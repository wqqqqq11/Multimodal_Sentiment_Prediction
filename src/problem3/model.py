from __future__ import annotations

from typing import Any

import torch
from torch import nn

MODALITIES = ("text", "audio", "vision")


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    ordered, _ = torch.sort(shifted, descending=True, dim=dim)
    size = ordered.size(dim)
    ranks = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.ndim
    shape[dim] = size
    ranks = ranks.view(shape)
    cumulative = ordered.cumsum(dim)
    support = 1 + ranks * ordered > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0)


class ScheduledSparseDistribution(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.register_buffer("sparsity_mix", torch.tensor(0.0), persistent=True)

    def set_mix(self, value: float) -> None:
        self.sparsity_mix.fill_(float(max(0.0, min(1.0, value))))

    def forward(self, scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        scaled = scores / self.temperature
        dense = torch.softmax(scaled, dim=-1)
        sparse = sparsemax(scaled, dim=-1)
        weights = dense.lerp(sparse, self.sparsity_mix.to(dense.dtype))
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden, dropout = int(cfg["hidden_dim"]), float(cfg["dropout"])
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                                              nn.GELU(), nn.Dropout(dropout))
        self.local3 = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.local5 = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.local_projection = nn.Linear(hidden * 2, hidden)
        self.local_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=int(cfg["modality_heads"]),
            dim_feedforward=int(cfg["modality_intermediate_dim"]), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(cfg["modality_layers"]),
                                              enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, int(cfg["max_steps"]), hidden))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(values.float()) + self.position[:, :values.size(1)]
        local = x.transpose(1, 2)
        local = torch.cat((self.local3(local), self.local5(local)), dim=1).transpose(1, 2)
        x = self.local_norm(x + self.local_projection(local))
        x = self.temporal(x, src_key_padding_mask=~valid.bool())
        return x * valid.unsqueeze(-1).to(x.dtype)


class SparseTemporalPool(nn.Module):
    def __init__(self, hidden: int, dropout: float, temperature: float) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.Tanh(),
                                   nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.distribution = ScheduledSparseDistribution(temperature)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.distribution(self.score(sequence).squeeze(-1), mask)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1), weights


class HSAIGNet(nn.Module):
    """Three-modality model whose prediction paths all pass through the gate."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden, fusion, dropout = int(cfg["hidden_dim"]), int(cfg["fusion_dim"]), float(cfg["dropout"])
        dimensions = {"text": int(cfg["text_dim"]), "audio": int(cfg["audio_dim"]), "vision": int(cfg["vision_dim"])}
        self.encoders = nn.ModuleDict({name: SequenceEncoder(dimensions[name], cfg) for name in MODALITIES})
        self.pools = nn.ModuleDict({name: SparseTemporalPool(hidden, dropout, float(cfg["time_attention_temperature"]))
                                    for name in MODALITIES})
        self.gate_score = nn.Sequential(nn.LayerNorm(hidden * 3), nn.Linear(hidden * 3, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.gate_distribution = ScheduledSparseDistribution(float(cfg["modality_gate_temperature"]))
        self.fusion = nn.Sequential(nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, fusion), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(fusion, fusion), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(fusion, int(cfg["num_classes"]))
        self.regressor, self.uncertainty = nn.Linear(fusion, 1), nn.Linear(fusion, 1)
        self.aux_classifiers = nn.ModuleDict({name: nn.Linear(hidden, int(cfg["num_classes"])) for name in MODALITIES})
        self.aux_regressors = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in MODALITIES})

    def set_sparsity_mix(self, value: float) -> None:
        for pool in self.pools.values():
            pool.distribution.set_mix(value)
        self.gate_distribution.set_mix(value)

    def freeze_text_bottom_layers(self, _: int) -> None:
        return

    @staticmethod
    def _mask(batch: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        mask = batch[f"{name}_evidence_candidate_mask"].bool()
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = torch.where(empty.unsqueeze(-1), ~batch["padding_mask"].bool(), mask)
        return mask

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        values = {"text": batch["text_features"], "audio": batch["audio"], "vision": batch["vision"]}
        pooled, attentions = {}, {}
        for name in MODALITIES:
            mask = self._mask(batch, name)
            pooled[name], attentions[name] = self.pools[name](self.encoders[name](values[name], mask), mask)
        stacked = torch.stack([pooled[name] for name in MODALITIES], dim=1)
        gate = self.gate_distribution(self.gate_score(torch.cat([pooled[name] for name in MODALITIES], dim=-1)))
        gated = stacked * gate.unsqueeze(-1)
        fused = self.fusion(torch.cat((gated.sum(dim=1), gated.flatten(1)), dim=-1))
        return {
            "logits": self.classifier(fused),
            "regression": 3.0 * torch.tanh(self.regressor(fused).squeeze(-1) / 3.0),
            "log_variance": self.uncertainty(fused).squeeze(-1).clamp(-5.0, 3.0),
            "modality_gate": gate,
            "aux_logits": torch.stack([self.aux_classifiers[n](pooled[n]) for n in MODALITIES], dim=1),
            "aux_regression": torch.stack([3.0 * torch.tanh(self.aux_regressors[n](pooled[n]).squeeze(-1) / 3.0)
                                             for n in MODALITIES], dim=1),
            **{f"time_attention_{name}": attentions[name] for name in MODALITIES},
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
