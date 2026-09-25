from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


MODALITIES = ("text", "audio", "vision")


def sparsemax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection with exact zeros and autograd support."""
    shifted = values - values.max(dim=dim, keepdim=True).values
    sorted_values = torch.sort(shifted, dim=dim, descending=True).values
    size = values.size(dim)
    ranks_shape = [1] * values.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(1, size + 1, device=values.device, dtype=values.dtype).view(ranks_shape)
    cumulative = sorted_values.cumsum(dim)
    support = (1 + ranks * sorted_values) > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold_sum = cumulative.gather(dim, support_size - 1)
    threshold = (threshold_sum - 1) / support_size.to(values.dtype)
    return torch.clamp(shifted - threshold, min=0)


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    safe_mask = mask.clone()
    empty = ~safe_mask.any(dim=dim, keepdim=True)
    if bool(empty.any()):
        first = torch.zeros_like(safe_mask)
        index = [slice(None)] * safe_mask.ndim
        index[dim] = 0
        first[tuple(index)] = True
        safe_mask = safe_mask | (empty & first)
    weights = sparsemax(logits.masked_fill(~safe_mask, -1e4), dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class SparseTemporalPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.fallback = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence).squeeze(-1) / self.temperature
        weights = masked_sparsemax(logits, mask.bool(), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        empty = ~mask.bool().any(dim=1)
        pooled = torch.where(empty.unsqueeze(1), self.fallback.unsqueeze(0), pooled)
        return pooled, weights


class TextEncoder(nn.Module):
    def __init__(self, model_path: str | Path, cfg: dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("问题三模型需要 transformers，请先安装 requirements.txt") from exc
        self.backbone = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=bool(cfg.get("text_local_files_only", True)),
            add_pooling_layer=False,
            use_safetensors=True,
        )
        self.hidden_size = int(self.backbone.config.hidden_size)

    @property
    def layers(self) -> nn.ModuleList:
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if layers is None:
            raise TypeError("文本模型必须提供 BERT 兼容的 encoder.layer")
        return layers

    def freeze_bottom_layers(self, count: int) -> None:
        count = max(0, min(int(count), len(self.layers)))
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        for parameter in self.backbone.embeddings.parameters():
            parameter.requires_grad_(count == 0)
        for index, layer in enumerate(self.layers):
            for parameter in layer.parameters():
                parameter.requires_grad_(index >= count)

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(input_ids.long())

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor] = {
            "attention_mask": batch["attention_mask"].long(),
            "token_type_ids": batch["token_type_ids"].long().clamp(0, 1),
        }
        if inputs_embeds is None:
            kwargs["input_ids"] = batch["input_ids"].long()
        else:
            kwargs["inputs_embeds"] = inputs_embeds
        return self.backbone(**kwargs, return_dict=True).last_hidden_state


class ContinuousEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        dropout = float(cfg["dropout"])
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.position = nn.Embedding(int(cfg["max_steps"]), hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(cfg["modality_heads"]),
            dim_feedforward=int(cfg["modality_intermediate_dim"]),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(cfg["modality_layers"]), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        encoded = self.input(values.float()) + self.position(positions)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask.bool())
        return self.norm(encoded)


class HSAIGNet(nn.Module):
    """Hierarchical sparse-attention classifier/regressor with inspectable weights."""

    def __init__(self, cfg: dict[str, Any], text_model_path: str | Path) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        fusion = int(cfg["fusion_dim"])
        dropout = float(cfg["dropout"])
        self.text_encoder = TextEncoder(text_model_path, cfg)
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_size, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.audio_encoder = ContinuousEncoder(int(cfg["audio_dim"]), hidden, cfg)
        self.vision_encoder = ContinuousEncoder(int(cfg["vision_dim"]), hidden, cfg)
        temperature = float(cfg.get("time_attention_temperature", 1.0))
        self.time_pools = nn.ModuleDict({
            modality: SparseTemporalPool(hidden, dropout, temperature) for modality in MODALITIES
        })
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 3, fusion), nn.LayerNorm(fusion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion, 3),
        )
        self.gate_temperature = max(float(cfg.get("modality_gate_temperature", 1.0)), 1e-3)
        self.interaction = nn.Sequential(
            nn.Linear(hidden * 3, fusion), nn.LayerNorm(fusion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion, hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, fusion), nn.LayerNorm(fusion), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion, int(cfg["num_classes"]))
        self.regressor = nn.Linear(fusion, 1)
        self.uncertainty = nn.Linear(fusion, 1)

    def freeze_text_bottom_layers(self, count: int) -> None:
        self.text_encoder.freeze_bottom_layers(count)

    def text_word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.word_embeddings(input_ids)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        text_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_sequence = self.text_projection(self.text_encoder(batch, text_embeddings))
        audio_sequence = self.audio_encoder(batch["audio"], batch["padding_mask"])
        vision_sequence = self.vision_encoder(batch["vision"], batch["padding_mask"])
        sequences = {"text": text_sequence, "audio": audio_sequence, "vision": vision_sequence}
        pooled: dict[str, torch.Tensor] = {}
        time_weights: dict[str, torch.Tensor] = {}
        available: list[torch.Tensor] = []
        availability_ratio: list[torch.Tensor] = []
        content_count = batch["content_mask"].sum(dim=1).clamp_min(1).float()
        for modality in MODALITIES:
            mask = batch[f"{modality}_evidence_candidate_mask"].bool()
            pooled[modality], time_weights[modality] = self.time_pools[modality](sequences[modality], mask)
            available.append(mask.any(dim=1))
            availability_ratio.append(mask.sum(dim=1).float() / content_count)
        concatenated = torch.cat([pooled[m] for m in MODALITIES], dim=1)
        ratios = torch.stack(availability_ratio, dim=1)
        gate_logits = self.modality_gate(torch.cat((concatenated, ratios), dim=1)) / self.gate_temperature
        gate = masked_sparsemax(gate_logits, torch.stack(available, dim=1), dim=1)
        stacked = torch.stack([pooled[m] for m in MODALITIES], dim=1)
        weighted = torch.sum(stacked * gate.unsqueeze(-1), dim=1)
        interaction = self.interaction(concatenated)
        fused = self.fusion(torch.cat((weighted, interaction), dim=1))
        logits = self.classifier(fused)
        regression = 3.0 * torch.tanh(self.regressor(fused).squeeze(-1) / 3.0)
        log_variance = self.uncertainty(fused).squeeze(-1).clamp(-5.0, 3.0)
        return {
            "logits": logits,
            "regression": regression,
            "log_variance": log_variance,
            "modality_gate": gate,
            "gate_logits": gate_logits,
            "time_attention_text": time_weights["text"],
            "time_attention_audio": time_weights["audio"],
            "time_attention_vision": time_weights["vision"],
            "fused": fused,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
