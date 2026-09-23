from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.fallback = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        logits = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        empty = ~mask.any(dim=1)
        return torch.where(empty.unsqueeze(1), self.fallback.unsqueeze(0), pooled)


class TextEncoder(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        self.token = nn.Embedding(int(cfg["vocab_size"]), hidden, padding_idx=0)
        self.token_type = nn.Embedding(2, hidden)
        self.position = nn.Embedding(int(cfg["max_steps"]), hidden)
        self.state = nn.Embedding(5, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg["text_heads"]),
            dim_feedforward=hidden * 3,
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(cfg["text_layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.pool = MaskedAttentionPool(hidden)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = batch["input_ids"].long()
        token_type = batch["token_type_ids"].long().clamp(0, 1)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        state = torch.ones_like(ids)
        state = torch.where(batch["structural_mask"].bool(), torch.full_like(state, 2), state)
        state = torch.where(batch["padding_mask"].bool(), torch.zeros_like(state), state)
        encoded = self.token(ids) + self.token_type(token_type) + self.position(positions) + self.state(state)
        encoded = self.encoder(encoded, src_key_padding_mask=~batch["attention_mask"].bool())
        encoded = self.norm(encoded)
        return encoded, self.pool(encoded, batch["content_mask"].bool())


class ContinuousEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_steps: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.position = nn.Embedding(max_steps, hidden_dim)
        self.state = nn.Embedding(5, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = MaskedAttentionPool(hidden_dim)

    @staticmethod
    def state_ids(batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
        state = torch.ones_like(batch["content_mask"], dtype=torch.long)  # observed
        state = torch.where(batch["structural_mask"].bool(), torch.full_like(state, 2), state)
        state = torch.where(batch[f"{modality}_natural_zero_mask"].bool(), torch.full_like(state, 3), state)
        state = torch.where(batch[f"{modality}_missing_mask"].bool(), torch.full_like(state, 4), state)
        state = torch.where(batch["padding_mask"].bool(), torch.zeros_like(state), state)
        return state

    def forward(self, batch: dict[str, torch.Tensor], modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch[modality].float()
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        encoded = self.projection(values) + self.position(positions) + self.state(self.state_ids(batch, modality))
        encoded, _ = self.gru(encoded)
        encoded = self.norm(self.dropout(encoded))
        observed = batch[f"{modality}_observed_mask"].bool() & batch["content_mask"].bool()
        return encoded, self.pool(encoded, observed)


class ReliabilityGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, modalities: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        scores = self.network(torch.cat((modalities, reliability), dim=-1)).squeeze(-1)
        scores = scores + self.gamma * torch.log(reliability[:, :, 0].clamp_min(1e-4))
        return torch.softmax(scores, dim=1)


class MRCDNet(nn.Module):
    """Mask-aware Reliability-gated Complete-to-Missing Distillation Network."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.ablation_mode = "none"
        hidden = int(cfg["hidden_dim"])
        fusion = int(cfg["fusion_dim"])
        dropout = float(cfg["dropout"])
        self.text_encoder = TextEncoder(cfg)
        self.audio_encoder = ContinuousEncoder(int(cfg["audio_dim"]), hidden, int(cfg["max_steps"]), dropout)
        self.vision_encoder = ContinuousEncoder(int(cfg["vision_dim"]), hidden, int(cfg["max_steps"]), dropout)
        self.gate = ReliabilityGate(hidden, dropout, float(cfg["gate_reliability_gamma"]))
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 6, fusion),
            nn.LayerNorm(fusion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion, fusion),
            nn.GELU(),
        )
        self.classification_head = nn.Linear(fusion, int(cfg["num_classes"]))
        self.regression_head = nn.Linear(fusion, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _, text = self.text_encoder(batch)
        _, audio = self.audio_encoder(batch, "audio")
        _, vision = self.vision_encoder(batch, "vision")
        modalities = torch.stack((text, audio, vision), dim=1)
        reliability = batch["modality_reliability"].float()
        if reliability.shape[-1] != 8:
            from .data import reliability_features
            reliability = reliability_features(batch)
        gates = self.gate(modalities, reliability)
        if self.ablation_mode == "uniform_gate":
            availability = reliability[:, :, 0].clamp_min(1e-4)
            gates = availability / availability.sum(dim=1, keepdim=True)
        elif self.ablation_mode == "text_only":
            gates = torch.zeros_like(gates)
            gates[:, 0] = 1.0
        elif self.ablation_mode != "none":
            raise ValueError(f"未知消融模式: {self.ablation_mode}")
        weighted = modalities * gates.unsqueeze(-1)
        pairwise = torch.cat((text * audio, text * vision, audio * vision), dim=-1)
        fused = self.fusion(torch.cat((weighted.flatten(1), pairwise), dim=-1))
        logits = self.classification_head(fused)
        regression = 3.0 * torch.tanh(self.regression_head(fused).squeeze(-1))
        return {
            "logits": logits,
            "regression": regression,
            "fused": fused,
            "gates": gates,
            "modalities": modalities,
            "reliability": reliability,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
