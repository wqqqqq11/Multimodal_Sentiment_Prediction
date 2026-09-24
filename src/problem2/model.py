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


def _state_ids(batch: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
    """0 padding, 1 observed, 2 structural, 3 natural zero, 4 artificial missing."""
    state = torch.ones_like(batch["content_mask"], dtype=torch.long)
    state = torch.where(batch["structural_mask"].bool(), torch.full_like(state, 2), state)
    state = torch.where(batch[f"{modality}_natural_zero_mask"].bool(), torch.full_like(state, 3), state)
    state = torch.where(batch[f"{modality}_missing_mask"].bool(), torch.full_like(state, 4), state)
    state = torch.where(batch["padding_mask"].bool(), torch.zeros_like(state), state)
    return state


class PretrainedTextEncoder(nn.Module):
    """BERT-compatible encoder with an explicit missing-state embedding."""

    def __init__(self, cfg: dict[str, Any], model_name: str | None = None) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover - exercised in deployment
            raise ImportError("问题二的新模型需要 transformers，请先安装 requirements.txt") from exc

        name = model_name or str(cfg["text_pretrained_model"])
        use_pretrained = bool(cfg.get("text_pretrained", True))
        if use_pretrained:
            is_teacher = name == str(cfg.get("teacher_text_pretrained_model", ""))
            revision_key = "teacher_text_pretrained_revision" if is_teacher else "text_pretrained_revision"
            revision = str(cfg.get(revision_key, "main"))
            self.backbone = AutoModel.from_pretrained(
                name,
                revision=revision,
                local_files_only=bool(cfg.get("text_local_files_only", False)),
                add_pooling_layer=False,
                use_safetensors=True,
            )
        else:
            hidden = int(cfg.get("text_hidden_dim", 256))
            heads = int(cfg.get("text_heads", 4))
            self.backbone = BertModel(
                BertConfig(
                    vocab_size=int(cfg.get("vocab_size", 30522)),
                    hidden_size=hidden,
                    num_hidden_layers=int(cfg.get("text_layers", 4)),
                    num_attention_heads=heads,
                    intermediate_size=int(cfg.get("text_intermediate_dim", hidden * 4)),
                    hidden_dropout_prob=float(cfg["dropout"]),
                    attention_probs_dropout_prob=float(cfg["dropout"]),
                    max_position_embeddings=max(64, int(cfg["max_steps"])),
                    type_vocab_size=2,
                    pad_token_id=0,
                ),
                add_pooling_layer=False,
            )
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.state = nn.Embedding(5, self.hidden_size, padding_idx=0)
        self.pool = MaskedAttentionPool(self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        nn.init.normal_(self.state.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.state.weight[0].zero_()

    @property
    def layers(self) -> nn.ModuleList:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise TypeError("所选文本模型不是 BERT 兼容的 encoder.layer 结构")
        return layers

    def freeze_bottom_layers(self, count: int) -> None:
        count = max(0, min(int(count), len(self.layers)))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        embeddings = getattr(self.backbone, "embeddings", None)
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = not (count > 0)
        for index, layer in enumerate(self.layers):
            requires_grad = index >= count
            for parameter in layer.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = batch["input_ids"].long()
        attention = batch["attention_mask"].long()
        token_type = batch["token_type_ids"].long().clamp(0, 1)
        word = self.backbone.get_input_embeddings()(ids)
        word = word + self.state(_state_ids(batch, "text"))
        encoded = self.backbone(
            inputs_embeds=word,
            attention_mask=attention,
            token_type_ids=token_type,
            return_dict=True,
        ).last_hidden_state
        observed = batch["text_observed_mask"].bool() & attention.bool()
        attended = self.pool(encoded, observed)
        cls = encoded[:, 0]
        return encoded, self.norm(0.5 * cls + 0.5 * attended)


class ContinuousEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        dropout = float(cfg["dropout"])
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.position = nn.Embedding(int(cfg["max_steps"]), hidden_dim)
        self.state = nn.Embedding(5, hidden_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(cfg.get("modality_heads", 4)),
            dim_feedforward=int(cfg.get("modality_intermediate_dim", hidden_dim * 2)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(cfg.get("modality_layers", 1)),
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = MaskedAttentionPool(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor], modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        values = batch[modality].float()
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        encoded = self.projection(values) + self.position(positions) + self.state(_state_ids(batch, modality))
        encoded = self.encoder(encoded, src_key_padding_mask=batch["padding_mask"].bool())
        encoded = self.norm(encoded)
        observed = batch[f"{modality}_observed_mask"].bool() & batch["content_mask"].bool()
        return encoded, self.pool(encoded, observed)


class SafeCrossAttention(nn.Module):
    """Text queries a modality; artificial missing positions never become keys/values."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        source_observed: torch.Tensor,
    ) -> torch.Tensor:
        valid = source_observed.bool().clone()
        empty = ~valid.any(dim=1)
        if empty.any():
            source = source.clone()
            valid[empty, 0] = True
            source[empty, 0] = self.missing_token[0, 0].to(source.dtype)
        delta, _ = self.attention(query, source, source, key_padding_mask=~valid, need_weights=False)
        gate = self.gate(torch.cat((query, delta), dim=-1))
        return self.norm(query + gate * delta)


class TextAnchoredFusion(nn.Module):
    def __init__(self, hidden_dim: int, fusion_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        heads = int(cfg.get("cross_attention_heads", 4))
        dropout = float(cfg["dropout"])
        self.audio_cross = SafeCrossAttention(hidden_dim, heads, dropout)
        self.vision_cross = SafeCrossAttention(hidden_dim, heads, dropout)
        self.pool = MaskedAttentionPool(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 5, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )

    def forward(
        self,
        text_sequence: torch.Tensor,
        text_mask: torch.Tensor,
        audio_sequence: torch.Tensor,
        audio_pool: torch.Tensor,
        audio_observed: torch.Tensor,
        vision_sequence: torch.Tensor,
        vision_pool: torch.Tensor,
        vision_observed: torch.Tensor,
    ) -> torch.Tensor:
        text_pool = self.pool(text_sequence, text_mask)
        text_audio = self.pool(self.audio_cross(text_sequence, audio_sequence, audio_observed), text_mask)
        text_vision = self.pool(self.vision_cross(text_sequence, vision_sequence, vision_observed), text_mask)
        return self.network(torch.cat((text_pool, text_audio, text_vision, audio_pool, vision_pool), dim=-1))


class ExpertHead(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(input_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_dim, classes)
        self.regressor = nn.Linear(fusion_dim, 1)
        self.log_variance = nn.Linear(fusion_dim, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.representation(value)
        return (
            feature,
            self.classifier(feature),
            self.regressor(feature).squeeze(-1),
            self.log_variance(feature).squeeze(-1).clamp(-5.0, 3.0),
        )


class ExpertRouter(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("router_hidden_dim", 64))
        self.prior_strength = float(cfg.get("router_prior_strength", 1.5))
        self.network = nn.Sequential(
            nn.Linear(35, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(float(cfg["dropout"])), nn.Linear(hidden, 3)
        )

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)

    def forward(
        self,
        reliability: torch.Tensor,
        expert_logits: torch.Tensor,
        expert_log_variance: torch.Tensor,
        synchrony: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        availability = reliability[:, :, 0]
        audio_visual_availability = availability[:, 1:].mean(dim=1)
        missing_degree = 1.0 - audio_visual_availability
        prior_text = (0.15 + missing_degree).clamp_max(1.0)
        prior_full = (0.1 + audio_visual_availability * (1.0 - synchrony * missing_degree)).clamp_max(1.0)
        prior_missing = (0.1 + missing_degree * (0.5 + synchrony)).clamp_max(1.0)
        prior = torch.stack((prior_text, prior_full, prior_missing), dim=1)
        prior = prior / prior.sum(dim=1, keepdim=True)
        entropy = torch.stack([self._entropy(expert_logits[:, index]) for index in range(3)], dim=1)
        uncertainty = torch.exp(0.5 * expert_log_variance).clamp_max(10.0)
        rate_gap = (availability[:, 1] - availability[:, 2]).abs().unsqueeze(1)
        router_input = torch.cat(
            (reliability.flatten(1), entropy, uncertainty, synchrony.unsqueeze(1), rate_gap), dim=1
        )
        scores = self.network(router_input) + self.prior_strength * torch.log(prior.clamp_min(1e-5))
        return torch.softmax(scores, dim=1), prior


class MRCDNet(nn.Module):
    """Compact-BERT, text-anchored, three-expert model for complete and missing inputs."""

    def __init__(self, cfg: dict[str, Any], *, text_model_name: str | None = None) -> None:
        super().__init__()
        self.ablation_mode = "none"
        hidden = int(cfg["hidden_dim"])
        fusion = int(cfg["fusion_dim"])
        dropout = float(cfg["dropout"])
        classes = int(cfg["num_classes"])
        self.text_encoder = PretrainedTextEncoder(cfg, text_model_name)
        self.text_projection = nn.Sequential(nn.Linear(self.text_encoder.hidden_size, hidden), nn.LayerNorm(hidden))
        self.text_pool = MaskedAttentionPool(hidden)
        self.audio_encoder = ContinuousEncoder(int(cfg["audio_dim"]), hidden, cfg)
        self.vision_encoder = ContinuousEncoder(int(cfg["vision_dim"]), hidden, cfg)
        self.fusion = TextAnchoredFusion(hidden, fusion, cfg)
        self.text_expert = ExpertHead(hidden, fusion, classes, dropout)
        self.full_expert = ExpertHead(fusion, fusion, classes, dropout)
        self.missing_expert = ExpertHead(fusion, fusion, classes, dropout)
        self.router = ExpertRouter(cfg)

    def freeze_text_bottom_layers(self, count: int) -> None:
        self.text_encoder.freeze_bottom_layers(count)

    def encode_text_features(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw_tokens, raw_pooled = self.text_encoder(batch)
        tokens = self.text_projection(raw_tokens)
        mask = batch["text_observed_mask"].bool() & batch["attention_mask"].bool()
        pooled = self.text_pool(tokens, mask)
        feature, logits, regression, log_variance = self.text_expert(pooled)
        return {
            "raw_tokens": raw_tokens,
            "raw_cls": raw_pooled,
            "tokens": tokens,
            "cls": pooled,
            "feature": feature,
            "logits": logits,
            "regression": regression,
            "log_variance": log_variance,
        }

    @staticmethod
    def _synchrony(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio = batch["audio_missing_mask"].bool() & batch["content_mask"].bool()
        vision = batch["vision_missing_mask"].bool() & batch["content_mask"].bool()
        union = (audio | vision).sum(dim=1).float()
        intersection = (audio & vision).sum(dim=1).float()
        return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(union))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text = self.encode_text_features(batch)
        audio_sequence, audio_pool = self.audio_encoder(batch, "audio")
        vision_sequence, vision_pool = self.vision_encoder(batch, "vision")
        content = batch["content_mask"].bool()
        fused_input = self.fusion(
            text["tokens"],
            batch["attention_mask"].bool(),
            audio_sequence,
            audio_pool,
            batch["audio_observed_mask"].bool() & content,
            vision_sequence,
            vision_pool,
            batch["vision_observed_mask"].bool() & content,
        )
        text_values = (text["feature"], text["logits"], text["regression"], text["log_variance"])
        full_values = self.full_expert(fused_input)
        missing_values = self.missing_expert(fused_input)
        features = torch.stack((text_values[0], full_values[0], missing_values[0]), dim=1)
        expert_logits = torch.stack((text_values[1], full_values[1], missing_values[1]), dim=1)
        expert_regression = torch.stack((text_values[2], full_values[2], missing_values[2]), dim=1)
        expert_log_variance = torch.stack((text_values[3], full_values[3], missing_values[3]), dim=1)
        reliability = batch["modality_reliability"].float()
        if reliability.shape[-1] != 9:
            from .data import reliability_features

            reliability = reliability_features(batch)
        synchrony = self._synchrony(batch)
        gates, prior = self.router(reliability, expert_logits, expert_log_variance, synchrony)
        if self.ablation_mode == "uniform_gate":
            gates = torch.full_like(gates, 1.0 / 3.0)
        elif self.ablation_mode == "text_only":
            gates = torch.zeros_like(gates)
            gates[:, 0] = 1.0
        elif self.ablation_mode != "none":
            raise ValueError(f"未知消融模式: {self.ablation_mode}")
        logits = (expert_logits * gates.unsqueeze(-1)).sum(dim=1)
        regression = (expert_regression * gates).sum(dim=1)
        log_variance = torch.logsumexp(expert_log_variance + torch.log(gates.clamp_min(1e-8)), dim=1)
        fused = (features * gates.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": logits,
            "regression": regression,
            "log_variance": log_variance,
            "fused": fused,
            "gates": gates,
            "router_prior": prior,
            "expert_logits": expert_logits,
            "expert_regression": expert_regression,
            "expert_log_variance": expert_log_variance,
            "modalities": torch.stack((text["cls"], audio_pool, vision_pool), dim=1),
            "text_vector": text["cls"],
            "text_tokens": text["tokens"],
            "text_raw_tokens": text["raw_tokens"],
            "text_raw_cls": text["raw_cls"],
            "text_logits": text["logits"],
            "text_regression": text["regression"],
            "reliability": reliability,
            "synchrony": synchrony,
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
