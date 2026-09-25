from __future__ import annotations

from typing import Any

import torch
from torch import nn


MODALITIES = ("text", "audio", "vision")


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class ContinuousContextEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        self.input_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.position = nn.Embedding(int(cfg["max_steps"]), hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg["attention_heads"]),
            dim_feedforward=int(cfg["intermediate_dim"]),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(cfg["context_layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        encoded = self.input_projection(values.float()) + self.position(positions)
        return self.norm(self.encoder(encoded, src_key_padding_mask=padding_mask.bool()))


class SalienceHead(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.network(sequence.transpose(1, 2)).squeeze(1)


class SparseTopK(nn.Module):
    """Global availability-aware hard Top-K with an iterative soft Top-K gradient."""

    def __init__(
        self,
        top_k: int,
        temperature: float,
        *,
        noise_scale: float = 1.0,
        adaptive_ratio: float = 0.0,
        minimum_k: int = 1,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.noise_scale = float(noise_scale)
        self.adaptive_ratio = float(adaptive_ratio)
        self.minimum_k = min(max(int(minimum_k), 1), self.top_k)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def set_noise_scale(self, noise_scale: float) -> None:
        self.noise_scale = max(float(noise_scale), 0.0)

    def _budget(self, mask: torch.Tensor) -> torch.Tensor:
        available = mask.sum(dim=-1)
        if self.adaptive_ratio <= 0:
            return available.clamp(max=self.top_k)
        budget = torch.ceil(available.float() * self.adaptive_ratio).long()
        budget = budget.clamp(min=self.minimum_k, max=self.top_k)
        return torch.minimum(budget, available)

    def _soft_topk(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        working = logits.masked_fill(~mask, -1e4) / self.temperature
        soft = torch.zeros_like(working)
        for _ in range(self.top_k):
            probability = torch.softmax(working, dim=-1) * mask.to(working.dtype)
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            soft = soft + probability
            working = working + torch.log((1.0 - probability).clamp_min(1e-6))
        return soft.clamp(max=1.0)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        perturbed = logits
        if self.training and self.noise_scale > 0:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            perturbed = logits - self.noise_scale * torch.log(-torch.log(uniform))
        hard_indices = perturbed.masked_fill(~mask, -1e4).topk(self.top_k, dim=-1).indices
        rank = torch.arange(self.top_k, device=logits.device).unsqueeze(0)
        budget = self._budget(mask)
        within_budget = rank < budget.unsqueeze(1)
        selected_valid = torch.gather(mask, 1, hard_indices) & within_budget
        hard = torch.zeros_like(logits).scatter(1, hard_indices, selected_valid.to(logits.dtype))
        soft = self._soft_topk(logits, mask)
        soft = soft * (budget.to(soft.dtype) / float(self.top_k)).unsqueeze(1)
        straight_through = hard - soft.detach() + soft
        return {
            "indices": hard_indices, "selected_valid": selected_valid, "hard_mask": hard,
            "soft_mask": soft, "straight_through_mask": straight_through,
        }


class EvidencePredictor(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        self.reasoning_tokens = nn.Parameter(torch.empty(1, int(cfg["reasoning_tokens"]), hidden))
        nn.init.normal_(self.reasoning_tokens, std=0.02)
        self.empty_evidence = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg["attention_heads"]),
            dim_feedforward=int(cfg["intermediate_dim"]),
            dropout=float(cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(cfg["reasoning_layers"]), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, int(cfg["num_classes"]))
        self.regressor = nn.Linear(hidden, 1)
        self.ordinal_alpha = float(cfg.get("ordinal_logit_alpha", 0.0))
        self.ordinal_threshold = float(cfg.get("ordinal_neutral_threshold", 0.25))
        self.ordinal_temperature = max(float(cfg.get("ordinal_temperature", 0.35)), 1e-3)

    def _ordinal_logits(self, regression: torch.Tensor) -> torch.Tensor:
        threshold = self.ordinal_threshold
        temperature = self.ordinal_temperature
        return torch.stack(
            (
                (-regression - threshold) / temperature,
                (threshold - regression.abs()) / temperature,
                (regression - threshold) / temperature,
            ),
            dim=-1,
        )

    def forward(self, evidence: torch.Tensor, evidence_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = evidence.shape[0]
        valid = evidence_mask.bool()
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            evidence = evidence.clone()
            valid = valid.clone()
            evidence[empty, 0] = self.empty_evidence[0, 0]
            valid[empty, 0] = True
        reasoning = self.reasoning_tokens.expand(batch, -1, -1)
        joined = torch.cat((reasoning, evidence), dim=1)
        reasoning_valid = torch.ones(batch, reasoning.shape[1], dtype=torch.bool, device=evidence.device)
        joined_mask = torch.cat((reasoning_valid, valid), dim=1)
        encoded = self.encoder(joined, src_key_padding_mask=~joined_mask)
        representation = self.norm(encoded[:, : reasoning.shape[1]].mean(dim=1))
        raw_logits = self.classifier(representation)
        regression = 3.0 * torch.tanh(self.regressor(representation).squeeze(-1))
        ordinal_logits = self._ordinal_logits(regression)
        logits = (1.0 - self.ordinal_alpha) * raw_logits + self.ordinal_alpha * ordinal_logits
        return {
            "representation": representation,
            "logits": logits,
            "raw_logits": raw_logits,
            "ordinal_logits": ordinal_logits,
            "regression": regression,
        }


class SEPCNet(nn.Module):
    """Sparse Evidence Prototype Counterfactual Network.

    Contextual features are used only by the selector and dense training reference.
    The evidence predictor receives gathered fixed-radius local payloads, never the full
    contextual sequence or the complete salience vector.
    """

    def __init__(self, cfg: dict[str, Any], text_model_path: str | None = None) -> None:
        super().__init__()
        model_cfg = cfg["model"] if "model" in cfg else cfg
        self.cfg = model_cfg
        hidden = int(model_cfg["hidden_dim"])
        self.max_steps = int(model_cfg["max_steps"])
        try:
            from transformers import AutoModel, BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("问题三模型需要 transformers，请在项目 uv 环境安装依赖") from exc
        if bool(model_cfg.get("text_pretrained", True)):
            if not text_model_path:
                raise ValueError("启用预训练文本模型时必须提供 text_model_path")
            self.text_backbone = AutoModel.from_pretrained(
                text_model_path,
                local_files_only=bool(model_cfg.get("text_local_files_only", True)),
                add_pooling_layer=False,
            )
        else:
            text_hidden = int(model_cfg.get("text_hidden_dim", hidden))
            self.text_backbone = BertModel(
                BertConfig(
                    vocab_size=int(model_cfg["vocab_size"]), hidden_size=text_hidden,
                    num_hidden_layers=int(model_cfg.get("text_layers", 1)),
                    num_attention_heads=int(model_cfg["attention_heads"]),
                    intermediate_size=int(model_cfg["intermediate_dim"]),
                    max_position_embeddings=max(64, self.max_steps), pad_token_id=0,
                    hidden_dropout_prob=float(model_cfg["dropout"]),
                    attention_probs_dropout_prob=float(model_cfg["dropout"]),
                ),
                add_pooling_layer=False,
            )
        text_hidden = int(self.text_backbone.config.hidden_size)
        self.text_context_projection = nn.Linear(text_hidden, hidden)
        self.audio_context = ContinuousContextEncoder(int(model_cfg["audio_dim"]), model_cfg)
        self.vision_context = ContinuousContextEncoder(int(model_cfg["vision_dim"]), model_cfg)

        self.text_payload = nn.Sequential(nn.Linear(text_hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.audio_payload = nn.Sequential(nn.Linear(int(model_cfg["audio_dim"]), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.vision_payload = nn.Sequential(nn.Linear(int(model_cfg["vision_dim"]), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.salience = nn.ModuleList(SalienceHead(hidden, float(model_cfg["dropout"])) for _ in MODALITIES)
        self.selector = SparseTopK(
            int(model_cfg["top_k"]),
            float(model_cfg["selector_temperature_initial"]),
            noise_scale=float(model_cfg.get("selector_noise_initial", 1.0)),
            adaptive_ratio=float(model_cfg.get("adaptive_top_k_ratio", 0.0)),
            minimum_k=int(model_cfg.get("minimum_top_k", 1)),
        )
        self.evidence_window_radius = max(int(model_cfg.get("evidence_window_radius", 0)), 0)
        self.local_window_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.modality_embedding = nn.Embedding(3, hidden)
        self.position_embedding = nn.Embedding(self.max_steps, hidden)
        self.predictor = EvidencePredictor(model_cfg)
        self.query_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

        dense_input = hidden * 3
        self.dense_reference = nn.Sequential(nn.Linear(dense_input, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(float(model_cfg["dropout"])))
        self.dense_classifier = nn.Linear(hidden, int(model_cfg["num_classes"]))
        self.dense_regressor = nn.Linear(hidden, 1)
        self.modality_classifier = nn.Linear(hidden, int(model_cfg["num_classes"]))
        self.modality_regressor = nn.Linear(hidden, 1)
        self.selector_classifier = nn.Linear(hidden, int(model_cfg["num_classes"]))
        self.selector_regressor = nn.Linear(hidden, 1)

    def freeze_text_layers(self, frozen: bool) -> None:
        for parameter in self.text_backbone.parameters():
            parameter.requires_grad = not frozen

    def set_selector_temperature(self, temperature: float) -> None:
        self.selector.set_temperature(temperature)

    def set_selector_noise(self, noise_scale: float) -> None:
        self.selector.set_noise_scale(noise_scale)

    def _candidate_masks(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([batch[f"{name}_evidence_candidate_mask"].bool() for name in MODALITIES], dim=1)

    def _encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].long()
        attention = batch["attention_mask"].long()
        token_type = batch.get("token_type_ids", torch.zeros_like(input_ids)).long()
        text_context_raw = self.text_backbone(
            input_ids=input_ids, attention_mask=attention, token_type_ids=token_type, return_dict=True,
        ).last_hidden_state
        text_context = self.text_context_projection(text_context_raw)
        text_word = self.text_backbone.get_input_embeddings()(input_ids)
        text_payload = self.text_payload(text_word)
        padding = batch["padding_mask"].bool()
        audio_context = self.audio_context(batch["audio"], padding)
        vision_context = self.vision_context(batch["vision"], padding)
        context = torch.stack((text_context, audio_context, vision_context), dim=1)
        payload = torch.stack((text_payload, self.audio_payload(batch["audio"].float()), self.vision_payload(batch["vision"].float())), dim=1)
        candidates = self._candidate_masks(batch)
        return context, self._local_window_payload(payload, candidates), candidates

    def _local_window_payload(self, payload: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Build local evidence units without leaking the full sequence."""
        radius = self.evidence_window_radius
        weighted = payload * candidates.to(payload.dtype).unsqueeze(-1)
        local_sum = torch.zeros_like(payload)
        local_count = torch.zeros_like(candidates, dtype=payload.dtype)
        steps = payload.shape[2]
        for offset in range(-radius, radius + 1):
            source_start = max(0, -offset)
            source_end = min(steps, steps - offset)
            target_start = source_start + offset
            target_end = source_end + offset
            if source_end <= source_start:
                continue
            local_sum[:, :, target_start:target_end] += weighted[:, :, source_start:source_end]
            local_count[:, :, target_start:target_end] += candidates[:, :, source_start:source_end].to(payload.dtype)
        local_mean = local_sum / local_count.clamp_min(1.0).unsqueeze(-1)
        return self.local_window_projection(torch.cat((payload, local_mean), dim=-1))

    def _add_identity(self, payload: torch.Tensor, modality: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return payload + self.modality_embedding(modality) + self.position_embedding(position)

    def predict_set(
        self, payload: torch.Tensor, modality: torch.Tensor, position: torch.Tensor, evidence_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence = self._add_identity(payload, modality.long(), position.long())
        result = self.predictor(evidence, evidence_mask.bool())
        result["query"] = torch.nn.functional.normalize(self.query_projection(evidence), dim=-1)
        return result

    def _fuse_prototypes(
        self, prediction: dict[str, torch.Tensor], modality: torch.Tensor, valid: torch.Tensor, prototype_memory: Any | None,
    ) -> dict[str, torch.Tensor]:
        if prototype_memory is None or not prototype_memory.ready:
            return prediction
        proto = prototype_memory.retrieve(prediction["query"], modality, valid)
        alpha = float(self.cfg["prototype_fusion_alpha"]) * proto["prototype_available"].to(prediction["logits"].dtype)
        prediction.update(proto)
        prediction["base_logits"] = prediction["logits"]
        prediction["base_regression"] = prediction["regression"]
        prediction["logits"] = (1.0 - alpha[:, None]) * prediction["base_logits"] + alpha[:, None] * proto["prototype_logits"]
        prediction["regression"] = (1.0 - alpha) * prediction["base_regression"] + alpha * proto["prototype_regression"]
        return prediction

    def predict_selected_subset(
        self, outputs: dict[str, torch.Tensor], keep_mask: torch.Tensor, prototype_memory: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        valid = keep_mask.bool() & outputs["selected_valid"].bool()
        prediction = self.predict_set(
            outputs["selected_payload"], outputs["selected_modality"], outputs["selected_position"],
            valid,
        )
        return self._fuse_prototypes(prediction, outputs["selected_modality"], valid, prototype_memory)

    def predict_complement(self, outputs: dict[str, torch.Tensor], prototype_memory: Any | None = None) -> dict[str, torch.Tensor]:
        complement = outputs["all_candidate_mask"] & ~outputs["hard_mask"].bool()
        batch, modalities, steps, hidden = outputs["all_payload"].shape
        payload = outputs["all_payload"].reshape(batch, modalities * steps, hidden)
        modality = torch.arange(modalities, device=payload.device).view(1, modalities, 1).expand(batch, modalities, steps).reshape(batch, -1)
        position = torch.arange(steps, device=payload.device).view(1, 1, steps).expand(batch, modalities, steps).reshape(batch, -1)
        valid = complement.reshape(batch, -1)
        prediction = self.predict_set(payload, modality, position, valid)
        return self._fuse_prototypes(prediction, modality, valid, prototype_memory)

    def forward(self, batch: dict[str, torch.Tensor], prototype_memory: Any | None = None) -> dict[str, torch.Tensor]:
        context, payload, candidates = self._encode(batch)
        batch_size, _, steps, hidden = context.shape
        score = torch.stack([self.salience[index](context[:, index]) for index in range(3)], dim=1)
        flat_score = score.reshape(batch_size, -1)
        flat_candidate = candidates.reshape(batch_size, -1)
        selection = self.selector(flat_score, flat_candidate)
        indices = selection["indices"]
        flat_payload = payload.reshape(batch_size, -1, hidden)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, hidden)
        selected_payload = torch.gather(flat_payload, 1, gather_index)
        selected_modality = torch.div(indices, steps, rounding_mode="floor")
        selected_position = indices.remainder(steps)
        gate = torch.gather(selection["straight_through_mask"], 1, indices).unsqueeze(-1)
        selected_payload = selected_payload * gate
        selected = self.predict_set(
            selected_payload,
            selected_modality,
            selected_position,
            selection["selected_valid"],
        )

        pooled_context = [masked_mean(context[:, index], candidates[:, index]) for index in range(3)]
        dense_feature = self.dense_reference(torch.cat(pooled_context, dim=-1))
        dense_logits = self.dense_classifier(dense_feature)
        dense_regression = 3.0 * torch.tanh(self.dense_regressor(dense_feature).squeeze(-1))
        # Direct supervised gradient for the selector while the main path stays hard Top-K.
        soft_mask = selection["soft_mask"].reshape(batch_size, 3, steps)
        soft_weight = soft_mask * candidates.to(soft_mask.dtype)
        modality_feature = (
            (payload * soft_weight.unsqueeze(-1)).sum(dim=2)
            / soft_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        )
        modality_logits = self.modality_classifier(modality_feature)
        modality_regression = 3.0 * torch.tanh(self.modality_regressor(modality_feature).squeeze(-1))
        selector_feature = (
            (payload * soft_weight.unsqueeze(-1)).sum(dim=(1, 2))
            / soft_weight.sum(dim=(1, 2), keepdim=False).clamp_min(1e-6).unsqueeze(-1)
        )
        selector_logits = self.selector_classifier(selector_feature)
        selector_regression = 3.0 * torch.tanh(self.selector_regressor(selector_feature).squeeze(-1))

        result: dict[str, torch.Tensor] = {
            "logits": selected["logits"], "regression": selected["regression"],
            "raw_logits": selected["raw_logits"], "ordinal_logits": selected["ordinal_logits"],
            "base_logits": selected["logits"], "base_regression": selected["regression"],
            "representation": selected["representation"], "selected_query": selected["query"],
            "dense_logits": dense_logits, "dense_regression": dense_regression,
            "modality_logits": modality_logits, "modality_regression": modality_regression,
            "selector_logits": selector_logits, "selector_regression": selector_regression,
            "salience_scores": score, "soft_mask": soft_mask,
            "hard_mask": selection["hard_mask"].reshape(batch_size, 3, steps),
            "selected_indices": indices, "selected_modality": selected_modality,
            "selected_position": selected_position, "selected_payload": selected_payload,
            "selected_valid": selection["selected_valid"],
            "selected_scores": torch.gather(flat_score, 1, indices),
            "all_payload": payload, "all_candidate_mask": candidates,
        }
        if prototype_memory is not None and prototype_memory.ready:
            proto = prototype_memory.retrieve(selected["query"], selected_modality, selection["selected_valid"])
            alpha = float(self.cfg["prototype_fusion_alpha"]) * proto["prototype_available"].to(result["logits"].dtype)
            result.update(proto)
            result["logits"] = (1.0 - alpha[:, None]) * result["base_logits"] + alpha[:, None] * proto["prototype_logits"]
            result["regression"] = (1.0 - alpha) * result["base_regression"] + alpha * proto["prototype_regression"]
        return result
