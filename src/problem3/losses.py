from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = torch.sqrt(prediction.square().sum() * target.square().sum()).clamp_min(1e-8)
    return 1.0 - (prediction * target).sum() / denominator


def _normalized_entropy(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1e-8)
    entropy = -(weights * safe.log()).sum(dim=1)
    maximum = mask.sum(dim=1).clamp_min(2).float().log()
    return (entropy / maximum).mean()


def supervised_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = cfg["loss_weights"]
    target_cls = batch["classification_labels"].long()
    target_reg = batch["regression_labels"].float()
    classification = F.cross_entropy(
        outputs["logits"], target_cls, weight=class_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    regression = F.smooth_l1_loss(
        outputs["regression"], target_reg, beta=float(cfg.get("regression_beta", 0.5))
    )
    correlation = _pearson_loss(outputs["regression"], target_reg)
    probabilities = torch.softmax(outputs["logits"], dim=1)
    class_axis = torch.tensor([-1.0, 0.0, 1.0], device=probabilities.device)
    expected_polarity = (probabilities * class_axis).sum(dim=1)
    coherence = F.smooth_l1_loss(expected_polarity, outputs["regression"] / 3.0, beta=0.25)
    temporal = torch.stack([
        _normalized_entropy(outputs[f"time_attention_{modality}"], batch[f"{modality}_evidence_candidate_mask"].bool())
        for modality in ("text", "audio", "vision")
    ]).mean()
    modality = _normalized_entropy(
        outputs["modality_gate"],
        torch.stack([
            batch[f"{name}_evidence_candidate_mask"].bool().any(dim=1)
            for name in ("text", "audio", "vision")
        ], dim=1),
    )
    total = (
        float(weights["classification"]) * classification
        + float(weights["regression"]) * regression
        + float(weights["correlation"]) * correlation
        + float(weights["coherence"]) * coherence
        + float(weights["temporal_sparsity"]) * temporal
        + float(weights["modality_sparsity"]) * modality
    )
    return total, {
        "classification": classification,
        "regression": regression,
        "correlation": correlation,
        "coherence": coherence,
        "temporal_sparsity": temporal,
        "modality_sparsity": modality,
    }
