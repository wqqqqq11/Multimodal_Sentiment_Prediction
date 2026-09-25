from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F

def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = prediction - prediction.mean(), target - target.mean()
    return 1.0 - (x * y).sum() / (torch.sqrt(x.square().sum() * y.square().sum()) + 1e-8)

def _ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    covariance = ((prediction - prediction.mean()) * (target - target.mean())).mean()
    denominator = prediction.var(unbiased=False) + target.var(unbiased=False) + (prediction.mean() - target.mean()).square()
    return 1.0 - 2.0 * covariance / (denominator + 1e-8)

def _entropy(probability: torch.Tensor) -> torch.Tensor:
    safe = probability.clamp_min(1e-8)
    return -(probability * safe.log()).sum(dim=-1).mean()

def supervised_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], class_weights: torch.Tensor,
                    cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels_cls, labels_reg = batch["classification_labels"].long(), batch["regression_labels"].float()
    classification = F.cross_entropy(outputs["logits"], labels_cls, weight=class_weights,
                                     label_smoothing=float(cfg["label_smoothing"]))
    regression = F.smooth_l1_loss(outputs["regression"], labels_reg, beta=float(cfg["regression_beta"]))
    mae = F.l1_loss(outputs["regression"], labels_reg)
    correlation, ccc = _correlation_loss(outputs["regression"], labels_reg), _ccc_loss(outputs["regression"], labels_reg)
    expected = (torch.softmax(outputs["logits"], dim=-1) *
                torch.tensor([-1.0, 0.0, 1.0], device=labels_reg.device)).sum(dim=-1)
    coherence = F.smooth_l1_loss(outputs["regression"] / 3.0, expected)
    aux_cls = torch.stack([F.cross_entropy(outputs["aux_logits"][:, i], labels_cls, weight=class_weights,
                                           label_smoothing=float(cfg["label_smoothing"])) for i in range(3)]).mean()
    aux_reg = F.smooth_l1_loss(outputs["aux_regression"], labels_reg.unsqueeze(1).expand(-1, 3),
                               beta=float(cfg["regression_beta"]))
    temporal = torch.stack([_entropy(outputs[f"time_attention_{name}"]) for name in ("text", "audio", "vision")]).mean()
    modality = _entropy(outputs["modality_gate"])
    parts = {"classification": classification, "regression": regression, "mae": mae,
             "correlation": correlation, "ccc": ccc, "coherence": coherence,
             "aux_classification": aux_cls, "aux_regression": aux_reg,
             "temporal_sparsity": temporal, "modality_sparsity": modality}
    total = sum(float(cfg["loss_weights"][name]) * value for name, value in parts.items())
    return total, parts
