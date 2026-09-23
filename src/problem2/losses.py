from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = torch.sqrt((prediction.square().sum() + 1e-8) * (target.square().sum() + 1e-8))
    return 1.0 - (prediction * target).sum() / denominator


def supervised_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels_cls = batch["classification_labels"].long()
    labels_reg = batch["regression_labels"].float()
    classification = F.cross_entropy(outputs["logits"], labels_cls, weight=class_weights)
    regression = F.smooth_l1_loss(outputs["regression"], labels_reg, beta=0.5)
    correlation = _pearson_loss(outputs["regression"], labels_reg)
    probabilities = torch.softmax(outputs["logits"], dim=-1)
    class_axis = torch.tensor([-1.0, 0.0, 1.0], device=probabilities.device)
    polarity_expectation = (probabilities * class_axis).sum(dim=-1)
    coherence = F.smooth_l1_loss(outputs["regression"] / 3.0, polarity_expectation)
    text_classification = F.cross_entropy(outputs["text_logits"], labels_cls, weight=class_weights)
    text_regression = F.smooth_l1_loss(outputs["text_regression"], labels_reg, beta=0.5)
    total = (
        weights["classification"] * classification
        + weights["regression"] * regression
        + weights["correlation"] * correlation
        + weights["coherence"] * coherence
        + weights.get("text_aux_classification", 0.0) * text_classification
        + weights.get("text_aux_regression", 0.0) * text_regression
    )
    return total, {
        "classification": classification,
        "regression": regression,
        "correlation": correlation,
        "coherence": coherence,
        "text_aux_classification": text_classification,
        "text_aux_regression": text_regression,
    }


def distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    temperature: float,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_probability = torch.softmax(teacher["logits"] / temperature, dim=-1)
    entropy = -(teacher_probability * torch.log(teacher_probability.clamp_min(1e-8))).sum(dim=-1)
    confidence = (1.0 - entropy / math.log(teacher_probability.shape[-1])).detach().clamp(0.05, 1.0)
    class_per_sample = F.kl_div(
        F.log_softmax(student["logits"] / temperature, dim=-1),
        teacher_probability,
        reduction="none",
    ).sum(dim=-1) * (temperature ** 2)
    kd_classification = (class_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    regression_per_sample = F.smooth_l1_loss(student["regression"], teacher["regression"], reduction="none", beta=0.25)
    kd_regression = (regression_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    feature_per_sample = 1.0 - F.cosine_similarity(student["fused"], teacher["fused"], dim=-1)
    kd_feature = (feature_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    modality_per_sample = 1.0 - F.cosine_similarity(
        student["modalities"], teacher["modalities"], dim=-1
    ).mean(dim=-1)
    kd_modality = (modality_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    student_normalized = F.normalize(student["fused"], dim=-1)
    teacher_normalized = F.normalize(teacher["fused"], dim=-1)
    kd_relation = F.mse_loss(student_normalized @ student_normalized.T, teacher_normalized @ teacher_normalized.T)
    target_gate = teacher["gates"] * student["reliability"][:, :, 0].detach().clamp_min(1e-4)
    target_gate = target_gate / target_gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
    kd_gate = F.kl_div(torch.log(student["gates"].clamp_min(1e-8)), target_gate, reduction="batchmean")
    total = (
        weights["kd_classification"] * kd_classification
        + weights["kd_regression"] * kd_regression
        + weights["kd_feature"] * kd_feature
        + weights.get("kd_modality", 0.0) * kd_modality
        + weights["kd_relation"] * kd_relation
        + weights["kd_gate"] * kd_gate
    )
    return total, {
        "kd_classification": kd_classification,
        "kd_regression": kd_regression,
        "kd_feature": kd_feature,
        "kd_modality": kd_modality,
        "kd_relation": kd_relation,
        "kd_gate": kd_gate,
    }
