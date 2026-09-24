from __future__ import annotations

import math

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

    error_square = (outputs["regression"] - labels_reg).square()
    log_variance = outputs["log_variance"]
    uncertainty = (0.5 * torch.exp(-log_variance) * error_square + 0.5 * log_variance).mean()

    expert_logits = outputs["expert_logits"]
    expert_regression = outputs["expert_regression"]
    expert_classification = torch.stack(
        [F.cross_entropy(expert_logits[:, index], labels_cls, weight=class_weights) for index in range(3)]
    ).mean()
    expert_regression_loss = torch.stack(
        [F.smooth_l1_loss(expert_regression[:, index], labels_reg, beta=0.5) for index in range(3)]
    ).mean()
    router_prior = F.kl_div(
        torch.log(outputs["gates"].clamp_min(1e-8)), outputs["router_prior"].detach(), reduction="batchmean"
    )
    total = (
        weights["classification"] * classification
        + weights["regression"] * regression
        + weights["correlation"] * correlation
        + weights["coherence"] * coherence
        + weights.get("uncertainty", 0.0) * uncertainty
        + weights.get("expert_aux_classification", 0.0) * expert_classification
        + weights.get("expert_aux_regression", 0.0) * expert_regression_loss
        + weights.get("router_prior", 0.0) * router_prior
    )
    return total, {
        "classification": classification,
        "regression": regression,
        "correlation": correlation,
        "coherence": coherence,
        "uncertainty": uncertainty,
        "expert_aux_classification": expert_classification,
        "expert_aux_regression": expert_regression_loss,
        "router_prior": router_prior,
    }


def distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    temperature: float,
    weights: dict[str, float],
    batch: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_probability = torch.softmax(teacher["logits"] / temperature, dim=-1)
    entropy = -(teacher_probability * torch.log(teacher_probability.clamp_min(1e-8))).sum(dim=-1)
    confidence = (1.0 - entropy / math.log(teacher_probability.shape[-1])).detach().clamp(0.05, 1.0)
    class_per_sample = F.kl_div(
        F.log_softmax(student["logits"] / temperature, dim=-1), teacher_probability, reduction="none"
    ).sum(dim=-1) * (temperature**2)
    kd_classification = (class_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    regression_per_sample = F.smooth_l1_loss(
        student["regression"], teacher["regression"], reduction="none", beta=0.25
    )
    kd_regression = (regression_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)

    feature_per_sample = 1.0 - F.cosine_similarity(student["fused"], teacher["fused"], dim=-1)
    kd_feature = (feature_per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)
    student_normalized = F.normalize(student["fused"], dim=-1)
    teacher_normalized = F.normalize(teacher["fused"], dim=-1)
    kd_relation = F.mse_loss(student_normalized @ student_normalized.T, teacher_normalized @ teacher_normalized.T)

    token_error = F.smooth_l1_loss(student["text_tokens"], teacher["text_tokens"], reduction="none", beta=0.25).mean(-1)
    token_cosine = 1.0 - F.cosine_similarity(student["text_tokens"], teacher["text_tokens"], dim=-1)
    if batch is None:
        token_mask = torch.ones_like(token_error)
    else:
        token_mask = batch["attention_mask"].float()
    kd_token = ((token_error + token_cosine) * token_mask).sum() / token_mask.sum().clamp_min(1.0)
    kd_cls = (
        F.smooth_l1_loss(student["text_vector"], teacher["text_vector"], beta=0.25)
        + (1.0 - F.cosine_similarity(student["text_vector"], teacher["text_vector"], dim=-1)).mean()
    )
    total = (
        weights["kd_classification"] * kd_classification
        + weights["kd_regression"] * kd_regression
        + weights.get("kd_token", 0.0) * kd_token
        + weights.get("kd_cls", 0.0) * kd_cls
        + weights["kd_feature"] * kd_feature
        + weights["kd_relation"] * kd_relation
    )
    return total, {
        "kd_classification": kd_classification,
        "kd_regression": kd_regression,
        "kd_token": kd_token,
        "kd_cls": kd_cls,
        "kd_feature": kd_feature,
        "kd_relation": kd_relation,
    }
