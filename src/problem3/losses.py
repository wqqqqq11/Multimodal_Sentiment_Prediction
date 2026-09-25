from __future__ import annotations

from typing import Any

import torch
from torch import nn


def pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = torch.sqrt((prediction.square().sum() * target.square().sum()).clamp_min(1e-8))
    return 1.0 - (prediction * target).sum() / denominator


def coherence_loss(logits: torch.Tensor, regression: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=-1)
    expected_polarity = probability @ logits.new_tensor([-1.0, 0.0, 1.0])
    normalized_intensity = regression / 3.0
    return torch.nn.functional.smooth_l1_loss(normalized_intensity, expected_polarity)


def js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = torch.softmax(first, dim=-1)
    second = torch.softmax(second, dim=-1)
    middle = 0.5 * (first + second)
    return 0.5 * (
        torch.nn.functional.kl_div(middle.log(), first, reduction="batchmean")
        + torch.nn.functional.kl_div(middle.log(), second, reduction="batchmean")
    )


class Problem3Loss(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        training = cfg["training"]
        self.weights = {key: float(value) for key, value in training["loss_weights"].items()}
        self.classification_margin = float(training["classification_margin"])
        self.regression_margin = float(training["regression_margin"])
        self.distillation_temperature = float(training.get("distillation_temperature", 2.0))
        self.register_buffer("class_weights", torch.tensor(training["class_weights"], dtype=torch.float32))

    def _teacher_distillation(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = outputs["logits"].new_zeros(())
        if "teacher_logits" not in outputs:
            return zero, zero
        temperature = self.distillation_temperature
        teacher_classification = torch.nn.functional.kl_div(
            torch.log_softmax(outputs["logits"] / temperature, dim=-1),
            torch.softmax(outputs["teacher_logits"].detach() / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        teacher_regression = torch.nn.functional.smooth_l1_loss(
            outputs["regression"], outputs["teacher_regression"].detach()
        )
        return teacher_classification, teacher_regression

    def _supervised(self, logits: torch.Tensor, regression: torch.Tensor, labels: torch.Tensor, intensity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        classification = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        regression_loss = torch.nn.functional.smooth_l1_loss(regression, intensity)
        correlation = pearson_loss(regression, intensity)
        coherence = coherence_loss(logits, regression)
        return classification, regression_loss, correlation, coherence

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        complement: dict[str, torch.Tensor] | None = None,
        perturbed: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["classification_labels"].long()
        intensity = batch["regression_labels"].float()
        classification, regression, correlation, coherence = self._supervised(outputs["logits"], outputs["regression"], labels, intensity)
        teacher_classification, teacher_regression = self._teacher_distillation(outputs)
        ordinal_consistency = torch.nn.functional.cross_entropy(
            outputs["ordinal_logits"], labels, weight=self.class_weights
        )
        dense_terms = self._supervised(outputs["dense_logits"], outputs["dense_regression"], labels, intensity)
        dense_auxiliary = dense_terms[0] + dense_terms[1] + 0.2 * dense_terms[2]
        modality_labels = labels[:, None].expand(-1, 3).reshape(-1)
        modality_intensity = intensity[:, None].expand(-1, 3).reshape(-1)
        modality_auxiliary = (
            torch.nn.functional.cross_entropy(outputs["modality_logits"].reshape(-1, 3), modality_labels, weight=self.class_weights)
            + torch.nn.functional.smooth_l1_loss(outputs["modality_regression"].reshape(-1), modality_intensity)
        )
        selector_auxiliary = (
            torch.nn.functional.cross_entropy(outputs["selector_logits"], labels, weight=self.class_weights)
            + torch.nn.functional.smooth_l1_loss(outputs["selector_regression"], intensity)
        )
        reference_logits = outputs.get("teacher_logits", outputs["dense_logits"]).detach()
        reference_regression = outputs.get("teacher_regression", outputs["dense_regression"]).detach()
        dense_probability = torch.softmax(reference_logits, dim=-1)
        sufficiency = js_divergence(outputs["logits"], reference_logits)
        sufficiency = sufficiency + torch.nn.functional.smooth_l1_loss(outputs["regression"], reference_regression)

        comprehensiveness = outputs["logits"].new_zeros(())
        if complement is not None:
            reference_class = dense_probability.argmax(dim=-1)
            selected_probability = torch.softmax(outputs["logits"], dim=-1).gather(1, reference_class[:, None]).squeeze(1)
            complement_probability = torch.softmax(complement["logits"], dim=-1).gather(1, reference_class[:, None]).squeeze(1)
            class_comp = torch.relu(self.classification_margin - (selected_probability - complement_probability)).mean()
            selected_distance = (outputs["regression"] - reference_regression).abs()
            complement_distance = (complement["regression"] - reference_regression).abs()
            reg_comp = torch.relu(self.regression_margin - (complement_distance - selected_distance)).mean()
            comprehensiveness = class_comp + reg_comp

        stability = outputs["logits"].new_zeros(())
        if perturbed is not None:
            stability = torch.nn.functional.mse_loss(outputs["soft_mask"], perturbed["soft_mask"])
            stability = stability + js_divergence(outputs["logits"], perturbed["logits"])
            stability = stability + 0.5 * torch.nn.functional.smooth_l1_loss(outputs["regression"], perturbed["regression"])

        prototype = outputs["logits"].new_zeros(())
        if "prototype_logits" in outputs:
            available = outputs["prototype_available"].bool()
            if bool(available.any()):
                prototype = (
                    torch.nn.functional.cross_entropy(outputs["prototype_logits"][available], labels[available], weight=self.class_weights)
                    + torch.nn.functional.smooth_l1_loss(outputs["prototype_regression"][available], intensity[available])
                )
        selected_distribution = outputs["soft_mask"].sum(dim=(0, 2)) / outputs["soft_mask"].sum().clamp_min(1.0)
        candidate_distribution = outputs["all_candidate_mask"].float().sum(dim=(0, 2))
        candidate_distribution = candidate_distribution / candidate_distribution.sum().clamp_min(1.0)
        sparsity_balance = torch.nn.functional.mse_loss(selected_distribution, candidate_distribution)

        terms = {
            "classification": classification,
            "regression": regression,
            "correlation": correlation,
            "coherence": coherence,
            "ordinal_consistency": ordinal_consistency,
            "teacher_classification": teacher_classification,
            "teacher_regression": teacher_regression,
            "dense_auxiliary": dense_auxiliary,
            "modality_auxiliary": modality_auxiliary,
            "selector_auxiliary": selector_auxiliary,
            "prototype": prototype,
            "sufficiency": sufficiency,
            "comprehensiveness": comprehensiveness,
            "stability": stability,
            "sparsity_balance": sparsity_balance,
        }
        total = sum(self.weights.get(name, 0.0) * value for name, value in terms.items())
        return total, {name: float(value.detach()) for name, value in {"total": total, **terms}.items()}


def perturb_batch(
    batch: dict[str, Any],
    mask_token_id: int = 103,
    probability: float = 0.05,
    noise_std: float = 0.02,
) -> dict[str, Any]:
    result = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    text_candidate = result["text_evidence_candidate_mask"].bool()
    replace = (torch.rand_like(text_candidate.float()) < probability) & text_candidate
    result["input_ids"][replace] = mask_token_id
    for modality in ("audio", "vision"):
        candidate = result[f"{modality}_evidence_candidate_mask"].bool().unsqueeze(-1)
        noise = torch.randn_like(result[modality].float()) * noise_std
        result[modality] = result[modality].float() + noise * candidate.to(noise.dtype)
    return result
