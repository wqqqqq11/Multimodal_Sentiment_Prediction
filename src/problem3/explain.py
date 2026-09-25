from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .metrics import apply_calibration
from .model import HSAIGNet, MODALITIES
from .explain_utils import (_cosine, _normalize, _rank_correlation, _top_mask, load_evidence_mapping,
                      make_single_batch)


def _target(outputs: dict[str, torch.Tensor], name: str, predicted_class: int) -> torch.Tensor:
    return outputs["logits"][0, predicted_class] if name == "classification" else outputs["regression"][0]


def _integrated_gradient(model: HSAIGNet, batch: dict[str, Any], modality: str,
                         actual: torch.Tensor, baseline: torch.Tensor, target: str,
                         predicted_class: int, steps: int) -> torch.Tensor:
    gradients, delta = [], actual - baseline
    key = "text_features" if modality == "text" else modality
    for alpha in torch.linspace(0.0, 1.0, steps, device=actual.device):
        value = (baseline + alpha * delta).detach().requires_grad_(True)
        modified = dict(batch); modified[key] = value
        scalar = _target(model(modified), target, predicted_class)
        gradients.append(torch.autograd.grad(scalar, value, retain_graph=False)[0].detach())
    return (delta * torch.stack(gradients).mean(dim=0)).detach()


def _baselines(modality: str, actual: torch.Tensor, candidate: torch.Tensor,
               statistics: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    position = statistics[f"{modality}_position_median"].to(actual.device).unsqueeze(0)
    global_value = statistics[f"{modality}_global_median"].to(actual.device).view(1, 1, -1).expand_as(actual)
    return [torch.where(candidate.unsqueeze(-1), value, actual) for value in (position, global_value)]


@torch.inference_mode()
def _predict(model: HSAIGNet, batch: dict[str, Any], calibration: dict[str, Any] | None) -> dict[str, Any]:
    outputs = model(batch)
    probability, classes, regression = apply_calibration(
        outputs["logits"].float().cpu().numpy(), outputs["regression"].float().cpu().numpy(), calibration)
    return {"outputs": outputs, "probabilities": probability[0], "prediction_class": int(classes[0]),
            "prediction_regression": float(regression[0])}


def _perturb(batch: dict[str, Any], selected: dict[str, np.ndarray], statistics: dict[str, torch.Tensor],
             retain: bool) -> dict[str, Any]:
    modified = dict(batch)
    for modality in MODALITIES:
        key = "text_features" if modality == "text" else modality
        candidate = batch[f"{modality}_evidence_candidate_mask"].bool()
        chosen = torch.as_tensor(selected[modality], device=candidate.device).unsqueeze(0)
        change = (candidate & ~chosen) if retain else chosen
        reference = statistics[f"{modality}_position_median"].to(candidate.device).unsqueeze(0)
        modified[key] = torch.where(change.unsqueeze(-1), reference, batch[key])
    return modified


def _effect(original: dict[str, Any], changed: dict[str, Any]) -> float:
    cls = original["prediction_class"]
    probability_drop = max(0.0, float(original["probabilities"][cls] - changed["probabilities"][cls]))
    regression_change = abs(original["prediction_regression"] - changed["prediction_regression"]) / 6.0
    return probability_drop + regression_change


def explain_sample(model: HSAIGNet, item: dict[str, Any], device: torch.device,
                   statistics: dict[str, torch.Tensor], cfg: dict[str, Any],
                   calibration: dict[str, Any] | None,
                   mapping: dict[tuple[str, int], dict[str, str]]) -> dict[str, Any]:
    model.eval()
    batch = make_single_batch(item, device)
    original = _predict(model, batch, calibration)
    predicted_class, steps = original["prediction_class"], int(cfg["integrated_gradient_steps"])
    modality_scores, baseline_agreements = {}, []
    for modality in MODALITIES:
        key = "text_features" if modality == "text" else modality
        actual = batch[key].float().detach()
        candidate = batch[f"{modality}_evidence_candidate_mask"].bool()
        scores = []
        for baseline in _baselines(modality, actual, candidate, statistics):
            cls_ig = _integrated_gradient(model, batch, modality, actual, baseline, "classification",
                                          predicted_class, steps).abs().sum(-1)[0]
            reg_ig = _integrated_gradient(model, batch, modality, actual, baseline, "regression",
                                          predicted_class, steps).abs().sum(-1)[0]
            combined = (float(cfg["classification_weight"]) * cls_ig +
                        float(cfg["regression_weight"]) * reg_ig) * candidate[0].float()
            scores.append(combined.cpu().numpy())
        modality_scores[modality] = np.mean(scores, axis=0)
        baseline_agreements.append(_cosine(_normalize(scores[0]), _normalize(scores[1])))
    selected = {name: _top_mask(modality_scores[name],
                                 batch[f"{name}_evidence_candidate_mask"][0].bool().cpu().numpy(),
                                 float(cfg["top_k_ratio"])) for name in MODALITIES}
    deletion_by_modality = []
    for modality in MODALITIES:
        isolated = {name: np.zeros(50, dtype=bool) for name in MODALITIES}
        isolated[modality] = selected[modality]
        deletion_by_modality.append(_effect(original, _predict(model, _perturb(batch, isolated, statistics, False), calibration)))
    deleted = _predict(model, _perturb(batch, selected, statistics, False), calibration)
    retained = _predict(model, _perturb(batch, selected, statistics, True), calibration)
    gate = original["outputs"]["modality_gate"][0].detach().cpu().numpy()
    ig_total = _normalize(np.asarray([modality_scores[name].sum() for name in MODALITIES]))
    deletion_total = _normalize(np.asarray(deletion_by_modality))
    weights = cfg["contribution_weights"]
    contribution = _normalize(float(weights["gate"]) * gate + float(weights["integrated_gradients"]) * ig_total +
                              float(weights["deletion"]) * deletion_total)
    attentions = {name: original["outputs"][f"time_attention_{name}"][0].detach().cpu().numpy()
                  for name in MODALITIES}
    agreements, evidence, sample_id = [], [], str(item["sample_id"])
    for modality_index, modality in enumerate(MODALITIES):
        candidate = batch[f"{modality}_evidence_candidate_mask"][0].bool().cpu().numpy()
        normalized = _normalize(modality_scores[modality])
        agreements.append(_rank_correlation(attentions[modality], normalized, candidate))
        positions = np.flatnonzero(candidate)
        positions = positions[np.argsort(normalized[positions])[::-1]]
        for rank, position in enumerate(positions[:int(cfg["max_evidence_per_modality"])], 1):
            mapped = mapping.get((sample_id, int(position)), {})
            evidence.append({
                "sample_id": sample_id, "modality": modality, "rank": rank, "position": int(position),
                "position_score": float(normalized[position]), "attention_score": float(attentions[modality][position]),
                "modality_score": float(contribution[modality_index]), "text_span": mapped.get("text_span", ""),
                "start_sec": float(mapped["start_sec"]) if mapped.get("start_sec") else None,
                "end_sec": float(mapped["end_sec"]) if mapped.get("end_sec") else None,
                "start_frame": int(mapped["start_frame"]) if mapped.get("start_frame") else None,
                "end_frame": int(mapped["end_frame"]) if mapped.get("end_frame") else None,
                "representative_frame": int(mapped["representative_frame"]) if mapped.get("representative_frame") else None,
                "mapping_method": mapped.get("mapping_method", ""),
                "mapping_confidence": float(mapped["mapping_confidence"]) if mapped.get("mapping_confidence") else None})
    main = int(np.argmax(contribution))
    return {
        "sample_id": sample_id, "raw_text": str(item["raw_text"]), "predicted_polarity": predicted_class,
        "predicted_label": ("negative", "neutral", "positive")[predicted_class],
        "predicted_intensity": original["prediction_regression"],
        "prediction_confidence": float(max(original["probabilities"])),
        "probabilities": original["probabilities"].tolist(), "main_modality": MODALITIES[main],
        "modality_contribution": {name: float(contribution[i]) for i, name in enumerate(MODALITIES)},
        "gate_weight": {name: float(gate[i]) for i, name in enumerate(MODALITIES)},
        "ig_weight": {name: float(ig_total[i]) for i, name in enumerate(MODALITIES)},
        "deletion_weight": {name: float(deletion_total[i]) for i, name in enumerate(MODALITIES)},
        "comprehensiveness": float(_effect(original, deleted)),
        "sufficiency": float(np.clip(1.0 - _effect(original, retained), 0.0, 1.0)),
        "stability": float(np.mean(baseline_agreements)), "attention_ig_agreement": float(np.mean(agreements)),
        "position_scores": {name: _normalize(modality_scores[name]).tolist() for name in MODALITIES},
        "time_attention": {name: attentions[name].tolist() for name in MODALITIES}, "key_evidence": evidence}


__all__ = ["explain_sample", "load_evidence_mapping"]
