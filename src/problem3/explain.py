from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import apply_calibration
from .model import HSAIGNet, MODALITIES
from .utils import move_to_device


def load_evidence_mapping(path: str | Path) -> dict[tuple[str, int], dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(str(row["sample_id"]), int(row["position"])): row for row in rows}


def make_single_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif key in {"sample_id", "raw_text", "video_path"}:
            batch[key] = [value]
        else:
            batch[key] = value
    return move_to_device(batch, device)


def _target_scalar(outputs: dict[str, torch.Tensor], target: str, predicted_class: int) -> torch.Tensor:
    if target == "classification":
        return outputs["logits"][0, predicted_class]
    if target == "regression":
        return outputs["regression"][0]
    raise ValueError(f"未知积分梯度目标: {target}")


def _integrated_gradient(
    model: HSAIGNet,
    batch: dict[str, Any],
    modality: str,
    actual: torch.Tensor,
    baseline: torch.Tensor,
    *,
    target: str,
    predicted_class: int,
    steps: int,
) -> torch.Tensor:
    gradients: list[torch.Tensor] = []
    delta = actual - baseline
    for alpha in torch.linspace(0.0, 1.0, int(steps), device=actual.device):
        interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
        if modality == "text":
            outputs = model(batch, text_embeddings=interpolated)
        else:
            modified = dict(batch)
            modified[modality] = interpolated
            outputs = model(modified)
        scalar = _target_scalar(outputs, target, predicted_class)
        gradient = torch.autograd.grad(scalar, interpolated, retain_graph=False, create_graph=False)[0]
        gradients.append(gradient.detach())
    average_gradient = torch.stack(gradients).mean(dim=0)
    return (delta * average_gradient).detach()


def _continuous_baselines(
    modality: str,
    actual: torch.Tensor,
    candidate_mask: torch.Tensor,
    baselines: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    position = baselines[f"{modality}_position_median"].to(actual.device).unsqueeze(0)
    global_value = baselines[f"{modality}_global_median"].to(actual.device).view(1, 1, -1).expand_as(actual)
    mask = candidate_mask.unsqueeze(-1)
    return [torch.where(mask, value, actual) for value in (position, global_value)]


def _text_baselines(
    model: HSAIGNet,
    batch: dict[str, Any],
    token_ids: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    actual = model.text_word_embeddings(batch["input_ids"].long()).detach()
    content = batch["content_mask"].bool()
    result: list[torch.Tensor] = []
    for token_id in token_ids:
        ids = batch["input_ids"].long().clone()
        ids[content] = int(token_id)
        result.append(model.text_word_embeddings(ids).detach())
    return actual, result


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    total = float(vector.sum())
    return vector / total if total > 1e-12 else np.zeros_like(vector)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 1.0


def _rank_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(left)[mask]
    y = np.asarray(right)[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.inference_mode()
def _predict_one(
    model: HSAIGNet, batch: dict[str, Any], calibration: dict[str, Any] | None
) -> dict[str, Any]:
    outputs = model(batch)
    logits = outputs["logits"].float().cpu().numpy()
    regression = outputs["regression"].float().cpu().numpy()
    probabilities, classes, calibrated_regression = apply_calibration(logits, regression, calibration)
    return {
        "outputs": outputs,
        "probabilities": probabilities[0],
        "prediction_class": int(classes[0]),
        "prediction_regression": float(calibrated_regression[0]),
    }


def _top_mask(scores: np.ndarray, candidate: np.ndarray, ratio: float) -> np.ndarray:
    indexes = np.flatnonzero(candidate)
    result = np.zeros_like(candidate, dtype=bool)
    if not len(indexes):
        return result
    count = max(1, min(len(indexes), int(math.ceil(len(indexes) * float(ratio)))))
    selected = indexes[np.argsort(scores[indexes])[-count:]]
    result[selected] = True
    return result


def _perturb(
    model: HSAIGNet,
    batch: dict[str, Any],
    selected: dict[str, np.ndarray],
    baselines: dict[str, torch.Tensor],
    text_token_id: int,
    *,
    retain: bool,
) -> dict[str, Any]:
    modified = dict(batch)
    content = batch["content_mask"].bool()
    text_selected = torch.as_tensor(selected["text"], device=content.device).unsqueeze(0)
    text_change = (content & ~text_selected) if retain else text_selected
    ids = batch["input_ids"].clone()
    ids[text_change] = int(text_token_id)
    modified["input_ids"] = ids
    for modality in ("audio", "vision"):
        candidate = batch[f"{modality}_evidence_candidate_mask"].bool()
        chosen = torch.as_tensor(selected[modality], device=candidate.device).unsqueeze(0)
        change = (candidate & ~chosen) if retain else chosen
        reference = baselines[f"{modality}_position_median"].to(candidate.device).unsqueeze(0)
        modified[modality] = torch.where(change.unsqueeze(-1), reference, batch[modality])
    return modified


def _effect(original: dict[str, Any], changed: dict[str, Any]) -> float:
    cls = original["prediction_class"]
    probability_drop = max(0.0, float(original["probabilities"][cls] - changed["probabilities"][cls]))
    regression_change = abs(float(original["prediction_regression"] - changed["prediction_regression"])) / 6.0
    return probability_drop + regression_change


def explain_sample(
    model: HSAIGNet,
    item: dict[str, Any],
    device: torch.device,
    baselines: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    calibration: dict[str, Any] | None,
    mapping: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    model.eval()
    batch = make_single_batch(item, device)
    original = _predict_one(model, batch, calibration)
    predicted_class = int(original["prediction_class"])
    steps = int(cfg["integrated_gradient_steps"])
    class_weight = float(cfg["classification_weight"])
    regression_weight = float(cfg["regression_weight"])
    modality_scores: dict[str, np.ndarray] = {}
    baseline_agreements: list[float] = []
    for modality in MODALITIES:
        candidate_tensor = batch[f"{modality}_evidence_candidate_mask"].bool()
        if modality == "text":
            actual, references = _text_baselines(model, batch, [int(v) for v in cfg["text_baseline_token_ids"]])
        else:
            actual = batch[modality].float().detach()
            references = _continuous_baselines(modality, actual, candidate_tensor, baselines)
        per_baseline: list[np.ndarray] = []
        for reference in references:
            classification = _integrated_gradient(
                model, batch, modality, actual, reference,
                target="classification", predicted_class=predicted_class, steps=steps,
            ).abs().sum(dim=-1)[0]
            regression = _integrated_gradient(
                model, batch, modality, actual, reference,
                target="regression", predicted_class=predicted_class, steps=steps,
            ).abs().sum(dim=-1)[0]
            combined = class_weight * classification + regression_weight * regression
            combined = combined * candidate_tensor[0].float()
            per_baseline.append(combined.cpu().numpy())
        modality_scores[modality] = np.mean(per_baseline, axis=0)
        if len(per_baseline) > 1:
            baseline_agreements.append(_cosine(_normalize(per_baseline[0]), _normalize(per_baseline[1])))
    selected = {
        modality: _top_mask(
            modality_scores[modality],
            batch[f"{modality}_evidence_candidate_mask"][0].bool().cpu().numpy(),
            float(cfg["top_k_ratio"]),
        )
        for modality in MODALITIES
    }
    deletion_by_modality: list[float] = []
    for modality in MODALITIES:
        isolated = {name: np.zeros(50, dtype=bool) for name in MODALITIES}
        isolated[modality] = selected[modality]
        changed = _predict_one(
            model,
            _perturb(model, batch, isolated, baselines, int(cfg["text_baseline_token_ids"][0]), retain=False),
            calibration,
        )
        deletion_by_modality.append(_effect(original, changed))
    deleted = _predict_one(
        model,
        _perturb(model, batch, selected, baselines, int(cfg["text_baseline_token_ids"][0]), retain=False),
        calibration,
    )
    retained = _predict_one(
        model,
        _perturb(model, batch, selected, baselines, int(cfg["text_baseline_token_ids"][0]), retain=True),
        calibration,
    )
    comprehensiveness = _effect(original, deleted)
    sufficiency = float(np.clip(1.0 - _effect(original, retained), 0.0, 1.0))
    gate = original["outputs"]["modality_gate"][0].detach().cpu().numpy()
    ig_total = _normalize(np.asarray([modality_scores[name].sum() for name in MODALITIES]))
    deletion_total = _normalize(np.asarray(deletion_by_modality))
    contribution_cfg = cfg["contribution_weights"]
    contribution = _normalize(
        float(contribution_cfg["gate"]) * gate
        + float(contribution_cfg["integrated_gradients"]) * ig_total
        + float(contribution_cfg["deletion"]) * deletion_total
    )
    attentions = {
        name: original["outputs"][f"time_attention_{name}"][0].detach().cpu().numpy() for name in MODALITIES
    }
    agreements = []
    evidence: list[dict[str, Any]] = []
    sample_id = str(item["sample_id"])
    for modality_index, modality in enumerate(MODALITIES):
        candidate = batch[f"{modality}_evidence_candidate_mask"][0].bool().cpu().numpy()
        normalized_scores = _normalize(modality_scores[modality])
        agreements.append(_rank_correlation(attentions[modality], normalized_scores, candidate))
        positions = np.flatnonzero(candidate)
        positions = positions[np.argsort(normalized_scores[positions])[::-1]]
        for rank, position in enumerate(positions[: int(cfg["max_evidence_per_modality"])], start=1):
            mapped = mapping.get((sample_id, int(position)), {})
            evidence.append({
                "sample_id": sample_id,
                "modality": modality,
                "rank": rank,
                "position": int(position),
                "position_score": float(normalized_scores[position]),
                "attention_score": float(attentions[modality][position]),
                "modality_score": float(contribution[modality_index]),
                "text_span": mapped.get("text_span", ""),
                "start_sec": float(mapped["start_sec"]) if mapped.get("start_sec") else None,
                "end_sec": float(mapped["end_sec"]) if mapped.get("end_sec") else None,
                "start_frame": int(mapped["start_frame"]) if mapped.get("start_frame") else None,
                "end_frame": int(mapped["end_frame"]) if mapped.get("end_frame") else None,
                "representative_frame": int(mapped["representative_frame"]) if mapped.get("representative_frame") else None,
                "mapping_method": mapped.get("mapping_method", ""),
                "mapping_confidence": float(mapped["mapping_confidence"]) if mapped.get("mapping_confidence") else None,
            })
    main_index = int(np.argmax(contribution))
    return {
        "sample_id": sample_id,
        "raw_text": str(item["raw_text"]),
        "predicted_polarity": predicted_class,
        "predicted_label": ("negative", "neutral", "positive")[predicted_class],
        "predicted_intensity": float(original["prediction_regression"]),
        "prediction_confidence": float(max(original["probabilities"])),
        "probabilities": original["probabilities"].tolist(),
        "main_modality": MODALITIES[main_index],
        "modality_contribution": {name: float(contribution[index]) for index, name in enumerate(MODALITIES)},
        "gate_weight": {name: float(gate[index]) for index, name in enumerate(MODALITIES)},
        "ig_weight": {name: float(ig_total[index]) for index, name in enumerate(MODALITIES)},
        "deletion_weight": {name: float(deletion_total[index]) for index, name in enumerate(MODALITIES)},
        "comprehensiveness": float(comprehensiveness),
        "sufficiency": sufficiency,
        "stability": float(np.mean(baseline_agreements)) if baseline_agreements else 1.0,
        "attention_ig_agreement": float(np.mean(agreements)),
        "position_scores": {name: _normalize(modality_scores[name]).tolist() for name in MODALITIES},
        "time_attention": {name: attentions[name].tolist() for name in MODALITIES},
        "key_evidence": evidence,
    }
