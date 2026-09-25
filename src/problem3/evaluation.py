from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .losses import perturb_batch
from .metrics import all_metrics, bootstrap_intervals, selection_score
from .model import MODALITIES
from .utils import move_to_device


POLARITY_NAMES = ("negative", "neutral", "positive")


def _subset_key(subset: tuple[int, ...]) -> int:
    return sum(1 << item for item in subset)


@torch.no_grad()
def exact_modality_shapley(
    model: Any, outputs: dict[str, torch.Tensor], prototype_memory: Any | None = None,
) -> dict[str, torch.Tensor]:
    batch, top_k = outputs["selected_modality"].shape
    target = outputs["logits"].argmax(dim=-1)
    utilities_cls: dict[int, torch.Tensor] = {}
    utilities_reg: dict[int, torch.Tensor] = {}
    for size in range(4):
        for subset in itertools.combinations(range(3), size):
            keep = torch.zeros(batch, top_k, dtype=torch.bool, device=target.device)
            for modality in subset:
                keep |= outputs["selected_modality"] == modality
            prediction = model.predict_selected_subset(outputs, keep, prototype_memory)
            log_probability = torch.log_softmax(prediction["logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
            utilities_cls[_subset_key(subset)] = log_probability
            utilities_reg[_subset_key(subset)] = prediction["regression"]
    factorial = math.factorial
    shapley_cls = outputs["logits"].new_zeros(batch, 3)
    shapley_reg = outputs["logits"].new_zeros(batch, 3)
    for modality in range(3):
        others = [item for item in range(3) if item != modality]
        for size in range(3):
            weight = factorial(size) * factorial(2 - size) / factorial(3)
            for subset in itertools.combinations(others, size):
                base = _subset_key(subset)
                added = base | (1 << modality)
                shapley_cls[:, modality] += weight * (utilities_cls[added] - utilities_cls[base])
                shapley_reg[:, modality] += weight * (utilities_reg[added] - utilities_reg[base])
    raw = shapley_cls.abs() + 0.25 * shapley_reg.abs()
    total = raw.sum(dim=-1, keepdim=True)
    selected_valid = outputs["selected_valid"].to(raw.dtype)
    fallback = torch.stack([
        ((outputs["selected_modality"] == modality).to(raw.dtype) * selected_valid).sum(dim=1)
        for modality in range(3)
    ], dim=1)
    fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
    contribution = torch.where(total > 1e-8, raw / total.clamp_min(1e-8), fallback)
    return {"shapley_class": shapley_cls, "shapley_regression": shapley_reg, "modality_contribution": contribution}


@torch.no_grad()
def local_leave_one_out(
    model: Any, outputs: dict[str, torch.Tensor], prototype_memory: Any | None = None,
) -> torch.Tensor:
    batch, top_k = outputs["selected_modality"].shape
    target = outputs["logits"].argmax(dim=-1)
    baseline_log_probability = torch.log_softmax(outputs["logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
    importance = outputs["logits"].new_zeros(batch, top_k)
    for rank in range(top_k):
        keep = torch.ones(batch, top_k, dtype=torch.bool, device=target.device)
        keep[:, rank] = False
        deleted = model.predict_selected_subset(outputs, keep, prototype_memory)
        deleted_log_probability = torch.log_softmax(deleted["logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
        class_effect = baseline_log_probability - deleted_log_probability
        regression_effect = (outputs["regression"] - deleted["regression"]).abs()
        importance[:, rank] = torch.relu(class_effect) + 0.25 * regression_effect
    importance = importance * outputs["selected_valid"].to(importance.dtype)
    total = importance.sum(dim=-1, keepdim=True)
    fallback = outputs["selected_valid"].to(importance.dtype)
    fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.where(total > 1e-8, importance / total.clamp_min(1e-8), fallback)


def _fidelity(outputs: dict[str, torch.Tensor], complement: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = outputs["logits"].argmax(dim=-1)
    selected_probability = torch.softmax(outputs["logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
    dense_probability = torch.softmax(outputs["dense_logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
    complement_probability = torch.softmax(complement["logits"], dim=-1).gather(1, target[:, None]).squeeze(1)
    return {
        "sufficiency_probability_gap": (selected_probability - dense_probability).abs(),
        "sufficiency_regression_gap": (outputs["regression"] - outputs["dense_regression"]).abs(),
        "comprehensiveness_probability_drop": selected_probability - complement_probability,
        "comprehensiveness_regression_change": (outputs["regression"] - complement["regression"]).abs(),
    }


@torch.no_grad()
def evaluate_model(
    model: Any,
    loader: Any,
    device: torch.device,
    cfg: dict[str, Any],
    prototype_memory: Any | None = None,
    explain: bool = True,
    teacher: Any | None = None,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    labels_cls: list[int] = []
    labels_reg: list[float] = []
    all_fidelity: dict[str, list[float]] = defaultdict(list)
    stability_jaccard: list[float] = []
    prototype_evidence_total = 0
    prototype_evidence_matched = 0
    prototype_enabled = prototype_memory is not None and prototype_memory.ready
    for batch in loader:
        sample_ids = list(batch["sample_id"])
        raw_texts = list(batch["raw_text"])
        batch = move_to_device(batch, device)
        outputs = model(batch, prototype_memory=prototype_memory)
        if teacher is not None:
            teacher_outputs = teacher(batch)
            outputs["teacher_logits"] = teacher_outputs["logits"]
            outputs["teacher_regression"] = teacher_outputs["regression"]
            outputs["dense_logits"] = teacher_outputs["logits"]
            outputs["dense_regression"] = teacher_outputs["regression"]
        complement = model.predict_complement(outputs, prototype_memory)
        probability = torch.softmax(outputs["logits"], dim=-1)
        prediction = probability.argmax(dim=-1)
        contributions = exact_modality_shapley(model, outputs, prototype_memory) if explain else None
        local_importance = local_leave_one_out(model, outputs, prototype_memory) if explain else None
        fidelity = _fidelity(outputs, complement)
        for key, value in fidelity.items():
            all_fidelity[key].extend(value.detach().cpu().tolist())

        if explain:
            repeats = int(cfg["evaluation"].get("stability_repeats", 0))
            original = outputs["hard_mask"].bool()
            for _ in range(repeats):
                counterfactual_cfg = cfg.get("counterfactual", {})
                perturbed_batch = perturb_batch(
                    batch,
                    mask_token_id=int(counterfactual_cfg.get("mask_token_id", 103)),
                    probability=float(counterfactual_cfg.get("non_evidence_mask_ratio", 0.05)),
                    noise_std=float(counterfactual_cfg.get("gaussian_noise_std", 0.02)),
                )
                perturbed = model(perturbed_batch, prototype_memory=prototype_memory)
                other = perturbed["hard_mask"].bool()
                intersection = (original & other).sum(dim=(1, 2)).float()
                union = (original | other).sum(dim=(1, 2)).float().clamp_min(1.0)
                stability_jaccard.extend((intersection / union).cpu().tolist())

        for row, sample_id in enumerate(sample_ids):
            contribution = contributions["modality_contribution"][row].cpu().tolist() if contributions else [0.0, 0.0, 0.0]
            main_modality = int(np.argmax(contribution)) if explain else -1
            record = {
                "sample_id": str(sample_id), "raw_text": str(raw_texts[row]),
                "predicted_class": int(prediction[row]), "predicted_polarity": POLARITY_NAMES[int(prediction[row])],
                "class_probability_negative": float(probability[row, 0]),
                "class_probability_neutral": float(probability[row, 1]),
                "class_probability_positive": float(probability[row, 2]),
                "predicted_intensity": float(outputs["regression"][row]),
                "main_modality": MODALITIES[main_modality] if main_modality >= 0 else "not_computed",
                "text_contribution": float(contribution[0]), "audio_contribution": float(contribution[1]),
                "vision_contribution": float(contribution[2]),
                **{key: float(value[row]) for key, value in fidelity.items()},
            }
            if "classification_labels" in batch:
                true_cls = int(batch["classification_labels"][row])
                true_reg = float(batch["regression_labels"][row])
                record.update({
                    "true_class": true_cls, "true_intensity": true_reg,
                    "classification_correct": true_cls == int(prediction[row]),
                    "absolute_error": abs(true_reg - float(outputs["regression"][row])),
                })
                labels_cls.append(true_cls)
                labels_reg.append(true_reg)
            records.append(record)
            if not explain:
                continue
            for rank in range(outputs["selected_indices"].shape[1]):
                if not bool(outputs["selected_valid"][row, rank]):
                    continue
                modality_id = int(outputs["selected_modality"][row, rank])
                prototype_index = int(outputs["nearest_prototype_index"][row, rank]) if "nearest_prototype_index" in outputs else -1
                prototype_matched = prototype_index >= 0 and prototype_enabled
                prototype_status = "matched" if prototype_matched else (
                    "no_modality_prototype" if prototype_enabled else "memory_not_loaded"
                )
                prototype_meta = prototype_memory.metadata[prototype_index] if prototype_matched else {}
                center_position = int(outputs["selected_position"][row, rank])
                window_radius = int(cfg["model"].get("evidence_window_radius", 0))
                prototype_evidence_total += 1
                prototype_evidence_matched += int(prototype_matched)
                evidence_rows.append({
                    "sample_id": str(sample_id), "rank": rank + 1, "modality": MODALITIES[modality_id],
                    "modality_id": modality_id, "position": center_position,
                    "window_start": max(0, center_position - window_radius),
                    "window_end": min(int(cfg["model"]["max_steps"]) - 1, center_position + window_radius),
                    "selector_score": float(outputs["selected_scores"][row, rank]),
                    "local_importance": float(local_importance[row, rank]),
                    "prototype_available": prototype_matched,
                    "prototype_status": prototype_status,
                    "prototype_index": prototype_index if prototype_matched else None,
                    "prototype_similarity": float(outputs["nearest_prototype_similarity"][row, rank]) if prototype_matched else None,
                    "prototype_sample_id": prototype_meta.get("sample_id") if prototype_matched else None,
                    "prototype_position": prototype_meta.get("position") if prototype_matched else None,
                })
    result: dict[str, Any] = {"predictions": records, "evidence": evidence_rows}
    result["fidelity_metrics"] = {key: float(np.mean(value)) for key, value in all_fidelity.items()}
    result["fidelity_metrics"]["selection_stability_jaccard"] = float(np.mean(stability_jaccard)) if stability_jaccard else None
    result["prototype_metrics"] = {
        "memory_enabled": prototype_enabled,
        "prototype_count": int(prototype_memory.embedding.shape[0]) if prototype_enabled else 0,
        "evidence_total": prototype_evidence_total,
        "evidence_matched": prototype_evidence_matched,
        "evidence_coverage": (
            prototype_evidence_matched / prototype_evidence_total if prototype_evidence_total else None
        ),
    }
    if labels_cls:
        predicted_cls = np.asarray([row["predicted_class"] for row in records])
        predicted_reg = np.asarray([row["predicted_intensity"] for row in records])
        true_cls = np.asarray(labels_cls)
        true_reg = np.asarray(labels_reg)
        metrics = all_metrics(true_cls, predicted_cls, true_reg, predicted_reg)
        metrics["bootstrap_95"] = bootstrap_intervals(
            true_cls, predicted_cls, true_reg, predicted_reg,
            int(cfg["evaluation"]["bootstrap_samples"]), int(cfg["project"]["seed"]),
        )
        fidelity_score = 1.0 - min(1.0, result["fidelity_metrics"]["sufficiency_probability_gap"])
        metrics["selection_score"] = selection_score(metrics, fidelity_score, cfg["evaluation"]["selection_score"])
        result["metrics"] = metrics
    return result


def _read_csv_index(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {tuple(row[key] for key in keys): row for row in csv.DictReader(handle)}


def attach_source_locations(result: dict[str, Any], cfg: dict[str, Any], split: str) -> None:
    root = Path(cfg["paths"]["preprocessed_root"])
    token_index = _read_csv_index(root / cfg["data"]["token_mapping_file"], ("split", "sample_id", "position"))
    time_index = _read_csv_index(root / cfg["data"]["time_mapping_file"], ("sample_id", "position"))
    for row in result["evidence"]:
        sample_id = str(row["sample_id"])
        position = str(row["position"])
        token = token_index.get((split, sample_id, position), {})
        timing = time_index.get((sample_id, position), {}) if split == "attachment4" else {}
        row.update({
            "text_span": token.get("text_span", ""), "char_start": token.get("char_start", ""),
            "char_end": token.get("char_end", ""), "start_sec": timing.get("start_sec", ""),
            "end_sec": timing.get("end_sec", ""), "start_frame": timing.get("start_frame", ""),
            "end_frame": timing.get("end_frame", ""), "representative_frame": timing.get("representative_frame", ""),
            "mapping_method": timing.get("mapping_method", "tokenizer_offset_exact" if row["modality"] == "text" else "unavailable"),
            "mapping_confidence": timing.get("mapping_confidence", token.get("mapping_confidence", "")),
        })
