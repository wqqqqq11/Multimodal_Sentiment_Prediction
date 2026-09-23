from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import apply_fixed_scenario
from .metrics import all_metrics
from .utils import move_to_device


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    scenario: tuple[str, float, str] | None = None,
) -> dict[str, Any]:
    model.eval()
    sample_ids: list[str] = []
    raw_texts: list[str] = []
    logits: list[np.ndarray] = []
    regressions: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    labels_cls: list[np.ndarray] = []
    labels_reg: list[np.ndarray] = []
    for source_batch in loader:
        sample_ids.extend(list(source_batch["sample_id"]))
        if "raw_text" in source_batch:
            raw_texts.extend(list(source_batch["raw_text"]))
        batch = move_to_device(source_batch, device)
        if scenario is not None:
            batch = apply_fixed_scenario(batch, *scenario)
        outputs = model(batch)
        logits.append(outputs["logits"].float().cpu().numpy())
        regressions.append(outputs["regression"].float().cpu().numpy())
        gates.append(outputs["gates"].float().cpu().numpy())
        reliabilities.append(outputs["reliability"].float().cpu().numpy())
        if "classification_labels" in batch:
            labels_cls.append(batch["classification_labels"].cpu().numpy())
            labels_reg.append(batch["regression_labels"].cpu().numpy())
    logit_values = np.concatenate(logits)
    probabilities = _softmax(logit_values)
    result: dict[str, Any] = {
        "sample_id": np.asarray(sample_ids),
        "raw_text": np.asarray(raw_texts) if raw_texts else None,
        "logits": logit_values,
        "probabilities": probabilities,
        "prediction_class": probabilities.argmax(axis=1),
        "prediction_regression": np.concatenate(regressions),
        "gates": np.concatenate(gates),
        "reliability": np.concatenate(reliabilities),
    }
    if labels_cls:
        result["label_class"] = np.concatenate(labels_cls)
        result["label_regression"] = np.concatenate(labels_reg)
        result["metrics"] = all_metrics(
            result["label_class"], result["prediction_class"],
            result["label_regression"], result["prediction_regression"],
        )
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def prediction_frame(result: dict[str, Any], include_labels: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame({
        "sample_id": result["sample_id"],
        "predicted_polarity": result["prediction_class"].astype(int),
        "predicted_intensity": result["prediction_regression"],
        "prob_negative": result["probabilities"][:, 0],
        "prob_neutral": result["probabilities"][:, 1],
        "prob_positive": result["probabilities"][:, 2],
        "gate_text": result["gates"][:, 0],
        "gate_audio": result["gates"][:, 1],
        "gate_vision": result["gates"][:, 2],
        "observed_text_ratio": result["reliability"][:, 0, 0],
        "observed_audio_ratio": result["reliability"][:, 1, 0],
        "observed_vision_ratio": result["reliability"][:, 2, 0],
    })
    if include_labels and "label_class" in result:
        frame["true_polarity"] = result["label_class"].astype(int)
        frame["true_intensity"] = result["label_regression"]
        frame["classification_correct"] = frame["predicted_polarity"] == frame["true_polarity"]
        frame["absolute_error"] = np.abs(frame["predicted_intensity"] - frame["true_intensity"])
    return frame


def evaluate_robustness(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    patterns: Iterable[str],
    rates: Iterable[float],
    positions: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = predict(model, loader, device)
    rows.append({"pattern": "none", "position": "none", "missing_rate": 0.0, **_flat_metrics(complete["metrics"])})
    for pattern in patterns:
        for position in positions:
            for rate in rates:
                result = predict(model, loader, device, (pattern, float(rate), position))
                rows.append({
                    "pattern": pattern,
                    "position": position,
                    "missing_rate": float(rate),
                    **_flat_metrics(result["metrics"]),
                })
    frame = pd.DataFrame(rows)
    complete_row = frame.iloc[0]
    frame["accuracy_drop"] = float(complete_row["accuracy"]) - frame["accuracy"]
    frame["macro_f1_drop"] = float(complete_row["macro_f1"]) - frame["macro_f1"]
    frame["mae_increase"] = frame["mae"] - float(complete_row["mae"])
    frame["pearson_drop"] = float(complete_row["pearson"]) - frame["pearson"]
    return frame


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")}


def missingness_attribution(robustness: pd.DataFrame) -> dict[str, Any]:
    incomplete = robustness[robustness["pattern"] != "none"].copy()
    worst_f1 = incomplete.loc[incomplete["macro_f1"].idxmin()]
    worst_mae = incomplete.loc[incomplete["mae"].idxmax()]
    by_pattern = incomplete.groupby("pattern")[["macro_f1_drop", "mae_increase", "pearson_drop"]].mean().reset_index()
    by_position = incomplete.groupby("position")[["macro_f1_drop", "mae_increase", "pearson_drop"]].mean().reset_index()
    return {
        "worst_macro_f1_scenario": worst_f1.to_dict(),
        "worst_mae_scenario": worst_mae.to_dict(),
        "average_by_pattern": by_pattern.to_dict(orient="records"),
        "average_by_position": by_position.to_dict(orient="records"),
    }


def evaluate_ablations(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    scenarios: list[tuple[str, tuple[str, float, str] | None]] = [
        ("complete", None),
        ("audio_20_middle", ("audio", 0.20, "middle")),
        ("vision_20_middle", ("vision", 0.20, "middle")),
        ("audio_vision_20_middle", ("audio_vision", 0.20, "middle")),
        ("audio_vision_40_middle", ("audio_vision", 0.40, "middle")),
        ("all_modalities_30_middle", ("all_modalities", 0.30, "middle")),
    ]
    variants = [
        ("teacher_without_missing_training", teacher, "none"),
        ("student_full", student, "none"),
        ("student_uniform_gate", student, "uniform_gate"),
        ("student_text_only", student, "text_only"),
    ]
    rows: list[dict[str, Any]] = []
    for variant, model, mode in variants:
        previous = getattr(model, "ablation_mode", "none")
        model.ablation_mode = mode
        try:
            for scenario_name, scenario in scenarios:
                result = predict(model, loader, device, scenario)
                rows.append({"variant": variant, "scenario": scenario_name, **_flat_metrics(result["metrics"])})
        finally:
            model.ablation_mode = previous
    return pd.DataFrame(rows)
