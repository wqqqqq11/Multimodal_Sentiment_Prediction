from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 3) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= true < classes and 0 <= pred < classes:
            matrix[true, pred] += 1
    return matrix


def all_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    goals: dict[str, float] | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true_cls, y_pred_cls)
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index in range(3):
        tp = float(matrix[index, index])
        p = tp / max(float(matrix[:, index].sum()), 1.0)
        r = tp / max(float(matrix[index].sum()), 1.0)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))
    truth = np.asarray(y_true_reg, dtype=np.float64)
    prediction = np.asarray(y_pred_reg, dtype=np.float64)
    mae = float(np.mean(np.abs(truth - prediction))) if truth.size else 0.0
    pearson = 0.0
    if truth.size > 1 and np.std(truth) > 1e-12 and np.std(prediction) > 1e-12:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
    result: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)),
        "mae": mae,
        "pearson": pearson,
        "neutral_recall": recall[1],
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": matrix.tolist(),
    }
    result["selection_score"] = target_score(result, goals)
    if goals:
        result["goal_audit"] = goal_audit(result, goals)
    return result


def target_score(metrics: dict[str, Any], goals: dict[str, float] | None = None) -> float:
    base = (
        0.45 * float(metrics["macro_f1"])
        + 0.25 * float(metrics["pearson"])
        + 0.20 * float(metrics["neutral_recall"])
        + 0.10 * float(metrics["accuracy"])
        - 0.30 * float(metrics["mae"])
    )
    if not goals:
        return base
    audit = goal_audit(metrics, goals)
    ratios = [
        float(metrics["accuracy"]) / float(goals["accuracy"]),
        float(metrics["macro_f1"]) / float(goals["macro_f1"]),
        float(goals["mae"]) / max(float(metrics["mae"]), 1e-8),
        float(metrics["pearson"]) / float(goals["pearson"]),
        float(metrics["neutral_recall"]) / float(goals["neutral_recall"]),
    ]
    return base + 0.08 * audit["met_count"] + 0.05 * float(np.mean(np.clip(ratios, 0.0, 1.25)))


def goal_audit(metrics: dict[str, Any], goals: dict[str, float]) -> dict[str, Any]:
    status = {
        "accuracy": float(metrics["accuracy"]) >= float(goals["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]) >= float(goals["macro_f1"]),
        "mae": float(metrics["mae"]) <= float(goals["mae"]),
        "pearson": float(metrics["pearson"]) >= float(goals["pearson"]),
        "neutral_recall": float(metrics["neutral_recall"]) >= float(goals["neutral_recall"]),
    }
    return {"targets": goals, "met": status, "met_count": int(sum(status.values())), "all_met": all(status.values())}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def apply_calibration(logits: np.ndarray, regression: np.ndarray,
                            calibration: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_reg = np.asarray(regression, dtype=np.float64)
    if calibration is None:
        adjusted, calibrated_reg = np.asarray(logits, dtype=np.float64), np.clip(raw_reg, -3.0, 3.0)
    else:
        calibrated_reg = np.clip(float(calibration["regression_slope"]) * raw_reg +
                                 float(calibration["regression_intercept"]), -3.0, 3.0)
        adjusted = np.asarray(logits, dtype=np.float64) / float(calibration["temperature"])
        adjusted = adjusted.copy()
        adjusted[:, 1] += float(calibration["neutral_bias"])
        scale = float(calibration["neutral_scale"])
        adjusted[:, 1] += float(calibration["neutral_strength"]) * np.exp(-np.abs(calibrated_reg) / scale)
        polarity = float(calibration["polarity_strength"]) * calibrated_reg / 3.0
        adjusted[:, 0] -= polarity
        adjusted[:, 2] += polarity
    probabilities = softmax(adjusted)
    return probabilities, probabilities.argmax(axis=1), calibrated_reg


def fit_calibration(logits: np.ndarray, regression: np.ndarray, labels_cls: np.ndarray,
                          labels_reg: np.ndarray, cfg: dict[str, Any], goals: dict[str, float]) -> dict[str, Any]:
    raw_reg = np.asarray(regression, dtype=np.float64)
    regression_candidates = [(1.0, 0.0)]
    if np.std(raw_reg) > 1e-8:
        slope, intercept = np.polyfit(raw_reg, labels_reg.astype(np.float64), 1)
        regression_candidates.append((float(np.clip(slope, 0.4, 1.8)), float(np.clip(intercept, -0.6, 0.6))))
    neutral_biases = np.linspace(float(cfg["neutral_bias_min"]), float(cfg["neutral_bias_max"]),
                                 int(cfg["neutral_bias_steps"]))
    best: dict[str, Any] | None = None
    for slope, intercept in regression_candidates:
        for temperature in cfg["temperature_values"]:
            for neutral_bias in neutral_biases:
                for neutral_strength in cfg["neutral_strength_values"]:
                    for polarity_strength in cfg["polarity_strength_values"]:
                        candidate = {"temperature": float(temperature), "neutral_bias": float(neutral_bias),
                                     "neutral_strength": float(neutral_strength),
                                     "polarity_strength": float(polarity_strength),
                                     "neutral_scale": float(cfg["neutral_scale"]),
                                     "regression_slope": slope, "regression_intercept": intercept}
                        _, predicted_cls, predicted_reg = apply_calibration(logits, raw_reg, candidate)
                        metrics = all_metrics(labels_cls, predicted_cls, labels_reg, predicted_reg, goals)
                        candidate["metrics"] = metrics
                        if best is None or metrics["selection_score"] > best["metrics"]["selection_score"]:
                            best = candidate
    assert best is not None
    return best
