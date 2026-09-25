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


def fit_calibration(
    logits: np.ndarray,
    regression: np.ndarray,
    labels_cls: np.ndarray,
    labels_reg: np.ndarray,
    cfg: dict[str, Any],
    goals: dict[str, float],
) -> dict[str, Any]:
    temperatures = np.linspace(float(cfg["temperature_min"]), float(cfg["temperature_max"]), int(cfg["temperature_steps"]))
    neutral_biases = np.linspace(float(cfg["neutral_bias_min"]), float(cfg["neutral_bias_max"]), int(cfg["neutral_bias_steps"]))
    best: dict[str, Any] | None = None
    raw_regression = np.asarray(regression, dtype=np.float64)
    candidates = [(1.0, 0.0)]
    if np.std(raw_regression) > 1e-8:
        slope, intercept = np.polyfit(raw_regression, labels_reg.astype(np.float64), 1)
        candidates.append((float(np.clip(slope, 0.5, 1.5)), float(np.clip(intercept, -0.5, 0.5))))
    for slope, intercept in candidates:
        calibrated_regression = np.clip(slope * raw_regression + intercept, -3.0, 3.0)
        for temperature in temperatures:
            for neutral_bias in neutral_biases:
                adjusted = logits / float(temperature)
                adjusted = adjusted.copy()
                adjusted[:, 1] += float(neutral_bias)
                prediction = adjusted.argmax(axis=1)
                metrics = all_metrics(labels_cls, prediction, labels_reg, calibrated_regression, goals)
                candidate = {
                    "temperature": float(temperature),
                    "class_bias": [0.0, float(neutral_bias), 0.0],
                    "regression_slope": slope,
                    "regression_intercept": intercept,
                    "metrics": metrics,
                }
                if best is None or float(metrics["selection_score"]) > float(best["metrics"]["selection_score"]):
                    best = candidate
    assert best is not None
    return best


def apply_calibration(
    logits: np.ndarray, regression: np.ndarray, calibration: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if calibration is None:
        adjusted = np.asarray(logits)
        calibrated_regression = np.clip(regression, -3.0, 3.0)
    else:
        adjusted = np.asarray(logits) / float(calibration["temperature"])
        adjusted = adjusted + np.asarray(calibration["class_bias"], dtype=np.float64)[None, :]
        calibrated_regression = np.clip(
            float(calibration["regression_slope"]) * np.asarray(regression)
            + float(calibration["regression_intercept"]),
            -3.0,
            3.0,
        )
    probabilities = softmax(adjusted)
    return probabilities, probabilities.argmax(axis=1), calibrated_regression
