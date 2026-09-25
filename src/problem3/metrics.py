from __future__ import annotations

from typing import Any
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 3) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= true < classes and 0 <= pred < classes: matrix[true, pred] += 1
    return matrix


def all_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_true_reg: np.ndarray,
                y_pred_reg: np.ndarray, goals: dict[str, float] | None = None) -> dict[str, Any]:
    matrix = confusion_matrix(y_true_cls, y_pred_cls)
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    f1: list[float] = []
    for index in range(3):
        tp = float(matrix[index, index])
        precision = tp / max(float(matrix[:, index].sum()), 1.0)
        recall = tp / max(float(matrix[index].sum()), 1.0)
        f1.append(2 * precision * recall / max(precision + recall, 1e-12))
    truth, prediction = np.asarray(y_true_reg, dtype=np.float64), np.asarray(y_pred_reg, dtype=np.float64)
    mae = float(np.mean(np.abs(truth - prediction))) if truth.size else 0.0
    pearson = 0.0
    if truth.size > 1 and np.std(truth) > 1e-12 and np.std(prediction) > 1e-12:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
    result: dict[str, Any] = {"accuracy": accuracy, "macro_f1": float(np.mean(f1)), "mae": mae,
                              "pearson": pearson, "per_class_f1": f1, "confusion_matrix": matrix.tolist()}
    result["selection_score"] = target_score(result, goals)
    if goals: result["goal_audit"] = goal_audit(result, goals)
    return result


def target_score(metrics: dict[str, Any], goals: dict[str, float] | None = None) -> float:
    if not goals:
        return (0.32 * float(metrics["accuracy"]) + 0.32 * float(metrics["macro_f1"])
                + 0.18 * float(metrics["pearson"]) - 0.18 * float(metrics["mae"]))
    ratios = np.asarray([
        float(metrics["accuracy"]) / float(goals["accuracy"]),
        float(metrics["macro_f1"]) / float(goals["macro_f1"]),
        float(goals["mae"]) / max(float(metrics["mae"]), 1e-8),
        float(metrics["pearson"]) / float(goals["pearson"]),
    ])
    weights = np.asarray([0.30, 0.30, 0.25, 0.15])
    return float(np.sum(weights * np.clip(ratios, 0.0, 1.15)))


def goal_audit(metrics: dict[str, Any], goals: dict[str, float]) -> dict[str, Any]:
    status = {"accuracy": float(metrics["accuracy"]) >= float(goals["accuracy"]),
              "macro_f1": float(metrics["macro_f1"]) >= float(goals["macro_f1"]),
              "mae": float(metrics["mae"]) <= float(goals["mae"]),
              "pearson": float(metrics["pearson"]) >= float(goals["pearson"])}
    return {"targets": goals, "met": status, "met_count": int(sum(status.values())), "all_met": all(status.values())}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def apply_calibration(logits: np.ndarray, regression: np.ndarray,
                      calibration: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = softmax(np.asarray(logits, dtype=np.float64))
    raw = np.asarray(regression, dtype=np.float64)
    if calibration is None:
        calibrated = np.clip(raw, -3.0, 3.0)
    else:
        calibrated = np.clip(float(calibration["regression_slope"]) * raw
                             + float(calibration["regression_intercept"]), -3.0, 3.0)
    return probabilities, probabilities.argmax(axis=1), calibrated


def fit_calibration(logits: np.ndarray, regression: np.ndarray, labels_cls: np.ndarray,
                    labels_reg: np.ndarray, cfg: dict[str, Any], goals: dict[str, float]) -> dict[str, Any]:
    raw = np.asarray(regression, dtype=np.float64)
    slopes = list(np.linspace(float(cfg["regression_slope_min"]), float(cfg["regression_slope_max"]),
                              int(cfg["regression_slope_steps"])))
    if np.std(raw) > 1e-8:
        fitted_slope, fitted_intercept = np.polyfit(raw, labels_reg.astype(np.float64), 1)
        slopes.append(float(np.clip(fitted_slope, cfg["regression_slope_min"], cfg["regression_slope_max"])))
    else:
        fitted_intercept = 0.0
    intercepts = list(np.linspace(float(cfg["regression_intercept_min"]), float(cfg["regression_intercept_max"]),
                                  int(cfg["regression_intercept_steps"])))
    intercepts.append(float(np.clip(fitted_intercept, cfg["regression_intercept_min"], cfg["regression_intercept_max"])))
    predicted_cls = np.asarray(logits).argmax(axis=1)
    best: dict[str, Any] | None = None
    for slope in sorted(set(slopes)):
        for intercept in sorted(set(intercepts)):
            calibrated = np.clip(slope * raw + intercept, -3.0, 3.0)
            metrics = all_metrics(labels_cls, predicted_cls, labels_reg, calibrated, goals)
            candidate = {"regression_slope": float(slope), "regression_intercept": float(intercept), "metrics": metrics}
            if best is None or (metrics["selection_score"], -metrics["mae"]) > (
                    best["metrics"]["selection_score"], -best["metrics"]["mae"]):
                best = candidate
    assert best is not None
    return best
