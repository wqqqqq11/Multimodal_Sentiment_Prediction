from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 3) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true, predicted in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= true < classes and 0 <= predicted < classes:
            matrix[true, predicted] += 1
    return matrix


def all_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> dict[str, object]:
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
        f1.append(2.0 * p * r / max(p + r, 1e-12))
    true_reg = y_true_reg.astype(np.float64)
    pred_reg = y_pred_reg.astype(np.float64)
    mae = float(np.mean(np.abs(true_reg - pred_reg)))
    pearson = 0.0 if len(true_reg) < 2 or np.std(true_reg) < 1e-12 or np.std(pred_reg) < 1e-12 else float(np.corrcoef(true_reg, pred_reg)[0, 1])
    return {
        "accuracy": accuracy, "macro_f1": float(np.mean(f1)), "mae": mae, "pearson": pearson,
        "per_class_precision": precision, "per_class_recall": recall, "per_class_f1": f1,
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_intervals(
    y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray,
    repeats: int, seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("accuracy", "macro_f1", "mae", "pearson")}
    for _ in range(repeats):
        indices = rng.integers(0, len(y_true_cls), size=len(y_true_cls))
        metrics = all_metrics(y_true_cls[indices], y_pred_cls[indices], y_true_reg[indices], y_pred_reg[indices])
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        key: {"lower_95": float(np.quantile(item, 0.025)), "upper_95": float(np.quantile(item, 0.975))}
        for key, item in values.items()
    }


def selection_score(metrics: dict[str, object], fidelity: float, weights: dict[str, float]) -> float:
    return (
        float(weights["macro_f1"]) * float(metrics["macro_f1"])
        + float(weights["pearson"]) * float(metrics["pearson"])
        + float(weights["mae"]) * float(metrics["mae"])
        + float(weights["fidelity"]) * float(fidelity)
    )
