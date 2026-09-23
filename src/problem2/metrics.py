from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(num_classes):
        tp = float(matrix[index, index])
        precision = tp / max(float(matrix[:, index].sum()), 1.0)
        recall = tp / max(float(matrix[index, :].sum()), 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": matrix.tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size else 0.0
    if y_true.size < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mae": mae, "pearson": pearson}


def all_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> dict[str, object]:
    result = classification_metrics(y_true_cls, y_pred_cls)
    result.update(regression_metrics(y_true_reg, y_pred_reg))
    result["selection_score"] = selection_score(result)
    return result


def selection_score(metrics: dict[str, object]) -> float:
    return float(metrics["macro_f1"]) + float(metrics["pearson"]) - 0.25 * float(metrics["mae"])
