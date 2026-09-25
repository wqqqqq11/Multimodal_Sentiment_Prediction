from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch
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


def _top_mask(scores: np.ndarray, candidate: np.ndarray, ratio: float) -> np.ndarray:
    indexes = np.flatnonzero(candidate)
    result = np.zeros_like(candidate, dtype=bool)
    if not len(indexes):
        return result
    count = max(1, min(len(indexes), int(math.ceil(len(indexes) * float(ratio)))))
    selected = indexes[np.argsort(scores[indexes])[-count:]]
    result[selected] = True
    return result
