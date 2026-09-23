"""Timestamp validation and normalization."""

from __future__ import annotations
import numpy as np


def centers(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != end.shape or np.any(end < start) or np.any(np.diff(start) < 0):
        raise ValueError("时间区间必须等长、非负长度且按开始时间单调")
    return (start + end) / 2.0


def normalize_time(values: np.ndarray, duration: float) -> np.ndarray:
    if duration <= 0:
        raise ValueError("duration必须>0")
    return np.clip(np.asarray(values, dtype=float) / duration, 0.0, 1.0)
