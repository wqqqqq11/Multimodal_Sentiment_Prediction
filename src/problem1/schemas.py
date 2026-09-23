"""Data contracts shared by extraction, alignment and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeatureSequence:
    sample_id: str
    modality: str
    features: np.ndarray
    positions: np.ndarray
    start: np.ndarray
    end: np.ndarray
    quality: np.ndarray
    valid_mask: np.ndarray
    source_index: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[0] == 0:
            raise ValueError(f"{self.sample_id}/{self.modality}: features必须为非空二维数组")
        length = self.features.shape[0]
        for name in ("positions", "start", "end", "quality", "valid_mask", "source_index"):
            value = np.asarray(getattr(self, name))
            if value.shape != (length,):
                raise ValueError(f"{self.sample_id}/{self.modality}: {name}长度错误")
        if not np.isfinite(self.features).all():
            raise ValueError(f"{self.sample_id}/{self.modality}: 特征含NaN/Inf")
        if not np.isfinite(self.positions).all() or np.any(np.diff(self.positions) < 0):
            raise ValueError(f"{self.sample_id}/{self.modality}: 位置必须有限且单调")
        if np.any(self.quality < 0) or np.any(self.quality > 1):
            raise ValueError(f"{self.sample_id}/{self.modality}: quality越界")


@dataclass
class AlignmentResult:
    sample_id: str
    consensus: np.ndarray
    consensus_time: np.ndarray
    aligned: dict[str, np.ndarray]
    valid_mask: np.ndarray
    uncertainty: np.ndarray
    plans: dict[str, np.ndarray]
    mappings: list[dict[str, Any]]
    metrics: dict[str, Any]
    history: list[dict[str, float]]
