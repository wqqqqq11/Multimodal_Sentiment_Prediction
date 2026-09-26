"""Leakage-safe robust scaling for acoustic and visual feature streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModalityScaler:
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalityScaler":
        return cls(**{name: np.asarray(payload[name], dtype=np.float64) for name in ("lower", "upper", "center", "scale")})


class RobustFeatureScaler:
    """Winsorize and IQR-scale using observed training time steps only."""

    def __init__(
        self,
        *,
        lower_quantile: float,
        upper_quantile: float,
        minimum_scale: float,
        normalized_clip: float,
    ) -> None:
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.minimum_scale = float(minimum_scale)
        self.normalized_clip = float(normalized_clip)
        self.modalities: dict[str, ModalityScaler] = {}

    def fit(self, name: str, features: np.ndarray, observed_mask: np.ndarray) -> ModalityScaler:
        array = np.asarray(features)
        mask = np.asarray(observed_mask, dtype=bool)
        if array.ndim != 3 or mask.shape != array.shape[:2]:
            raise ValueError(f"{name}: expected features (N,T,D) and mask (N,T)")
        values = np.asarray(array[mask], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"{name}: no observed training values available to fit scaler")
        values[~np.isfinite(values)] = np.nan
        lower = np.nanquantile(values, self.lower_quantile, axis=0)
        upper = np.nanquantile(values, self.upper_quantile, axis=0)
        center = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        fallback = np.nanstd(values, axis=0)
        for vector, default in ((lower, 0.0), (upper, 0.0), (center, 0.0), (q25, 0.0), (q75, 0.0), (fallback, 1.0)):
            vector[~np.isfinite(vector)] = default
        scale = q75 - q25
        use_fallback = scale < self.minimum_scale
        scale[use_fallback] = fallback[use_fallback]
        scale[scale < self.minimum_scale] = 1.0
        stats = ModalityScaler(lower=lower, upper=upper, center=center, scale=scale)
        self.modalities[name] = stats
        return stats

    def transform(
        self,
        name: str,
        features: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        if name not in self.modalities:
            raise ValueError(f"Scaler has not been fit for {name}")
        stats = self.modalities[name]
        array = np.asarray(features)
        mask = np.asarray(observed_mask, dtype=bool)
        if array.ndim != 3 or mask.shape != array.shape[:2]:
            raise ValueError(f"{name}: expected features (N,T,D) and mask (N,T)")
        output = np.zeros(array.shape, dtype=np.float32)
        values = np.asarray(array[mask], dtype=np.float64).copy()
        if values.size == 0:
            return output, {"observed_steps": 0, "nonfinite_elements": 0, "winsorized_elements": 0, "winsorized_ratio": 0.0}
        invalid = ~np.isfinite(values)
        invalid_count = int(invalid.sum())
        if invalid_count:
            rows, cols = np.nonzero(invalid)
            values[rows, cols] = stats.center[cols]
        outside = (values < stats.lower) | (values > stats.upper)
        winsorized_count = int(outside.sum())
        values = np.clip(values, stats.lower, stats.upper)
        values = (values - stats.center) / stats.scale
        values = np.clip(values, -self.normalized_clip, self.normalized_clip)
        output[mask] = values.astype(np.float32)
        return output, {
            "observed_steps": int(mask.sum()),
            "nonfinite_elements": invalid_count,
            "winsorized_elements": winsorized_count,
            "winsorized_ratio": float(winsorized_count / values.size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "training-observed-only quantile winsorization plus IQR scaling",
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "minimum_scale": self.minimum_scale,
            "normalized_clip": self.normalized_clip,
            "modalities": {name: value.to_dict() for name, value in self.modalities.items()},
        }


def scaler_from_dict(payload: dict[str, Any]) -> RobustFeatureScaler:
    scaler = RobustFeatureScaler(
        lower_quantile=float(payload["lower_quantile"]),
        upper_quantile=float(payload["upper_quantile"]),
        minimum_scale=float(payload["minimum_scale"]),
        normalized_clip=float(payload["normalized_clip"]),
    )
    scaler.modalities = {
        name: ModalityScaler.from_dict(value) for name, value in payload["modalities"].items()
    }
    return scaler
