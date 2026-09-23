"""Sample-level reliability descriptors and label checks."""

from __future__ import annotations

import numpy as np


RELIABILITY_COLUMNS = (
    "valid_ratio",
    "observed_ratio",
    "natural_zero_ratio",
    "detected_missing_ratio",
)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )


def reliability_tensor(
    content_mask: np.ndarray,
    audio_observed: np.ndarray,
    audio_natural: np.ndarray,
    vision_observed: np.ndarray,
    vision_natural: np.ndarray,
    *,
    audio_missing: np.ndarray | None = None,
    vision_missing: np.ndarray | None = None,
) -> np.ndarray:
    """Build (N, modality, reliability_feature) inputs for the gating network."""
    content = np.asarray(content_mask, dtype=bool)
    n, steps = content.shape
    denominator = content.sum(axis=1).astype(np.float32)
    valid_ratio = denominator / max(steps, 1)
    audio_missing = np.zeros_like(content) if audio_missing is None else np.asarray(audio_missing, dtype=bool)
    vision_missing = np.zeros_like(content) if vision_missing is None else np.asarray(vision_missing, dtype=bool)

    result = np.zeros((n, 3, len(RELIABILITY_COLUMNS)), dtype=np.float32)
    result[:, :, 0] = valid_ratio[:, None]
    result[:, 0, 1] = np.where(denominator > 0, 1.0, 0.0)
    for index, (observed, natural, missing) in enumerate(
        (
            (content, np.zeros_like(content), np.zeros_like(content)),
            (audio_observed, audio_natural, audio_missing),
            (vision_observed, vision_natural, vision_missing),
        )
    ):
        result[:, index, 1] = _safe_ratio(np.asarray(observed).sum(axis=1), denominator)
        result[:, index, 2] = _safe_ratio(np.asarray(natural).sum(axis=1), denominator)
        result[:, index, 3] = _safe_ratio(np.asarray(missing).sum(axis=1), denominator)
    return result


def expected_class_from_regression(regression: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(regression, dtype=float)
    result = np.ones(values.shape, dtype=np.int64)
    result[values < -eps] = 0
    result[values > eps] = 2
    return result


def balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=3).astype(np.float64)
    weights = np.divide(values.size, 3.0 * counts, out=np.zeros(3, dtype=np.float64), where=counts > 0)
    return weights.astype(np.float32)
