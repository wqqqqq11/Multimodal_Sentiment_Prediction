"""Modality-independent temporal signatures and multi-scale smoothing."""

from __future__ import annotations

import numpy as np

from ..schemas import FeatureSequence


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) == 1:
        return values.copy()
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        left, right = max(0, index - radius), min(len(values), index + radius + 1)
        result[index] = (cumulative[right] - cumulative[left]) / (right - left)
    return result


def temporal_signature(sequence: FeatureSequence, radius: int = 0) -> np.ndarray:
    """Map arbitrary-dimensional modality features into 16 comparable statistics."""
    raw = np.asarray(sequence.features, dtype=np.float64)
    center = np.median(raw, axis=0, keepdims=True)
    scale = np.median(np.abs(raw - center), axis=0, keepdims=True) * 1.4826
    normalized = np.clip((raw - center) / np.maximum(scale, 1e-6), -6.0, 6.0)
    smooth = _moving_average(normalized, radius)
    delta = np.vstack((np.zeros((1, smooth.shape[1])), np.diff(smooth, axis=0)))
    quantiles = np.quantile(smooth, [0.25, 0.5, 0.75], axis=1).T
    descriptors = np.column_stack((
        smooth.mean(axis=1), smooth.std(axis=1), np.sqrt(np.mean(smooth * smooth, axis=1)),
        smooth.min(axis=1), smooth.max(axis=1), quantiles,
        np.mean(smooth > 0, axis=1), np.mean(np.abs(smooth), axis=1),
        np.sqrt(np.mean(delta * delta, axis=1)), delta.mean(axis=1),
        np.mean((normalized - smooth) ** 2, axis=1), sequence.quality,
        sequence.positions, np.sin(np.pi * sequence.positions),
    ))
    median = np.median(descriptors, axis=0, keepdims=True)
    mad = np.median(np.abs(descriptors - median), axis=0, keepdims=True) * 1.4826
    signature = np.clip((descriptors - median) / np.maximum(mad, 1e-4), -6.0, 6.0)
    return (signature / np.maximum(np.linalg.norm(signature, axis=1, keepdims=True), 1e-8)).astype(np.float64)


def multiscale_signatures(sequence: FeatureSequence, radii: list[int]) -> list[np.ndarray]:
    return [temporal_signature(sequence, int(radius)) for radius in radii]


def interpolate_signature(sequence: FeatureSequence, signature: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(target_positions, sequence.positions, signature[:, column])
                            for column in range(signature.shape[1])])
