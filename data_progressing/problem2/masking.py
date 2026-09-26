"""Mask semantics and empirical continuous-span missingness augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODALITIES = ("text", "audio", "vision")
PATTERN_TO_ID = {"audio": 1, "vision": 2, "audio_vision": 3}
POSITION_TO_ID = {"begin": 0, "middle": 1, "end": 2}


def timestep_all_zero(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return whether every feature at a time step is effectively zero."""
    array = np.asarray(features)
    if array.ndim < 2:
        raise ValueError("features must end in (time, feature)")
    return np.all(np.isfinite(array) & (np.abs(array) <= eps), axis=-1)


def masks_from_attention(attention_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Separate padding from the valid BERT region and its CLS/SEP slots."""
    attention = np.asarray(attention_mask) > 0
    if attention.ndim != 2:
        raise ValueError("attention_mask must have shape (N, T)")
    valid = attention.copy()
    structural = np.zeros_like(valid)
    content = valid.copy()
    for index, row in enumerate(valid):
        positions = np.flatnonzero(row)
        if positions.size == 0:
            continue
        structural[index, positions[0]] = True
        content[index, positions[0]] = False
        if positions.size > 1:
            structural[index, positions[-1]] = True
            content[index, positions[-1]] = False
    return {
        "valid": valid,
        "content": content,
        "structural": structural,
        "padding": ~valid,
    }


def modality_availability(
    features: np.ndarray,
    content_mask: np.ndarray,
    *,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Identify observed and naturally unavailable steps in organizer source data."""
    zero = timestep_all_zero(features, eps=eps)
    content = np.asarray(content_mask, dtype=bool)
    if zero.shape != content.shape:
        raise ValueError("feature time axes and content_mask do not agree")
    natural_zero = content & zero
    observed = content & ~zero
    return {"observed": observed, "natural_zero": natural_zero, "zero": zero}


def mask_to_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indexes.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[indexes[0], indexes[breaks + 1]]
    ends = np.r_[indexes[breaks], indexes[-1]] + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


@dataclass(frozen=True)
class MissingnessConfig:
    bank_size: int
    pattern_probabilities: dict[str, float]
    position_probabilities: dict[str, float]
    ratio_min: float
    ratio_max: float
    beta_alpha: float
    beta_beta: float
    min_remaining: int


class ContinuousSpanMaskGenerator:
    """Generate deterministic teacher/student masks from the Attachment 3 regime."""

    def __init__(self, config: MissingnessConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)

    def generate(
        self,
        audio_observed: np.ndarray,
        vision_observed: np.ndarray,
    ) -> dict[str, np.ndarray]:
        audio = np.asarray(audio_observed, dtype=bool)
        vision = np.asarray(vision_observed, dtype=bool)
        if audio.shape != vision.shape or audio.ndim != 2:
            raise ValueError("audio and vision observed masks must share shape (N, T)")
        sample_count, steps = audio.shape
        k = self.config.bank_size
        bank = np.zeros((sample_count, k, len(MODALITIES), steps), dtype=bool)
        pattern_id = np.zeros((sample_count, k), dtype=np.uint8)
        position_id = np.zeros((sample_count, k, len(MODALITIES)), dtype=np.int8)
        relative_start = np.full((sample_count, k, len(MODALITIES)), -1.0, dtype=np.float32)
        actual_ratio = np.zeros((sample_count, k, len(MODALITIES)), dtype=np.float32)
        pattern_names = list(self.config.pattern_probabilities)
        pattern_p = np.asarray([self.config.pattern_probabilities[name] for name in pattern_names], dtype=float)
        position_names = list(self.config.position_probabilities)
        position_p = np.asarray([self.config.position_probabilities[name] for name in position_names], dtype=float)

        for sample_index in range(sample_count):
            for bank_index in range(k):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, sample_index, bank_index]))
                pattern = str(rng.choice(pattern_names, p=pattern_p))
                pattern_id[sample_index, bank_index] = PATTERN_TO_ID[pattern]
                targets = ("audio", "vision") if pattern == "audio_vision" else (pattern,)
                for modality in targets:
                    modality_index = MODALITIES.index(modality)
                    observed = audio[sample_index] if modality == "audio" else vision[sample_index]
                    position = str(rng.choice(position_names, p=position_p))
                    ratio = self.config.ratio_min + (
                        self.config.ratio_max - self.config.ratio_min
                    ) * float(rng.beta(self.config.beta_alpha, self.config.beta_beta))
                    generated, start = self._one_span(observed, ratio, position, rng)
                    bank[sample_index, bank_index, modality_index] = generated
                    position_id[sample_index, bank_index, modality_index] = POSITION_TO_ID[position]
                    observed_count = int(observed.sum())
                    actual_ratio[sample_index, bank_index, modality_index] = (
                        float(generated.sum() / observed_count) if observed_count else 0.0
                    )
                    relative_start[sample_index, bank_index, modality_index] = start
        return {
            "synthetic_missing_mask": bank,
            "pattern_id": pattern_id,
            "position_id": position_id,
            "relative_start": relative_start,
            "actual_missing_ratio": actual_ratio,
        }

    def _one_span(
        self,
        observed: np.ndarray,
        ratio: float,
        position: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        result = np.zeros_like(observed, dtype=bool)
        indexes = np.flatnonzero(observed)
        available = int(indexes.size)
        max_masked = max(0, available - self.config.min_remaining)
        target = min(max_masked, max(1, int(round(ratio * available))))
        if target <= 0:
            return result, -1.0
        lowest = int(indexes[0])
        highest = int(indexes[-1])
        span_width = max(1, target)
        last_start = max(lowest, highest - span_width + 1)
        third = max(1, (last_start - lowest + 1) // 3)
        if position == "begin":
            lo, hi = lowest, min(last_start, lowest + third - 1)
        elif position == "middle":
            center = (lowest + last_start) // 2
            lo, hi = max(lowest, center - third // 2), min(last_start, center + third // 2)
        else:
            lo, hi = max(lowest, last_start - third + 1), last_start
        start = int(rng.integers(lo, hi + 1)) if hi >= lo else lowest
        end = min(observed.size, start + span_width)
        result[start:end] = observed[start:end]
        if int(result.sum()) < target:
            ordered = indexes[np.argsort(np.abs(indexes - (start + span_width / 2)))]
            result[ordered[:target]] = True
        if int(result.sum()) > target:
            chosen = np.flatnonzero(result)
            result[chosen[target:]] = False
        relative = float(start / max(observed.size - 1, 1))
        return result, relative


def missingness_config(raw: dict) -> MissingnessConfig:
    return MissingnessConfig(
        bank_size=int(raw["mask_bank_size"]),
        pattern_probabilities={str(k): float(v) for k, v in raw["pattern_probabilities"].items()},
        position_probabilities={str(k): float(v) for k, v in raw["position_probabilities"].items()},
        ratio_min=float(raw["ratio_min"]),
        ratio_max=float(raw["ratio_max"]),
        beta_alpha=float(raw["ratio_beta_alpha"]),
        beta_beta=float(raw["ratio_beta_beta"]),
        min_remaining=int(raw["min_observed_steps_after_mask"]),
    )
