"""Acceptance checks for the Problem 2 preprocessing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ValidationError(ValueError):
    """Raised when an output artifact violates the mask/data contract."""


def validate_split_arrays(arrays: dict[str, np.ndarray], *, labeled: bool) -> dict[str, Any]:
    required = {
        "sample_id",
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "content_mask",
        "structural_mask",
        "padding_mask",
        "text_observed_mask",
        "text_natural_zero_mask",
        "text_missing_mask",
        "audio",
        "vision",
        "audio_observed_mask",
        "audio_natural_zero_mask",
        "audio_missing_mask",
        "vision_observed_mask",
        "vision_natural_zero_mask",
        "vision_missing_mask",
        "modality_reliability",
    }
    if labeled:
        required |= {"classification_labels", "regression_labels", "raw_text", "privileged_text"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValidationError(f"Missing arrays: {missing}")
    n, steps = arrays["input_ids"].shape
    if arrays["audio"].shape != (n, steps, 74) or arrays["vision"].shape != (n, steps, 35):
        raise ValidationError("Feature shapes do not match the aligned organizer schema")
    if arrays["modality_reliability"].shape != (n, 3, 4):
        raise ValidationError("modality_reliability must have shape (N,3,4)")
    for name in ("audio", "vision", "modality_reliability"):
        if not np.isfinite(arrays[name]).all():
            raise ValidationError(f"{name} contains NaN or Inf")
    attention = arrays["attention_mask"].astype(bool)
    content = arrays["content_mask"].astype(bool)
    structural = arrays["structural_mask"].astype(bool)
    padding = arrays["padding_mask"].astype(bool)
    if np.any(content & structural) or np.any(attention != (content | structural)):
        raise ValidationError("content/structural masks do not partition the attention region")
    if np.any(attention & padding) or np.any(~(attention | padding)):
        raise ValidationError("attention and padding masks do not form a partition")
    text_observed = arrays["text_observed_mask"].astype(bool)
    text_natural = arrays["text_natural_zero_mask"].astype(bool)
    text_missing = arrays["text_missing_mask"].astype(bool)
    if np.any(text_observed & text_natural) or np.any(text_observed & text_missing) or np.any(text_natural & text_missing):
        raise ValidationError("text availability masks overlap")
    if np.any((text_observed | text_natural | text_missing) & ~content):
        raise ValidationError("text availability mask reaches structural/padding slots")
    for modality in ("audio", "vision"):
        observed = arrays[f"{modality}_observed_mask"].astype(bool)
        natural = arrays[f"{modality}_natural_zero_mask"].astype(bool)
        detected = arrays[f"{modality}_missing_mask"].astype(bool)
        if np.any(observed & natural) or np.any(observed & detected) or np.any(natural & detected):
            raise ValidationError(f"{modality} availability masks overlap")
        if np.any((observed | natural | detected) & ~content):
            raise ValidationError(f"{modality} availability mask reaches structural/padding slots")
        if np.any(arrays[modality][~observed] != 0):
            raise ValidationError(f"{modality} unavailable slots must stay exactly zero after scaling")
    if labeled:
        labels = arrays["classification_labels"]
        regression = arrays["regression_labels"]
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValidationError("Classification labels must be 0/1/2")
        if not np.isfinite(regression).all() or np.any((regression < -3) | (regression > 3)):
            raise ValidationError("Regression labels must be finite and within [-3,3]")
        if arrays["privileged_text"].shape != (n, 768) or not np.isfinite(arrays["privileged_text"]).all():
            raise ValidationError("privileged_text must be finite with shape (N,768)")
    return {
        "sample_count": int(n),
        "steps": int(steps),
        "text_observed_steps": int(arrays["text_observed_mask"].sum()),
        "audio_observed_steps": int(arrays["audio_observed_mask"].sum()),
        "vision_observed_steps": int(arrays["vision_observed_mask"].sum()),
        "audio_missing_steps": int(arrays["audio_missing_mask"].sum()),
        "vision_missing_steps": int(arrays["vision_missing_mask"].sum()),
    }


def validate_mask_bank(mask_bank: dict[str, np.ndarray], base: dict[str, np.ndarray]) -> dict[str, Any]:
    masks = mask_bank["synthetic_missing_mask"].astype(bool)
    if masks.ndim != 4 or masks.shape[2] != 3:
        raise ValidationError("synthetic_missing_mask must have shape (N,K,3,T)")
    if np.any(masks[:, :, 0]):
        raise ValidationError("Primary Attachment 3 regime must not mask text")
    content = base["content_mask"].astype(bool)[:, None, None, :]
    if np.any(masks & ~content):
        raise ValidationError("Synthetic masks may not touch structural/padding positions")
    audio_observed = base["audio_observed_mask"].astype(bool)[:, None, :]
    vision_observed = base["vision_observed_mask"].astype(bool)[:, None, :]
    if np.any(masks[:, :, 1] & ~audio_observed) or np.any(masks[:, :, 2] & ~vision_observed):
        raise ValidationError("Synthetic masks may only hide truly observed source steps")
    return {
        "sample_count": int(masks.shape[0]),
        "bank_size": int(masks.shape[1]),
        "audio_masked_steps": int(masks[:, :, 1].sum()),
        "vision_masked_steps": int(masks[:, :, 2].sum()),
    }


def validate_saved_root(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        with np.load(root / f"{split}.npz", allow_pickle=False) as artifact:
            result[split] = validate_split_arrays(dict(artifact), labeled=True)
    with np.load(root / "challenge_aligned.npz", allow_pickle=False) as artifact:
        result["challenge"] = validate_split_arrays(dict(artifact), labeled=False)
    with np.load(root / "train.npz", allow_pickle=False) as train, np.load(
        root / "train_mask_bank.npz", allow_pickle=False
    ) as bank:
        result["mask_bank"] = validate_mask_bank(dict(bank), dict(train))
    return result
