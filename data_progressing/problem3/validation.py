"""Acceptance checks for Problem 3 preprocessing artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class ValidationError(ValueError):
    """Raised when a Problem 3 artifact violates its data contract."""


def validate_split_arrays(arrays: dict[str, np.ndarray], *, labeled: bool) -> dict[str, Any]:
    required = {
        "sample_id", "raw_text", "input_ids", "attention_mask", "token_type_ids",
        "content_mask", "structural_mask", "padding_mask", "audio", "vision",
        "text_observed_mask", "audio_observed_mask", "vision_observed_mask",
        "text_natural_zero_mask", "audio_natural_zero_mask", "vision_natural_zero_mask",
        "text_failure_mask", "audio_failure_mask", "vision_failure_mask",
        "text_evidence_candidate_mask", "audio_evidence_candidate_mask",
        "vision_evidence_candidate_mask", "truncation_flag", "tokenizer_exact_match",
        "covered_char_end", "token_mapping_confidence",
    }
    if labeled:
        required |= {"classification_labels", "regression_labels", "intensity_bin"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValidationError(f"Missing arrays: {missing}")
    forbidden = {"privileged_text", "modality_reliability"} & arrays.keys()
    if forbidden:
        raise ValidationError(f"Problem 3 main inputs contain forbidden bypass fields: {sorted(forbidden)}")
    n, steps = arrays["input_ids"].shape
    if arrays["audio"].shape != (n, steps, 74) or arrays["vision"].shape != (n, steps, 35):
        raise ValidationError("Feature shapes do not match aligned_50")
    if not np.isfinite(arrays["audio"]).all() or not np.isfinite(arrays["vision"]).all():
        raise ValidationError("Continuous features contain NaN or Inf")
    attention = arrays["attention_mask"].astype(bool)
    content = arrays["content_mask"].astype(bool)
    structural = arrays["structural_mask"].astype(bool)
    padding = arrays["padding_mask"].astype(bool)
    if np.any(content & structural) or np.any(attention != (content | structural)):
        raise ValidationError("content/structural do not partition attention")
    if np.any(attention & padding) or np.any(~(attention | padding)):
        raise ValidationError("attention/padding do not form a partition")
    for modality in ("text", "audio", "vision"):
        observed = arrays[f"{modality}_observed_mask"].astype(bool)
        natural = arrays[f"{modality}_natural_zero_mask"].astype(bool)
        failure = arrays[f"{modality}_failure_mask"].astype(bool)
        candidate = arrays[f"{modality}_evidence_candidate_mask"].astype(bool)
        if np.any((observed | natural | failure | candidate) & ~content):
            raise ValidationError(f"{modality} mask reaches structural/padding positions")
        if np.any(candidate & ~observed) or np.any(candidate & failure):
            raise ValidationError(f"{modality} candidate mask is not a subset of healthy observations")
        if modality in {"audio", "vision"} and np.any(arrays[modality][~observed] != 0):
            raise ValidationError(f"{modality} unavailable positions are not exact zero")
    if labeled:
        labels = arrays["classification_labels"]
        regression = arrays["regression_labels"]
        bins = arrays["intensity_bin"]
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValidationError("Classification labels must be 0/1/2")
        if not np.isfinite(regression).all() or np.any((regression < -3) | (regression > 3)):
            raise ValidationError("Regression labels must lie in [-3,3]")
        if not np.isin(bins, np.arange(7)).all():
            raise ValidationError("Intensity bins must be 0..6")
    return {
        "sample_count": int(n),
        "steps": int(steps),
        "content_steps": int(content.sum()),
        "audio_candidates": int(arrays["audio_evidence_candidate_mask"].sum()),
        "vision_candidates": int(arrays["vision_evidence_candidate_mask"].sum()),
        "truncation_flags": int(arrays["truncation_flag"].sum()),
        "tokenizer_mismatches": int((~arrays["tokenizer_exact_match"].astype(bool)).sum()),
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_saved_root(root: Path, expected_counts: dict[str, int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_counts = expected_counts or {"train": 3395, "valid": 728, "test": 727}
    for split in ("train", "valid", "test"):
        with np.load(root / f"{split}.npz", allow_pickle=False) as artifact:
            audit = validate_split_arrays(dict(artifact), labeled=True)
        if audit["sample_count"] != int(expected_counts[split]):
            raise ValidationError(f"{split}: expected {expected_counts[split]} samples")
        result[split] = audit
    with np.load(root / "attachment4_aligned.npz", allow_pickle=False) as artifact:
        result["attachment4"] = validate_split_arrays(dict(artifact), labeled=False)
    if result["attachment4"]["sample_count"] != 20:
        raise ValidationError("Attachment 4 must contain 20 samples")

    prototype_rows = _csv_rows(root / "prototypes" / "prototype_source_manifest.csv")
    if any(row.get("source_split") != "train" for row in prototype_rows):
        raise ValidationError("Prototype manifest contains a non-training source")
    result["prototype_candidates"] = len(prototype_rows)

    mapping_rows = _csv_rows(root / "mappings" / "attachment4_time_mapping.csv")
    if not mapping_rows:
        raise ValidationError("Attachment 4 time mapping is empty")
    for row in mapping_rows:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if start < 0 or end < start:
            raise ValidationError("Invalid time interval in Attachment 4 mapping")
    result["attachment4_mapping_rows"] = len(mapping_rows)

    leakage = _csv_rows(root / "leakage_watchlist.csv")
    result["leakage_matches"] = len(leakage)
    return result
