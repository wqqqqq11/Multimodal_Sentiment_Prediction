"""Strict admission gate for the competition-grade official-tokenizer artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from data_progressing.problem3.config import Problem3Config
from data_progressing.problem3.validation import ValidationError, validate_saved_root


EXPECTED_TOKENIZER = "google-bert/bert-base-uncased"
EXPECTED_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def validate_official_artifacts(config: Problem3Config) -> dict[str, Any]:
    tokenizer = config.section("tokenizer")
    if tokenizer.get("name") != EXPECTED_TOKENIZER:
        raise ValidationError(f"Official run requires tokenizer {EXPECTED_TOKENIZER}")
    if tokenizer.get("revision") != EXPECTED_REVISION:
        raise ValidationError(f"Official run requires pinned revision {EXPECTED_REVISION}")
    if not bool(tokenizer.get("local_files_only")):
        raise ValidationError("Official preprocessing must be offline after tokenizer caching")

    root = config.path("preprocessed_root")
    audit = validate_saved_root(root, config.section("validation")["expected_split_counts"])
    for split in ("train", "valid", "test", "attachment4"):
        if int(audit[split]["tokenizer_mismatches"]) != 0:
            raise ValidationError(f"{split}: tokenizer mismatch forbids competition admission")
        path = root / ("attachment4_aligned.npz" if split == "attachment4" else f"{split}.npz")
        with np.load(path, allow_pickle=False) as artifact:
            if not np.all(artifact["tokenizer_exact_match"]):
                raise ValidationError(f"{split}: approximate token mapping detected")
            if not np.allclose(artifact["token_mapping_confidence"], 1.0):
                raise ValidationError(f"{split}: token mapping confidence must be 1.0")

    token_path = root / "mappings" / "token_offsets.csv"
    with token_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValidationError("Token-offset mapping is empty")
    if any(row["tokenizer_exact_match"].casefold() != "true" for row in rows):
        raise ValidationError("Token-offset CSV contains an approximate mapping")
    if any(abs(float(row["mapping_confidence"]) - 1.0) > 1e-8 for row in rows):
        raise ValidationError("Token-offset CSV contains confidence below 1.0")
    audit["official_tokenizer_gate"] = {
        "name": EXPECTED_TOKENIZER,
        "revision": EXPECTED_REVISION,
        "token_rows": len(rows),
        "mismatches": 0,
        "minimum_token_mapping_confidence": 1.0,
        "passed": True,
    }
    return audit

