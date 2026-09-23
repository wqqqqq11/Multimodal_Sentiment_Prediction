from __future__ import annotations

import numpy as np

from data_progressing.problem2.validation import validate_mask_bank, validate_split_arrays


def _artifact() -> dict[str, np.ndarray]:
    content = np.array([[0, 1, 1, 0, 0]], dtype=bool)
    structural = np.array([[1, 0, 0, 1, 0]], dtype=bool)
    observed = content.copy()
    return {
        "sample_id": np.array(["x"]),
        "raw_text": np.array(["hello"]),
        "input_ids": np.ones((1, 5), dtype=np.int32),
        "attention_mask": content | structural,
        "token_type_ids": np.zeros((1, 5), dtype=np.int8),
        "content_mask": content,
        "structural_mask": structural,
        "padding_mask": np.array([[0, 0, 0, 0, 1]], dtype=bool),
        "text_observed_mask": observed,
        "text_natural_zero_mask": np.zeros_like(content),
        "text_missing_mask": np.zeros_like(content),
        "privileged_text": np.zeros((1, 768), dtype=np.float32),
        "audio": np.where(observed[..., None], 1.0, 0.0).repeat(74, axis=2).astype(np.float32),
        "vision": np.where(observed[..., None], 1.0, 0.0).repeat(35, axis=2).astype(np.float32),
        "audio_observed_mask": observed,
        "audio_natural_zero_mask": np.zeros_like(content),
        "audio_missing_mask": np.zeros_like(content),
        "vision_observed_mask": observed,
        "vision_natural_zero_mask": np.zeros_like(content),
        "vision_missing_mask": np.zeros_like(content),
        "modality_reliability": np.ones((1, 3, 4), dtype=np.float32),
        "classification_labels": np.array([2]),
        "regression_labels": np.array([1.0], dtype=np.float32),
    }


def test_split_contract_and_mask_bank() -> None:
    artifact = _artifact()
    result = validate_split_arrays(artifact, labeled=True)
    assert result["sample_count"] == 1
    bank = {
        "synthetic_missing_mask": np.zeros((1, 2, 3, 5), dtype=bool),
        "pattern_id": np.ones((1, 2), dtype=np.uint8),
        "position_id": np.zeros((1, 2, 3), dtype=np.int8),
        "relative_start": np.zeros((1, 2, 3), dtype=np.float32),
        "actual_missing_ratio": np.zeros((1, 2, 3), dtype=np.float32),
    }
    bank["synthetic_missing_mask"][0, 0, 1, 1] = True
    validated = validate_mask_bank(bank, artifact)
    assert validated["audio_masked_steps"] == 1
