from __future__ import annotations

import numpy as np

from data_progressing.problem3.media_mapping import reconstruct_to_content, reconstruction_metrics
from data_progressing.problem3.text_mapping import intensity_bin
from data_progressing.problem3.validation import ValidationError, validate_split_arrays


def _minimal_arrays() -> dict[str, np.ndarray]:
    attention = np.asarray([[1, 1, 1, 0]], dtype=np.uint8)
    content = np.asarray([[0, 1, 0, 0]], dtype=bool)
    structural = np.asarray([[1, 0, 1, 0]], dtype=bool)
    padding = ~attention.astype(bool)
    observed = content.copy()
    zeros = np.zeros_like(content)
    return {
        "sample_id": np.asarray(["x"]),
        "raw_text": np.asarray(["hello"]),
        "input_ids": np.asarray([[101, 7592, 102, 0]], dtype=np.int32),
        "attention_mask": attention,
        "token_type_ids": np.zeros((1, 4), dtype=np.uint8),
        "content_mask": content,
        "structural_mask": structural,
        "padding_mask": padding,
        "audio": np.zeros((1, 4, 74), dtype=np.float32),
        "vision": np.zeros((1, 4, 35), dtype=np.float32),
        "text_observed_mask": observed,
        "audio_observed_mask": zeros,
        "vision_observed_mask": zeros,
        "text_natural_zero_mask": zeros,
        "audio_natural_zero_mask": content,
        "vision_natural_zero_mask": content,
        "text_failure_mask": zeros,
        "audio_failure_mask": zeros,
        "vision_failure_mask": zeros,
        "text_evidence_candidate_mask": observed,
        "audio_evidence_candidate_mask": zeros,
        "vision_evidence_candidate_mask": zeros,
        "truncation_flag": np.asarray([False]),
        "tokenizer_exact_match": np.asarray([True]),
        "covered_char_end": np.asarray([5], dtype=np.int32),
        "token_mapping_confidence": np.asarray([1.0], dtype=np.float32),
        "classification_labels": np.asarray([2], dtype=np.int8),
        "regression_labels": np.asarray([1.25], dtype=np.float32),
        "intensity_bin": np.asarray([5], dtype=np.int8),
    }


def test_intensity_bin_boundaries() -> None:
    values = np.asarray([-3, -2, -1, 0, 0.1, 1, 2, 3], dtype=np.float32)
    assert intensity_bin(values).tolist() == [0, 1, 2, 3, 4, 4, 5, 6]


def test_unaligned_reconstruction_preserves_padding_zero() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
    content = np.asarray([False, True, True, True, False])
    reconstructed, length = reconstruct_to_content(source, content, eps=1e-8)
    assert length == 2
    assert reconstructed.shape == (5, 2)
    assert np.all(reconstructed[~content] == 0)
    assert np.allclose(reconstructed[content][0], source[0])
    assert np.allclose(reconstructed[content][-1], source[1])


def test_reconstruction_metrics_identity() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = reconstruction_metrics(reference, reference.copy(), np.asarray([True, True]))
    assert np.isclose(result["median_cosine"], 1.0)
    assert np.isclose(result["mse"], 0.0)


def test_main_contract_rejects_privileged_bypass() -> None:
    arrays = _minimal_arrays()
    arrays["privileged_text"] = np.zeros((1, 768), dtype=np.float32)
    try:
        validate_split_arrays(arrays, labeled=True)
    except ValidationError as exc:
        assert "forbidden bypass" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("privileged_text must be rejected")


def test_main_contract_accepts_valid_minimal_case() -> None:
    audit = validate_split_arrays(_minimal_arrays(), labeled=True)
    assert audit["sample_count"] == 1
    assert audit["content_steps"] == 1

