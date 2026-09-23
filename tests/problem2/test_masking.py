from __future__ import annotations

import numpy as np

from data_progressing.problem2.masking import (
    ContinuousSpanMaskGenerator,
    MissingnessConfig,
    masks_from_attention,
    modality_availability,
)


def _config() -> MissingnessConfig:
    return MissingnessConfig(
        bank_size=3,
        pattern_probabilities={"audio": 0.2, "vision": 0.2, "audio_vision": 0.6},
        position_probabilities={"begin": 1 / 3, "middle": 1 / 3, "end": 1 / 3},
        ratio_min=0.05,
        ratio_max=0.48,
        beta_alpha=2.0,
        beta_beta=4.0,
        min_remaining=2,
    )


def test_attention_masks_separate_cls_sep_and_padding() -> None:
    attention = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    result = masks_from_attention(attention)
    np.testing.assert_array_equal(result["structural"], [[1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(result["content"], [[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    assert not np.any(result["valid"] & result["padding"])


def test_natural_zero_is_only_inside_content() -> None:
    features = np.array([[[0.0], [2.0], [0.0], [0.0], [0.0]]])
    content = np.array([[0, 1, 1, 0, 0]], dtype=bool)
    result = modality_availability(features, content)
    np.testing.assert_array_equal(result["observed"], [[0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(result["natural_zero"], [[0, 0, 1, 0, 0]])


def test_mask_bank_is_deterministic_and_masks_only_observed_steps() -> None:
    audio = np.array([[0, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    vision = np.array([[0, 1, 1, 0, 1, 1, 1, 0]], dtype=bool)
    first = ContinuousSpanMaskGenerator(_config(), seed=2026).generate(audio, vision)
    second = ContinuousSpanMaskGenerator(_config(), seed=2026).generate(audio, vision)
    np.testing.assert_array_equal(first["synthetic_missing_mask"], second["synthetic_missing_mask"])
    masks = first["synthetic_missing_mask"]
    assert not masks[:, :, 0].any()
    assert not np.any(masks[:, :, 1] & ~audio[:, None, :])
    assert not np.any(masks[:, :, 2] & ~vision[:, None, :])
    assert np.all((audio.sum(axis=1)[:, None] - masks[:, :, 1].sum(axis=2)) >= 2)
    assert np.all((vision.sum(axis=1)[:, None] - masks[:, :, 2].sum(axis=2)) >= 2)
