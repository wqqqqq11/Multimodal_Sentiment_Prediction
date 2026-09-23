from __future__ import annotations

import numpy as np

from data_progressing.problem2.scaling import RobustFeatureScaler


def test_scaler_uses_observed_values_and_preserves_unavailable_zero() -> None:
    features = np.array(
        [
            [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 30.0], [1000.0, np.nan], [0.0, 0.0]],
        ]
    )
    observed = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
    scaler = RobustFeatureScaler(
        lower_quantile=0.0,
        upper_quantile=0.75,
        minimum_scale=1e-6,
        normalized_clip=8.0,
    )
    scaler.fit("audio", features, observed)
    transformed, audit = scaler.transform("audio", features, observed)
    assert np.isfinite(transformed).all()
    assert np.all(transformed[~observed] == 0)
    assert audit["nonfinite_elements"] == 1
    assert audit["winsorized_elements"] >= 1


def test_scaler_does_not_fit_padding_zeros() -> None:
    features = np.array([[[0.0], [10.0], [20.0], [0.0]]])
    observed = np.array([[0, 1, 1, 0]], dtype=bool)
    scaler = RobustFeatureScaler(
        lower_quantile=0.0,
        upper_quantile=1.0,
        minimum_scale=1e-6,
        normalized_clip=8.0,
    )
    stats = scaler.fit("vision", features, observed)
    assert stats.center[0] == 15.0
