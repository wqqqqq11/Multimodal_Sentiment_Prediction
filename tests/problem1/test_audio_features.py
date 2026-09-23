import unittest
import numpy as np

from src.problem1.feature_extraction.audio.extractor import (
    _quality_per_step,
    _step_audio_diagnostics,
    _wavlm_time_intervals,
)


class AudioFeatureTest(unittest.TestCase):
    def test_quality_tracks_energy(self):
        signal = np.concatenate((np.zeros(100), np.full(100, 0.3, dtype=np.float32)))
        quality = _quality_per_step(signal, np.asarray([0, 100, 200]))
        self.assertEqual(quality.shape, (2,))
        self.assertGreater(quality[1], quality[0])

    def test_activity_mask_excludes_silence_and_keeps_speech(self):
        signal = np.concatenate((np.zeros(200), np.full(200, 0.3, dtype=np.float32)))
        starts = np.asarray([0, 100, 200, 300])
        ends = np.asarray([100, 200, 300, 400])
        quality, activity, valid = _step_audio_diagnostics(
            signal, starts, ends,
            {"vad_enabled": True, "vad_threshold": 0.15,
             "vad_noise_percentile": 20.0, "vad_margin_db": 3.0},
        )
        self.assertEqual(quality.shape, (4,))
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[-1]))
        self.assertGreater(activity[-1], activity[0])

    def test_wavlm_intervals_follow_convolution_receptive_field(self):
        starts, ends, positions = _wavlm_time_intervals(
            output_steps=2, input_samples=20, sample_rate=10,
            kernels=[3, 3], strides=[2, 2],
        )
        np.testing.assert_allclose(starts, [0.0, 0.4])
        np.testing.assert_allclose(ends, [0.7, 1.1])
        np.testing.assert_allclose(positions, [0.175, 0.375])


if __name__ == "__main__":
    unittest.main()
