import unittest
import numpy as np

from src.problem1.feature_extraction.audio.extractor import _quality_per_step


class AudioFeatureTest(unittest.TestCase):
    def test_quality_tracks_energy(self):
        signal = np.concatenate((np.zeros(100), np.full(100, 0.3, dtype=np.float32)))
        quality = _quality_per_step(signal, np.asarray([0, 100, 200]))
        self.assertEqual(quality.shape, (2,))
        self.assertGreater(quality[1], quality[0])


if __name__ == "__main__":
    unittest.main()
