import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.problem1.io import load_feature, save_feature
from src.problem1.schemas import FeatureSequence


class FeaturePipelineTest(unittest.TestCase):
    def test_feature_roundtrip(self):
        sequence = FeatureSequence("id", "text", np.ones((2, 4), np.float32), np.array([0.25, 0.75]),
                                  np.array([0, 1]), np.array([1, 2]), np.ones(2), np.ones(2, bool),
                                  np.arange(2), {"tokens": ["a", "b"]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feature.npz"
            save_feature(path, sequence)
            restored = load_feature(path)
        np.testing.assert_array_equal(sequence.features, restored.features)
        self.assertEqual(restored.metadata["tokens"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
