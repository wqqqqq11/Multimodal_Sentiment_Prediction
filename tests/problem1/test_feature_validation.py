import unittest
import numpy as np

from src.problem1.schemas import FeatureSequence


class FeatureValidationTest(unittest.TestCase):
    def test_rejects_non_monotone_positions(self):
        sequence = FeatureSequence("s", "text", np.zeros((2, 3)), np.array([0.8, 0.2]),
                                  np.array([0, 1]), np.array([1, 2]), np.ones(2),
                                  np.ones(2, bool), np.arange(2))
        with self.assertRaises(ValueError):
            sequence.validate()


if __name__ == "__main__":
    unittest.main()
