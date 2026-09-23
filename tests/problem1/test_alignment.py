import unittest
import numpy as np

from src.problem1.alignment.sinkhorn import sinkhorn
from src.problem1.alignment.soft_dtw import soft_dtw_occupancy
from src.problem1.visualization import _matplotlib_plain_text


class AlignmentPrimitiveTest(unittest.TestCase):
    def test_soft_dtw_has_monotone_diagonal_support(self):
        axis = np.linspace(0, 1, 12)
        cost = (axis[:, None] - axis[None, :]) ** 2
        occupancy, value = soft_dtw_occupancy(cost, 0.1)
        self.assertEqual(occupancy.shape, cost.shape)
        self.assertTrue(np.isfinite(value))
        expected = np.sum(occupancy * np.arange(12)[:, None], axis=0) / np.maximum(occupancy.sum(axis=0), 1e-12)
        self.assertTrue(np.all(np.diff(expected) >= -1e-8))

    def test_sinkhorn_marginals(self):
        cost = np.abs(np.arange(8)[:, None] / 7 - np.arange(5)[None, :] / 4)
        a, b = np.arange(1, 9, dtype=float), np.ones(5)
        plan, metrics = sinkhorn(cost, a, b, epsilon=0.15, iterations=500, tolerance=1e-7)
        np.testing.assert_allclose(plan.sum(axis=1), a / a.sum(), atol=2e-6)
        np.testing.assert_allclose(plan.sum(axis=0), b / b.sum(), atol=2e-6)
        self.assertLessEqual(metrics["raw_marginal_residual"], 1e-7)
        self.assertLess(metrics["marginal_residual"], 2e-6)

    def test_sample_id_is_safe_for_matplotlib_mathtext(self):
        self.assertEqual(_matplotlib_plain_text("-3g5yACwYnA$_$13"), r"-3g5yACwYnA\$_\$13")


if __name__ == "__main__":
    unittest.main()
