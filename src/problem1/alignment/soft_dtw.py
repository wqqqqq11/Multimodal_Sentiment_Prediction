"""Log-domain differentiable DTW path occupancy."""

from __future__ import annotations

import math
import numpy as np


def _logadd3(a: float, b: float, c: float) -> float:
    maximum = max(a, b, c)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum) + math.exp(c - maximum))


def soft_dtw_occupancy(cost: np.ndarray, gamma: float) -> tuple[np.ndarray, float]:
    """Return node marginals of the Gibbs distribution over monotone DTW paths."""
    if gamma <= 0:
        raise ValueError("Soft-DTW gamma必须>0")
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Soft-DTW代价矩阵必须为有限二维数组")
    n, m = matrix.shape
    log_weight = -np.clip(matrix, 0.0, 100.0) / gamma
    forward = np.full((n, m), -np.inf, dtype=np.float64)
    forward[0, 0] = log_weight[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            forward[i, j] = log_weight[i, j] + _logadd3(
                forward[i - 1, j] if i else -math.inf,
                forward[i, j - 1] if j else -math.inf,
                forward[i - 1, j - 1] if i and j else -math.inf)
    backward = np.full((n, m), -np.inf, dtype=np.float64)
    backward[-1, -1] = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if i == n - 1 and j == m - 1:
                continue
            backward[i, j] = _logadd3(
                log_weight[i + 1, j] + backward[i + 1, j] if i + 1 < n else -math.inf,
                log_weight[i, j + 1] + backward[i, j + 1] if j + 1 < m else -math.inf,
                log_weight[i + 1, j + 1] + backward[i + 1, j + 1]
                if i + 1 < n and j + 1 < m else -math.inf)
    log_partition = forward[-1, -1]
    occupancy = np.exp(np.clip(forward + backward - log_partition, -745.0, 0.0))
    return occupancy, float(-gamma * log_partition)
