"""Entropy-regularized optimal transport solver."""

from __future__ import annotations

from typing import Any
import numpy as np


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _round_marginals(plan: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Altschuler-style nonnegative rounding to exact requested marginals."""
    rounded = plan * np.minimum(a / np.maximum(plan.sum(axis=1), 1e-300), 1.0)[:, None]
    rounded *= np.minimum(b / np.maximum(rounded.sum(axis=0), 1e-300), 1.0)[None, :]
    row_error = np.maximum(a - rounded.sum(axis=1), 0.0)
    column_error = np.maximum(b - rounded.sum(axis=0), 0.0)
    missing = float(row_error.sum())
    if missing > 1e-15:
        rounded += np.outer(row_error, column_error) / missing
    return rounded


def sinkhorn(cost: np.ndarray, source_mass: np.ndarray, target_mass: np.ndarray, *, epsilon: float,
             iterations: int, tolerance: float) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(cost, dtype=np.float64)
    a = np.maximum(np.asarray(source_mass, dtype=np.float64), 1e-12)
    b = np.maximum(np.asarray(target_mass, dtype=np.float64), 1e-12)
    a, b = a / a.sum(), b / b.sum()
    log_kernel = -(matrix - np.min(matrix)) / epsilon
    log_a, log_b = np.log(a), np.log(b)
    log_u, log_v, residual = np.zeros_like(a), np.zeros_like(b), np.inf
    used = 0
    for used in range(1, iterations + 1):
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel.T + log_u[None, :], axis=1)
        if used == 1 or used % 5 == 0 or used == iterations:
            plan = np.exp(np.clip(log_u[:, None] + log_kernel + log_v[None, :], -745.0, 700.0))
            residual = max(float(np.max(np.abs(plan.sum(axis=1) - a))),
                           float(np.max(np.abs(plan.sum(axis=0) - b))))
            if residual <= tolerance:
                break
    plan = np.exp(np.clip(log_u[:, None] + log_kernel + log_v[None, :], -745.0, 700.0))
    raw_residual = residual
    plan = _round_marginals(plan, a, b)
    residual = max(float(np.max(np.abs(plan.sum(axis=1) - a))),
                   float(np.max(np.abs(plan.sum(axis=0) - b))))
    entropy = -float(np.sum(plan * np.log(np.maximum(plan, 1e-30))))
    return plan, {"iterations": used, "raw_marginal_residual": raw_residual,
                  "marginal_residual": residual, "entropy": entropy,
                  "transport_cost": float(np.sum(plan * matrix))}


def monotone_coupling(source_mass: np.ndarray, target_mass: np.ndarray) -> np.ndarray:
    """Exact 1-D increasing (quantile) coupling with both marginals preserved."""
    a = np.maximum(np.asarray(source_mass, dtype=np.float64), 0.0)
    b = np.maximum(np.asarray(target_mass, dtype=np.float64), 0.0)
    a, b = a / a.sum(), b / b.sum()
    plan = np.zeros((len(a), len(b)), dtype=np.float64)
    remaining_a, remaining_b = a.copy(), b.copy()
    i = j = 0
    while i < len(a) and j < len(b):
        mass = min(remaining_a[i], remaining_b[j])
        plan[i, j] += mass
        remaining_a[i] -= mass
        remaining_b[j] -= mass
        if remaining_a[i] <= 1e-14:
            i += 1
        if remaining_b[j] <= 1e-14:
            j += 1
    return plan
