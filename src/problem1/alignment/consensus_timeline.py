"""Soft-DTW constrained Sinkhorn alignment on a tri-modal consensus timeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import Problem1Config
from ..schemas import AlignmentResult, FeatureSequence
from .multiscale import interpolate_signature, multiscale_signatures
from .sinkhorn import monotone_coupling, sinkhorn
from .soft_dtw import soft_dtw_occupancy

MODALITIES = ("text", "audio", "vision")


def _effective_quality(sequence: FeatureSequence, quality_floor: float) -> np.ndarray:
    """Keep observed evidence above the numerical floor and exclude missing frames."""
    quality = np.asarray(sequence.quality, dtype=np.float64)
    valid = np.asarray(sequence.valid_mask, dtype=np.bool_)
    if not np.any(valid):
        raise ValueError(f"{sequence.sample_id}/{sequence.modality}: 没有可用于对齐的有效时间步")
    return np.where(valid, np.maximum(quality, quality_floor), 0.0)


def _pairwise_cost(source: np.ndarray, target: np.ndarray, source_positions: np.ndarray,
                   target_positions: np.ndarray, quality: np.ndarray, band: float,
                   params: dict[str, Any]) -> np.ndarray:
    cosine = np.clip(source @ target.T, -1.0, 1.0)
    semantic = 1.0 - cosine
    time_delta = np.abs(source_positions[:, None] - target_positions[None, :])
    time_cost = np.where(time_delta <= band, 0.5 * (time_delta / max(band, 1e-8)) ** 2,
                         time_delta / max(band, 1e-8) - 0.5)
    quality_cost = -np.log(np.clip(quality[:, None], 1e-6, 1.0))
    outside = np.maximum(time_delta - band, 0.0) / max(band, 1e-8)
    return (float(params["semantic_weight"]) * semantic + float(params["time_weight"]) * time_cost +
            float(params["quality_weight"]) * quality_cost + 3.0 * outside)


def _smooth(values: np.ndarray, weight: float) -> np.ndarray:
    if len(values) < 3 or weight <= 0:
        return values
    result = values.copy()
    result[1:-1] = ((1.0 - weight) * values[1:-1] +
                    0.5 * weight * (values[:-2] + values[2:]))
    return result


def _evidence_indices(plan: np.ndarray, column: int, mass: float) -> tuple[np.ndarray, float]:
    conditional = plan[:, column] / max(float(plan[:, column].sum()), 1e-12)
    order = np.argsort(conditional)[::-1]
    count = int(np.searchsorted(np.cumsum(conditional[order]), mass, side="left") + 1)
    chosen = np.sort(order[:max(1, count)])
    return chosen, float(conditional[chosen].sum())


def align_sample(sequences: dict[str, FeatureSequence], cfg: Problem1Config,
                 logger: logging.Logger | None = None) -> AlignmentResult:
    logger = logger or logging.getLogger("problem1.alignment")
    if set(sequences) != set(MODALITIES):
        raise ValueError(f"需要三模态特征，实际={sorted(sequences)}")
    sample_id = sequences["text"].sample_id
    if any(sequence.sample_id != sample_id for sequence in sequences.values()):
        raise ValueError("三模态sample_id不一致")
    align_cfg, multi_cfg = cfg.section("alignment"), cfg.section("multiscale")
    radii = {name: list(multi_cfg[f"{name}_radii"]) for name in MODALITIES}
    scales = {name: multiscale_signatures(sequences[name], radii[name]) for name in MODALITIES}
    scale_weights = np.asarray(multi_cfg["scale_weights"], dtype=np.float64)
    duration = max(float(sequences["audio"].end[-1]), float(sequences["vision"].end[-1]))
    steps = min(int(align_cfg["max_consensus_steps"]),
                max(2, int(np.ceil(duration / float(align_cfg["consensus_interval_sec"]))) + 1))
    target_positions = np.linspace(0.0, 1.0, steps)
    consensus_time = target_positions * duration
    effective_quality = {
        name: _effective_quality(sequences[name], float(align_cfg["quality_floor"]))
        for name in MODALITIES
    }
    modality_quality = {
        name: float(effective_quality[name][sequences[name].valid_mask].mean())
        for name in MODALITIES
    }
    modality_weights = np.asarray([modality_quality[name] for name in MODALITIES], dtype=np.float64)
    modality_weights /= modality_weights.sum()
    interpolated = [interpolate_signature(sequences[name], scales[name][0], target_positions)
                    for name in MODALITIES]
    consensus = np.average(np.stack(interpolated), axis=0, weights=modality_weights)
    consensus /= np.maximum(np.linalg.norm(consensus, axis=1, keepdims=True), 1e-8)

    # Soft-DTW is evaluated at all three temporal scales. Its Gibbs path occupancy is
    # then held as a monotonic prior during the alternating Sinkhorn/barycenter updates.
    priors: dict[str, np.ndarray] = {}
    soft_values: dict[str, list[float]] = {}
    bands = list(align_cfg["time_bands"])
    gammas = list(align_cfg["soft_dtw_gamma"])
    for name in MODALITIES:
        paths, values = [], []
        sequence = sequences[name]
        for index, signature in enumerate(scales[name]):
            cost = _pairwise_cost(signature, consensus, sequence.positions, target_positions,
                                  effective_quality[name], float(bands[index]), align_cfg)
            occupancy, value = soft_dtw_occupancy(cost, float(gammas[index]))
            paths.append(occupancy / max(float(occupancy.sum()), 1e-12))
            values.append(value)
        priors[name] = np.average(np.stack(paths), axis=0, weights=scale_weights)
        soft_values[name] = values

    plans: dict[str, np.ndarray] = {}
    aligned: dict[str, np.ndarray] = {}
    metrics_by_modality: dict[str, dict[str, Any]] = {}
    history: list[dict[str, float]] = []
    target_mass = np.full(steps, 1.0 / steps)
    converged = False
    for iteration in range(1, int(align_cfg["consensus_iterations"]) + 1):
        aligned_current, objective = {}, 0.0
        time_candidates, time_weights = [], []
        for name in MODALITIES:
            sequence = sequences[name]
            scale_costs = [_pairwise_cost(signature, consensus, sequence.positions, target_positions,
                                          effective_quality[name], float(bands[index]), align_cfg)
                           for index, signature in enumerate(scales[name])]
            fused = np.average(np.stack(scale_costs), axis=0, weights=scale_weights)
            prior = priors[name] / max(float(priors[name].max()), 1e-12)
            fused = fused - float(align_cfg["path_prior_weight"]) * np.log(prior + 1e-8)
            # Missing visual frames retain an auditable zero feature row, but receive
            # only Sinkhorn's numerical epsilon mass and therefore cannot steer alignment.
            source_mass = effective_quality[name]
            plan, transport_metrics = sinkhorn(
                fused, source_mass, target_mass, epsilon=float(align_cfg["sinkhorn_epsilon"]),
                iterations=int(align_cfg["sinkhorn_iterations"]),
                tolerance=float(align_cfg["sinkhorn_tolerance"]),
            )
            projection_weight = float(align_cfg.get("monotone_projection_weight", 0.0))
            if projection_weight > 0:
                increasing = monotone_coupling(source_mass, target_mass)
                semantic_plan = plan
                while True:
                    plan = (1.0 - projection_weight) * semantic_plan + projection_weight * increasing
                    expected_index = np.sum(plan * np.arange(plan.shape[0])[:, None], axis=0) / target_mass
                    if np.all(np.diff(expected_index) >= -1e-8) or projection_weight >= 1.0 - 1e-8:
                        break
                    projection_weight = min(1.0, 1.0 - 0.5 * (1.0 - projection_weight))
                normalized_mass = source_mass / source_mass.sum()
                transport_metrics["marginal_residual"] = max(
                    float(np.max(np.abs(plan.sum(axis=1) - normalized_mass))),
                    float(np.max(np.abs(plan.sum(axis=0) - target_mass))),
                )
                transport_metrics["transport_cost"] = float(np.sum(plan * fused))
                transport_metrics["entropy"] = -float(np.sum(plan * np.log(np.maximum(plan, 1e-30))))
                transport_metrics["monotone_projection_weight_used"] = projection_weight
            plans[name] = plan
            metrics_by_modality[name] = transport_metrics
            aligned_current[name] = (plan.T @ scales[name][0]) / target_mass[:, None]
            objective += modality_quality[name] * float(transport_metrics["transport_cost"])
            if name in ("audio", "vision"):
                centers = (sequence.start + sequence.end) / 2
                time_candidates.append((plan.T @ centers) / target_mass)
                time_weights.append(modality_quality[name])
        updated = np.average(np.stack([aligned_current[name] for name in MODALITIES]), axis=0,
                             weights=modality_weights)
        updated = _smooth(updated, float(align_cfg["smooth_weight"]))
        updated /= np.maximum(np.linalg.norm(updated, axis=1, keepdims=True), 1e-8)
        delta = float(np.sqrt(np.mean((updated - consensus) ** 2)))
        consensus = updated
        aligned = aligned_current
        consensus_time = np.average(np.stack(time_candidates), axis=0, weights=np.asarray(time_weights))
        consensus_time = np.clip(np.maximum.accumulate(consensus_time), 0.0, duration)
        history.append({"iteration": float(iteration), "objective": float(objective), "consensus_delta": delta})
        logger.info("%s 对齐迭代 %d/%d: objective=%.6f, delta=%.6f", sample_id, iteration,
                    int(align_cfg["consensus_iterations"]), objective, delta)
        if delta <= float(align_cfg["consensus_tolerance"]):
            converged = True
            break

    uncertainty_parts = []
    for name in MODALITIES:
        conditional = plans[name] / np.maximum(plans[name].sum(axis=0, keepdims=True), 1e-12)
        entropy = -np.sum(conditional * np.log(np.maximum(conditional, 1e-30)), axis=0)
        uncertainty_parts.append(entropy / max(np.log(max(2, conditional.shape[0])), 1e-8))
    uncertainty = np.average(np.stack(uncertainty_parts), axis=0, weights=modality_weights)
    mappings: list[dict[str, Any]] = []
    for column in range(steps):
        item: dict[str, Any] = {"consensus_index": column, "consensus_time_sec": float(consensus_time[column]),
                                "uncertainty": float(uncertainty[column]), "modalities": {}}
        for name in MODALITIES:
            indices, covered = _evidence_indices(plans[name], column, float(align_cfg["evidence_mass"]))
            sequence = sequences[name]
            item["modalities"][name] = {
                "source_row_indices": indices.tolist(), "source_indices": sequence.source_index[indices].tolist(),
                "source_start": float(sequence.start[indices].min()), "source_end": float(sequence.end[indices].max()),
                "covered_mass": covered,
            }
        mappings.append(item)
    residual = max(float(value["marginal_residual"]) for value in metrics_by_modality.values())
    expected = {name: np.sum(plans[name] * np.arange(plans[name].shape[0])[:, None], axis=0) / target_mass
                for name in MODALITIES}
    monotonic_violations = {name: int(np.sum(np.diff(values) < -1e-8)) for name, values in expected.items()}
    metrics: dict[str, Any] = {
        "duration_sec": duration, "consensus_steps": steps, "iterations": len(history), "converged": converged,
        "objective": history[-1]["objective"], "consensus_delta": history[-1]["consensus_delta"],
        "mean_uncertainty": float(uncertainty.mean()), "max_marginal_residual": residual,
        "modality_quality": modality_quality, "modality_weights": dict(zip(MODALITIES, modality_weights.tolist())),
        "soft_dtw_values": soft_values, "sinkhorn": metrics_by_modality,
        "monotonic_violations": monotonic_violations,
    }
    result = AlignmentResult(sample_id=sample_id, consensus=consensus.astype(np.float32),
                             consensus_time=consensus_time.astype(np.float32),
                             aligned={name: aligned[name].astype(np.float32) for name in MODALITIES},
                             valid_mask=np.ones(steps, dtype=np.bool_), uncertainty=uncertainty.astype(np.float32),
                             plans={name: plans[name].astype(np.float32) for name in MODALITIES},
                             mappings=mappings, metrics=metrics, history=history)
    return result
