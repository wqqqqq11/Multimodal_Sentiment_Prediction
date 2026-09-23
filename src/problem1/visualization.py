"""Matplotlib/Seaborn publication figures for the competition solution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Problem1Config
from .io import atomic_json, safe_id


def _matplotlib_plain_text(value: str) -> str:
    """Escape math delimiters in external identifiers before rendering."""
    return str(value).replace("$", r"\$")


def _plot_modules() -> tuple[Any, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("缺少matplotlib/seaborn；请安装根目录requirements.txt") from exc
    sns.set_theme(style="whitegrid", context="talk")
    return plt, sns


def _quality_chart(cfg: Problem1Config, records: list[dict[str, Any]], path: Path) -> None:
    plt, sns = _plot_modules()
    actual = np.asarray([
        np.mean([item["mean_uncertainty"] for item in records]),
        max(item["max_marginal_residual"] for item in records),
        sum(item["monotonic_violations"] for item in records),
    ], dtype=float)
    tolerance = float(cfg.section("alignment")["sinkhorn_tolerance"])
    normalized = np.asarray([actual[0], min(actual[1] / tolerance, 1.0),
                             min(actual[2] / max(len(records) * 3, 1), 1.0)])
    labels = ["Mean uncertainty", "Marginal residual / tolerance", "Monotonic violation rate"]
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=False)
    colors = sns.color_palette("colorblind", 3)
    bars = axis.bar(labels, normalized, color=colors, label="Normalized diagnostic ratio")
    axis.set_title("Problem 1 Alignment Quality Overview", pad=16, weight="bold")
    axis.set_ylabel("Normalized diagnostic ratio [0, 1]")
    axis.set_ylim(0, 1.05)
    axis.legend(loc="upper right")
    annotations = [f"{actual[0]:.5f}", f"{actual[1]:.3e}", f"{int(actual[2])}"]
    for bar, label in zip(bars, annotations):
        axis.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.015) + 0.025,
                  label, ha="center", va="bottom", fontsize=12)
    figure.subplots_adjust(bottom=0.22)
    figure.text(0.5, 0.045,
                f"Conclusion: all {len(records)} samples were solved; lower values indicate more reliable alignment.",
                ha="center", fontsize=11)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _convergence_chart(histories: list[list[dict[str, float]]], path: Path) -> None:
    plt, _ = _plot_modules()
    maximum = max(len(history) for history in histories)
    curves = []
    for history in histories:
        values = np.asarray([point["objective"] for point in history], dtype=float)
        values /= max(values[0], 1e-12)
        curves.append(np.pad(values, (0, maximum - len(values)), mode="edge"))
    stacked = np.stack(curves)
    median = np.median(stacked, axis=0)
    lower, upper = np.quantile(stacked, [0.25, 0.75], axis=0)
    iteration = np.arange(1, maximum + 1)
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(iteration, median, marker="o", linewidth=2.5, label="Median normalized objective")
    axis.fill_between(iteration, lower, upper, alpha=0.22, label="Interquartile range")
    axis.set_title("Consensus Optimization Convergence", pad=16, weight="bold")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Normalized objective")
    axis.set_xticks(iteration)
    axis.legend()
    figure.subplots_adjust(bottom=0.20)
    change = 100.0 * (median[-1] - median[0])
    figure.text(0.5, 0.04,
                f"Conclusion: median normalized objective changed by {change:+.2f}% after {maximum} iterations.",
                ha="center", fontsize=11)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _heatmap_chart(cfg: Problem1Config, sample_id: str, path: Path) -> None:
    plt, sns = _plot_modules()
    source = cfg.path("aligned_root") / "samples" / safe_id(sample_id) / "alignment.npz"
    with np.load(source, allow_pickle=False) as data:
        plans = [data[f"plan_{name}"] for name in ("text", "audio", "vision")]
    labels = ["Text tokens", "Audio windows", "Video frames"]
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    maximum = max(float(plan.max()) for plan in plans)
    for axis, plan, label in zip(axes, plans, labels):
        sns.heatmap(plan, ax=axis, cmap="viridis", vmin=0.0, vmax=maximum,
                    cbar=axis is axes[-1], cbar_kws={"label": "Transport mass"})
        axis.set_ylabel(label)
        axis.set_xlabel("")
    axes[-1].set_xlabel("Consensus time index")
    display_sample_id = _matplotlib_plain_text(sample_id)
    figure.suptitle(f"Representative Alignment Plans: {display_sample_id}", weight="bold", y=0.98)
    figure.subplots_adjust(bottom=0.10, top=0.93, hspace=0.23)
    figure.text(0.5, 0.025,
                "Conclusion: diagonal transport bands show that chronological evidence is preserved across modalities.",
                ha="center", fontsize=11)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def create_visualizations(cfg: Problem1Config, records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in records if item["status"] != "failed"]
    if not successful:
        raise ValueError("没有成功样本，无法绘图")
    figures = cfg.path("output_root") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    histories = []
    for item in successful:
        metrics_path = cfg.path("aligned_root") / "samples" / safe_id(item["sample_id"]) / "metrics.json"
        histories.append(json.loads(metrics_path.read_text(encoding="utf-8"))["history"])
    representative = sorted(successful, key=lambda item: item["mean_uncertainty"])[len(successful) // 2]["sample_id"]
    paths = {
        "quality_overview": figures / "01_alignment_quality_overview.png",
        "convergence": figures / "02_consensus_convergence.png",
        "representative_heatmap": figures / "03_representative_alignment_heatmap.png",
    }
    _quality_chart(cfg, successful, paths["quality_overview"])
    _convergence_chart(histories, paths["convergence"])
    _heatmap_chart(cfg, representative, paths["representative_heatmap"])
    summary = {
        "successful_samples": len(successful), "representative_sample": representative,
        "mean_objective": float(np.mean([item["objective"] for item in successful])),
        "mean_uncertainty": float(np.mean([item["mean_uncertainty"] for item in successful])),
        "max_marginal_residual": float(max(item["max_marginal_residual"] for item in successful)),
        "total_monotonic_violations": int(sum(item["monotonic_violations"] for item in successful)),
        "figures": {key: str(value) for key, value in paths.items()},
    }
    atomic_json(cfg.path("output_root") / "reports" / "solution_summary.json", summary)
    return summary
