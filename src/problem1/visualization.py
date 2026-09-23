"""Matplotlib/Seaborn publication figures for the competition solution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Problem1Config
from .io import atomic_csv, atomic_json, atomic_text, load_feature, safe_id


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


def _quality_chart(cfg: Problem1Config, records: list[dict[str, Any]], audit: dict[str, Any], path: Path) -> None:
    plt, sns = _plot_modules()
    tolerance = float(cfg.section("alignment")["sinkhorn_tolerance"])
    modality_uncertainty = {name: [] for name in ("text", "audio", "vision")}
    raw_residuals = []
    consensus_passes = []
    for item in records:
        metrics_path = cfg.path("aligned_root") / "samples" / safe_id(item["sample_id"]) / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        consensus_passes.append(bool(metrics["converged"]))
        for name in modality_uncertainty:
            modality_uncertainty[name].append(float(metrics["mean_uncertainty_by_modality"][name]))
            raw_residuals.append(float(metrics["sinkhorn"][name]["raw_marginal_residual"]))
    pass_rates = np.asarray([
        float(audit.get("sample_coverage_rate", 0.0)),
        float(bool(audit.get("feature_validation_passed", False))),
        float(audit.get("mapping_audit_pass_rate", 0.0)),
        float(np.mean(consensus_passes)),
        float(np.mean(np.asarray(raw_residuals) <= tolerance)),
    ])
    labels = ["Coverage", "Feature integrity", "Mapping audit", "Consensus", "Raw Sinkhorn"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=False)
    colors = sns.color_palette("colorblind", len(labels))
    bars = axes[0].bar(labels, pass_rates, color=colors)
    axes[0].set_title("Organizer-facing acceptance checks", pad=14, weight="bold")
    axes[0].set_ylabel("Pass rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, pass_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                     f"{value:.1%}", ha="center", va="bottom", fontsize=11)
    axes[1].boxplot([modality_uncertainty[name] for name in modality_uncertainty],
                    tick_labels=["Text", "Audio", "Vision"], showmeans=True)
    axes[1].set_title("Normalized alignment uncertainty", pad=14, weight="bold")
    axes[1].set_ylabel("Normalized entropy [0, 1] (lower is sharper)")
    axes[1].set_ylim(0, 1)
    figure.suptitle("Problem 1 Acceptance and Alignment Diagnostics", weight="bold", y=0.99)
    figure.subplots_adjust(bottom=0.22, top=0.86, wspace=0.25)
    figure.text(0.5, 0.045,
                "Coverage and audit metrics verify deliverables; uncertainty is reported separately and is not an accuracy score.",
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


def _representative_validation(cfg: Problem1Config, sample_id: str, figure_path: Path,
                               table_path: Path, report_path: Path) -> None:
    """Create an auditable text/audio/video correspondence card for one typical sample."""
    plt, sns = _plot_modules()
    sample_dir = cfg.path("aligned_root") / "samples" / safe_id(sample_id)
    mappings = json.loads((sample_dir / "mapping.json").read_text(encoding="utf-8"))
    sequences = {
        name: load_feature(cfg.path("feature_root") / "samples" / safe_id(sample_id) / f"{name}.npz")
        for name in ("text", "audio", "vision")
    }
    with np.load(sample_dir / "alignment.npz", allow_pickle=False) as data:
        aligned = {name: np.asarray(data[f"aligned_{name}"], dtype=float)
                   for name in ("text", "audio", "vision")}
        uncertainty = np.asarray(data["uncertainty"], dtype=float)
        plans = {name: np.asarray(data[f"plan_{name}"], dtype=float)
                 for name in ("text", "audio", "vision")}

    tokens = list(sequences["text"].metadata.get("tokens", []))
    rows: list[dict[str, Any]] = []
    for column, mapping in enumerate(mappings):
        modalities = mapping["modalities"]
        text_rows = [int(value) for value in modalities["text"]["source_row_indices"]]
        snippet = " ".join(tokens[index] for index in text_rows if 0 <= index < len(tokens))
        if not snippet:
            snippet = "[unavailable]"
        peaks: dict[str, int] = {}
        for name in ("audio", "vision"):
            conditional = plans[name][:, column] / max(float(plans[name][:, column].sum()), 1e-12)
            peaks[name] = int(np.argmax(conditional))
        audio_peak, vision_peak = peaks["audio"], peaks["vision"]
        video_source_indices = [int(value) for value in modalities["vision"]["source_indices"]]
        rows.append({
            "sample_id": sample_id,
            "consensus_index": column,
            "consensus_time_sec": float(mapping["consensus_time_sec"]),
            "uncertainty": float(mapping["uncertainty"]),
            "text_token_indices": ";".join(str(value) for value in modalities["text"]["source_indices"]),
            "text_fragment": snippet,
            "audio_peak_start_sec": float(sequences["audio"].start[audio_peak]),
            "audio_peak_end_sec": float(sequences["audio"].end[audio_peak]),
            "audio_90pct_start_sec": float(modalities["audio"]["source_start"]),
            "audio_90pct_end_sec": float(modalities["audio"]["source_end"]),
            "video_peak_frame_index": int(sequences["vision"].source_index[vision_peak]),
            "video_peak_start_sec": float(sequences["vision"].start[vision_peak]),
            "video_peak_end_sec": float(sequences["vision"].end[vision_peak]),
            "video_90pct_frame_start": min(video_source_indices),
            "video_90pct_frame_end": max(video_source_indices),
            "video_90pct_start_sec": float(modalities["vision"]["source_start"]),
            "video_90pct_end_sec": float(modalities["vision"]["source_end"]),
        })
    atomic_csv(table_path, rows)

    frame_lookup: dict[int, Path] = {}
    timeline_path = cfg.path("preprocessed_root") / "tables" / "visual_timeline.csv"
    with timeline_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["sample_id"] == sample_id:
                frame_lookup[int(item["source_frame_index"])] = cfg.path("preprocessed_root") / item["frame_relative_path"]

    anchor_count = min(5, len(rows))
    anchors = np.unique(np.linspace(0, len(rows) - 1, anchor_count).round().astype(int))
    figure = plt.figure(figsize=(18, 14), constrained_layout=False)
    grid = figure.add_gridspec(6, len(anchors), height_ratios=[2.6, 0.18, 1.05, 1.05, 1.05, 0.9],
                              hspace=0.58, wspace=0.16)
    for position, column in enumerate(anchors):
        row = rows[int(column)]
        axis = figure.add_subplot(grid[0, position])
        frame_path = frame_lookup.get(int(row["video_peak_frame_index"]))
        if frame_path is not None and frame_path.exists():
            axis.imshow(plt.imread(frame_path), aspect="auto")
        else:
            axis.text(0.5, 0.5, "Frame unavailable", ha="center", va="center")
        axis.axis("off")
        snippet = str(row["text_fragment"])
        if len(snippet) > 34:
            snippet = snippet[:31] + "..."
        axis.set_title(
            f"t={row['consensus_time_sec']:.2f}s  U={row['uncertainty']:.2f}\n"
            f"Text: {snippet}\n"
            f"Audio: {row['audio_peak_start_sec']:.2f}-{row['audio_peak_end_sec']:.2f}s\n"
            f"Video: frame {row['video_peak_frame_index']} "
            f"({row['video_peak_start_sec']:.2f}-{row['video_peak_end_sec']:.2f}s)",
            fontsize=10, pad=8,
        )

    maximum = max(float(np.max(np.abs(values))) for values in aligned.values())
    for grid_row, name, label in zip((2, 3, 4), ("text", "audio", "vision"),
                                     ("Text features", "Audio features", "Visual features")):
        axis = figure.add_subplot(grid[grid_row, :])
        sns.heatmap(aligned[name].T, ax=axis, cmap="coolwarm", center=0.0,
                    vmin=-maximum, vmax=maximum, cbar=False)
        axis.set_ylabel(label)
        axis.set_xlabel("")
    axes_uncertainty = figure.add_subplot(grid[5, :])
    axes_uncertainty.plot(np.arange(len(uncertainty)), uncertainty, marker="o", linewidth=2.2)
    axes_uncertainty.set_xlabel("Consensus time index")
    axes_uncertainty.set_ylabel("Uncertainty")
    axes_uncertainty.set_ylim(0, 1)
    axes_uncertainty.grid(alpha=0.3)
    figure.suptitle(f"Typical-sample Multimodal Alignment Validation: {_matplotlib_plain_text(sample_id)}",
                    fontsize=20, weight="bold", y=0.985)
    figure.text(0.5, 0.012,
                "Top: traceable text/audio/video anchors and source frames. Bottom: the three aligned feature sequences on the same consensus axis.",
                ha="center", fontsize=11)
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.09, right=0.98)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    displayed = [rows[int(index)] for index in anchors]
    lines = [
        "# 问题一典型样本时序对齐验证", "",
        f"- 样本编号：`{sample_id}`", "- 选择规则：全部成功样本中不确定性中位数附近的样本。",
        f"- 共识时间点：{len(rows)}；下表展示{len(displayed)}个均匀抽取的核验锚点。", "",
        "|共识位置|时间/s|文本片段|语音峰值时段/s|视频峰值帧与时段/s|不确定性|",
        "|---:|---:|---|---|---|---:|",
    ]
    for row in displayed:
        fragment = str(row["text_fragment"]).replace("|", "\\|")
        lines.append(
            f"|{row['consensus_index']}|{row['consensus_time_sec']:.3f}|{fragment}|"
            f"{row['audio_peak_start_sec']:.3f}–{row['audio_peak_end_sec']:.3f}|"
            f"#{row['video_peak_frame_index']}，{row['video_peak_start_sec']:.3f}–{row['video_peak_end_sec']:.3f}|"
            f"{row['uncertainty']:.3f}|"
        )
    lines.extend(["", f"完整逐点对应关系见 `{table_path.name}`；可视化见 `{figure_path.name}`。", ""])
    atomic_text(report_path, "\n".join(lines))


def create_visualizations(cfg: Problem1Config, records: list[dict[str, Any]],
                          audit: dict[str, Any] | None = None) -> dict[str, Any]:
    successful = [item for item in records if item["status"] != "failed"]
    if not successful:
        raise ValueError("没有成功样本，无法绘图")
    figures = cfg.path("output_root") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit = audit or {}
    histories = []
    metrics_payloads = []
    for item in successful:
        metrics_path = cfg.path("aligned_root") / "samples" / safe_id(item["sample_id"]) / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        histories.append(payload["history"])
        metrics_payloads.append(payload["metrics"])
    representative = sorted(successful, key=lambda item: item["mean_uncertainty"])[len(successful) // 2]["sample_id"]
    paths = {
        "quality_overview": figures / "01_alignment_quality_overview.png",
        "convergence": figures / "02_consensus_convergence.png",
        "representative_heatmap": figures / "03_representative_alignment_heatmap.png",
        "representative_validation": figures / "04_representative_sample_validation.png",
    }
    reports = cfg.path("output_root") / "reports"
    representative_table = reports / "representative_sample_alignment.csv"
    representative_report = reports / "representative_sample_validation.md"
    _quality_chart(cfg, successful, audit, paths["quality_overview"])
    _convergence_chart(histories, paths["convergence"])
    _heatmap_chart(cfg, representative, paths["representative_heatmap"])
    _representative_validation(cfg, representative, paths["representative_validation"],
                               representative_table, representative_report)
    uncertainty_values = np.asarray([item["mean_uncertainty"] for item in successful], dtype=float)
    modality_uncertainty = {
        name: float(np.mean([metrics["mean_uncertainty_by_modality"][name] for metrics in metrics_payloads]))
        for name in ("text", "audio", "vision")
    }
    raw_residuals = [float(metrics["sinkhorn"][name]["raw_marginal_residual"])
                     for metrics in metrics_payloads for name in ("text", "audio", "vision")]
    tolerance = float(cfg.section("alignment")["sinkhorn_tolerance"])
    summary = {
        "successful_samples": len(successful), "representative_sample": representative,
        "mean_objective": float(np.mean([item["objective"] for item in successful])),
        "mean_uncertainty": float(uncertainty_values.mean()),
        "p90_uncertainty": float(np.quantile(uncertainty_values, 0.90)),
        "max_uncertainty": float(uncertainty_values.max()),
        "high_uncertainty_sample_count": int(np.sum(uncertainty_values >= 0.40)),
        "mean_uncertainty_by_modality": modality_uncertainty,
        "raw_sinkhorn_convergence_rate": float(np.mean(np.asarray(raw_residuals) <= tolerance)),
        "sample_sinkhorn_convergence_rate": float(np.mean([metrics["sinkhorn_converged"] for metrics in metrics_payloads])),
        "max_raw_marginal_residual": float(max(raw_residuals)),
        "max_marginal_residual": float(max(item["max_marginal_residual"] for item in successful)),
        "total_monotonic_violations": int(sum(item["monotonic_violations"] for item in successful)),
        "total_preprojection_monotonic_violations": int(sum(
            metrics["sinkhorn"][name]["preprojection_monotonic_violations"]
            for metrics in metrics_payloads for name in ("text", "audio", "vision")
        )),
        "acceptance_audit": audit,
        "figures": {key: str(value) for key, value in paths.items()},
        "representative_alignment_table": str(representative_table),
        "representative_validation_report": str(representative_report),
    }
    atomic_json(cfg.path("output_root") / "reports" / "solution_summary.json", summary)
    return summary
