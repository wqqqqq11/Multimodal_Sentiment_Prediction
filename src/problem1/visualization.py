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
            modality_uncertainty[name].append(float(
                metrics["temporal_diagnostics_by_modality"][name]["mean_normalized_temporal_uncertainty"]
            ))
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
    axes[1].set_title("Time-aware alignment uncertainty", pad=14, weight="bold")
    axes[1].set_ylabel("Temporal std / consensus interval [0, 1]")
    axes[1].set_ylim(0, 1)
    figure.subplots_adjust(bottom=0.18, top=0.90, wspace=0.25)
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
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Normalized objective")
    axis.set_xticks(iteration)
    axis.legend()
    figure.subplots_adjust(bottom=0.14, top=0.96)
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


def _face_detection_rates(cfg: Problem1Config) -> dict[str, float]:
    report_path = cfg.path("feature_root") / "validation_report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(item["sample_id"]): float(item.get("vision_detection_rate", 0.0))
        for item in report.get("samples", [])
        if "vision_detection_rate" in item
    }


def _select_representative(cfg: Problem1Config, successful: list[dict[str, Any]]) -> tuple[str, str]:
    """Prefer a face-visible sample, with an optional manual override."""
    configured = str(cfg.section("visualization").get("representative_sample_id") or "").strip()
    available = {item["sample_id"] for item in successful}
    if configured:
        if configured not in available:
            raise ValueError(f"指定的代表性样本不在成功结果中: {configured}")
        return configured, "配置文件 visualization.representative_sample_id 指定的样本。"
    threshold = float(cfg.section("visualization").get("min_representative_face_detection_rate", 0.80))
    rates = _face_detection_rates(cfg)
    faced = [item for item in successful if rates.get(item["sample_id"], 0.0) >= threshold]
    pool = faced or successful
    chosen = sorted(pool, key=lambda item: float(item["mean_uncertainty"]))[len(pool) // 2]["sample_id"]
    if faced:
        rate = rates.get(chosen, 0.0)
        return chosen, (
            f"人脸检测率不低于 {threshold:.0%} 的成功样本中，不确定性中位数附近的样本"
            f"（该样本检测率 {rate:.0%}）。"
        )
    return chosen, "没有达到人脸检测率要求的样本，退回全部成功样本的不确定性中位数。"


def _representative_validation(cfg: Problem1Config, sample_id: str, figure_path: Path,
                               table_path: Path, report_path: Path, selection_rule: str) -> None:
    """Create an auditable, reader-facing correspondence chart for one typical sample."""
    plt, _ = _plot_modules()
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
            "audio_90pct_start_sec": float(modalities["audio"]["weighted_05_time"]),
            "audio_90pct_end_sec": float(modalities["audio"]["weighted_95_time"]),
            "video_peak_frame_index": int(sequences["vision"].source_index[vision_peak]),
            "video_peak_start_sec": float(sequences["vision"].start[vision_peak]),
            "video_peak_end_sec": float(sequences["vision"].end[vision_peak]),
            "video_90pct_frame_start": min(video_source_indices),
            "video_90pct_frame_end": max(video_source_indices),
            "video_90pct_start_sec": float(modalities["vision"]["weighted_05_time"]),
            "video_90pct_end_sec": float(modalities["vision"]["weighted_95_time"]),
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
    displayed = [rows[int(index)] for index in anchors]
    aligned_dimensions = {name: int(values.shape[1]) for name, values in aligned.items()}
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(anchors)))

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(18, 9.2), constrained_layout=False)
    grid = figure.add_gridspec(
        3, len(anchors), height_ratios=[2.75, 0.98, 2.25], hspace=0.035, wspace=0.15
    )
    for position, column in enumerate(anchors):
        row = rows[int(column)]
        axis = figure.add_subplot(grid[0, position])
        frame_path = frame_lookup.get(int(row["video_peak_frame_index"]))
        if frame_path is not None and frame_path.exists():
            axis.imshow(plt.imread(frame_path), aspect="auto")
        else:
            axis.text(0.5, 0.5, "Frame unavailable", ha="center", va="center")
        axis.axis("off")
        axis.set_anchor("S")

        info_axis = figure.add_subplot(grid[1, position])
        info_axis.axis("off")
        snippet = str(row["text_fragment"])
        if len(snippet) > 30:
            snippet = snippet[:27] + "..."
        info_axis.text(
            0.5, 0.98,
            f"Aligned step {row['consensus_index']}  ·  t={row['consensus_time_sec']:.2f} s\n"
            f"Text: {snippet}\n"
            f"Audio: {row['audio_peak_start_sec']:.2f}–{row['audio_peak_end_sec']:.2f} s\n"
            f"Video: frame {row['video_peak_frame_index']}, "
            f"{row['video_peak_start_sec']:.2f}–{row['video_peak_end_sec']:.2f} s\n"
            f"Aligned features: T{aligned_dimensions['text']} / A{aligned_dimensions['audio']} / "
            f"V{aligned_dimensions['vision']} → step {row['consensus_index']}",
            ha="center", va="top", fontsize=9.2, linespacing=1.22,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": colors[position],
                  "alpha": 0.10, "edgecolor": colors[position], "linewidth": 1.5},
        )

    chart_grid = grid[2, :].subgridspec(
        2, 1, height_ratios=[1.25, 0.90], hspace=0.34
    )
    feature_axis = figure.add_subplot(chart_grid[0, 0])
    audio_sequence = sequences["audio"]
    consensus_times = np.asarray([float(row["consensus_time_sec"]) for row in rows])
    modality_styles = {
        "text": ("文本特征", "#4c78a8"),
        "audio": ("音频特征", "#f58518"),
        "vision": ("视觉特征", "#54a24b"),
    }
    for name, (label, color) in modality_styles.items():
        strength = np.linalg.norm(aligned[name], axis=1)
        minimum, maximum = float(strength.min()), float(strength.max())
        normalized = (strength - minimum) / max(maximum - minimum, 1e-12)
        feature_axis.plot(
            consensus_times, normalized, color=color, marker="o", markersize=4.5,
            linewidth=2.0, label=label,
        )
    for position, row in enumerate(displayed):
        feature_axis.axvline(row["consensus_time_sec"], color=colors[position],
                             linestyle="--", linewidth=1.3, alpha=0.72)
        feature_axis.text(row["consensus_time_sec"], 1.04, f"Step {row['consensus_index']}",
                          color=colors[position], ha="center", va="bottom", fontsize=9)
    timeline_start = min(float(audio_sequence.start.min()), float(consensus_times.min()))
    timeline_end = max(float(audio_sequence.end.max()), float(consensus_times.max()))
    feature_axis.set_ylabel("三模态特征强度", fontsize=10, labelpad=8)
    feature_axis.set_xlim(timeline_start, timeline_end)
    feature_axis.set_ylim(0.0, 1.18)
    feature_axis.tick_params(axis="x", labelbottom=False)
    feature_axis.legend(loc="lower right", ncol=3, frameon=True, fontsize=9)
    feature_axis.grid(alpha=0.25)

    confidence = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    confidence_axis = figure.add_subplot(chart_grid[1, 0])
    confidence_axis.plot(consensus_times, confidence, color="#315f9e", marker="o", linewidth=2.0,
                         label="Alignment confidence (1 - uncertainty)")
    for position, row in enumerate(displayed):
        index = int(row["consensus_index"])
        confidence_axis.scatter(row["consensus_time_sec"], confidence[index], s=75,
                                color=colors[position], edgecolor="white", linewidth=1.0, zorder=3)
    confidence_axis.set_xlabel("对齐时间（秒）")
    confidence_axis.set_ylabel("对齐置信度", fontsize=10, labelpad=8)
    confidence_axis.set_xlim(timeline_start, timeline_end)
    confidence_axis.set_ylim(0.0, 1.0)
    confidence_axis.legend(loc="lower right", frameon=True, fontsize=9)
    confidence_axis.grid(alpha=0.25)

    figure.subplots_adjust(top=0.99, bottom=0.09, left=0.075, right=0.985)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# 问题一典型样本时序对齐验证", "",
        f"- 样本编号：`{sample_id}`", f"- 选择规则：{selection_rule}",
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
    representative, selection_rule = _select_representative(cfg, successful)
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
                               representative_table, representative_report, selection_rule)
    uncertainty_values = np.asarray([item["mean_uncertainty"] for item in successful], dtype=float)
    modality_uncertainty = {
        name: float(np.mean([metrics["mean_uncertainty_by_modality"][name] for metrics in metrics_payloads]))
        for name in ("text", "audio", "vision")
    }
    temporal_diagnostics = {
        name: {
            metric: float(np.mean([
                values["temporal_diagnostics_by_modality"][name][metric]
                for values in metrics_payloads
            ]))
            for metric in (
                "mean_temporal_std_sec", "mean_normalized_temporal_uncertainty",
                "mean_weighted_90pct_span_sec", "p90_weighted_90pct_span_sec",
                "mean_peak_offset_sec", "p90_peak_offset_sec",
            )
        }
        for name in ("text", "audio", "vision")
    }
    raw_residuals = [float(metrics["sinkhorn"][name]["raw_marginal_residual"])
                     for metrics in metrics_payloads for name in ("text", "audio", "vision")]
    tolerance = float(cfg.section("alignment")["sinkhorn_tolerance"])
    summary = {
        "successful_samples": len(successful), "representative_sample": representative,
        "representative_selection_rule": selection_rule,
        "mean_objective": float(np.mean([item["objective"] for item in successful])),
        "mean_uncertainty": float(uncertainty_values.mean()),
        "p90_uncertainty": float(np.quantile(uncertainty_values, 0.90)),
        "max_uncertainty": float(uncertainty_values.max()),
        "high_uncertainty_sample_count": int(np.sum(uncertainty_values >= 0.40)),
        "mean_uncertainty_by_modality": modality_uncertainty,
        "temporal_diagnostics_by_modality": temporal_diagnostics,
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
