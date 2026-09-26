"""Descriptive analysis for the independent Problem 3 preprocessing artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_progressing.problem3.config import Problem3Config
from data_progressing.problem3.io import ensure_dir, write_csv, write_json


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {name: artifact[name] for name in artifact.files}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _label_table(splits: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, arrays in splits.items():
        counts = Counter(arrays["classification_labels"].astype(int).tolist())
        bin_counts = Counter(arrays["intensity_bin"].astype(int).tolist())
        for label in range(3):
            rows.append({
                "split": split, "target": "polarity", "category": label,
                "count": counts[label], "ratio": counts[label] / len(arrays["sample_id"]),
            })
        for label in range(7):
            rows.append({
                "split": split, "target": "intensity_bin", "category": label,
                "count": bin_counts[label], "ratio": bin_counts[label] / len(arrays["sample_id"]),
            })
    return rows


def _sequence_table(all_splits: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, arrays in all_splits.items():
        for index, sample_id in enumerate(arrays["sample_id"].astype(str)):
            row: dict[str, Any] = {
                "split": split, "sample_id": sample_id,
                "content_steps": int(arrays["content_mask"][index].sum()),
                "truncated": bool(arrays["truncation_flag"][index]),
            }
            for modality in ("text", "audio", "vision"):
                row[f"{modality}_candidate_steps"] = int(
                    arrays[f"{modality}_evidence_candidate_mask"][index].sum()
                )
            rows.append(row)
    return rows


def _plot_labels(rows: list[dict[str, Any]], path: Path) -> None:
    polarity = [row for row in rows if row["target"] == "polarity"]
    splits = ("train", "valid", "test")
    x = np.arange(3)
    width = 0.24
    for offset, split in enumerate(splits):
        values = [next(row["count"] for row in polarity if row["split"] == split and row["category"] == label) for label in x]
        plt.bar(x + (offset - 1) * width, values, width, label=split)
    plt.xticks(x, ["negative", "neutral", "positive"])
    plt.ylabel("samples")
    plt.title("Problem 2 labels reused for Problem 3 supervision")
    plt.legend()
    _save_figure(path)


def _plot_lengths(rows: list[dict[str, Any]], path: Path) -> None:
    for split in ("train", "valid", "test", "attachment4"):
        values = [int(row["content_steps"]) for row in rows if row["split"] == split]
        plt.hist(values, bins=np.arange(0.5, 50.5, 2), alpha=0.45, label=split)
    plt.xlabel("content tokens")
    plt.ylabel("samples")
    plt.title("Aligned content-length distribution")
    plt.legend()
    _save_figure(path)


def _plot_candidate_ratios(all_splits: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    names = list(all_splits)
    modalities = ("text", "audio", "vision")
    values = np.zeros((len(names), len(modalities)), dtype=float)
    for i, split in enumerate(names):
        arrays = all_splits[split]
        denominator = max(int(arrays["content_mask"].sum()), 1)
        for j, modality in enumerate(modalities):
            values[i, j] = arrays[f"{modality}_evidence_candidate_mask"].sum() / denominator
    x = np.arange(len(names))
    width = 0.24
    for j, modality in enumerate(modalities):
        plt.bar(x + (j - 1) * width, values[:, j], width, label=modality)
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel("candidate/content ratio")
    plt.title("Healthy evidence-candidate coverage")
    plt.legend()
    _save_figure(path)


def _plot_attachment4_media(arrays: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].bar(arrays["sample_id"].astype(str), arrays["duration_sec"])
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Attachment 4 video duration")
    resolutions = [f"{w}x{h}" for w, h in zip(arrays["video_width"], arrays["video_height"])]
    counts = Counter(resolutions)
    axes[1].bar(list(counts), list(counts.values()))
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("videos")
    axes[1].set_title("Resolution distribution")
    _save_figure(path)


def _plot_drift(rows: list[dict[str, str]], path: Path) -> None:
    sample_ids = sorted({row["sample_id"] for row in rows})
    x = np.arange(len(sample_ids))
    width = 0.38
    for offset, modality in enumerate(("audio", "vision")):
        lookup = {row["sample_id"]: float(row["outside_train_quantile_ratio"]) for row in rows if row["modality"] == modality}
        plt.bar(x + (offset - 0.5) * width, [lookup[sid] for sid in sample_ids], width, label=modality)
    plt.xticks(x, sample_ids, rotation=90)
    plt.ylabel("outside training quantiles")
    plt.title("Attachment 4 feature drift after training-only reference")
    plt.legend()
    _save_figure(path)


def _plot_prototypes(rows: list[dict[str, str]], path: Path) -> None:
    keys = [(modality, label) for modality in ("text", "audio", "vision") for label in range(3)]
    counts = Counter((row["modality"], int(row["classification_label"])) for row in rows)
    labels = [f"{modality[0].upper()}-{label}" for modality, label in keys]
    plt.bar(labels, [counts[key] for key in keys])
    plt.ylabel("candidate positions")
    plt.title("Training-only prototype candidate coverage")
    _save_figure(path)


def run_analysis(config: Problem3Config) -> dict[str, Any]:
    source = config.path("preprocessed_root")
    output = ensure_dir(config.path("output_root") / "data_analysis")
    tables = ensure_dir(output / "tables")
    figures = ensure_dir(output / "figures")
    labeled = {split: _load_npz(source / f"{split}.npz") for split in ("train", "valid", "test")}
    attachment4 = _load_npz(source / "attachment4_aligned.npz")
    all_splits = {**labeled, "attachment4": attachment4}

    label_rows = _label_table(labeled)
    sequence_rows = _sequence_table(all_splits)
    write_csv(tables / "dataset02_label_distribution.csv", label_rows, list(label_rows[0]))
    write_csv(tables / "sequence_and_candidate_counts.csv", sequence_rows, list(sequence_rows[0]))

    quality_rows = _read_csv(source / "quality" / "sample_quality.csv")
    attachment4_audit = [row for row in quality_rows if row["split"] == "attachment4"]
    write_csv(tables / "attachment4_sample_audit.csv", attachment4_audit, list(attachment4_audit[0]))
    drift_rows = _read_csv(source / "quality" / "feature_drift.csv")
    write_csv(tables / "feature_distribution_shift.csv", drift_rows, list(drift_rows[0]))
    prototype_rows = _read_csv(source / "prototypes" / "prototype_source_manifest.csv")
    prototype_counts = Counter(
        (row["modality"], int(row["classification_label"]), int(row["intensity_bin"]))
        for row in prototype_rows
    )
    prototype_table = [
        {"modality": key[0], "classification_label": key[1], "intensity_bin": key[2], "count": count}
        for key, count in sorted(prototype_counts.items())
    ]
    write_csv(tables / "prototype_cell_counts.csv", prototype_table, list(prototype_table[0]))
    shutil.copyfile(source / "leakage_watchlist.csv", tables / "leakage_watchlist.csv")

    _plot_labels(label_rows, figures / "label_distribution.png")
    _plot_lengths(sequence_rows, figures / "content_length_distribution.png")
    _plot_candidate_ratios(all_splits, figures / "evidence_candidate_coverage.png")
    _plot_attachment4_media(attachment4, figures / "attachment4_video_profile.png")
    _plot_drift(drift_rows, figures / "attachment4_feature_shift.png")
    _plot_prototypes(prototype_rows, figures / "prototype_candidate_coverage.png")

    leakage_rows = _read_csv(source / "leakage_watchlist.csv")
    failed_vision = [
        str(sid) for sid, mask in zip(attachment4["sample_id"], attachment4["vision_failure_mask"])
        if np.asarray(mask).any()
    ]
    summary = {
        "sample_counts": {name: len(arrays["sample_id"]) for name, arrays in all_splits.items()},
        "attachment4_truncated_ids": attachment4["sample_id"][attachment4["truncation_flag"]].astype(str).tolist(),
        "attachment4_visual_failure_ids": failed_vision,
        "attachment4_duplicate_text_matches": len(leakage_rows),
        "prototype_candidates": len(prototype_rows),
        "mapping_caveat": "Time/frame evidence uses a proportional fallback and must be reported as low-confidence.",
        "modeling_boundary": "This analysis creates no classifier, regressor, selector, or fusion model.",
    }
    write_json(output / "summary.json", summary)
    report = f"""# 问题三数据分析报告

## 结论摘要

- 问题三监督数据沿用问题二划分：训练 {summary['sample_counts']['train']}、验证 {summary['sample_counts']['valid']}、测试 {summary['sample_counts']['test']}；附件4为 {summary['sample_counts']['attachment4']} 个最终待预测样本。
- 附件4中发生截断风险的样本：{', '.join(summary['attachment4_truncated_ids']) or '无'}。
- 附件4中被规则判定为视觉流严重失效的样本：{', '.join(failed_vision) or '无'}。修复数据只作为独立候选，不覆盖官方对齐特征。
- 发现 {len(leakage_rows)} 条附件4与历史划分的规范化文本精确匹配，仅列入泄漏监控，禁止复制历史标签。
- 训练集生成 {len(prototype_rows)} 个稀疏原型候选位置，来源严格限定为 train。

## 对建模的约束

1. 主输入不得包含问题二的 `privileged_text` 或手工 `modality_reliability`，防止绕开稀疏证据瓶颈。
2. 模态原型只能由训练集候选构建；验证集只用于超参数选择和一致性评估。
3. 文本字符区间可靠；音视频时间区间目前是按字符覆盖比例映射的低置信度回退结果，论文与结果文件必须显式披露。
4. 反事实模板只定义扰动语义和可复现种子，本阶段未执行任何模型推断。

## 产物索引

- `tables/dataset02_label_distribution.csv`：极性与强度分箱分布。
- `tables/sequence_and_candidate_counts.csv`：各样本长度与证据候选数。
- `tables/attachment4_sample_audit.csv`：最终集逐样本质量审计。
- `tables/feature_distribution_shift.csv`：附件4相对训练尺度的越界率。
- `tables/prototype_cell_counts.csv`：模态×极性×强度候选覆盖。
- `figures/`：上述关键关系的可视化。
"""
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    return summary

