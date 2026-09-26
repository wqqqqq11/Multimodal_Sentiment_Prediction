from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"text": "#31688E", "audio": "#35B779", "vision": "#FDE725", "audio_vision": "#D1495B", "all_modalities": "#7B2CBF"}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    teacher = history[history["stage"] == "teacher"]
    student = history[history["stage"] == "student"]
    axes[0, 0].plot(teacher["epoch"], teacher["train_loss"], marker="o", label="教师训练损失")
    axes[0, 0].plot(student["epoch"], student["train_loss"], marker="o", label="学生训练损失")
    axes[0, 0].set(title="训练损失收敛曲线", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(teacher["epoch"], teacher["valid_macro_f1"], label="教师完整验证 F1")
    axes[0, 1].plot(student["epoch"], student["complete_macro_f1"], label="学生完整验证 F1")
    axes[0, 1].plot(student["epoch"], student["target30_macro_f1"], label="学生三模态同步缺失30% F1")
    axes[0, 1].set(title="分类性能随训练变化", xlabel="Epoch", ylabel="Macro-F1")
    axes[0, 1].legend()
    axes[1, 0].plot(teacher["epoch"], teacher["valid_mae"], label="教师完整验证 MAE")
    axes[1, 0].plot(student["epoch"], student["complete_mae"], label="学生完整验证 MAE")
    axes[1, 0].plot(student["epoch"], student["target30_mae"], label="学生三模态同步缺失30% MAE")
    axes[1, 0].set(title="回归误差随训练变化", xlabel="Epoch", ylabel="MAE")
    axes[1, 0].legend()
    axes[1, 1].plot(student["epoch"], student["train_supervised"], label="监督损失")
    axes[1, 1].plot(student["epoch"], student["train_distillation"], label="蒸馏损失")
    axes[1, 1].set(title="学生监督与蒸馏损失", xlabel="Epoch", ylabel="损失分量")
    axes[1, 1].legend()
    _save(fig, path)


def plot_robustness_curves(frame: pd.DataFrame, path: Path, primary_position: str = "middle") -> None:
    configure_matplotlib()
    data = frame[(frame["position"] == primary_position) & (frame["pattern"] != "none")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = {"text": "文本缺失", "audio": "语音缺失", "vision": "视觉缺失", "audio_vision": "语音+视觉缺失", "all_modalities": "三模态同步缺失"}
    for pattern, group in data.groupby("pattern"):
        group = group.sort_values("missing_rate")
        axes[0].plot(group["missing_rate"] * 100, group["macro_f1"], marker="o", label=labels[pattern], color=COLORS[pattern])
        axes[1].plot(group["missing_rate"] * 100, group["mae"], marker="o", label=labels[pattern], color=COLORS[pattern])
    axes[0].set(title=f"{primary_position}位置缺失率与分类性能", xlabel="人工缺失率（%）", ylabel="Macro-F1")
    axes[1].set(title=f"{primary_position}位置缺失率与回归误差", xlabel="人工缺失率（%）", ylabel="MAE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    _save(fig, path)


def plot_position_effect(frame: pd.DataFrame, path: Path) -> None:
    configure_matplotlib()
    data = frame[frame["pattern"] != "none"].groupby("position", as_index=False)[["macro_f1_drop", "mae_increase"]].mean()
    positions = [value for value in ("begin", "middle", "end") if value in set(data["position"])]
    data = data.set_index("position").loc[positions].reset_index()
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x, data["macro_f1_drop"], color="#4C78A8")
    axes[1].bar(x, data["mae_increase"], color="#F58518")
    for axis, title, ylabel in zip(axes, ("不同缺失位置的平均 F1 降幅", "不同缺失位置的平均 MAE 增量"), ("Macro-F1 降幅", "MAE 增量"), strict=True):
        axis.set_xticks(x, data["position"])
        axis.set(title=title, xlabel="连续缺失位置", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_ablation(frame: pd.DataFrame, path: Path) -> None:
    configure_matplotlib()
    scenario = "all_modalities_30_middle"
    data = frame[frame["scenario"] == scenario]
    labels = {
        "teacher_without_missing_training": "完整教师",
        "student_full": "完整学生",
        "student_uniform_gate": "去可靠性门控",
        "student_text_only": "仅文本",
    }
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, data["macro_f1"], color="#4C78A8")
    axes[1].bar(x, data["mae"], color="#F58518")
    tick_labels = [labels[value] for value in data["variant"]]
    for axis, title, ylabel in zip(axes, ("三模态同步缺失30%分类消融", "三模态同步缺失30%回归消融"), ("Macro-F1（越高越好）", "MAE（越低越好）"), strict=True):
        axis.set_xticks(x, tick_labels, rotation=18, ha="right")
        axis.set(title=title, xlabel="模型变体", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_confusion_and_regression(result: dict[str, Any], path: Path, split_name: str) -> None:
    configure_matplotlib()
    matrix = np.asarray(result["metrics"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set(title=f"{split_name}情感极性混淆矩阵", xlabel="预测类别", ylabel="真实类别")
    axes[0].set_xticks([0, 1, 2], ["负向", "中性", "正向"])
    axes[0].set_yticks([0, 1, 2], ["负向", "中性", "正向"])
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].scatter(result["label_regression"], result["prediction_regression"], s=12, alpha=0.45, color="#2A9D8F")
    axes[1].plot([-3, 3], [-3, 3], linestyle="--", color="#D1495B", label="理想预测")
    axes[1].set(title=f"{split_name}情感强度预测", xlabel="真实情感强度", ylabel="预测情感强度", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    axes[1].legend()
    _save(fig, path)


def plot_challenge_predictions(frame: pd.DataFrame, path: Path) -> None:
    configure_matplotlib()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(x, frame["predicted_intensity"], marker="o", label="预测情感强度", color="#D1495B")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="附件3全量样本情感强度预测", ylabel="情感强度 [-3,3]")
    axes[0].legend()
    bottom = np.zeros(len(frame))
    for key, label, color in (("gate_text_expert", "文本专家", COLORS["text"]), ("gate_full_expert", "完整专家", COLORS["audio"]), ("gate_missing_expert", "缺失专家", COLORS["vision"])):
        axes[1].bar(x, frame[key], bottom=bottom, label=label, color=color)
        bottom += frame[key].to_numpy()
    axes[1].set(ylabel="路由权重")
    axes[1].legend(ncol=3)
    axes[1].set_xticks(x, frame["sample_id"], rotation=90, fontsize=7)
    _save(fig, path)
