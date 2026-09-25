from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"text": "#31688E", "audio": "#35B779", "vision": "#F39C34"}


def _configure() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 180


def plot_training_history(history: list[dict[str, Any]], path: Path) -> None:
    _configure()
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(frame["epoch"], frame["train_loss"], marker="o")
    axes[0, 0].set(title="训练损失", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].plot(frame["epoch"], frame["valid_accuracy"], label="Accuracy")
    axes[0, 1].plot(frame["epoch"], frame["valid_macro_f1"], label="Macro-F1")
    axes[0, 1].set(title="分类指标", xlabel="Epoch", ylabel="Score")
    axes[0, 1].legend()
    axes[1, 0].plot(frame["epoch"], frame["valid_mae"], label="MAE")
    axes[1, 0].plot(frame["epoch"], frame["valid_pearson"], label="Pearson")
    axes[1, 0].set(title="回归指标", xlabel="Epoch", ylabel="Score")
    axes[1, 0].legend()
    axes[1, 1].plot(frame["epoch"], frame["valid_selection_score"], label="目标感知选模分数")
    axes[1, 1].bar(frame["epoch"], frame["goals_met"] / 4.0, alpha=0.25, label="达标比例")
    axes[1, 1].set(title="选模与目标达成", xlabel="Epoch", ylabel="Score")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_validation(result: dict[str, Any], path: Path) -> None:
    _configure()
    matrix = np.asarray(result["metrics"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set_xticks([0, 1, 2], ["负向", "中性", "正向"])
    axes[0].set_yticks([0, 1, 2], ["负向", "中性", "正向"])
    axes[0].set(title="验证集混淆矩阵", xlabel="预测", ylabel="真实")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].scatter(result["label_regression"], result["prediction_regression"], s=14, alpha=0.45)
    axes[1].plot([-3, 3], [-3, 3], "--", color="#D1495B")
    axes[1].set(title="验证集强度预测", xlabel="真实强度", ylabel="预测强度", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    metrics = result["metrics"]
    fig.suptitle(
        f"Acc={metrics['accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  MAE={metrics['mae']:.3f}  "
        f"r={metrics['pearson']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_modality_contributions(explanations: list[dict[str, Any]], path: Path) -> None:
    _configure()
    ids = [row["sample_id"] for row in explanations]
    x = np.arange(len(ids))
    bottom = np.zeros(len(ids))
    fig, axis = plt.subplots(figsize=(max(12, len(ids) * 0.55), 5.5))
    for modality in ("text", "audio", "vision"):
        values = np.asarray([row["modality_contribution"][modality] for row in explanations])
        axis.bar(x, values, bottom=bottom, label=modality, color=COLORS[modality])
        bottom += values
    axis.set_xticks(x, ids, rotation=60, ha="right")
    axis.set(title="附件4样本级三模态作用程度", xlabel="样本", ylabel="归一化贡献")
    axis.legend(ncol=3)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_explanation_card(explanation: dict[str, Any], path: Path) -> None:
    _configure()
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    for row, modality in enumerate(("text", "audio", "vision")):
        importance = np.asarray(explanation["position_scores"][modality])[None, :]
        attention = np.asarray(explanation["time_attention"][modality])[None, :]
        axes[row, 0].imshow(attention, aspect="auto", cmap="Blues", vmin=0)
        axes[row, 1].imshow(importance, aspect="auto", cmap="Oranges", vmin=0)
        axes[row, 0].set_ylabel(modality)
        axes[row, 0].set_yticks([])
        axes[row, 1].set_yticks([])
    axes[0, 0].set_title("模型内稀疏时间注意力")
    axes[0, 1].set_title("积分梯度局部重要性")
    axes[-1, 0].set_xlabel("对齐位置 0–49")
    axes[-1, 1].set_xlabel("对齐位置 0–49")
    contribution = explanation["modality_contribution"]
    fig.suptitle(
        f"样本 {explanation['sample_id']} | {explanation['predicted_label']} | intensity={explanation['predicted_intensity']:.3f} | "
        f"主模态={explanation['main_modality']} | T/A/V={contribution['text']:.2f}/{contribution['audio']:.2f}/{contribution['vision']:.2f}\n"
        f"comprehensiveness={explanation['comprehensiveness']:.3f} | sufficiency={explanation['sufficiency']:.3f} | stability={explanation['stability']:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
