"""统一的中文图表样式。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _setup_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


_setup_font()

COLORS = ["#1f4e79", "#c47b2b", "#2f6f4e", "#8c3a4b", "#5c6b73", "#6b4c9a"]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def bar_chart(labels, values, path: Path, title: str, xlabel: str, ylabel: str, rotation: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.55), 4.2))
    ax.bar(range(len(labels)), values, color=COLORS[0])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(path)


def hist_chart(values, path: Path, title: str, xlabel: str, ylabel: str = "样本数", bins: int = 30) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(arr, bins=bins, color=COLORS[0], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(path)


def overlay_hist(series: dict, path: Path, title: str, xlabel: str, bins) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (name, values) in enumerate(series.items()):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        ax.hist(arr, bins=bins, density=True, alpha=0.45, label=name, color=COLORS[i % len(COLORS)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("密度")
    ax.legend()
    savefig(path)


def grouped_bar(categories, group_to_values: dict, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = list(group_to_values)
    x = np.arange(len(categories))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        ax.bar(x + i * width, group_to_values[name], width=width, label=name, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    savefig(path)


def heatmap(matrix, row_labels, col_labels, path: Path, title: str, cbar_label: str, vmin=None, vmax=None) -> None:
    fig_w = max(6, len(col_labels) * 0.85)
    fig_h = max(4, len(row_labels) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=cbar_label)
    savefig(path)


def scatter_chart(x, y, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(x, y, s=18, alpha=0.75, color=COLORS[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(path)
