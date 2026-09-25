from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#2F5597", "#ED7D31", "#70AD47")


def _save(fig: Any, path: Path, conclusion: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.015, f"Conclusion: {conclusion}", ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training(history: list[dict[str, Any]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(epochs, [row["train"]["total"] for row in history], label="train loss", color=COLORS[0])
    axes[0].set(title="SEPC-Net Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation"]["macro_f1"] for row in history], label="Macro-F1", color=COLORS[0])
    axes[1].plot(epochs, [row["validation"]["pearson"] for row in history], label="Pearson", color=COLORS[2])
    axes[1].plot(epochs, [row["validation"]["mae"] for row in history], label="MAE", color=COLORS[1])
    axes[1].set(title="Validation Performance", xlabel="Epoch", ylabel="Metric")
    axes[1].legend()
    best = max(history, key=lambda row: float(row["validation"]["selection_score"]))
    _save(fig, path, f"The composite validation score peaks at epoch {best['epoch']}.")


def plot_validation_performance(result: dict[str, Any], path: Path) -> None:
    predictions = result["predictions"]
    matrix = np.asarray(result["metrics"]["confusion_matrix"])
    true_reg = np.asarray([row["true_intensity"] for row in predictions])
    pred_reg = np.asarray([row["predicted_intensity"] for row in predictions])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    image = axes[0].imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0].set(title="Polarity Confusion Matrix", xlabel="Predicted class", ylabel="True class", xticks=range(3), yticks=range(3))
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].scatter(true_reg, pred_reg, alpha=0.45, s=18, color=COLORS[0], label="samples")
    axes[1].plot([-3, 3], [-3, 3], "--", color="black", label="ideal")
    axes[1].set(title="Emotion Intensity Prediction", xlabel="True intensity", ylabel="Predicted intensity", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    axes[1].legend()
    metrics = result["metrics"]
    _save(fig, path, f"Accuracy={metrics['accuracy']:.3f}, Macro-F1={metrics['macro_f1']:.3f}, MAE={metrics['mae']:.3f}, Pearson={metrics['pearson']:.3f}.")


def plot_modality_contributions(result: dict[str, Any], path: Path) -> None:
    rows = result["predictions"]
    names = ("text", "audio", "vision")
    means = [np.mean([float(row[f"{name}_contribution"]) for row in rows]) for name in names]
    primary = [sum(row["main_modality"] == name for row in rows) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].bar(names, means, color=COLORS)
    axes[0].set(title="Mean Modality Contribution", xlabel="Modality", ylabel="Normalized Shapley contribution", ylim=(0, 1))
    axes[1].bar(names, primary, color=COLORS)
    axes[1].set(title="Primary Modality Frequency", xlabel="Modality", ylabel="Samples")
    dominant = names[int(np.argmax(means))]
    _save(fig, path, f"{dominant} has the largest mean contribution; sample-level values remain the authoritative explanation.")


def plot_fidelity(result: dict[str, Any], path: Path) -> None:
    metrics = result["fidelity_metrics"]
    labels = ["Suff. prob gap", "Suff. intensity gap", "Comp. prob drop", "Comp. intensity change", "Stability Jaccard"]
    keys = ["sufficiency_probability_gap", "sufficiency_regression_gap", "comprehensiveness_probability_drop", "comprehensiveness_regression_change", "selection_stability_jaccard"]
    values = [float(metrics[key]) for key in keys]
    fig, axis = plt.subplots(figsize=(9, 4.6))
    axis.bar(labels, values, color=(COLORS[0], COLORS[0], COLORS[1], COLORS[1], COLORS[2]))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="Explanation Fidelity and Stability", xlabel="Diagnostic", ylabel="Mean value")
    axis.tick_params(axis="x", rotation=20)
    _save(fig, path, "Smaller sufficiency gaps, larger positive comprehensiveness effects, and higher Jaccard indicate more faithful evidence.")


def plot_evidence_heatmap(result: dict[str, Any], path: Path, maximum_samples: int = 20) -> None:
    sample_ids = [str(row["sample_id"]) for row in result["predictions"][:maximum_samples]]
    index = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    matrix = np.zeros((len(sample_ids), 50), dtype=np.float32)
    for evidence in result["evidence"]:
        sample_id = str(evidence["sample_id"])
        if sample_id in index:
            matrix[index[sample_id], int(evidence["position"])] += float(evidence["local_importance"])
    fig, axis = plt.subplots(figsize=(12, max(4.5, len(sample_ids) * 0.30)))
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axis.set(title="Local Evidence Importance by Aligned Position", xlabel="Aligned position", ylabel="Sample", yticks=np.arange(len(sample_ids)), yticklabels=sample_ids)
    fig.colorbar(image, ax=axis, label="Leave-one-out importance")
    peak = int(np.argmax(matrix.sum(axis=0))) if matrix.size else 0
    _save(fig, path, f"Across the displayed samples, aligned position {peak} has the largest accumulated local importance.")


def plot_error_attribution(rows: list[dict[str, Any]], path: Path) -> None:
    filtered = [row for row in rows if row["samples"] >= 2]
    names = [row["group"] for row in filtered]
    x = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(8, len(names) * 1.25), 4.8))
    axis.bar(x - width / 2, [row["classification_error_rate"] for row in filtered], width, label="classification error", color=COLORS[0])
    axis.bar(x + width / 2, [row["intensity_mae"] for row in filtered], width, label="intensity MAE", color=COLORS[1])
    axis.set(title="Validation Error Attribution", xlabel="Quality group", ylabel="Error", xticks=x, xticklabels=names)
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    worst = max(filtered, key=lambda row: row["classification_error_rate"] + row["intensity_mae"]) if filtered else {"group": "none"}
    _save(fig, path, f"The highest combined error among groups with at least two samples is associated with {worst['group']}.")


def generate_visualizations(
    run_dir: Path, history: list[dict[str, Any]], validation: dict[str, Any], attachment4: dict[str, Any], attribution: list[dict[str, Any]],
) -> list[str]:
    paths = {
        "training": run_dir / "figures" / "training_history.png",
        "validation": run_dir / "figures" / "validation_performance.png",
        "modality": run_dir / "figures" / "validation_modality_contributions.png",
        "fidelity": run_dir / "figures" / "validation_explanation_fidelity.png",
        "evidence": run_dir / "figures" / "attachment4_evidence_heatmap.png",
        "errors": run_dir / "figures" / "validation_error_attribution.png",
    }
    generated: list[Path] = []
    if history:
        plot_training(history, paths["training"])
        generated.append(paths["training"])
    plot_validation_performance(validation, paths["validation"])
    plot_modality_contributions(validation, paths["modality"])
    plot_fidelity(validation, paths["fidelity"])
    plot_evidence_heatmap(attachment4, paths["evidence"])
    plot_error_attribution(attribution, paths["errors"])
    generated.extend(paths[key] for key in ("validation", "modality", "fidelity", "evidence", "errors"))
    return [str(path) for path in generated]
