from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import all_metrics, apply_calibration
from .utils import move_to_device


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    goals: dict[str, float] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    sample_ids: list[str] = []
    raw_texts: list[str] = []
    logits: list[np.ndarray] = []
    regression: list[np.ndarray] = []
    log_variance: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    attentions = {name: [] for name in ("text", "audio", "vision")}
    labels_cls: list[np.ndarray] = []
    labels_reg: list[np.ndarray] = []
    for source in loader:
        sample_ids.extend(list(source["sample_id"]))
        raw_texts.extend(list(source["raw_text"]))
        batch = move_to_device(source, device)
        outputs = model(batch)
        logits.append(outputs["logits"].float().cpu().numpy())
        regression.append(outputs["regression"].float().cpu().numpy())
        log_variance.append(outputs["log_variance"].float().cpu().numpy())
        gates.append(outputs["modality_gate"].float().cpu().numpy())
        for modality in attentions:
            attentions[modality].append(outputs[f"time_attention_{modality}"].float().cpu().numpy())
        if "classification_labels" in batch:
            labels_cls.append(batch["classification_labels"].cpu().numpy())
            labels_reg.append(batch["regression_labels"].cpu().numpy())
    raw_logits = np.concatenate(logits)
    raw_regression = np.concatenate(regression)
    probabilities, prediction_class, prediction_regression = apply_calibration(raw_logits, raw_regression, calibration)
    result: dict[str, Any] = {
        "sample_id": np.asarray(sample_ids),
        "raw_text": np.asarray(raw_texts),
        "logits": raw_logits,
        "probabilities": probabilities,
        "prediction_class": prediction_class,
        "prediction_regression": prediction_regression,
        "prediction_regression_raw": raw_regression,
        "regression_uncertainty": np.exp(0.5 * np.concatenate(log_variance)),
        "modality_gate": np.concatenate(gates),
        **{f"time_attention_{name}": np.concatenate(values) for name, values in attentions.items()},
    }
    if labels_cls:
        result["label_class"] = np.concatenate(labels_cls)
        result["label_regression"] = np.concatenate(labels_reg)
        result["metrics"] = all_metrics(
            result["label_class"], prediction_class,
            result["label_regression"], prediction_regression,
            goals,
        )
    return result


def prediction_frame(result: dict[str, Any], include_labels: bool = True) -> pd.DataFrame:
    names = ("negative", "neutral", "positive")
    frame = pd.DataFrame({
        "sample_id": result["sample_id"],
        "raw_text": result["raw_text"],
        "predicted_polarity": result["prediction_class"].astype(int),
        "predicted_label": [names[index] for index in result["prediction_class"].astype(int)],
        "predicted_intensity": result["prediction_regression"],
        "prob_negative": result["probabilities"][:, 0],
        "prob_neutral": result["probabilities"][:, 1],
        "prob_positive": result["probabilities"][:, 2],
        "prediction_confidence": result["probabilities"].max(axis=1),
        "regression_uncertainty": result["regression_uncertainty"],
        "gate_text": result["modality_gate"][:, 0],
        "gate_audio": result["modality_gate"][:, 1],
        "gate_vision": result["modality_gate"][:, 2],
    })
    if include_labels and "label_class" in result:
        frame["true_polarity"] = result["label_class"].astype(int)
        frame["true_intensity"] = result["label_regression"]
        frame["classification_correct"] = frame["predicted_polarity"] == frame["true_polarity"]
        frame["absolute_error"] = np.abs(frame["predicted_intensity"] - frame["true_intensity"])
    return frame
