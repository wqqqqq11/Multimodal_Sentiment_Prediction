from __future__ import annotations
from typing import Any
import pandas as pd

def _explanation_frames(explanations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for row in explanations:
        contribution = row["modality_contribution"]
        top = sorted(row["key_evidence"], key=lambda item: item["position_score"], reverse=True)[:5]
        summary_rows.append({
            "sample_id": row["sample_id"],
            "predicted_polarity": row["predicted_polarity"],
            "predicted_label": row["predicted_label"],
            "predicted_intensity": row["predicted_intensity"],
            "prediction_confidence": row["prediction_confidence"],
            "main_modality": row["main_modality"],
            "text_contribution": contribution["text"],
            "audio_contribution": contribution["audio"],
            "vision_contribution": contribution["vision"],
            "comprehensiveness": row["comprehensiveness"],
            "sufficiency": row["sufficiency"],
            "stability": row["stability"],
            "attention_ig_agreement": row["attention_ig_agreement"],
            "key_evidence": " | ".join(
                f"{item['modality']}@{item['position']}:{item['text_span']}"
                for item in top
            ),
        })
        evidence_rows.extend(row["key_evidence"])
    return pd.DataFrame(summary_rows), pd.DataFrame(evidence_rows)


def _error_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("true_polarity"):
        rows.append({
            "group": f"class_{int(label)}",
            "samples": len(group),
            "classification_accuracy": float(group["classification_correct"].mean()),
            "mean_absolute_error": float(group["absolute_error"].mean()),
            "mean_confidence": float(group["prediction_confidence"].mean()),
        })
    hardest = frame.sort_values(["classification_correct", "absolute_error"], ascending=[True, False]).head(30)
    for _, row in hardest.iterrows():
        rows.append({
            "group": "hard_sample",
            "sample_id": row["sample_id"],
            "true_polarity": row["true_polarity"],
            "predicted_polarity": row["predicted_polarity"],
            "true_intensity": row["true_intensity"],
            "predicted_intensity": row["predicted_intensity"],
            "classification_accuracy": float(row["classification_correct"]),
            "mean_absolute_error": float(row["absolute_error"]),
            "mean_confidence": float(row["prediction_confidence"]),
        })
    return pd.DataFrame(rows)
