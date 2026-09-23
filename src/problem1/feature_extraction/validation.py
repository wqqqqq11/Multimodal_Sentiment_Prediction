"""Feature artifact integrity and temporal consistency validation."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from ..config import Problem1Config
from ..io import atomic_json, load_feature, safe_id


def validate_features(cfg: Problem1Config, *, limit: int | None = None) -> dict[str, Any]:
    with (cfg.path("preprocessed_root") / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        rows = rows[:limit]
    errors, warnings, samples = [], [], []
    expected_backends = {modality: str(cfg.section(modality)["backend"])
                         for modality in ("text", "audio", "vision")}
    for row in rows:
        sample_id = row["sample_id"]
        item: dict[str, Any] = {"sample_id": sample_id}
        sequences = {}
        for modality in ("text", "audio", "vision"):
            path = cfg.path("feature_root") / "samples" / safe_id(sample_id) / f"{modality}.npz"
            try:
                sequences[modality] = load_feature(path)
                item[f"{modality}_shape"] = list(sequences[modality].features.shape)
                item[f"{modality}_valid_ratio"] = float(sequences[modality].valid_mask.mean())
                backend = sequences[modality].metadata.get("backend")
                if backend != expected_backends[modality]:
                    errors.append({"sample_id": sample_id, "modality": modality,
                                   "error": f"特征后端不匹配: expected={expected_backends[modality]}, actual={backend}"})
                if not bool(sequences[modality].valid_mask.any()):
                    errors.append({"sample_id": sample_id, "modality": modality,
                                   "error": "该模态没有任何有效时间步"})
            except Exception as exc:
                errors.append({"sample_id": sample_id, "modality": modality, "error": f"{type(exc).__name__}: {exc}"})
        if len(sequences) == 3:
            item["audio_visual_end_delta_sec"] = abs(float(sequences["audio"].end[-1]) - float(sequences["vision"].end[-1]))
            item["all_finite"] = all(np.isfinite(value.features).all() for value in sequences.values())
            detection_rate = float(sequences["vision"].metadata.get("detection_rate", 0.0))
            item["vision_detection_rate"] = detection_rate
            warning_threshold = float(cfg.section("vision")["min_face_detection_rate_warning"])
            if detection_rate < warning_threshold:
                warnings.append({"sample_id": sample_id, "modality": "vision",
                                 "warning": f"人脸检测率偏低: {detection_rate:.1%}"})
        samples.append(item)
    report = {"passed": not errors, "sample_count": len(rows), "error_count": len(errors),
              "warning_count": len(warnings), "errors": errors, "warnings": warnings, "samples": samples}
    atomic_json(cfg.path("feature_root") / "validation_report.json", report)
    return report
