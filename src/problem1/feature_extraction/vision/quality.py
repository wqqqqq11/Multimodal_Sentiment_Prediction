"""Per-frame visual evidence quality."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any


def visual_quality(image: np.ndarray, box: tuple[int, int, int, int], detected: bool,
                   config: dict[str, Any]) -> float:
    x, y, w, h = box
    roi = image[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean()) / 255.0
    exposure = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
    sharpness = min(
        float(cv2.Laplacian(gray, cv2.CV_64F).var()) /
        float(config["quality_sharpness_reference"]),
        1.0,
    )
    base = float(config["face_quality_base"] if detected else config["scene_quality_base"])
    value = (base + float(config["quality_exposure_weight"]) * exposure +
             float(config["quality_sharpness_weight"]) * sharpness)
    return float(np.clip(value, 0.0, 1.0))
