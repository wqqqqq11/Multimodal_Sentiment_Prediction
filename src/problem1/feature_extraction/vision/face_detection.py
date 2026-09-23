"""Strict MediaPipe Face Landmarker wrapper; no synthetic face fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MediaPipeFaceLandmarker:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("缺少mediapipe；请安装根目录requirements.txt") from exc
        model_path = Path(str(config["landmarker_model_path"]))
        if not model_path.is_absolute():
            model_path = Path(str(config["project_root"])) / model_path
        if not model_path.is_file():
            raise FileNotFoundError(
                f"缺少MediaPipe Face Landmarker模型: {model_path}；"
                "请按README下载face_landmarker.task，不再启用中心ROI降级"
            )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=int(config["num_faces"]),
            min_face_detection_confidence=float(config["min_face_detection_confidence"]),
            min_face_presence_confidence=float(config["min_face_presence_confidence"]),
            min_tracking_confidence=float(config["min_tracking_confidence"]),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr_image: np.ndarray, timestamp_ms: int) -> Any | None:
        import cv2
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.face_landmarks:
            return None
        return result

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeFaceLandmarker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
