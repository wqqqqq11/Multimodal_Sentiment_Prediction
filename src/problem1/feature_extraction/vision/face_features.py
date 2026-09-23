"""ConvNeXt face appearance plus MediaPipe geometry/blendshape descriptors."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from ...common.device import resolve_torch_device

_CRITICAL_LANDMARKS = (1, 13, 14, 33, 61, 70, 152, 263, 291, 300)
_CACHE: dict[tuple[str, str, str], tuple[Any, Any, int]] = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _cache_dir(config: dict[str, Any]) -> str:
    path = Path(str(config["cache_dir"]))
    if not path.is_absolute():
        path = Path(str(config["project_root"])) / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class DeepFaceAppearanceEncoder:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise RuntimeError("缺少timm/torchvision/Pillow；请安装根目录requirements.txt") from exc
        self.device = resolve_torch_device(str(config.get("device", "auto")))
        name, cache_dir = str(config["appearance_model_name"]), _cache_dir(config)
        key = (name, self.device, cache_dir)
        with _CACHE_LOCK:
            if key not in _CACHE:
                model = timm.create_model(name, pretrained=True, num_classes=0, cache_dir=cache_dir).eval().to(self.device)
                transform = create_transform(**resolve_model_data_config(model), is_training=False)
                dimension = int(getattr(model, "num_features"))
                _CACHE[key] = (model, transform, dimension)
        self.model, self.transform, self.dimension = _CACHE[key]
        self.mixed_precision = bool(config.get("mixed_precision", True)) and self.device == "cuda"

    def encode(self, bgr_face: np.ndarray) -> np.ndarray:
        try:
            import torch
            import cv2
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("缺少torch/Pillow/OpenCV；请安装根目录requirements.txt") from exc
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with _INFERENCE_LOCK, torch.inference_mode(), torch.autocast(
            device_type=self.device, dtype=torch.float16, enabled=self.mixed_precision
        ):
            value = self.model(tensor).float().cpu().numpy()[0]
        return np.asarray(value, dtype=np.float32)


def face_bbox(landmarks: list[Any], image_shape: tuple[int, ...], padding_ratio: float) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    xs = np.asarray([point.x for point in landmarks], dtype=float) * width
    ys = np.asarray([point.y for point in landmarks], dtype=float) * height
    x1, x2, y1, y2 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    padding = max(x2 - x1, y2 - y1) * padding_ratio
    x1, y1 = max(0, int(np.floor(x1 - padding))), max(0, int(np.floor(y1 - padding)))
    x2, y2 = min(width, int(np.ceil(x2 + padding))), min(height, int(np.ceil(y2 + padding)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("MediaPipe返回了空人脸边界框")
    return x1, y1, x2 - x1, y2 - y1


def structured_face_features(result: Any, bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    landmarks = result.face_landmarks[0]
    x, y, width, height = bbox
    image_height, image_width = image_shape[:2]
    landmark_values = []
    for index in _CRITICAL_LANDMARKS:
        point = landmarks[index]
        landmark_values.extend(((point.x * image_width - x) / max(width, 1),
                                (point.y * image_height - y) / max(height, 1),
                                point.z * image_width / max(width, 1)))
    categories = sorted(result.face_blendshapes[0], key=lambda item: item.category_name)
    if len(categories) != 52:
        raise RuntimeError(f"MediaPipe应输出52维Blendshape，实际={len(categories)}")
    blendshape_names = [str(item.category_name) for item in categories]
    blendshapes = [float(item.score) for item in categories]
    if result.facial_transformation_matrixes:
        pose = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(-1)
    else:
        raise RuntimeError("MediaPipe未输出面部变换矩阵，请检查output_facial_transformation_matrixes")
    if pose.size != 16:
        raise RuntimeError(f"面部变换矩阵应为4×4，实际元素数={pose.size}")
    geometry = [
        (x + width / 2) / image_width, (y + height / 2) / image_height,
        width / image_width, height / image_height, width / max(height, 1),
    ]
    vector = np.asarray(blendshapes + landmark_values + pose.tolist() + geometry, dtype=np.float32)
    return vector, blendshape_names


def full_feature_dimension(appearance_dimension: int) -> int:
    return int(appearance_dimension + 52 + 3 * len(_CRITICAL_LANDMARKS) + 16 + 5)
