"""MediaPipe expression/geometry and ConvNeXt face appearance extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...schemas import FeatureSequence
from .face_detection import MediaPipeFaceLandmarker
from .face_features import DeepFaceAppearanceEncoder, face_bbox, full_feature_dimension, structured_face_features
from .quality import visual_quality


def extract_vision(sample_id: str, root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> FeatureSequence:
    if not rows:
        raise ValueError(f"{sample_id}: 没有视觉时间轴记录")
    ordered = sorted(rows, key=lambda row: int(row["timeline_index"]))
    appearance = DeepFaceAppearanceEncoder(config)
    dimension = full_feature_dimension(appearance.dimension)
    features, quality, valid, face_detected, boxes, appearance_sources = [], [], [], [], [], []
    blendshape_names: list[str] | None = None
    previous_timestamp = -1
    with MediaPipeFaceLandmarker(config) as landmarker:
        for row in ordered:
            path = root / str(row["frame_relative_path"])
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取帧: {path}")
            timestamp = max(previous_timestamp + 1, int(round(float(row["center_sec"]) * 1000.0)))
            previous_timestamp = timestamp
            result = landmarker.detect(image, timestamp)
            if result is None:
                # A frame without a detectable face is still genuine visual
                # evidence (scene, body, objects, captions). Encode the complete
                # frame and leave only the face-specific block at zero.
                appearance_vector = appearance.encode(image)
                appearance_vector /= max(float(np.linalg.norm(appearance_vector)), 1e-8)
                structured = np.zeros(dimension - appearance.dimension, dtype=np.float32)
                features.append(np.concatenate((appearance_vector, structured)).astype(np.float32))
                height, width = image.shape[:2]
                quality.append(visual_quality(image, (0, 0, width, height), False, config))
                valid.append(True)
                face_detected.append(False)
                boxes.append(None)
                appearance_sources.append("full_frame")
                continue
            bbox = face_bbox(result.face_landmarks[0], image.shape, float(config["face_padding_ratio"]))
            x, y, width, height = bbox
            face = image[y:y + height, x:x + width]
            appearance_vector = appearance.encode(face)
            appearance_vector /= max(float(np.linalg.norm(appearance_vector)), 1e-8)
            structured, names = structured_face_features(result, bbox, image.shape)
            if blendshape_names is None:
                blendshape_names = names
            elif names != blendshape_names:
                raise RuntimeError(f"{sample_id}: MediaPipe Blendshape字段顺序在帧间发生变化")
            vector = np.concatenate((appearance_vector, structured)).astype(np.float32)
            if vector.size != dimension:
                raise RuntimeError(f"{sample_id}: 视觉特征维数错误，期望{dimension}，实际{vector.size}")
            features.append(vector)
            quality.append(visual_quality(image, bbox, True, config))
            valid.append(True)
            face_detected.append(True)
            boxes.append(list(bbox))
            appearance_sources.append("face_crop")
    starts = np.asarray([float(row["start_sec"]) for row in ordered], dtype=np.float32)
    ends = np.asarray([float(row["end_sec"]) for row in ordered], dtype=np.float32)
    duration = max(float(ends.max()), 1e-8)
    valid_array = np.asarray(valid, dtype=np.bool_)
    face_detected_array = np.asarray(face_detected, dtype=np.bool_)
    sequence = FeatureSequence(
        sample_id=sample_id, modality="vision", features=np.asarray(features, dtype=np.float32),
        positions=((starts + ends) / 2 / duration).astype(np.float32), start=starts, end=ends,
        quality=np.asarray(quality, dtype=np.float32), valid_mask=valid_array,
        source_index=np.asarray([int(row["source_frame_index"]) for row in ordered], dtype=np.int32),
        metadata={
            "backend": "mediapipe_timm", "landmarker": "MediaPipe Face Landmarker",
            "appearance_model_name": str(config["appearance_model_name"]),
            "appearance_dimension": appearance.dimension, "feature_dimension": dimension,
            "duration_sec": duration, "face_detected": face_detected_array.tolist(), "face_boxes": boxes,
            "detection_rate": float(face_detected_array.mean()), "blendshape_names": blendshape_names or [],
            "appearance_sources": appearance_sources, "no_face_policy": str(config["no_face_policy"]),
            "scene_encoded_rate": float((~face_detected_array).mean()),
            "unit": "second", "fallback_used": False,
        },
    )
    sequence.validate()
    return sequence
