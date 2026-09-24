"""Video metadata, proportional time fallback, and unaligned-vision repair helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        readable = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if readable else 0
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) if readable else 0
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) if readable else 0
    finally:
        capture.release()
    duration = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return {
        "readable": readable,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def proportional_time_mapping(
    *,
    sample_id: str,
    token_rows: list[dict[str, Any]],
    content_mask: np.ndarray,
    audio_candidate: np.ndarray,
    vision_candidate: np.ndarray,
    metadata: dict[str, Any],
    confidence: float,
) -> list[dict[str, Any]]:
    """Build an explicitly low-confidence token/second/frame fallback mapping."""
    duration = float(metadata["duration_sec"])
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    rows_by_position = {int(row["position"]): row for row in token_rows}
    content_positions = np.flatnonzero(np.asarray(content_mask, dtype=bool))
    covered_end = max(
        (int(rows_by_position[pos]["char_end"]) for pos in content_positions if pos in rows_by_position),
        default=0,
    )
    result: list[dict[str, Any]] = []
    for ordinal, position in enumerate(content_positions):
        token = rows_by_position.get(int(position), {})
        char_start = int(token.get("char_start", 0))
        char_end = int(token.get("char_end", 0))
        if covered_end > 0 and char_end > char_start:
            start = duration * char_start / covered_end
            end = duration * char_end / covered_end
        else:
            denominator = max(len(content_positions), 1)
            start = duration * ordinal / denominator
            end = duration * (ordinal + 1) / denominator
        start = max(0.0, min(float(start), duration))
        end = max(start, min(float(end), duration))
        if frame_count > 0 and fps > 0:
            start_frame = min(frame_count - 1, max(0, int(math.floor(start * fps))))
            end_frame = min(frame_count - 1, max(start_frame, int(math.ceil(end * fps)) - 1))
            representative = int(round((start_frame + end_frame) / 2))
        else:
            start_frame = end_frame = representative = -1
        result.append(
            {
                "sample_id": str(sample_id),
                "position": int(position),
                "token_id": int(token.get("token_id", -1)),
                "token_text": str(token.get("token_text", "")),
                "text_span": str(token.get("text_span", "")),
                "char_start": char_start,
                "char_end": char_end,
                "start_sec": start,
                "end_sec": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "representative_frame": representative,
                "mapping_method": "token_offset_proportional_fallback",
                "mapping_confidence": float(confidence),
                "text_candidate": True,
                "audio_candidate": bool(audio_candidate[position]),
                "vision_candidate": bool(vision_candidate[position]),
            }
        )
    return result


def inferred_nonzero_length(features: np.ndarray, eps: float) -> int:
    array = np.asarray(features)
    nonzero = np.any(np.isfinite(array) & (np.abs(array) > eps), axis=-1)
    positions = np.flatnonzero(nonzero)
    return int(positions[-1] + 1) if positions.size else 0


def reconstruct_to_content(
    unaligned_features: np.ndarray,
    content_mask: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Linearly resample the nonzero unaligned prefix onto aligned content slots."""
    source = np.asarray(unaligned_features, dtype=np.float64)
    content = np.asarray(content_mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("unaligned features must have shape (T,D)")
    output = np.zeros((content.size, source.shape[1]), dtype=np.float32)
    length = inferred_nonzero_length(source, eps)
    targets = np.flatnonzero(content)
    if length == 0 or targets.size == 0:
        return output, length
    valid = source[:length]
    if length == 1:
        values = np.repeat(valid, targets.size, axis=0)
    else:
        source_axis = np.linspace(0.0, 1.0, length)
        target_axis = np.linspace(0.0, 1.0, targets.size)
        values = np.column_stack(
            [np.interp(target_axis, source_axis, valid[:, dimension]) for dimension in range(valid.shape[1])]
        )
    output[targets] = values.astype(np.float32)
    return output, length


def reconstruction_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return {"median_cosine": 0.0, "mse": float("nan"), "positions": 0}
    ref = reference[selected]
    cand = candidate[selected]
    denominator = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    cosine = np.divide(
        np.sum(ref * cand, axis=1), denominator,
        out=np.zeros(ref.shape[0], dtype=np.float64), where=denominator > 1e-12,
    )
    return {
        "median_cosine": float(np.median(cosine)),
        "mse": float(np.mean((ref - cand) ** 2)),
        "positions": int(selected.sum()),
    }
