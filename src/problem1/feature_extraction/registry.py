"""Explicit baseline backend registry."""

from __future__ import annotations

from .audio.extractor import extract_audio
from .text.extractor import extract_text
from .vision.extractor import extract_vision

EXTRACTORS = {
    ("text", "deberta_v3"): extract_text,
    ("audio", "wavlm"): extract_audio,
    ("vision", "mediapipe_timm"): extract_vision,
}


def get_extractor(modality: str, backend: str):
    try:
        return EXTRACTORS[(modality, backend)]
    except KeyError as exc:
        raise ValueError(f"未注册的特征后端: {modality}/{backend}") from exc
