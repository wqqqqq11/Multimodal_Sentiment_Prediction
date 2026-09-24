"""Deterministic WordPiece-to-source-text mapping for Problem 3 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextMappingResult:
    rows: list[dict[str, Any]]
    exact_match: np.ndarray
    truncation_flag: np.ndarray
    covered_char_end: np.ndarray
    confidence: np.ndarray


def load_tokenizer(config: dict[str, Any]):
    """Load the pinned fast tokenizer without silently falling back to the network."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise ImportError("Problem 3 text mapping requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["name"]),
        revision=str(config.get("revision", "main")),
        use_fast=True,
        local_files_only=bool(config.get("local_files_only", True)),
    )
    if not tokenizer.is_fast:
        raise ValueError("Problem 3 requires a fast tokenizer with offset mappings")
    return tokenizer


def build_token_mappings(
    *,
    tokenizer,
    split: str,
    sample_ids: np.ndarray,
    raw_text: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    max_length: int,
) -> TextMappingResult:
    """Map organizer token positions to original character spans and audit token equality."""
    ids = np.asarray(sample_ids).astype(str)
    texts = np.asarray(raw_text).astype(str)
    organizer_ids = np.asarray(input_ids, dtype=np.int64)
    organizer_attention = np.asarray(attention_mask, dtype=bool)
    if organizer_ids.shape != organizer_attention.shape or organizer_ids.shape[0] != len(ids):
        raise ValueError("Token arrays and sample IDs do not share a batch dimension")
    encoded = tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=int(max_length),
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    generated_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    generated_attention = np.asarray(encoded["attention_mask"], dtype=bool)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int32)
    if generated_ids.shape != organizer_ids.shape:
        raise ValueError(f"Tokenizer generated {generated_ids.shape}, expected {organizer_ids.shape}")
    exact = np.all(generated_ids == organizer_ids, axis=1) & np.all(
        generated_attention == organizer_attention, axis=1
    )
    full_lengths = np.asarray(
        [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts],
        dtype=np.int32,
    )
    truncated = full_lengths > int(max_length)
    confidence = np.where(exact, 1.0, 0.25).astype(np.float32)
    covered_end = np.zeros(len(ids), dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(ids):
        token_strings = tokenizer.convert_ids_to_tokens(organizer_ids[sample_index].tolist())
        for position in np.flatnonzero(organizer_attention[sample_index]):
            start, end = (int(value) for value in offsets[sample_index, position])
            special = start == end
            if not special:
                covered_end[sample_index] = max(covered_end[sample_index], end)
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "position": int(position),
                    "token_id": int(organizer_ids[sample_index, position]),
                    "token_text": str(token_strings[position]),
                    "char_start": start,
                    "char_end": end,
                    "text_span": texts[sample_index][start:end] if end > start else "",
                    "is_special": bool(special),
                    "tokenizer_exact_match": bool(exact[sample_index]),
                    "truncated": bool(truncated[sample_index]),
                    "mapping_confidence": float(confidence[sample_index]),
                }
            )
    return TextMappingResult(
        rows=rows,
        exact_match=exact,
        truncation_flag=truncated,
        covered_char_end=covered_end,
        confidence=confidence,
    )


def intensity_bin(values: np.ndarray) -> np.ndarray:
    """Map continuous labels to seven ordered bins with zero in its own bin."""
    labels = np.asarray(values, dtype=np.float32)
    result = np.empty(labels.shape, dtype=np.int8)
    result[labels < -2.0] = 0
    result[(labels >= -2.0) & (labels < -1.0)] = 1
    result[(labels >= -1.0) & (labels < 0.0)] = 2
    result[labels == 0.0] = 3
    result[(labels > 0.0) & (labels <= 1.0)] = 4
    result[(labels > 1.0) & (labels <= 2.0)] = 5
    result[labels > 2.0] = 6
    return result
