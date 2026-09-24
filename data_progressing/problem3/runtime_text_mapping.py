"""Runtime-safe token mapping with an explicit offline character fallback.

This module exists so a missing local Hugging Face vocabulary never changes organizer
input IDs or forces an untracked network download.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_progressing.problem3.text_mapping import TextMappingResult


def load_tokenizer(config: dict[str, Any]):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(config["name"]),
            revision=str(config.get("revision", "main")),
            use_fast=True,
            local_files_only=bool(config.get("local_files_only", True)),
        )
        return tokenizer if tokenizer.is_fast else None
    except (ImportError, OSError, TypeError, ValueError):
        return None


def build_token_mappings(
    *, tokenizer: Any, split: str, sample_ids: np.ndarray, raw_text: np.ndarray,
    input_ids: np.ndarray, attention_mask: np.ndarray, max_length: int,
) -> TextMappingResult:
    ids = np.asarray(sample_ids).astype(str)
    texts = np.asarray(raw_text).astype(str)
    organizer_ids = np.asarray(input_ids, dtype=np.int64)
    organizer_attention = np.asarray(attention_mask, dtype=bool)
    if tokenizer is not None:
        from data_progressing.problem3.text_mapping import build_token_mappings as exact_mapping
        return exact_mapping(
            tokenizer=tokenizer, split=split, sample_ids=ids, raw_text=texts,
            input_ids=organizer_ids, attention_mask=organizer_attention,
            max_length=max_length,
        )
    exact = np.zeros(len(ids), dtype=bool)
    truncated = organizer_attention.sum(axis=1) >= int(max_length)
    covered_end = np.asarray([len(text) for text in texts], dtype=np.int32)
    confidence = np.full(len(ids), 0.35, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(ids):
        positions = np.flatnonzero(organizer_attention[sample_index])
        content = positions[1:-1] if positions.size >= 2 else np.asarray([], dtype=int)
        ordinal_by_position = {int(position): ordinal for ordinal, position in enumerate(content)}
        text = texts[sample_index]
        for position in positions:
            special = int(position) not in ordinal_by_position
            if special or content.size == 0:
                start = end = 0
            else:
                ordinal = ordinal_by_position[int(position)]
                start = int(round(len(text) * ordinal / len(content)))
                end = int(round(len(text) * (ordinal + 1) / len(content)))
            token_id = int(organizer_ids[sample_index, position])
            rows.append({
                "split": split, "sample_id": sample_id, "position": int(position),
                "token_id": token_id, "token_text": f"[id={token_id}]",
                "char_start": start, "char_end": end,
                "text_span": text[start:end] if end > start else "",
                "is_special": bool(special), "tokenizer_exact_match": False,
                "truncated": bool(truncated[sample_index]),
                "mapping_confidence": float(confidence[sample_index]),
            })
    return TextMappingResult(
        rows=rows, exact_match=exact, truncation_flag=truncated,
        covered_char_end=covered_end, confidence=confidence,
    )

