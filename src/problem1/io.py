"""Atomic I/O helpers for problem 1 artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import FeatureSequence

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_id(value: str) -> str:
    return _INVALID.sub("_", value).strip().rstrip(".") or "unnamed"


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(name, **arrays)
        Path(name).replace(path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def save_feature(path: Path, seq: FeatureSequence, compressed: bool = True) -> None:
    seq.validate()
    atomic_npz(
        path,
        compressed=compressed,
        sample_id=np.asarray(seq.sample_id), modality=np.asarray(seq.modality),
        features=np.asarray(seq.features, dtype=np.float32),
        positions=np.asarray(seq.positions, dtype=np.float32),
        start=np.asarray(seq.start, dtype=np.float32), end=np.asarray(seq.end, dtype=np.float32),
        quality=np.asarray(seq.quality, dtype=np.float32),
        valid_mask=np.asarray(seq.valid_mask, dtype=np.bool_),
        source_index=np.asarray(seq.source_index, dtype=np.int32),
        metadata=np.asarray(json.dumps(seq.metadata, ensure_ascii=False)),
    )


def load_feature(path: Path) -> FeatureSequence:
    with np.load(path, allow_pickle=False) as data:
        seq = FeatureSequence(
            sample_id=str(data["sample_id"]), modality=str(data["modality"]),
            features=data["features"], positions=data["positions"], start=data["start"],
            end=data["end"], quality=data["quality"], valid_mask=data["valid_mask"],
            source_index=data["source_index"], metadata=json.loads(str(data["metadata"])),
        )
    seq.validate()
    return seq
