"""结果落盘与通用数值转换。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_output(name: str) -> dict[str, Path]:
    from config import OUTPUT_ROOT

    root = ensure_dir(OUTPUT_ROOT / name)
    return {
        "root": root,
        "tables": ensure_dir(root / "tables"),
        "figures": ensure_dir(root / "figures"),
    }


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2)


def save_df(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_str_array(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.array([str(v) for v in values.tolist()], dtype=object)
    return np.array([str(v) for v in values], dtype=object)


def describe_series(values) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    qs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    quantiles = np.quantile(arr, qs)
    centered = arr - arr.mean()
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    skew = m3 / (m2**1.5) if m2 > 0 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(arr.max()),
        "skewness": float(skew),
    }


def value_counts(values) -> dict:
    series = pd.Series(list(values))
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}
