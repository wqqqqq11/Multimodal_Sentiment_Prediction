"""序列特征的填充、有效长度与局部全零区间统计。"""

from __future__ import annotations

import numpy as np

from config import ZERO_EPS


def timestep_all_zero(array: np.ndarray, eps: float = ZERO_EPS) -> np.ndarray:
    """最后一维全接近 0 的时间步。支持 (T, D) 或 (N, T, D)。"""
    return np.all(np.abs(array) <= eps, axis=-1)


def mask_to_runs(mask: np.ndarray) -> list[dict]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [
        {"start": int(s), "end": int(e), "length": int(e - s + 1)}
        for s, e in zip(starts, ends)
    ]


def last_nonzero_length(zero_mask: np.ndarray) -> np.ndarray:
    """zero_mask: (N, T) 或 (T,)。返回最后一个非零步的下标 + 1；全零则为 0。"""
    single = zero_mask.ndim == 1
    if single:
        zero_mask = zero_mask[None, :]
    has = ~zero_mask
    index = np.arange(zero_mask.shape[1])[None, :]
    last = np.where(has, index, -1).max(axis=1)
    length = (last + 1).astype(np.int32)
    return length[0] if single else length


def sequence_zero_profile(
    array: np.ndarray,
    provided_length: int | None = None,
    structural_leading_zero: bool = False,
) -> dict:
    """分析单条 (T, D) 序列的有效区、尾部填充和内部全零区间。"""
    zero_mask = timestep_all_zero(array)
    length = int(zero_mask.shape[0])
    inferred = int(last_nonzero_length(zero_mask))
    if provided_length is None:
        valid_length = inferred
    else:
        valid_length = int(max(0, min(length, provided_length)))

    positions = np.arange(length)
    padding = positions >= valid_length
    structural = np.zeros(length, dtype=bool)
    if structural_leading_zero and length > 0 and bool(zero_mask[0]):
        structural[0] = True
    valid = ~padding
    missing = zero_mask & valid & ~structural
    runs = mask_to_runs(missing)
    valid_steps = int(valid.sum())
    return {
        "timesteps": length,
        "feature_dim": int(array.shape[-1]) if array.ndim >= 2 else 1,
        "inferred_valid_length": inferred,
        "provided_length": None if provided_length is None else int(provided_length),
        "valid_length": valid_length,
        "padding_steps": int(padding.sum()),
        "structural_leading_zero": bool(structural.any()),
        "nonzero_steps": int((~zero_mask & valid).sum()),
        "missing_steps": int(missing.sum()),
        "missing_ratio_in_valid": float(missing.sum() / valid_steps) if valid_steps else None,
        "element_zero_ratio": float(np.mean(np.abs(array) <= ZERO_EPS)),
        "runs": runs,
        "missing_mask": missing,
        "zero_mask": zero_mask,
    }


def batch_length_and_internal_zeros(
    array: np.ndarray,
    provided_lengths: np.ndarray | None = None,
    structural_leading_zero: bool = False,
) -> dict[str, np.ndarray]:
    """array: (N, T, D)。返回每条样本的长度与内部全零步数。"""
    zero_mask = timestep_all_zero(array)
    n, timesteps = zero_mask.shape
    inferred = last_nonzero_length(zero_mask)
    if provided_lengths is None:
        valid = inferred
    else:
        valid = np.clip(np.asarray(provided_lengths, dtype=np.int32), 0, timesteps)
    positions = np.arange(timesteps)[None, :]
    internal = zero_mask & (positions < valid[:, None])
    if structural_leading_zero:
        internal = internal.copy()
        internal[:, 0] = False
    nonzero = (~zero_mask) & (positions < valid[:, None])
    return {
        "inferred_valid_length": inferred,
        "valid_length": valid,
        "internal_zero_steps": internal.sum(axis=1).astype(np.int32),
        "nonzero_steps": nonzero.sum(axis=1).astype(np.int32),
        "element_zero_ratio": (np.abs(array) <= ZERO_EPS).mean(axis=(1, 2)),
    }


def _chunk_reduce(array: np.ndarray, sample_limit: int = 2_000_000) -> dict:
    """分块计算，避免在未对齐语音这种上亿元素数组上再复制一整份。"""
    flat = np.asarray(array)
    floating = np.issubdtype(flat.dtype, np.floating)
    nan_count = int(np.isnan(flat).sum()) if floating else 0
    inf_count = int(np.isinf(flat).sum()) if floating else 0
    ravel = flat.ravel()
    total = 0
    sum_ = 0.0
    sumsq = 0.0
    min_ = np.inf
    max_ = -np.inf
    zeros = 0
    # 约 8e6 个元素一块，float64 约 64MB。
    chunk = 8_000_000
    for start in range(0, ravel.size, chunk):
        part = ravel[start : start + chunk].astype(np.float64, copy=False)
        if floating and (nan_count or inf_count):
            part = part[np.isfinite(part)]
        if part.size == 0:
            continue
        total += int(part.size)
        sum_ += float(part.sum())
        sumsq += float(np.square(part).sum())
        min_ = min(min_, float(part.min()))
        max_ = max(max_, float(part.max()))
        zeros += int(np.count_nonzero(np.abs(part) <= ZERO_EPS))
    if total == 0:
        return {"count": 0, "nan_count": nan_count, "inf_count": inf_count}
    mean = sum_ / total
    variance = max(sumsq / total - mean**2, 0.0)
    rng = np.random.default_rng(0)
    if ravel.size > sample_limit:
        picked = ravel[rng.choice(ravel.size, sample_limit, replace=False)].astype(np.float64, copy=False)
        sampled = True
    else:
        picked = ravel.astype(np.float64, copy=False)
        sampled = False
    if floating:
        picked = picked[np.isfinite(picked)]
    quantiles = np.quantile(picked, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if picked.size else [None] * 7
    return {
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "count": total,
        "min": float(min_),
        "max": float(max_),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "zero_ratio": float(zeros / total),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "quantile_sampled": sampled,
        "quantile_sample_size": int(picked.size),
        "p01": None if quantiles[0] is None else float(quantiles[0]),
        "p05": None if quantiles[1] is None else float(quantiles[1]),
        "p25": None if quantiles[2] is None else float(quantiles[2]),
        "median": None if quantiles[3] is None else float(quantiles[3]),
        "p75": None if quantiles[4] is None else float(quantiles[4]),
        "p95": None if quantiles[5] is None else float(quantiles[5]),
        "p99": None if quantiles[6] is None else float(quantiles[6]),
    }


def numeric_overview(array: np.ndarray, sample_limit: int = 2_000_000) -> dict:
    """全量均值/方差/零占比，分位数在大数组上抽样。"""
    return _chunk_reduce(array, sample_limit=sample_limit)


def per_dimension_stats(array: np.ndarray, chunk_samples: int = 64) -> dict[str, np.ndarray]:
    """array (N, T, D) 或 (T, D)，在样本和时间上汇总每一维。"""
    if array.ndim == 2:
        array = array[None, ...]
    sample_count, _, dims = array.shape
    total = 0
    sum_ = np.zeros(dims, dtype=np.float64)
    sumsq = np.zeros(dims, dtype=np.float64)
    min_ = np.full(dims, np.inf)
    max_ = np.full(dims, -np.inf)
    zeros = np.zeros(dims, dtype=np.float64)
    for start in range(0, sample_count, chunk_samples):
        chunk = np.asarray(array[start : start + chunk_samples], dtype=np.float64)
        flat = chunk.reshape(-1, dims)
        sum_ += flat.sum(axis=0)
        sumsq += np.square(flat).sum(axis=0)
        min_ = np.minimum(min_, flat.min(axis=0))
        max_ = np.maximum(max_, flat.max(axis=0))
        zeros += np.count_nonzero(np.abs(flat) <= ZERO_EPS, axis=0)
        total += flat.shape[0]
    mean = sum_ / max(total, 1)
    variance = np.maximum(sumsq / max(total, 1) - mean**2, 0)
    return {
        "mean": mean,
        "std": np.sqrt(variance),
        "min": min_,
        "max": max_,
        "zero_ratio": zeros / max(total, 1),
    }


def polarity_from_regression(values, eps: float = 1e-6) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    labels = np.full(scores.shape, "neutral", dtype=object)
    labels[scores < -eps] = "negative"
    labels[scores > eps] = "positive"
    return labels


def text_lengths(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chars, words = [], []
    for text in texts:
        raw = "" if text is None else str(text)
        chars.append(len(raw))
        words.append(len(raw.split()))
    return np.asarray(chars, dtype=np.int32), np.asarray(words, dtype=np.int32)


def parse_sample_id(sample_id: str) -> tuple[str, str]:
    text = str(sample_id)
    if "$_$" in text:
        video_id, clip_id = text.split("$_$", 1)
        return video_id, clip_id
    return text, ""
