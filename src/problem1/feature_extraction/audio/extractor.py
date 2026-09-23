"""WavLM contextual speech representations with exact temporal metadata."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ...common.device import resolve_torch_device
from ...schemas import FeatureSequence

_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _cache_dir(config: dict[str, Any]) -> str:
    path = Path(str(config["cache_dir"]))
    if not path.is_absolute():
        path = Path(str(config["project_root"])) / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_encoder(config: dict[str, Any]) -> tuple[Any, Any, str]:
    device = resolve_torch_device(str(config.get("device", "auto")))
    model_name, revision = str(config["model_name"]), str(config["revision"])
    if config.get("use_safetensors") is not True:
        raise RuntimeError("audio.use_safetensors必须为true，禁止加载不安全的.bin权重")
    cache_dir = _cache_dir(config)
    key = (model_name, revision, device, cache_dir)
    with _CACHE_LOCK:
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError(f"无法导入transformers AutoFeatureExtractor/AutoModel: {exc}") from exc
        if key not in _CACHE:
            processor = AutoFeatureExtractor.from_pretrained(
                model_name, revision=revision, cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
            )
            model = AutoModel.from_pretrained(
                model_name, revision=revision, use_safetensors=True, cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
            ).eval().to(device)
            _CACHE[key] = (processor, model)
    processor, model = _CACHE[key]
    return processor, model, device


def _read_audio(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("缺少soundfile/scipy；请安装根目录requirements.txt") from exc
    signal, rate = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        raise ValueError(f"音频为空: {path}")
    if rate != target_rate:
        divisor = math.gcd(int(rate), int(target_rate))
        signal = resample_poly(signal, target_rate // divisor, int(rate) // divisor).astype(np.float32)
        rate = target_rate
    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        signal = signal / peak
    return np.asarray(signal, dtype=np.float32), int(rate)


def _step_audio_diagnostics(signal: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                            config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate signal quality and speech activity without discarding timeline positions."""
    energy_quality, levels = [], []
    if len(starts) != len(ends):
        raise ValueError("音频诊断区间起止长度不一致")
    for left, right in zip(starts, ends):
        segment = signal[int(left):max(int(left) + 1, int(right))]
        rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
        db = 20.0 * np.log10(rms + 1e-8)
        clipping = float(np.mean(np.abs(segment) >= 0.99))
        levels.append(db)
        energy_quality.append(
            np.clip(0.10 + 0.90 * np.clip((db + 60.0) / 45.0, 0.0, 1.0)
                    * (1.0 - clipping), 0.05, 1.0)
        )
    quality = np.asarray(energy_quality, dtype=np.float64)
    db_values = np.asarray(levels, dtype=np.float64)
    if not bool(config.get("vad_enabled", False)):
        activity = np.ones_like(db_values)
        return quality.astype(np.float32), activity.astype(np.float32), np.ones(len(activity), dtype=np.bool_)

    noise = float(np.percentile(db_values, float(config.get("vad_noise_percentile", 20.0))))
    speech_level = float(np.percentile(db_values, 95.0))
    margin = float(config.get("vad_margin_db", 3.0))
    denominator = max(speech_level - noise - margin, 6.0)
    relative_activity = np.clip((db_values - noise - margin) / denominator, 0.0, 1.0)
    # The absolute term keeps continuously voiced clips from being mistaken for noise
    # when their within-clip dynamic range is small.
    absolute_activity = np.clip((db_values + 55.0) / 35.0, 0.0, 1.0)
    activity = np.maximum(relative_activity, 0.5 * absolute_activity)
    threshold = float(config.get("vad_threshold", 0.15))
    valid = activity >= threshold
    if not np.any(valid):
        valid[int(np.argmax(db_values))] = True
    quality = np.clip(quality * (0.25 + 0.75 * activity), 0.01, 1.0)
    return quality.astype(np.float32), activity.astype(np.float32), valid.astype(np.bool_)


def _quality_per_step(signal: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Backward-compatible signal-quality helper used by primitive tests."""
    quality, _, _ = _step_audio_diagnostics(
        signal, np.asarray(edges[:-1]), np.asarray(edges[1:]), {"vad_enabled": False}
    )
    return quality


def _wavlm_time_intervals(output_steps: int, input_samples: int, sample_rate: int,
                          kernels: list[int], strides: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map encoder outputs to exact convolutional receptive-field intervals."""
    if output_steps < 1 or input_samples < 1 or len(kernels) != len(strides) or not kernels:
        raise ValueError("WavLM时间映射参数非法")
    receptive_field, jump = 1, 1
    for kernel, stride in zip(kernels, strides):
        if int(kernel) < 1 or int(stride) < 1:
            raise ValueError("WavLM卷积kernel/stride必须为正整数")
        receptive_field += (int(kernel) - 1) * jump
        jump *= int(stride)
    starts_sample = np.arange(output_steps, dtype=np.float64) * jump
    ends_sample = starts_sample + receptive_field
    if starts_sample[-1] >= input_samples:
        raise ValueError("WavLM输出长度与卷积时间映射不一致")
    starts = starts_sample / sample_rate
    ends = np.minimum(ends_sample, float(input_samples)) / sample_rate
    centers = 0.5 * (starts + ends)
    duration = input_samples / sample_rate
    positions = np.clip(centers / max(duration, 1e-8), 0.0, 1.0)
    return starts.astype(np.float32), ends.astype(np.float32), positions.astype(np.float32)


def _pool(sequence: FeatureSequence, maximum: int) -> FeatureSequence:
    length = sequence.features.shape[0]
    if length <= maximum:
        return sequence
    edges = np.linspace(0, length, maximum + 1).astype(int)
    matrix, positions, starts, ends, quality, valid, indices = [], [], [], [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        right = max(left + 1, right)
        matrix.append(sequence.features[left:right].mean(axis=0))
        positions.append(sequence.positions[left:right].mean())
        starts.append(sequence.start[left]); ends.append(sequence.end[right - 1])
        quality.append(sequence.quality[left:right].mean())
        valid.append(bool(sequence.valid_mask[left:right].any()))
        indices.append(int(np.round(sequence.source_index[left:right].mean())))
    sequence.features = np.asarray(matrix, dtype=np.float32)
    sequence.positions = np.asarray(positions, dtype=np.float32)
    sequence.start = np.asarray(starts, dtype=np.float32)
    sequence.end = np.asarray(ends, dtype=np.float32)
    sequence.quality = np.asarray(quality, dtype=np.float32)
    sequence.source_index = np.asarray(indices, dtype=np.int32)
    sequence.valid_mask = np.asarray(valid, dtype=np.bool_)
    sequence.metadata["original_steps"] = length
    return sequence


def extract_audio(sample_id: str, path: Path, config: dict[str, Any]) -> FeatureSequence:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少PyTorch；请安装根目录requirements.txt") from exc
    target_rate = int(config["sample_rate"])
    signal, rate = _read_audio(path, target_rate)
    duration = len(signal) / rate
    if duration > float(config["max_seconds"]):
        raise ValueError(
            f"{sample_id}: 音频{duration:.2f}s超过audio.max_seconds={config['max_seconds']}，"
            "为避免静默截断，请增大配置值"
        )
    processor, model, device = _load_encoder(config)
    inputs = processor(signal, sampling_rate=rate, return_tensors="pt", padding=False)
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    use_amp = bool(config.get("mixed_precision", True)) and device == "cuda"
    with _INFERENCE_LOCK, torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=use_amp
    ):
        hidden = model(**model_inputs).last_hidden_state[0].float().cpu().numpy()
    if hidden.shape[0] == 0:
        raise RuntimeError(f"{sample_id}: WavLM未输出任何时间步")
    kernels = [int(value) for value in getattr(model.config, "conv_kernel", [])]
    strides = [int(value) for value in getattr(model.config, "conv_stride", [])]
    starts, ends, positions = _wavlm_time_intervals(
        hidden.shape[0], len(signal), rate, kernels, strides
    )
    sample_starts = np.rint(starts.astype(np.float64) * rate).astype(int)
    sample_ends = np.rint(ends.astype(np.float64) * rate).astype(int)
    quality, activity, valid = _step_audio_diagnostics(signal, sample_starts, sample_ends, config)
    sequence = FeatureSequence(
        sample_id=sample_id, modality="audio", features=np.asarray(hidden, dtype=np.float32),
        positions=positions, start=starts, end=ends,
        quality=quality, valid_mask=valid,
        source_index=np.arange(hidden.shape[0], dtype=np.int32),
        metadata={"backend": "wavlm", "model_name": str(config["model_name"]),
                  "revision": str(config["revision"]), "weight_format": "safetensors", "sample_rate": rate,
                  "duration_sec": duration, "embedding_dimension": int(hidden.shape[1]), "unit": "second",
                  "timestamp_method": "convolution_receptive_field", "conv_kernel": kernels,
                  "conv_stride": strides, "vad_method": "adaptive_energy",
                  "speech_activity_ratio": float(valid.mean()),
                  "mean_speech_probability": float(activity.mean())},
    )
    sequence = _pool(sequence, int(config["max_steps"]))
    sequence.validate()
    return sequence
