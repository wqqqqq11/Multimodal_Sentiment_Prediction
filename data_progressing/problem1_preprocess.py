"""问题一：原始视频与标签的可复现预处理管线。

本模块只负责进入三模态特征提取之前的确定性处理：

1. 校验 ``label-100.xlsx`` 与 100 个 MP4 是否一一对应；
2. 生成带 SHA256、视频属性和质量标记的样本清单；
3. 在不改写语义的前提下规范化文本，并保存词元字符偏移；
4. 生成按真实时间戳定义的视觉采样时间轴；
5. 可选地抽取视觉帧、调用 FFmpeg 生成 16 kHz 单声道 WAV；
6. 保存配置和验证报告，保证后续特征提取可追溯。

原始赛题数据始终只读。BERT、openSMILE、LibreFace、MFA 等模型属于
下一阶段的特征提取/对齐工具，不在本预处理脚本中隐式下载或运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


PIPELINE_VERSION = "1.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "original_data_from_the_competition_organizer"
    / "dataset01"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "preprocessed_data" / "problem1"
REQUIRED_LABEL_COLUMNS = {"video_id", "clip_id", "text", "label", "annotation"}
ANNOTATION_TO_CLASS = {"negative": 0, "neutral": 1, "positive": 2}
WORD_OR_PUNCT = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]",
    flags=re.UNICODE,
)
WINDOWS_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class PreprocessError(RuntimeError):
    """输入结构或预处理依赖不满足要求。"""


@dataclass(frozen=True)
class PipelineConfig:
    input_root: str
    output_root: str
    label_file: str
    frame_sample_fps: float
    extract_frames: bool
    extract_audio: bool
    audio_sample_rate: int
    overwrite_media: bool
    strict: bool
    workers: int
    ffmpeg: str | None


def normalize_id(value: Any) -> str:
    """稳定地把 Excel 中的 ID 转成文件系统使用的字符串。"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(text: Any) -> str:
    """只做 Unicode 和空白规范化，不改变大小写、词义或标点。"""
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_with_offsets(text: str) -> list[dict[str, Any]]:
    """生成可回映到规范化文本的轻量词元，不替代 BERT tokenizer。"""
    tokens: list[dict[str, Any]] = []
    for index, match in enumerate(WORD_OR_PUNCT.finditer(text)):
        token = match.group(0)
        if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", token):
            kind = "word"
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            kind = "number"
        else:
            kind = "punct"
        tokens.append(
            {
                "token_index": index,
                "token": token,
                "token_kind": kind,
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )
    return tokens


def polarity_from_score(score: float) -> str:
    if score < 0:
        return "negative"
    if score > 0:
        return "positive"
    return "neutral"


def safe_filename(value: str) -> str:
    cleaned = WINDOWS_INVALID_FILENAME.sub("_", value).strip().rstrip(".")
    return cleaned or "unnamed"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary_name, index=False, encoding="utf-8-sig")
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _estimate_frame_interval(timestamps_sec: list[float], nominal_fps: float) -> float:
    """根据实际时间戳估计帧间隔，失败时回退到名义帧率。"""
    if len(timestamps_sec) >= 2:
        differences = np.diff(np.asarray(timestamps_sec, dtype=np.float64))
        positive = differences[np.isfinite(differences) & (differences > 1e-9)]
        if positive.size:
            return float(np.median(positive))
    if nominal_fps > 0 and math.isfinite(nominal_fps):
        return 1.0 / nominal_fps
    return 0.0


def _decode_frame_timestamps(
    cap: cv2.VideoCapture,
    nominal_fps: float,
) -> tuple[list[float], int]:
    """顺序解码视频流，返回每个可解码帧的时间戳和回退计数。

    部分赛题 MP4 的 ``CAP_PROP_FRAME_COUNT`` 按容器时长估算，会显著大于
    实际视频流帧数。顺序解码是后续抽帧可执行性的最终依据。
    """
    timestamps: list[float] = []
    fallback_count = 0
    fallback_interval = 1.0 / nominal_fps if nominal_fps > 0 else 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        previous = timestamps[-1] if timestamps else None
        invalid = not math.isfinite(timestamp) or timestamp < 0
        if previous is not None and timestamp <= previous:
            invalid = True
        if invalid:
            fallback_count += 1
            if previous is None:
                timestamp = 0.0
            elif fallback_interval > 0:
                timestamp = previous + fallback_interval
            else:
                timestamp = previous + 1e-6
        timestamps.append(float(timestamp))
    return timestamps, fallback_count


def _sample_quality_metrics(
    path: Path,
    decoded_frame_count: int,
) -> tuple[list[float], list[float]]:
    """从实际可解码帧范围的五个分位位置计算亮度和清晰度。"""
    if decoded_frame_count <= 0:
        return [], []

    target_indices = sorted(
        {
            min(
                decoded_frame_count - 1,
                max(0, round((decoded_frame_count - 1) * ratio)),
            )
            for ratio in (0.02, 0.25, 0.50, 0.75, 0.98)
        }
    )
    target_set = set(target_indices)
    brightness: list[float] = []
    blur: list[float] = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return brightness, blur

    frame_index = 0
    final_target = target_indices[-1]
    while frame_index <= final_target:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_index in target_set:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        frame_index += 1
    cap.release()
    return brightness, blur


def inspect_video(path: Path) -> dict[str, Any]:
    """读取视频元数据，并以实际顺序解码结果确定视觉流范围。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {
            "video_readable": False,
            "width": 0,
            "height": 0,
            "fps": None,
            "frame_count": 0,
            "metadata_frame_count": 0,
            "duration_sec": None,
            "visual_duration_sec": None,
            "visual_coverage_ratio": None,
            "timestamp_fallback_count": 0,
            "codec": "",
            "sampled_frame_count": 0,
            "brightness_mean": None,
            "brightness_std": None,
            "blur_laplacian_mean": None,
            "dark_frame_ratio": None,
            "_decoded_frame_timestamps_sec": [],
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    duration = (
        metadata_frame_count / fps
        if fps > 0 and metadata_frame_count > 0
        else math.nan
    )
    timestamps_sec, timestamp_fallback_count = _decode_frame_timestamps(cap, fps)
    cap.release()

    decoded_frame_count = len(timestamps_sec)
    frame_interval = _estimate_frame_interval(timestamps_sec, fps)
    visual_duration = (
        timestamps_sec[-1] + frame_interval if timestamps_sec else math.nan
    )
    coverage_ratio = (
        visual_duration / duration
        if math.isfinite(duration) and duration > 0 and math.isfinite(visual_duration)
        else math.nan
    )
    brightness, blur = _sample_quality_metrics(path, decoded_frame_count)

    dark_ratio = (
        float(np.mean(np.asarray(brightness) < 25.0)) if brightness else math.nan
    )
    return {
        "video_readable": decoded_frame_count > 0,
        "width": width,
        "height": height,
        "fps": _finite_or_none(fps),
        "frame_count": decoded_frame_count,
        "metadata_frame_count": metadata_frame_count,
        "duration_sec": _finite_or_none(duration),
        "visual_duration_sec": _finite_or_none(visual_duration),
        "visual_coverage_ratio": _finite_or_none(coverage_ratio),
        "timestamp_fallback_count": timestamp_fallback_count,
        "codec": codec,
        "sampled_frame_count": len(brightness),
        "brightness_mean": _finite_or_none(float(np.mean(brightness))) if brightness else None,
        "brightness_std": _finite_or_none(float(np.std(brightness))) if brightness else None,
        "blur_laplacian_mean": _finite_or_none(float(np.mean(blur))) if blur else None,
        "dark_frame_ratio": _finite_or_none(dark_ratio),
        "_decoded_frame_timestamps_sec": timestamps_sec,
    }


def build_visual_timeline(
    sample_id: str,
    frame_timestamps_sec: Iterable[float],
    sample_fps: float,
) -> list[dict[str, Any]]:
    """按实际可解码帧时间戳生成视觉采样计划。"""
    timestamps = np.asarray(list(frame_timestamps_sec), dtype=np.float64)
    timestamps = timestamps[np.isfinite(timestamps) & (timestamps >= 0)]
    if timestamps.size == 0 or sample_fps <= 0:
        return []
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame_timestamps_sec 必须严格递增")

    interval = 1.0 / sample_fps
    frame_interval = _estimate_frame_interval(timestamps.tolist(), 0.0)
    visual_duration = float(timestamps[-1] + frame_interval)
    targets = np.arange(0.0, visual_duration, interval, dtype=np.float64)
    if targets.size == 0:
        targets = np.asarray([0.0], dtype=np.float64)

    timeline: list[dict[str, Any]] = []
    used_source_frames: set[int] = set()
    artifact_id = safe_filename(sample_id)
    for target in targets:
        right = int(np.searchsorted(timestamps, target, side="left"))
        candidates = [min(right, timestamps.size - 1)]
        if right > 0:
            candidates.append(right - 1)
        source_frame = min(
            candidates,
            key=lambda index: (abs(float(timestamps[index]) - float(target)), index),
        )
        if source_frame in used_source_frames:
            continue
        used_source_frames.add(source_frame)
        timeline_index = len(timeline)
        start_sec = float(target)
        end_sec = min(visual_duration, start_sec + interval)
        source_timestamp = float(timestamps[source_frame])
        timeline.append(
            {
                "sample_id": sample_id,
                "timeline_index": timeline_index,
                "source_frame_index": source_frame,
                "source_timestamp_sec": source_timestamp,
                "start_sec": start_sec,
                "end_sec": float(end_sec),
                "center_sec": float((start_sec + end_sec) / 2.0),
                "timestamp_error_sec": source_timestamp - start_sec,
                "frame_relative_path": (
                    f"frames/{artifact_id}/frame_{timeline_index:05d}.jpg"
                ),
            }
        )
    return timeline


def read_label_table(label_path: Path) -> tuple[pd.DataFrame, str]:
    if not label_path.is_file():
        raise PreprocessError(f"标注文件不存在：{label_path}")

    sheets = pd.read_excel(label_path, sheet_name=None)
    selected_name: str | None = None
    selected_frame: pd.DataFrame | None = None
    for sheet_name, frame in sheets.items():
        if REQUIRED_LABEL_COLUMNS.issubset(set(map(str, frame.columns))):
            if sheet_name == "label":
                selected_name, selected_frame = sheet_name, frame
                break
            if selected_frame is None:
                selected_name, selected_frame = sheet_name, frame

    if selected_frame is None or selected_name is None:
        raise PreprocessError(
            f"没有工作表同时包含字段：{sorted(REQUIRED_LABEL_COLUMNS)}"
        )

    frame = selected_frame.copy()
    frame["video_id"] = frame["video_id"].map(normalize_id)
    frame["clip_id"] = frame["clip_id"].map(normalize_id)
    frame["sample_id"] = frame["video_id"] + "$_$" + frame["clip_id"]
    frame["text_original"] = frame["text"].fillna("").astype(str)
    frame["text_normalized"] = frame["text_original"].map(normalize_text)
    frame["label_reg"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["annotation_normalized"] = (
        frame["annotation"].fillna("").astype(str).str.strip().str.lower()
    )
    frame["polarity_from_score"] = frame["label_reg"].map(
        lambda value: polarity_from_score(float(value)) if pd.notna(value) else "invalid"
    )
    frame["label_cls"] = frame["annotation_normalized"].map(ANNOTATION_TO_CLASS)
    frame["annotation_matches_score"] = (
        frame["annotation_normalized"] == frame["polarity_from_score"]
    )
    return frame, selected_name


def scan_videos(input_root: Path) -> tuple[dict[tuple[str, str], Path], list[str]]:
    video_map: dict[tuple[str, str], Path] = {}
    duplicate_keys: list[str] = []
    for path in sorted(input_root.glob("*/*.mp4")):
        key = (path.parent.name, path.stem)
        if key in video_map:
            duplicate_keys.append(f"{key[0]}$_${key[1]}")
        video_map[key] = path
    return video_map, duplicate_keys


def _label_validation_errors(labels: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    duplicates = labels.loc[labels["sample_id"].duplicated(keep=False), "sample_id"].tolist()
    if duplicates:
        errors.append(f"标注表存在重复样本ID：{sorted(set(duplicates))}")
    if (labels["video_id"] == "").any() or (labels["clip_id"] == "").any():
        errors.append("标注表存在空 video_id 或 clip_id")
    invalid_scores = labels["label_reg"].isna() | ~labels["label_reg"].between(-3.0, 3.0)
    if invalid_scores.any():
        errors.append(
            "情感强度非法："
            + ", ".join(labels.loc[invalid_scores, "sample_id"].astype(str).tolist())
        )
    invalid_annotations = ~labels["annotation_normalized"].isin(ANNOTATION_TO_CLASS)
    if invalid_annotations.any():
        errors.append(
            "情感极性非法："
            + ", ".join(labels.loc[invalid_annotations, "sample_id"].astype(str).tolist())
        )
    mismatch = ~labels["annotation_matches_score"]
    if mismatch.any():
        errors.append(
            "情感强度与极性不一致："
            + ", ".join(labels.loc[mismatch, "sample_id"].astype(str).tolist())
        )
    if (labels["text_normalized"] == "").any():
        errors.append(
            "存在空转写："
            + ", ".join(
                labels.loc[labels["text_normalized"] == "", "sample_id"].astype(str).tolist()
            )
        )
    return errors


def _process_sample(
    row: dict[str, Any],
    video_path: Path | None,
    input_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    sample_id = str(row["sample_id"])
    text = str(row["text_normalized"])
    tokens = tokenize_with_offsets(text)
    token_rows = [{"sample_id": sample_id, **token} for token in tokens]

    label_reg = float(row["label_reg"]) if pd.notna(row["label_reg"]) else None
    label_cls = int(row["label_cls"]) if pd.notna(row["label_cls"]) else None

    base = {
        "sample_id": sample_id,
        "video_id": str(row["video_id"]),
        "clip_id": str(row["clip_id"]),
        "text_original": str(row["text_original"]),
        "text_normalized": text,
        "text_char_length": len(text),
        "text_token_count": len(tokens),
        "text_word_count": sum(t["token_kind"] in {"word", "number"} for t in tokens),
        "label_reg": label_reg,
        "label_cls": label_cls,
        "annotation": str(row["annotation"]),
        "annotation_normalized": str(row["annotation_normalized"]),
        "annotation_matches_score": bool(row["annotation_matches_score"]),
        "video_exists": video_path is not None,
    }
    if video_path is None:
        inspection = inspect_video(Path("__missing__"))
        timestamps = inspection.pop("_decoded_frame_timestamps_sec")
        return {
            **base,
            "relative_path": "",
            "file_size_bytes": 0,
            "sha256": "",
            **inspection,
        }, token_rows, timestamps

    inspection = inspect_video(video_path)
    timestamps = inspection.pop("_decoded_frame_timestamps_sec")
    return {
        **base,
        "relative_path": video_path.relative_to(input_root).as_posix(),
        "file_size_bytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
        **inspection,
    }, token_rows, timestamps


def derive_quality_flags(record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not record.get("video_exists", False):
        flags.append("missing_video")
    if not record.get("video_readable", False):
        flags.append("unreadable_video")
    duration = record.get("duration_sec")
    if duration is None or not (2.648 - 0.05 <= float(duration) <= 34.567 + 0.05):
        flags.append("duration_outside_contest_range")
    fps = record.get("fps")
    if fps is None or float(fps) <= 0:
        flags.append("invalid_fps")
    frame_count = int(record.get("frame_count") or 0)
    metadata_frame_count = int(record.get("metadata_frame_count") or 0)
    if frame_count <= 0:
        flags.append("no_decodable_frames")
    if metadata_frame_count > 0 and frame_count < metadata_frame_count:
        flags.append("decoded_frame_count_below_metadata")
    coverage = record.get("visual_coverage_ratio")
    if coverage is not None and float(coverage) < 0.95:
        flags.append("visual_stream_shorter_than_container")
    expected_quality_samples = min(5, frame_count)
    if int(record.get("sampled_frame_count") or 0) < expected_quality_samples:
        flags.append("incomplete_quality_sampling")
    width = int(record.get("width") or 0)
    height = int(record.get("height") or 0)
    if width < 640 or height < 360:
        flags.append("low_resolution")
    if int(record.get("text_word_count") or 0) > 50:
        flags.append("text_over_50_words")
    if int(record.get("text_word_count") or 0) == 0:
        flags.append("empty_text")
    if not bool(record.get("annotation_matches_score", False)):
        flags.append("label_annotation_mismatch")
    dark_ratio = record.get("dark_frame_ratio")
    if dark_ratio is not None and float(dark_ratio) >= 0.8:
        flags.append("mostly_dark_sampled_frames")
    return flags


def resolve_ffmpeg(explicit_path: str | None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise PreprocessError(f"指定的 FFmpeg 不存在：{candidate}")
        return candidate
    located = shutil.which("ffmpeg")
    return Path(located).resolve() if located else None


def extract_audio_wav(
    ffmpeg: Path,
    video_path: Path,
    output_path: Path,
    sample_rate: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        status = "reused"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if completed.returncode != 0 or not output_path.is_file():
            output_path.unlink(missing_ok=True)
            return {
                "audio_status": "failed",
                "audio_error": completed.stderr.strip()[-1000:],
                "audio_relative_path": "",
                "audio_frames": 0,
                "audio_duration_sec": None,
                "audio_sample_rate": None,
                "audio_channels": None,
            }
        status = "extracted"

    try:
        with wave.open(str(output_path), "rb") as stream:
            frames = int(stream.getnframes())
            rate = int(stream.getframerate())
            channels = int(stream.getnchannels())
    except (wave.Error, OSError) as exc:
        return {
            "audio_status": "invalid_wav",
            "audio_error": str(exc),
            "audio_relative_path": "",
            "audio_frames": 0,
            "audio_duration_sec": None,
            "audio_sample_rate": None,
            "audio_channels": None,
        }
    return {
        "audio_status": status,
        "audio_error": "",
        "audio_relative_path": output_path.as_posix(),
        "audio_frames": frames,
        "audio_duration_sec": frames / rate if rate > 0 else None,
        "audio_sample_rate": rate,
        "audio_channels": channels,
    }


def extract_sampled_frames(
    video_path: Path,
    output_root: Path,
    timeline_rows: Iterable[dict[str, Any]],
    overwrite: bool,
) -> tuple[int, int, int]:
    rows = sorted(
        list(timeline_rows),
        key=lambda row: int(row["source_frame_index"]),
    )
    if not rows:
        return 0, 0, 0

    expected_paths = {
        (output_root / str(row["frame_relative_path"])).resolve() for row in rows
    }
    removed_stale = 0
    if overwrite:
        frame_directories = {path.parent for path in expected_paths}
        resolved_root = output_root.resolve()
        for directory in frame_directories:
            if resolved_root != directory and resolved_root not in directory.parents:
                raise PreprocessError(f"帧输出目录越界：{directory}")
            if directory.is_dir():
                for existing in directory.glob("frame_*.jpg"):
                    if existing.resolve() not in expected_paths:
                        existing.unlink()
                        removed_stale += 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0, len(rows), removed_stale

    written = 0
    pending: dict[int, list[tuple[dict[str, Any], Path]]] = {}
    for row in rows:
        output_path = output_root / str(row["frame_relative_path"])
        if output_path.is_file() and not overwrite:
            written += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pending.setdefault(int(row["source_frame_index"]), []).append((row, output_path))

    frame_index = 0
    while pending:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        targets = pending.pop(frame_index, [])
        for _, output_path in targets:
            if cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                written += 1
        frame_index += 1
    cap.release()
    failed = len(rows) - written
    return written, failed, removed_stale


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    input_root = Path(config.input_root).resolve()
    output_root = Path(config.output_root).resolve()
    label_path = Path(config.label_file).resolve()
    if not input_root.is_dir():
        raise PreprocessError(f"附件1目录不存在：{input_root}")
    if input_root == output_root or input_root in output_root.parents:
        raise PreprocessError("输出目录不能位于附件1原始目录内部")
    if config.frame_sample_fps <= 0:
        raise PreprocessError("frame_sample_fps 必须大于0")
    if config.audio_sample_rate <= 0:
        raise PreprocessError("audio_sample_rate 必须大于0")

    ffmpeg = resolve_ffmpeg(config.ffmpeg)
    if config.extract_audio and ffmpeg is None:
        raise PreprocessError(
            "请求了音频抽取，但未找到 FFmpeg。请安装 FFmpeg 或通过 --ffmpeg 指定路径。"
        )

    labels, sheet_name = read_label_table(label_path)
    validation_errors = _label_validation_errors(labels)
    video_map, duplicate_video_keys = scan_videos(input_root)
    if duplicate_video_keys:
        validation_errors.append(f"视频键重复：{sorted(set(duplicate_video_keys))}")

    label_keys = set(zip(labels["video_id"], labels["clip_id"], strict=True))
    video_keys = set(video_map)
    missing_video_keys = sorted(label_keys - video_keys)
    extra_video_keys = sorted(video_keys - label_keys)
    if missing_video_keys:
        validation_errors.append(
            "标注无视频：" + ", ".join(f"{a}$_${b}" for a, b in missing_video_keys)
        )
    if extra_video_keys:
        validation_errors.append(
            "视频无标注：" + ", ".join(f"{a}$_${b}" for a, b in extra_video_keys)
        )

    label_records = labels.to_dict(orient="records")

    def process(
        row: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
        path = video_map.get((str(row["video_id"]), str(row["clip_id"])))
        return _process_sample(row, path, input_root)

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as executor:
        processed = list(executor.map(process, label_records))

    manifest_records = [item[0] for item in processed]
    token_records = [token for item in processed for token in item[1]]
    decoded_timestamps_by_sample = {
        str(item[0]["sample_id"]): item[2] for item in processed
    }
    manifest_records.sort(key=lambda item: item["sample_id"])
    token_records.sort(key=lambda item: (item["sample_id"], item["token_index"]))

    timeline_records: list[dict[str, Any]] = []
    timeline_by_sample: dict[str, list[dict[str, Any]]] = {}
    for record in manifest_records:
        sample_id = str(record["sample_id"])
        rows = build_visual_timeline(
            sample_id=sample_id,
            frame_timestamps_sec=decoded_timestamps_by_sample[sample_id],
            sample_fps=config.frame_sample_fps,
        )
        timeline_by_sample[sample_id] = rows
        timeline_records.extend(rows)

    if config.extract_audio:
        assert ffmpeg is not None
        for record in manifest_records:
            relative_path = str(record.get("relative_path") or "")
            artifact_id = safe_filename(str(record["sample_id"]))
            audio_path = output_root / "audio" / f"{artifact_id}.wav"
            if not relative_path:
                result = {
                    "audio_status": "missing_video",
                    "audio_error": "",
                    "audio_relative_path": "",
                    "audio_frames": 0,
                    "audio_duration_sec": None,
                    "audio_sample_rate": None,
                    "audio_channels": None,
                }
            else:
                result = extract_audio_wav(
                    ffmpeg=ffmpeg,
                    video_path=input_root / relative_path,
                    output_path=audio_path,
                    sample_rate=config.audio_sample_rate,
                    overwrite=config.overwrite_media,
                )
                if result["audio_relative_path"]:
                    result["audio_relative_path"] = audio_path.relative_to(output_root).as_posix()
            record.update(result)
    else:
        for record in manifest_records:
            record.update(
                {
                    "audio_status": "not_requested",
                    "audio_error": "",
                    "audio_relative_path": "",
                    "audio_frames": 0,
                    "audio_duration_sec": None,
                    "audio_sample_rate": None,
                    "audio_channels": None,
                }
            )

    frame_failures = 0
    frame_writes = 0
    stale_frames_removed = 0
    if config.extract_frames:
        for record in manifest_records:
            relative_path = str(record.get("relative_path") or "")
            rows = timeline_by_sample[str(record["sample_id"])]
            if not relative_path:
                frame_failures += len(rows)
                continue
            written, failed, removed_stale = extract_sampled_frames(
                video_path=input_root / relative_path,
                output_root=output_root,
                timeline_rows=rows,
                overwrite=config.overwrite_media,
            )
            frame_writes += written
            frame_failures += failed
            stale_frames_removed += removed_stale

    unreadable = [
        str(record["sample_id"])
        for record in manifest_records
        if not bool(record.get("video_readable"))
    ]
    if unreadable:
        validation_errors.append(f"视频不可解码：{unreadable}")
    if config.extract_audio:
        audio_failures = [
            str(record["sample_id"])
            for record in manifest_records
            if record.get("audio_status") in {"failed", "invalid_wav", "missing_video"}
        ]
        if audio_failures:
            validation_errors.append(f"音频抽取失败：{audio_failures}")
    else:
        audio_failures = []
    if config.extract_frames and frame_failures:
        validation_errors.append(f"抽帧失败数：{frame_failures}")

    for record in manifest_records:
        flags = derive_quality_flags(record)
        if record.get("audio_status") in {"failed", "invalid_wav", "missing_video"}:
            flags.append("audio_extraction_failed")
        record["quality_flags"] = ";".join(sorted(set(flags)))
        record["quality_ok"] = len(flags) == 0

    output_root.mkdir(parents=True, exist_ok=True)
    tables_root = output_root / "tables"
    reports_root = output_root / "reports"

    manifest_frame = pd.DataFrame(manifest_records)
    token_frame = pd.DataFrame(token_records)
    timeline_frame = pd.DataFrame(timeline_records)
    normalized_labels = labels[
        [
            "sample_id",
            "video_id",
            "clip_id",
            "text_original",
            "text_normalized",
            "label_reg",
            "label_cls",
            "annotation",
            "annotation_normalized",
            "annotation_matches_score",
        ]
    ].sort_values("sample_id")

    atomic_write_csv(output_root / "manifest.csv", manifest_frame)
    atomic_write_csv(tables_root / "labels_normalized.csv", normalized_labels)
    atomic_write_csv(tables_root / "tokens.csv", token_frame)
    atomic_write_csv(tables_root / "visual_timeline.csv", timeline_frame)

    flag_counts: Counter[str] = Counter()
    for value in manifest_frame["quality_flags"].fillna("").astype(str):
        flag_counts.update(flag for flag in value.split(";") if flag)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "label_sheet": sheet_name,
        "label_rows": int(len(labels)),
        "video_files": int(len(video_map)),
        "manifest_rows": int(len(manifest_frame)),
        "token_rows": int(len(token_frame)),
        "visual_timeline_rows": int(len(timeline_frame)),
        "missing_video_count": len(missing_video_keys),
        "extra_video_count": len(extra_video_keys),
        "unreadable_video_count": len(unreadable),
        "audio_failure_count": len(audio_failures),
        "sampled_frames_written": frame_writes,
        "sampled_frame_failures": frame_failures,
        "stale_frames_removed": stale_frames_removed,
        "decoded_frame_count_total": int(
            sum(int(record.get("frame_count") or 0) for record in manifest_records)
        ),
        "metadata_frame_count_mismatch_count": int(
            sum(
                int(record.get("frame_count") or 0)
                != int(record.get("metadata_frame_count") or 0)
                for record in manifest_records
            )
        ),
        "visual_stream_shorter_count": int(
            sum(
                record.get("visual_coverage_ratio") is not None
                and float(record["visual_coverage_ratio"]) < 0.95
                for record in manifest_records
            )
        ),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "passed": len(validation_errors) == 0,
        "outputs": {
            "manifest": "manifest.csv",
            "labels": "tables/labels_normalized.csv",
            "tokens": "tables/tokens.csv",
            "visual_timeline": "tables/visual_timeline.csv",
        },
    }
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
    }
    config_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "config": asdict(config),
        "runtime": runtime,
        "source_label_sha256": sha256_file(label_path),
        "assumptions": {
            "text_normalization": "Unicode NFKC + whitespace collapse; semantics unchanged",
            "visual_timeline": (
                "nearest actually decodable frame selected from sequential OpenCV "
                "timestamps; container frame-count estimates are audit-only"
            ),
            "audio": "optional FFmpeg PCM s16le mono resampling",
            "raw_data_mutated": False,
        },
    }
    atomic_write_json(reports_root / "preprocess_report.json", report)
    atomic_write_json(output_root / "preprocess_config.json", config_payload)

    if config.strict and validation_errors:
        raise PreprocessError(
            f"预处理已写出审计文件，但严格校验失败，共 {len(validation_errors)} 项；"
            f"详见 {reports_root / 'preprocess_report.json'}"
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="问题一：附件1原始视频与标注的可复现预处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=None,
        help="默认使用 <input-root>/label-100.xlsx",
    )
    parser.add_argument(
        "--frame-sample-fps",
        type=float,
        default=5.0,
        help="视觉采样计划及可选抽帧的目标帧率",
    )
    parser.add_argument(
        "--extract-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否按视觉时间轴实际写出JPEG帧",
    )
    parser.add_argument(
        "--extract-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否调用FFmpeg写出16kHz单声道WAV",
    )
    parser.add_argument("--audio-sample-rate", type=int, default=16_000)
    parser.add_argument("--ffmpeg", type=str, default=None, help="FFmpeg可执行文件路径")
    parser.add_argument(
        "--overwrite-media",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否覆盖已存在的WAV和JPEG",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="发现缺文件、坏标签、不可解码或媒体抽取失败时返回非零状态",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    label_file = (args.label_file or (input_root / "label-100.xlsx")).resolve()
    config = PipelineConfig(
        input_root=str(input_root),
        output_root=str(args.output_root.resolve()),
        label_file=str(label_file),
        frame_sample_fps=float(args.frame_sample_fps),
        extract_frames=bool(args.extract_frames),
        extract_audio=bool(args.extract_audio),
        audio_sample_rate=int(args.audio_sample_rate),
        overwrite_media=bool(args.overwrite_media),
        strict=bool(args.strict),
        workers=max(1, int(args.workers)),
        ffmpeg=args.ffmpeg,
    )
    try:
        report = run_pipeline(config)
    except PreprocessError as exc:
        print(f"[problem1-preprocess] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[problem1-preprocess] "
        f"样本={report['manifest_rows']}，词元={report['token_rows']}，"
        f"视觉时间片={report['visual_timeline_rows']}，"
        f"校验错误={report['validation_error_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
