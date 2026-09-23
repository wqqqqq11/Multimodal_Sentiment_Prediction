"""Concurrent, restartable tri-modal feature extraction pipeline."""

from __future__ import annotations

import csv
import json
import logging
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np

from ..config import Problem1Config
from ..io import atomic_csv, atomic_json, load_feature, safe_id, save_feature
from .audio.extractor import extract_audio
from .text.extractor import extract_text
from .vision.extractor import extract_vision


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["sample_id"], []).append(row)
    return groups


def _feature_paths(root: Path, sample_id: str) -> dict[str, Path]:
    directory = root / "samples" / safe_id(sample_id)
    return {name: directory / f"{name}.npz" for name in ("text", "audio", "vision")}


def _encoder_config(cfg: Problem1Config, modality: str) -> dict[str, Any]:
    return {
        **cfg.section(modality),
        "project_root": str(cfg.project_root),
        "device": cfg.section("runtime")["device"],
        "mixed_precision": bool(cfg.section("runtime")["mixed_precision"]),
    }


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _process_sample(row: dict[str, str], token_rows: list[dict[str, str]], vision_rows: list[dict[str, str]],
                    cfg: Problem1Config, overwrite: bool) -> dict[str, Any]:
    started, sample_id = time.perf_counter(), row["sample_id"]
    paths = _feature_paths(cfg.path("feature_root"), sample_id)
    try:
        if not overwrite and bool(cfg.section("runtime")["resume"]) and all(path.exists() for path in paths.values()):
            sequences = {name: load_feature(path) for name, path in paths.items()}
            status = "resumed"
        else:
            root = cfg.path("preprocessed_root")
            sequences = {
                "text": extract_text(sample_id, row["text_normalized"], _encoder_config(cfg, "text"), token_rows),
                "audio": extract_audio(sample_id, root / row["audio_relative_path"], _encoder_config(cfg, "audio")),
                "vision": extract_vision(sample_id, root, vision_rows, _encoder_config(cfg, "vision")),
            }
            for modality, sequence in sequences.items():
                save_feature(paths[modality], sequence, bool(cfg.section("output")["compressed"]))
            status = "completed"
        record: dict[str, Any] = {
            "sample_id": sample_id, "status": status, "label_reg": float(row["label_reg"]),
            "label_cls": int(row["label_cls"]), "elapsed_sec": round(time.perf_counter() - started, 4), "error": "",
        }
        for modality, sequence in sequences.items():
            record[f"{modality}_steps"] = int(sequence.features.shape[0])
            record[f"{modality}_dimension"] = int(sequence.features.shape[1])
            record[f"{modality}_quality_mean"] = round(float(sequence.quality.mean()), 6)
        return record
    except Exception as exc:
        return {"sample_id": sample_id, "status": "failed", "label_reg": row.get("label_reg", ""),
                "label_cls": row.get("label_cls", ""), "elapsed_sec": round(time.perf_counter() - started, 4),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}


def run_feature_extraction(cfg: Problem1Config, *, overwrite: bool = False, limit: int | None = None,
                           logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    logger = logger or logging.getLogger("problem1.features")
    root, output = cfg.path("preprocessed_root"), cfg.path("feature_root")
    manifest = _read_csv(root / "manifest.csv")
    if limit is not None:
        manifest = manifest[:limit]
    token_groups = _group(_read_csv(root / "tables" / "tokens.csv"))
    vision_groups = _group(_read_csv(root / "tables" / "visual_timeline.csv"))
    workers = int(cfg.section("runtime")["workers"])
    previous_config = output / "resolved_config.json"
    if previous_config.exists() and not overwrite:
        try:
            previous_fingerprint = json.loads(previous_config.read_text(encoding="utf-8"))["fingerprint"]
        except (OSError, KeyError, json.JSONDecodeError):
            previous_fingerprint = None
        if previous_fingerprint != cfg.fingerprint:
            logger.warning("配置指纹已变化，禁止复用旧特征并自动重新提取")
            overwrite = True
    logger.info("特征提取开始: 样本=%d, workers=%d, overwrite=%s", len(manifest), workers, overwrite)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p1-feature") as executor:
        futures = {executor.submit(_process_sample, row, token_groups.get(row["sample_id"], []),
                                   vision_groups.get(row["sample_id"], []), cfg, overwrite): row["sample_id"]
                   for row in manifest}
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            if record["status"] == "failed":
                logger.error("[%d/%d] %s 失败: %s", number, len(manifest), record["sample_id"], record["error"])
            else:
                logger.info("[%d/%d] %s %s (%.2fs)", number, len(manifest), record["sample_id"],
                            record["status"], float(record["elapsed_sec"]))
    records.sort(key=lambda item: item["sample_id"])
    fields = ["sample_id", "status", "label_reg", "label_cls", "elapsed_sec", "error",
              "text_steps", "text_dimension", "text_quality_mean", "audio_steps", "audio_dimension",
              "audio_quality_mean", "vision_steps", "vision_dimension", "vision_quality_mean"]
    atomic_csv(output / "feature_manifest.csv", records, fields)
    atomic_json(output / "resolved_config.json", {"fingerprint": cfg.fingerprint, "config": cfg.raw})
    atomic_json(output / "model_versions.json", {"python": platform.python_version(), "numpy": np.__version__,
                                                   "opencv": cv2.__version__,
                                                   "torch": _installed_version("torch"),
                                                   "transformers": _installed_version("transformers"),
                                                   "mediapipe": _installed_version("mediapipe"),
                                                   "timm": _installed_version("timm"), "backends": {
                                                       "text": cfg.section("text")["backend"],
                                                       "audio": cfg.section("audio")["backend"],
                                                       "vision": cfg.section("vision")["backend"]}})
    failed = [record for record in records if record["status"] == "failed"]
    atomic_json(output / "feature_errors.json", failed)
    if failed and bool(cfg.section("runtime")["strict"]):
        raise RuntimeError(f"特征提取严格校验失败: {len(failed)}/{len(records)}，详见feature_errors.json")
    return records
