"""Concurrent alignment orchestration and artifact persistence."""

from __future__ import annotations

import csv
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Problem1Config
from ..io import atomic_csv, atomic_json, atomic_npz, load_feature, safe_id
from .consensus_timeline import MODALITIES, align_sample


def _read_manifest(cfg: Problem1Config, limit: int | None) -> list[dict[str, str]]:
    with (cfg.path("preprocessed_root") / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows if limit is None else rows[:limit]


def _result_paths(cfg: Problem1Config, sample_id: str) -> tuple[Path, Path, Path]:
    directory = cfg.path("aligned_root") / "samples" / safe_id(sample_id)
    return directory / "alignment.npz", directory / "mapping.json", directory / "metrics.json"


def _save_result(cfg: Problem1Config, result: Any) -> None:
    aligned_path, mapping_path, metrics_path = _result_paths(cfg, result.sample_id)
    atomic_npz(
        aligned_path, compressed=bool(cfg.section("output")["compressed"]),
        sample_id=np.asarray(result.sample_id), consensus=result.consensus, consensus_time=result.consensus_time,
        valid_mask=result.valid_mask, uncertainty=result.uncertainty,
        aligned_text=result.aligned["text"], aligned_audio=result.aligned["audio"], aligned_vision=result.aligned["vision"],
        plan_text=result.plans["text"], plan_audio=result.plans["audio"], plan_vision=result.plans["vision"],
    )
    atomic_json(mapping_path, result.mappings)
    atomic_json(metrics_path, {"metrics": result.metrics, "history": result.history})


def _align_one(row: dict[str, str], cfg: Problem1Config, overwrite: bool,
               logger: logging.Logger) -> dict[str, Any]:
    sample_id, started = row["sample_id"], time.perf_counter()
    aligned_path, _, metrics_path = _result_paths(cfg, sample_id)
    try:
        if aligned_path.exists() and metrics_path.exists() and not overwrite and bool(cfg.section("runtime")["resume"]):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            status = "resumed"
        else:
            directory = cfg.path("feature_root") / "samples" / safe_id(sample_id)
            sequences = {name: load_feature(directory / f"{name}.npz") for name in MODALITIES}
            result = align_sample(sequences, cfg, logger)
            _save_result(cfg, result)
            metrics, status = result.metrics, "completed"
        return {
            "sample_id": sample_id, "status": status, "label_reg": float(row["label_reg"]),
            "label_cls": int(row["label_cls"]), "consensus_steps": int(metrics["consensus_steps"]),
            "iterations": int(metrics["iterations"]), "converged": bool(metrics["converged"]),
            "objective": float(metrics["objective"]), "mean_uncertainty": float(metrics["mean_uncertainty"]),
            "max_marginal_residual": float(metrics["max_marginal_residual"]),
            "monotonic_violations": sum(int(x) for x in metrics["monotonic_violations"].values()),
            "elapsed_sec": round(time.perf_counter() - started, 4), "error": "",
        }
    except Exception as exc:
        return {"sample_id": sample_id, "status": "failed", "label_reg": row.get("label_reg", ""),
                "label_cls": row.get("label_cls", ""), "elapsed_sec": round(time.perf_counter() - started, 4),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=10)}


def _build_dataset(cfg: Problem1Config, rows: list[dict[str, str]], records: list[dict[str, Any]]) -> None:
    successful = {item["sample_id"] for item in records if item["status"] != "failed"}
    selected = [row for row in rows if row["sample_id"] in successful]
    if not selected:
        return
    maximum = int(cfg.section("alignment")["max_consensus_steps"])
    dimension = int(cfg.section("multiscale")["common_dimension"])
    count = len(selected)
    consensus = np.zeros((count, maximum, dimension), dtype=np.float32)
    aligned = {name: np.zeros_like(consensus) for name in MODALITIES}
    times = np.zeros((count, maximum), dtype=np.float32)
    uncertainty = np.zeros((count, maximum), dtype=np.float32)
    mask = np.zeros((count, maximum), dtype=np.bool_)
    for index, row in enumerate(selected):
        path, _, _ = _result_paths(cfg, row["sample_id"])
        with np.load(path, allow_pickle=False) as data:
            length = min(maximum, len(data["consensus_time"]))
            consensus[index, :length] = data["consensus"][:length]
            times[index, :length] = data["consensus_time"][:length]
            uncertainty[index, :length] = data["uncertainty"][:length]
            mask[index, :length] = True
            for name in MODALITIES:
                aligned[name][index, :length] = data[f"aligned_{name}"][:length]
    atomic_npz(
        cfg.path("aligned_root") / "aligned_dataset.npz", compressed=True,
        sample_ids=np.asarray([row["sample_id"] for row in selected]),
        labels_reg=np.asarray([float(row["label_reg"]) for row in selected], dtype=np.float32),
        labels_cls=np.asarray([int(row["label_cls"]) for row in selected], dtype=np.int8),
        consensus=consensus, aligned_text=aligned["text"], aligned_audio=aligned["audio"],
        aligned_vision=aligned["vision"], consensus_time=times, uncertainty=uncertainty, valid_mask=mask,
    )


def run_alignment(cfg: Problem1Config, *, overwrite: bool = False, limit: int | None = None,
                  logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    logger = logger or logging.getLogger("problem1.alignment")
    rows = _read_manifest(cfg, limit)
    workers = int(cfg.section("runtime")["alignment_workers"])
    resolved = cfg.path("aligned_root") / "resolved_config.json"
    if resolved.exists() and not overwrite:
        try:
            previous_fingerprint = json.loads(resolved.read_text(encoding="utf-8"))["fingerprint"]
        except (OSError, KeyError, json.JSONDecodeError):
            previous_fingerprint = None
        if previous_fingerprint != cfg.fingerprint:
            logger.warning("配置指纹已变化，禁止复用旧对齐结果并自动重新求解")
            overwrite = True
    logger.info("模型求解开始: 样本=%d, alignment_workers=%d, overwrite=%s", len(rows), workers, overwrite)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p1-align") as executor:
        futures = {executor.submit(_align_one, row, cfg, overwrite, logger): row["sample_id"] for row in rows}
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            if record["status"] == "failed":
                logger.error("[%d/%d] %s 求解失败: %s", number, len(rows), record["sample_id"], record["error"])
            else:
                logger.info("[%d/%d] %s %s: objective=%.6f, uncertainty=%.4f (%.2fs)", number, len(rows),
                            record["sample_id"], record["status"], record["objective"],
                            record["mean_uncertainty"], record["elapsed_sec"])
    records.sort(key=lambda item: item["sample_id"])
    fields = ["sample_id", "status", "label_reg", "label_cls", "consensus_steps", "iterations", "converged",
              "objective", "mean_uncertainty", "max_marginal_residual", "monotonic_violations", "elapsed_sec", "error"]
    report_root = cfg.path("alignment_output_root")
    atomic_csv(report_root / "alignment_manifest.csv", records, fields)
    atomic_json(resolved, {"fingerprint": cfg.fingerprint, "config": cfg.raw})
    failures = [item for item in records if item["status"] == "failed"]
    atomic_json(report_root / "alignment_errors.json", failures)
    _build_dataset(cfg, rows, records)
    if failures and bool(cfg.section("runtime")["strict"]):
        raise RuntimeError(
            f"模型求解严格校验失败: {len(failures)}/{len(records)}，"
            f"详见{report_root / 'alignment_errors.json'}"
        )
    return records
