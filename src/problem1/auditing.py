"""Acceptance audit for Problem 1 coverage, mappings, masks, and reproducibility."""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np

from .alignment.consensus_timeline import MODALITIES
from .config import Problem1Config
from .io import atomic_csv, atomic_json, load_feature, safe_id
from .reporting import write_feature_summary_table


def _read_csv(path: Any) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_problem1(cfg: Problem1Config, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the minimum organizer-facing acceptance contract for every result."""
    manifest = _read_csv(cfg.path("preprocessed_root") / "manifest.csv")
    expected_ids = [row["sample_id"] for row in manifest]
    expected_set = set(expected_ids)
    record_ids = [str(row["sample_id"]) for row in records]
    successful = [row for row in records if row["status"] != "failed"]
    successful_ids = [str(row["sample_id"]) for row in successful]
    successful_set = set(successful_ids)

    validation_path = cfg.path("feature_root") / "validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    feature_resolved_path = cfg.path("feature_root") / "resolved_config.json"
    feature_resolved = json.loads(feature_resolved_path.read_text(encoding="utf-8")) if feature_resolved_path.exists() else {}
    resolved_path = cfg.path("aligned_root") / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}

    global_path = cfg.path("aligned_root") / "aligned_dataset.npz"
    global_data: dict[str, np.ndarray] = {}
    global_index: dict[str, int] = {}
    global_errors: list[str] = []
    if global_path.exists():
        with np.load(global_path, allow_pickle=False) as data:
            global_data = {name: np.asarray(data[name]) for name in data.files}
        global_ids = [str(value) for value in global_data.get("sample_ids", np.asarray([]))]
        global_index = {sample_id: index for index, sample_id in enumerate(global_ids)}
        if len(global_ids) != len(set(global_ids)):
            global_errors.append("aligned_dataset.npz包含重复sample_id")
        if set(global_ids) != successful_set:
            global_errors.append("aligned_dataset.npz的sample_id集合与成功清单不一致")
    else:
        global_errors.append("缺少aligned_dataset.npz")

    audit_rows: list[dict[str, Any]] = []
    total_mapping_errors = 0
    padding_violations = 0
    tolerance = 1e-5
    evidence_mass = float(cfg.section("alignment")["evidence_mass"])

    for record in successful:
        sample_id = str(record["sample_id"])
        feature_dir = cfg.path("feature_root") / "samples" / safe_id(sample_id)
        aligned_dir = cfg.path("aligned_root") / "samples" / safe_id(sample_id)
        errors: list[str] = []
        sequences = {}
        try:
            sequences = {name: load_feature(feature_dir / f"{name}.npz") for name in MODALITIES}
        except Exception as exc:
            errors.append(f"特征读取失败: {type(exc).__name__}: {exc}")

        aligned_path = aligned_dir / "alignment.npz"
        mapping_path = aligned_dir / "mapping.json"
        metrics_path = aligned_dir / "metrics.json"
        if not aligned_path.exists() or not mapping_path.exists() or not metrics_path.exists():
            errors.append("缺少alignment.npz、mapping.json或metrics.json")
        elif len(sequences) == len(MODALITIES):
            try:
                mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
                with np.load(aligned_path, allow_pickle=False) as data:
                    aligned = {name: np.asarray(data[name]) for name in data.files}
                consensus_time = np.asarray(aligned["consensus_time"], dtype=float)
                uncertainty = np.asarray(aligned["uncertainty"], dtype=float)
                length = len(consensus_time)
                if length != int(record["consensus_steps"]):
                    errors.append("样本级共识长度与清单不一致")
                if len(mappings) != length:
                    errors.append("mapping条数与共识长度不一致")
                if not np.isfinite(consensus_time).all() or np.any(np.diff(consensus_time) < -tolerance):
                    errors.append("consensus_time包含非有限值或非单调")
                if not np.isfinite(uncertainty).all() or np.any((uncertainty < 0) | (uncertainty > 1 + tolerance)):
                    errors.append("uncertainty包含非有限值或越界")

                for name in MODALITIES:
                    plan = np.asarray(aligned[f"plan_{name}"], dtype=float)
                    if plan.shape != (len(sequences[name].features), length):
                        errors.append(f"{name}传输矩阵形状错误")
                    if not np.isfinite(plan).all() or np.any(plan < -tolerance):
                        errors.append(f"{name}传输矩阵包含非法值")

                if len(mappings) == length:
                    for column, item in enumerate(mappings):
                        if int(item.get("consensus_index", -1)) != column:
                            errors.append(f"mapping[{column}]共识索引错误")
                            continue
                        if abs(float(item["consensus_time_sec"]) - consensus_time[column]) > tolerance:
                            errors.append(f"mapping[{column}]共识时间不一致")
                        if abs(float(item["uncertainty"]) - uncertainty[column]) > tolerance:
                            errors.append(f"mapping[{column}]不确定性不一致")
                        for name in MODALITIES:
                            evidence = item.get("modalities", {}).get(name, {})
                            rows = np.asarray(evidence.get("source_row_indices", []), dtype=int)
                            sequence = sequences[name]
                            if rows.size == 0 or np.any(rows < 0) or np.any(rows >= len(sequence.features)):
                                errors.append(f"mapping[{column}]/{name}源行索引越界或为空")
                                continue
                            expected_indices = sequence.source_index[rows].tolist()
                            if expected_indices != evidence.get("source_indices"):
                                errors.append(f"mapping[{column}]/{name}原始索引不一致")
                            if abs(float(evidence["source_start"]) - float(sequence.start[rows].min())) > tolerance:
                                errors.append(f"mapping[{column}]/{name}起点不一致")
                            if abs(float(evidence["source_end"]) - float(sequence.end[rows].max())) > tolerance:
                                errors.append(f"mapping[{column}]/{name}终点不一致")
                            if float(evidence["covered_mass"]) + tolerance < evidence_mass:
                                errors.append(f"mapping[{column}]/{name}证据覆盖质量不足")
            except Exception as exc:
                errors.append(f"映射校验异常: {type(exc).__name__}: {exc}")

        sample_padding_ok = True
        if sample_id not in global_index:
            errors.append("样本未写入aligned_dataset.npz")
            sample_padding_ok = False
        elif global_data:
            index = global_index[sample_id]
            length = int(record["consensus_steps"])
            mask = np.asarray(global_data["valid_mask"][index], dtype=bool)
            if not np.all(mask[:length]) or np.any(mask[length:]):
                errors.append("全量数据集valid_mask与有效长度不一致")
                sample_padding_ok = False
            padded_names = ("consensus", "aligned_text", "aligned_audio", "aligned_vision",
                            "consensus_time", "uncertainty")
            for name in padded_names:
                if np.any(np.asarray(global_data[name][index, length:]) != 0):
                    errors.append(f"全量数据集{name}填充区非零")
                    sample_padding_ok = False
        if not sample_padding_ok:
            padding_violations += 1

        total_mapping_errors += len(errors)
        audit_rows.append({
            "sample_id": sample_id,
            "feature_files_complete": len(sequences) == len(MODALITIES),
            "result_files_complete": aligned_path.exists() and mapping_path.exists() and metrics_path.exists(),
            "mapping_audit_passed": not errors,
            "padding_audit_passed": sample_padding_ok,
            "error_count": len(errors),
            "errors": " | ".join(errors),
        })

    passed_samples = sum(bool(row["mapping_audit_passed"]) for row in audit_rows)
    duplicate_count = len(record_ids) - len(set(record_ids))
    missing_ids = sorted(expected_set - successful_set)
    unexpected_ids = sorted(set(record_ids) - expected_set)
    full_coverage = not missing_ids and not unexpected_ids and duplicate_count == 0
    fingerprint_match = resolved.get("fingerprint") == cfg.fingerprint
    feature_fingerprint_match = feature_resolved.get("feature_fingerprint") == cfg.feature_fingerprint
    warning_counts: dict[str, int] = {}
    for warning in validation.get("warnings", []):
        modality = str(warning.get("modality", "unknown"))
        warning_counts[modality] = warning_counts.get(modality, 0) + 1
    summary = {
        "expected_samples": len(expected_ids),
        "processed_samples": len(records),
        "successful_samples": len(successful),
        "sample_coverage_rate": len(expected_set & successful_set) / max(len(expected_set), 1),
        "duplicate_sample_id_count": duplicate_count,
        "missing_sample_ids": missing_ids,
        "unexpected_sample_ids": unexpected_ids,
        "feature_validation_passed": bool(validation.get("passed", False)),
        "feature_validation_error_count": int(validation.get("error_count", 0)),
        "feature_validation_warning_count": int(validation.get("warning_count", 0)),
        "feature_validation_warning_count_by_modality": warning_counts,
        "mapping_audit_passed_samples": passed_samples,
        "mapping_audit_pass_rate": passed_samples / max(len(audit_rows), 1),
        "mapping_error_count": total_mapping_errors + len(global_errors),
        "padding_violation_count": padding_violations,
        "global_errors": global_errors,
        "config_fingerprint": cfg.fingerprint,
        "resolved_config_fingerprint_match": fingerprint_match,
        "feature_config_fingerprint_match": feature_fingerprint_match,
        "overall_passed": bool(
            full_coverage and len(successful) == len(records) and validation.get("passed", False)
            and passed_samples == len(audit_rows) and not global_errors and fingerprint_match
            and feature_fingerprint_match
        ),
    }
    report_root = cfg.path("alignment_output_root")
    atomic_csv(report_root / "acceptance_audit.csv", audit_rows)
    feature_summary_path = write_feature_summary_table(cfg, records, audit_rows)
    try:
        feature_summary_display = feature_summary_path.relative_to(cfg.project_root).as_posix()
    except ValueError:
        feature_summary_display = str(feature_summary_path)
    summary["feature_summary_table"] = feature_summary_display
    atomic_json(report_root / "acceptance_audit.json", summary)
    return summary
