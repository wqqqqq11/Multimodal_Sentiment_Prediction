"""Problem 3 preprocessing: sparse evidence, prototype memory, and counterfactual assets.

The main model inputs deliberately reuse Problem 2's training-only scaler and aligned
feature contract.  Problem 3 outputs are independent and contain no privileged text
embedding or hand-crafted reliability vector that could bypass the evidence bottleneck.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data_progressing.problem2.masking import masks_from_attention, modality_availability
from data_progressing.problem2.scaling import scaler_from_dict
from data_progressing.problem3.config import Problem3Config
from data_progressing.problem3.io import ensure_dir, file_sha256, write_csv, write_json, write_npz
from data_progressing.problem3.media_mapping import (
    inferred_nonzero_length,
    proportional_time_mapping,
    reconstruct_to_content,
    reconstruction_metrics,
    video_metadata,
)
from data_progressing.problem3.text_mapping import build_token_mappings, intensity_bin, load_tokenizer
from data_progressing.problem3.validation import validate_saved_root, validate_split_arrays


SPLITS = ("train", "valid", "test")
MASK_MODALITIES = ("text", "audio", "vision")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        return {name: artifact[name] for name in artifact.files}


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return value


def _text_bert_fields(text_bert: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(text_bert)
    if array.shape != (3, 50):
        raise ValueError(f"Expected text_bert shape (3, 50), got {array.shape}")
    return (
        np.asarray(array[0], dtype=np.int32),
        np.asarray(array[1], dtype=np.uint8),
        np.asarray(array[2], dtype=np.uint8),
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _failure_mask(observed: np.ndarray, content: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    """Flag a stream as failed when fewer than threshold of content steps are observed."""
    observed = np.asarray(observed, dtype=bool)
    content = np.asarray(content, dtype=bool)
    denominator = np.maximum(content.sum(axis=1), 1)
    ratio = observed.sum(axis=1) / denominator
    failed_sample = ratio < float(threshold)
    return content & failed_sample[:, None]


def _with_problem3_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(arrays[name])
        for name in (
            "sample_id", "raw_text", "input_ids", "attention_mask", "token_type_ids",
            "content_mask", "structural_mask", "padding_mask", "audio", "vision",
            "text_observed_mask", "text_natural_zero_mask", "audio_observed_mask",
            "audio_natural_zero_mask", "vision_observed_mask", "vision_natural_zero_mask",
            "classification_labels", "regression_labels",
        )
    }
    content = result["content_mask"].astype(bool)
    for modality in MASK_MODALITIES:
        observed = result[f"{modality}_observed_mask"].astype(bool)
        failure = np.zeros_like(content, dtype=bool)
        result[f"{modality}_failure_mask"] = failure
        result[f"{modality}_evidence_candidate_mask"] = content & observed & ~failure
    result["intensity_bin"] = intensity_bin(result["regression_labels"])
    return result


def _add_text_mapping(
    arrays: dict[str, np.ndarray], *, tokenizer: Any, split: str, max_length: int
) -> list[dict[str, Any]]:
    mapping = build_token_mappings(
        tokenizer=tokenizer,
        split=split,
        sample_ids=arrays["sample_id"],
        raw_text=arrays["raw_text"],
        input_ids=arrays["input_ids"],
        attention_mask=arrays["attention_mask"],
        max_length=max_length,
    )
    arrays["truncation_flag"] = mapping.truncation_flag.astype(bool)
    arrays["tokenizer_exact_match"] = mapping.exact_match.astype(bool)
    arrays["covered_char_end"] = mapping.covered_char_end.astype(np.int32)
    arrays["token_mapping_confidence"] = mapping.confidence.astype(np.float32)
    return mapping.rows


def _stack_attachment4(
    aligned_dir: Path,
    *,
    scaler: Any,
    eps: float,
) -> tuple[dict[str, np.ndarray], list[Path], list[dict[str, Any]]]:
    paths = sorted(aligned_dir.glob("*.pkl"), key=lambda path: path.stem)
    if not paths:
        raise FileNotFoundError(f"No aligned Attachment 4 PKL files in {aligned_dir}")
    records = [_load_pickle(path) for path in paths]
    input_ids, attention, token_types = zip(*[_text_bert_fields(row["text_bert"]) for row in records])
    input_ids_array = np.stack(input_ids)
    attention_array = np.stack(attention)
    token_type_array = np.stack(token_types)
    masks = masks_from_attention(attention_array)
    audio_raw = np.stack([np.asarray(row["audio"], dtype=np.float64) for row in records])
    vision_raw = np.stack([np.asarray(row["vision"], dtype=np.float64) for row in records])
    audio_availability = modality_availability(audio_raw, masks["content"], eps=eps)
    vision_availability = modality_availability(vision_raw, masks["content"], eps=eps)
    audio_scaled, audio_scale_audit = scaler.transform("audio", audio_raw, audio_availability["observed"])
    vision_scaled, vision_scale_audit = scaler.transform("vision", vision_raw, vision_availability["observed"])
    audio_failure = _failure_mask(audio_availability["observed"], masks["content"])
    vision_failure = _failure_mask(vision_availability["observed"], masks["content"])
    text_observed = masks["content"].copy()
    text_failure = np.zeros_like(text_observed)
    arrays = {
        "sample_id": np.asarray([str(row["id"]) for row in records], dtype="U32"),
        "raw_text": np.asarray([str(row["raw_text"]) for row in records], dtype="U1024"),
        "input_ids": input_ids_array.astype(np.int32),
        "attention_mask": attention_array.astype(np.uint8),
        "token_type_ids": token_type_array.astype(np.uint8),
        "content_mask": masks["content"].astype(bool),
        "structural_mask": masks["structural"].astype(bool),
        "padding_mask": masks["padding"].astype(bool),
        "audio": audio_scaled,
        "vision": vision_scaled,
        "text_observed_mask": text_observed,
        "text_natural_zero_mask": np.zeros_like(text_observed),
        "audio_observed_mask": audio_availability["observed"],
        "audio_natural_zero_mask": audio_availability["natural_zero"],
        "vision_observed_mask": vision_availability["observed"],
        "vision_natural_zero_mask": vision_availability["natural_zero"],
        "text_failure_mask": text_failure,
        "audio_failure_mask": audio_failure,
        "vision_failure_mask": vision_failure,
        "text_evidence_candidate_mask": text_observed & ~text_failure,
        "audio_evidence_candidate_mask": audio_availability["observed"] & ~audio_failure,
        "vision_evidence_candidate_mask": vision_availability["observed"] & ~vision_failure,
    }
    scaling_audit = [
        {"modality": "audio", **audio_scale_audit},
        {"modality": "vision", **vision_scale_audit},
    ]
    return arrays, paths, scaling_audit


def _even_positions(mask: np.ndarray, cap: int) -> np.ndarray:
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    if positions.size <= cap:
        return positions
    indexes = np.rint(np.linspace(0, positions.size - 1, cap)).astype(int)
    return positions[indexes]


def _prototype_candidates(
    train: dict[str, np.ndarray], token_rows: list[dict[str, Any]], cap: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    token_lookup = {
        (str(row["sample_id"]), int(row["position"])): str(row["text_span"] or row["token_text"])
        for row in token_rows if row["split"] == "train"
    }
    rows: list[dict[str, Any]] = []
    sample_indexes: list[int] = []
    modality_ids: list[int] = []
    positions_out: list[int] = []
    labels: list[int] = []
    regression: list[float] = []
    bins: list[int] = []
    for sample_index, sample_id in enumerate(train["sample_id"].astype(str)):
        for modality_id, modality in enumerate(MASK_MODALITIES):
            mask = train[f"{modality}_evidence_candidate_mask"][sample_index]
            for position in _even_positions(mask, cap):
                rows.append({
                    "source_split": "train",
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "modality": modality,
                    "modality_id": modality_id,
                    "position": int(position),
                    "classification_label": int(train["classification_labels"][sample_index]),
                    "regression_label": float(train["regression_labels"][sample_index]),
                    "intensity_bin": int(train["intensity_bin"][sample_index]),
                    "token_context": token_lookup.get((sample_id, int(position)), ""),
                })
                sample_indexes.append(sample_index)
                modality_ids.append(modality_id)
                positions_out.append(int(position))
                labels.append(int(train["classification_labels"][sample_index]))
                regression.append(float(train["regression_labels"][sample_index]))
                bins.append(int(train["intensity_bin"][sample_index]))
    arrays = {
        "sample_index": np.asarray(sample_indexes, dtype=np.int32),
        "modality_id": np.asarray(modality_ids, dtype=np.int8),
        "position": np.asarray(positions_out, dtype=np.int16),
        "classification_label": np.asarray(labels, dtype=np.int8),
        "regression_label": np.asarray(regression, dtype=np.float32),
        "intensity_bin": np.asarray(bins, dtype=np.int8),
    }
    return rows, arrays


def _masked_statistics(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(features, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    dimension = array.shape[-1]
    global_median = np.median(array[selected], axis=0).astype(np.float32)
    position_median = np.zeros((array.shape[1], dimension), dtype=np.float32)
    for position in range(array.shape[1]):
        values = array[selected[:, position], position]
        position_median[position] = np.median(values, axis=0) if len(values) else global_median
    return global_median, position_median


def _counterfactual_assets(
    train: dict[str, np.ndarray], valid: dict[str, np.ndarray], config: Problem3Config, root: Path
) -> None:
    statistics: dict[str, np.ndarray] = {}
    for modality in ("audio", "vision"):
        global_median, position_median = _masked_statistics(
            train[modality], train[f"{modality}_evidence_candidate_mask"]
        )
        statistics[f"{modality}_global_median"] = global_median
        statistics[f"{modality}_position_median"] = position_median
    write_npz(root / "counterfactual" / "baseline_statistics.npz", statistics, compressed=True)
    raw = config.section("counterfactual")
    write_json(root / "counterfactual" / "perturbation_templates.json", {
        "scope": "preprocessing templates only; no model inference is performed",
        "principle": "modify non-selected evidence while preserving selected evidence",
        "text": {"replacement": "mask_token", "mask_token_id": int(raw["mask_token_id"])},
        "audio": {"replacement": "training_position_median", "gaussian_noise_std": float(raw["gaussian_noise_std"])},
        "vision": {"replacement": "training_position_median", "time_jitter_steps": int(raw["time_jitter_steps"])},
        "non_evidence_mask_ratio": float(raw["non_evidence_mask_ratio"]),
    })
    repeats = int(raw["validation_repeats"])
    seeds = np.empty((len(valid["sample_id"]), repeats), dtype=np.uint32)
    for sample_index in range(seeds.shape[0]):
        for repeat in range(repeats):
            seeds[sample_index, repeat] = np.random.SeedSequence(
                [config.seed, 3, sample_index, repeat]
            ).generate_state(1, dtype=np.uint32)[0]
    write_npz(root / "counterfactual" / "validation_seeds.npz", {
        "sample_id": valid["sample_id"], "seed": seeds,
    }, compressed=True)


def _quality_rows(
    split: str,
    arrays: dict[str, np.ndarray],
    scaling: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scaling = scaling or {}
    for index, sample_id in enumerate(arrays["sample_id"].astype(str)):
        row: dict[str, Any] = {
            "split": split,
            "sample_id": sample_id,
            "content_steps": int(arrays["content_mask"][index].sum()),
            "truncation_flag": bool(arrays["truncation_flag"][index]),
            "tokenizer_exact_match": bool(arrays["tokenizer_exact_match"][index]),
            "token_mapping_confidence": float(arrays["token_mapping_confidence"][index]),
        }
        for modality in MASK_MODALITIES:
            row[f"{modality}_observed_steps"] = int(arrays[f"{modality}_observed_mask"][index].sum())
            row[f"{modality}_natural_zero_steps"] = int(arrays[f"{modality}_natural_zero_mask"][index].sum())
            row[f"{modality}_failure_steps"] = int(arrays[f"{modality}_failure_mask"][index].sum())
            row[f"{modality}_candidate_steps"] = int(arrays[f"{modality}_evidence_candidate_mask"][index].sum())
        for modality in ("audio", "vision"):
            row[f"{modality}_winsorized_ratio"] = float(scaling.get(modality, {}).get("winsorized_ratio", 0.0))
        rows.append(row)
    return rows


def _feature_drift_rows(
    arrays: dict[str, np.ndarray], raw_features: dict[str, np.ndarray], scaler: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in ("audio", "vision"):
        stats = scaler.modalities[modality]
        for index, sample_id in enumerate(arrays["sample_id"].astype(str)):
            mask = arrays[f"{modality}_observed_mask"][index].astype(bool)
            values = np.asarray(raw_features[modality][index][mask], dtype=np.float64)
            if values.size:
                outside = (values < stats.lower) | (values > stats.upper)
                normalized = (np.clip(values, stats.lower, stats.upper) - stats.center) / stats.scale
                clip_ratio = float((np.abs(normalized) >= scaler.normalized_clip).mean())
                outside_ratio = float(outside.mean())
            else:
                outside_ratio = clip_ratio = 0.0
            rows.append({
                "split": "attachment4", "sample_id": sample_id, "modality": modality,
                "observed_steps": int(mask.sum()), "outside_train_quantile_ratio": outside_ratio,
                "normalized_clip_ratio": clip_ratio,
            })
    return rows


def _leakage_watchlist(
    labeled: dict[str, dict[str, np.ndarray]], attachment4: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, arrays in labeled.items():
        for sample_id, text in zip(arrays["sample_id"].astype(str), arrays["raw_text"].astype(str)):
            index[_normalize_text(text)].append((split, sample_id))
    rows: list[dict[str, Any]] = []
    for sample_id, text in zip(attachment4["sample_id"].astype(str), attachment4["raw_text"].astype(str)):
        for split, matched_id in index.get(_normalize_text(text), []):
            rows.append({
                "attachment4_id": sample_id, "matched_split": split,
                "matched_sample_id": matched_id, "match_type": "normalized_exact_text",
                "policy": "audit_only_never_copy_labels",
            })
    return rows


def _repair_candidates(
    *, config: Problem3Config, attachment4: dict[str, np.ndarray], scaler: Any, root: Path
) -> dict[str, Any]:
    unaligned_dir = config.path("dataset04_unaligned_dir")
    eps = float(config.section("data")["zero_epsilon"])
    requested = str(config.section("repair")["sample_id"])
    rows: list[dict[str, Any]] = []
    healthy_cosines: list[float] = []
    target_artifact: dict[str, np.ndarray] | None = None
    for index, sample_id in enumerate(attachment4["sample_id"].astype(str)):
        path = unaligned_dir / f"{sample_id}.pkl"
        source = _load_pickle(path)
        source_vision = np.asarray(source["vision"], dtype=np.float64)
        reconstructed, inferred_length = reconstruct_to_content(
            source_vision, attachment4["content_mask"][index], eps=eps
        )
        provided_length = int(source.get("vision_lengths", 0))
        reference_mask = attachment4["vision_observed_mask"][index]
        official_scaled = attachment4["vision"][index]
        reconstructed_batch = reconstructed[None, ...]
        reconstructed_observed = attachment4["content_mask"][index][None, :]
        reconstructed_scaled, _ = scaler.transform("vision", reconstructed_batch, reconstructed_observed)
        metrics = reconstruction_metrics(official_scaled, reconstructed_scaled[0], reference_mask)
        official_observed = int(reference_mask.sum())
        severe_official_failure = bool(attachment4["vision_failure_mask"][index].any())
        if not severe_official_failure and official_observed > 0:
            healthy_cosines.append(metrics["median_cosine"])
        rows.append({
            "sample_id": sample_id,
            "provided_vision_length": provided_length,
            "inferred_nonzero_length": inferred_length,
            "official_observed_steps": official_observed,
            "official_failure": severe_official_failure,
            "reconstruction_median_cosine": metrics["median_cosine"],
            "reconstruction_mse": metrics["mse"],
            "is_requested_repair": sample_id == requested,
            "policy": "candidate_only_no_silent_replacement",
        })
        if sample_id == requested:
            target_artifact = {
                "sample_id": np.asarray([sample_id], dtype="U32"),
                "official_scaled_vision": official_scaled[None, ...].astype(np.float32),
                "official_observed_mask": reference_mask[None, ...].astype(bool),
                "repair_candidate_raw_vision": reconstructed[None, ...].astype(np.float32),
                "repair_candidate_scaled_vision": reconstructed_scaled.astype(np.float32),
                "repair_candidate_mask": attachment4["content_mask"][index][None, ...].astype(bool),
                "provided_vision_length": np.asarray([provided_length], dtype=np.int32),
                "inferred_nonzero_length": np.asarray([inferred_length], dtype=np.int32),
            }
    if target_artifact is None:
        raise ValueError(f"Requested repair sample {requested} is absent")
    fieldnames = list(rows[0])
    write_csv(root / "repair" / "length_overrides.csv", rows, fieldnames)
    write_npz(root / "repair" / f"sample_{requested}_vision_candidate.npz", target_artifact, compressed=True)
    median_healthy = float(np.median(healthy_cosines)) if healthy_cosines else 0.0
    threshold = float(config.section("repair")["minimum_median_cosine"])
    audit = {
        "method": "unaligned nonzero-prefix linear resampling to aligned content positions",
        "healthy_sample_count": len(healthy_cosines),
        "healthy_median_of_sample_cosines": median_healthy,
        "required_minimum": threshold,
        "validation_passed": bool(median_healthy >= threshold),
        "deployment_policy": "repair remains a separate candidate regardless of validation result",
    }
    write_json(root / "repair" / "repair_validation.json", audit)
    return audit


def run_preprocessing(config: Problem3Config) -> dict[str, Any]:
    p2_root = config.path("problem2_preprocessed_root")
    aligned_dir = config.path("dataset04_aligned_dir")
    root = config.path("preprocessed_root")
    compressed = bool(config.section("output").get("compressed", True))
    overwrite = bool(config.section("output").get("overwrite", False))
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Problem 3 output already exists at {root}. Set output.overwrite=true only after review."
        )
    ensure_dir(root)
    tokenizer = load_tokenizer(config.section("tokenizer"))
    max_length = int(config.section("tokenizer")["max_length"])
    scaler_path = p2_root / "scaler.json"
    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = scaler_from_dict(scaler_payload)

    labeled: dict[str, dict[str, np.ndarray]] = {}
    token_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    split_audits: dict[str, Any] = {}
    for split in SPLITS:
        arrays = _with_problem3_masks(_load_npz(p2_root / f"{split}.npz"))
        rows = _add_text_mapping(arrays, tokenizer=tokenizer, split=split, max_length=max_length)
        token_rows.extend(rows)
        split_audits[split] = validate_split_arrays(arrays, labeled=True)
        write_npz(root / f"{split}.npz", arrays, compressed=compressed)
        labeled[split] = arrays
        quality_rows.extend(_quality_rows(split, arrays))

    attachment4, p4_paths, p4_scale_audit = _stack_attachment4(
        aligned_dir,
        scaler=scaler,
        eps=float(config.section("data")["zero_epsilon"]),
    )
    p4_token_rows = _add_text_mapping(
        attachment4, tokenizer=tokenizer, split="attachment4", max_length=max_length
    )
    token_rows.extend(p4_token_rows)

    raw_features = {
        "audio": np.stack([np.asarray(_load_pickle(path)["audio"]) for path in p4_paths]),
        "vision": np.stack([np.asarray(_load_pickle(path)["vision"]) for path in p4_paths]),
    }
    metadata_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    token_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in p4_token_rows:
        token_by_id[str(row["sample_id"])].append(row)
    for index, sample_id in enumerate(attachment4["sample_id"].astype(str)):
        video_path = aligned_dir / "videos" / f"{sample_id}.mp4"
        metadata = video_metadata(video_path)
        metadata_rows.append({"sample_id": sample_id, "video_path": str(video_path), **metadata})
        time_rows.extend(proportional_time_mapping(
            sample_id=sample_id,
            token_rows=token_by_id[sample_id],
            content_mask=attachment4["content_mask"][index],
            audio_candidate=attachment4["audio_evidence_candidate_mask"][index],
            vision_candidate=attachment4["vision_evidence_candidate_mask"][index],
            metadata=metadata,
            confidence=float(config.section("mapping")["fallback_confidence"]),
        ))
    attachment4.update({
        "video_path": np.asarray([row["video_path"] for row in metadata_rows], dtype="U1024"),
        "video_readable": np.asarray([row["readable"] for row in metadata_rows], dtype=bool),
        "duration_sec": np.asarray([row["duration_sec"] for row in metadata_rows], dtype=np.float32),
        "fps": np.asarray([row["fps"] for row in metadata_rows], dtype=np.float32),
        "frame_count": np.asarray([row["frame_count"] for row in metadata_rows], dtype=np.int32),
        "video_width": np.asarray([row["width"] for row in metadata_rows], dtype=np.int32),
        "video_height": np.asarray([row["height"] for row in metadata_rows], dtype=np.int32),
    })
    split_audits["attachment4"] = validate_split_arrays(attachment4, labeled=False)
    write_npz(root / "attachment4_aligned.npz", attachment4, compressed=compressed)

    write_csv(root / "mappings" / "token_offsets.csv", token_rows, list(token_rows[0]))
    write_csv(root / "mappings" / "attachment4_time_mapping.csv", time_rows, list(time_rows[0]))
    write_csv(root / "mappings" / "attachment4_video_metadata.csv", metadata_rows, list(metadata_rows[0]))

    prototypes, prototype_arrays = _prototype_candidates(
        labeled["train"], token_rows, int(config.section("prototype")["max_positions_per_sample_modality"])
    )
    write_csv(
        root / "prototypes" / "prototype_source_manifest.csv", prototypes, list(prototypes[0])
    )
    write_npz(root / "prototypes" / "prototype_candidates.npz", prototype_arrays, compressed=True)
    _counterfactual_assets(labeled["train"], labeled["valid"], config, root)

    scale_by_modality = {str(row["modality"]): row for row in p4_scale_audit}
    quality_rows.extend(_quality_rows("attachment4", attachment4, scale_by_modality))
    quality_fields = list(quality_rows[0])
    write_csv(root / "quality" / "sample_quality.csv", quality_rows, quality_fields)
    drift_rows = _feature_drift_rows(attachment4, raw_features, scaler)
    write_csv(root / "quality" / "feature_drift.csv", drift_rows, list(drift_rows[0]))
    truncation_rows = [
        {
            "split": row["split"], "sample_id": row["sample_id"],
            "truncated": bool(arrays["truncation_flag"][index]),
            "covered_char_end": int(arrays["covered_char_end"][index]),
            "text_length_chars": len(str(arrays["raw_text"][index])),
            "tokenizer_exact_match": bool(arrays["tokenizer_exact_match"][index]),
        }
        for row_split, arrays in [*labeled.items(), ("attachment4", attachment4)]
        for index, row in enumerate(
            {"split": row_split, "sample_id": sid} for sid in arrays["sample_id"].astype(str)
        )
    ]
    write_csv(root / "quality" / "truncation_report.csv", truncation_rows, list(truncation_rows[0]))

    leakage = _leakage_watchlist(labeled, attachment4)
    leakage_fields = ["attachment4_id", "matched_split", "matched_sample_id", "match_type", "policy"]
    write_csv(root / "leakage_watchlist.csv", leakage, leakage_fields)
    repair_audit = _repair_candidates(config=config, attachment4=attachment4, scaler=scaler, root=root)

    source_paths = [p2_root / f"{split}.npz" for split in SPLITS] + [scaler_path] + p4_paths
    source_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in source_paths
    ]
    write_csv(root / "source_manifest.csv", source_rows, ["path", "size_bytes", "sha256"])
    write_json(root / "preprocess_config.json", config.raw)
    write_json(root / "scaler_reference.json", {
        "source": str(scaler_path), "sha256": file_sha256(scaler_path),
        "fit_scope": "Problem 2 training observed positions only", "payload": scaler_payload,
    })
    acceptance = validate_saved_root(root, config.section("validation")["expected_split_counts"])
    report = {
        "project": config.section("project"),
        "config_fingerprint": config.fingerprint,
        "independence": "Problem 3 writes only to its own roots; Problem 1/2 artifacts are read-only",
        "problem2_dependency": "schema and training-fitted scaler only",
        "split_audits": split_audits,
        "saved_artifact_acceptance": acceptance,
        "attachment4_scaling": p4_scale_audit,
        "repair_validation": repair_audit,
        "leakage_matches": len(leakage),
        "prototype_candidate_count": len(prototypes),
    }
    write_json(root / "preprocessing_report.json", report)
    write_json(root / "acceptance_audit.json", acceptance)
    return report

