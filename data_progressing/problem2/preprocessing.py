"""End-to-end preprocessing for mask-aware gating and consistency distillation."""

from __future__ import annotations

import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import Problem2Config
from .io import ensure_dir, file_sha256, write_csv, write_json, write_npz
from .masking import (
    ContinuousSpanMaskGenerator,
    mask_to_runs,
    masks_from_attention,
    missingness_config,
    modality_availability,
)
from .quality import RELIABILITY_COLUMNS, balanced_class_weights, expected_class_from_regression, reliability_tensor
from .scaling import RobustFeatureScaler
from .validation import validate_mask_bank, validate_saved_root, validate_split_arrays


LOGGER = logging.getLogger(__name__)


def _load_pickle(path: Path) -> Any:
    LOGGER.info("Loading %s", path.name)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _text_fields(text_bert: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(text_bert)
    if array.ndim != 3 or array.shape[1:] != (3, 50):
        raise ValueError(f"text_bert must have shape (N,3,50), got {array.shape}")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-4):
        raise ValueError("text_bert contains non-integral token metadata")
    input_ids = rounded[:, 0].astype(np.int32)
    attention = (rounded[:, 1] > 0).astype(bool)
    token_types = rounded[:, 2].astype(np.int8)
    if np.any(input_ids < 0):
        raise ValueError("input_ids must be non-negative")
    return input_ids, attention, token_types


def _base_masks(block: dict[str, Any], eps: float) -> dict[str, np.ndarray]:
    _, attention, _ = _text_fields(np.asarray(block["text_bert"]))
    timeline = masks_from_attention(attention)
    audio = modality_availability(np.asarray(block["audio"]), timeline["content"], eps=eps)
    vision = modality_availability(np.asarray(block["vision"]), timeline["content"], eps=eps)
    return {**timeline, "audio_observed": audio["observed"], "audio_natural": audio["natural_zero"],
            "vision_observed": vision["observed"], "vision_natural": vision["natural_zero"]}


def _privileged_text_pool(block: dict[str, Any], content: np.ndarray) -> np.ndarray:
    """Pool organizer-provided BERT representations for training-only privileged supervision."""
    if "text" not in block:
        raise ValueError("Attachment 2 labeled split is missing organizer-provided `text`")
    text = np.asarray(block["text"], dtype=np.float32)
    if text.ndim != 3 or text.shape[:2] != content.shape or text.shape[2] != 768:
        raise ValueError(f"text must have shape (N,50,768), got {text.shape}")
    if not np.isfinite(text).all():
        raise ValueError("organizer-provided text contains NaN or Inf")
    denominator = np.maximum(content.sum(axis=1, keepdims=True), 1)
    return ((text * content[..., None]).sum(axis=1) / denominator).astype(np.float32)


def _build_privileged_initialization(
    train_block: dict[str, Any],
    content: np.ndarray,
    *,
    vocab_size: int,
    hidden_dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Compress contextual BERT vectors into a small organizer-only token initialization."""
    text = np.asarray(train_block["text"], dtype=np.float32)
    input_ids, _, _ = _text_fields(np.asarray(train_block["text_bert"]))
    rng = np.random.default_rng(seed + 731)
    gaussian = rng.normal(size=(text.shape[-1], hidden_dim)).astype(np.float64)
    projection, _ = np.linalg.qr(gaussian, mode="reduced")
    projection = projection.astype(np.float32)
    projected = np.einsum("ntd,dh->nth", text, projection, optimize=True)
    flat_ids = input_ids[content].astype(np.int64)
    flat_vectors = projected[content]
    if np.any(flat_ids >= vocab_size):
        raise ValueError("text_bert token id exceeds configured privileged vocabulary")
    prototypes = np.zeros((vocab_size, hidden_dim), dtype=np.float64)
    raw_prototypes = np.zeros((vocab_size, text.shape[-1]), dtype=np.float64)
    counts = np.bincount(flat_ids, minlength=vocab_size).astype(np.int64)
    for dimension in range(hidden_dim):
        prototypes[:, dimension] = np.bincount(
            flat_ids, weights=flat_vectors[:, dimension], minlength=vocab_size
        )
    flat_raw = text[content]
    for dimension in range(text.shape[-1]):
        raw_prototypes[:, dimension] = np.bincount(
            flat_ids, weights=flat_raw[:, dimension], minlength=vocab_size
        )
    seen = counts > 0
    prototypes[seen] /= counts[seen, None]
    raw_prototypes[seen] /= counts[seen, None]
    return {
        "projection": projection,
        "token_prototypes": prototypes.astype(np.float32),
        "token_prototypes_raw": raw_prototypes.astype(np.float16),
        "token_counts": counts,
    }


def _prepare_labeled_split(
    split: str,
    block: dict[str, Any],
    scaler: RobustFeatureScaler,
    eps: float,
    severe_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    ids = np.asarray(block["id"]).astype(str)
    raw_text = np.asarray(block["raw_text"]).astype(str)
    input_ids, attention, token_types = _text_fields(np.asarray(block["text_bert"]))
    masks = _base_masks(block, eps)
    audio, audio_audit = scaler.transform("audio", np.asarray(block["audio"]), masks["audio_observed"])
    vision, vision_audit = scaler.transform("vision", np.asarray(block["vision"]), masks["vision_observed"])
    labels = np.asarray(block["classification_labels"]).reshape(-1).astype(np.int64)
    regression = np.asarray(block["regression_labels"]).reshape(-1).astype(np.float32)
    expected = expected_class_from_regression(regression)
    mismatch = labels != expected
    missing = np.zeros_like(masks["content"], dtype=bool)
    privileged_text = _privileged_text_pool(block, masks["content"])
    reliability = reliability_tensor(
        masks["content"], masks["audio_observed"], masks["audio_natural"],
        masks["vision_observed"], masks["vision_natural"],
    )
    arrays = {
        "sample_id": ids,
        "raw_text": raw_text,
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": token_types,
        "content_mask": masks["content"],
        "structural_mask": masks["structural"],
        "padding_mask": masks["padding"],
        "text_observed_mask": masks["content"].copy(),
        "text_natural_zero_mask": missing.copy(),
        "text_missing_mask": missing.copy(),
        "privileged_text": privileged_text,
        "audio": audio,
        "vision": vision,
        "audio_observed_mask": masks["audio_observed"],
        "audio_natural_zero_mask": masks["audio_natural"],
        "audio_missing_mask": missing.copy(),
        "vision_observed_mask": masks["vision_observed"],
        "vision_natural_zero_mask": masks["vision_natural"],
        "vision_missing_mask": missing.copy(),
        "modality_reliability": reliability,
        "classification_labels": labels,
        "regression_labels": regression,
    }
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        content_steps = int(masks["content"][index].sum())
        audio_zero = int(masks["audio_natural"][index].sum())
        vision_zero = int(masks["vision_natural"][index].sum())
        denominator = max(content_steps, 1)
        rows.append({
            "split": split,
            "sample_id": sample_id,
            "content_steps": content_steps,
            "audio_natural_zero_steps": audio_zero,
            "vision_natural_zero_steps": vision_zero,
            "audio_natural_zero_ratio": audio_zero / denominator,
            "vision_natural_zero_ratio": vision_zero / denominator,
            "empty_text_content": content_steps == 0,
            "label_disagreement": bool(mismatch[index]),
            "severe_audio_unavailable": audio_zero / denominator >= severe_threshold,
            "severe_vision_unavailable": vision_zero / denominator >= severe_threshold,
        })
    audit = {
        "sample_count": int(len(ids)),
        "duplicate_ids": int(len(ids) - len(np.unique(ids))),
        "label_disagreements": int(mismatch.sum()),
        "audio": audio_audit,
        "vision": vision_audit,
        "audio_natural_zero_steps": int(masks["audio_natural"].sum()),
        "vision_natural_zero_steps": int(masks["vision_natural"].sum()),
    }
    validate_split_arrays(arrays, labeled=True)
    return arrays, audit, rows


def _load_challenge(folder: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    blocks: dict[str, list[np.ndarray]] = {"text_bert": [], "audio": [], "vision": []}
    names: list[str] = []
    for path in sorted(folder.glob("*.pkl")):
        sample = _load_pickle(path)
        if set(sample) == {"test"} and isinstance(sample["test"], dict):
            sample = sample["test"]
        if not all(name in sample for name in blocks):
            raise ValueError(f"{path.name}: expected text_bert/audio/vision")
        for name in blocks:
            value = np.asarray(sample[name])
            if value.shape[0] != 1:
                raise ValueError(f"{path.name}/{name}: each challenge file must contain exactly one sample")
            blocks[name].append(value)
        names.append(path.stem)
    if not names:
        raise ValueError(f"No challenge pickle files found in {folder}")
    return {name: np.concatenate(values, axis=0) for name, values in blocks.items()}, names


def _prepare_challenge(
    block: dict[str, np.ndarray],
    names: list[str],
    scaler: RobustFeatureScaler,
    eps: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    input_ids, attention, token_types = _text_fields(block["text_bert"])
    timeline = masks_from_attention(attention)
    audio_state = modality_availability(block["audio"], timeline["content"], eps=eps)
    vision_state = modality_availability(block["vision"], timeline["content"], eps=eps)
    # Attachment 3 omits source IDs and complete counterparts for most files. Internal zero
    # steps therefore mean "unavailable" at inference; natural vs injected causes cannot be
    # identified without leaking unavailable information.
    audio_missing = audio_state["natural_zero"]
    vision_missing = vision_state["natural_zero"]
    natural = np.zeros_like(timeline["content"], dtype=bool)
    audio, audio_audit = scaler.transform("audio", block["audio"], audio_state["observed"])
    vision, vision_audit = scaler.transform("vision", block["vision"], vision_state["observed"])
    reliability = reliability_tensor(
        timeline["content"], audio_state["observed"], natural,
        vision_state["observed"], natural,
        audio_missing=audio_missing, vision_missing=vision_missing,
    )
    arrays = {
        "sample_id": np.asarray(names, dtype=str),
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": token_types,
        "content_mask": timeline["content"],
        "structural_mask": timeline["structural"],
        "padding_mask": timeline["padding"],
        "text_observed_mask": timeline["content"].copy(),
        "text_natural_zero_mask": natural.copy(),
        "text_missing_mask": natural.copy(),
        "audio": audio,
        "vision": vision,
        "audio_observed_mask": audio_state["observed"],
        "audio_natural_zero_mask": natural.copy(),
        "audio_missing_mask": audio_missing,
        "vision_observed_mask": vision_state["observed"],
        "vision_natural_zero_mask": natural.copy(),
        "vision_missing_mask": vision_missing,
        "modality_reliability": reliability,
    }
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        content_steps = int(timeline["content"][index].sum())
        denominator = max(content_steps, 1)
        audio_runs = mask_to_runs(audio_missing[index])
        vision_runs = mask_to_runs(vision_missing[index])
        rows.append({
            "sample_id": name,
            "content_steps": content_steps,
            "audio_missing_steps": int(audio_missing[index].sum()),
            "audio_missing_ratio": float(audio_missing[index].sum() / denominator),
            "audio_span_count": len(audio_runs),
            "audio_longest_span": max((end - start for start, end in audio_runs), default=0),
            "vision_missing_steps": int(vision_missing[index].sum()),
            "vision_missing_ratio": float(vision_missing[index].sum() / denominator),
            "vision_span_count": len(vision_runs),
            "vision_longest_span": max((end - start for start, end in vision_runs), default=0),
        })
    audit = {
        "sample_count": len(names),
        "zero_semantics": "effective unavailability; injected and natural causes are not identifiable without complete counterparts",
        "audio": audio_audit,
        "vision": vision_audit,
        "audio_missing_steps": int(audio_missing.sum()),
        "vision_missing_steps": int(vision_missing.sum()),
    }
    validate_split_arrays(arrays, labeled=False)
    return arrays, audit, rows


def _mask_bank_report(bank: dict[str, np.ndarray]) -> dict[str, Any]:
    ratios = bank["actual_missing_ratio"]
    patterns = bank["pattern_id"]
    return {
        "bank_size": int(ratios.shape[1]),
        "pattern_counts": {
            "audio": int((patterns == 1).sum()),
            "vision": int((patterns == 2).sum()),
            "audio_vision": int((patterns == 3).sum()),
        },
        "audio_mean_ratio_when_selected": float(ratios[:, :, 1][ratios[:, :, 1] > 0].mean()),
        "vision_mean_ratio_when_selected": float(ratios[:, :, 2][ratios[:, :, 2] > 0].mean()),
    }


def run_preprocessing(cfg: Problem2Config, *, overwrite: bool = False) -> dict[str, Any]:
    source_path = cfg.path("dataset02_aligned")
    challenge_dir = cfg.path("dataset03_aligned_dir")
    output_root = ensure_dir(cfg.path("preprocessed_root"))
    report_root = ensure_dir(cfg.path("output_root") / "preprocessing")
    expected = [output_root / f"{split}.npz" for split in ("train", "valid", "test")]
    expected += [output_root / "challenge_aligned.npz", output_root / "train_mask_bank.npz"]
    allow_overwrite = overwrite or bool(cfg.section("output").get("overwrite", False))
    existing = [path for path in expected if path.exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(f"Preprocessed artifacts already exist; pass --overwrite: {existing[0]}")

    data = _load_pickle(source_path)
    if list(data) != ["train", "valid", "test"]:
        raise ValueError("aligned_50.pkl must contain train/valid/test in organizer order")
    eps = float(cfg.section("data")["zero_epsilon"])
    train_masks = _base_masks(data["train"], eps)
    scaling_cfg = cfg.section("scaling")
    scaler = RobustFeatureScaler(
        lower_quantile=float(scaling_cfg["lower_quantile"]),
        upper_quantile=float(scaling_cfg["upper_quantile"]),
        minimum_scale=float(scaling_cfg["minimum_scale"]),
        normalized_clip=float(scaling_cfg["normalized_clip"]),
    )
    scaler.fit("audio", np.asarray(data["train"]["audio"]), train_masks["audio_observed"])
    scaler.fit("vision", np.asarray(data["train"]["vision"]), train_masks["vision_observed"])
    write_json(output_root / "scaler.json", scaler.to_dict())

    privileged_cfg = cfg.section("privileged_text")
    privileged_init = _build_privileged_initialization(
        data["train"], train_masks["content"],
        vocab_size=int(privileged_cfg["vocab_size"]),
        hidden_dim=int(privileged_cfg["hidden_dim"]),
        seed=cfg.seed,
    )
    write_npz(
        output_root / str(privileged_cfg["initialization_file"]),
        privileged_init,
        compressed=True,
    )

    compressed = bool(cfg.section("output").get("compressed", True))
    split_audits: dict[str, Any] = {}
    quality_rows: list[dict[str, Any]] = []
    prepared_train: dict[str, np.ndarray] | None = None
    quality_cfg = cfg.section("quality")
    severe_threshold = float(quality_cfg["severe_natural_zero_ratio"])
    for split in ("train", "valid", "test"):
        arrays, audit, rows = _prepare_labeled_split(
            split, data[split], scaler, eps, severe_threshold
        )
        if bool(quality_cfg.get("reject_label_disagreement", True)) and audit["label_disagreements"]:
            raise ValueError(f"{split}: classification/regression label disagreement detected")
        write_npz(output_root / f"{split}.npz", arrays, compressed=compressed)
        split_audits[split] = audit
        quality_rows.extend(rows)
        if split == "train":
            prepared_train = arrays
        LOGGER.info("Prepared %s: %d samples", split, len(arrays["sample_id"]))

    if prepared_train is None:
        raise RuntimeError("Training split was not prepared")
    generator = ContinuousSpanMaskGenerator(missingness_config(cfg.section("missingness")), cfg.seed)
    bank = generator.generate(prepared_train["audio_observed_mask"], prepared_train["vision_observed_mask"])
    validate_mask_bank(bank, prepared_train)
    write_npz(output_root / "train_mask_bank.npz", bank, compressed=compressed)

    challenge_block, challenge_names = _load_challenge(challenge_dir)
    challenge_arrays, challenge_audit, challenge_rows = _prepare_challenge(
        challenge_block, challenge_names, scaler, eps
    )
    expected_challenge = int(cfg.section("validation")["expected_dataset03_samples"])
    if len(challenge_names) != expected_challenge:
        raise ValueError(f"Expected {expected_challenge} aligned challenge files, found {len(challenge_names)}")
    write_npz(output_root / "challenge_aligned.npz", challenge_arrays, compressed=compressed)

    quality_fields = list(quality_rows[0])
    write_csv(output_root / "sample_quality.csv", quality_rows, quality_fields)
    write_csv(output_root / "challenge_manifest.csv", challenge_rows, list(challenge_rows[0]))
    train_labels = np.asarray(data["train"]["classification_labels"]).astype(np.int64)
    report = {
        "project": cfg.section("project"),
        "config_fingerprint": cfg.fingerprint,
        "source": {
            "dataset02_aligned": source_path.relative_to(cfg.project_root).as_posix(),
            "dataset02_sha256": file_sha256(source_path),
            "dataset03_aligned_dir": challenge_dir.relative_to(cfg.project_root).as_posix(),
        },
        "design": {
            "text_interface": "text_bert token ids, attention mask and token type ids",
            "privileged_text": "Attachment 2 organizer text pooled for training-only teacher; compressed token prototypes initialize the deployable student",
            "scaled_modalities": ["audio", "vision"],
            "scaler_fit_scope": "Attachment 2 training split, observed content steps only",
            "sample_policy": "retain all samples; flag severe unavailability instead of deleting",
            "mask_semantics": ["padding", "structural CLS/SEP", "observed", "natural zero", "synthetic/detected missing"],
            "reliability_columns": list(RELIABILITY_COLUMNS),
        },
        "splits": split_audits,
        "challenge": challenge_audit,
        "augmentation": _mask_bank_report(bank),
        "class_weights": balanced_class_weights(train_labels).tolist(),
    }
    clip_warning = float(quality_cfg["warn_clipped_element_ratio"])
    report["warnings"] = [
        f"{split}/{modality}: winsorized ratio {split_audits[split][modality]['winsorized_ratio']:.4f} exceeds {clip_warning:.4f}"
        for split in ("train", "valid", "test")
        for modality in ("audio", "vision")
        if split_audits[split][modality]["winsorized_ratio"] > clip_warning
    ]
    write_json(output_root / "preprocessing_report.json", report)
    write_json(report_root / "preprocessing_report.json", report)
    readme = "# Problem 2 preprocessed data\n\n"
    readme += "Primary text input is `text_bert`; this is the only text representation shared by Attachment 2 and Attachment 3.\n\n"
    readme += "Attachment 2 labeled splits additionally contain training-only `privileged_text`; `text_privileged_init.npz` compresses organizer BERT semantics into the compact student vocabulary. Neither is required by Attachment 3 inference.\n\n"
    readme += "- `train.npz`, `valid.npz`, `test.npz`: clean teacher inputs, labels, base masks and gate reliability features.\n"
    readme += "- `train_mask_bank.npz`: deterministic continuous-span masks for missing-student views. Index `[sample, view]` pairs with the same clean training sample.\n"
    readme += "- `challenge_aligned.npz`: all 30 Attachment 3 aligned samples with detected unavailable intervals.\n"
    readme += "- `scaler.json`: audio/vision statistics fit only on observed training steps.\n"
    readme += "- `sample_quality.csv` and `challenge_manifest.csv`: sample-level audit trails.\n\n"
    readme += "Unavailable feature slots remain exact zero after transformation. Apply each mask-bank view to the clean sample at training time so the same sample supplies the teacher and student targets.\n"
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    del data, challenge_block
    gc.collect()

    acceptance = validate_saved_root(output_root)
    write_json(output_root / "acceptance_audit.json", acceptance)
    write_json(report_root / "acceptance_audit.json", acceptance)
    LOGGER.info("Problem 2 preprocessing acceptance checks passed")
    return {"report": report, "acceptance": acceptance, "output_root": str(output_root)}
