from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _transform(values: np.ndarray, mask: np.ndarray, mean: np.ndarray, components: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = np.zeros((values.shape[0], values.shape[1], components.shape[1]), dtype=np.float32)
    selected = np.asarray(values[mask], dtype=np.float32)
    result[mask] = np.clip(((selected - mean) @ components) / scale, -8.0, 8.0).astype(np.float32)
    return result


def build_text_features(root: Path, dimensions: int = 128, overwrite: bool = False) -> Path:
    source_path = root / "datasets/original_data_from_the_competition_organizer/dataset02/aligned_50.pkl"
    p3_root = root / "datasets/preprocessed_data/problem3"
    attachment_dir = root / "datasets/original_data_from_the_competition_organizer/dataset04/aligned"
    output = p3_root / "text_features"
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"文本特征已存在: {output}；如需重建请使用 --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    dataset = _load_pickle(source_path)
    base: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid", "test"):
        with np.load(p3_root / f"{split}.npz", allow_pickle=False) as archive:
            base[split] = {key: archive[key] for key in ("sample_id", "content_mask", "text_evidence_candidate_mask")}
        source_ids = np.asarray(dataset[split]["id"]).astype(str)
        if not np.array_equal(source_ids, base[split]["sample_id"].astype(str)):
            raise ValueError(f"{split} 的原始文本特征与问题三预处理ID顺序不一致")
    train_values = np.asarray(dataset["train"]["text"], dtype=np.float32)
    train_mask = base["train"]["content_mask"].astype(bool)
    selected = train_values[train_mask]
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = selected - mean
    covariance = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: int(dimensions)]
    components = eigenvectors[:, order].astype(np.float32)
    explained = eigenvalues[order].clip(min=1e-8).astype(np.float32)
    scale = np.sqrt(explained).astype(np.float32)
    total_variance = float(np.maximum(eigenvalues, 0).sum())
    explained_ratio = float(np.maximum(eigenvalues[order], 0).sum() / max(total_variance, 1e-12))
    for split in ("train", "valid", "test"):
        values = np.asarray(dataset[split]["text"], dtype=np.float32)
        transformed = _transform(values, base[split]["content_mask"].astype(bool), mean, components, scale)
        np.savez_compressed(output / f"{split}_text.npz", sample_id=base[split]["sample_id"], text_features=transformed)
    attachment_paths = sorted(attachment_dir.glob("*.pkl"), key=lambda path: path.stem)
    records = [_load_pickle(path) for path in attachment_paths]
    with np.load(p3_root / "attachment4_aligned.npz", allow_pickle=False) as archive:
        attachment_ids = archive["sample_id"].astype(str)
        attachment_mask = archive["content_mask"].astype(bool)
    record_ids = np.asarray([str(record["id"]) for record in records])
    if not np.array_equal(record_ids, attachment_ids):
        raise ValueError("附件4原始文本特征与问题三预处理ID顺序不一致")
    attachment_values = np.stack([np.asarray(record["text"], dtype=np.float32) for record in records])
    attachment_features = _transform(attachment_values, attachment_mask, mean, components, scale)
    np.savez_compressed(output / "attachment4_text.npz", sample_id=attachment_ids, text_features=attachment_features)
    train_text_features = np.load(output / "train_text.npz", allow_pickle=False)["text_features"]
    candidate = base["train"]["text_evidence_candidate_mask"].astype(bool)
    global_median = np.median(train_text_features[candidate], axis=0).astype(np.float32)
    position_median = np.zeros((50, int(dimensions)), dtype=np.float32)
    for position in range(50):
        values = train_text_features[candidate[:, position], position]
        position_median[position] = np.median(values, axis=0) if len(values) else global_median
    np.savez_compressed(
        output / "text_baselines.npz",
        text_global_median=global_median,
        text_position_median=position_median,
    )
    np.savez_compressed(output / "pca_state.npz", mean=mean, components=components, scale=scale)
    report = {
        "source": str(source_path),
        "fit_scope": "attachment2 train content positions only",
        "input_dimension": 768,
        "output_dimension": int(dimensions),
        "fit_positions": int(train_mask.sum()),
        "explained_variance_ratio": explained_ratio,
        "splits": {split: int(len(base[split]["sample_id"])) for split in base},
        "attachment4": int(len(attachment_ids)),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="构造问题三紧凑PCA文本特征")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = build_text_features(PROJECT_ROOT, args.dimensions, args.overwrite)
    print(f"[OK] 文本特征已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
