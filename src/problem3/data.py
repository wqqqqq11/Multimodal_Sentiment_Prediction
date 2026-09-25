from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker


TENSOR_FIELDS = (
    "input_ids", "attention_mask", "token_type_ids", "content_mask", "structural_mask", "padding_mask",
    "audio", "vision", "text_observed_mask", "audio_observed_mask", "vision_observed_mask",
    "text_failure_mask", "audio_failure_mask", "vision_failure_mask",
    "text_evidence_candidate_mask", "audio_evidence_candidate_mask", "vision_evidence_candidate_mask",
    "classification_labels", "regression_labels", "intensity_bin", "truncation_flag",
    "tokenizer_exact_match", "token_mapping_confidence", "duration_sec", "fps", "frame_count",
)


class Problem3Dataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"数据文件不存在: {self.path}")
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: source[key] for key in source.files}
        required = {
            "sample_id", "raw_text", "input_ids", "attention_mask", "audio", "vision", "content_mask",
            "text_evidence_candidate_mask", "audio_evidence_candidate_mask", "vision_evidence_candidate_mask",
        }
        missing = required.difference(self.arrays)
        if missing:
            raise ValueError(f"{self.path.name} 缺少字段: {sorted(missing)}")
        length = len(self.arrays["sample_id"])
        if any(len(value) != length for value in self.arrays.values()):
            raise ValueError(f"{self.path.name} 各字段样本数不一致")

    def __len__(self) -> int:
        return len(self.arrays["sample_id"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "sample_id": str(self.arrays["sample_id"][index]),
            "raw_text": str(self.arrays["raw_text"][index]),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
        for key in TENSOR_FIELDS:
            if key not in self.arrays:
                continue
            value = self.arrays[key][index]
            tensor = torch.as_tensor(value)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            item[key] = tensor
        return item

    @property
    def labeled(self) -> bool:
        return "classification_labels" in self.arrays and "regression_labels" in self.arrays


def build_dataloader(
    dataset: Dataset[dict[str, Any]], *, batch_size: int, shuffle: bool, num_workers: int,
    pin_memory: bool, seed: int, drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=drop_last,
    )


def load_splits(cfg: dict[str, Any]) -> dict[str, Problem3Dataset]:
    root = Path(cfg["paths"]["preprocessed_root"])
    files = {
        "train": cfg["data"]["train_file"],
        "valid": cfg["data"]["valid_file"],
        "test": cfg["data"]["test_file"],
        "attachment4": cfg["data"]["attachment4_file"],
    }
    return {name: Problem3Dataset(root / filename) for name, filename in files.items()}
