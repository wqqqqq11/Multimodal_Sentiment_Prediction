from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


TENSOR_KEYS = (
    "input_ids", "attention_mask", "token_type_ids", "content_mask", "structural_mask", "padding_mask",
    "audio", "vision", "text_observed_mask", "audio_observed_mask", "vision_observed_mask",
    "text_natural_zero_mask", "audio_natural_zero_mask", "vision_natural_zero_mask",
    "text_failure_mask", "audio_failure_mask", "vision_failure_mask",
    "text_evidence_candidate_mask", "audio_evidence_candidate_mask", "vision_evidence_candidate_mask",
    "classification_labels", "regression_labels", "truncation_flag", "token_mapping_confidence",
    "duration_sec", "fps", "frame_count", "video_readable",
)


class Problem3Dataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"问题三预处理数据不存在: {self.path}")
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        required = {
            "sample_id", "raw_text", "input_ids", "attention_mask", "token_type_ids",
            "content_mask", "structural_mask", "padding_mask", "audio", "vision",
            "text_evidence_candidate_mask", "audio_evidence_candidate_mask", "vision_evidence_candidate_mask",
        }
        missing = required.difference(self.arrays)
        if missing:
            raise KeyError(f"{self.path.name} 缺少字段: {sorted(missing)}")
        n = len(self.arrays["sample_id"])
        if self.arrays["input_ids"].shape != (n, 50):
            raise ValueError(f"{self.path.name} 文本形状不是 (N,50)")
        if self.arrays["audio"].shape != (n, 50, 74) or self.arrays["vision"].shape != (n, 50, 35):
            raise ValueError(f"{self.path.name} 音视频形状与 aligned_50 不一致")

    def __len__(self) -> int:
        return int(len(self.arrays["sample_id"]))

    @property
    def has_labels(self) -> bool:
        return "classification_labels" in self.arrays and "regression_labels" in self.arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "sample_id": str(self.arrays["sample_id"][index]),
            "raw_text": str(self.arrays["raw_text"][index]),
        }
        if "video_path" in self.arrays:
            item["video_path"] = str(self.arrays["video_path"][index])
        for key in TENSOR_KEYS:
            if key in self.arrays:
                item[key] = torch.from_numpy(np.asarray(self.arrays[key][index]))
        return item


def class_balanced_weights(dataset: Problem3Dataset) -> torch.Tensor:
    if not dataset.has_labels:
        raise ValueError("无标签数据不能构造类别均衡采样器")
    labels = np.asarray(dataset.arrays["classification_labels"], dtype=np.int64)
    counts = np.bincount(labels, minlength=3).clip(min=1)
    return torch.as_tensor(1.0 / counts[labels], dtype=torch.double)


def make_loader(
    dataset: Problem3Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    balanced: bool = False,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if balanced:
        sampler = WeightedRandomSampler(
            class_balanced_weights(dataset), num_samples=len(dataset), replacement=True, generator=generator
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        generator=generator,
        drop_last=False,
    )


def load_baselines(path: str | Path) -> dict[str, torch.Tensor]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"积分梯度基线不存在: {source}")
    with np.load(source, allow_pickle=False) as archive:
        result = {key: torch.from_numpy(np.asarray(archive[key], dtype=np.float32)) for key in archive.files}
    required = {
        "audio_global_median", "audio_position_median", "vision_global_median", "vision_position_median"
    }
    missing = required.difference(result)
    if missing:
        raise KeyError(f"积分梯度基线缺少字段: {sorted(missing)}")
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("积分梯度基线包含 NaN 或 Inf")
    return result
