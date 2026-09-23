from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TENSOR_KEYS = (
    "input_ids", "attention_mask", "token_type_ids", "content_mask", "structural_mask", "padding_mask",
    "audio", "vision", "audio_observed_mask", "audio_natural_zero_mask", "audio_missing_mask",
    "vision_observed_mask", "vision_natural_zero_mask", "vision_missing_mask", "modality_reliability",
    "classification_labels", "regression_labels",
)


class Problem2Dataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, mask_bank_path: str | Path | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"预处理数据不存在: {self.path}")
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        required = {
            "sample_id", "input_ids", "attention_mask", "content_mask", "audio", "vision",
            "audio_observed_mask", "audio_natural_zero_mask", "audio_missing_mask",
            "vision_observed_mask", "vision_natural_zero_mask", "vision_missing_mask",
        }
        missing = required.difference(self.arrays)
        if missing:
            raise KeyError(f"{self.path.name} 缺少字段: {sorted(missing)}")
        self.mask_bank: np.ndarray | None = None
        if mask_bank_path is not None:
            bank_path = Path(mask_bank_path)
            if not bank_path.is_file():
                raise FileNotFoundError(f"训练掩码银行不存在: {bank_path}")
            with np.load(bank_path, allow_pickle=False) as archive:
                self.mask_bank = archive["synthetic_missing_mask"]
            if len(self.mask_bank) != len(self):
                raise ValueError("训练数据与掩码银行样本数不一致")

    def __len__(self) -> int:
        return int(len(self.arrays["sample_id"]))

    @property
    def has_labels(self) -> bool:
        return "classification_labels" in self.arrays and "regression_labels" in self.arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {"sample_id": str(self.arrays["sample_id"][index])}
        if "raw_text" in self.arrays:
            result["raw_text"] = str(self.arrays["raw_text"][index])
        for key in TENSOR_KEYS:
            if key in self.arrays:
                result[key] = torch.from_numpy(np.asarray(self.arrays[key][index]))
        if self.mask_bank is not None:
            result["synthetic_missing_mask_bank"] = torch.from_numpy(self.mask_bank[index])
        return result


def make_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
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
        generator=generator,
        drop_last=False,
    )


def _longest_run(mask: torch.Tensor) -> torch.Tensor:
    # T is only 50, so the explicit scan is faster and clearer than CPU round-trips.
    current = torch.zeros(mask.shape[0], device=mask.device, dtype=torch.float32)
    longest = torch.zeros_like(current)
    for step in range(mask.shape[1]):
        current = torch.where(mask[:, step], current + 1.0, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest


def reliability_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    content = batch["content_mask"].bool()
    denominator = content.sum(dim=1).clamp_min(1).float()
    positions = torch.arange(content.shape[1], device=content.device).unsqueeze(0)
    valid_rank = content.cumsum(dim=1).float() / denominator.unsqueeze(1)
    modality_features: list[torch.Tensor] = []
    for modality in ("text", "audio", "vision"):
        if modality == "text":
            observed = content
            natural = torch.zeros_like(content)
            missing = torch.zeros_like(content)
            quality = torch.ones_like(denominator)
        else:
            observed = batch[f"{modality}_observed_mask"].bool() & content
            natural = batch[f"{modality}_natural_zero_mask"].bool() & content
            missing = batch[f"{modality}_missing_mask"].bool() & content
            values = batch[modality].abs().mean(dim=-1)
            quality = (values * observed.float()).sum(dim=1) / observed.sum(dim=1).clamp_min(1)
            quality = torch.tanh(quality)
        observed_ratio = observed.sum(dim=1).float() / denominator
        natural_ratio = natural.sum(dim=1).float() / denominator
        missing_ratio = missing.sum(dim=1).float() / denominator
        begin = (missing & (valid_rank <= 1.0 / 3.0)).sum(dim=1).float() / denominator
        middle = (missing & (valid_rank > 1.0 / 3.0) & (valid_rank <= 2.0 / 3.0)).sum(dim=1).float() / denominator
        end = (missing & (valid_rank > 2.0 / 3.0)).sum(dim=1).float() / denominator
        longest = _longest_run(missing) / denominator
        modality_features.append(torch.stack((observed_ratio, natural_ratio, missing_ratio, quality, begin, middle, end, longest), dim=1))
    return torch.stack(modality_features, dim=1)


def apply_random_mask_view(
    batch: dict[str, Any],
    probability: float,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    if "synthetic_missing_mask_bank" not in batch:
        raise KeyError("训练批次缺少 synthetic_missing_mask_bank")
    output = dict(batch)
    bank = batch["synthetic_missing_mask_bank"].bool()  # [B,K,3,T]
    batch_size, bank_size = bank.shape[:2]
    choices = torch.randint(bank_size, (batch_size,), generator=generator, device=bank.device)
    selected = bank[torch.arange(batch_size, device=bank.device), choices]
    use_missing = torch.rand(batch_size, generator=generator, device=bank.device) < probability
    selected = selected & use_missing[:, None, None]
    for channel, modality in ((1, "audio"), (2, "vision")):
        artificial = selected[:, channel] & batch["content_mask"].bool()
        prior_missing = batch[f"{modality}_missing_mask"].bool()
        total_missing = prior_missing | artificial
        observed = batch[f"{modality}_observed_mask"].bool() & ~artificial
        values = batch[modality].clone()
        values = values.masked_fill(total_missing.unsqueeze(-1), 0.0)
        output[modality] = values
        output[f"{modality}_observed_mask"] = observed
        output[f"{modality}_missing_mask"] = total_missing
    output["modality_reliability"] = reliability_features(output)
    return output


def apply_fixed_scenario(
    batch: dict[str, Any],
    pattern: str,
    rate: float,
    position: str,
) -> dict[str, Any]:
    if pattern not in {"audio", "vision", "audio_vision", "none"}:
        raise ValueError(f"未知缺失类型: {pattern}")
    if position not in {"begin", "middle", "end"}:
        raise ValueError(f"未知缺失位置: {position}")
    output = dict(batch)
    modalities = [] if pattern == "none" else (["audio", "vision"] if pattern == "audio_vision" else [pattern])
    content = batch["content_mask"].bool()
    for modality in modalities:
        artificial = torch.zeros_like(content)
        base_observed = batch[f"{modality}_observed_mask"].bool() & content
        for row in range(content.shape[0]):
            eligible = torch.where(base_observed[row])[0]
            if eligible.numel() == 0:
                continue
            length = min(max(int(round(float(content[row].sum()) * rate)), 1), max(int(eligible.numel()) - 1, 1))
            if position == "begin":
                start = 0
            elif position == "end":
                start = max(int(eligible.numel()) - length, 0)
            else:
                start = max((int(eligible.numel()) - length) // 2, 0)
            artificial[row, eligible[start:start + length]] = True
        total_missing = batch[f"{modality}_missing_mask"].bool() | artificial
        output[modality] = batch[modality].masked_fill(total_missing.unsqueeze(-1), 0.0)
        output[f"{modality}_observed_mask"] = batch[f"{modality}_observed_mask"].bool() & ~artificial
        output[f"{modality}_missing_mask"] = total_missing
    output["modality_reliability"] = reliability_features(output)
    return output


def iter_loader(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield from loader
