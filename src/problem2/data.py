from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TENSOR_KEYS = (
    "input_ids", "attention_mask", "token_type_ids", "content_mask", "structural_mask", "padding_mask",
    "text_observed_mask", "text_natural_zero_mask", "text_missing_mask", "privileged_text",
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
            "text_observed_mask", "text_natural_zero_mask", "text_missing_mask",
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


def _segment_count(mask: torch.Tensor) -> torch.Tensor:
    previous = torch.cat((torch.zeros_like(mask[:, :1]), mask[:, :-1]), dim=1)
    return (mask & ~previous).sum(dim=1).float()


def reliability_features(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    content = batch["content_mask"].bool()
    denominator = content.sum(dim=1).clamp_min(1).float()
    positions = torch.arange(content.shape[1], device=content.device).unsqueeze(0)
    valid_rank = content.cumsum(dim=1).float() / denominator.unsqueeze(1)
    modality_features: list[torch.Tensor] = []
    for modality in ("text", "audio", "vision"):
        if modality == "text":
            observed = batch["text_observed_mask"].bool() & content
            natural = batch["text_natural_zero_mask"].bool() & content
            missing = batch["text_missing_mask"].bool() & content
            quality = observed.sum(dim=1).float() / denominator
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
        segments = _segment_count(missing) / denominator
        longest = _longest_run(missing) / denominator
        modality_features.append(
            torch.stack((observed_ratio, natural_ratio, missing_ratio, quality, begin, middle, end, segments, longest), dim=1)
        )
    return torch.stack(modality_features, dim=1)


def apply_random_mask_view(
    batch: dict[str, Any],
    probability: float,
    generator: torch.Generator | None = None,
    synchronized_probability: float = 0.0,
    trimodal_probability: float = 0.0,
    synchronized_rate_min: float = 0.20,
    synchronized_rate_max: float = 0.40,
    multi_span_probability: float = 0.0,
    multi_span_count_min: int = 2,
    multi_span_count_max: int = 6,
    multi_span_length_max: int = 4,
) -> dict[str, Any]:
    if "synthetic_missing_mask_bank" not in batch:
        raise KeyError("训练批次缺少 synthetic_missing_mask_bank")
    output = dict(batch)
    bank = batch["synthetic_missing_mask_bank"].bool()  # [B,K,3,T]
    batch_size, bank_size = bank.shape[:2]
    choices = torch.randint(bank_size, (batch_size,), generator=generator, device=bank.device)
    selected = bank[torch.arange(batch_size, device=bank.device), choices]
    use_missing = torch.rand(batch_size, generator=generator, device=bank.device) < probability
    kind_draw = torch.rand(batch_size, generator=generator, device=bank.device)
    use_trimodal = (kind_draw < trimodal_probability) & use_missing
    use_synchronized = (
        (kind_draw >= trimodal_probability)
        & (kind_draw < trimodal_probability + synchronized_probability)
        & use_missing
    )
    use_independent = use_missing & ~use_trimodal & ~use_synchronized
    selected = selected & use_independent[:, None, None]
    synchronized = torch.zeros((batch_size, bank.shape[-1]), dtype=torch.bool, device=bank.device)
    for row in range(batch_size):
        if not bool(use_synchronized[row] or use_trimodal[row]):
            continue
        ratio = float(torch.empty(1).uniform_(synchronized_rate_min, synchronized_rate_max, generator=generator))
        use_multi = float(torch.rand(1, generator=generator)) < multi_span_probability
        if use_multi:
            synchronized[row] = _random_multi_span(
                batch["content_mask"][row].bool(), ratio, generator,
                multi_span_count_min, multi_span_count_max, multi_span_length_max,
            )
        else:
            position_id = int(torch.randint(3, (1,), generator=generator))
            position = ("begin", "middle", "end")[position_id]
            synchronized[row] = _absolute_content_span(batch["content_mask"][row].bool(), ratio, position)
    text_shared = synchronized & use_trimodal[:, None]
    text_artificial = (selected[:, 0] | text_shared) & batch["text_observed_mask"].bool()
    output["input_ids"] = batch["input_ids"].clone().masked_fill(text_artificial, 103)
    output["text_observed_mask"] = batch["text_observed_mask"].bool() & ~text_artificial
    output["text_missing_mask"] = batch["text_missing_mask"].bool() | text_artificial
    shared_rows = use_synchronized | use_trimodal
    shared_total_missing = (
        batch["audio_missing_mask"].bool() | batch["vision_missing_mask"].bool() | synchronized
    )
    for channel, modality in ((1, "audio"), (2, "vision")):
        artificial = (selected[:, channel] | synchronized) & batch[f"{modality}_observed_mask"].bool()
        prior_missing = batch[f"{modality}_missing_mask"].bool()
        total_missing = prior_missing | artificial
        total_missing = torch.where(shared_rows[:, None], shared_total_missing, total_missing)
        observed = batch[f"{modality}_observed_mask"].bool() & ~total_missing
        values = batch[modality].clone()
        values = values.masked_fill(total_missing.unsqueeze(-1), 0.0)
        output[modality] = values
        output[f"{modality}_observed_mask"] = observed
        output[f"{modality}_missing_mask"] = total_missing
    output["modality_reliability"] = reliability_features(output)
    return output


def _random_multi_span(
    content: torch.Tensor,
    rate: float,
    generator: torch.Generator | None,
    count_min: int,
    count_max: int,
    length_max: int,
) -> torch.Tensor:
    """Generate several short synchronized segments, matching Attachment 3 more closely."""
    result = torch.zeros_like(content, dtype=torch.bool)
    indexes = torch.where(content)[0]
    if indexes.numel() <= 1 or rate <= 0:
        return result
    target = min(max(int(round(float(indexes.numel()) * rate)), 1), int(indexes.numel()) - 1)
    span_count = int(torch.randint(count_min, count_max + 1, (1,), generator=generator))
    attempts = 0
    while int((result & content).sum()) < target and attempts < span_count * 8:
        start_index = int(torch.randint(int(indexes.numel()), (1,), generator=generator))
        length = int(torch.randint(1, max(int(length_max), 1) + 1, (1,), generator=generator))
        start = int(indexes[start_index])
        result[start:start + length] = True
        result &= content
        attempts += 1
    if int(result.sum()) > target:
        chosen = torch.where(result)[0]
        keep = chosen[torch.randperm(chosen.numel(), generator=generator)[:target]]
        result.zero_()
        result[keep] = True
    return result


def apply_fixed_scenario(
    batch: dict[str, Any],
    pattern: str,
    rate: float,
    position: str,
) -> dict[str, Any]:
    pattern_modalities = {
        "none": [],
        "text": ["text"],
        "audio": ["audio"],
        "vision": ["vision"],
        "text_audio": ["text", "audio"],
        "text_vision": ["text", "vision"],
        "audio_vision": ["audio", "vision"],
        "all_modalities": ["text", "audio", "vision"],
    }
    if pattern not in pattern_modalities:
        raise ValueError(f"未知缺失类型: {pattern}")
    if position not in {"begin", "middle", "end"}:
        raise ValueError(f"未知缺失位置: {position}")
    output = dict(batch)
    modalities = pattern_modalities[pattern]
    content = batch["content_mask"].bool()
    shared_span = torch.stack([_absolute_content_span(row, rate, position) for row in content])
    for modality in modalities:
        artificial = shared_span & batch[f"{modality}_observed_mask"].bool()
        total_missing = batch[f"{modality}_missing_mask"].bool() | artificial
        output[f"{modality}_observed_mask"] = batch[f"{modality}_observed_mask"].bool() & ~artificial
        output[f"{modality}_missing_mask"] = total_missing
        if modality == "text":
            output["input_ids"] = batch["input_ids"].clone().masked_fill(total_missing, 103)
        else:
            output[modality] = batch[modality].masked_fill(total_missing.unsqueeze(-1), 0.0)
    output["modality_reliability"] = reliability_features(output)
    return output


def _absolute_content_span(content: torch.Tensor, rate: float, position: str) -> torch.Tensor:
    """Create one physically contiguous interval, then intersect it with each modality."""
    result = torch.zeros_like(content, dtype=torch.bool)
    indexes = torch.where(content)[0]
    if indexes.numel() <= 1 or rate <= 0:
        return result
    length = min(max(int(round(float(indexes.numel()) * rate)), 1), int(indexes.numel()) - 1)
    first, last = int(indexes[0]), int(indexes[-1])
    if position == "begin":
        start = first
    elif position == "end":
        start = last - length + 1
    else:
        start = first + max((int(indexes.numel()) - length) // 2, 0)
    result[start:start + length] = True
    return result & content


def iter_loader(loader: DataLoader[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield from loader
