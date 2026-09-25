from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .utils import move_to_device


class PrototypeMemory(nn.Module):
    """Traceable train-only medoid memory separated by modality/class/intensity."""

    def __init__(self, hidden_dim: int, temperature: float, intensity_centers: list[float]) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.register_buffer("embedding", torch.empty(0, hidden_dim))
        self.register_buffer("modality", torch.empty(0, dtype=torch.long))
        self.register_buffer("class_label", torch.empty(0, dtype=torch.long))
        self.register_buffer("intensity_bin", torch.empty(0, dtype=torch.long))
        self.register_buffer("intensity_value", torch.empty(0))
        self.register_buffer("intensity_centers", torch.tensor(intensity_centers, dtype=torch.float32))
        self.metadata: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.embedding.shape[0] > 0

    def clear(self) -> None:
        hidden = self.embedding.shape[-1]
        device = self.embedding.device
        self.embedding = torch.empty(0, hidden, device=device)
        self.modality = torch.empty(0, dtype=torch.long, device=device)
        self.class_label = torch.empty(0, dtype=torch.long, device=device)
        self.intensity_bin = torch.empty(0, dtype=torch.long, device=device)
        self.intensity_value = torch.empty(0, device=device)
        self.metadata = []

    def set_entries(self, arrays: dict[str, np.ndarray], metadata: list[dict[str, Any]]) -> None:
        embedding = np.asarray(arrays["embedding"])
        expected_hidden_dim = int(self.embedding.shape[-1])
        if embedding.ndim != 2 or embedding.shape[1] != expected_hidden_dim:
            raise ValueError(
                f"原型嵌入维度不匹配: expected=(*,{expected_hidden_dim}) actual={embedding.shape}"
            )
        if embedding.shape[0] == 0:
            raise ValueError("原型库不能为空")
        if not np.isfinite(embedding).all():
            raise ValueError("原型嵌入包含 NaN 或 Infinity")
        if len(metadata) != embedding.shape[0]:
            raise ValueError("原型元数据条数与嵌入条数不一致")
        for key in ("modality", "class_label", "intensity_bin", "intensity_value"):
            if len(arrays[key]) != embedding.shape[0]:
                raise ValueError(f"原型字段 {key} 的条数与嵌入条数不一致")
        device = self.embedding.device
        self.embedding = torch.as_tensor(embedding, dtype=torch.float32, device=device)
        self.embedding = torch.nn.functional.normalize(self.embedding, dim=-1)
        self.modality = torch.as_tensor(arrays["modality"], dtype=torch.long, device=device)
        self.class_label = torch.as_tensor(arrays["class_label"], dtype=torch.long, device=device)
        self.intensity_bin = torch.as_tensor(arrays["intensity_bin"], dtype=torch.long, device=device)
        self.intensity_value = torch.as_tensor(arrays["intensity_value"], dtype=torch.float32, device=device)
        self.metadata = metadata

    def retrieve(
        self, query: torch.Tensor, query_modality: torch.Tensor, query_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError("原型记忆库尚未构建")
        if self.temperature <= 0:
            raise ValueError("原型检索温度必须大于 0")
        # Prototype softmax must stay in FP32. Under CUDA autocast, -1e4 / 0.12
        # overflows FP16 to -inf; an all-masked evidence row would then produce
        # softmax(-inf, ...) = NaN and contaminate the complement regression.
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_fp32 = torch.nn.functional.normalize(query.float(), dim=-1)
            embedding_fp32 = torch.nn.functional.normalize(self.embedding.float(), dim=-1)
            similarity = torch.einsum("bkd,pd->bkp", query_fp32, embedding_fp32)
            valid = query_modality.unsqueeze(-1) == self.modality.view(1, 1, -1)
            if query_mask is not None:
                valid = valid & query_mask.bool().unsqueeze(-1)
            available_evidence = valid.any(dim=-1)
            scaled = (similarity / self.temperature).masked_fill(~valid, -1e9)
            weights = torch.softmax(scaled, dim=-1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            class_one_hot = torch.nn.functional.one_hot(self.class_label, num_classes=3).float()
            evidence_weight = available_evidence.float()
            denominator = evidence_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            class_per_evidence = torch.einsum("bkp,pc->bkc", weights, class_one_hot)
            class_per_evidence = torch.where(
                available_evidence.unsqueeze(-1), class_per_evidence, torch.zeros_like(class_per_evidence)
            )
            class_probability = class_per_evidence.sum(dim=1) / denominator
            sample_available = available_evidence.any(dim=1)
            class_probability = torch.where(
                sample_available.unsqueeze(-1), class_probability,
                torch.full_like(class_probability, 1.0 / class_probability.shape[-1]),
            )
            prototype_logits = torch.log(class_probability.clamp_min(1e-6))
            regression_per_evidence = torch.einsum("bkp,p->bk", weights, self.intensity_value.float())
            regression_per_evidence = torch.where(
                available_evidence, regression_per_evidence, torch.zeros_like(regression_per_evidence)
            )
            prototype_regression = (regression_per_evidence.sum(dim=1) / denominator.squeeze(1)).clamp(-3.0, 3.0)
            masked_similarity = similarity.masked_fill(~valid, -1e9)
            nearest_similarity, nearest_index = masked_similarity.max(dim=-1)
        nearest_index = torch.where(available_evidence, nearest_index, torch.full_like(nearest_index, -1))
        # Availability is carried explicitly. Keep tensors finite so no downstream
        # CSV/JSON writer can accidentally receive a NaN sentinel.
        nearest_similarity = torch.where(available_evidence, nearest_similarity, torch.zeros_like(nearest_similarity))
        return {
            "prototype_logits": prototype_logits,
            "prototype_regression": prototype_regression,
            "nearest_prototype_index": nearest_index,
            "nearest_prototype_similarity": nearest_similarity,
            "prototype_available": sample_available,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embedding=self.embedding.detach().cpu().numpy(),
            modality=self.modality.detach().cpu().numpy(),
            class_label=self.class_label.detach().cpu().numpy(),
            intensity_bin=self.intensity_bin.detach().cpu().numpy(),
            intensity_value=self.intensity_value.detach().cpu().numpy(),
            sample_id=np.asarray([row["sample_id"] for row in self.metadata], dtype="U32"),
            position=np.asarray([row["position"] for row in self.metadata], dtype=np.int16),
            source_split=np.asarray([row.get("source_split", "train") for row in self.metadata], dtype="U8"),
        )

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in ("embedding", "modality", "class_label", "intensity_bin", "intensity_value")}
            metadata = [
                {"sample_id": str(sample), "position": int(position), "source_split": str(split)}
                for sample, position, split in zip(source["sample_id"], source["position"], source["source_split"], strict=True)
            ]
        self.set_entries(arrays, metadata)


def _farthest_first(embedding: np.ndarray, count: int) -> np.ndarray:
    """Return actual observations; deterministic scalable k-center medoids."""
    if len(embedding) <= count:
        return np.arange(len(embedding), dtype=np.int64)
    normalized = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    center = normalized.mean(axis=0, keepdims=True)
    first = int(np.argmax((normalized * center).sum(axis=1)))
    selected = [first]
    nearest_distance = 1.0 - normalized @ normalized[first]
    for _ in range(1, count):
        index = int(np.argmax(nearest_distance))
        selected.append(index)
        nearest_distance = np.minimum(nearest_distance, 1.0 - normalized @ normalized[index])
    return np.asarray(selected, dtype=np.int64)


@torch.no_grad()
def build_prototype_memory(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    memory: PrototypeMemory,
    prototypes_per_cell: int,
    minimum_confidence: float,
    require_correct: bool = True,
) -> dict[str, int]:
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        identifiers = list(batch["sample_id"])
        batch = move_to_device(batch, device)
        outputs = model(batch, prototype_memory=None)
        probability = torch.softmax(outputs["base_logits"], dim=-1)
        confidence, prediction = probability.max(dim=-1)
        correct = prediction == batch["classification_labels"].long()
        keep_sample = confidence >= minimum_confidence
        if require_correct:
            keep_sample = keep_sample & correct
        for row in range(len(identifiers)):
            if not bool(keep_sample[row]):
                continue
            for rank in range(outputs["selected_query"].shape[1]):
                if not bool(outputs["selected_valid"][row, rank]):
                    continue
                records.append({
                    "embedding": outputs["selected_query"][row, rank].detach().cpu().numpy(),
                    "modality": int(outputs["selected_modality"][row, rank]),
                    "class_label": int(batch["classification_labels"][row]),
                    "intensity_bin": int(batch["intensity_bin"][row]),
                    "intensity_value": float(batch["regression_labels"][row]),
                    "sample_id": str(identifiers[row]),
                    "position": int(outputs["selected_position"][row, rank]),
                    "source_split": "train",
                })
    if not records:
        requirement = "正确分类与置信度阈值" if require_correct else "置信度阈值"
        raise RuntimeError(f"没有满足{requirement}的训练证据，无法构建原型库")
    chosen: list[dict[str, Any]] = []
    cells = sorted({(r["modality"], r["class_label"], r["intensity_bin"]) for r in records})
    for cell in cells:
        candidates = [r for r in records if (r["modality"], r["class_label"], r["intensity_bin"]) == cell]
        embedding = np.stack([r["embedding"] for r in candidates])
        chosen.extend(candidates[index] for index in _farthest_first(embedding, prototypes_per_cell))
    arrays = {
        "embedding": np.stack([r["embedding"] for r in chosen]).astype(np.float32),
        "modality": np.asarray([r["modality"] for r in chosen], dtype=np.int64),
        "class_label": np.asarray([r["class_label"] for r in chosen], dtype=np.int64),
        "intensity_bin": np.asarray([r["intensity_bin"] for r in chosen], dtype=np.int64),
        "intensity_value": np.asarray([r["intensity_value"] for r in chosen], dtype=np.float32),
    }
    memory.set_entries(arrays, chosen)
    model.train(was_training)
    return {"candidate_evidence": len(records), "occupied_cells": len(cells), "prototypes": len(chosen)}
