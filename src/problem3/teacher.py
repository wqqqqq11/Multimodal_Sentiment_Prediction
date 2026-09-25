from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..problem2.data import reliability_features
from ..problem2.model import MRCDNet


def prepare_problem2_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Adapt complete Problem 3 inputs to the frozen Problem 2 model contract."""
    result = dict(batch)
    input_ids = result["input_ids"]
    attention = result["attention_mask"].bool()
    content = result.get("content_mask", attention).bool()
    result["content_mask"] = content
    result.setdefault("token_type_ids", torch.zeros_like(input_ids))
    result.setdefault("structural_mask", attention & ~content)
    result.setdefault("padding_mask", ~attention)
    for modality in ("text", "audio", "vision"):
        candidate_key = f"{modality}_evidence_candidate_mask"
        default_observed = result.get(candidate_key, content).bool() & content
        result.setdefault(f"{modality}_observed_mask", default_observed)
        result.setdefault(f"{modality}_natural_zero_mask", torch.zeros_like(content))
        result[f"{modality}_missing_mask"] = torch.zeros_like(content)
    result["modality_reliability"] = reliability_features(result)
    return result


class FrozenProblem2Teacher(nn.Module):
    """Full frozen MRCD-Net teacher used for P1 response distillation."""

    def __init__(self, model: MRCDNet) -> None:
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.model.eval()
        outputs = self.model(prepare_problem2_batch(batch))
        return {
            "logits": outputs["logits"].float(),
            "regression": outputs["regression"].float(),
            "fused": outputs["fused"].float(),
        }


def load_problem2_teacher(
    checkpoint_path: Path,
    config_path: Path,
    text_model_path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> FrozenProblem2Teacher:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"问题二教师检查点不存在: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"问题二教师配置不存在: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = MRCDNet(cfg["model"], text_model_name=str(text_model_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get(
        "model_state",
        checkpoint.get("student_state", checkpoint.get("state_dict", checkpoint)),
    )
    incompatible = model.load_state_dict(source, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "问题二教师检查点与模型结构不匹配: "
            f"missing={incompatible.missing_keys[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )
    teacher = FrozenProblem2Teacher(model.float()).to(device)
    logger.info("已加载并冻结问题二完整教师: %s", checkpoint_path)
    return teacher
