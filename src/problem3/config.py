from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the Problem 3 model configuration is invalid."""


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"问题三模型配置不存在: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置须为 JSON 兼容 YAML: {source}: {exc}") from exc
    cfg = _deep_update(raw, overrides or {})
    validate_config(cfg)
    cfg["_config_path"] = str(source)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = ("project", "paths", "data", "model", "training", "evaluation", "explanation", "output")
    for section in required:
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"缺少配置段: {section}")
    model = cfg["model"]
    if int(model["max_steps"]) != 50 or int(model["audio_dim"]) != 74 or int(model["vision_dim"]) != 35:
        raise ConfigError("模型输入必须与 aligned_50 的 50/74/35 维度一致")
    if int(model["hidden_dim"]) % int(model["modality_heads"]) != 0:
        raise ConfigError("hidden_dim 必须能被 modality_heads 整除")
    training = cfg["training"]
    if int(training["epochs"]) < 1 or int(training["patience"]) < 1:
        raise ConfigError("epochs 和 patience 必须为正整数")
    if len(training["class_weights"]) != 3 or any(float(v) <= 0 for v in training["class_weights"]):
        raise ConfigError("class_weights 必须包含三个正数")
    if any(float(v) < 0 for v in training["loss_weights"].values()):
        raise ConfigError("损失权重不能为负")
    explanation = cfg["explanation"]
    if int(explanation["integrated_gradient_steps"]) < 2:
        raise ConfigError("积分梯度步数至少为2")
    if not 0 < float(explanation["top_k_ratio"]) <= 1:
        raise ConfigError("top_k_ratio 必须位于 (0,1]")
    contribution_total = sum(float(v) for v in explanation["contribution_weights"].values())
    if abs(contribution_total - 1.0) > 1e-6:
        raise ConfigError("三类模态贡献权重之和必须为1")
    goals = cfg["evaluation"]["goals"]
    for key in ("accuracy", "macro_f1", "pearson", "neutral_recall"):
        if not 0 <= float(goals[key]) <= 1:
            raise ConfigError(f"目标 {key} 必须位于 [0,1]")
    if float(goals["mae"]) <= 0:
        raise ConfigError("MAE目标必须大于0")


def resolve_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result = deepcopy(cfg)
    for key in ("preprocessed_root", "output_root", "text_pretrained_model"):
        value = Path(result["paths"][key])
        result["paths"][key] = str(value if value.is_absolute() else (project_root / value).resolve())
    return result
