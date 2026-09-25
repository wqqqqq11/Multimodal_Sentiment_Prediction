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
        result[key] = _deep_update(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"问题三模型配置不存在: {config_path}")
    try:
        cfg = _deep_update(json.loads(config_path.read_text(encoding="utf-8")), overrides or {})
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置须为 JSON 兼容 YAML: {config_path}: {exc}") from exc
    validate_config(cfg)
    cfg["_config_path"] = str(config_path)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("project", "paths", "data", "model", "training", "evaluation"):
        if section not in cfg:
            raise ConfigError(f"缺少配置段: {section}")
    model = cfg["model"]
    training = cfg["training"]
    top_k = int(model["top_k"])
    if not 1 <= top_k <= 3 * int(model["max_steps"]):
        raise ConfigError("model.top_k 必须位于 [1, 3*max_steps]")
    minimum_top_k = int(model.get("minimum_top_k", 1))
    if not 1 <= minimum_top_k <= top_k:
        raise ConfigError("model.minimum_top_k 必须位于 [1, top_k]")
    if float(model.get("adaptive_top_k_ratio", 0.0)) < 0:
        raise ConfigError("model.adaptive_top_k_ratio 不能为负")
    if int(model.get("evidence_window_radius", 0)) < 0:
        raise ConfigError("model.evidence_window_radius 不能为负")
    alpha = float(model["prototype_fusion_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError("model.prototype_fusion_alpha 必须位于 [0,1]")
    ordinal_alpha = float(model.get("ordinal_logit_alpha", 0.0))
    if not 0.0 <= ordinal_alpha <= 1.0:
        raise ConfigError("model.ordinal_logit_alpha 必须位于 [0,1]")
    if int(model["hidden_dim"]) % int(model["attention_heads"]) != 0:
        raise ConfigError("hidden_dim 必须能被 attention_heads 整除")
    for key in ("counterfactual_batch_probability", "stability_batch_probability"):
        if not 0.0 <= float(training[key]) <= 1.0:
            raise ConfigError(f"training.{key} 必须位于 [0,1]")
    for key, value in training["loss_weights"].items():
        if float(value) < 0:
            raise ConfigError(f"损失权重 {key} 不能为负")
    if len(model["intensity_bin_centers"]) != 7:
        raise ConfigError("intensity_bin_centers 必须包含 7 个中心")


def resolve_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result = deepcopy(cfg)
    for key, value in result["paths"].items():
        path = Path(value)
        result["paths"][key] = str(path if path.is_absolute() else (project_root / path).resolve())
    return result
