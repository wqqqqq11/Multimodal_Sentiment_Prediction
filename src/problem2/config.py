from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a Problem 2 configuration is invalid."""


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"模型配置不存在: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置须为 JSON 兼容 YAML: {config_path}: {exc}") from exc
    cfg = _deep_update(raw, overrides or {})
    validate_config(cfg)
    cfg["_config_path"] = str(config_path)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    for section in ("project", "paths", "data", "model", "training", "evaluation"):
        if section not in cfg:
            raise ConfigError(f"缺少配置段: {section}")
    probability = float(cfg["training"]["missing_view_probability"])
    if not 0.0 <= probability <= 1.0:
        raise ConfigError("missing_view_probability 必须在 [0,1]")
    synchronized = float(cfg["training"].get("synchronized_missing_probability", 0.0))
    trimodal = float(cfg["training"].get("trimodal_missing_probability", 0.0))
    if synchronized < 0.0 or trimodal < 0.0 or synchronized + trimodal > 1.0:
        raise ConfigError("同步音视频与三模态缺失概率之和必须在 [0,1]")
    for key in (
        "text_pretrained_model", "text_pretrained_revision",
        "teacher_text_pretrained_model", "teacher_text_pretrained_revision",
    ):
        if not str(cfg["model"].get(key, "")).strip():
            raise ConfigError(f"model.{key} 不能为空")
    residual_scale = float(cfg["model"].get("classification_residual_scale", 0.35))
    if not 0.0 <= residual_scale <= 1.0:
        raise ConfigError("model.classification_residual_scale 必须在 [0,1]")
    temperature = float(cfg["training"]["distillation_temperature"])
    if temperature <= 0:
        raise ConfigError("distillation_temperature 必须大于 0")
    for key, value in cfg["training"]["loss_weights"].items():
        if float(value) < 0:
            raise ConfigError(f"损失权重 {key} 不能为负")
    for rate in cfg["evaluation"]["missing_rates"]:
        if not 0.0 <= float(rate) < 1.0:
            raise ConfigError(f"缺失率必须在 [0,1): {rate}")
    evaluation = cfg["evaluation"]
    if "stress_scenario" not in evaluation:
        raise ConfigError("evaluation.stress_scenario 缺失")
    for key in ("pattern", "rate", "position"):
        if key not in evaluation["stress_scenario"]:
            raise ConfigError(f"evaluation.stress_scenario 缺少 {key}")
    if "selection" not in evaluation:
        raise ConfigError("evaluation.selection 缺失")
    selection = evaluation["selection"]
    for key in ("complete_weight", "robust_weight", "scenarios"):
        if key not in selection:
            raise ConfigError(f"evaluation.selection 缺少 {key}")
    if not selection["scenarios"]:
        raise ConfigError("evaluation.selection.scenarios 不能为空")


def resolve_paths(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result = deepcopy(cfg)
    for key in ("preprocessed_root", "output_root", "strategy_document"):
        value = Path(result["paths"][key])
        result["paths"][key] = str(value if value.is_absolute() else (project_root / value).resolve())
    return result
