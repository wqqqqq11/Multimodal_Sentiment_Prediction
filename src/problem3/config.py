from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file(): raise FileNotFoundError(f"问题三模型配置不存在: {source}")
    cfg = _merge(json.loads(source.read_text(encoding="utf-8")), overrides or {})
    model = cfg["model"]
    if (int(model["max_steps"]), int(model["text_dim"]), int(model["audio_dim"]), int(model["vision_dim"])) != (50, 256, 74, 35):
        raise ValueError("问题三模型输入必须为 50 步以及 256/74/35 维")
    if int(model["hidden_dim"]) % int(model["modality_heads"]): raise ValueError("hidden_dim必须整除注意力头数")
    if len(cfg["training"]["class_weights"]) != 3: raise ValueError("class_weights必须有三项")
    if abs(sum(float(v) for v in cfg["explanation"]["contribution_weights"].values()) - 1.0) > 1e-6:
        raise ValueError("解释贡献权重之和必须为1")
    cfg["_config_path"] = str(source)
    return cfg


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    result = deepcopy(cfg)
    for key in ("preprocessed_root", "feature_root", "output_root"):
        path = Path(result["paths"][key])
        result["paths"][key] = str(path if path.is_absolute() else (root / path).resolve())
    return result
