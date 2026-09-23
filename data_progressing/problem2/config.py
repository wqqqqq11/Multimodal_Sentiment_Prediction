"""Configuration loading and validation for Problem 2 preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a preprocessing configuration is invalid."""


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Problem2Config:
    project_root: Path
    source_path: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing object section: {name}")
        return value

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"Invalid path setting: paths.{name}")
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.raw)

    @property
    def seed(self) -> int:
        return int(self.section("project")["seed"])


def _probability_map(section: dict[str, Any], key: str) -> None:
    values = section.get(key)
    if not isinstance(values, dict) or not values:
        raise ConfigError(f"{key} must be a non-empty object")
    probabilities = [float(value) for value in values.values()]
    if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ConfigError(f"{key} probabilities must be non-negative and sum to one")


def load_config(path: str | Path) -> Problem2Config:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to read JSON-compatible YAML {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be an object")
    cfg = Problem2Config(project_root=source.parents[1], source_path=source, raw=raw)
    for name in ("project", "paths", "data", "scaling", "missingness", "quality", "output", "validation"):
        cfg.section(name)

    data = cfg.section("data")
    if data.get("version") != "aligned":
        raise ConfigError("Problem 2 primary pipeline requires the aligned feature version")
    if int(data.get("max_steps", 0)) != 50:
        raise ConfigError("data.max_steps must be 50 for the organizer's aligned feature file")
    if int(data.get("audio_dim", 0)) != 74 or int(data.get("vision_dim", 0)) != 35:
        raise ConfigError("Organizer feature dimensions must be audio=74 and vision=35")
    if float(data.get("zero_epsilon", 0)) <= 0:
        raise ConfigError("data.zero_epsilon must be positive")

    scaling = cfg.section("scaling")
    lower = float(scaling.get("lower_quantile", -1))
    upper = float(scaling.get("upper_quantile", 2))
    if not 0 <= lower < upper <= 1:
        raise ConfigError("Scaling quantiles must satisfy 0 <= lower < upper <= 1")
    if scaling.get("scale") not in {"iqr"}:
        raise ConfigError("Only leakage-safe IQR scaling is supported")
    if set(scaling.get("modalities", [])) != {"audio", "vision"}:
        raise ConfigError("Only audio and vision are scaled; text_bert is the common text interface")

    missingness = cfg.section("missingness")
    _probability_map(missingness, "pattern_probabilities")
    _probability_map(missingness, "position_probabilities")
    if int(missingness.get("mask_bank_size", 0)) < 1:
        raise ConfigError("missingness.mask_bank_size must be at least one")
    ratio_min = float(missingness.get("ratio_min", -1))
    ratio_max = float(missingness.get("ratio_max", 2))
    if not 0 < ratio_min <= ratio_max < 1:
        raise ConfigError("Missing ratios must satisfy 0 < min <= max < 1")
    if float(missingness.get("ratio_beta_alpha", 0)) <= 0 or float(missingness.get("ratio_beta_beta", 0)) <= 0:
        raise ConfigError("Beta distribution parameters must be positive")
    return cfg
