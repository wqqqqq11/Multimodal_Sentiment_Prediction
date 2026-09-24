"""Configuration loading and validation for Problem 3 preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the Problem 3 preprocessing configuration is invalid."""


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Problem3Config:
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


def load_config(path: str | Path) -> Problem3Config:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to read JSON-compatible YAML {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be an object")
    cfg = Problem3Config(project_root=source.parents[1], source_path=source, raw=raw)
    required = (
        "project", "paths", "data", "tokenizer", "mapping", "prototype",
        "counterfactual", "repair", "output", "validation",
    )
    for name in required:
        cfg.section(name)
    data = cfg.section("data")
    if data.get("version") != "aligned":
        raise ConfigError("Problem 3 primary preprocessing requires aligned features")
    if int(data.get("max_steps", 0)) != 50:
        raise ConfigError("data.max_steps must be 50")
    if int(data.get("audio_dim", 0)) != 74 or int(data.get("vision_dim", 0)) != 35:
        raise ConfigError("Organizer dimensions must be audio=74 and vision=35")
    if float(data.get("zero_epsilon", 0.0)) <= 0:
        raise ConfigError("data.zero_epsilon must be positive")
    prototype = cfg.section("prototype")
    bins = [float(value) for value in prototype.get("intensity_bins", [])]
    if bins != sorted(bins) or bins[0] != -3.0 or bins[-1] != 3.0:
        raise ConfigError("prototype.intensity_bins must be ordered from -3 to 3")
    if int(prototype.get("max_positions_per_sample_modality", 0)) < 1:
        raise ConfigError("prototype position cap must be positive")
    counterfactual = cfg.section("counterfactual")
    if int(counterfactual.get("validation_repeats", 0)) < 1:
        raise ConfigError("counterfactual.validation_repeats must be positive")
    if float(counterfactual.get("gaussian_noise_std", -1.0)) < 0:
        raise ConfigError("counterfactual.gaussian_noise_std must be non-negative")
    return cfg
