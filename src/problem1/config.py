"""Unified configuration loading and validation for problem 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_fingerprint(raw: dict[str, Any]) -> str:
    """Hash only settings that can change extracted feature artifacts."""
    runtime = raw.get("runtime", {})
    output = raw.get("output", {})
    payload = {
        "project_seed": raw.get("project", {}).get("seed"),
        "text": raw.get("text"),
        "audio": raw.get("audio"),
        "vision": raw.get("vision"),
        "runtime": {
            "device": runtime.get("device"),
            "mixed_precision": runtime.get("mixed_precision"),
        },
        "output": {"dtype": output.get("dtype")},
    }
    return _hash_payload(payload)


@dataclass(frozen=True)
class Problem1Config:
    project_root: Path
    source_path: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"配置缺少对象节：{name}")
        return value

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"路径配置非法：paths.{name}")
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def fingerprint(self) -> str:
        return _hash_payload(self.raw)

    @property
    def feature_fingerprint(self) -> str:
        return feature_fingerprint(self.raw)


def _validate_probability(value: Any, key: str, *, inclusive_zero: bool = True) -> float:
    number = float(value)
    lower_ok = number >= 0 if inclusive_zero else number > 0
    if not lower_ok or number > 1:
        raise ConfigError(f"{key} 必须位于 {'[0,1]' if inclusive_zero else '(0,1]'}")
    return number


def load_config(path: str | Path) -> Problem1Config:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"无法读取JSON兼容YAML配置 {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须是对象")
    project_root = source.parents[1]
    cfg = Problem1Config(project_root=project_root, source_path=source, raw=raw)
    for section in ("project", "paths", "runtime", "text", "audio", "vision", "multiscale", "alignment", "output", "validation", "visualization"):
        cfg.section(section)
    if int(cfg.section("runtime")["workers"]) < 1:
        raise ConfigError("runtime.workers 必须>=1")
    if cfg.section("runtime").get("device") not in ("auto", "cpu", "cuda"):
        raise ConfigError("runtime.device 必须是 auto/cpu/cuda")
    expected_backends = {"text": "deberta_v3", "audio": "wavlm", "vision": "mediapipe_timm"}
    for modality, backend in expected_backends.items():
        if cfg.section(modality).get("backend") != backend:
            raise ConfigError(f"竞赛配置要求 {modality}.backend={backend}，不再提供降级后端")
    for modality in ("text", "audio"):
        section = cfg.section(modality)
        revision = section.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ConfigError(f"{modality}.revision 不能为空，必须固定safetensors权重修订版本")
        if section.get("use_safetensors") is not True:
            raise ConfigError(f"{modality}.use_safetensors 必须为true，禁止回退到.bin权重")
    audio = cfg.section("audio")
    _validate_probability(audio["vad_threshold"], "audio.vad_threshold")
    noise_percentile = float(audio["vad_noise_percentile"])
    if not 0 <= noise_percentile < 100:
        raise ConfigError("audio.vad_noise_percentile 必须位于[0,100)")
    if float(audio["vad_margin_db"]) < 0:
        raise ConfigError("audio.vad_margin_db 必须>=0")
    model_path = cfg.section("vision").get("landmarker_model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ConfigError("vision.landmarker_model_path 不能为空")
    vision = cfg.section("vision")
    if vision.get("no_face_policy") != "full_frame_appearance":
        raise ConfigError("vision.no_face_policy 必须为full_frame_appearance，禁止使用全零或中心ROI降级")
    for key in ("face_quality_base", "scene_quality_base", "quality_exposure_weight",
                "quality_sharpness_weight", "min_face_detection_rate_warning"):
        _validate_probability(vision[key], f"vision.{key}")
    if float(vision["quality_sharpness_reference"]) <= 0:
        raise ConfigError("vision.quality_sharpness_reference 必须>0")
    if int(cfg.section("alignment")["max_consensus_steps"]) < 1:
        raise ConfigError("alignment.max_consensus_steps 必须>=1")
    _validate_probability(cfg.section("alignment")["semantic_weight"], "alignment.semantic_weight")
    _validate_probability(cfg.section("alignment")["time_weight"], "alignment.time_weight")
    _validate_probability(cfg.section("alignment")["quality_weight"], "alignment.quality_weight")
    _validate_probability(cfg.section("alignment")["monotone_projection_weight"],
                          "alignment.monotone_projection_weight")
    _validate_probability(cfg.section("alignment")["evidence_mass"], "alignment.evidence_mass", inclusive_zero=False)
    alignment = cfg.section("alignment")
    if float(alignment["audio_sinkhorn_epsilon"]) <= 0:
        raise ConfigError("alignment.audio_sinkhorn_epsilon 必须>0")
    for key in ("time_bands", "audio_time_bands"):
        bands = [float(value) for value in alignment[key]]
        if len(bands) != 3 or any(value <= 0 or value > 1 for value in bands):
            raise ConfigError(f"alignment.{key} 必须是三个位于(0,1]的数")
    weights = [float(x) for x in cfg.section("multiscale")["scale_weights"]]
    if len(weights) != 3 or any(x < 0 for x in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ConfigError("multiscale.scale_weights 必须为和等于1的三个非负数")
    return cfg
