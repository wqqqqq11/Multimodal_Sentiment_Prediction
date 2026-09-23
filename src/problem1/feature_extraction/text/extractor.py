"""DeBERTa-v3 contextual word representations for competition use."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from ...common.device import resolve_torch_device
from ...schemas import FeatureSequence
from .quality import token_quality
from .token_mapping import tokens_from_rows

_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _resolve_cache_dir(config: dict[str, Any]) -> str:
    path = Path(str(config["cache_dir"]))
    if not path.is_absolute():
        path = Path(str(config["project_root"])) / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_encoder(config: dict[str, Any]) -> tuple[Any, Any, str]:
    device = resolve_torch_device(str(config.get("device", "auto")))
    model_name = str(config["model_name"])
    revision = str(config["revision"])
    if config.get("use_safetensors") is not True:
        raise RuntimeError("text.use_safetensors必须为true，禁止加载不安全的.bin权重")
    cache_dir = _resolve_cache_dir(config)
    key = (model_name, revision, device, cache_dir)
    with _CACHE_LOCK:
        # Transformers uses lazy module imports. Importing it concurrently from
        # worker threads can expose a partially initialized module, so both the
        # first import and first model construction are serialized here.
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(f"无法导入transformers AutoModel/AutoTokenizer: {exc}") from exc
        if key not in _CACHE:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, use_fast=True, cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
            )
            if not tokenizer.is_fast:
                raise RuntimeError(f"{model_name}必须使用Fast tokenizer，才能保持子词到原词元的严格映射")
            model = AutoModel.from_pretrained(
                model_name, revision=revision, use_safetensors=True, cache_dir=cache_dir,
                local_files_only=bool(config.get("local_files_only", False)),
                output_hidden_states=True,
            ).eval().to(device)
            _CACHE[key] = (tokenizer, model)
    tokenizer, model = _CACHE[key]
    return tokenizer, model, device


def extract_text(sample_id: str, text: str, config: dict[str, Any],
                 token_rows: list[dict[str, Any]] | None = None) -> FeatureSequence:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少PyTorch；请安装根目录requirements.txt") from exc
    tokens = tokens_from_rows(text, token_rows)
    if not tokens:
        raise ValueError(f"{sample_id}: 文本词元为空")
    tokenizer, model, device = _load_encoder(config)
    token_text = [str(item["token"]) for item in tokens]
    encoded = tokenizer(
        token_text, is_split_into_words=True, return_tensors="pt", truncation=True,
        max_length=int(config["max_length"]), add_special_tokens=True,
    )
    word_ids = encoded.word_ids(batch_index=0)
    model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    use_amp = bool(config.get("mixed_precision", True)) and device == "cuda"
    with _INFERENCE_LOCK, torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16, enabled=use_amp
    ):
        output = model(**model_inputs)
        last_n = int(config["layer_mix_last_n"])
        hidden = torch.stack(output.hidden_states[-last_n:], dim=0).mean(dim=0)[0].float().cpu().numpy()
    features = []
    for word_index in range(len(tokens)):
        subword_indices = [index for index, mapped in enumerate(word_ids) if mapped == word_index]
        if not subword_indices:
            raise ValueError(
                f"{sample_id}: 第{word_index}个词元未被编码，可能是max_length过小导致截断；请增大text.max_length"
            )
        features.append(hidden[subword_indices].mean(axis=0))
    matrix = np.asarray(features, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    starts = np.asarray([item["start"] for item in tokens], dtype=np.float32)
    ends = np.asarray([item["end"] for item in tokens], dtype=np.float32)
    sequence = FeatureSequence(
        sample_id=sample_id, modality="text", features=matrix,
        positions=(np.arange(len(tokens), dtype=np.float32) + 0.5) / len(tokens),
        start=starts, end=ends, quality=token_quality(tokens),
        valid_mask=np.ones(len(tokens), dtype=np.bool_), source_index=np.arange(len(tokens), dtype=np.int32),
        metadata={"backend": "deberta_v3", "model_name": str(config["model_name"]),
                  "revision": str(config["revision"]), "weight_format": "safetensors",
                  "tokens": token_text, "layer_mix_last_n": int(config["layer_mix_last_n"]),
                  "embedding_dimension": int(matrix.shape[1]), "unit": "character"},
    )
    sequence.validate()
    return sequence
