"""Strict PyTorch device selection for competition encoders."""

from __future__ import annotations

from typing import Any


def resolve_torch_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少PyTorch；请先执行 uv pip install -r requirements.txt") from exc
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但PyTorch没有检测到可用CUDA设备")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cpu", "cuda"):
        raise ValueError("device必须是auto/cpu/cuda")
    return requested


def runtime_device(requested: str = "auto") -> dict[str, Any]:
    import torch
    selected = resolve_torch_device(requested)
    return {"device": selected, "backend": "torch", "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
