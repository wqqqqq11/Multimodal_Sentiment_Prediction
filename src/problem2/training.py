from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import apply_random_mask_view
from .evaluation import predict
from .losses import distillation_loss, supervised_loss
from .model import MRCDNet
from .utils import atomic_torch_save, move_to_device


def _autocast(device: torch.device):
    return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")


def _scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


def _epoch_average(sums: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / max(batches, 1) for key, value in sums.items()}


def _checkpoint(model: nn.Module, epoch: int, metrics: dict[str, Any], path: Path) -> None:
    atomic_torch_save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": metrics,
    }, path)


def load_model_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_teacher(
    model: MRCDNet,
    train_loader: DataLoader[dict[str, Any]],
    valid_loader: DataLoader[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    logger: Any,
) -> list[dict[str, Any]]:
    train_cfg = cfg["training"]
    weights = train_cfg["loss_weights"]
    class_weights = torch.tensor(train_cfg["class_weights"], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["teacher_lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg["teacher_epochs"]), 1))
    scaler = _scaler(device)
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoints" / "teacher_best.pt"
    for epoch in range(1, int(train_cfg["teacher_epochs"]) + 1):
        started = time.perf_counter()
        model.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        for source_batch in train_loader:
            batch = move_to_device(source_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                outputs = model(batch)
                total, parts = supervised_loss(outputs, batch, class_weights, weights)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(total.detach())
            for key, value in parts.items():
                sums[key] += float(value.detach())
            batches += 1
        scheduler.step()
        valid = predict(model, valid_loader, device)
        metrics = valid["metrics"]
        row = {
            "stage": "teacher", "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in _epoch_average(sums, batches).items()},
            **{f"valid_{key}": float(metrics[key]) for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")},
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        logger.info(
            "教师 Epoch %02d/%02d | loss=%.4f | Acc=%.4f F1=%.4f MAE=%.4f r=%.4f | %.1fs",
            epoch, int(train_cfg["teacher_epochs"]), row["train_loss"], metrics["accuracy"], metrics["macro_f1"],
            metrics["mae"], metrics["pearson"], row["seconds"],
        )
        score = float(metrics["selection_score"])
        if score > best_score + float(train_cfg["min_delta"]):
            best_score, stale = score, 0
            _checkpoint(model, epoch, metrics, checkpoint_path)
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                logger.info("教师早停：连续 %d 个 epoch 未改善", stale)
                break
    load_model_checkpoint(model, checkpoint_path, device)
    return history


def train_student(
    student: MRCDNet,
    teacher: MRCDNet,
    train_loader: DataLoader[dict[str, Any]],
    valid_loader: DataLoader[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    logger: Any,
) -> list[dict[str, Any]]:
    train_cfg = cfg["training"]
    weights = train_cfg["loss_weights"]
    class_weights = torch.tensor(train_cfg["class_weights"], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(train_cfg["student_lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg["student_epochs"]), 1))
    scaler = _scaler(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoints" / "student_best.pt"
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(int(cfg["project"]["seed"]) + 17)
    for epoch in range(1, int(train_cfg["student_epochs"]) + 1):
        started = time.perf_counter()
        student.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        distillation_scale = min(1.0, epoch / max(int(train_cfg["distillation_warmup_epochs"]), 1))
        for source_batch in train_loader:
            masked_cpu = apply_random_mask_view(source_batch, float(train_cfg["missing_view_probability"]), mask_generator)
            complete_batch = move_to_device(source_batch, device)
            masked_batch = move_to_device(masked_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode(), _autocast(device):
                teacher_outputs = teacher(complete_batch)
            with _autocast(device):
                student_outputs = student(masked_batch)
                supervised, supervised_parts = supervised_loss(student_outputs, masked_batch, class_weights, weights)
                distilled, distilled_parts = distillation_loss(
                    student_outputs, teacher_outputs,
                    float(train_cfg["distillation_temperature"]), weights,
                )
                total = supervised + distillation_scale * distilled
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(train_cfg["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(total.detach())
            sums["supervised"] += float(supervised.detach())
            sums["distillation"] += float(distilled.detach())
            for key, value in {**supervised_parts, **distilled_parts}.items():
                sums[key] += float(value.detach())
            batches += 1
        scheduler.step()
        complete = predict(student, valid_loader, device)
        missing = predict(student, valid_loader, device, ("audio_vision", 0.20, "middle"))
        complete_metrics, missing_metrics = complete["metrics"], missing["metrics"]
        robust_score = 0.5 * float(complete_metrics["selection_score"]) + 0.5 * float(missing_metrics["selection_score"])
        row = {
            "stage": "student", "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "distillation_scale": distillation_scale,
            **{f"train_{key}": value for key, value in _epoch_average(sums, batches).items()},
            **{f"complete_{key}": float(complete_metrics[key]) for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")},
            **{f"missing20_{key}": float(missing_metrics[key]) for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")},
            "robust_selection_score": robust_score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        logger.info(
            "学生 Epoch %02d/%02d | loss=%.4f sup=%.4f kd=%.4f | 完整F1=%.4f MAE=%.4f | 双缺失20%% F1=%.4f MAE=%.4f | %.1fs",
            epoch, int(train_cfg["student_epochs"]), row["train_loss"], row["train_supervised"], row["train_distillation"],
            complete_metrics["macro_f1"], complete_metrics["mae"], missing_metrics["macro_f1"], missing_metrics["mae"], row["seconds"],
        )
        if robust_score > best_score + float(train_cfg["min_delta"]):
            best_score, stale = robust_score, 0
            _checkpoint(student, epoch, {"complete": complete_metrics, "missing20": missing_metrics, "robust_score": robust_score}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                logger.info("学生早停：连续 %d 个 epoch 未改善", stale)
                break
    load_model_checkpoint(student, checkpoint_path, device)
    return history
