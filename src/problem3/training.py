from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import predict
from .losses import supervised_loss
from .model import HSAIGNet
from .utils import atomic_torch_save, move_to_device


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def _make_optimizer(model: HSAIGNet, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    text_ids = {id(parameter) for parameter in model.text_encoder.backbone.parameters()}
    text_parameters = [parameter for parameter in model.parameters() if id(parameter) in text_ids]
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_ids]
    return torch.optim.AdamW(
        [
            {"params": text_parameters, "lr": float(cfg["text_lr"])},
            {"params": other_parameters, "lr": float(cfg["head_lr"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )


def load_checkpoint(model: HSAIGNet, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_model(
    model: HSAIGNet,
    train_loader: DataLoader[dict[str, Any]],
    valid_loader: DataLoader[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    logger: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training = cfg["training"]
    goals = cfg["evaluation"]["goals"]
    class_weights = torch.tensor(training["class_weights"], device=device, dtype=torch.float32)
    model.freeze_text_bottom_layers(int(training.get("freeze_text_bottom_layers", 0)))
    optimizer = _make_optimizer(model, training)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.55, patience=2,
        threshold=float(training["min_delta"]), min_lr=1e-7,
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    checkpoint_path = run_dir / "checkpoints" / "hsaig_best.pt"
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        started = time.perf_counter()
        if epoch == int(training.get("unfreeze_text_epoch", 1)):
            model.freeze_text_bottom_layers(0)
            logger.info("文本编码器从第 %d 轮起全部解冻", epoch)
        model.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        for source in train_loader:
            batch = move_to_device(source, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                outputs = model(batch)
                total, parts = supervised_loss(outputs, batch, class_weights, training)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(total.detach())
            for key, value in parts.items():
                sums[key] += float(value.detach())
            batches += 1
        validation = predict(model, valid_loader, device, goals=goals)
        metrics = validation["metrics"]
        score = float(metrics["selection_score"])
        scheduler.step(score)
        averages = {key: value / max(batches, 1) for key, value in sums.items()}
        row = {
            "epoch": epoch,
            "text_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in averages.items()},
            **{f"valid_{key}": float(metrics[key]) for key in (
                "accuracy", "macro_f1", "mae", "pearson", "neutral_recall", "selection_score"
            )},
            "goals_met": int(metrics["goal_audit"]["met_count"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        logger.info(
            "Epoch %02d/%02d | loss=%.4f | Acc=%.4f F1=%.4f MAE=%.4f r=%.4f Neutral-R=%.4f | goals=%d/5 | %.1fs",
            epoch, int(training["epochs"]), row["train_loss"], metrics["accuracy"], metrics["macro_f1"],
            metrics["mae"], metrics["pearson"], metrics["neutral_recall"], row["goals_met"], row["seconds"],
        )
        if score > best_score + float(training["min_delta"]):
            best_score = score
            stale = 0
            atomic_torch_save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics, "selection_score": score},
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= int(training["patience"]):
                logger.info("验证集目标分数连续 %d 轮未改善，提前停止", stale)
                break
    if not checkpoint_path.is_file():
        raise RuntimeError("训练未生成最佳模型检查点")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    pd.DataFrame(history).to_csv(run_dir / "metrics" / "training_history.csv", index=False, encoding="utf-8-sig")
    return history, checkpoint
