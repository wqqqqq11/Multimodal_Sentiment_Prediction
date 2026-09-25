from __future__ import annotations

import math
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


def _mix(epoch: int, cfg: dict[str, Any]) -> float:
    warmup, transition = int(cfg["warmup_epochs"]), int(cfg["sparsity_transition_epochs"])
    return max(0.0, min(1.0, (epoch - warmup) / max(transition, 1)))


def _lr_factor(epoch_index: int, cfg: dict[str, Any]) -> float:
    warmup, total = int(cfg["scheduler_warmup_epochs"]), int(cfg["epochs"])
    floor = float(cfg["minimum_lr_ratio"])
    if epoch_index < warmup: return float(epoch_index + 1) / max(warmup, 1)
    progress = (epoch_index - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def load_checkpoint(model: HSAIGNet, path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_model(model: HSAIGNet, train_loader: DataLoader[dict[str, Any]],
                valid_loader: DataLoader[dict[str, Any]], cfg: dict[str, Any], device: torch.device,
                run_dir: Path, logger: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training, goals = cfg["training"], cfg["evaluation"]["goals"]
    class_weights = torch.tensor(training["class_weights"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: _lr_factor(epoch, training))
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp, init_scale=4096.0, growth_interval=1000)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "hsaig_best.pt"
    best_score, stale, history = -float("inf"), 0, []
    for epoch in range(1, int(training["epochs"]) + 1):
        started, sparsity_mix = time.perf_counter(), _mix(epoch, training)
        model.set_sparsity_mix(sparsity_mix)
        model.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        for source in train_loader:
            batch = move_to_device(source, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch)
                total, parts = supervised_loss(outputs, batch, class_weights, training)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(total.detach())
            for key, value in parts.items(): sums[key] += float(value.detach())
            batches += 1
        validation = predict(model, valid_loader, device, goals=goals)
        metrics, score = validation["metrics"], float(validation["metrics"]["selection_score"])
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "sparsity_mix": sparsity_mix,
               **{f"train_{key}": value / max(batches, 1) for key, value in sums.items()},
               **{f"valid_{key}": float(metrics[key]) for key in
                  ("accuracy", "macro_f1", "mae", "pearson", "selection_score")},
               "goals_met": int(metrics["goal_audit"]["met_count"]), "seconds": time.perf_counter() - started}
        history.append(row)
        logger.info("Epoch %02d/%02d entmax=%.2f | Acc=%.4f Macro-F1=%.4f MAE=%.4f Pearson=%.4f | goals=%d/4",
                    epoch, int(training["epochs"]), sparsity_mix, metrics["accuracy"], metrics["macro_f1"],
                    metrics["mae"], metrics["pearson"], row["goals_met"])
        if score > best_score + float(training["min_delta"]):
            best_score, stale = score, 0
            atomic_torch_save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics,
                               "selection_score": score, "sparsity_mix": sparsity_mix,
                               "selection_weights": {"accuracy": 0.30, "macro_f1": 0.30,
                                                     "mae": 0.25, "pearson": 0.15}}, checkpoint_path)
            logger.info("保存最优模型 epoch=%d selection_score=%.6f", epoch, score)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(training["minimum_epochs"]) and stale >= int(training["patience"]):
            logger.info("正式指标选模分数连续%d轮未改善，提前停止", stale)
            break
    if not checkpoint_path.is_file(): raise RuntimeError("训练未生成最优检查点")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    pd.DataFrame(history).to_csv(run_dir / "metrics" / "training_history.csv", index=False, encoding="utf-8-sig")
    metrics = checkpoint["metrics"]
    logger.info("加载最优单模型 epoch=%d entmax=%.2f | Acc=%.4f Macro-F1=%.4f MAE=%.4f Pearson=%.4f",
                checkpoint["epoch"], checkpoint["sparsity_mix"], metrics["accuracy"], metrics["macro_f1"],
                metrics["mae"], metrics["pearson"])
    return history, checkpoint
