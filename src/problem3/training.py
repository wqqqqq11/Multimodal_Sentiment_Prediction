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


def _mix(epoch: int, cfg: dict[str, Any]) -> float:
    warmup, transition = int(cfg["warmup_epochs"]), int(cfg["sparsity_transition_epochs"])
    return max(0.0, min(1.0, (epoch - warmup) / max(transition, 1)))


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.55, patience=3,
                                                            threshold=float(training["min_delta"]), min_lr=2e-6)
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    checkpoint_path = run_dir / "checkpoints" / "hsaig_best.pt"
    best_score, stale, history = -float("inf"), 0, []
    for epoch in range(1, int(training["epochs"]) + 1):
        started, sparsity_mix = time.perf_counter(), _mix(epoch, training)
        model.set_sparsity_mix(sparsity_mix)
        model.train()
        sums, batches = defaultdict(float), 0
        for source in train_loader:
            batch = move_to_device(source, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch)
                total, parts = supervised_loss(outputs, batch, class_weights, training)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            scaler.step(optimizer); scaler.update()
            sums["loss"] += float(total.detach())
            for key, value in parts.items(): sums[key] += float(value.detach())
            batches += 1
        validation = predict(model, valid_loader, device, goals=goals)
        metrics, score = validation["metrics"], float(validation["metrics"]["selection_score"])
        scheduler.step(score)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "sparsity_mix": sparsity_mix,
               **{f"train_{k}": v / max(batches, 1) for k, v in sums.items()},
               **{f"valid_{k}": float(metrics[k]) for k in ("accuracy", "macro_f1", "mae", "pearson",
                                                               "neutral_recall", "selection_score")},
               "goals_met": int(metrics["goal_audit"]["met_count"]), "seconds": time.perf_counter() - started}
        history.append(row)
        logger.info("Epoch %02d/%02d mix=%.2f | Acc=%.4f F1=%.4f MAE=%.4f r=%.4f Neutral-R=%.4f | goals=%d/5",
                    epoch, int(training["epochs"]), sparsity_mix, metrics["accuracy"], metrics["macro_f1"],
                    metrics["mae"], metrics["pearson"], metrics["neutral_recall"], row["goals_met"])
        if score > best_score + float(training["min_delta"]):
            best_score, stale = score, 0
            atomic_torch_save({"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics,
                               "selection_score": score, "sparsity_mix": sparsity_mix}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(training["patience"]): break
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    pd.DataFrame(history).to_csv(run_dir / "metrics" / "training_history.csv", index=False, encoding="utf-8-sig")
    return history, checkpoint
