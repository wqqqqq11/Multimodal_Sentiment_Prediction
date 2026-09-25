from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .evaluation import evaluate_model
from .losses import Problem3Loss, perturb_batch
from .prototypes import PrototypeMemory, build_prototype_memory
from .utils import atomic_torch_save, file_sha256, move_to_device, write_json


def warm_start_text_backbone(model: nn.Module, checkpoint_path: Path, logger: logging.Logger) -> dict[str, int]:
    """Warm-start the shared BERT only; P1 uses the complete checkpoint as a frozen teacher."""
    if not checkpoint_path.is_file():
        logger.warning("问题二检查点不存在，跳过热启动: %s", checkpoint_path)
        return {"loaded": 0, "skipped": 0}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state", checkpoint)
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skipped = 0
    prefix = "text_encoder.backbone."
    for key, value in source.items():
        if not key.startswith(prefix):
            continue
        target_key = "text_backbone." + key[len(prefix):]
        if target_key in target and target[target_key].shape == value.shape:
            mapped[target_key] = value.to(target[target_key].dtype)
        else:
            skipped += 1
    model.load_state_dict(mapped, strict=False)
    logger.info("问题二文本骨干热启动: loaded=%d skipped=%d", len(mapped), skipped)
    return {"loaded": len(mapped), "skipped": skipped}


def _optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    base: list[nn.Parameter] = []
    text: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (text if name.startswith("text_backbone.") else base).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": base, "lr": float(cfg["training"]["learning_rate"])},
            {"params": text, "lr": float(cfg["training"]["text_learning_rate"])},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


def _anneal(initial: float, final: float, epoch: int, total_epochs: int, exponential: bool) -> float:
    ratio = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
    if exponential and initial > 0 and final > 0:
        return initial * (final / initial) ** ratio
    return initial + (final - initial) * ratio


def _temperature(cfg: dict[str, Any], epoch: int, total_epochs: int) -> float:
    return _anneal(
        float(cfg["model"]["selector_temperature_initial"]),
        float(cfg["model"]["selector_temperature_final"]),
        epoch,
        total_epochs,
        True,
    )


def _selector_noise(cfg: dict[str, Any], epoch: int, total_epochs: int) -> float:
    return _anneal(
        float(cfg["model"].get("selector_noise_initial", 1.0)),
        float(cfg["model"].get("selector_noise_final", 0.0)),
        epoch,
        total_epochs,
        False,
    )


def _performance_gate(metrics: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    evaluation = cfg["evaluation"]
    baseline = evaluation.get("problem2_baseline", {})
    ratio = float(evaluation.get("performance_floor_ratio", 0.95))
    checks = {
        "accuracy": float(metrics["accuracy"]) >= ratio * float(baseline.get("accuracy", 0.0)),
        "macro_f1": float(metrics["macro_f1"]) >= ratio * float(baseline.get("macro_f1", 0.0)),
        "mae": float(metrics["mae"]) <= float(baseline.get("mae", float("inf"))) / max(ratio, 1e-6),
        "pearson": float(metrics["pearson"]) >= ratio * float(baseline.get("pearson", 0.0)),
        "neutral_recall": float(metrics["per_class_recall"][1]) >= float(evaluation.get("minimum_neutral_recall", 0.0)),
    }
    return all(checks.values()), checks


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: Problem3Loss,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    use_counterfactual: bool,
    teacher: Any | None,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    totals: dict[str, float] = {}
    steps = 0
    amp_enabled = bool(cfg["training"]["amp"]) and device.type == "cuda"
    counterfactual_cfg = cfg.get("counterfactual", {})
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        teacher_outputs = teacher(batch) if teacher is not None else None
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            # P0: prototypes are never fed back into the predictor during training.
            outputs = model(batch, prototype_memory=None)
            if teacher_outputs is not None:
                outputs["teacher_logits"] = teacher_outputs["logits"]
                outputs["teacher_regression"] = teacher_outputs["regression"]
            complement = None
            if use_counterfactual and random.random() < float(cfg["training"]["counterfactual_batch_probability"]):
                complement = model.predict_complement(outputs, prototype_memory=None)
            perturbed = None
            if use_counterfactual and random.random() < float(cfg["training"]["stability_batch_probability"]):
                perturbed_batch = perturb_batch(
                    batch,
                    mask_token_id=int(counterfactual_cfg.get("mask_token_id", 103)),
                    probability=float(counterfactual_cfg.get("non_evidence_mask_ratio", 0.05)),
                    noise_std=float(counterfactual_cfg.get("gaussian_noise_std", 0.02)),
                )
                perturbed = model(perturbed_batch, prototype_memory=None)
            loss, terms = criterion(outputs, batch, complement=complement, perturbed=perturbed)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"训练损失出现非有限值: {terms}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip"]))
        scaler.step(optimizer)
        scaler.update()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
    scheduler.step()
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train_model(
    model: Any,
    train_loader: Any,
    train_eval_loader: Any,
    valid_loader: Any,
    device: torch.device,
    cfg: dict[str, Any],
    run_dir: Path,
    logger: logging.Logger,
    resume: Path | None = None,
    teacher: Any | None = None,
) -> tuple[dict[str, Any], PrototypeMemory]:
    model.to(device)
    memory = PrototypeMemory(
        int(cfg["model"]["hidden_dim"]),
        float(cfg["model"]["prototype_temperature"]),
        list(cfg["model"]["intensity_bin_centers"]),
    ).to(device)
    criterion = Problem3Loss(cfg).to(device)
    optimizer = _optimizer(model, cfg)
    warmup_epochs = int(cfg["training"]["warmup_epochs"])
    total_epochs = warmup_epochs + int(cfg["training"]["joint_epochs"])
    if total_epochs < 1:
        raise ValueError("问题三总训练轮数必须大于 0")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")

    start_epoch = 0
    best_score = float("-inf")
    best_gate_passed = False
    best_epoch: int | None = None
    best_stage = ""
    patience = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "checkpoints" / "best.pt"
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state.get("best_score", best_score))
        best_gate_passed = bool(state.get("best_gate_passed", False))
        best_epoch = state.get("best_epoch")
        best_stage = str(state.get("best_stage", ""))
        patience = int(state.get("patience", 0))
        history = list(state.get("history", []))
        source_best = resume.parent / "best.pt"
        if source_best.is_file():
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_best, best_path)
        logger.info("从 epoch=%d 恢复训练", start_epoch)

    freeze_epochs = int(cfg["training"]["text_freeze_epochs"])
    enable_counterfactual = bool(cfg["training"].get("enable_counterfactual", False))
    for epoch in range(start_epoch, total_epochs):
        model.freeze_text_layers(epoch < freeze_epochs)
        temperature = _temperature(cfg, epoch, total_epochs)
        noise_scale = _selector_noise(cfg, epoch, total_epochs)
        model.set_selector_temperature(temperature)
        model.set_selector_noise(noise_scale)
        stage = "warmup" if epoch < warmup_epochs else "selector"
        if epoch == warmup_epochs:
            patience = 0
            logger.info("进入 selector 阶段：保留全阶段最优模型，继续退火稀疏选择器")
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            cfg,
            scaler,
            use_counterfactual=enable_counterfactual and stage == "selector",
            teacher=teacher,
        )
        validation = evaluate_model(
            model,
            valid_loader,
            device,
            cfg,
            prototype_memory=None,
            explain=False,
            teacher=teacher,
        )
        validation_metrics = validation["metrics"]
        score = float(validation_metrics["selection_score"])
        gate_passed, gate_checks = _performance_gate(validation_metrics, cfg)
        epoch_row = {
            "epoch": epoch + 1,
            "stage": stage,
            "temperature": temperature,
            "noise_scale": noise_scale,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "performance_gate_passed": gate_passed,
            "performance_gate_checks": gate_checks,
        }
        history.append(epoch_row)
        logger.info(
            "epoch=%03d stage=%s loss=%.5f acc=%.4f f1=%.4f mae=%.4f r=%.4f "
            "score=%.4f gate=%s temp=%.3f noise=%.3f",
            epoch + 1,
            stage,
            train_metrics["total"],
            validation_metrics["accuracy"],
            validation_metrics["macro_f1"],
            validation_metrics["mae"],
            validation_metrics["pearson"],
            score,
            gate_passed,
            temperature,
            noise_scale,
        )
        improved = gate_passed and not best_gate_passed
        improved = improved or (gate_passed == best_gate_passed and score > best_score + float(cfg["training"]["min_delta"]))
        if improved:
            best_score = score
            best_gate_passed = gate_passed
            best_epoch = epoch + 1
            best_stage = stage
            patience = 0
        else:
            patience += 1
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "best_gate_passed": best_gate_passed,
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "patience": patience,
            "history": history,
            "checkpoint_stage": stage,
            "checkpoint_role": "last",
            "prototype_required": False,
            "prototype_ready": False,
            "prototype_pair_validated": False,
            "config": cfg,
        }
        atomic_torch_save(checkpoint, run_dir / "checkpoints" / "last.pt")
        if improved:
            checkpoint["checkpoint_role"] = "best_performance_candidate"
            checkpoint["selected_validation_metrics"] = validation_metrics
            checkpoint["selected_gate_checks"] = gate_checks
            atomic_torch_save(checkpoint, best_path)
        write_json(run_dir / "metrics" / "training_history.json", history)
        if stage == "selector" and patience >= int(cfg["training"]["patience"]):
            logger.info("验证集连续 %d 轮无提升，提前停止", patience)
            break

    if not best_path.is_file():
        raise RuntimeError("训练结束后没有可用的最优检查点")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])

    prototype_stats: dict[str, int] | None = None
    prototype_path = run_dir / "prototypes" / "problem3_prototype_memory.npz"
    if bool(cfg["training"].get("build_prototypes_after_training", True)):
        try:
            prototype_stats = build_prototype_memory(
                model,
                train_eval_loader,
                device,
                memory,
                int(cfg["model"]["prototypes_per_cell"]),
                float(cfg["training"].get("posthoc_prototype_min_confidence", 0.0)),
                require_correct=bool(cfg["training"].get("posthoc_prototype_require_correct", False)),
            )
            memory.save(prototype_path)
            logger.info("训练后构建只读解释原型库: %s", prototype_stats)
        except RuntimeError as exc:
            logger.warning("训练后原型库构建失败；预测仍可独立运行: %s", exc)

    best.update({
        "checkpoint_role": "best_performance_finalized",
        "prototype_required": False,
        "prototype_ready": memory.ready,
        "prototype_pair_validated": False,
        "prototype_path": str(prototype_path.resolve()) if memory.ready else "",
        "prototype_sha256": file_sha256(prototype_path) if memory.ready else "",
        "prototype_count": int(memory.embedding.shape[0]) if memory.ready else 0,
        "prototype_stats": prototype_stats,
    })
    atomic_torch_save(best, best_path)
    logger.info(
        "加载全阶段最优模型: epoch=%d stage=%s score=%.4f gate=%s",
        int(best["epoch"]) + 1,
        str(best.get("checkpoint_stage", "unknown")),
        float(best_score),
        best_gate_passed,
    )
    if not best_gate_passed:
        logger.warning(
            "最优模型未通过 P0 性能门槛，不应作为最终国奖方案提交；失败项=%s",
            best.get("selected_gate_checks", {}),
        )
    return {
        "best_score": best_score,
        "best_epoch": int(best["epoch"]) + 1,
        "best_stage": str(best.get("checkpoint_stage", "unknown")),
        "performance_gate_passed": best_gate_passed,
        "selected_validation_metrics": best.get("selected_validation_metrics", {}),
        "selected_gate_checks": best.get("selected_gate_checks", {}),
        "prototype_stats": prototype_stats,
        "prototype_count": int(memory.embedding.shape[0]) if memory.ready else 0,
        "history": history,
    }, memory
