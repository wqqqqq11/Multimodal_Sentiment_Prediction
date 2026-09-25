from __future__ import annotations

import logging
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from .config import load_config, resolve_paths
from .data import build_dataloader, load_splits
from .evaluation import attach_source_locations, evaluate_model
from .model import SEPCNet
from .prototypes import PrototypeMemory
from .reporting import error_attribution, export_result, write_explanation_cards, write_solution_summary
from .teacher import load_problem2_teacher
from .training import train_model, warm_start_text_backbone
from .utils import atomic_torch_save, choose_device, create_run_directory, file_sha256, set_seed, setup_logger, write_json
from .visualization import generate_visualizations


def _loaders(datasets: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    common = {
        "num_workers": int(data["num_workers"]), "pin_memory": bool(data["pin_memory"]),
        "seed": int(cfg["project"]["seed"]),
    }
    return {
        "train": build_dataloader(datasets["train"], batch_size=int(data["batch_size"]), shuffle=True, **common),
        "train_eval": build_dataloader(datasets["train"], batch_size=int(data["eval_batch_size"]), shuffle=False, **common),
        "valid": build_dataloader(datasets["valid"], batch_size=int(data["eval_batch_size"]), shuffle=False, **common),
        "test": build_dataloader(datasets["test"], batch_size=int(data["eval_batch_size"]), shuffle=False, **common),
        "attachment4": build_dataloader(datasets["attachment4"], batch_size=int(data["eval_batch_size"]), shuffle=False, **common),
    }


def _load_checkpoint(
    model: SEPCNet, memory: PrototypeMemory, checkpoint_path: Path, device: torch.device, logger: logging.Logger,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    configured_prototype = str(state.get("prototype_path", "")).strip()
    expected_sha256 = str(state.get("prototype_sha256", "")).strip().lower()
    rejected_candidates: list[str] = []
    candidates = [
        Path(configured_prototype) if configured_prototype else None,
        checkpoint_path.parent.parent / "prototypes" / "memory_best.npz",
        checkpoint_path.parent.parent / "prototypes" / "problem3_prototype_memory.npz",
        checkpoint_path.parent.parent / "prototypes" / "memory_current.npz",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            if expected_sha256 and file_sha256(candidate).lower() != expected_sha256:
                rejected_candidates.append(str(candidate))
                continue
            memory.load(candidate)
            expected_count = state.get("prototype_count")
            if expected_count is not None and int(memory.embedding.shape[0]) != int(expected_count):
                memory.clear()
                raise RuntimeError(
                    f"原型数量与 checkpoint 不一致: expected={expected_count} "
                    f"actual={int(memory.embedding.shape[0])}"
                )
            logger.info("已加载原型记忆: %s", candidate)
            break
    if not memory.ready and bool(state.get("prototype_required", False)):
        if rejected_candidates:
            raise RuntimeError(
                "原型库 SHA-256 与 checkpoint 不匹配，已拒绝加载: " + ", ".join(rejected_candidates)
            )
        raise RuntimeError("该旧检查点声明原型为必需组件，但没有找到匹配的原型库")
    if not memory.ready:
        logger.info("检查点未附带原型库；将按纯证据模型运行，预测不受影响")
    return state


def _export_deployment(model: SEPCNet, memory: PrototypeMemory, cfg: dict[str, Any], run_dir: Path) -> Path:
    prototype_path = run_dir / "prototypes" / "problem3_prototype_memory.npz"
    if memory.ready:
        memory.save(prototype_path)
    state = {
        key: value.detach().cpu().half() if torch.is_floating_point(value) else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    path = run_dir / "checkpoints" / "problem3_sepcnet_best_fp16.pt"
    atomic_torch_save({
        "model_state": state,
        "config": cfg,
        "precision": "float16",
        "checkpoint_stage": "best_any_stage",
        "checkpoint_role": "deployment_sparse_evidence_model",
        "prototype_required": False,
        "prototype_ready": memory.ready,
        "prototype_pair_validated": False,
        "prototype_path": str(prototype_path.resolve()) if memory.ready else "",
        "prototype_sha256": file_sha256(prototype_path) if memory.ready else "",
        "prototype_count": int(memory.embedding.shape[0]) if memory.ready else 0,
    }, path)
    return path


def _run_evaluations(
    model: SEPCNet, memory: PrototypeMemory, loaders: dict[str, Any], device: torch.device,
    cfg: dict[str, Any], run_dir: Path, logger: logging.Logger,
    teacher: Any | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for split in ("valid", "test", "attachment4"):
        logger.info("开始 %s 推理与解释计算", split)
        result = evaluate_model(
            model, loaders[split], device, cfg, memory if memory.ready else None,
            explain=True, teacher=teacher,
        )
        attach_source_locations(result, cfg, split)
        output = export_result(result, split, run_dir, cfg)
        card = write_explanation_cards(result, split, run_dir)
        results[split] = result
        artifacts[split] = {**output, "explanation_cards": str(card)}
        prototype_metrics = result["prototype_metrics"]
        prototype_coverage = prototype_metrics["evidence_coverage"]
        logger.info(
            "%s 原型覆盖: %d/%d (%s)", split,
            prototype_metrics["evidence_matched"], prototype_metrics["evidence_total"],
            f"{prototype_coverage:.4f}" if prototype_coverage is not None else "N/A",
        )
        if "metrics" in result:
            metrics = result["metrics"]
            logger.info(
                "%s: Acc=%.4f F1=%.4f MAE=%.4f Pearson=%.4f",
                split, metrics["accuracy"], metrics["macro_f1"], metrics["mae"], metrics["pearson"],
            )
    attribution = error_attribution(results["valid"])
    attribution_path = run_dir / "metrics" / "validation_error_attribution.json"
    write_json(attribution_path, attribution)
    artifacts["error_attribution"] = str(attribution_path)
    return {"results": results, "artifacts": artifacts, "attribution": attribution}


def run_pipeline(
    config_path: str | Path,
    *,
    stage: str = "all",
    checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    device_name: str = "auto",
    run_name: str | None = None,
    num_workers: int | None = None,
    warm_start: bool = True,
) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    raw_cfg = load_config(config_path)
    cfg = resolve_paths(raw_cfg, project_root)
    if num_workers is not None:
        cfg["data"]["num_workers"] = int(num_workers)
    output_root = Path(cfg["paths"]["modeling_output_root"])
    run_dir = create_run_directory(output_root, run_name)
    logger = setup_logger(run_dir)
    try:
        if stage not in {"all", "train", "evaluate", "infer"}:
            raise ValueError(f"未知 stage={stage}")
        set_seed(int(cfg["project"]["seed"]))
        device = choose_device(device_name)
        logger.info("Problem 3 pipeline stage=%s device=%s run=%s", stage, device, run_dir.name)
        shutil.copy2(config_path, run_dir / "reports" / "problem3_config_snapshot.yaml")
        write_json(run_dir / "reports" / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "device": str(device),
        })
        datasets = load_splits(cfg)
        loaders = _loaders(datasets, cfg)
        model = SEPCNet(cfg, text_model_path=cfg["paths"]["text_model"]).to(device)
        teacher = None
        if bool(cfg["training"].get("use_problem2_teacher", True)):
            teacher = load_problem2_teacher(
                Path(cfg["paths"]["problem2_checkpoint"]),
                Path(cfg["paths"]["problem2_config"]),
                Path(cfg["paths"]["text_model"]),
                device,
                logger,
            )
        memory = PrototypeMemory(
            int(cfg["model"]["hidden_dim"]), float(cfg["model"]["prototype_temperature"]),
            list(cfg["model"]["intensity_bin_centers"]),
        ).to(device)
        training_result: dict[str, Any] | None = None
        if stage in {"all", "train"}:
            if warm_start and resume is None:
                warm_start_text_backbone(model, Path(cfg["paths"]["problem2_checkpoint"]), logger)
            training_result, memory = train_model(
                model, loaders["train"], loaders["train_eval"], loaders["valid"], device, cfg,
                run_dir, logger, Path(resume).resolve() if resume else None, teacher=teacher,
            )
            write_json(run_dir / "metrics" / "training_summary.json", training_result)
            deployment_path = _export_deployment(model, memory, cfg, run_dir)
            logger.info("已保存部署模型: %s", deployment_path)
        else:
            if checkpoint is None:
                raise ValueError("evaluate/infer 阶段必须提供 --checkpoint")
            _load_checkpoint(model, memory, Path(checkpoint).resolve(), device, logger)

        artifacts: dict[str, Any] = {}
        if stage in {"all", "evaluate"}:
            evaluation = _run_evaluations(model, memory, loaders, device, cfg, run_dir, logger, teacher)
            artifacts.update(evaluation["artifacts"])
            history = training_result["history"] if training_result else []
            figures = generate_visualizations(
                run_dir, history, evaluation["results"]["valid"], evaluation["results"]["attachment4"], evaluation["attribution"],
            )
            artifacts["figures"] = figures
        elif stage == "infer":
            result = evaluate_model(
                model, loaders["attachment4"], device, cfg, memory if memory.ready else None,
                explain=True, teacher=teacher,
            )
            attach_source_locations(result, cfg, "attachment4")
            artifacts["attachment4"] = export_result(result, "attachment4", run_dir, cfg)
            artifacts["attachment4_cards"] = str(write_explanation_cards(result, "attachment4", run_dir))
        summary = write_solution_summary(run_dir, artifacts)
        write_json(output_root / "latest_run.json", {"status": "completed", "stage": stage, "run_dir": str(run_dir), "summary": str(summary)})
        logger.info("问题三求解完成: %s", run_dir)
        return run_dir
    except KeyboardInterrupt:
        write_json(run_dir / "failure.json", {"type": "KeyboardInterrupt", "message": "用户中断"})
        logger.warning("运行被用户中断")
        raise
    except Exception as exc:
        logger.exception("问题三求解失败")
        write_json(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        write_json(output_root / "latest_run.json", {"status": "failed", "stage": stage, "run_dir": str(run_dir), "error": str(exc)})
        raise
