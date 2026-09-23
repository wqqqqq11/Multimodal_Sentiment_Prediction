from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch

from .config import load_config, resolve_paths
from .data import Problem2Dataset, make_loader
from .evaluation import (
    evaluate_ablations,
    evaluate_robustness,
    missingness_attribution,
    predict,
    prediction_frame,
)
from .model import MRCDNet, count_parameters
from .reporting import write_solution_report
from .training import train_student, train_teacher, train_text_student
from .utils import choose_device, create_run_directory, set_seed, setup_logger, write_json
from .visualization import (
    plot_ablation,
    plot_challenge_predictions,
    plot_confusion_and_regression,
    plot_position_effect,
    plot_robustness_curves,
    plot_training_history,
)


def _build_datasets(cfg: dict[str, Any]) -> dict[str, Problem2Dataset]:
    root = Path(cfg["paths"]["preprocessed_root"])
    data_cfg = cfg["data"]
    return {
        "train": Problem2Dataset(root / data_cfg["train_file"], root / data_cfg["mask_bank_file"]),
        "valid": Problem2Dataset(root / data_cfg["valid_file"]),
        "test": Problem2Dataset(root / data_cfg["test_file"]),
        "challenge": Problem2Dataset(root / data_cfg["challenge_file"]),
    }


def _build_loaders(datasets: dict[str, Problem2Dataset], cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    seed = int(cfg["project"]["seed"])
    return {
        "train": make_loader(datasets["train"], int(data_cfg["batch_size"]), True, int(data_cfg["num_workers"]), bool(data_cfg["pin_memory"]), seed),
        "valid": make_loader(datasets["valid"], int(data_cfg["eval_batch_size"]), False, int(data_cfg["num_workers"]), bool(data_cfg["pin_memory"]), seed + 1),
        "test": make_loader(datasets["test"], int(data_cfg["eval_batch_size"]), False, int(data_cfg["num_workers"]), bool(data_cfg["pin_memory"]), seed + 2),
        "challenge": make_loader(datasets["challenge"], int(data_cfg["eval_batch_size"]), False, int(data_cfg["num_workers"]), bool(data_cfg["pin_memory"]), seed + 3),
    }


def _write_submission(challenge: dict[str, Any], output_path: Path) -> None:
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    frame = pd.DataFrame({
        "id": challenge["sample_id"],
        "sentiment_polarity": [label_names[int(value)] for value in challenge["prediction_class"]],
        "sentiment_intensity": challenge["prediction_regression"],
    })
    if len(frame) != 30:
        raise ValueError(f"附件3预测应为30条，实际为 {len(frame)} 条")
    if frame["id"].duplicated().any():
        raise ValueError("附件3预测存在重复 id")
    if not frame["sentiment_intensity"].between(-3.0, 3.0).all():
        raise ValueError("附件3预测强度超出 [-3,3]")
    frame.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.6f")


def _copy_submission_artifacts(run_dir: Path, output_root: Path) -> None:
    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "predictions" / "problem2_attachment3_predictions.csv", submission / "problem2_attachment3_predictions.csv")
    shutil.copy2(run_dir / "checkpoints" / "student_best.pt", submission / "problem2_student_best.pt")
    shutil.copy2(run_dir / "resolved_config.json", submission / "problem2_model_config.json")
    (submission / "README.md").write_text(
        "# 问题2提交材料\n\n"
        "- `problem2_attachment3_predictions.csv`：附件3共30条预测。\n"
        "- `problem2_student_best.pt`：MRCD-Net学生模型参数。\n"
        "- `problem2_model_config.json`：复现实验配置。\n\n"
        "主结果列：`id`、`sentiment_polarity`（Negative/Neutral/Positive）、`sentiment_intensity`（[-3,3]）。\n",
        encoding="utf-8",
    )


def run_solution(
    project_root: Path,
    config_path: Path,
    requested_device: str = "auto",
    smoke: bool = False,
    run_name: str | None = None,
) -> Path:
    overrides: dict[str, Any] = {}
    if smoke:
        overrides = {
            "training": {"teacher_epochs": 1, "text_distill_epochs": 1, "student_epochs": 1, "patience": 1},
            "evaluation": {"missing_rates": [0.20], "missing_positions": ["middle"]},
        }
    cfg = resolve_paths(load_config(config_path, overrides), project_root)
    output_root = Path(cfg["paths"]["output_root"])
    run_dir = create_run_directory(output_root, run_name)
    logger = setup_logger(run_dir)
    try:
        write_json(run_dir / "resolved_config.json", cfg)
        set_seed(int(cfg["project"]["seed"]))
        device = choose_device(requested_device)
        logger.info("[步骤1/4 数据输入] 读取对齐版预处理数据；只从附件2训练/验证集学习参数")
        datasets = _build_datasets(cfg)
        loaders = _build_loaders(datasets, cfg)
        logger.info(
            "数据规模：train=%d valid=%d test=%d attachment3=%d；DataLoader workers=%d",
            len(datasets["train"]), len(datasets["valid"]), len(datasets["test"]), len(datasets["challenge"]), cfg["data"]["num_workers"],
        )
        logger.info("[步骤2/4 参数初始化] seed=%d device=%s", cfg["project"]["seed"], device)
        teacher = MRCDNet(cfg["model"], privileged_text=True).to(device)
        student = MRCDNet(cfg["model"], privileged_text=False).to(device)
        privileged_path = Path(cfg["paths"]["privileged_text_init"])
        if not privileged_path.is_file():
            raise FileNotFoundError(f"缺少P0特权文本初始化文件，请重新运行问题2预处理: {privileged_path}")
        with np.load(privileged_path, allow_pickle=False) as privileged:
            projection = torch.from_numpy(privileged["projection"]).to(device)
            prototypes = torch.from_numpy(privileged["token_prototypes"]).to(device)
            raw_prototypes = torch.from_numpy(privileged["token_prototypes_raw"]).to(device)
            token_counts = torch.from_numpy(privileged["token_counts"]).to(device)
        teacher.initialize_privileged_text(projection)
        student.initialize_privileged_text(projection)
        student.initialize_student_tokens(prototypes, token_counts)
        parameter_count, trainable_count = count_parameters(student)
        logger.info("MRCD-Net 参数：total=%s trainable=%s", f"{parameter_count:,}", f"{trainable_count:,}")
        logger.info("[步骤3/4 模型调用] 开始训练完整教师")
        teacher_history = train_teacher(teacher, loaders["train"], loaders["valid"], cfg, device, run_dir, logger)
        transferable = {
            key: value for key, value in teacher.state_dict().items()
            if not key.startswith("text_encoder.") and not key.startswith("privileged_text_encoder.")
        }
        student.load_state_dict(transferable, strict=False)
        with torch.no_grad():
            mapped_parts = []
            for start in range(0, raw_prototypes.shape[0], 2048):
                mapped_parts.append(teacher.privileged_text_encoder(raw_prototypes[start:start + 2048]).float())
            final_teacher_prototypes = torch.cat(mapped_parts, dim=0)
        student.initialize_student_tokens(final_teacher_prototypes, token_counts)
        logger.info("P0文本特权蒸馏：赛方预计算text教师 -> 可部署text_bert学生")
        text_history = train_text_student(student, teacher, loaders["train"], loaders["valid"], cfg, device, run_dir, logger)
        logger.info("开始训练缺失学生：由最佳教师初始化并执行完整—缺失一致性蒸馏")
        student_history = train_student(student, teacher, loaders["train"], loaders["valid"], cfg, device, run_dir, logger)
        history = pd.DataFrame(teacher_history + text_history + student_history)
        history.to_csv(run_dir / "metrics" / "training_history.csv", index=False, encoding="utf-8-sig")
        plot_training_history(history, run_dir / "figures" / "training_history.png")

        logger.info("[步骤4/4 结果输出] 计算验证/测试性能、缺失网格、消融与附件3预测")
        teacher_validation = predict(teacher, loaders["valid"], device)
        student_validation = predict(student, loaders["valid"], device)
        test_result = predict(student, loaders["test"], device)
        challenge_result = predict(student, loaders["challenge"], device)
        target_cfg = cfg["evaluation"]["target_scenario"]
        target_scenario = (str(target_cfg["pattern"]), float(target_cfg["rate"]), str(target_cfg["position"]))
        target_result = predict(student, loaders["valid"], device, target_scenario)
        prediction_frame(student_validation).to_csv(run_dir / "predictions" / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        prediction_frame(test_result).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False, encoding="utf-8-sig")
        challenge_diagnostics = prediction_frame(challenge_result, include_labels=False)
        challenge_diagnostics.to_csv(run_dir / "predictions" / "problem2_attachment3_diagnostics.csv", index=False, encoding="utf-8-sig")
        _write_submission(challenge_result, run_dir / "predictions" / "problem2_attachment3_predictions.csv")

        eval_cfg = cfg["evaluation"]
        robustness = evaluate_robustness(
            student, loaders["valid"], device,
            eval_cfg["missing_patterns"], eval_cfg["missing_rates"], eval_cfg["missing_positions"],
        )
        robustness.to_csv(run_dir / "metrics" / "robustness_results.csv", index=False, encoding="utf-8-sig")
        attribution = missingness_attribution(robustness)
        write_json(run_dir / "metrics" / "missingness_attribution.json", attribution)
        ablation = evaluate_ablations(student, teacher, loaders["valid"], device)
        ablation.to_csv(run_dir / "metrics" / "ablation_results.csv", index=False, encoding="utf-8-sig")
        summary = {
            "run_id": run_dir.name,
            "device": str(device),
            "parameter_count": parameter_count,
            "teacher_validation": teacher_validation["metrics"],
            "student_validation": student_validation["metrics"],
            "student_test": test_result["metrics"],
            "target_synchronized_30": target_result["metrics"],
            "target_thresholds": {
                "minimum_accuracy": target_cfg["minimum_accuracy"],
                "minimum_macro_f1": target_cfg["minimum_macro_f1"],
                "maximum_mae": target_cfg["maximum_mae"],
            },
            "attachment3_count": len(challenge_result["sample_id"]),
            "smoke": smoke,
        }
        summary["target_passed"] = bool(
            float(target_result["metrics"]["accuracy"]) >= float(target_cfg["minimum_accuracy"])
            and float(target_result["metrics"]["macro_f1"]) >= float(target_cfg["minimum_macro_f1"])
            and float(target_result["metrics"]["mae"]) <= float(target_cfg["maximum_mae"])
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        plot_robustness_curves(robustness, run_dir / "figures" / "missing_rate_effect.png", eval_cfg["primary_position"])
        plot_position_effect(robustness, run_dir / "figures" / "missing_position_effect.png")
        plot_ablation(ablation, run_dir / "figures" / "ablation_comparison.png")
        plot_confusion_and_regression(student_validation, run_dir / "figures" / "validation_performance.png", "验证集")
        plot_confusion_and_regression(test_result, run_dir / "figures" / "test_performance.png", "测试集")
        plot_challenge_predictions(challenge_diagnostics, run_dir / "figures" / "attachment3_predictions.png")
        write_solution_report(
            run_dir / "reports" / "model_solution_report.md", run_dir.name, str(device), parameter_count,
            student_validation["metrics"], test_result["metrics"], robustness, attribution,
        )
        write_json(output_root / "latest_run.json", {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "completed", "smoke": smoke})
        if not smoke:
            _copy_submission_artifacts(run_dir, output_root)
        logger.info(
            "求解完成：valid Acc=%.4f F1=%.4f MAE=%.4f r=%.4f；结果目录=%s",
            student_validation["metrics"]["accuracy"], student_validation["metrics"]["macro_f1"],
            student_validation["metrics"]["mae"], student_validation["metrics"]["pearson"], run_dir,
        )
        logger.info(
            "目标验收（三模态同步缺失30%%）：Acc=%.4f/%0.4f F1=%.4f/%0.4f MAE=%.4f/%0.4f，状态=%s",
            target_result["metrics"]["accuracy"], target_cfg["minimum_accuracy"],
            target_result["metrics"]["macro_f1"], target_cfg["minimum_macro_f1"],
            target_result["metrics"]["mae"], target_cfg["maximum_mae"],
            "PASS" if summary["target_passed"] else "NOT_REACHED",
        )
        return run_dir
    except Exception as exc:
        logger.exception("问题2求解失败：%s", exc)
        write_json(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        write_json(output_root / "latest_run.json", {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "failed"})
        raise
