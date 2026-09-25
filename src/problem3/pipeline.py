from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config, resolve_paths
from .data import Problem3Dataset, load_baselines, make_loader
from .evaluation import predict, prediction_frame
from .explain import explain_sample, load_evidence_mapping
from .metrics import fit_calibration
from .model import HSAIGNet, count_parameters
from .training import train_model
from .utils import (
    atomic_torch_save,
    choose_device,
    create_run_directory,
    set_seed,
    setup_logger,
    write_json,
)
from .visualization import (
    plot_explanation_card,
    plot_modality_contributions,
    plot_training_history,
    plot_validation,
)


def _datasets(cfg: dict[str, Any]) -> dict[str, Problem3Dataset]:
    root = Path(cfg["paths"]["preprocessed_root"])
    data = cfg["data"]
    return {
        "train": Problem3Dataset(root / data["train_file"]),
        "valid": Problem3Dataset(root / data["valid_file"]),
        "test": Problem3Dataset(root / data["test_file"]),
        "attachment4": Problem3Dataset(root / data["attachment4_file"]),
    }


def _loaders(datasets: dict[str, Problem3Dataset], cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    seed = int(cfg["project"]["seed"])
    return {
        name: make_loader(
            dataset,
            batch_size=int(data["batch_size"] if name == "train" else data["eval_batch_size"]),
            shuffle=name == "train",
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
            seed=seed + index,
            balanced=name == "train" and bool(data.get("class_balanced_sampler", False)),
        )
        for index, (name, dataset) in enumerate(datasets.items())
    }


def _explanation_frames(explanations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for row in explanations:
        contribution = row["modality_contribution"]
        top = sorted(row["key_evidence"], key=lambda item: item["position_score"], reverse=True)[:5]
        summary_rows.append({
            "sample_id": row["sample_id"],
            "predicted_polarity": row["predicted_polarity"],
            "predicted_label": row["predicted_label"],
            "predicted_intensity": row["predicted_intensity"],
            "prediction_confidence": row["prediction_confidence"],
            "main_modality": row["main_modality"],
            "text_contribution": contribution["text"],
            "audio_contribution": contribution["audio"],
            "vision_contribution": contribution["vision"],
            "comprehensiveness": row["comprehensiveness"],
            "sufficiency": row["sufficiency"],
            "stability": row["stability"],
            "attention_ig_agreement": row["attention_ig_agreement"],
            "key_evidence": " | ".join(
                f"{item['modality']}@{item['position']}:{item['text_span']}"
                for item in top
            ),
        })
        evidence_rows.extend(row["key_evidence"])
    return pd.DataFrame(summary_rows), pd.DataFrame(evidence_rows)


def _error_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("true_polarity"):
        rows.append({
            "group": f"class_{int(label)}",
            "samples": len(group),
            "classification_accuracy": float(group["classification_correct"].mean()),
            "mean_absolute_error": float(group["absolute_error"].mean()),
            "mean_confidence": float(group["prediction_confidence"].mean()),
        })
    hardest = frame.sort_values(["classification_correct", "absolute_error"], ascending=[True, False]).head(30)
    for _, row in hardest.iterrows():
        rows.append({
            "group": "hard_sample",
            "sample_id": row["sample_id"],
            "true_polarity": row["true_polarity"],
            "predicted_polarity": row["predicted_polarity"],
            "true_intensity": row["true_intensity"],
            "predicted_intensity": row["predicted_intensity"],
            "classification_accuracy": float(row["classification_correct"]),
            "mean_absolute_error": float(row["absolute_error"]),
            "mean_confidence": float(row["prediction_confidence"]),
        })
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    validation: dict[str, Any],
    test: dict[str, Any],
    explanations: list[dict[str, Any]],
    parameter_count: int,
    calibration: dict[str, Any],
) -> None:
    valid = validation["metrics"]
    heldout = test["metrics"]
    dominant = pd.Series([row["main_modality"] for row in explanations]).value_counts().to_dict()
    text = f"""# 问题三层次稀疏注意力与积分梯度模型运行报告

## 运行结论

模型参数量为 {parameter_count:,}。结构由三模态时序编码器、模态内Sparsemax时间注意力、样本级Sparsemax模态门控、分类回归双头、积分梯度复核以及删除和保留实验组成。结构与阈值选择仅使用附件2验证集；附件4不参与训练和调参。

| 数据划分 | Accuracy | Macro-F1 | MAE | Pearson | 中性Recall | 达标数 |
|---|---:|---:|---:|---:|---:|---:|
| 验证集 | {valid['accuracy']:.4f} | {valid['macro_f1']:.4f} | {valid['mae']:.4f} | {valid['pearson']:.4f} | {valid['neutral_recall']:.4f} | {valid['goal_audit']['met_count']}/5 |
| 独立测试集 | {heldout['accuracy']:.4f} | {heldout['macro_f1']:.4f} | {heldout['mae']:.4f} | {heldout['pearson']:.4f} | {heldout['neutral_recall']:.4f} | {heldout['goal_audit']['met_count']}/5 |

验证集校准参数为温度 {calibration['temperature']:.3f}、中性logit偏置 {calibration['class_bias'][1]:.3f}、回归斜率 {calibration['regression_slope']:.3f}、截距 {calibration['regression_intercept']:.3f}。这些参数没有使用独立测试集或附件4标签。

## 解释结果

附件4共生成 {len(explanations)} 条预测与解释。主要参考模态计数为 `{json.dumps(dominant, ensure_ascii=False)}`。每条解释同时保留模型门控、积分梯度、删除效应、局部证据位置、comprehensiveness、sufficiency、跨基线stability以及注意力和积分梯度排序一致性。

音视频位置使用预处理映射表。当前比例回退映射的置信度较低，论文和结果文件中必须披露，不能把它描述成精确强制对齐边界。

## 输出索引

- `predictions/validation_predictions.csv`：验证集逐样本结果和错误字段。
- `predictions/test_predictions.csv`：附件2独立测试集结果。
- `predictions/problem3_attachment4_predictions_and_explanations.csv`：附件4全量主结果。
- `explanations/attachment4_evidence.csv`：每个模态的局部证据位置与原始材料映射。
- `explanations/attachment4_explanations.json`：注意力、积分梯度和忠实度完整记录。
- `metrics/goal_audit.json`：五项目标逐项验收。
- `metrics/error_analysis.csv`：验证集分组错误和困难样本。
- `figures/`：训练、性能、模态贡献与典型解释卡。
"""
    path.write_text(text, encoding="utf-8")


def _export_submission(
    model: HSAIGNet,
    run_dir: Path,
    output_root: Path,
    cfg: dict[str, Any],
) -> None:
    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    model_path = submission / "problem3_hsaig_fp16.pt"
    atomic_torch_save({"model_state": state, "precision": "float16"}, model_path)
    size_mb = model_path.stat().st_size / (1024 ** 2)
    if size_mb > float(cfg["output"]["max_submission_model_mb"]):
        raise RuntimeError(f"问题三部署权重 {size_mb:.2f} MiB 超过配置上限")
    shutil.copy2(run_dir / "predictions" / "problem3_attachment4_predictions_and_explanations.csv", submission)
    shutil.copy2(run_dir / "explanations" / "attachment4_evidence.csv", submission)
    shutil.copy2(run_dir / "resolved_config.json", submission / "problem3_model_config.json")
    (submission / "README.md").write_text(
        "# 问题3提交材料\n\n"
        "包含FP16模型权重、附件4全量预测与解释、局部证据表和复现配置。"
        "附件4仅用于最终推理，所有结构、阈值与校准参数均由附件2训练/验证集确定。\n",
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
            "training": {"epochs": 1, "patience": 1, "unfreeze_text_epoch": 99},
            "explanation": {"integrated_gradient_steps": 4, "typical_card_count": 2},
        }
    cfg = resolve_paths(load_config(config_path, overrides), project_root)
    output_root = Path(cfg["paths"]["output_root"])
    run_dir = create_run_directory(output_root, run_name)
    logger = setup_logger(run_dir)
    write_json(run_dir / "resolved_config.json", cfg)
    try:
        set_seed(int(cfg["project"]["seed"]))
        device = choose_device(requested_device)
        datasets = _datasets(cfg)
        loaders = _loaders(datasets, cfg)
        logger.info(
            "数据规模 train=%d valid=%d test=%d attachment4=%d；device=%s",
            len(datasets["train"]), len(datasets["valid"]), len(datasets["test"]), len(datasets["attachment4"]), device,
        )
        model = HSAIGNet(cfg["model"], cfg["paths"]["text_pretrained_model"]).to(device)
        total_parameters, trainable_parameters = count_parameters(model)
        logger.info("HSAIG-Net参数 total=%s trainable=%s", f"{total_parameters:,}", f"{trainable_parameters:,}")
        history, checkpoint = train_model(model, loaders["train"], loaders["valid"], cfg, device, run_dir, logger)
        raw_valid = predict(model, loaders["valid"], device, goals=cfg["evaluation"]["goals"])
        calibration = fit_calibration(
            raw_valid["logits"], raw_valid["prediction_regression_raw"],
            raw_valid["label_class"], raw_valid["label_regression"],
            cfg["evaluation"]["calibration"], cfg["evaluation"]["goals"],
        )
        write_json(run_dir / "metrics" / "calibration.json", calibration)
        validation = predict(model, loaders["valid"], device, goals=cfg["evaluation"]["goals"], calibration=calibration)
        test = predict(model, loaders["test"], device, goals=cfg["evaluation"]["goals"], calibration=calibration)
        attachment4 = predict(model, loaders["attachment4"], device, calibration=calibration)
        valid_frame = prediction_frame(validation)
        test_frame = prediction_frame(test)
        attachment_frame = prediction_frame(attachment4, include_labels=False)
        valid_frame.to_csv(run_dir / "predictions" / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        test_frame.to_csv(run_dir / "predictions" / "test_predictions.csv", index=False, encoding="utf-8-sig")
        _error_analysis(valid_frame).to_csv(run_dir / "metrics" / "error_analysis.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "metrics" / "goal_audit.json", {
            "validation": validation["metrics"]["goal_audit"],
            "test": test["metrics"]["goal_audit"],
            "validation_metrics": validation["metrics"],
            "test_metrics": test["metrics"],
        })
        preprocessed = Path(cfg["paths"]["preprocessed_root"])
        baselines = load_baselines(preprocessed / cfg["data"]["baseline_file"])
        mapping = load_evidence_mapping(preprocessed / cfg["data"]["time_mapping_file"])
        explanation_limit = 2 if smoke else len(datasets["attachment4"])
        explanations: list[dict[str, Any]] = []
        logger.info("开始附件4积分梯度与删除/保留复核：%d条", explanation_limit)
        for index in range(explanation_limit):
            explanation = explain_sample(
                model, datasets["attachment4"][index], device, baselines,
                cfg["explanation"], calibration, mapping,
            )
            explanations.append(explanation)
            logger.info(
                "解释 %s | main=%s | comp=%.3f suff=%.3f stability=%.3f",
                explanation["sample_id"], explanation["main_modality"], explanation["comprehensiveness"],
                explanation["sufficiency"], explanation["stability"],
            )
        summary_frame, evidence_frame = _explanation_frames(explanations)
        attachment_output = attachment_frame.merge(summary_frame.drop(columns=["predicted_polarity", "predicted_label", "predicted_intensity", "prediction_confidence"]), on="sample_id", how="left")
        attachment_output.to_csv(
            run_dir / "predictions" / "problem3_attachment4_predictions_and_explanations.csv",
            index=False, encoding="utf-8-sig",
        )
        evidence_frame.to_csv(run_dir / "explanations" / "attachment4_evidence.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "explanations" / "attachment4_explanations.json", explanations)
        plot_training_history(history, run_dir / "figures" / "training_history.png")
        plot_validation(validation, run_dir / "figures" / "validation_performance.png")
        plot_modality_contributions(explanations, run_dir / "figures" / "attachment4_modality_contributions.png")
        for explanation in explanations[: int(cfg["explanation"]["typical_card_count"])]:
            plot_explanation_card(
                explanation, run_dir / "figures" / f"explanation_card_{explanation['sample_id']}.png"
            )
        summary = {
            "run_id": run_dir.name,
            "device": str(device),
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_epoch": int(checkpoint["epoch"]),
            "validation": validation["metrics"],
            "test": test["metrics"],
            "attachment4_count": len(attachment4["sample_id"]),
            "explained_attachment4_count": len(explanations),
            "calibration": calibration,
            "smoke": smoke,
        }
        write_json(run_dir / "metrics" / "summary.json", summary)
        _write_report(
            run_dir / "reports" / "model_solution_report.md", validation, test, explanations,
            total_parameters, calibration,
        )
        if not smoke:
            _export_submission(model, run_dir, output_root, cfg)
        write_json(output_root / "latest_model_run.json", {
            "run_id": run_dir.name, "run_dir": str(run_dir), "status": "completed", "smoke": smoke,
        })
        logger.info(
            "完成：valid Acc=%.4f F1=%.4f MAE=%.4f r=%.4f Neutral-R=%.4f；达标%d/5",
            validation["metrics"]["accuracy"], validation["metrics"]["macro_f1"],
            validation["metrics"]["mae"], validation["metrics"]["pearson"],
            validation["metrics"]["neutral_recall"], validation["metrics"]["goal_audit"]["met_count"],
        )
        return run_dir
    except Exception as exc:
        logger.exception("问题三求解失败：%s", exc)
        write_json(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        write_json(output_root / "latest_model_run.json", {
            "run_id": run_dir.name, "run_dir": str(run_dir), "status": "failed", "smoke": smoke,
        })
        raise
