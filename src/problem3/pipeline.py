from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config, resolve_paths
from .data import Problem3Dataset
from .evaluation import predict, prediction_frame
from .explain import explain_sample, load_evidence_mapping
from .metrics import fit_calibration
from .model import HSAIGNet, count_parameters
from .training import train_model
from .data import load_baselines, make_loader
from .reporting import _error_analysis, _explanation_frames
from .utils import atomic_torch_save, choose_device, create_run_directory, set_seed, setup_logger, write_json
from .visualization import (plot_explanation_card, plot_modality_contributions, plot_training_history,
                            plot_validation)


def _datasets(cfg: dict[str, Any]) -> dict[str, Problem3Dataset]:
    base, features, data = Path(cfg["paths"]["preprocessed_root"]), Path(cfg["paths"]["feature_root"]), cfg["data"]
    pairs = {"train": ("train_file", "train_text_file"), "valid": ("valid_file", "valid_text_file"),
             "test": ("test_file", "test_text_file"),
             "attachment4": ("attachment4_file", "attachment4_text_file")}
    missing = [features / data[text_key] for _, text_key in pairs.values() if not (features / data[text_key]).is_file()]
    if missing:
        raise FileNotFoundError("缺少文本特征，请先运行 python -m data_progressing.problem3.build_text_features：" +
                                ", ".join(str(path) for path in missing))
    return {name: Problem3Dataset(base / data[base_key], features / data[text_key])
            for name, (base_key, text_key) in pairs.items()}


def _loaders(datasets: dict[str, Problem3Dataset], cfg: dict[str, Any]) -> dict[str, Any]:
    data, seed = cfg["data"], int(cfg["project"]["seed"])
    return {name: make_loader(dataset, batch_size=int(data["batch_size"] if name == "train" else data["eval_batch_size"]),
                              shuffle=name == "train", num_workers=int(data["num_workers"]),
                              pin_memory=bool(data["pin_memory"]), seed=seed + index, balanced=False)
            for index, (name, dataset) in enumerate(datasets.items())}


def _statistics(cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    base, features, data = Path(cfg["paths"]["preprocessed_root"]), Path(cfg["paths"]["feature_root"]), cfg["data"]
    result = load_baselines(base / data["baseline_file"])
    with np.load(features / data["text_baseline_file"], allow_pickle=False) as archive:
        result.update({key: torch.from_numpy(np.asarray(archive[key], dtype=np.float32)) for key in archive.files})
    return result


def _export(model: HSAIGNet, run_dir: Path, output_root: Path, cfg: dict[str, Any]) -> float:
    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
             for key, value in model.state_dict().items()}
    target = submission / "problem3_hsaig_fp16.pt"
    atomic_torch_save({"model_state": state, "precision": "float16", "architecture": "HSAIG-Net-v3-wide"}, target)
    size = target.stat().st_size / 1024 ** 2
    if size > float(cfg["output"]["max_submission_model_mb"]):
        target.unlink(); raise RuntimeError(f"模型权重 {size:.2f} MiB 超过上限")
    for source in (run_dir / "predictions" / "problem3_attachment4_predictions_and_explanations.csv",
                   run_dir / "explanations" / "attachment4_evidence.csv"):
        shutil.copy2(source, submission / source.name)
    shutil.copy2(run_dir / "resolved_config.json", submission / "problem3_model_config.json")
    return size


def run_solution(project_root: Path, config_path: Path, requested_device: str = "auto", smoke: bool = False,
                 run_name: str | None = None) -> Path:
    overrides = ({"training": {"epochs": 1, "patience": 1},
                  "explanation": {"integrated_gradient_steps": 4, "typical_card_count": 2}}
                 if smoke else {})
    cfg = resolve_paths(load_config(config_path, overrides), project_root)
    output_root = Path(cfg["paths"]["output_root"])
    run_dir = create_run_directory(output_root, run_name or ("problem3_smoke" if smoke else None))
    logger = setup_logger(run_dir); write_json(run_dir / "resolved_config.json", cfg)
    try:
        set_seed(int(cfg["project"]["seed"])); device = choose_device(requested_device)
        datasets, loaders = _datasets(cfg), None
        loaders = _loaders(datasets, cfg)
        model = HSAIGNet(cfg["model"]).to(device)
        total, trainable = count_parameters(model)
        logger.info("HSAIG-Net-v3-wide train=%d valid=%d test=%d attachment4=%d device=%s params=%s",
                    *(len(datasets[name]) for name in ("train", "valid", "test", "attachment4")), device, f"{total:,}")
        history, checkpoint = train_model(model, loaders["train"], loaders["valid"], cfg, device, run_dir, logger)
        raw_valid = predict(model, loaders["valid"], device, goals=cfg["evaluation"]["goals"])
        calibration = fit_calibration(raw_valid["logits"], raw_valid["prediction_regression_raw"],
                                            raw_valid["label_class"], raw_valid["label_regression"],
                                            cfg["evaluation"]["calibration"], cfg["evaluation"]["goals"])
        write_json(run_dir / "metrics" / "calibration.json", calibration)
        validation = predict(model, loaders["valid"], device, goals=cfg["evaluation"]["goals"], calibration=calibration)
        test = predict(model, loaders["test"], device, goals=cfg["evaluation"]["goals"], calibration=calibration)
        attachment = predict(model, loaders["attachment4"], device, calibration=calibration)
        valid_frame, test_frame = prediction_frame(validation), prediction_frame(test)
        attachment_frame = prediction_frame(attachment, include_labels=False)
        valid_frame.to_csv(run_dir / "predictions" / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        test_frame.to_csv(run_dir / "predictions" / "test_predictions.csv", index=False, encoding="utf-8-sig")
        _error_analysis(valid_frame).to_csv(run_dir / "metrics" / "error_analysis.csv", index=False, encoding="utf-8-sig")
        statistics = _statistics(cfg)
        mapping = load_evidence_mapping(Path(cfg["paths"]["preprocessed_root"]) / cfg["data"]["time_mapping_file"])
        explanations = [explain_sample(model, datasets["attachment4"][index], device, statistics,
                                       cfg["explanation"], calibration, mapping)
                        for index in range(min(2, len(datasets["attachment4"])) if smoke else len(datasets["attachment4"]))]
        summary_frame, evidence_frame = _explanation_frames(explanations)
        drop = ["predicted_polarity", "predicted_label", "predicted_intensity", "prediction_confidence"]
        attachment_frame.merge(summary_frame.drop(columns=drop), on="sample_id", how="left").to_csv(
            run_dir / "predictions" / "problem3_attachment4_predictions_and_explanations.csv",
            index=False, encoding="utf-8-sig")
        evidence_frame.to_csv(run_dir / "explanations" / "attachment4_evidence.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "explanations" / "attachment4_explanations.json", explanations)
        plot_training_history(history, run_dir / "figures" / "training_history.png")
        plot_validation(validation, run_dir / "figures" / "validation_performance.png")
        plot_modality_contributions(explanations, run_dir / "figures" / "attachment4_modality_contributions.png")
        for explanation in explanations[:int(cfg["explanation"]["typical_card_count"])]:
            plot_explanation_card(explanation, run_dir / "figures" / f"explanation_card_{explanation['sample_id']}.png")
        accepted = bool(validation["metrics"]["goal_audit"]["all_met"])
        model_size = None if smoke else _export(model, run_dir, output_root, cfg)
        audit = {"accepted": accepted, "submission_exported": model_size is not None,
                 "reason": ("四项验证集目标全部满足，已导出最佳模型" if accepted else
                            "四项验证指标未全部满足，仍已导出当前最佳模型与附件4结果"),
                 "validation": validation["metrics"], "test": test["metrics"]}
        write_json(run_dir / "metrics" / "goal_audit.json", audit)
        write_json(run_dir / "metrics" / "summary.json", {
            "run_id": run_dir.name, "device": str(device), "parameters": total, "trainable_parameters": trainable,
            "estimated_fp16_mib": total * 2 / 1024 ** 2, "exported_model_mib": model_size,
            "best_epoch": int(checkpoint["epoch"]), "validation": validation["metrics"], "test": test["metrics"],
            "calibration": calibration, "accepted": accepted, "smoke": smoke})
        write_json(output_root / "latest_model_run.json", {"run_id": run_dir.name, "run_dir": str(run_dir),
                                                                    "status": "accepted" if accepted else "goals_not_met"})
        logger.info("验收：Acc=%.4f Macro-F1=%.4f MAE=%.4f Pearson=%.4f；%d/4，accepted=%s",
                    validation["metrics"]["accuracy"], validation["metrics"]["macro_f1"],
                    validation["metrics"]["mae"], validation["metrics"]["pearson"],
                    validation["metrics"]["goal_audit"]["met_count"], accepted)
        return run_dir
    except Exception as exc:
        logger.exception("问题三流水线失败：%s", exc)
        write_json(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
