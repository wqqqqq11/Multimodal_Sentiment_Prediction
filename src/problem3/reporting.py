from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .utils import json_safe, write_json, write_jsonl


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_safe(rows))
    os.replace(temporary, path)


def _quality_index(cfg: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(cfg["paths"]["preprocessed_root"]) / cfg["data"]["quality_file"]
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {(row["split"], row["sample_id"]): row for row in csv.DictReader(handle)}


def enrich_quality_flags(result: dict[str, Any], cfg: dict[str, Any], split: str) -> None:
    quality = _quality_index(cfg)
    for row in result["predictions"]:
        source = quality.get((split, str(row["sample_id"])), {})
        flags: list[str] = []
        if source.get("truncation_flag", "False").lower() == "true":
            flags.append("text_truncated")
        if int(float(source.get("vision_failure_steps", 0) or 0)) > 0:
            flags.append("vision_failure")
        if int(float(source.get("audio_failure_steps", 0) or 0)) > 0:
            flags.append("audio_failure")
        confidence = float(source.get("token_mapping_confidence", 1.0) or 1.0)
        if confidence < 0.8:
            flags.append("low_mapping_confidence")
        row["quality_flags"] = ";".join(flags) if flags else "none"


def export_result(result: dict[str, Any], split: str, run_dir: Path, cfg: dict[str, Any]) -> dict[str, str]:
    enrich_quality_flags(result, cfg, split)
    predictions_path = run_dir / "predictions" / f"{split}_predictions.csv"
    evidence_path = run_dir / "explanations" / f"{split}_evidence.csv"
    _write_csv(predictions_path, result["predictions"])
    _write_csv(evidence_path, result["evidence"])
    output = {"predictions": str(predictions_path), "evidence": str(evidence_path)}
    if split == "attachment4":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evidence in result["evidence"]:
            grouped[str(evidence["sample_id"])].append(evidence)
        combined: list[dict[str, Any]] = []
        jsonl: list[dict[str, Any]] = []
        for prediction in result["predictions"]:
            sample_id = str(prediction["sample_id"])
            evidence = sorted(grouped[sample_id], key=lambda row: int(row["rank"]))
            summary = []
            for item in evidence:
                if item["modality"] == "text":
                    location = f"chars[{item.get('char_start','')}:{item.get('char_end','')}]={item.get('text_span','')}"
                elif item["modality"] == "audio":
                    location = f"{item.get('start_sec','')}-{item.get('end_sec','')}s"
                else:
                    location = f"frames[{item.get('start_frame','')}:{item.get('end_frame','')}]"
                summary.append(f"{item['rank']}:{item['modality']}@{location}|importance={float(item['local_importance']):.6f}")
            row = {**prediction, "key_evidence": "; ".join(summary)}
            combined.append(row)
            jsonl.append({"prediction": prediction, "evidence": evidence})
        combined_path = run_dir / "predictions" / "attachment4_prediction_and_explanation.csv"
        jsonl_path = run_dir / "explanations" / "attachment4_explanations.jsonl"
        _write_csv(combined_path, combined)
        write_jsonl(jsonl_path, jsonl)
        output.update({"combined": str(combined_path), "jsonl": str(jsonl_path)})
        total_size = sum(Path(path).stat().st_size for path in output.values())
        if total_size > int(cfg["evaluation"]["max_submission_bytes"]):
            raise RuntimeError(f"附件4结果文件总计 {total_size} bytes，超过配置上限")
    if "metrics" in result:
        metrics_path = run_dir / "metrics" / f"{split}_metrics.json"
        write_json(metrics_path, {
            "metrics": result["metrics"], "fidelity_metrics": result["fidelity_metrics"],
            "prototype_metrics": result["prototype_metrics"],
        })
        output["metrics"] = str(metrics_path)
    return output


def error_attribution(result: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["predictions"]:
        if "true_class" not in row:
            continue
        flags = str(row.get("quality_flags", "none")).split(";")
        for flag in flags:
            groups[flag].append(row)
        groups["all"].append(row)
    output: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        output.append({
            "group": group, "samples": len(rows),
            "classification_error_rate": float(np.mean([not bool(row["classification_correct"]) for row in rows])),
            "intensity_mae": float(np.mean([float(row["absolute_error"]) for row in rows])),
            "mean_sufficiency_gap": float(np.mean([float(row["sufficiency_probability_gap"]) for row in rows])),
        })
    return output


def write_explanation_cards(result: dict[str, Any], split: str, run_dir: Path, count: int = 6) -> Path:
    predictions = result["predictions"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["evidence"]:
        grouped[str(row["sample_id"])].append(row)
    if not predictions:
        chosen: list[dict[str, Any]] = []
    elif "classification_correct" in predictions[0]:
        ordered = sorted(predictions, key=lambda row: (bool(row["classification_correct"]), -float(row["absolute_error"])))
        indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
        chosen = [ordered[index] for index in indices]
    else:
        confidence = lambda row: max(float(row[f"class_probability_{name}"]) for name in ("negative", "neutral", "positive"))
        ordered = sorted(predictions, key=confidence)
        indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
        chosen = [ordered[index] for index in indices]
    lines = [f"# {split} 典型样本解释卡", "", "解释卡中的局部重要性来自逐证据删除，模态作用程度来自三模态精确 Shapley；音视频位置映射置信度需与定位结果一并解读。", ""]
    for item in chosen:
        sample_id = str(item["sample_id"])
        lines.extend([
            f"## 样本 {sample_id}", "",
            f"- 预测：{item['predicted_polarity']}，强度 {float(item['predicted_intensity']):.4f}",
            f"- 主要参考模态：{item['main_modality']}",
            f"- 模态作用：文本 {float(item['text_contribution']):.3f}，音频 {float(item['audio_contribution']):.3f}，视觉 {float(item['vision_contribution']):.3f}",
            f"- 质量标记：{item.get('quality_flags', 'none')}", "", "|排名|模态|位置|局部重要性|原型样本|原型相似度|可回看定位|", "|---:|---|---:|---:|---|---:|---|",
        ])
        for evidence in sorted(grouped[sample_id], key=lambda row: int(row["rank"])):
            if evidence["modality"] == "text":
                location = f"字符 {evidence.get('char_start','')}-{evidence.get('char_end','')}：{evidence.get('text_span','')}"
            elif evidence["modality"] == "audio":
                location = f"{evidence.get('start_sec','')}-{evidence.get('end_sec','')} 秒"
            else:
                location = f"帧 {evidence.get('start_frame','')}-{evidence.get('end_frame','')}（代表帧 {evidence.get('representative_frame','')}）"
            similarity = evidence.get("prototype_similarity")
            similarity_text = f"{float(similarity):.4f}" if similarity is not None else "N/A"
            prototype_sample = evidence.get("prototype_sample_id") or "N/A"
            lines.append(
                f"|{evidence['rank']}|{evidence['modality']}|{evidence['position']}|"
                f"{float(evidence['local_importance']):.4f}|{prototype_sample}|{similarity_text}|{location}|"
            )
        lines.append("")
    path = run_dir / "reports" / f"{split}_explanation_cards.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_solution_summary(run_dir: Path, artifacts: dict[str, Any]) -> Path:
    path = run_dir / "reports" / "solution_run_summary.md"
    lines = [
        "# 问题三模型求解运行摘要", "",
        "## 数据输入", "", "读取问题三独立预处理目录中的 train、valid、test 与 attachment4_aligned；训练集只用于参数和原型学习，验证集用于早停与超参数选择，附件4只做最终推理。", "",
        "## 参数初始化", "", "文本骨干由问题二同构参数热启动，完整问题二模型作为冻结教师；选择器温度与 Gumbel 噪声共同退火。Top-K 上限、局部窗口和各损失权重只在验证集调试。", "",
        "## 模型调用", "", "上下文选择器按样本分配自适应 Top-K，再把中心前后固定半径的局部载荷交给受限预测器；极性与强度共享有序约束。原型仅在训练后用于相似案例解释，不反馈预测。", "",
        "## 结果输出", "", "输出极性、强度、三模态 Shapley 作用程度、主要模态、逐证据删除重要性、训练原型匹配及原始片段定位。低置信时间映射和模态失效均显式标记。", "",
        "## 本次产物", "",
    ]
    for key, value in artifacts.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
