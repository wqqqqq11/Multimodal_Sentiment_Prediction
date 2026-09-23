"""附件2：aligned_50.pkl / unaligned_50.pkl 与 label.xlsx。"""

from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd

from config import (
    ALIGNED_STRUCTURAL_LEADING_ZERO,
    CLASS_ID_TO_POLARITY,
    DATASET_DIRS,
    POLARITY_ZH,
    SPLIT_ZH,
)
from feature_stats import (
    batch_length_and_internal_zeros,
    numeric_overview,
    parse_sample_id,
    per_dimension_stats,
    polarity_from_regression,
    text_lengths,
)
from io_utils import dataset_output, describe_series, save_df, save_json, value_counts
from plotting import bar_chart, grouped_bar, hist_chart, overlay_hist


def _load(path):
    print(f"[dataset02] 读取 {path.name} ...")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _optional_lengths(block: dict, key: str, n: int) -> np.ndarray | None:
    if key not in block:
        return None
    values = np.asarray(block[key]).reshape(-1)
    if values.size != n:
        return None
    return values.astype(np.int32)


def _schema_rows(version: str, data: dict) -> list[dict]:
    rows = []
    for split, block in data.items():
        for field, value in block.items():
            array = np.asarray(value)
            rows.append(
                {
                    "version": version,
                    "split": split,
                    "field": field,
                    "shape": "x".join(str(x) for x in array.shape),
                    "dtype": str(array.dtype),
                    "ndim": int(array.ndim),
                }
            )
    return rows


def analyze_version(version: str, filename: str) -> dict:
    path = DATASET_DIRS["dataset02"] / filename
    data = _load(path)
    out = dataset_output("dataset02")
    structural = version == "aligned"
    sample_frames = []
    numeric_rows = []
    dim_rows = []
    split_polarity = {}
    regression_by_split = {}

    for split, block in data.items():
        ids = np.array([str(x) for x in np.asarray(block["id"]).tolist()], dtype=object)
        n = len(ids)
        raw_text = np.array([str(x) for x in np.asarray(block["raw_text"]).tolist()], dtype=object)
        regression = np.asarray(block["regression_labels"], dtype=float).reshape(-1)
        classification = np.asarray(block["classification_labels"], dtype=float).reshape(-1).astype(int)
        polarity = polarity_from_regression(regression)
        class_polarity = np.array([CLASS_ID_TO_POLARITY.get(int(v), "unknown") for v in classification], dtype=object)
        char_len, word_len = text_lengths(raw_text)
        text_bert = np.asarray(block["text_bert"])
        attention = text_bert[:, 1, :]
        text_valid = attention.sum(axis=1).astype(np.int32)
        token_ids = text_bert[:, 0, :]
        # 注意力掩码是否只在尾部填充：有效区内不应再出现 0。
        mask_holes = []
        for row in attention:
            positive = np.flatnonzero(row > 0)
            if positive.size == 0:
                mask_holes.append(True)
                continue
            mask_holes.append(bool(np.any(row[: positive[-1] + 1] == 0)))
        mask_holes = np.asarray(mask_holes)

        modality_stats = {}
        for modality in ("audio", "vision", "text"):
            array = np.asarray(block[modality])
            length_key = f"{modality}_lengths"
            provided = _optional_lengths(block, length_key, n)
            profile = batch_length_and_internal_zeros(
                array,
                provided_lengths=provided,
                structural_leading_zero=structural and modality in ALIGNED_STRUCTURAL_LEADING_ZERO,
            )
            modality_stats[modality] = profile
            numeric = numeric_overview(array)
            numeric.update({"version": version, "split": split, "modality": modality})
            numeric_rows.append(numeric)
            dims = per_dimension_stats(array)
            for index in range(array.shape[-1]):
                dim_rows.append(
                    {
                        "version": version,
                        "split": split,
                        "modality": modality,
                        "dimension": index,
                        "mean": float(dims["mean"][index]),
                        "std": float(dims["std"][index]),
                        "min": float(dims["min"][index]),
                        "max": float(dims["max"][index]),
                        "zero_ratio": float(dims["zero_ratio"][index]),
                    }
                )

        parsed = [parse_sample_id(item) for item in ids]
        frame = pd.DataFrame(
            {
                "version": version,
                "split": split,
                "id": ids,
                "video_id": [item[0] for item in parsed],
                "clip_id": [item[1] for item in parsed],
                "raw_text": raw_text,
                "text_char_len": char_len,
                "text_word_len": word_len,
                "text_bert_valid_len": text_valid,
                "attention_mask_has_internal_zero": mask_holes,
                "regression_label": regression,
                "classification_label": classification,
                "polarity_from_regression": polarity,
                "polarity_from_classification": class_polarity,
                "label_sources_agree": polarity == class_polarity,
                "audio_valid_length": modality_stats["audio"]["valid_length"],
                "audio_inferred_length": modality_stats["audio"]["inferred_valid_length"],
                "audio_nonzero_steps": modality_stats["audio"]["nonzero_steps"],
                "audio_internal_zero_steps": modality_stats["audio"]["internal_zero_steps"],
                "vision_valid_length": modality_stats["vision"]["valid_length"],
                "vision_inferred_length": modality_stats["vision"]["inferred_valid_length"],
                "vision_nonzero_steps": modality_stats["vision"]["nonzero_steps"],
                "vision_internal_zero_steps": modality_stats["vision"]["internal_zero_steps"],
                "text_inferred_length": modality_stats["text"]["inferred_valid_length"],
                "text_internal_zero_steps": modality_stats["text"]["internal_zero_steps"],
            }
        )
        # 对齐版本的经验关系：有效 BERT 长度 = 音频非零步 + [CLS] + [SEP]。
        frame["audio_matches_bert_length"] = frame["text_bert_valid_len"] == frame["audio_nonzero_steps"] + 2
        cls_token = token_ids[:, 0]
        frame["text_bert_starts_with_101"] = cls_token == 101
        sample_frames.append(frame)
        split_polarity[SPLIT_ZH.get(split, split)] = [POLARITY_ZH[item] for item in polarity]
        regression_by_split[SPLIT_ZH.get(split, split)] = regression
        print(f"[dataset02] {version}/{split}: {n} 条")

    samples = pd.concat(sample_frames, ignore_index=True)
    save_df(out["tables"] / f"samples_{version}.csv", samples.drop(columns=["raw_text"]))
    save_df(
        out["tables"] / f"texts_{version}.csv",
        samples[["version", "split", "id", "video_id", "clip_id", "raw_text", "text_char_len", "text_word_len"]],
    )
    save_df(out["tables"] / f"modality_numeric_{version}.csv", pd.DataFrame(numeric_rows))
    dim_df = pd.DataFrame(dim_rows)
    save_df(out["tables"] / f"dimension_stats_{version}.csv", dim_df)
    constant_dims = dim_df[dim_df["std"] <= 1e-8][
        ["version", "split", "modality", "dimension", "mean", "std", "zero_ratio"]
    ]
    save_df(out["tables"] / f"near_constant_dimensions_{version}.csv", constant_dims)

    polarity_table = (
        samples.groupby(["split", "polarity_from_regression"]).size().rename("count").reset_index()
    )
    save_df(out["tables"] / f"polarity_by_split_{version}.csv", polarity_table)

    length_summary_rows = []
    for column in (
        "text_word_len",
        "text_bert_valid_len",
        "audio_valid_length",
        "vision_valid_length",
        "audio_internal_zero_steps",
        "vision_internal_zero_steps",
    ):
        for split, part in samples.groupby("split"):
            stats = describe_series(part[column])
            stats.update({"version": version, "split": split, "metric": column})
            length_summary_rows.append(stats)
    save_df(out["tables"] / f"length_summary_{version}.csv", pd.DataFrame(length_summary_rows))

    grouped_bar(
        [SPLIT_ZH.get(s, s) for s in ("train", "valid", "test")],
        {
            POLARITY_ZH[p]: [
                int(((samples["split"] == s) & (samples["polarity_from_regression"] == p)).sum())
                for s in ("train", "valid", "test")
            ]
            for p in ("negative", "neutral", "positive")
        },
        out["figures"] / f"polarity_by_split_{version}.png",
        f"附件2 {version} 各划分的情感极性",
        "数据划分",
        "样本数",
    )
    overlay_hist(
        regression_by_split,
        out["figures"] / f"regression_by_split_{version}.png",
        f"附件2 {version} 情感强度分布",
        "情感强度",
        bins=np.linspace(-3, 3, 25),
    )
    overlay_hist(
        {
            "音频有效长度": samples["audio_valid_length"],
            "视觉有效长度": samples["vision_valid_length"],
            "文本 BERT 有效长度": samples["text_bert_valid_len"],
        },
        out["figures"] / f"sequence_lengths_{version}.png",
        f"附件2 {version} 序列有效长度",
        "有效时间步",
        bins=30,
    )
    hist_chart(
        samples["text_word_len"],
        out["figures"] / f"text_word_length_{version}.png",
        f"附件2 {version} 转写词数",
        "词数",
    )

    duplicate_ids = int(samples["id"].duplicated().sum())
    summary = {
        "dataset": "dataset02",
        "version": version,
        "file": filename,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "top_level_keys": list(data.keys()),
        "fields_by_split": {split: sorted(block.keys()) for split, block in data.items()},
        "split_counts": samples.groupby("split").size().astype(int).to_dict(),
        "sample_count": int(len(samples)),
        "duplicate_ids_within_version": duplicate_ids,
        "unique_video_ids": int(samples["video_id"].nunique()),
        "regression": describe_series(samples["regression_label"]),
        "polarity_counts": value_counts(samples["polarity_from_regression"]),
        "classification_counts": value_counts(samples["classification_label"]),
        "regression_classification_agreement": float(samples["label_sources_agree"].mean()),
        "attention_mask_internal_zero_samples": int(samples["attention_mask_has_internal_zero"].sum()),
        "text_bert_cls_101_ratio": float(samples["text_bert_starts_with_101"].mean()),
        "audio_bert_length_agreement": float(samples["audio_matches_bert_length"].mean()),
        "samples_with_audio_internal_zeros": int((samples["audio_internal_zero_steps"] > 0).sum()),
        "samples_with_vision_internal_zeros": int((samples["vision_internal_zero_steps"] > 0).sum()),
        "samples_with_text_feature_internal_zeros": int((samples["text_internal_zero_steps"] > 0).sum()),
        "vision_all_zero_samples": int((samples["vision_nonzero_steps"] == 0).sum()),
        "audio_all_zero_samples": int((samples["audio_nonzero_steps"] == 0).sum()),
        "text_word_len": describe_series(samples["text_word_len"]),
        "text_bert_valid_len": describe_series(samples["text_bert_valid_len"]),
        "audio_valid_length": describe_series(samples["audio_valid_length"]),
        "vision_valid_length": describe_series(samples["vision_valid_length"]),
        "near_constant_dimension_rows": int(len(constant_dims)),
        "schema": _schema_rows(version, data),
    }
    save_json(out["root"] / f"summary_{version}.json", summary)
    save_df(out["tables"] / f"schema_{version}.csv", pd.DataFrame(summary["schema"]))
    del data
    gc.collect()
    return summary


def analyze_label_table(aligned_samples: pd.DataFrame | None = None) -> dict:
    """label.xlsx 与对齐特征样本逐条对照。样本表已落盘时直接读取。"""
    out = dataset_output("dataset02")
    excel = pd.read_excel(DATASET_DIRS["dataset02"] / "label.xlsx")
    excel["video_id"] = excel["video_id"].astype(str)
    excel["clip_id"] = excel["clip_id"].astype(str)
    excel["id"] = excel["video_id"] + "$_$" + excel["clip_id"]
    excel["polarity_from_regression"] = polarity_from_regression(excel["label"].to_numpy())
    excel["annotation_norm"] = excel["annotation"].astype(str).str.strip().str.lower()
    if aligned_samples is None:
        aligned_samples = pd.read_csv(out["tables"] / "samples_aligned.csv")
    merged = excel.merge(
        aligned_samples[
            [
                "id",
                "split",
                "regression_label",
                "classification_label",
                "polarity_from_regression",
            ]
        ].rename(columns={"polarity_from_regression": "pkl_polarity", "split": "pkl_split"}),
        on="id",
        how="outer",
        indicator=True,
    )
    both = merged[merged["_merge"] == "both"].copy()
    both["score_abs_diff"] = (both["label"] - both["regression_label"]).abs()
    both["text_available"] = True
    save_df(out["tables"] / "label_excel.csv", excel)
    save_df(
        out["tables"] / "label_excel_vs_aligned_pkl.csv",
        both[
            [
                "id",
                "video_id",
                "clip_id",
                "mode",
                "pkl_split",
                "label",
                "regression_label",
                "score_abs_diff",
                "annotation",
                "pkl_polarity",
                "classification_label",
            ]
        ],
    )
    mode_vs_split = pd.crosstab(both["mode"], both["pkl_split"])
    mode_vs_split.to_csv(out["tables"] / "excel_mode_vs_pkl_split.csv", encoding="utf-8-sig")
    summary = {
        "excel_rows": int(len(excel)),
        "excel_only_ids": int((merged["_merge"] == "left_only").sum()),
        "pkl_only_ids": int((merged["_merge"] == "right_only").sum()),
        "matched_ids": int(len(both)),
        "max_abs_score_diff": float(both["score_abs_diff"].max()) if len(both) else None,
        "scores_not_exactly_equal": int((both["score_abs_diff"] > 1e-4).sum()) if len(both) else None,
        "annotation_matches_pkl_polarity": float((both["annotation_norm"] == both["pkl_polarity"]).mean()) if len(both) else None,
        "mode_equals_pkl_split": float((both["mode"].astype(str) == both["pkl_split"].astype(str)).mean()) if len(both) else None,
        "excel_polarity_counts": value_counts(excel["polarity_from_regression"]),
        "excel_mode_counts": value_counts(excel["mode"]),
    }
    bar_chart(
        [POLARITY_ZH.get(x, x) for x in summary["excel_polarity_counts"]],
        list(summary["excel_polarity_counts"].values()),
        out["figures"] / "excel_polarity_counts.png",
        "附件2 label.xlsx 情感极性",
        "极性",
        "样本数",
    )
    return summary


def _compare_aligned_and_unaligned(out) -> dict:
    columns = ["id", "split", "regression_label", "classification_label", "text_word_len", "text_bert_valid_len"]
    aligned = pd.read_csv(out["tables"] / "samples_aligned.csv", usecols=columns)
    unaligned = pd.read_csv(out["tables"] / "samples_unaligned.csv", usecols=columns)
    merged = aligned.merge(unaligned, on="id", suffixes=("_aligned", "_unaligned"))
    if merged.empty:
        return {"shared_rows": 0}
    score_diff = (merged["regression_label_aligned"] - merged["regression_label_unaligned"]).abs()
    return {
        "shared_rows": int(len(merged)),
        "same_split": float((merged["split_aligned"] == merged["split_unaligned"]).mean()),
        "same_regression": float((score_diff <= 1e-4).mean()),
        "max_abs_regression_diff": float(score_diff.max()),
        "same_classification": float(
            (merged["classification_label_aligned"] == merged["classification_label_unaligned"]).mean()
        ),
        "same_text_word_len": float((merged["text_word_len_aligned"] == merged["text_word_len_unaligned"]).mean()),
        "same_text_bert_valid_len": float(
            (merged["text_bert_valid_len_aligned"] == merged["text_bert_valid_len_unaligned"]).mean()
        ),
    }


def analyze() -> dict:
    aligned = analyze_version("aligned", "aligned_50.pkl")
    label_summary = analyze_label_table()
    unaligned = analyze_version("unaligned", "unaligned_50.pkl")
    out = dataset_output("dataset02")
    version_agreement = _compare_aligned_and_unaligned(out)
    aligned_ids = set(pd.read_csv(out["tables"] / "samples_aligned.csv", usecols=["id"])["id"].astype(str))
    unaligned_ids = set(pd.read_csv(out["tables"] / "samples_unaligned.csv", usecols=["id"])["id"].astype(str))
    comparison = {
        "aligned_count": len(aligned_ids),
        "unaligned_count": len(unaligned_ids),
        "only_in_aligned": sorted(aligned_ids - unaligned_ids),
        "only_in_unaligned": sorted(unaligned_ids - aligned_ids),
        "shared_ids": len(aligned_ids & unaligned_ids),
    }
    summary = {
        "dataset": "dataset02",
        "aligned": aligned,
        "unaligned": unaligned,
        "label_excel": label_summary,
        "id_set_comparison": {
            "aligned_count": comparison["aligned_count"],
            "unaligned_count": comparison["unaligned_count"],
            "shared_ids": comparison["shared_ids"],
            "only_in_aligned_count": len(comparison["only_in_aligned"]),
            "only_in_unaligned_count": len(comparison["only_in_unaligned"]),
        },
        "aligned_unaligned_label_agreement": version_agreement,
    }
    save_json(out["root"] / "summary.json", summary)
    save_df(
        out["tables"] / "id_set_difference.csv",
        pd.DataFrame(
            {
                "only_in_aligned": pd.Series(comparison["only_in_aligned"]),
                "only_in_unaligned": pd.Series(comparison["only_in_unaligned"]),
            }
        ),
    )
    print(
        f"[dataset02] 对齐 {comparison['aligned_count']}，未对齐 {comparison['unaligned_count']}，"
        f"共有 ID {comparison['shared_ids']}"
    )
    return summary


if __name__ == "__main__":
    analyze()
