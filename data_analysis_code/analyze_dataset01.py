"""附件1：100 条原始视频与标注表。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATASET_DIRS, POLARITY_ZH
from feature_stats import polarity_from_regression, text_lengths
from io_utils import dataset_output, describe_series, save_df, save_json, value_counts
from plotting import bar_chart, hist_chart, scatter_chart


def _video_meta(path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    duration = frames / fps if fps > 0 else np.nan
    brightness = []
    if frames > 0:
        for ratio in (0.1, 0.5, 0.9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frames - 1, max(0, int(frames * ratio))))
            ok, frame = cap.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness.append(float(gray.mean()))
    cap.release()
    return {
        "readable": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_sec": duration,
        "codec": codec,
        "sampled_brightness_mean": float(np.mean(brightness)) if brightness else np.nan,
    }


def analyze() -> dict:
    root = DATASET_DIRS["dataset01"]
    out = dataset_output("dataset01")
    label_path = root / "label-100.xlsx"
    sheets = pd.read_excel(label_path, sheet_name=None)
    label_df = sheets["label"].copy()
    label_df["video_id"] = label_df["video_id"].astype(str)
    label_df["clip_id"] = label_df["clip_id"].astype(str)
    label_df["sample_key"] = label_df["video_id"] + "/" + label_df["clip_id"]
    char_len, word_len = text_lengths(label_df["text"].astype(str).to_numpy())
    label_df["text_char_len"] = char_len
    label_df["text_word_len"] = word_len
    label_df["polarity"] = polarity_from_regression(label_df["label"].to_numpy())
    label_df["annotation_norm"] = label_df["annotation"].astype(str).str.strip().str.lower()
    label_df["annotation_matches_score"] = label_df["annotation_norm"] == label_df["polarity"]

    files = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for video in sorted(folder.glob("*.mp4")):
            meta = _video_meta(video)
            files.append(
                {
                    "video_id": folder.name,
                    "clip_id": video.stem,
                    "sample_key": f"{folder.name}/{video.stem}",
                    "relative_path": str(video.relative_to(root)).replace("\\", "/"),
                    "file_size_mb": video.stat().st_size / (1024 * 1024),
                    **meta,
                }
            )
    file_df = pd.DataFrame(files)
    merged = label_df.merge(file_df, on=["video_id", "clip_id", "sample_key"], how="outer", indicator=True)
    missing_video = merged.loc[merged["_merge"] == "left_only", "sample_key"].tolist()
    extra_video = merged.loc[merged["_merge"] == "right_only", "sample_key"].tolist()

    clips_per_video = (
        label_df.groupby("video_id").size().rename("clip_count").reset_index().sort_values("clip_count", ascending=False)
    )
    polarity_counts = label_df["polarity"].value_counts().rename_axis("polarity").reset_index(name="count")
    annotation_counts = label_df["annotation"].value_counts().rename_axis("annotation").reset_index(name="count")

    resolution = file_df[file_df["readable"] == True].copy() if len(file_df) else file_df
    if len(resolution):
        resolution["resolution"] = resolution["width"].astype(str) + "x" + resolution["height"].astype(str)
        resolution_counts = resolution["resolution"].value_counts().rename_axis("resolution").reset_index(name="count")
    else:
        resolution_counts = pd.DataFrame(columns=["resolution", "count"])

    notes = []
    for name, frame in sheets.items():
        if name == "label":
            continue
        exported = frame.copy()
        exported.insert(0, "sheet", name)
        notes.append(exported)
    if notes:
        save_df(out["tables"] / "label_workbook_notes.csv", pd.concat(notes, ignore_index=True))

    save_df(out["tables"] / "samples.csv", merged.drop(columns=["_merge"]))
    save_df(out["tables"] / "polarity_counts.csv", polarity_counts)
    save_df(out["tables"] / "annotation_counts.csv", annotation_counts)
    save_df(out["tables"] / "clips_per_video.csv", clips_per_video)
    save_df(out["tables"] / "resolution_counts.csv", resolution_counts)

    score = label_df["label"].to_numpy(dtype=float)
    bins = np.round(np.arange(-3, 3.0001, 0.5), 2)
    hist, edges = np.histogram(score, bins=bins)
    save_df(
        out["tables"] / "regression_histogram.csv",
        pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}),
    )

    if len(file_df) and file_df["readable"].any():
        hist_chart(
            file_df.loc[file_df["readable"], "duration_sec"],
            out["figures"] / "duration_hist.png",
            "附件1 视频时长分布",
            "时长（秒）",
        )
        hist_chart(
            file_df["file_size_mb"],
            out["figures"] / "file_size_hist.png",
            "附件1 视频文件大小分布",
            "文件大小（MB）",
        )
        if len(resolution_counts):
            bar_chart(
                resolution_counts["resolution"].tolist(),
                resolution_counts["count"].tolist(),
                out["figures"] / "resolution_counts.png",
                "附件1 视频分辨率计数",
                "分辨率（宽x高）",
                "视频数",
                rotation=30,
            )
        paired = merged.dropna(subset=["label", "duration_sec"])
        if len(paired):
            scatter_chart(
                paired["duration_sec"],
                paired["label"],
                out["figures"] / "duration_vs_label.png",
                "附件1 视频时长与情感强度",
                "时长（秒）",
                "情感强度",
            )
    hist_chart(score, out["figures"] / "regression_hist.png", "附件1 情感强度分布", "情感强度", bins=24)
    hist_chart(label_df["text_word_len"], out["figures"] / "text_word_length_hist.png", "附件1 转写词数分布", "词数")
    bar_chart(
        [POLARITY_ZH.get(x, x) for x in polarity_counts["polarity"]],
        polarity_counts["count"].tolist(),
        out["figures"] / "polarity_counts.png",
        "附件1 情感极性计数",
        "极性",
        "样本数",
    )
    bar_chart(
        clips_per_video["clip_count"].value_counts().sort_index().index.astype(str).tolist(),
        clips_per_video["clip_count"].value_counts().sort_index().tolist(),
        out["figures"] / "clips_per_video_hist.png",
        "附件1 同一 video_id 下的片段数",
        "每个视频包含的片段数",
        "video_id 数",
    )

    readable = file_df[file_df.get("readable", False) == True] if len(file_df) else file_df
    summary = {
        "dataset": "dataset01",
        "description": "附件1，CMU-MOSEI 原始英文视频片段及连续情感标注。",
        "label_rows": int(len(label_df)),
        "video_files": int(len(file_df)),
        "video_id_count": int(label_df["video_id"].nunique()),
        "duplicate_sample_keys": int(label_df["sample_key"].duplicated().sum()),
        "labels_without_video": missing_video,
        "videos_without_label": extra_video,
        "unreadable_videos": file_df.loc[~file_df["readable"], "sample_key"].tolist() if len(file_df) else [],
        "polarity_counts": value_counts(label_df["polarity"]),
        "annotation_counts": value_counts(label_df["annotation"]),
        "annotation_score_agreement": float(label_df["annotation_matches_score"].mean()),
        "regression": describe_series(score),
        "text_char_len": describe_series(label_df["text_char_len"]),
        "text_word_len": describe_series(label_df["text_word_len"]),
        "duration_sec": describe_series(readable["duration_sec"]) if len(readable) else {},
        "fps": describe_series(readable["fps"]) if len(readable) else {},
        "file_size_mb": describe_series(file_df["file_size_mb"]) if len(file_df) else {},
        "clips_per_video": describe_series(clips_per_video["clip_count"]),
        "contest_duration_range_sec": [2.648, 34.567],
        "duration_outside_contest_range": int(
            ((readable["duration_sec"] < 2.648 - 0.05) | (readable["duration_sec"] > 34.567 + 0.05)).sum()
        )
        if len(readable)
        else None,
    }
    save_json(out["root"] / "summary.json", summary)
    print(f"[dataset01] 标注 {summary['label_rows']} 条，视频 {summary['video_files']} 个，极性 {summary['polarity_counts']}")
    return summary


if __name__ == "__main__":
    analyze()
