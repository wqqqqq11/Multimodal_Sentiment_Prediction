"""附件4：20 条完整三模态可解释性测试样本及原始视频。"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd

from analyze_dataset01 import _video_meta
from config import DATASET_DIRS
from feature_stats import sequence_zero_profile, text_lengths
from io_utils import dataset_output, describe_series, save_df, save_json
from plotting import hist_chart, scatter_chart


def _md5(path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {key: np.asarray(value) if not isinstance(value, (str, int, np.str_, np.integer)) else value for key, value in payload.items()}


def _as_text(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.reshape(-1)[0])
    return str(value)


def analyze() -> dict:
    root = DATASET_DIRS["dataset04"]
    out = dataset_output("dataset04")
    text_index = {}
    text_path = dataset_output("dataset02")["tables"] / "texts_aligned.csv"
    if text_path.exists():
        reference = pd.read_csv(text_path, usecols=["id", "split", "raw_text"])
        for row in reference.itertuples(index=False):
            text_index.setdefault(str(row.raw_text), []).append((str(row.split), str(row.id)))

    rows = []
    video_rows = []
    hash_rows = []
    aligned_dir = root / "aligned"
    unaligned_dir = root / "unaligned"
    feature_names = sorted(p.stem for p in aligned_dir.glob("*.pkl"))
    for stem in feature_names:
        aligned = _load(aligned_dir / f"{stem}.pkl")
        unaligned = _load(unaligned_dir / f"{stem}.pkl")
        raw_text = _as_text(aligned["raw_text"])
        char_len, word_len = text_lengths(np.array([raw_text]))
        text_bert = np.asarray(aligned["text_bert"])
        mask = text_bert[1]
        audio = np.asarray(aligned["audio"])
        vision = np.asarray(aligned["vision"])
        audio_profile = sequence_zero_profile(audio, structural_leading_zero=True)
        vision_profile = sequence_zero_profile(vision, structural_leading_zero=True)
        text_profile = sequence_zero_profile(np.asarray(aligned["text"]))
        u_audio = np.asarray(unaligned["audio"])
        u_vision = np.asarray(unaligned["vision"])
        u_audio_len = int(unaligned["audio_lengths"]) if "audio_lengths" in unaligned else None
        u_vision_len = int(unaligned["vision_lengths"]) if "vision_lengths" in unaligned else None
        u_audio_profile = sequence_zero_profile(u_audio, provided_length=u_audio_len)
        u_vision_profile = sequence_zero_profile(u_vision, provided_length=u_vision_len)
        matches = text_index.get(raw_text, [])
        rows.append(
            {
                "id": _as_text(aligned.get("id", stem)),
                "file_stem": stem,
                "raw_text": raw_text,
                "text_char_len": int(char_len[0]),
                "text_word_len": int(word_len[0]),
                "raw_text_equal_across_versions": raw_text == _as_text(unaligned["raw_text"]),
                "text_features_equal": bool(np.allclose(aligned["text"], unaligned["text"])),
                "text_bert_equal": bool(np.array_equal(aligned["text_bert"], unaligned["text_bert"])),
                "id_equal": _as_text(aligned.get("id", "")) == _as_text(unaligned.get("id", "")),
                "text_bert_valid_len": int(mask.sum()),
                "text_bert_starts_with_101": bool(np.rint(text_bert[0, 0]) == 101),
                "aligned_audio_shape": "x".join(str(x) for x in audio.shape),
                "aligned_vision_shape": "x".join(str(x) for x in vision.shape),
                "aligned_text_shape": "x".join(str(x) for x in np.asarray(aligned["text"]).shape),
                "aligned_audio_nonzero_steps": audio_profile["nonzero_steps"],
                "aligned_audio_internal_zero_steps": audio_profile["missing_steps"],
                "aligned_vision_nonzero_steps": vision_profile["nonzero_steps"],
                "aligned_vision_internal_zero_steps": vision_profile["missing_steps"],
                "aligned_text_internal_zero_steps": text_profile["missing_steps"],
                "aligned_audio_matches_bert": int(mask.sum()) == audio_profile["nonzero_steps"] + 2,
                "unaligned_audio_shape": "x".join(str(x) for x in u_audio.shape),
                "unaligned_vision_shape": "x".join(str(x) for x in u_vision.shape),
                "unaligned_audio_lengths": u_audio_len,
                "unaligned_vision_lengths": u_vision_len,
                "unaligned_audio_inferred_length": u_audio_profile["inferred_valid_length"],
                "unaligned_vision_inferred_length": u_vision_profile["inferred_valid_length"],
                "unaligned_audio_internal_zero_steps": u_audio_profile["missing_steps"],
                "unaligned_vision_internal_zero_steps": u_vision_profile["missing_steps"],
                "audio_length_matches_inferred": u_audio_len == u_audio_profile["inferred_valid_length"],
                "vision_length_matches_inferred": u_vision_len == u_vision_profile["inferred_valid_length"],
                "found_in_dataset02": bool(matches),
                "dataset02_match_count": len(matches),
                "dataset02_matches": ";".join(f"{split}:{sample_id}" for split, sample_id in matches[:5]),
                "aligned_fields": ",".join(sorted(aligned.keys())),
                "unaligned_fields": ",".join(sorted(str(k) for k in unaligned.keys())),
            }
        )

    for version, folder in (("aligned", aligned_dir), ("unaligned", unaligned_dir)):
        video_dir = folder / "videos"
        for video in sorted(video_dir.glob("*.mp4")):
            meta = _video_meta(video)
            video_rows.append(
                {
                    "version": version,
                    "file_stem": video.stem,
                    "file_size_mb": video.stat().st_size / (1024 * 1024),
                    "md5": _md5(video),
                    **meta,
                }
            )
    videos = pd.DataFrame(video_rows)
    if len(videos):
        wide = videos.pivot(index="file_stem", columns="version", values="md5")
        for stem, row in wide.iterrows():
            hash_rows.append(
                {
                    "file_stem": stem,
                    "aligned_md5": row.get("aligned"),
                    "unaligned_md5": row.get("unaligned"),
                    "videos_identical": row.get("aligned") == row.get("unaligned"),
                }
            )
    hashes = pd.DataFrame(hash_rows)
    samples = pd.DataFrame(rows)
    save_df(out["tables"] / "samples.csv", samples)
    save_df(out["tables"] / "videos.csv", videos)
    save_df(out["tables"] / "video_hash_comparison.csv", hashes)

    if len(samples):
        hist_chart(samples["text_word_len"], out["figures"] / "text_word_length.png", "附件4 转写词数", "词数")
        hist_chart(
            samples["unaligned_audio_lengths"].dropna(),
            out["figures"] / "unaligned_audio_length.png",
            "附件4 未对齐语音有效长度",
            "audio_lengths",
        )
        hist_chart(
            samples["unaligned_vision_lengths"].dropna(),
            out["figures"] / "unaligned_vision_length.png",
            "附件4 未对齐视觉有效长度",
            "vision_lengths",
        )
    readable = videos[(videos["version"] == "aligned") & (videos["readable"] == True)] if len(videos) else videos
    if len(readable):
        hist_chart(readable["duration_sec"], out["figures"] / "video_duration.png", "附件4 原始视频时长", "时长（秒）")
        joined = samples.merge(readable, left_on="file_stem", right_on="file_stem")
        if len(joined):
            scatter_chart(
                joined["duration_sec"],
                joined["unaligned_audio_lengths"],
                out["figures"] / "duration_vs_audio_length.png",
                "附件4 视频时长与未对齐语音长度",
                "视频时长（秒）",
                "audio_lengths",
            )

    summary = {
        "dataset": "dataset04",
        "description": "附件4，无标签、三模态完整的可解释性测试集，含对齐/未对齐特征和原始视频。",
        "sample_count": int(len(samples)),
        "raw_text_consistent": int(samples["raw_text_equal_across_versions"].sum()) if len(samples) else 0,
        "text_features_consistent": int(samples["text_features_equal"].sum()) if len(samples) else 0,
        "text_bert_consistent": int(samples["text_bert_equal"].sum()) if len(samples) else 0,
        "aligned_audio_internal_zero_samples": int((samples["aligned_audio_internal_zero_steps"] > 0).sum()) if len(samples) else 0,
        "aligned_vision_internal_zero_samples": int((samples["aligned_vision_internal_zero_steps"] > 0).sum()) if len(samples) else 0,
        "aligned_text_internal_zero_samples": int((samples["aligned_text_internal_zero_steps"] > 0).sum()) if len(samples) else 0,
        "unaligned_audio_internal_zero_samples": int((samples["unaligned_audio_internal_zero_steps"] > 0).sum()) if len(samples) else 0,
        "unaligned_vision_internal_zero_samples": int((samples["unaligned_vision_internal_zero_steps"] > 0).sum()) if len(samples) else 0,
        "audio_length_equals_inferred": int(samples["audio_length_matches_inferred"].sum()) if len(samples) else 0,
        "vision_length_equals_inferred": int(samples["vision_length_matches_inferred"].sum()) if len(samples) else 0,
        "bert_audio_length_agreement": int(samples["aligned_audio_matches_bert"].sum()) if len(samples) else 0,
        "found_in_dataset02": int(samples["found_in_dataset02"].sum()) if len(samples) else 0,
        "text_word_len": describe_series(samples["text_word_len"]) if len(samples) else {},
        "text_bert_valid_len": describe_series(samples["text_bert_valid_len"]) if len(samples) else {},
        "unaligned_audio_lengths": describe_series(samples["unaligned_audio_lengths"]) if len(samples) else {},
        "unaligned_vision_lengths": describe_series(samples["unaligned_vision_lengths"]) if len(samples) else {},
        "videos_per_version": videos.groupby("version").size().astype(int).to_dict() if len(videos) else {},
        "identical_video_pairs": int(hashes["videos_identical"].sum()) if len(hashes) else 0,
        "video_pairs": int(len(hashes)),
        "duration_sec": describe_series(readable["duration_sec"]) if len(readable) else {},
        "resolution_counts": readable.assign(resolution=readable["width"].astype(str) + "x" + readable["height"].astype(str))["resolution"]
        .value_counts()
        .astype(int)
        .to_dict()
        if len(readable)
        else {},
        "unreadable_videos": videos.loc[~videos["readable"], "file_stem"].astype(str).tolist() if len(videos) else [],
        "aligned_fields": samples["aligned_fields"].iloc[0].split(",") if len(samples) else [],
        "unaligned_fields": samples["unaligned_fields"].iloc[0].split(",") if len(samples) else [],
    }
    save_json(out["root"] / "summary.json", summary)
    print(
        f"[dataset04] 样本 {summary['sample_count']}，视频对完全相同 {summary['identical_video_pairs']}/"
        f"{summary['video_pairs']}，出现在附件2中 {summary['found_in_dataset02']}"
    )
    return summary


if __name__ == "__main__":
    analyze()
