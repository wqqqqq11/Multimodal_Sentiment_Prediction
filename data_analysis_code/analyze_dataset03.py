"""附件3：30 条局部模态缺失测试样本，对齐与未对齐各一套。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALIGNED_STRUCTURAL_LEADING_ZERO, DATASET_DIRS, MATCH_ATOL, SPLIT_ZH
from feature_stats import mask_to_runs, sequence_zero_profile, timestep_all_zero
from io_utils import dataset_output, describe_series, save_df, save_json
from plotting import bar_chart, heatmap, hist_chart


def _squeeze_feature(value, key: str):
    array = np.asarray(value)
    if key == "raw_text":
        return str(array.reshape(-1)[0])
    if array.ndim >= 1 and array.shape[0] == 1 and key in {"audio", "vision", "text", "text_bert"}:
        return array[0]
    return array


def _load_sample(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and list(payload.keys()) == ["test"] and isinstance(payload["test"], dict):
        block = payload["test"]
        top = "test"
    else:
        block = payload
        top = "root"
    sample = {key: _squeeze_feature(value, key) for key, value in block.items()}
    sample["__top__"] = top
    sample["__fields__"] = sorted(block.keys())
    return sample


def _index_dataset02(data: dict) -> dict:
    text_index = {}
    bert_index = {}
    for split, block in data.items():
        ids = [str(x) for x in np.asarray(block["id"]).tolist()]
        texts = [str(x) for x in np.asarray(block["raw_text"]).tolist()]
        bert = np.asarray(block["text_bert"])
        for index, sample_id in enumerate(ids):
            text_index.setdefault(texts[index], []).append((split, index, sample_id))
            if bert.ndim == 3:
                bert_index.setdefault(tuple(bert[index, 0].tolist()), []).append((split, index, sample_id))
    return {"text": text_index, "bert": bert_index, "data": data}


def _best_feature_match(query: np.ndarray, data: dict, modality: str) -> dict | None:
    zero_rows = timestep_all_zero(query)
    kept = np.flatnonzero(~zero_rows)
    if kept.size < 3:
        return None
    best = None
    second = None
    for split, block in data.items():
        if modality not in block:
            continue
        source = np.asarray(block[modality])
        if source.ndim != 3 or source.shape[1:] != query.shape:
            continue
        ids = [str(x) for x in np.asarray(block["id"]).tolist()]
        batch = 256
        for start in range(0, source.shape[0], batch):
            chunk = source[start : start + batch]
            max_abs = np.abs(chunk[:, kept, :] - query[kept]).max(axis=(1, 2))
            order = np.argsort(max_abs)[:2]
            for row in order:
                item = {
                    "split": split,
                    "index": int(start + row),
                    "id": ids[start + row],
                    "max_abs": float(max_abs[row]),
                }
                if best is None or item["max_abs"] < best["max_abs"]:
                    second = best
                    best = item
                elif second is None or item["max_abs"] < second["max_abs"]:
                    second = item
    if best is None:
        return None
    best["second_max_abs"] = None if second is None else second["max_abs"]
    best["method"] = f"{modality}_nonzero_positions"
    return best


def _match_sample(sample: dict, index: dict) -> dict:
    data = index["data"]
    raw_text = sample.get("raw_text")
    if isinstance(raw_text, str) and raw_text in index["text"]:
        split, row, sample_id = index["text"][raw_text][0]
        return {
            "matched": True,
            "method": "raw_text",
            "split": split,
            "index": row,
            "id": sample_id,
            "candidate_count": len(index["text"][raw_text]),
            "max_abs": 0.0,
        }
    text_bert = sample.get("text_bert")
    if isinstance(text_bert, np.ndarray) and text_bert.ndim == 2:
        key = tuple(np.rint(text_bert[0]).astype(np.int64).tolist())
        if key in index["bert"]:
            split, row, sample_id = index["bert"][key][0]
            return {
                "matched": True,
                "method": "text_bert_input_ids",
                "split": split,
                "index": row,
                "id": sample_id,
                "candidate_count": len(index["bert"][key]),
                "max_abs": 0.0,
            }
    best_effort = None
    for modality in ("audio", "vision"):
        if modality not in sample or not isinstance(sample[modality], np.ndarray):
            continue
        found = _best_feature_match(sample[modality], data, modality)
        if found is None:
            continue
        if best_effort is None or found["max_abs"] < best_effort["max_abs"]:
            best_effort = found
        if found["max_abs"] <= MATCH_ATOL:
            found["matched"] = True
            found["candidate_count"] = 1
            return found
    unmatched = {"matched": False, "method": "unmatched"}
    if best_effort is not None:
        unmatched.update(
            {
                "nearest_id": best_effort["id"],
                "nearest_split": best_effort["split"],
                "max_abs": best_effort["max_abs"],
                "nearest_method": best_effort["method"],
            }
        )
    return unmatched


def _original_length(original: np.ndarray, provided) -> int:
    if provided is not None:
        return int(provided)
    profile = sequence_zero_profile(original)
    return int(profile["inferred_valid_length"])


def _injected_mask(original: np.ndarray, current: np.ndarray, valid_length: int, structural: bool) -> np.ndarray:
    steps = min(original.shape[0], current.shape[0])
    valid_length = min(valid_length, steps)
    region = np.zeros(steps, dtype=bool)
    region[:valid_length] = True
    if structural and steps > 0:
        region[0] = False
    original_zero = timestep_all_zero(original[:steps])
    current_zero = timestep_all_zero(current[:steps])
    return region & (~original_zero) & current_zero


def _length_field(block: dict, key: str, index: int):
    if key not in block:
        return None
    values = np.asarray(block[key]).reshape(-1)
    if index >= values.size:
        return None
    return int(values[index])


def analyze_version(version: str, folder: Path, index: dict) -> dict:
    structural_modalities = set(ALIGNED_STRUCTURAL_LEADING_ZERO) if version == "aligned" else set()
    rows = []
    spans = []
    numeric_modalities = ("audio", "vision", "text", "text_bert")
    for path in sorted(folder.glob("*.pkl")):
        sample = _load_sample(path)
        match = _match_sample(sample, index)
        original_block = None
        if match.get("matched"):
            original_block = index["data"][match["split"]]
        record = {
            "version": version,
            "file_name": path.name,
            "sample_no": path.stem.split("_")[-1],
            "top_key": sample["__top__"],
            "fields": ",".join(sample["__fields__"]),
            "matched": bool(match.get("matched")),
            "match_method": match.get("method"),
            "matched_id": match.get("id"),
            "matched_split": match.get("split"),
            "match_max_abs": match.get("max_abs"),
            "match_candidate_count": match.get("candidate_count"),
            "nearest_id": match.get("nearest_id"),
            "nearest_split": match.get("nearest_split"),
        }
        if "raw_text" in sample:
            record["raw_text"] = sample["raw_text"]
            record["text_char_len"] = len(sample["raw_text"])
            record["text_word_len"] = len(sample["raw_text"].split())
            if original_block is not None:
                source_text = str(original_block["raw_text"][match["index"]])
                record["raw_text_equals_dataset02"] = sample["raw_text"] == source_text
        affected = []
        for modality in numeric_modalities:
            if modality not in sample or not isinstance(sample[modality], np.ndarray):
                record[f"{modality}_present"] = False
                continue
            current = sample[modality]
            record[f"{modality}_present"] = True
            record[f"{modality}_shape"] = "x".join(str(x) for x in current.shape)
            record[f"{modality}_dtype"] = str(current.dtype)
            if modality == "text_bert":
                _record_text_bert(record, spans, version, path.name, current, original_block, match)
                if record.get("text_bert_missing_steps", 0) > 0:
                    affected.append("text_bert")
                continue
            structural = modality in structural_modalities
            provided = None
            original = None
            if original_block is not None and modality in original_block:
                original = np.asarray(original_block[modality][match["index"]])
                provided = _length_field(original_block, f"{modality}_lengths", match["index"])
            if original is not None and original.shape == current.shape:
                valid_length = _original_length(original, provided)
                missing = _injected_mask(original, current, valid_length, structural)
                detection = "compared_with_dataset02"
                profile = sequence_zero_profile(current, provided_length=valid_length, structural_leading_zero=structural)
            else:
                profile = sequence_zero_profile(current, structural_leading_zero=structural)
                missing = profile["missing_mask"]
                valid_length = profile["valid_length"]
                detection = "heuristic_internal_zeros"
            runs = mask_to_runs(missing)
            record[f"{modality}_valid_length"] = int(valid_length)
            record[f"{modality}_missing_steps"] = int(missing.sum())
            record[f"{modality}_missing_ratio"] = float(missing.sum() / valid_length) if valid_length else None
            record[f"{modality}_run_count"] = len(runs)
            record[f"{modality}_longest_run"] = max((item["length"] for item in runs), default=0)
            record[f"{modality}_detection"] = detection
            record[f"{modality}_element_zero_ratio"] = profile["element_zero_ratio"]
            if int(missing.sum()) > 0:
                affected.append(modality)
            for run in runs:
                spans.append(
                    {
                        "version": version,
                        "file_name": path.name,
                        "matched_id": match.get("id"),
                        "modality": modality,
                        "detection": detection,
                        "start": run["start"],
                        "end": run["end"],
                        "length": run["length"],
                        "valid_length": int(valid_length),
                        "relative_start": float(run["start"] / valid_length) if valid_length else None,
                        "relative_length": float(run["length"] / valid_length) if valid_length else None,
                    }
                )
        record["affected_modalities"] = ",".join(affected) if affected else "none"
        record["affected_count"] = len(affected)
        rows.append(record)
        print(f"[dataset03] {path.name} 匹配={record['matched']} 缺失模态={record['affected_modalities']}")

    frame = pd.DataFrame(rows)
    span_df = pd.DataFrame(spans)
    return {"samples": frame, "spans": span_df}


def _record_text_bert(record, spans, version, file_name, current, original_block, match) -> None:
    ids = np.rint(current[0]).astype(np.int64)
    mask = np.rint(current[1]).astype(np.int64)
    record["text_bert_valid_len"] = int(mask.sum())
    record["text_bert_starts_with_101"] = bool(ids[0] == 101) if ids.size else False
    positive = np.flatnonzero(mask > 0)
    holes = False
    if positive.size:
        holes = bool(np.any(mask[: positive[-1] + 1] == 0))
    record["text_bert_mask_has_internal_zero"] = holes
    if original_block is None or "text_bert" not in original_block:
        record["text_bert_missing_steps"] = int(holes)
        record["text_bert_detection"] = "heuristic_mask_holes"
        return
    source = np.asarray(original_block["text_bert"][match["index"]])
    source_ids = source[0].astype(np.int64)
    source_mask = source[1].astype(np.int64)
    steps = min(ids.size, source_ids.size)
    missing = (source_mask[:steps] > 0) & ((ids[:steps] == 0) | (mask[:steps] == 0)) & (source_ids[:steps] != 0)
    runs = mask_to_runs(missing)
    valid = int(source_mask[:steps].sum())
    record["text_bert_valid_length"] = valid
    record["text_bert_missing_steps"] = int(missing.sum())
    record["text_bert_missing_ratio"] = float(missing.sum() / valid) if valid else None
    record["text_bert_run_count"] = len(runs)
    record["text_bert_longest_run"] = max((item["length"] for item in runs), default=0)
    record["text_bert_detection"] = "compared_with_dataset02"
    for run in runs:
        spans.append(
            {
                "version": version,
                "file_name": file_name,
                "matched_id": match.get("id"),
                "modality": "text_bert",
                "detection": "compared_with_dataset02",
                "start": run["start"],
                "end": run["end"],
                "length": run["length"],
                "valid_length": valid,
                "relative_start": float(run["start"] / valid) if valid else None,
                "relative_length": float(run["length"] / valid) if valid else None,
            }
        )


def _summarize(version: str, samples: pd.DataFrame, spans: pd.DataFrame) -> dict:
    summary = {
        "version": version,
        "files": int(len(samples)),
        "matched_files": int(samples["matched"].sum()) if len(samples) else 0,
        "match_methods": samples["match_method"].value_counts().astype(int).to_dict() if len(samples) else {},
        "affected_pattern_counts": samples["affected_modalities"].value_counts().astype(int).to_dict() if len(samples) else {},
        "fields": sorted({field for raw in samples["fields"] for field in str(raw).split(",")}) if len(samples) else [],
    }
    for modality in ("audio", "vision", "text", "text_bert"):
        column = f"{modality}_missing_steps"
        if column not in samples.columns:
            continue
        present = samples[f"{modality}_present"] if f"{modality}_present" in samples.columns else pd.Series(True, index=samples.index)
        part = samples[present.fillna(False)]
        if part.empty or part[column].isna().all():
            continue
        summary[modality] = {
            "samples_with_missing": int((part[column].fillna(0) > 0).sum()),
            "missing_steps": describe_series(part[column].fillna(0)),
            "missing_ratio": describe_series(part[f"{modality}_missing_ratio"].dropna())
            if f"{modality}_missing_ratio" in part
            else {},
        }
    if len(spans):
        summary["span_count"] = int(len(spans))
        summary["span_length"] = describe_series(spans["length"])
        summary["span_relative_start"] = describe_series(spans["relative_start"].dropna())
    else:
        summary["span_count"] = 0
    return summary


def _plot(version: str, samples: pd.DataFrame, spans: pd.DataFrame, figures: Path) -> None:
    if samples.empty:
        return
    pattern = samples["affected_modalities"].value_counts()
    bar_chart(
        pattern.index.tolist(),
        pattern.tolist(),
        figures / f"affected_patterns_{version}.png",
        f"附件3 {version} 缺失模态组合",
        "发生局部全零的模态",
        "样本数",
        rotation=25,
    )
    modalities = [name for name in ("text_bert", "audio", "vision", "text") if f"{name}_missing_ratio" in samples.columns]
    if modalities:
        matrix = samples[ [f"{name}_missing_ratio" for name in modalities] ].fillna(0).to_numpy(dtype=float)
        heatmap(
            matrix,
            samples["file_name"].str.replace(".pkl", "", regex=False).tolist(),
            modalities,
            figures / f"missing_ratio_heatmap_{version}.png",
            f"附件3 {version} 各样本有效区内缺失比例",
            "缺失时间步 / 有效长度",
            vmin=0,
            vmax=1,
        )
    if len(spans):
        hist_chart(spans["length"], figures / f"span_length_{version}.png", f"附件3 {version} 缺失区间长度", "连续全零时间步")
        hist_chart(
            spans["relative_start"].dropna(),
            figures / f"span_position_{version}.png",
            f"附件3 {version} 缺失区间起点（相对有效长度）",
            "起点 / 有效长度",
        )


def _analyze_one(version: str, filename: str, folder_name: str) -> dict:
    path = DATASET_DIRS["dataset02"] / filename
    print(f"[dataset03] 为对照缺失位置读取 {filename}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    index = _index_dataset02(data)
    result = analyze_version(version, DATASET_DIRS["dataset03"] / folder_name, index)
    del data, index
    return result


def analyze() -> dict:
    out = dataset_output("dataset03")
    aligned = _analyze_one("aligned", "aligned_50.pkl", "aligned")
    unaligned = _analyze_one("unaligned", "unaligned_50.pkl", "unaligned")
    samples = pd.concat([aligned["samples"], unaligned["samples"]], ignore_index=True)
    spans = pd.concat([aligned["spans"], unaligned["spans"]], ignore_index=True)
    save_df(out["tables"] / "samples.csv", samples)
    save_df(out["tables"] / "missing_spans.csv", spans)
    _plot("aligned", aligned["samples"], aligned["spans"], out["figures"])
    _plot("unaligned", unaligned["samples"], unaligned["spans"], out["figures"])

    pair_rows = []
    if len(aligned["samples"]) and len(unaligned["samples"]):
        left = aligned["samples"].set_index("sample_no")
        right = unaligned["samples"].set_index("sample_no")
        for sample_no in sorted(set(left.index) & set(right.index)):
            pair_rows.append(
                {
                    "sample_no": sample_no,
                    "aligned_file": left.loc[sample_no, "file_name"],
                    "unaligned_file": right.loc[sample_no, "file_name"],
                    "aligned_matched_id": left.loc[sample_no, "matched_id"],
                    "unaligned_matched_id": right.loc[sample_no, "matched_id"],
                    "same_source_sample": left.loc[sample_no, "matched_id"] == right.loc[sample_no, "matched_id"],
                    "aligned_affected": left.loc[sample_no, "affected_modalities"],
                    "unaligned_affected": right.loc[sample_no, "affected_modalities"],
                }
            )
    pairs = pd.DataFrame(pair_rows)
    save_df(out["tables"] / "aligned_unaligned_pairs.csv", pairs)
    summary = {
        "dataset": "dataset03",
        "description": "附件3，无标签局部模态缺失测试集。缺失指有效时间步内原本非零、当前被置零的连续区间。",
        "file_rename": {
            "aligned": "附件3_XX.pkl -> aligned_version_XX.pkl",
            "unaligned": "附件3_未对齐版本_XX.pkl -> unaligned_version_XX.pkl",
        },
        "aligned": _summarize("aligned", aligned["samples"], aligned["spans"]),
        "unaligned": _summarize("unaligned", unaligned["samples"], unaligned["spans"]),
        "paired_files": int(len(pairs)),
        "paired_same_source_sample": int(pairs["same_source_sample"].sum()) if len(pairs) else 0,
        "matched_split_counts": samples.loc[samples["matched"], "matched_split"].map(lambda x: SPLIT_ZH.get(x, x)).value_counts().astype(int).to_dict()
        if samples["matched"].any()
        else {},
    }
    save_json(out["root"] / "summary.json", summary)
    print(
        f"[dataset03] 对齐匹配 {summary['aligned']['matched_files']}/30，"
        f"未对齐匹配 {summary['unaligned']['matched_files']}/30，"
        f"同号样本同源 {summary['paired_same_source_sample']}"
    )
    return summary


if __name__ == "__main__":
    analyze()
