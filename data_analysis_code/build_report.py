"""汇总四个数据集的统计结果，写出总览表和中文分析报告。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUTPUT_ROOT, POLARITY_ZH, SPLIT_ZH
from io_utils import dataset_output, load_json, save_df, save_json
from plotting import bar_chart, grouped_bar


def _fmt(stats: dict | None, digits: int = 3) -> str:
    if not stats or not stats.get("count"):
        return "无有效数据"
    return (
        f"n={stats['count']}，均值 {stats['mean']:.{digits}f}，标准差 {stats['std']:.{digits}f}，"
        f"中位数 {stats['median']:.{digits}f}，范围 [{stats['min']:.{digits}f}, {stats['max']:.{digits}f}]"
    )


def _counts(mapping: dict | None) -> str:
    if not mapping:
        return "无"
    parts = []
    for key, value in mapping.items():
        label = POLARITY_ZH.get(key, SPLIT_ZH.get(key, key))
        parts.append(f"{label} {value}")
    return "，".join(parts)


def _read(name: str) -> dict:
    path = OUTPUT_ROOT / name / "summary.json"
    return load_json(path) if path.exists() else {}


def _cross_dataset(out: dict[str, Path]) -> dict:
    rows = []
    ds01_path = OUTPUT_ROOT / "dataset01" / "tables" / "samples.csv"
    ds02_path = OUTPUT_ROOT / "dataset02" / "tables" / "samples_aligned.csv"
    ds03_path = OUTPUT_ROOT / "dataset03" / "tables" / "samples.csv"
    cross = {"dataset01_in_dataset02": None, "dataset03_source_splits": None, "dataset04_in_dataset02": None}
    if ds01_path.exists() and ds02_path.exists():
        ds01 = pd.read_csv(ds01_path, dtype=str)
        ds02 = pd.read_csv(ds02_path, dtype=str)
        ds01["video_id"] = ds01["video_id"].astype(str)
        ds01["clip_id"] = ds01["clip_id"].astype(str)
        ds02["video_id"] = ds02["video_id"].astype(str)
        ds02["clip_id"] = ds02["clip_id"].astype(str)
        merged = ds01.merge(ds02[["video_id", "clip_id", "split", "regression_label"]], on=["video_id", "clip_id"], how="left")
        merged["found"] = merged["split"].notna()
        save_df(out["tables"] / "dataset01_in_dataset02.csv", merged[["video_id", "clip_id", "split", "label", "regression_label", "found"]])
        split_counts = merged.loc[merged["found"], "split"].value_counts().astype(int).to_dict()
        cross["dataset01_in_dataset02"] = {
            "labeled_rows": int(len(ds01)),
            "found": int(merged["found"].sum()),
            "missing": int((~merged["found"]).sum()),
            "split_counts": split_counts,
        }
        if split_counts:
            grouped_bar(
                [SPLIT_ZH.get(key, key) for key in split_counts],
                {"附件1样本": list(split_counts.values())},
                out["figures"] / "dataset01_split_membership.png",
                "附件1的100条样本落在附件2的哪个划分",
                "附件2划分",
                "样本数",
            )
    if ds03_path.exists():
        ds03 = pd.read_csv(ds03_path)
        if "matched_split" in ds03.columns:
            counts = ds03.loc[ds03["matched"] == True, "matched_split"].value_counts().astype(int).to_dict()
            cross["dataset03_source_splits"] = counts
    ds04 = _read("dataset04")
    cross["dataset04_in_dataset02"] = ds04.get("found_in_dataset02")
    for name, summary in (
        ("dataset01", _read("dataset01")),
        ("dataset02", _read("dataset02")),
        ("dataset03", _read("dataset03")),
        ("dataset04", _read("dataset04")),
    ):
        if name == "dataset02":
            rows.append({"dataset": name, "role": "训练/验证/测试特征", "samples": summary.get("aligned", {}).get("sample_count"), "labeled": True})
        elif name == "dataset01":
            rows.append({"dataset": name, "role": "原始视频，问题1", "samples": summary.get("label_rows"), "labeled": True})
        elif name == "dataset03":
            rows.append({"dataset": name, "role": "局部缺失测试，问题2", "samples": summary.get("aligned", {}).get("files"), "labeled": False})
        else:
            rows.append({"dataset": name, "role": "完整模态解释测试，问题3", "samples": summary.get("sample_count"), "labeled": False})
    save_df(out["tables"] / "dataset_roles.csv", pd.DataFrame(rows))
    bar_chart(
        [row["dataset"] for row in rows if row["samples"]],
        [row["samples"] for row in rows if row["samples"]],
        out["figures"] / "sample_counts.png",
        "四个附件的样本规模",
        "数据集",
        "样本数",
    )
    save_json(out["root"] / "cross_dataset.json", cross)
    return cross


def _section_dataset01(summary: dict) -> list[str]:
    if not summary:
        return ["附件1结果缺失。"]
    lines = [
        f"附件1对应 `dataset01`，是问题1的原始材料：{summary.get('video_id_count')} 个 video_id、"
        f"{summary.get('label_rows')} 条标注、{summary.get('video_files')} 个 mp4。",
        f"标注表与视频文件一一对应：缺视频 {len(summary.get('labels_without_video') or [])} 条，"
        f"无标注视频 {len(summary.get('videos_without_label') or [])} 个，"
        f"无法解码 {len(summary.get('unreadable_videos') or [])} 个，重复键 {summary.get('duplicate_sample_keys')}。",
        f"情感强度 {_fmt(summary.get('regression'))}。极性计数：{_counts(summary.get('polarity_counts'))}。"
        f"文字标签 annotation 与分数极性一致比例为 {summary.get('annotation_score_agreement'):.3f}。",
        f"转写词数 {_fmt(summary.get('text_word_len'), 2)}。视频时长 {_fmt(summary.get('duration_sec'), 2)} 秒，"
        f"帧率 {_fmt(summary.get('fps'), 2)}，文件大小 {_fmt(summary.get('file_size_mb'), 2)} MB。",
        f"赛题给出的时长范围是 2.648–34.567 秒，实测落在该范围外的视频有 {summary.get('duration_outside_contest_range')} 条"
        "（允许 0.05 秒容器计时误差）。分辨率以 1280×720 为主（56 条），其余为 640×360、480×270 等更低分辨率，帧率多数是 30，最低约 23.98。",
        f"同一 video_id 的片段数 {_fmt(summary.get('clips_per_video'), 2)}。建模时样本键必须是 video_id + clip_id，不能只用文件夹名。",
    ]
    return lines


def _section_dataset02(summary: dict) -> list[str]:
    if not summary:
        return ["附件2结果缺失。"]
    aligned = summary.get("aligned", {})
    unaligned = summary.get("unaligned", {})
    excel = summary.get("label_excel", {})
    ids = summary.get("id_set_comparison", {})
    agree = summary.get("aligned_unaligned_label_agreement", {})
    lines = [
        "附件2是问题2和问题3的监督数据。外层键为 train / valid / test，样本按字段堆叠，"
        "读取方式是 `data[split][field][index]`，不是 `data[split][index][field]`。",
        f"对齐版 {aligned.get('sample_count')} 条，划分 {aligned.get('split_counts')}；"
        f"未对齐版 {unaligned.get('sample_count')} 条，划分 {unaligned.get('split_counts')}。"
        f"两边共有 ID {ids.get('shared_ids')}，只在对齐版 {ids.get('only_in_aligned_count')}，"
        f"只在未对齐版 {ids.get('only_in_unaligned_count')}。",
        f"对齐版回归标签 {_fmt(aligned.get('regression'))}。极性：{_counts(aligned.get('polarity_counts'))}。"
        "分类编码与极性完全一致：0 负向 1380，1 中性 1100，2 正向 2370。"
        "正向约占 48.9%，负向 28.5%，中性 22.7%，准确率会被多数类抬高。",
        f"label.xlsx 有 {excel.get('excel_rows')} 行，与对齐 pkl 匹配 {excel.get('matched_ids')} 条，"
        f"分数最大绝对差 {excel.get('max_abs_score_diff')}，划分字段 mode 与 pkl 划分一致比例 {excel.get('mode_equals_pkl_split')}，"
        f"annotation 与 pkl 极性一致比例 {excel.get('annotation_matches_pkl_polarity')}。",
        f"对齐与未对齐在共有 ID 上的划分一致比例 {agree.get('same_split')}，回归标签一致比例 {agree.get('same_regression')}，"
        f"分类一致比例 {agree.get('same_classification')}，BERT 有效长度一致比例 {agree.get('same_text_bert_valid_len')}。",
        f"对齐版文本 BERT 有效长度 {_fmt(aligned.get('text_bert_valid_len'), 2)}，"
        f"句首 token 为 101（BERT 的 [CLS]）的比例 {aligned.get('text_bert_cls_101_ratio')}。"
        f"音频非零步数 + 2 等于 BERT 有效长度的比例 {aligned.get('audio_bert_length_agreement')}。"
        "这个差值对应 [CLS] 和 [SEP]：对齐后的音频/视觉第 0 步在训练集上是结构性全零，词级特征从第 1 步开始。",
        f"对齐版有效长度：音频 {_fmt(aligned.get('audio_valid_length'), 2)}，视觉 {_fmt(aligned.get('vision_valid_length'), 2)}。"
        f"有效区内仍有全零步的样本：音频 {aligned.get('samples_with_audio_internal_zeros')}，"
        f"视觉 {aligned.get('samples_with_vision_internal_zeros')}，文本特征 {aligned.get('samples_with_text_feature_internal_zeros')}。"
        f"视觉整段非零步为 0 的样本 {aligned.get('vision_all_zero_samples')}，音频 {aligned.get('audio_all_zero_samples')}。"
        "因此完整训练集里视觉本来就存在自然全零，问题2不能把所有全零都当成人工缺失。",
        f"未对齐版音频有效长度 {_fmt(unaligned.get('audio_valid_length'), 2)}，"
        f"视觉 {_fmt(unaligned.get('vision_valid_length'), 2)}，文本仍是 50 步 BERT。"
        f"未对齐版内部全零样本：音频 {unaligned.get('samples_with_audio_internal_zeros')}，"
        f"视觉 {unaligned.get('samples_with_vision_internal_zeros')}。",
        f"注意力掩码内部空洞样本：对齐 {aligned.get('attention_mask_internal_zero_samples')}，"
        f"未对齐 {unaligned.get('attention_mask_internal_zero_samples')}。"
        f"近常数特征维行数：对齐 {aligned.get('near_constant_dimension_rows')}，未对齐 {unaligned.get('near_constant_dimension_rows')}。"
        "各维均值、标准差和零占比见 dimension_stats_*.csv。没有标准差接近 0 的废维。",
        "未对齐音频在官方 audio_lengths 内部的全零时间步为 0。"
        "因此附件3未对齐音频里的内部全零不能用训练集的自然稀疏来解释，就是局部缺失。"
        "对齐音频在 4850 条里只有 5 条出现内部全零，比例约 0.06%；附件3对齐版有 27/30 条出现，平均缺失比例约 18.5%。",
        "转写词数中位数约 17，但存在极端长文本，最长样本 125344$_$0 有 309 个词，位于验证集。",
        "两套文件必须整题固定用一套。对齐版三模态都是最多 50 步且时间位置对齐；"
        "未对齐版文本仍是 50×768，语音 500×74、视觉 500×35，有效长度看 audio_lengths 和 vision_lengths。",
    ]
    return lines


def _section_dataset03(summary: dict) -> list[str]:
    if not summary:
        return ["附件3结果缺失。"]
    lines = [
        "附件3是问题2的无标签测试集，对齐和未对齐各 30 个文件，每个文件 1 条样本。"
        "中文文件名已改为 aligned_version_XX.pkl 与 unaligned_version_XX.pkl。",
        "缺失判定：先把测试样本配回附件2的原始特征，再把“原始非零、当前全零”的有效时间步记为被置零区间。"
        "对齐版音频/视觉第 0 步的结构性全零不计入缺失；未对齐版用附件2给出的有效长度，避免把尾部填充当成缺失。",
    ]
    for key, title in (("aligned", "对齐版"), ("unaligned", "未对齐版")):
        part = summary.get(key, {})
        lines.append(
            f"{title}字段 {part.get('fields')}。成功配回附件2的文件 {part.get('matched_files')}/{part.get('files')}，"
            f"匹配方式 {part.get('match_methods')}。缺失组合计数：{part.get('affected_pattern_counts')}。"
            f"缺失区间 {part.get('span_count')} 段，长度 {_fmt(part.get('span_length'), 2)}，"
            f"相对起点 {_fmt(part.get('span_relative_start'), 3)}。"
        )
        for modality in ("audio", "vision", "text_bert", "text"):
            block = part.get(modality)
            if not block:
                continue
            lines.append(
                f"{title}{modality}：有缺失的样本 {block.get('samples_with_missing')}，"
                f"缺失步数 {_fmt(block.get('missing_steps'), 2)}，缺失比例 {_fmt(block.get('missing_ratio'), 3)}。"
            )
    lines.append(
        f"同编号的对齐/未对齐文件配到同一条附件2样本的有 {summary.get('paired_same_source_sample')}/"
        f"{summary.get('paired_files')}，编号是 01、14、24、25，对应 "
        "221153$_$16、EO_5o9Gup6g$_$10、zhNksSReaQk$_$27、194299$_$14，四条都在附件2测试集。"
        "其余 26 条的原文和特征都对不上这 4850 条，最近邻音频绝对误差在 3 到 15 之间，不是同一条样本的浮点误差。"
    )
    lines.append(
        "配回成功的样本上，被置零的是原来非零的时间步，区间长度大多是 1 或 2，最长 4。"
        "对没有配回的样本，缺失比例用的是有效区内的内部全零，其中已去掉对齐版第 0 步和尾部填充。"
        "这些区间的起点均匀分布在整段有效时间上，不是固定出现在句首或句尾。"
        "文本侧没有发现缺失：对齐版 text_bert 的 30 条都没有被置零，未对齐版保留了完整 raw_text。"
        "对齐版另有 3 条音频和视觉都没有内部全零。若缺失被加在序列末尾，会和填充连在一起，这种方法会把它漏掉。"
    )
    lines.append(
        "未对齐版最长的两段视觉全零出现在 unaligned_version_03：第 3–49 步共 47 步，以及第 90–107 步共 18 步。"
        "其余长区间很少，长度达到 5 以上的只有个别视觉或音频片段。"
    )
    lines.append(
        "对齐版没有 raw_text 和 768 维 text，文本入口是 float32 的 text_bert；"
        "未对齐版没有 text / text_bert，只保留 raw_text 以及 float64 的 audio、vision。"
        "两套测试文件的字段集合并不相同，推理时要按版本读取，不能直接套用附件2的键。"
    )
    return lines


def _section_dataset04(summary: dict) -> list[str]:
    if not summary:
        return ["附件4结果缺失。"]
    return [
        f"附件4有 {summary.get('sample_count')} 条无标签样本。对齐字段 {summary.get('aligned_fields')}；"
        f"未对齐字段 {summary.get('unaligned_fields')}。",
        f"同一编号的两套特征中，原文一致 {summary.get('raw_text_consistent')}，768 维文本一致 {summary.get('text_features_consistent')}，"
        f"text_bert 一致 {summary.get('text_bert_consistent')}。",
        f"对齐版内部全零样本：音频 {summary.get('aligned_audio_internal_zero_samples')}，"
        f"视觉 {summary.get('aligned_vision_internal_zero_samples')}，文本 {summary.get('aligned_text_internal_zero_samples')}。"
        f"BERT 长度与音频非零步 + 2 一致的样本 {summary.get('bert_audio_length_agreement')}/{summary.get('sample_count')}。",
        f"未对齐长度字段与“最后一个非零步”一致：音频 {summary.get('audio_length_equals_inferred')}/20，"
        f"视觉 {summary.get('vision_length_equals_inferred')}/20。"
        "不一致的是样本 13 和 16：vision_lengths 都写成 1，但视觉特征里最后一个非零步分别在第 18 和第 44 步。"
        "若建模时直接按 vision_lengths 截断，这两条会丢掉后面的视觉内容。"
        f"未对齐有效区内全零样本：音频 {summary.get('unaligned_audio_internal_zero_samples')}，"
        f"视觉 {summary.get('unaligned_vision_internal_zero_samples')}。附件4本身没有附件3那种局部缺失。",
        f"未对齐语音长度 {_fmt(summary.get('unaligned_audio_lengths'), 2)}，视觉 {_fmt(summary.get('unaligned_vision_lengths'), 2)}，"
        f"转写词数 {_fmt(summary.get('text_word_len'), 2)}。",
        f"原始视频在对齐/未对齐目录中 MD5 相同的有 {summary.get('identical_video_pairs')}/{summary.get('video_pairs')}，"
        f"时长 {_fmt(summary.get('duration_sec'), 2)} 秒，分辨率 {summary.get('resolution_counts')}，"
        f"无法解码 {summary.get('unreadable_videos')}。",
        f"按原文能在附件2找到的样本有 {summary.get('found_in_dataset02')} 条，且都在测试集："
        "03→GAVpYuhMZAw$_$6，07→ChhZna-aBK4$_$8，08→2m58ShI1QSI$_$2，12→gE7kUqMqQ9g$_$2，15→SKTyBOhDX6U$_$9。"
        "另外 15 条是附件2未收录的新视频。问题3要求把关键证据映射回文本片段、语音时段或视频帧；"
        "视频文件名与 pkl 的 id 都是 01–20，可以直接对齐。两套目录里的 mp4 逐字节相同，保留一份即可。",
    ]


def _section_cross(cross: dict) -> list[str]:
    info = cross.get("dataset01_in_dataset02") or {}
    lines = [
        f"附件1的 {info.get('labeled_rows')} 条里，只有 {info.get('found')} 条能在附件2用 video_id + clip_id 找到，"
        f"另外 {info.get('missing')} 条没有预计算特征。能找到的划分是：{_counts(info.get('split_counts'))}，验证集为 0。",
        "label-100.xlsx 的说明页写着这 100 条在原始 CMU-MOSEI 中属于训练集 64、验证集 10、测试集 26。"
        "那是完整语料上的划分，不是本次下发的 4850 条特征文件里的划分。说明页还确认极性为负向 18、中性 25、正向 57，与分数规则一致。",
        "问题1要自行从 mp4 提取文本、语音、视觉特征并对齐，不能假设附件2已经给这 100 条准备好了特征。"
        "其中 7 条同时出现在附件2测试集，用附件1标签做特征抽查时不要拿这 7 条去调附件2的测试结果。",
        "附件3里能配回附件2的 4 条全部属于测试集；附件4里能配回的 5 条也全部属于测试集。"
        "专项测试和附件2测试集有交集，但大部分专项样本是 4850 条之外的新片段。",
    ]
    return lines


def build() -> Path:
    out = dataset_output("overview")
    cross = _cross_dataset(out)
    ds01, ds02, ds03, ds04 = _read("dataset01"), _read("dataset02"), _read("dataset03"), _read("dataset04")
    sections = [
        ("1. 附件1：原始视频", _section_dataset01(ds01)),
        ("2. 附件2：标准化特征", _section_dataset02(ds02)),
        ("3. 附件3：局部缺失测试集", _section_dataset03(ds03)),
        ("4. 附件4：可解释性测试集", _section_dataset04(ds04)),
        ("5. 跨数据集关系", _section_cross(cross)),
        (
            "6. 后续建模时直接可用的数据事实",
            [
                "情感强度是 [-3, 3] 的连续值，极性由符号决定，0 只算中性。分类评测用 Accuracy 和 F1，回归评测用 MAE 和 Pearson。",
                "附件2类别不平衡，正向样本明显多于负向和中性，F1 需要看各类而不是只看准确率。",
                "text 与 text_bert 二选一，text_bert 不是第四个模态。它的三行是 token id、attention mask、segment id。",
                "对齐序列要单独处理第 0 步和尾部填充；未对齐序列要用各自的 length，不能把 500 步全部当成有效帧。",
                "问题2的缺失是有效区间内部的连续全零，不是整模态删除。评估缺失率时应使用配回附件2之后的置零比例，而不是把填充零算进去。",
                "问题3的 20 条视频与特征编号一致，解释结果里的时间位置应能回指到转写、语音长度或视频帧。",
            ],
        ),
    ]
    lines = [
        "# 赛方数据集分析报告",
        "",
        "数据来自 2026 年中国研究生数学建模竞赛 E 题“复杂场景下多模态情感预测的数学建模与算法设计”。",
        "统计由 `data_analysis_code/run_all.py` 生成，明细表和分布图在各数据集子目录的 `tables` 与 `figures` 中。",
        "",
    ]
    for title, paragraphs in sections:
        lines.append(f"## {title}")
        lines.append("")
        for paragraph in paragraphs:
            lines.append(paragraph)
            lines.append("")
    lines.append("## 输出目录")
    lines.append("")
    lines.append("- `dataset01/`、`dataset02/`、`dataset03/`、`dataset04/`：各附件的 summary.json、tables、figures")
    lines.append("- `overview/`：样本量、附件1所在划分、跨数据集对照")
    lines.append("- `analysis_report.md`：本报告")
    lines.append("")
    path = OUTPUT_ROOT / "analysis_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")
    return path


if __name__ == "__main__":
    build()
