# 问题一数据预处理

`problem1_preprocess.py`处理附件1的100条原始视频和`label-100.xlsx`。它不会修改赛题原始数据，也不会隐式下载预训练模型。

## 默认运行

```bash
python data_progressing/problem1_preprocess.py
```

默认完成：

- 标签与视频一一对应校验；
- 视频SHA256、容器元数据、实际可解码帧数、分辨率、编码和抽样质量检查；
- 文本Unicode/空白规范化及可回映字符偏移；
- 5fps视觉采样时间轴；
- 清单、配置和验证报告写出。

默认输出到`datasets/preprocessed_data/problem1/`。

## 可选媒体抽取

写出5fps JPEG帧。若曾使用旧版时间轴抽过帧，请增加
`--overwrite-media`，它会覆盖当前帧并清理同一样本目录中的过期帧：

```bash
python data_progressing/problem1_preprocess.py --extract-frames --overwrite-media
```

写出16kHz单声道WAV需要本机FFmpeg：

```bash
python data_progressing/problem1_preprocess.py --extract-audio --ffmpeg C:\path\to\ffmpeg.exe
```

脚本不会自动安装FFmpeg。重复运行默认复用已有媒体文件；显式传入`--overwrite-media`才会覆盖。

## 输出

```text
datasets/preprocessed_data/problem1/
├── manifest.csv
├── preprocess_config.json
├── reports/preprocess_report.json
├── tables/labels_normalized.csv
├── tables/tokens.csv
├── tables/visual_timeline.csv
├── audio/                         # 仅--extract-audio
└── frames/                        # 仅--extract-frames
```

`visual_timeline.csv`依据顺序解码得到的真实帧时间戳，选择最接近5fps目标时刻的可解码帧，供后续LibreFace/OpenFace特征提取使用。容器估算帧数只保留在`metadata_frame_count`中用于审计，不再用于抽帧。`tokens.csv`只用于文本追溯，不替代后续BERT WordPiece分词。

## 测试

```bash
python -m unittest discover -s data_progressing -p "test_*.py"
```
