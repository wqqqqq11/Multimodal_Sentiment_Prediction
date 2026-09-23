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

# 问题二数据预处理

问题二的全部预处理实现位于`data_progressing/problem2/`，不与`src/`中的模型代码混放。入口文件为：

```powershell
python data_progressing/problem2_preprocess.py --overwrite
python data_progressing/problem2_validate.py
```

目录职责如下：

```text
data_progressing/
├── problem2_preprocess.py         # 附件2、3全量预处理入口
├── problem2_validate.py           # 已生成数据的独立验收入口
└── problem2/
    ├── config.py                  # 配置读取与约束检查
    ├── masking.py                 # 结构、padding、自然零值与连续缺失掩码
    ├── scaling.py                 # 训练集拟合的分位数裁剪与IQR缩放
    ├── quality.py                 # 标签复核、类别权重和门控可靠性特征
    ├── preprocessing.py           # 附件2、3流程编排与报告生成
    ├── validation.py              # 张量形状、掩码互斥和数值验收
    └── io.py                      # 原子化NPZ、JSON和CSV写出
```

输出仍位于`datasets/preprocessed_data/problem2/`，审计日志和报告位于`outputs/problem2/`。预处理只生成模型可消费的数据与掩码，不包含网络结构、训练循环或预测模型。

## 测试

```bash
python -m unittest discover -s data_progressing -p "test_*.py"
python -m pytest tests/problem2 -q
```
