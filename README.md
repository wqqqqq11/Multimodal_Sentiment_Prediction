# 复杂场景下多模态情感预测

## 目录结构

```text
Multimodal_Sentiment_Prediction/
├── configs/
│   └── problem1.yaml                    # 问题一统一配置入口
├── data_analysis_code/                   # 赛方数据集分析代码
├── data_progressing/
│   ├── problem1_preprocess.py            # 问题一基础媒体预处理
│   ├── test_problem1_preprocess.py
│   └── README.md
├── scripts/
│   └── problem1/
│       ├── extract_features.py           # 三模态特征提取入口（占位）
│       └── validate_features.py          # 三模态特征验收入口（占位）
├── src/
│   ├── problem1/
│   │   ├── config.py                     # 配置加载与校验（占位）
│   │   ├── schemas.py                    # 统一特征结构（占位）
│   │   ├── io.py                         # 特征与元数据读写（占位）
│   │   ├── feature_extraction/
│   │   │   ├── pipeline.py               # 三模态流程编排（占位）
│   │   │   ├── registry.py               # 提取器注册（占位）
│   │   │   ├── validation.py             # 特征校验（占位）
│   │   │   ├── text/                     # 文本特征、映射与质量
│   │   │   ├── audio/                    # 声学特征、分帧与质量
│   │   │   └── vision/                   # 人脸、表情与视觉质量
│   │   ├── alignment/                    # Soft-DTW、Sinkhorn、多尺度与共识时间轴
│   │   └── common/                       # 设备、批处理、时间戳与复现工具
│   └── utils/
├── tests/
│   └── problem1/                         # 问题一特征提取测试占位
├── datasets/
│   ├── original_data_from_the_competition_organizer/
│   │   ├── dataset01/                    # 附件1：原始视频、文本与标签
│   │   ├── dataset02/                    # 附件2：对齐/未对齐特征
│   │   ├── dataset03/
│   │   └── dataset04/
│   └── preprocessed_data/
│       └── problem1/
│           ├── audio/                    # 100条16 kHz单声道WAV
│           ├── frames/                   # 3977张可解码视觉帧
│           ├── tables/                   # 标签、词元和视觉时间轴
│           ├── features/                 # 三模态未对齐特征输出目录
│           │   └── samples/              # 每个样本的text/audio/vision特征
│           └── aligned/                  # 多模态软对齐结果预留目录
├── docs/                                 # 赛题文档
├── outputs/
│   ├── data_analysis_results/            # 数据分析结果
│   └── problem1/
│       ├── logs/
│       ├── reports/
│       └── figures/
├── strategy/                             # 解题策略
├── Problem_Analysis.md
├── main.py
├── requirements.txt
└── README.md
```

## 问题一代码组织约定

- 不按阶段拆分目录，统一按“问题—功能—模态”组织。
- `configs/problem1.yaml`是问题一唯一配置入口；实现后所有路径、模型、窗口、设备、输出和校验参数均从该文件读取。
- `data_progressing/`只负责基础媒体预处理。
- `src/problem1/feature_extraction/`负责文本、音频和视觉特征提取，各模态之间保持解耦。
- `src/problem1/alignment/`预留给多尺度编码、Soft-DTW、Sinkhorn最优传输和共识时间轴。
- `datasets/preprocessed_data/problem1/features/`保存未对齐的变长三模态特征。
- `datasets/preprocessed_data/problem1/aligned/`保存后续对齐特征、掩码和时间映射。
- `outputs/problem1/`只保存日志、报告和图表，不保存训练输入。

目前特征提取和对齐相关Python文件仅为目录与职责占位，不包含实际实现。

分析代码的运行方式：

```bash
python data_analysis_code/run_all.py
```

问题一数据预处理：

```bash
python data_progressing/problem1_preprocess.py
```

详细参数和输出说明见`data_progressing/README.md`。
