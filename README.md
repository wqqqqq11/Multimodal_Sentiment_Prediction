# 复杂场景下多模态情感预测

## 目录结构

```text
Multimodal_Sentiment_Prediction/
├── data_analysis_code/          # 赛方数据集分析代码
│   ├── run_all.py               # 依次分析四个附件并生成报告
│   ├── analyze_dataset01.py     # 附件1：原始视频与标注
│   ├── analyze_dataset02.py     # 附件2：对齐 / 未对齐特征
│   ├── analyze_dataset03.py     # 附件3：局部模态缺失测试集
│   ├── analyze_dataset04.py     # 附件4：可解释性测试集
│   ├── build_report.py          # 汇总统计并写出分析报告
│   ├── config.py                # 路径与常量
│   ├── feature_stats.py         # 序列长度、填充与全零区间
│   ├── io_utils.py              # 表格与 JSON 写出
│   ├── plotting.py              # 分布图
│   └── requirements.txt
├── data_progressing/               # 数据处理代码
├── datasets/
│   ├── original_data_from_the_competition_organizer/
│   │   ├── dataset01/           # 附件1：37 个 video_id 目录、100 个 mp4、label-100.xlsx
│   │   ├── dataset02/           # 附件2：aligned_50.pkl、unaligned_50.pkl、label.xlsx
│   │   ├── dataset03/
│   │   │   ├── aligned/         # aligned_version_01.pkl … aligned_version_30.pkl
│   │   │   └── unaligned/       # unaligned_version_01.pkl … unaligned_version_30.pkl
│   │   └── dataset04/
│   │       ├── aligned/         # 01.pkl … 20.pkl，videos/01.mp4 … 20.mp4
│   │       └── unaligned/       # 01.pkl … 20.pkl，videos/01.mp4 … 20.mp4
│   └── preprocessed_data/       # 预处理后的数据
├── docs/                        # 赛题文档
├── outputs/
│   ├── data_analysis_results/   # 分析结果
│   │   ├── analysis_report.md
│   │   ├── dataset01/           # summary.json、tables/、figures/
│   │   ├── dataset02/
│   │   ├── dataset03/
│   │   ├── dataset04/
│   │   └── overview/
│   └── logs/
├── src/
│   └── utils/
├── strategy/                    # 解题策略
├── main.py
├── requirements.txt
└── README.md
```

分析代码的运行方式：

```bash
python data_analysis_code/run_all.py
```
