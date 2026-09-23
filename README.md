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
│       ├── extract_features.py           # 三模态特征提取入口
│       ├── validate_features.py          # 三模态特征验收入口
│       ├── align_features.py             # Soft-DTW/Sinkhorn对齐入口
│       └── solve_problem1.py             # 问题一端到端求解入口
├── src/
│   ├── problem1/
│   │   ├── config.py                     # 配置加载与校验
│   │   ├── schemas.py                    # 统一特征结构
│   │   ├── io.py                         # 原子化特征与元数据读写
│   │   ├── feature_extraction/
│   │   │   ├── pipeline.py               # 三模态并发流程编排
│   │   │   ├── registry.py               # 可插拔特征提取器注册
│   │   │   ├── validation.py             # 特征校验
│   │   │   ├── text/                     # 文本特征、映射与质量
│   │   │   ├── audio/                    # 声学特征、分帧与质量
│   │   │   └── vision/                   # 人脸、表情与视觉质量
│   │   ├── alignment/                    # Soft-DTW、Sinkhorn、单调投影与共识时间轴
│   │   └── common/                       # 设备、批处理、时间戳与复现工具
│   └── utils/
├── tests/
│   └── problem1/                         # 问题一特征、I/O与对齐单元测试
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
│           └── aligned/                  # 多模态对齐结果、映射及定长数据集
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
- `outputs/problem1/`只保存求解清单、错误审计、日志、报告和图表，不保存训练输入。

## 问题一模型求解

### 环境安装

本实现使用竞赛级预训练表征，不提供哈希文本、手工音频谱或中心人脸区域等静默降级。先安装根目录依赖：

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip check --python .venv\Scripts\python.exe
```

首次运行还需下载 MediaPipe Face Landmarker 模型：

```powershell
New-Item -ItemType Directory -Force models\mediapipe
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile "models\mediapipe\face_landmarker.task"
```

三模态最终表征为：

- 文本：`microsoft/deberta-v3-base`（`refs/pr/14`）的 safetensors 权重、最后四层子词表示，并严格聚合回赛方词元。
- 音频：`microsoft/wavlm-base-plus`（`refs/pr/2`）的 safetensors 权重、上下文语音表示及真实秒级区间。
- 视觉：MediaPipe 52维 Blendshape、关键点几何、4×4头姿矩阵和预训练 ConvNeXt 外观表示；检测到人脸时编码人脸区域，无脸镜头编码完整画面并以较低质量权重参与对齐。

DeBERTa 和 WavLM 强制设置 `use_safetensors=true`，不会回退读取 `.bin`/pickle 权重，因此可与手动安装的 PyTorch 2.5.1 GPU 版本配合使用。缺少依赖、指定修订版权重或 `face_landmarker.task` 时程序会明确失败，不会使用低质量替代特征。第一次运行会从 Hugging Face 下载 DeBERTa、WavLM 和 ConvNeXt 权重。

端到端运行（自动完成特征提取、校验、对齐、结果汇总和绘图）：

```bash
python scripts/problem1/solve_problem1.py
```

首次调试建议只处理少量样本：

```bash
python scripts/problem1/solve_problem1.py --limit 3 --overwrite-features --overwrite-alignment
```

若三模态特征已经生成，可直接复用并重新对齐：

```bash
python scripts/problem1/solve_problem1.py --skip-features --overwrite-alignment
```

主要输出：

- `datasets/preprocessed_data/problem1/features/feature_manifest.csv`：特征维数、长度和质量审计。
- `datasets/preprocessed_data/problem1/aligned/aligned_dataset.npz`：可供后续模型直接读取的定长张量。
- `datasets/preprocessed_data/problem1/aligned/samples/<sample_id>/mapping.json`：共识点至三模态源证据的映射。
- `outputs/problem1/alignment/alignment_manifest.csv`：问题一模型求解状态、收敛性和对齐质量指标。
- `outputs/problem1/alignment/acceptance_audit.csv/json`：100条样本覆盖、映射、有效长度和零填充验收结果。
- `outputs/problem1/alignment/alignment_errors.json`：模型求解失败样本及异常信息。
- `outputs/problem1/reports/model_solution.md`：参数、步骤、诊断指标和论文结论素材。
- `outputs/problem1/reports/representative_sample_alignment.csv`：典型样本逐共识位置的文本、语音和视频对应关系。
- `outputs/problem1/reports/representative_sample_validation.md`：典型样本核验说明。
- `outputs/problem1/figures/`：对齐质量、收敛曲线和代表性传输矩阵。

音频特征使用WavLM卷积感受野生成可核验时间戳，并以自适应能量VAD降低静音段对传输计划的干扰。报告同时保留离散传输熵与按秒计算的时间不确定性；跨模态比较应优先使用后者。

所有参数统一由 `configs/problem1.yaml` 管理。完整环境要求记录在根目录 `requirements.txt`。并发数分别由 `runtime.workers` 和 `runtime.alignment_workers` 控制；预训练模型在进程内共享且推理段加锁，媒体读取和样本编排仍可并发。错误会写入独立审计文件，严格模式下任一失败都会返回非零退出码。

配置发生变化时，代码会比较配置指纹并拒绝复用旧特征和旧对齐结果。当前目录中此前由基线后端产生的结果只能视为历史结果；安装依赖后必须执行：

```powershell
python scripts/problem1/solve_problem1.py --overwrite-features --overwrite-alignment
```

分析代码的运行方式：

```bash
python data_analysis_code/run_all.py
```

问题一数据预处理：

```bash
python data_progressing/problem1_preprocess.py
```

详细参数和输出说明见`data_progressing/README.md`。

## 问题二数据预处理

问题二使用附件2对齐版训练特征和附件3对齐版局部缺失专项测试。文本统一采用两套数据都具备的`text_bert`接口，音频和视觉执行仅由训练集有效观测位置拟合的分位数裁剪与IQR稳健缩放。流水线显式区分内容、CLS/SEP结构位置、padding、自然零值以及人工或检测缺失，并生成完整教师—缺失学生一一配对的连续片段掩码库。

```powershell
python data_progressing/problem2_preprocess.py --overwrite
python data_progressing/problem2_validate.py
python -m pytest tests/problem2 -q
```

代码与配置：

- `configs/problem2.yaml`：数据路径、稳健缩放、缺失分布和验收阈值；
- `data_progressing/problem2/`：掩码、缩放、质量特征、预处理和验收实现；
- `data_progressing/problem2_preprocess.py`：全量预处理入口；
- `data_progressing/problem2_validate.py`：独立验收入口；
- `tests/problem2/`：掩码语义、异常值处理和数据契约测试；
- `strategy/problem2_modeling.md`：问题二变量、假设、公式推导、训练求解、缺失规律分析与模型接入说明。

主要数据产物位于`datasets/preprocessed_data/problem2/`，审计报告位于`outputs/problem2/preprocessing/`。训练输入不使用附件2独有的768维`text`字段，从而保证与附件3字段同构。
