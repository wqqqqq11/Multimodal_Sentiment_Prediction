# 多模态情感预测与可解释分析

本项目包含三个任务：问题一生成并对齐100条样本的文本、音频和视觉时序特征；问题二针对局部模态缺失场景完成附件3情感预测；问题三完成附件4情感预测与多模态解释。根目录中的 `submisson/` 为按竞赛要求整理的最终提交材料，训练缓存、赛方原始数据和非必要中间结果未纳入提交目录。

## 一、提交材料说明

当前 `submisson/` 共9个文件，总大小约31.459MB，低于竞赛规定的50MB上限。

### 1. 问题一

- `submisson/problem1/features/多模态时序特征文件.npz`：100条样本的多模态时序特征。文件包含样本编号、文本特征、音频特征、视觉特征、共识表示、对齐时间轴、不确定性和有效时间步掩码；三个模态的特征形状均为 `(100, 50, 16)`。
- `submisson/problem1/reports/问题1-100条样本多模态特征汇总.csv`：100条样本的模态类型、有效时长、原始特征维度、原始步数和对齐后步数汇总。
- `submisson/problem1/reports/问题1-100条样本特征验收明细.csv`：逐样本验收结果。
- `submisson/problem1/reports/问题1-100条样本特征验收汇总.json`：问题一整体验收统计。

问题一提交目录只保留赛方要求的特征文件及配套验收材料，不重复提交项目代码。

### 2. 问题二

- `submisson/problem2/model/problem2_best_model.pt`：问题二最终FP16学生模型参数。
- `submisson/problem2/results/问题2-预测结果.csv`：附件3全部30条样本的情感极性与情感强度预测结果。

问题二结果文件的主要字段为样本编号、情感极性和情感强度。附件3只用于最终推理，不参与模型训练、阈值选择或参数调整。

### 3. 问题三

- `submisson/problem3/model/problem3_best_model.pt`：问题三最终FP16模型参数。
- `submisson/problem3/results/问题3-预测与解释结果.csv`：附件4全部20条样本的预测与解释主文件，包含预测极性、预测强度、类别概率、置信度、模态权重、模态贡献和关键证据。
- `submisson/problem3/results/问题3-附件4多模态证据明细.csv`：附件4的文本片段、音频时间段和视觉帧级补充证据。

附件4只用于最终预测与解释，不参与训练集统计量、校准参数或模型结构的选择。

## 二、项目目录

```text
Multimodal_Sentiment_Prediction/
├── configs/          三个问题的参数配置
├── data_progressing/ 数据集预处理、映射与验收规则
├── scripts/          三个问题的运行入口
├── src/              特征提取、模型训练、评价与解释核心代码
├── submisson/        最终提交材料
├── README.md         项目与复现说明
└── requirements.txt  Python依赖
```

配置文件是实验参数的唯一来源：

- `configs/problem1.yaml`：问题一特征提取、对齐与验收参数。
- `configs/problem2_model.yaml`：问题二模型、训练、缺失增强与评价参数。
- `configs/problem3_model.yaml`：问题三模型、训练、校准与解释参数。
- `configs/problem2.yaml`、`configs/problem3.yaml`：赛方原始数据到模型输入的路径及预处理约束记录。

## 三、运行环境说明

项目已验证的运行环境如下：

| 项目 | 版本或要求 |
|---|---|
| 操作系统 | Windows 64位 |
| Python | 3.12.10 |
| PyTorch | 2.5.1+cu118 |
| TorchVision | 0.20.1+cu118 |
| CUDA运行时 | 11.8 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Transformers | 4.57.6 |
| 随机种子 | 2026 |

建议使用 `uv` 创建独立虚拟环境。GPU环境可按以下方式安装：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
uv pip check --python .\.venv\Scripts\python.exe
```

没有CUDA设备时，可从PyTorch官方源安装对应CPU版本，并在运行命令中使用 `--device cpu`。完整训练建议使用CUDA设备；问题二教师网络和问题一预训练编码器的显存占用较高。

问题一首次提取特征时需要根据 `configs/problem1.yaml` 下载固定模型，包括DeBERTa-v3、WavLM、ConvNeXt和MediaPipe人脸关键点模型。问题二首次从头训练时需要下载配置中指定的完整BERT教师模型和四层紧凑BERT学生模型。模型名称及固定修订版本均记录在配置文件中。

## 四、数据集预处理规则

赛方原始数据不包含在项目中。预处理程序始终只读原始数据，所有派生产物写入 `datasets/preprocessed_data/`，参数和路径由 `configs/problem1.yaml`、`configs/problem2.yaml`、`configs/problem3.yaml` 管理。所有命令均应在项目根目录执行。

### 1. 目录层级

```text
data_progressing/
├── problem1/
│   └── preprocess.py              问题一原始视频、标签、音频与视觉时间轴预处理
├── problem2/
│   ├── run_preprocessing.py       问题二预处理入口
│   ├── run_validation.py          问题二产物验收入口
│   ├── preprocessing.py           数据转换主流程
│   ├── masking.py                 缺失状态与连续缺失掩码
│   ├── scaling.py                 训练集稳健缩放
│   ├── quality.py                 模态质量与可靠性
│   └── validation.py              数组、标签和掩码契约校验
└── problem3/
    ├── run_pipeline.py            预处理与数据分析一体化入口
    ├── run_preprocessing.py       独立预处理入口
    ├── build_text_features.py     训练集拟合的PCA文本特征
    ├── run_analysis.py            数据质量与分布分析
    ├── run_validation.py          常规产物验收
    ├── run_official_validation.py 官方分词器映射验收
    ├── preprocessing.py           数据转换主流程
    ├── text_mapping.py            文本词元与字符区间映射
    ├── media_mapping.py           音频时间段与视觉帧映射
    └── validation.py              数据、映射和证据契约校验
```

所有预处理入口均建议从项目根目录通过 `python -m` 调用，避免工作目录变化导致相对路径失效。

### 2. 问题一预处理规则

1. 读取附件1的 `label-100.xlsx` 和100个MP4文件，使用 `video_id + "$_$" + clip_id` 构造稳定样本主键，并检查标签、视频和样本编号是否一一对应。
2. 文本只执行Unicode NFKC和连续空白规范化，不改变大小写、语义或标点；同时保存词元字符起止位置，保证后续解释可回映原文。
3. 视频必须按实际可解码帧顺序读取，不信任容器声明的总帧数；视觉时间轴依据真实时间戳生成，默认目标采样率为5帧/秒。
4. 音频可通过FFmpeg抽取为16kHz、单声道、PCM WAV；视觉帧和音频默认按需生成，不修改原始视频。
5. 为每个原始文件记录SHA-256、分辨率、帧率、时长、亮度、清晰度和质量标记；严格模式下缺文件、坏标签、不可解码视频或媒体抽取失败均返回非零状态。

执行：

```powershell
.\.venv\Scripts\python.exe -m data_progressing.problem1.preprocess
```

如需同时生成WAV和采样帧：

```powershell
.\.venv\Scripts\python.exe -m data_progressing.problem1.preprocess --extract-audio --extract-frames
```

### 3. 问题二预处理规则

1. 仅使用附件2的训练划分拟合音频和视觉缩放器；验证集、测试集和附件3不得参与统计量估计。
2. 对训练集观测位置执行分位数截尾和IQR稳健缩放；非有限值按训练统计处理，填充、自然全零和检测到的缺失位置在缩放后保持精确零值。
3. 文本统一使用 `text_bert` 的词元编号、注意力掩码和分段编号，明确区分内容位置、CLS/SEP结构位置和尾部填充位置。
4. 对三个模态分别保存“真实观测、自然零值、人工或检测缺失”掩码，三种状态互斥且不得扩展到结构位或填充位。
5. 训练掩码银行只遮蔽真实观测的音频和视觉连续片段，不遮蔽文本；缺失位置、比例和组合由固定随机种子生成，并保证遮蔽后仍保留最少观测步数。
6. 附件3的30条样本只生成无标签推理输入和质量审计，不参与训练、缩放器拟合或模型选择。

执行与验收：

```powershell
.\.venv\Scripts\python.exe -m data_progressing.problem2.run_preprocessing --overwrite
.\.venv\Scripts\python.exe -m data_progressing.problem2.run_validation
```

主要输出：

```text
datasets/preprocessed_data/problem2/
├── train.npz
├── valid.npz
├── test.npz
├── challenge_aligned.npz
├── train_mask_bank.npz
├── scaler.json
└── challenge_manifest.csv
```

### 4. 问题三预处理规则

1. 问题三沿用问题二的训练、验证、测试划分和训练集缩放器，禁止使用附件4重新拟合缩放参数。
2. 附件4共20条样本，统一生成文本、音频、视觉观测掩码、质量掩码和证据候选掩码；标签字段不写入附件4产物。
3. 文本使用固定分词器生成词元—字符区间映射，并记录截断、精确匹配范围和映射置信度；低置信度回退映射必须显式标记。
4. 音频证据映射到真实秒区间，视觉证据映射到原始帧号；映射表与模型输入使用相同样本顺序和有效内容掩码。
5. 768维文本表示只使用训练集有效内容位置拟合PCA并压缩到256维；验证集、测试集和附件4只应用已固定的均值、投影矩阵和尺度。
6. 反事实基线、原型候选和位置统计只由训练集构造；验证扰动种子固定，确保解释指标可复现。
7. 对附件4执行分布漂移、截断、视觉失效和历史文本重合审计；修复数据只作为独立候选，不覆盖官方对齐特征，也不复制历史标签。

推荐执行顺序：

```powershell
.\.venv\Scripts\python.exe -m data_progressing.problem3.run_pipeline
.\.venv\Scripts\python.exe -m data_progressing.problem3.build_text_features --overwrite
.\.venv\Scripts\python.exe -m data_progressing.problem3.run_validation
.\.venv\Scripts\python.exe -m data_progressing.problem3.run_official_validation
```

若只需单独执行预处理或分析：

```powershell
.\.venv\Scripts\python.exe -m data_progressing.problem3.run_preprocessing
.\.venv\Scripts\python.exe -m data_progressing.problem3.run_analysis
```

主要输出：

```text
datasets/preprocessed_data/problem3/
├── train.npz
├── valid.npz
├── test.npz
├── attachment4_aligned.npz
├── text_features/
│   ├── train_text.npz
│   ├── valid_text.npz
│   ├── test_text.npz
│   ├── attachment4_text.npz
│   ├── pca_state.npz
│   └── text_baselines.npz
├── counterfactual/baseline_statistics.npz
├── mappings/attachment4_time_mapping.csv
├── mappings/attachment4_video_metadata.csv
├── quality/sample_quality.csv
└── leakage_watchlist.csv
```
### 5. 通用预处理约束

- 原始赛题文件始终只读，禁止在原始目录中覆盖或改写数据。
- 必须先完成训练、验证和测试划分，再拟合缩放器、PCA、原型、反事实基线及其他统计量。
- 验证集、普通测试集、附件3和附件4不得参与训练统计量或模型参数的拟合。
- 样本编号和样本顺序必须显式保存，并在原始数据、预处理数组、预测结果和解释证据之间逐阶段校验。
- 标签只允许写入有监督划分；附件3和附件4的预处理产物不得包含未知真实标签。
- 所有数值数组必须检查形状、数据类型、有限值、标签范围、掩码互斥性和不可用位置的零值约束。
- 每次预处理均应保存实际配置、源文件SHA-256、质量报告和验收结果，保证数据来源与处理过程可追溯。
- 低置信度映射、异常样本和修复候选必须单独标记，不得静默覆盖官方数据。
## 五、运行方法

所有命令均在项目根目录执行。

### 1. 问题一

完整执行特征提取、特征校验、多模态对齐、验收与可视化：

```powershell
.\.venv\Scripts\python.exe scripts\problem1\solve_problem1.py
```

### 2. 问题二


正式训练、评价并生成附件3预测：

```powershell
.\.venv\Scripts\python.exe scripts\problem2\solve_problem2.py --device cuda
```

运行结果写入 `outputs/problem2/runs/<运行编号>/`。正式运行会输出最佳学生模型、验证与测试指标、缺失稳健性结果以及附件3预测文件。

### 3. 问题三

正式训练、评价并生成附件4预测与解释：

```powershell
.\.venv\Scripts\python.exe scripts\problem3\solve_problem3.py --device cuda
```

运行结果写入 `outputs/problem3/runs/<运行编号>/`，其中包括最佳模型、评价指标、附件4预测、证据明细和解释图表。

## 六、复现与合规说明

- 三个问题具体训练参数以 `configs/` 中的配置为准。
- 问题二和问题三的验证集、普通测试集与专项附件数据严格分离；附件3和附件4不参与调参。
- 提交目录未包含赛方原始数据、训练缓存、教师模型、未压缩检查点或其他大体积中间文件。
- 最终提交材料总大小约31.459MB，符合不超过50MB的限制。
- 提交前应再次以赛方最终发布的文件命名、字段模板和压缩包规则为准。