# 问题三预处理输出契约与验收标准

本文规定问题三预处理的目录、字段、独立性、验收门槛和执行顺序。与 `problem3_preprocessing.md`、`problem3_data_analysis.md`、`problem3_evidence_preprocessing.md` 共同构成完整方案。

## 1 项目目录

问题三新增独立入口和模块，既不修改问题一、二产物，也不在预处理目录实现模型。

```text
configs/
└── problem3.yaml
data_progressing/
├── problem3_preprocess.py
├── problem3_validate.py
└── problem3/
    ├── config.py
    ├── io.py
    ├── masking.py
    ├── scaling.py
    ├── text_mapping.py
    ├── media_mapping.py
    ├── prototype_index.py
    ├── counterfactual.py
    ├── quality.py
    ├── preprocessing.py
    └── validation.py
tests/
└── problem3/
datasets/preprocessed_data/
└── problem3/
outputs/
└── problem3/
```

问题三可只读引用问题二的 `scaler.json` 和干净划分，但输出必须写入自己的目录。配置记录上游文件SHA256；若上游哈希变化，问题三拒绝静默复用旧产物。

## 2 数据输出结构

```text
datasets/preprocessed_data/problem3/
├── preprocess_config.json
├── source_manifest.csv
├── scaler_reference.json
├── leakage_watchlist.csv
├── train.npz
├── valid.npz
├── test.npz
├── attachment4_aligned.npz
├── masks/
│   ├── train_candidate_masks.npz
│   ├── valid_candidate_masks.npz
│   ├── test_candidate_masks.npz
│   └── attachment4_candidate_masks.npz
├── mappings/
│   ├── token_offsets.csv
│   ├── attachment4_time_mapping.csv
│   ├── attachment4_frame_mapping.csv
│   └── mapping_audit.csv
├── prototypes/
│   ├── prototype_source_manifest.csv
│   ├── intensity_bins.json
│   └── prototype_sampling_index.npz
├── counterfactual/
│   ├── baseline_statistics.npz
│   ├── perturbation_templates.json
│   └── validation_seeds.npz
├── repair/
│   ├── length_overrides.csv
│   ├── sample13_official.npz
│   ├── sample13_repair_candidate.npz
│   └── repair_validation.json
├── quality/
│   ├── sample_quality.csv
│   ├── feature_drift.csv
│   └── truncation_report.csv
└── reports/
    ├── preprocessing_report.json
    └── acceptance_audit.json
```

## 3 主NPZ字段

训练、验证、测试主文件至少包含：

- `sample_id`、`raw_text`；
- `input_ids`、`attention_mask`、`token_type_ids`；
- `content_mask`、`structural_mask`、`padding_mask`；
- `audio`、`vision`；
- 三模态 `observed_mask`、`natural_zero_mask`、`failure_mask`；
- 三模态 `evidence_candidate_mask`；
- `classification_labels`、`regression_labels`；
- `intensity_bin`和基础质量字段。

附件4采用相同输入接口但不包含伪标签，额外保存视频路径索引、时间映射索引、帧映射索引、截断标记和修复分支标记。

`privileged_text`不进入问题三主NPZ，以防预测器绕过稀疏证据。如果对照实验需要，只能写入独立审计文件，问题三主加载器默认不可见。

## 4 数据处理验收

### 4.1 完整性与隔离

1. 附件2样本数严格为3395/728/727，附件4严格为20；
2. 官方划分、ID、标签和源文件哈希未改变；
3. 附件4和附件2测试集不进入训练索引与原型库；
4. 问题三运行不修改问题一、问题二产物；
5. 全部拟合统计量有证据证明只使用附件2训练集；
6. 对齐与未对齐版本没有在主输入中混用。

### 4.2 数值与掩码

1. 输入dtype符合契约且没有NaN/Inf；
2. 每条序列满足 `content + structural + padding = 50`；
3. `observed_mask`只能位于内容区；
4. 证据候选只能位于观测内容区且不与失败位置重叠；
5. 标准化后结构、padding和不可用位置保持精确零；
6. 原始值、缩放值、掩码和处理日志可逐样本追溯；
7. 训练集缩放器哈希与问题二冻结版本一致。

### 4.3 映射可复核性

1. 附件4全部可选文本位置具有字符区间；
2. 可选音频和视觉位置具有合法时间区间；
3. 时间区间单调、非负且不超过视频时长；
4. 帧范围可在实际视频中解码；
5. `[CLS]`、`[SEP]`和padding不能被选为证据；
6. 样本07和18保存模型可见文本范围和截断风险；
7. 样本13官方分支与修复候选都有完整记录；
8. 低置信度映射明确标记，不能伪装成精确边界。

### 4.4 原型与反事实

1. 原型候选来源划分全部为train；
2. 每个候选能回到训练ID、位置、标签和文本语境；
3. 验证、测试和附件4不出现在原型源索引；
4. 反事实删除不误删结构标记或改变有效长度定义；
5. 同一固定种子重复生成的扰动索引完全一致；
6. 预测器输入不存在未选特征或完整上下文旁路；
7. 选择器和预测器可见字段白名单通过自动检查。

## 5 自动停止条件

出现以下任一情况时停止处理，不得静默继续：

- 字段、样本数、形状或标签与预期不一致；
- 重新分词与赛方词元无法建立单调对应；
- 附件4特征与视频无法按编号一一配对；
- 训练集以外的数据进入缩放器、类别权重或原型统计；
- 时间区间越界或帧无法解码；
- 修复候选未通过阈值却被写入主输入；
- 输出路径将覆盖问题一或问题二文件；
- 模型可见数据契约中出现 `privileged_text` 或未选择的完整序列特征。

## 6 推荐执行顺序

### 阶段A 数据审计

1. 冻结源文件哈希、字段模式和官方划分；
2. 生成附件2标签、长度、零值和异常统计；
3. 生成附件4视频、特征、文本、交集和长度异常报告；
4. 固定泄漏监控名单。

### 阶段B 基础预处理

1. 核验问题二训练集缩放器；
2. 生成问题三独立train/valid/test和附件4主文件；
3. 构造结构、内容、padding、观测、自然零值和失败掩码；
4. 生成漂移和样本质量表。

### 阶段C 证据映射

1. 建立WordPiece到原文字符和空格词映射；
2. 对附件4执行词级语音强制对齐；
3. 通过真实PTS生成视频帧映射；
4. 人工核验代表样本和异常样本。

### 阶段D 原型与反事实准备

1. 生成只来自训练集的原型候选清单；
2. 固定强度分箱、候选采样和验证扰动种子；
3. 保存多基线统计和反事实规则；
4. 执行证据旁路与数据泄漏检查。

### 阶段E 验收与冻结

1. 运行独立验收；
2. 输出 `acceptance_audit.json`；
3. 冻结配置、源哈希和数据版本；
4. 验收通过后才允许开始问题三模型实现。

## 7 论文表述主线

问题三的数据处理可概括为：

> 训练集拟合的稳健标准化保证数值可比，互斥语义掩码保证证据候选真实有效，词元—语音—视频时间映射保证解释可回看，训练集原型清单保证案例来源可追溯，固定反事实模板保证解释忠实度可重复验证。

论文必须披露附件4的5条测试集文本交集、样本07和18的截断风险、样本13对齐视觉全零与未对齐长度异常、样本16未对齐视觉长度异常，以及所有异常的保留、修复候选和回退规则。
