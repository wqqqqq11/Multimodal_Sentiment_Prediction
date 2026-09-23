from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_solution_report(
    path: Path,
    run_id: str,
    device: str,
    parameter_count: int,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    robustness: pd.DataFrame,
    attribution: dict[str, Any],
) -> None:
    worst = attribution["worst_macro_f1_scenario"]
    text = f"""# 问题2模型求解运行报告

## 1 运行结论

本次运行 `{run_id}` 完成了完整教师训练、缺失学生蒸馏、验证集稳健性实验、消融实验、独立测试集评价以及附件3全量推理。模型参数量为 {parameter_count:,}，计算设备为 `{device}`。

| 数据划分 | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| 验证集 | {validation_metrics['accuracy']:.4f} | {validation_metrics['macro_f1']:.4f} | {validation_metrics['mae']:.4f} | {validation_metrics['pearson']:.4f} |
| 独立测试集 | {test_metrics['accuracy']:.4f} | {test_metrics['macro_f1']:.4f} | {test_metrics['mae']:.4f} | {test_metrics['pearson']:.4f} |

最弱分类场景是 `{worst['pattern']}` 模态在 `{worst['position']}` 位置连续缺失 {worst['missing_rate']:.0%}，对应 Macro-F1={worst['macro_f1']:.4f}。该结论由验证集受控遮蔽得到，不使用附件3无标签数据调参。

## 2 求解步骤与注意事项

### 2.1 数据输入

读取 `datasets/preprocessed_data/problem2` 下的对齐版 NPZ 数据和训练掩码银行。训练、验证、测试和附件3保持 `aligned_50` 同一接口；附件3只用于最终推理。

注意：不得把填充零、自然零值和人工缺失混为一类；不得使用测试集或附件3标签进行结构选择。

### 2.2 参数初始化

固定 NumPy、PyTorch 和 CUDA 随机种子；创建独立教师与学生网络。教师先在完整样本上训练，学生以教师最优权重作为初始化。

注意：缺失视图采样概率必须在 [0,1]；蒸馏温度必须大于0；全部损失系数必须非负。门控可靠性系数越大，模型越抑制高缺失模态，应在验证集内调试。

### 2.3 模型调用

教师优化分类、回归、相关性与极性一致性损失。学生在经验分布连续遮蔽样本上，同时优化真实标签监督损失以及分类概率、回归值、融合表示、样本关系和门控贡献蒸馏损失。早停指标取完整验证场景与语音视觉同时缺失20%场景综合分数。

注意：梯度裁剪阈值需大于0；蒸馏权重采用预热，避免训练初期错误教师信号主导；显存不足时优先下调 batch size，不改变序列接口。

### 2.4 结果输出

保存时间戳日志、教师/学生最优权重、历史指标、验证集稳健性网格、消融结果、测试集逐样本误差、附件3预测 CSV 和论文可用图表。

注意：竞赛总附件不超过50MB；提交前删除非必要中间检查点并确认预测 CSV 共30条、样本编号无重复、强度落在 [-3,3]。

## 3 缺失规律实验

稳健性网格共 {len(robustness)} 个场景，覆盖语音缺失、视觉缺失、双模态缺失，句首/句中/句尾位置，以及配置中的多个缺失率。完整明细见 `metrics/robustness_results.csv`，规律汇总见 `metrics/missingness_attribution.json`。

## 4 输出文件说明

- `logs/train.log`：含时间戳的完整训练日志。
- `checkpoints/teacher_best.pt` 与 `student_best.pt`：最优模型参数。
- `predictions/problem2_attachment3_predictions.csv`：附件3竞赛预测主文件。
- `predictions/problem2_attachment3_diagnostics.csv`：概率、门控与可用率诊断文件，不作为主提交文件。
- `metrics/*.csv|json`：性能、消融、稳健性和错误归因结果。
- `figures/*.png`：训练、缺失规律、消融和全量预测可视化。
"""
    path.write_text(text, encoding="utf-8")
