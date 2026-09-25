# 问题三 HSAIG-Net-v3

新版直接使用赛方 `aligned_50.pkl` 文本表征，并且只在附件2训练集上拟合 PCA。文本维度由
768 压缩到 256，累计解释方差约 79.46%。附件2验证、测试和附件4不参与 PCA 拟合。

模型包含三模态时序编码器、Softmax 到 Entmax-1.5 的渐进稀疏注意力、样本级模态门控、
分类与回归独立融合层、单模态辅助监督、积分梯度以及删除和保留实验。训练保留正式指标分数
最高的五个检查点，并生成单个权重平均检查点。

## 训练

256维文本特征已经生成。正式训练执行：

```powershell
.\.venv\Scripts\python.exe scripts\problem3\solve_problem3.py --device cuda --run-name hsaig_v3
```

如需重新生成文本特征：

```powershell
.\.venv\Scripts\python.exe data_progressing\problem3_text_features.py --overwrite
```

运行目录中的关键文件：

- `checkpoints/candidate_epoch_*.pt`：正式指标分数最高的多个候选检查点；
- `checkpoints/hsaig_topk_averaged.pt`：用于最终推理的权重平均检查点；
- `metrics/goal_audit.json`：四项正式目标验收；
- `metrics/summary.json`：参数量、平均轮次与验证/测试指标；
- `predictions/problem3_attachment4_predictions_and_explanations.csv`：附件4全量结果；
- `explanations/attachment4_evidence.csv`：原始文本、语音时段和视觉帧证据。

验收目标为 Accuracy ≥ 0.65、Macro-F1 ≥ 0.62、MAE ≤ 0.55、Pearson ≥ 0.65。
只有四项全部满足才写入 `outputs/problem3/submission`。
