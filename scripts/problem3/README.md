# 问题三新版 HSAIG-v2

## 为什么重构

旧版把本地 BERT 放进提交权重，FP16 约 23 MiB，验证集只满足 2/5 项目标。新版直接使用赛方
`aligned_50.pkl` 中的 768 维文本表征，并且只在附件2训练集上拟合 PCA，压缩到 128 维。
这消除了 BERT 参数，也避免附件2验证、测试和附件4进入降维拟合。

模型使用三路轻量时序编码器、由 Softmax 逐轮过渡到 Sparsemax 的时间注意力和模态门控、三模态辅助监督、
分类与强度联合损失，以及回归感知的验证集校准。所有融合特征都经过模态门控，使门控、积分梯度和删除实验
针对同一条预测路径。

## 运行

```powershell
python data_progressing/problem3_text_features.py
python scripts/problem3/solve_problem3.py --device cuda
```

快速检查执行链：

```powershell
python scripts/problem3/solve_problem3.py --device cuda --smoke
```

若文本特征目录已存在但需要重建：

```powershell
python data_progressing/problem3_text_features.py --overwrite
```

每次运行都会生成独立目录 `outputs/problem3/runs/<run_id>`。关键文件为：

- `metrics/goal_audit.json`：五项目标和是否通过验收；
- `metrics/summary.json`：参数量、FP16 估算大小、最佳轮次和完整指标；
- `predictions/problem3_attachment4_predictions_and_explanations.csv`：附件4全量结果；
- `explanations/attachment4_evidence.csv`：文本片段、语音时段和视觉帧证据。

只有验证集五项目标全部通过，正式运行才会写入 `outputs/problem3/submission`。未达标的模型和结果
仍留在本次 run 目录用于诊断，不会覆盖可提交版本。
