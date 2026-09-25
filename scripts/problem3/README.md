# 问题三建模入口

本目录实现“层次稀疏注意力与积分梯度复核”。训练、阈值和校准参数只使用附件2训练集与验证集；附件4仅做最终推理。

正式运行：

```powershell
.\.venv\Scripts\python.exe scripts\problem3\solve_problem3.py --device cuda
```

快速检查执行链：

```powershell
.\.venv\Scripts\python.exe scripts\problem3\solve_problem3.py --device cuda --smoke
```

主要输出位于`outputs/problem3/runs/<run_id>/`：

- `metrics/summary.json`和`goal_audit.json`：五项指标及目标验收；
- `predictions/problem3_attachment4_predictions_and_explanations.csv`：附件4全量主结果；
- `explanations/attachment4_evidence.csv`：文本、语音时段和视觉帧证据；
- `explanations/attachment4_explanations.json`：门控、积分梯度及忠实度完整记录；
- `figures/`：训练曲线、验证性能、模态贡献和典型解释卡；
- `reports/model_solution_report.md`：论文结果素材；
- `outputs/problem3/submission/`：FP16权重、配置和最终CSV。

配置入口为`configs/problem3_model.yaml`。当前音视频时间边界来自低置信度比例回退，正式论文和结果中必须披露该限制。
