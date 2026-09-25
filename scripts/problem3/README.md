# 问题三 P0–P2 修复版求解命令

问题三只使用统一配置 `configs/problem3.yaml`。正式运行前应确认问题三预处理验收已经通过，且 `datasets/preprocessed_data/problem3` 中存在 train、valid、test、attachment4_aligned 四个 NPZ 文件。

## 完整求解

```powershell
uv run python scripts/problem3/solve_problem3.py --stage all --device cuda --num-workers 2
```

流程严格为：加载并冻结问题二完整教师 → 文本骨干热启动 → 局部窗口证据训练 → 稀疏选择器退火 → 全阶段验证择优 → 训练后只读原型库 → 验证/内部测试 → 附件4全量推理。`--stage all` 不使用附件4选择模型或阈值。

注意事项：

- `model.top_k=8` 是上限；实际预算按候选数自适应，最少 4 条证据。
- 每条证据为中心位置前后各 2 步的局部窗口，预测器不会读取完整上下文序列。
- `model.prototype_fusion_alpha=0.0`：原型仅用于训练后的相似案例解释，禁止反馈到预测。
- 选择器温度从 1.5 退火到 0.35，Gumbel 噪声从 1.0 退火到 0.05。
- 最优模型可来自 warmup 或 selector 任一阶段；不再强制用较差的后期检查点覆盖前期最优。
- `training_summary.json` 中 `performance_gate_passed` 必须为 `true`，并检查中性类召回率不得低于 0.10。
- Windows 的 `--num-workers` 建议从 0 或 2 开始；脚本已支持 DataLoader 多进程并发。
- 正式比较必须报告 Accuracy、Macro-F1、MAE、Pearson 和 95% bootstrap 区间，同时查看充分性、完备性和稳定性诊断。

## 仅训练

```powershell
uv run python scripts/problem3/solve_problem3.py --stage train --device cuda
```

每次运行在 `outputs/problem3/modeling/runs/<时间戳>/` 建立独立目录，日志写入 `logs/train.log`，最佳与最近检查点分别写入 `checkpoints/best.pt` 和 `checkpoints/last.pt`。

手动运行后，优先提供以下文件用于下一轮诊断：`logs/train.log`、`metrics/training_summary.json`、`metrics/training_history.json`、验证集与测试集指标 JSON，以及 `metrics/validation_error_attribution.json`。

## 从中断点恢复

```powershell
uv run python scripts/problem3/solve_problem3.py --stage all --device cuda --resume outputs/problem3/modeling/runs/<运行ID>/checkpoints/last.pt
```

## 只做评价或附件4推理

```powershell
uv run python scripts/problem3/solve_problem3.py --stage evaluate --device cuda --checkpoint outputs/problem3/modeling/runs/<运行ID>/checkpoints/best.pt
uv run python scripts/problem3/solve_problem3.py --stage infer --device cuda --checkpoint outputs/problem3/modeling/runs/<运行ID>/checkpoints/best.pt
```

附件4核心文件为：

- `predictions/attachment4_prediction_and_explanation.csv`：全量预测与可读解释汇总；
- `explanations/attachment4_evidence.csv`：逐证据精确定位、局部重要性与训练原型；
- `explanations/attachment4_explanations.jsonl`：保留层级结构的复核文件；
- `reports/attachment4_explanation_cards.md`：典型样本解释卡；
- `figures/attachment4_evidence_heatmap.png`：局部证据重要性分布图。

音视频时段/帧号来自当前比例映射，映射置信度为 0.35，必须与定位一起提交说明；文本字符区间来自官方 BERT tokenizer 精确 offset。样本 13 的视觉失效会显式写入 `quality_flags`，不会被候选修复文件静默替换。
