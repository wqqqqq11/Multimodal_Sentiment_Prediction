"""Generate a reproducible Chinese solution report from numeric outputs."""

from __future__ import annotations

from typing import Any
import numpy as np

from .config import Problem1Config
from .io import atomic_text


def write_solution_report(cfg: Problem1Config, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    good = [item for item in records if item["status"] != "failed"]
    alignment = cfg.section("alignment")
    multi = cfg.section("multiscale")
    text_cfg, audio_cfg, vision_cfg = cfg.section("text"), cfg.section("audio"), cfg.section("vision")
    convergence = 100.0 * np.mean([bool(item["converged"]) for item in good]) if good else 0.0
    text = f"""# 问题一模型求解报告

## 1. 数据输入

- 输入样本：预处理后的文本词元表、16 kHz 单声道 WAV、视觉帧时间轴与帧图像。
- 实际成功求解：{len(good)} 个样本；失败：{len(records) - len(good)} 个样本。
- 特征文件位于 `datasets/preprocessed_data/problem1/features/samples/`，均保留原位置、起止时间、质量分数和原始索引。
- 文本编码器：`{text_cfg['model_name']}`；音频编码器：`{audio_cfg['model_name']}`。
- 视觉编码器：MediaPipe Face Landmarker + `{vision_cfg['appearance_model_name']}`，不启用中心 ROI 降级。

注意事项：不得打乱 `sample_id`；音频/视觉时间单位为秒，文本位置为归一化词元顺序；任何 NaN、Inf 或非单调时间戳均由严格校验拒绝。

## 2. 参数初始化

- 多尺度半径：文本 {multi['text_radii']}，音频 {multi['audio_radii']}，视觉 {multi['vision_radii']}。
- 尺度权重：{multi['scale_weights']}；公共时序签名维数：{multi['common_dimension']}。
- Soft-DTW 平滑参数 γ：{alignment['soft_dtw_gamma']}。
- Sinkhorn 熵正则 ε：{alignment['sinkhorn_epsilon']}，最大迭代 {alignment['sinkhorn_iterations']}，容差 {alignment['sinkhorn_tolerance']}。
- 代价权重（语义/时间/质量）：{alignment['semantic_weight']} / {alignment['time_weight']} / {alignment['quality_weight']}。
- 单调投影权重：{alignment['monotone_projection_weight']}（Sinkhorn 语义耦合与一维递增分位耦合的凸组合）。
- 共识时间轴最长 {alignment['max_consensus_steps']} 点，间隔目标 {alignment['consensus_interval_sec']} 秒。

注意事项：γ、ε 必须大于 0；三类代价权重均需位于 [0,1]；过小 ε 会导致数值下溢，过大 ε 会使对齐过度平滑。

## 3. 模型调用

1. 三模态原始特征分别转化为 16 维鲁棒时序统计签名，并在短、中、长三个尺度平滑。
2. 计算语义差异、相对时间 Huber 代价和质量惩罚，Soft-DTW 前向—后向算法给出多尺度单调路径先验。
3. 将路径先验写入联合代价，Sinkhorn 迭代求满足源/目标边缘约束的最优传输计划。
4. 将 Sinkhorn 计划与保持边缘质量的一维递增分位耦合做凸组合，以显式消除跨越式时间映射。
5. 三模态传输后的表示按数据质量加权形成 Wasserstein 型共识重心，循环更新至收敛或达到上限。
6. 对每个共识点保留覆盖 {alignment['evidence_mass']:.0%} 传输质量的源证据集合，形成可解释映射。

注意事项：Soft-DTW 路径先验只在初始化时计算一次，并在重心交替迭代中固定，从而避免路径和重心同时剧烈振荡；最终仍以每轮更新后的联合代价重新求 Sinkhorn 计划。

## 4. 结果输出与诊断

- 平均目标函数：{summary['mean_objective']:.6f}。
- 平均对齐不确定性：{summary['mean_uncertainty']:.6f}（越低越确定）。
- 最大边缘残差：{summary['max_marginal_residual']:.6e}（越低越满足质量守恒）。
- 单调违例总数：{summary['total_monotonic_violations']}。
- 提前达到共识容差的样本比例：{convergence:.2f}%（未提前停止并不表示失败，只表示运行到配置的迭代上限）。
- 代表性样本：`{summary['representative_sample']}`。

`aligned_dataset.npz` 是可直接用于后续情感预测的定长张量；每个样本目录中的 `mapping.json` 用于回溯词元、音频窗和视频帧，`metrics.json` 用于论文消融与误差分析。

## 5. 图表结论口径

- 图 1 比较平均不确定性、边缘残差和单调违例，三者越低表示对齐越可靠。
- 图 2 展示样本目标函数的中位归一化收敛轨迹，用于判断共识迭代是否稳定。
- 图 3 展示代表性样本三模态传输矩阵；亮色质量沿近对角单调带分布时，说明时间顺序得到保留。
"""
    atomic_text(cfg.path("output_root") / "reports" / "model_solution.md", text)
