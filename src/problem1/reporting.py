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
    audit = summary["acceptance_audit"]
    modality_uncertainty = summary["mean_uncertainty_by_modality"]
    temporal = summary["temporal_diagnostics_by_modality"]
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
- Sinkhorn 熵正则 ε：{alignment['sinkhorn_epsilon']}，最大迭代 {alignment['sinkhorn_iterations']}，原始边缘残差容差 {alignment['sinkhorn_tolerance']}。
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

- 样本覆盖率：{audit['sample_coverage_rate']:.2%}；重复ID：{audit['duplicate_sample_id_count']}；缺失ID：{len(audit['missing_sample_ids'])}。
- 特征完整性校验：{'通过' if audit['feature_validation_passed'] else '未通过'}；错误 {audit['feature_validation_error_count']} 项，质量警告 {audit['feature_validation_warning_count']} 项（分模态：{audit['feature_validation_warning_count_by_modality']}）。
- 映射与填充验收通过率：{audit['mapping_audit_pass_rate']:.2%}；映射/填充错误 {audit['mapping_error_count']} 项。
- 特征/对齐配置指纹一致性：{'通过' if audit['feature_config_fingerprint_match'] and audit['resolved_config_fingerprint_match'] else '未通过'}。
- 平均目标函数：{summary['mean_objective']:.6f}。
- 对齐不确定性：均值 {summary['mean_uncertainty']:.6f}，P90 {summary['p90_uncertainty']:.6f}，最大值 {summary['max_uncertainty']:.6f}；不确定性≥0.40的样本 {summary['high_uncertainty_sample_count']} 个。
- 分模态平均不确定性：文本 {modality_uncertainty['text']:.6f}，音频 {modality_uncertainty['audio']:.6f}，视觉 {modality_uncertainty['vision']:.6f}。
- 上述值是离散传输熵，受各模态时间步密度影响，不作为跨模态准确率。时间尺度感知不确定性（时间标准差/共识间隔）：文本 {temporal['text']['mean_normalized_temporal_uncertainty']:.6f}，音频 {temporal['audio']['mean_normalized_temporal_uncertainty']:.6f}，视觉 {temporal['vision']['mean_normalized_temporal_uncertainty']:.6f}。
- 音频时间诊断：平均时间标准差 {temporal['audio']['mean_temporal_std_sec']:.4f}s；加权90%区间均值 {temporal['audio']['mean_weighted_90pct_span_sec']:.4f}s；峰值偏差均值 {temporal['audio']['mean_peak_offset_sec']:.4f}s。
- 原始 Sinkhorn 收敛率：{summary['raw_sinkhorn_convergence_rate']:.2%}；样本级三模态全部收敛率：{summary['sample_sinkhorn_convergence_rate']:.2%}。
- 最大原始边缘残差：{summary['max_raw_marginal_residual']:.6e}；边缘修正后的最大残差：{summary['max_marginal_residual']:.6e}。
- 单调违例总数：{summary['total_monotonic_violations']}。
- 单调投影前违例总数：{summary['total_preprojection_monotonic_violations']}；最终值只用于确认约束满足。
- 提前达到共识容差的样本比例：{convergence:.2f}%（未提前停止并不表示失败，只表示运行到配置的迭代上限）。
- 代表性样本：`{summary['representative_sample']}`。

`aligned_dataset.npz` 是可直接用于后续情感预测的定长张量；每个样本目录中的 `mapping.json` 用于回溯词元、音频窗和视频帧，`metrics.json` 用于论文消融与误差分析。

`acceptance_audit.csv/json` 给出100条样本的覆盖、映射、有效长度和零填充验收结果。原始残差用于判断迭代求解是否收敛；修正后残差用于确认最终传输计划严格满足边缘约束，两者不再混用。

## 5. 图表结论口径

- 图 1 左侧展示赛方要求对应的覆盖、特征、映射和收敛通过率，右侧展示按秒计算后归一化的三模态时间不确定性分布。
- 图 2 展示样本目标函数的中位归一化收敛轨迹，用于判断共识迭代是否稳定。
- 图 3 展示代表性样本三模态传输矩阵；亮色质量沿近对角单调带分布时，说明时间顺序得到保留。
- 图 4 是赛方要求的典型样本验证图：上方展示可回溯的文本片段、语音峰值时段和视频源帧，下方展示三类特征在统一共识时间轴上的对应关系及位置级不确定性。

## 6. 典型样本核验材料

- 逐共识位置明细：`{summary['representative_alignment_table']}`。
- 图文核验说明：`{summary['representative_validation_report']}`。
- 典型样本图：`{summary['figures']['representative_validation']}`。
"""
    atomic_text(cfg.path("output_root") / "reports" / "model_solution.md", text)
