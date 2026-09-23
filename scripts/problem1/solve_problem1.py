"""One-command engineered solution for Problem 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem1.alignment.pipeline import run_alignment
from src.problem1.auditing import audit_problem1
from src.problem1.common.reproducibility import configure_logging, seed_everything
from src.problem1.config import load_config
from src.problem1.feature_extraction.pipeline import run_feature_extraction
from src.problem1.feature_extraction.validation import validate_features
from src.problem1.reporting import write_solution_report
from src.problem1.visualization import create_visualizations


def main() -> int:
    parser = argparse.ArgumentParser(description="求解问题一：多尺度Soft-DTW+Sinkhorn+三模态共识时间轴")
    parser.add_argument("--config", default="configs/problem1.yaml", help="统一配置文件")
    parser.add_argument("--limit", type=int, help="仅处理前N个样本，用于冒烟测试")
    parser.add_argument("--skip-features", action="store_true", help="复用已存在的特征")
    parser.add_argument("--overwrite-features", action="store_true", help="覆盖三模态特征")
    parser.add_argument("--overwrite-alignment", action="store_true", help="覆盖对齐结果")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    logger = configure_logging(cfg.path("output_root") / "logs" / "solve_problem1.log",
                               cfg.section("runtime")["log_level"])
    seed_everything(int(cfg.section("project")["seed"]))
    try:
        logger.info("步骤1/4 数据输入与特征提取")
        if not args.skip_features:
            run_feature_extraction(cfg, overwrite=args.overwrite_features, limit=args.limit, logger=logger)
        validation = validate_features(cfg, limit=args.limit)
        if not validation["passed"]:
            raise RuntimeError(f"特征校验失败: {validation['error_count']}项")
        logger.info("步骤2/4 参数初始化: config_fingerprint=%s", cfg.fingerprint[:12])
        logger.info("步骤3/4 模型调用")
        records = run_alignment(cfg, overwrite=args.overwrite_alignment, limit=args.limit, logger=logger)
        logger.info("步骤4/4 验收审计、结果输出与典型样本可视化")
        audit = audit_problem1(cfg, records)
        if not audit["overall_passed"] and args.limit is None:
            raise RuntimeError(f"问题一验收审计失败: {audit['mapping_error_count']}项映射/填充错误")
        summary = create_visualizations(cfg, records, audit)
        write_solution_report(cfg, records, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        logger.exception("问题一求解失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
