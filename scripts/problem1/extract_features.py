"""CLI for problem 1 feature extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem1.common.reproducibility import configure_logging, seed_everything
from src.problem1.config import load_config
from src.problem1.feature_extraction.pipeline import run_feature_extraction
from src.problem1.feature_extraction.validation import validate_features


def main() -> int:
    parser = argparse.ArgumentParser(description="问题一三模态特征提取")
    parser.add_argument("--config", default="configs/problem1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    logger = configure_logging(cfg.path("output_root") / "logs" / "feature_extraction.log",
                               cfg.section("runtime")["log_level"])
    seed_everything(int(cfg.section("project")["seed"]))
    try:
        records = run_feature_extraction(cfg, overwrite=args.overwrite, limit=args.limit, logger=logger)
        report = validate_features(cfg, limit=args.limit)
        logger.info("特征提取结束: 样本=%d, 校验错误=%d", len(records), report["error_count"])
        return 0 if report["passed"] else 2
    except Exception:
        logger.exception("特征提取失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
