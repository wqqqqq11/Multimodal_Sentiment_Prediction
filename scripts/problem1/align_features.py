"""CLI for Soft-DTW/Sinkhorn tri-modal alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem1.alignment.pipeline import run_alignment
from src.problem1.common.reproducibility import configure_logging, seed_everything
from src.problem1.config import load_config
from src.problem1.reporting import write_solution_report
from src.problem1.visualization import create_visualizations


def main() -> int:
    parser = argparse.ArgumentParser(description="问题一Soft-DTW+Sinkhorn三模态对齐")
    parser.add_argument("--config", default="configs/problem1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    logger = configure_logging(cfg.path("output_root") / "logs" / "alignment.log", cfg.section("runtime")["log_level"])
    seed_everything(int(cfg.section("project")["seed"]))
    try:
        records = run_alignment(cfg, overwrite=args.overwrite, limit=args.limit, logger=logger)
        summary = create_visualizations(cfg, records)
        write_solution_report(cfg, records, summary)
        logger.info("对齐完成: 样本=%d, mean_uncertainty=%.6f", len(records), summary["mean_uncertainty"])
        return 0
    except Exception:
        logger.exception("对齐求解失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
