"""Build leakage-safe Problem 2 teacher/student preprocessing artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_progressing.problem2.config import load_config
from data_progressing.problem2.preprocessing import run_preprocessing


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess Attachment 2/3 for mask-aware gating and complete-missing distillation"
    )
    parser.add_argument("--config", default="configs/problem2.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    _configure_logging(cfg.path("output_root") / "logs" / "preprocess.log")
    try:
        result = run_preprocessing(cfg, overwrite=args.overwrite)
    except Exception:
        logging.getLogger(__name__).exception("Problem 2 preprocessing failed")
        return 1
    print(json.dumps(result["acceptance"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
