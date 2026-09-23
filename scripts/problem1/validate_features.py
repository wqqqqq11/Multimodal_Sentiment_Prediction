"""CLI for feature validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem1.config import load_config
from src.problem1.feature_extraction.validation import validate_features


def main() -> int:
    parser = argparse.ArgumentParser(description="校验问题一三模态特征")
    parser.add_argument("--config", default="configs/problem1.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = validate_features(load_config(ROOT / args.config), limit=args.limit)
    print(json.dumps({k: report[k] for k in ("passed", "sample_count", "error_count", "warning_count")},
                     ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
