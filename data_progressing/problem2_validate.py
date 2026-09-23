"""Validate saved Problem 2 preprocessing artifacts without rebuilding them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_progressing.problem2.config import load_config
from data_progressing.problem2.validation import validate_saved_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Problem 2 preprocessed arrays and masks")
    parser.add_argument("--config", default="configs/problem2.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    result = validate_saved_root(cfg.path("preprocessed_root"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
