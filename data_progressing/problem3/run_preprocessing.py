"""CLI entry point for Problem 3 preprocessing."""

from __future__ import annotations

import argparse
import json

from data_progressing.problem3.config import load_config
from data_progressing.problem3.preprocessing import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/problem3.yaml")
    args = parser.parse_args()
    report = run_preprocessing(load_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

