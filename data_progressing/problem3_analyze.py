"""CLI entry point for Problem 3 descriptive data analysis."""

from __future__ import annotations

import argparse
import json

from data_progressing.problem3.analysis import run_analysis
from data_progressing.problem3.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/problem3.yaml")
    args = parser.parse_args()
    result = run_analysis(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

