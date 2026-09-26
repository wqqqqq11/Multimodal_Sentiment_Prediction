"""CLI for strict official-tokenizer Problem 3 admission validation."""

from __future__ import annotations

import argparse
import json

from data_progressing.problem3.config import load_config
from data_progressing.problem3.official_validation import validate_official_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/problem3.yaml")
    args = parser.parse_args()
    result = validate_official_artifacts(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

