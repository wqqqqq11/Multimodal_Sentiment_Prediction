"""CLI acceptance validation for generated Problem 3 artifacts."""

from __future__ import annotations

import argparse
import json

from data_progressing.problem3.config import load_config
from data_progressing.problem3.validation import validate_saved_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/problem3.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    audit = validate_saved_root(
        config.path("preprocessed_root"), config.section("validation")["expected_split_counts"]
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

