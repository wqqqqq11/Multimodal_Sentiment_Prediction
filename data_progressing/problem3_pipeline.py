"""Robust CLI for Problem 3 preprocessing and analysis.

It uses the pinned fast tokenizer when its files exist locally and otherwise activates
the audited low-confidence character mapping fallback without changing model inputs.
"""

from __future__ import annotations

import argparse
import json

from data_progressing.problem3.analysis import run_analysis
from data_progressing.problem3.config import load_config
from data_progressing.problem3 import preprocessing
from data_progressing.problem3.runtime_text_mapping import build_token_mappings, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/problem3.yaml")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    preprocessing.load_tokenizer = load_tokenizer
    preprocessing.build_token_mappings = build_token_mappings
    report = preprocessing.run_preprocessing(config)
    result = {"preprocessing": report}
    if not args.skip_analysis:
        result["analysis"] = run_analysis(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

