"""依次分析四个赛方数据集，并把图表、表格和报告写到 outputs/data_analysis_results。"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_dataset01 import analyze as analyze_dataset01
from analyze_dataset02 import analyze as analyze_dataset02
from analyze_dataset03 import analyze as analyze_dataset03
from analyze_dataset04 import analyze as analyze_dataset04
from build_report import build as build_report


def main() -> int:
    scratch = Path(__file__).resolve().parents[1] / "outputs" / "data_analysis_results" / "_doc_extract.txt"
    if scratch.exists():
        scratch.unlink()
    steps = [
        ("dataset01", analyze_dataset01),
        ("dataset02", analyze_dataset02),
        ("dataset03", analyze_dataset03),
        ("dataset04", analyze_dataset04),
        ("report", build_report),
    ]
    failed = []
    for name, func in steps:
        started = time.perf_counter()
        print(f"\n===== {name} =====")
        try:
            func()
        except Exception:
            failed.append(name)
            traceback.print_exc()
        else:
            print(f"===== {name} 完成，用时 {time.perf_counter() - started:.1f}s =====")
    if failed:
        print("失败步骤:", ", ".join(failed))
        return 1
    print("\n全部分析完成。结果目录: outputs/data_analysis_results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
