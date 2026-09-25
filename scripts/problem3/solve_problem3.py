from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.problem3.pipeline import run_solution  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="问题3HSAIG-v2训练、验收和解释")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "problem3_model.yaml")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="仅1轮训练和2条解释，用于执行链检查")
    args = parser.parse_args()
    try:
        run_dir = run_solution(PROJECT_ROOT, args.config, args.device, args.smoke, args.run_name)
    except KeyboardInterrupt:
        print("[ERROR] 用户中断训练，已保留日志。", file=sys.stderr); return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr); return 2
    print(f"[OK] 问题三运行完成，是否达标请查看 metrics/goal_audit.json：{run_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
