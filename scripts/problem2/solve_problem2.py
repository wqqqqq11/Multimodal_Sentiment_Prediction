from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problem2.pipeline import run_solution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="求解问题2：掩码感知门控网络与完整—缺失一致性蒸馏",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "problem2_model.yaml", help="模型配置文件")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--run-name", default=None, help="可选运行目录名；默认使用毫秒级时间戳")
    parser.add_argument("--smoke", action="store_true", help="仅训练1个epoch并缩小缺失网格，用于检查全流程")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = run_solution(PROJECT_ROOT, args.config, args.device, args.smoke, args.run_name)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[ERROR] 用户中断训练；已保留本次运行日志。", file=sys.stderr)
        return 130
    except Exception as exc:  # final safety boundary for CLI execution
        print(f"[ERROR] 未预期异常 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[OK] 问题2求解完成：{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
