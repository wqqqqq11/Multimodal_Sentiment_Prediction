from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problem3.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三 SEPC-Net 训练、评价与附件4解释推理")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "problem3.yaml"))
    parser.add_argument("--stage", choices=("all", "train", "evaluate", "infer"), default="all")
    parser.add_argument("--checkpoint", help="evaluate/infer 使用的检查点")
    parser.add_argument("--resume", help="恢复训练检查点")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--run-name", help="可选运行目录名；默认使用毫秒时间戳")
    parser.add_argument("--num-workers", type=int, help="覆盖 DataLoader 并发进程数；Windows 建议 0-4")
    parser.add_argument("--no-warm-start", action="store_true", help="不从问题二同构文本骨干热启动")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_dir = run_pipeline(
            args.config, stage=args.stage, checkpoint=args.checkpoint, resume=args.resume,
            device_name=args.device, run_name=args.run_name, num_workers=args.num_workers,
            warm_start=not args.no_warm_start,
        )
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[Problem3] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[Problem3] 输出目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
