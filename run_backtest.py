"""回测入口：指定实验配置，或在 IDE 中使用下方默认路径。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 切换实验时只需修改此路径。
EXPERIMENT_PATH = PROJECT_ROOT / "private_strategy" / "beginner_example" / "experiment.yaml"
SYSTEM_PATH = PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"


# 回测命令行入口：解析实验和系统路径，切换项目目录并调用 run_experiment()，打印摘要并返回退出码。
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行一个 Rule 或 Model 回测。")
    parser.add_argument(
        "experiment", nargs="?", type=Path, default=EXPERIMENT_PATH,
        help="实验 YAML；省略时使用脚本中的 EXPERIMENT_PATH",
    )
    parser.add_argument(
        "--system", type=Path, default=SYSTEM_PATH, help="系统配置 YAML；默认沿用原有配置",
    )
    args = parser.parse_args(argv)
    try:
        from etf_backtest.experiment import run_experiment

        os.chdir(PROJECT_ROOT)
        result = run_experiment(args.experiment, system_path=args.system, project_root=PROJECT_ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
