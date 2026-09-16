"""模拟盘入口：按现有配置每天自动运行，Ctrl+C 停止。"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path


# 模拟盘命令行入口：读取并校验配置，装配运行时后启动常驻引擎，处理中断与启动错误。
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 Rule/Model 模拟盘每日自动运行。")
    parser.add_argument("--config", type=Path, required=True, help="现有模拟盘 YAML 配置")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from etf_backtest.live.config import load_live_config
        from etf_backtest.live.service import build_production_runtime

        engine = build_production_runtime(load_live_config(args.config))
        engine.run_forever()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
