"""由配置驱动、负责三类日频 PAPER 业务任务的调度器。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import JobAlreadySucceeded, JobSkipped, LiveDailyJobs
from etf_backtest.live.state import JobTriggerSource

SHANGHAI = ZoneInfo("Asia/Shanghai")
LOGGER = logging.getLogger(__name__)


# 规定日作业调度需要的交易日判断接口。
class TradingDaySource(Protocol):
    # 判断日期是否为可执行作业的交易日，由具体日历来源实现。
    def is_trading_day(self, trade_date: date) -> bool: ...


BrokerJobRunner = Callable[[str, date, JobTriggerSource], object]


class LiveScheduler:
    """每个进程运行各业务任务一次，并通过 MySQL 持久化状态去重。"""

    # 保存交易日来源、作业函数、时间表与时钟，初始化当日触发记录。
    def __init__(
        self,
        *,
        jobs: LiveDailyJobs,
        calendar: TradingDaySource,
        config: LiveConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.jobs = jobs
        self.calendar = calendar
        self.config = config
        self.clock = clock or (lambda: datetime.now(SHANGHAI))
        self._running = False
        self._handled: set[tuple[date, str]] = set()
        self._missed: list[tuple[date, str]] = []
        self._broker_job_runner: BrokerJobRunner | None = None

    def set_broker_job_runner(self, runner: BrokerJobRunner) -> None:
        """将依赖券商的任务委托给引擎持有的会话边界。"""

        if self._running:
            raise RuntimeError("cannot replace broker job runner while scheduler is running")
        self._broker_job_runner = runner

    # 读取调度器是否处于运行状态。
    @property
    def running(self) -> bool:
        return self._running

    # 返回已记录的错过窗口作业，便于观察调度结果。
    @property
    def missed_jobs(self) -> tuple[tuple[date, str], ...]:
        return tuple(self._missed)

    # 启动调度状态，使后续 tick() 可以检查并触发作业。
    def start(self) -> None:
        if self._broker_job_runner is None:
            raise RuntimeError("scheduler must be bound to LiveTradingEngine before starting")
        # 重连时有意重新评估当日任务；持久化任务记录才是权威去重依据，本地缓存不是。
        self._handled.clear()
        self._running = True

    # 停止调度状态，阻止后续 tick() 发起新作业。
    def stop(self) -> None:
        self._running = False

    # 检查当前市场日期和时间，按尾盘执行、收盘对账、信号顺序触发到期作业，并处理去重与错过窗口。
    def tick(self, now: datetime | None = None) -> None:
        current = (now or self.clock()).astimezone(SHANGHAI)
        trade_date = current.date()
        local_time = current.time().replace(tzinfo=None)
        if not self.calendar.is_trading_day(trade_date):
            return
        # 执行顺序保持固定；配置只决定每项任务何时到期。
        schedule = (
            (self.config.execution.submit_start, "rebalance"),
            (self.config.eod.run_time, "eod"),
            (self.config.signal.run_time, "prepare_signal"),
        )
        for job_time, name in schedule:
            key = (trade_date, name)
            if key in self._handled or local_time < job_time:
                continue
            self._handled.add(key)
            try:
                if self.jobs.has_job_completed(name, trade_date):
                    continue
                if (
                    name == "rebalance"
                    and local_time >= self.config.execution.stop_new_orders
                    and not self.jobs.has_rebalance_activity(trade_date)
                ):
                    self._missed.append(key)
                    self.jobs.record_job_skipped("rebalance", trade_date, "MISSED_ORDER_WINDOW")
                    continue
                if name == "prepare_signal":
                    self.jobs.prepare_signal(trade_date, trigger_source=JobTriggerSource.SCHEDULED)
                else:
                    if self._broker_job_runner is None:
                        raise RuntimeError("broker jobs require LiveTradingEngine")
                    self._broker_job_runner(name, trade_date, JobTriggerSource.SCHEDULED)
            except (JobAlreadySucceeded, JobSkipped):
                # 两种结果都已持久化，且都是补执行期间的正常状态。
                continue
            except Exception:
                # 避免每秒重试；重连或重启会根据持久化状态重新进入。
                LOGGER.exception("scheduled live job failed: %s", name)


__all__ = ["BrokerJobRunner", "LiveScheduler", "TradingDaySource"]
