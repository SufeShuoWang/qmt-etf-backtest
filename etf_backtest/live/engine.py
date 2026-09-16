"""在主线程中持有账户锁、调度器和券商会话。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from sqlalchemy.engine import Connection, Engine

from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_account_lock,
    release_account_lock,
)
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.live.state import AccountStatus, JobTriggerSource

SHANGHAI = ZoneInfo("Asia/Shanghai")
LOGGER = logging.getLogger(__name__)
ACCOUNT_LOCK_HEARTBEAT_SECONDS = 300.0


if TYPE_CHECKING:
    from etf_backtest.live.broker.callbacks import BrokerEventConsumer


# 管理模拟盘进程生命周期、账户独占锁、券商健康状态和日作业调度。
class LiveTradingEngine:
    # 绑定配置、状态库、券商、日作业与调度器，准备生命周期状态。
    def __init__(
        self,
        *,
        config: LiveConfig,
        state_engine: Engine,
        repository: LiveStateRepository,
        broker: BrokerGateway,
        quote_provider: QuoteProvider,
        jobs: LiveDailyJobs,
        scheduler: LiveScheduler,
        event_consumer: BrokerEventConsumer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.state_engine = state_engine
        self.repository = repository
        self.broker = broker
        self.quote_provider = quote_provider
        self.jobs = jobs
        self.scheduler = scheduler
        self.event_consumer = event_consumer
        self.clock = clock or (lambda: datetime.now(SHANGHAI))
        self._connection: Connection | None = None
        self._lock_acquired = False
        self._broker_connected = False
        self._scheduler_started = False
        self._consumer_started = False
        self._broker_unhealthy = Event()
        self._broker_unhealthy_reason: str | None = None
        self._shutdown = Event()
        self.scheduler.set_broker_job_runner(self._run_broker_job)
        self.jobs.set_broker_health_check(self._require_broker_healthy)

    # 挂接券商事件消费者，使主循环能统一启动与停止回调处理。
    def set_event_consumer(self, event_consumer: BrokerEventConsumer) -> None:
        if self._connection is not None:
            raise RuntimeError("cannot replace the event consumer while engine is running")
        self.event_consumer = event_consumer

    def notify_broker_unhealthy(self, reason: str) -> None:
        """线程安全的回调目标；实际恢复仍在引擎线程执行。"""

        if not self._broker_connected:
            return
        if self._broker_unhealthy_reason is None:
            self._broker_unhealthy_reason = reason
        self._broker_unhealthy.set()

    # 获取账户独占锁并启动事件消费和调度，避免同一账户被多个进程重复执行。
    def start(self, trade_date: date) -> None:
        if self._connection is not None:
            raise RuntimeError("live engine is already started")
        self._shutdown.clear()
        self._broker_unhealthy.clear()
        self._broker_unhealthy_reason = None
        account_id = self.config.account.account_id()
        self._connection = self.state_engine.connect()
        try:
            if not acquire_account_lock(self._connection, account_id):
                raise RuntimeError("account lock is already held")
            self._lock_acquired = True
            if self.event_consumer is not None:
                self.event_consumer.start()
                self._consumer_started = True
            del trade_date
            self.scheduler.start()
            self._scheduler_started = True
        except Exception:
            self.stop()
            raise

    # 模拟盘常驻循环：检查账户锁与券商健康状态，推进调度器，具体券商作业在独立连接生命周期中运行。
    def run_forever(self) -> None:
        """在当前调用线程中负责调度 tick 和全部重连工作。"""

        self.start(self.clock().astimezone(SHANGHAI).date())
        next_lock_heartbeat = monotonic() + ACCOUNT_LOCK_HEARTBEAT_SECONDS
        try:
            while not self._shutdown.wait(1.0):
                current = monotonic()
                if current >= next_lock_heartbeat:
                    self._heartbeat_account_lock()
                    next_lock_heartbeat = current + ACCOUNT_LOCK_HEARTBEAT_SECONDS
                if self._scheduler_started:
                    self.scheduler.tick()
        finally:
            self.stop()

    # 停止调度与事件消费，断开券商并释放账户锁等资源。
    def stop(self) -> None:
        self._shutdown.set()
        connection = self._connection
        try:
            self._stop_scheduler()
        finally:
            try:
                self._disconnect_broker()
            finally:
                try:
                    if self._consumer_started and self.event_consumer is not None:
                        self.event_consumer.stop()
                        self._consumer_started = False
                finally:
                    try:
                        if connection is not None and self._lock_acquired:
                            release_account_lock(connection, self.config.account.account_id())
                            self._lock_acquired = False
                    finally:
                        if connection is not None:
                            connection.close()
                            self._connection = None

    # 连接并订阅券商后执行启动对账与目标业务，finally 中断连；调度等待期间不维持该作业的 Trader 会话。
    def _run_broker_job(
        self,
        job_name: str,
        trade_date: date,
        trigger_source: JobTriggerSource,
    ) -> object:
        """在全新的 Trader 生命周期中运行单个依赖券商的业务任务。"""

        assert self._connection is not None
        account_id = self.config.account.account_id()
        # 连接前先标记生命周期已开始。MiniQMT 适配器会在连接失败时停止部分创建的
        # Trader；下方无条件断开还能确保网关状态与实际情况一致。
        self._broker_connected = True
        self._broker_unhealthy.clear()
        self._broker_unhealthy_reason = None
        try:
            self.broker.connect()
            self.broker.subscribe_account(account_id)
            self._require_broker_healthy()
            account = self.jobs.startup_reconcile(trade_date, lock_connection=self._connection)
            self._require_broker_healthy()
            status = account["status"]
            if status not in {AccountStatus.ACTIVE, AccountStatus.ACTIVE.value}:
                raise RuntimeError("startup did not produce an ACTIVE account")
            if job_name == "rebalance":
                self.quote_provider.subscribe(self.jobs.all_symbols)
                run = self.jobs.execute_pending_target
            elif job_name == "eod":
                run = self.jobs.eod
            else:
                raise ValueError(f"unsupported broker job: {job_name}")
            result = run(
                trade_date, trigger_source=trigger_source, lock_connection=self._connection,
            )
            self._require_broker_healthy()
            return result
        finally:
            self._disconnect_broker()
            self._broker_unhealthy.clear()
            self._broker_unhealthy_reason = None

    # 停止调度器，同时更新引擎调度状态。
    def _stop_scheduler(self) -> None:
        if self._scheduler_started:
            self.scheduler.stop()
            self._scheduler_started = False

    # 定期检查并保持 MySQL 账户锁连接有效；连接失效时停止服务，避免丢锁后继续交易。
    def _heartbeat_account_lock(self) -> None:
        """Keep the MySQL advisory-lock session alive or stop the service."""

        connection = self._connection
        if connection is None or not self._lock_acquired:
            raise RuntimeError("account lock connection is not active")
        try:
            connection.exec_driver_sql("SELECT 1").scalar_one()
        except Exception as exc:
            # Lock ownership is now uncertain. Let stop() close the connection
            # without issuing RELEASE_LOCK on a broken session.
            self._lock_acquired = False
            raise RuntimeError(
                "account lock connection heartbeat failed; stopping live service"
            ) from exc

    # 在作业或委托前检查券商健康标记，异常时阻止继续交易。
    def _require_broker_healthy(self) -> None:
        if not self._broker_unhealthy.is_set():
            return
        reason = self._broker_unhealthy_reason or "BROKER_CALLBACK_UNHEALTHY"
        self.repository.pause_account(self.config.account.account_id(), reason)
        raise RuntimeError(f"MiniQMT session became unhealthy: {reason}")

    # 统一断开券商连接，供作业结束和引擎停止时清理资源。
    def _disconnect_broker(self) -> None:
        if self._broker_connected:
            try:
                self.broker.disconnect()
            finally:
                self._broker_connected = False


__all__ = ["LiveTradingEngine"]
