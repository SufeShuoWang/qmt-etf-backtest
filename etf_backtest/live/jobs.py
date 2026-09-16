"""单个策略使用独立资金与持仓账本的日频交易任务。"""

from __future__ import annotations

import hashlib
import time as time_module
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import TypeVar, cast

from sqlalchemy.engine import Connection, Engine

from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE, FeeConfig
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.order import OrderSide
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.execution.near_close_limit import NearCloseLimitPolicy
from etf_backtest.live.execution.planner import (
    LiveRebalancePlanner,
)
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_job_lock,
    release_job_lock,
)
from etf_backtest.live.reconciliation import (
    ReconciliationService,
    is_local_remark_token,
)
from etf_backtest.live.risk import LiveRiskManager
from etf_backtest.live.signals import StrategyRuntime
from etf_backtest.live.state import (
    AccountStatus,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    JobTriggerSource,
    LiveQuote,
    OrderIntent,
    OrderIntentStatus,
    QueryResult,
    ReconciliationReport,
    SubmitOrderStatus,
)

ResultT = TypeVar("ResultT")


class JobAlreadySucceeded(RuntimeError):
    """某个已持久化日频任务已经进入终态。"""


class JobSkipped(RuntimeError):
    """有意执行的安全跳过，不应按失败重试。"""


class AccountSafetyError(RuntimeError):
    """券商事实或账户账本不确定，必须立即中止当前交易任务。"""


# 把数据库状态字段还原为指定枚举，兼容已是枚举的值。
def _status[EnumT: Enum](value: object, enum_type: type[EnumT]) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(cast(str, value))


# 实现模拟盘每日闭环：启动对账、生成并保存信号、分阶段执行、撤单确认及收盘快照。
class LiveDailyJobs:
    # 组装券商、行情、策略运行时、规划器、风控、对账与状态仓库等作业依赖。
    def __init__(
        self,
        *,
        config: LiveConfig,
        broker: BrokerGateway,
        quote_provider: QuoteProvider,
        state_repository: LiveStateRepository,
        state_engine: Engine,
        strategy_runtime: StrategyRuntime,
        planner: LiveRebalancePlanner | None = None,
        risk_manager: LiveRiskManager | None = None,
        price_policy: NearCloseLimitPolicy | None = None,
        reconciliation_service: ReconciliationService | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        broker_health_check: Callable[[], None] | None = None,
        fee_model: FeeModel | None = None,
    ) -> None:
        if strategy_runtime.spec.strategy_id != config.strategy.strategy_id:
            raise ValueError("StrategyRuntime must match configured strategy_id")
        self.config, self.broker, self.quote_provider = config, broker, quote_provider
        self.repository, self.state_engine = state_repository, state_engine
        self.strategy_runtime = strategy_runtime
        self.fee_model = fee_model or FeeModel(FeeConfig())
        self.planner, self.risk = (
            planner or LiveRebalancePlanner(self.fee_model),
            risk_manager or LiveRiskManager(self.fee_model),
        )
        self.price_policy = price_policy or NearCloseLimitPolicy()
        self.reconciliation = reconciliation_service or ReconciliationService(
            strategy_runtime.turnover_rules
        )
        self.clock, self.sleep = (
            clock or (lambda: datetime.now(MARKET_TIMEZONE)),
            sleep or time_module.sleep,
        )
        self._broker_health_check = broker_health_check or (lambda: None)

    # 注册券商健康检查，供作业推进和委托前调用。
    def set_broker_health_check(self, check: Callable[[], None]) -> None:
        self._broker_health_check = check

    # 执行已注册的券商健康检查，阻止在连接异常时继续作业。
    def _require_broker_healthy(self) -> None:
        try:
            self._broker_health_check()
        except AccountSafetyError:
            raise
        except Exception as error:
            account_id = self.config.account.account_id()
            account = self.repository.get_account(account_id)
            if (
                account is not None
                and _status(account["status"], AccountStatus) is AccountStatus.ACTIVE
            ):
                self.repository.pause_account(account_id, "BROKER_CALLBACK_UNHEALTHY")
            raise AccountSafetyError(str(error)) from error

    # 统一取得券商查询记录，查询失败或异常时抛错而非当作空结果。
    def _broker_records(
        self, query: Callable[[], QueryResult[ResultT]], label: str
    ) -> tuple[ResultT, ...]:
        self._require_broker_healthy()
        try:
            result = query()
        except Exception as error:
            account_id = self.config.account.account_id()
            self.repository.pause_account(account_id, f"BROKER_{label.upper()}_QUERY_FAILED")
            raise AccountSafetyError(f"broker {label} query raised an exception") from error
        self._require_broker_healthy()
        if not result.success:
            account_id = self.config.account.account_id()
            self.repository.pause_account(account_id, f"BROKER_{label.upper()}_QUERY_FAILED")
            raise AccountSafetyError(f"broker {label} query failed: {result.error}")
        return result.records

    # 返回配置策略涉及的完整证券集合，供批量行情查询。
    @property
    def all_symbols(self) -> tuple[str, ...]:
        return self.strategy_runtime.spec.symbols

    # 同步配置账户与单策略虚拟账本，保留已存在的运行资金和持仓。
    def _sync_configured_account(self) -> Mapping[str, object]:
        strategy = self.config.strategy
        model = strategy.model
        return self.repository.sync_account(
            account_id=self.config.account.account_id(),
            mode=self.config.account.mode,
            account_type=self.config.account.account_type,
            strategy={
                "strategy_id": strategy.strategy_id,
                "case": strategy.case,
                "initial_capital": strategy.initial_capital,
                "experiment_path": str(strategy.experiment_path),
                "model_backend": None if model is None else model.backend,
                "bundle_path": None if model is None else str(model.bundle_path),
                "model_id": self.strategy_runtime.spec.model_id,
            },
        )

    # 在 MySQL 作业锁内执行任务，检查重复完成状态并记录成功、跳过或失败，最后释放锁。
    def _run_job(
        self,
        job_type: str,
        trade_date: date,
        trigger_source: JobTriggerSource,
        body: Callable[[], ResultT],
        lock_connection: Connection | None,
        *,
        deduplicate: bool = True,
    ) -> ResultT:
        manager = (
            self.state_engine.connect()
            if lock_connection is None
            else nullcontext(lock_connection)
        )
        account_id = self.config.account.account_id()
        # 先同步账户与策略账本，再写任务记录，满足现有外键关系；这里不建表。
        self._sync_configured_account()
        with manager as lock:
            if not acquire_job_lock(lock, account_id, job_type, trade_date):
                error = RuntimeError(f"job lock is already held: {job_type}")
                run_id = self.repository.start_job_run(
                    account_id=account_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                )
                self.repository.finish_job_run(run_id, error=error)
                raise error
            run_id = ""
            try:
                if deduplicate and self.repository.has_terminal_job(
                    account_id, job_type, trade_date
                ):
                    raise JobAlreadySucceeded(
                        f"job already completed: {job_type} {trade_date.isoformat()}"
                    )
                run_id = self.repository.start_job_run(
                    account_id=account_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                )
                result = body()
            except JobAlreadySucceeded:
                raise
            except JobSkipped as error:
                self.repository.finish_job_run(run_id, skip_reason=str(error))
                raise
            except Exception as error:
                self.repository.finish_job_run(run_id, error=error)
                raise
            else:
                self.repository.finish_job_run(run_id)
                return result
            finally:
                release_job_lock(lock, account_id, job_type, trade_date)

    # 查询指定日期的作业是否已经完成，供调度器去重。
    def has_job_completed(self, job_type: str, trade_date: date) -> bool:
        return self.repository.has_terminal_job(
            self.config.account.account_id(), job_type, trade_date
        )

    def _run_strategy_step(
        self,
        job_type: str,
        trade_date: date,
        strategy_id: str,
        trigger_source: JobTriggerSource,
        body: Callable[[], ResultT],
    ) -> ResultT | None:
        """在父 Job 锁内执行可恢复的策略步骤，不另取 MySQL advisory lock。"""

        account_id = self.config.account.account_id()
        if self.repository.has_terminal_job(
            account_id,
            job_type,
            trade_date,
            strategy_id=strategy_id,
        ):
            return None
        run_id = self.repository.start_job_run(
            account_id=account_id,
            strategy_id=strategy_id,
            job_type=job_type,
            trade_date=trade_date,
            trigger_source=trigger_source,
        )
        try:
            result = body()
        except JobSkipped as error:
            self.repository.finish_job_run(run_id, skip_reason=str(error))
            return None
        except Exception as error:
            self.repository.finish_job_run(run_id, error=error)
            raise
        self.repository.finish_job_run(run_id)
        return result

    # 将错过执行窗口等情况记录为跳过作业，避免后续重复触发。
    def record_job_skipped(self, job_type: str, trade_date: date, reason: str) -> None:
        # 用 JobSkipped 表达跳过原因，由统一作业包装器持久化。
        def skip() -> None:
            raise JobSkipped(reason)

        try:
            self._run_job(job_type, trade_date, JobTriggerSource.RECOVERY, skip, None)
        except (JobAlreadySucceeded, JobSkipped):
            return

    # 检查指定执行日是否已有意图或冻结目标，帮助区分尚未执行与中断恢复。
    def has_rebalance_activity(self, execution_date: date) -> bool:
        decision = self.repository.pending_decision_for_date(
            self.config.account.account_id(), execution_date
        )
        return decision is not None and bool(
            self.repository.list_order_intents_for_decision(str(decision["decision_id"]))
            or self.repository.load_execution_targets(str(decision["decision_id"])) is not None
        )

    # 通过统一作业包装执行启动对账。
    def startup_reconcile(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.RECOVERY,
        lock_connection: Connection | None = None,
    ) -> Mapping[str, object]:
        return self._run_job(
            "startup_reconcile",
            trade_date,
            trigger_source,
            lambda: self._startup_reconcile(trade_date),
            lock_connection,
            deduplicate=False,
        )

    # 同步虚拟账户并核对券商订单与成交，恢复已知状态或暂停异常账户。
    def _startup_reconcile(self, trade_date: date) -> Mapping[str, object]:
        self._require_current_trade_date(trade_date)
        account_id = self.config.account.account_id()
        configured = self.repository.get_account(account_id)
        if configured is None:
            raise RuntimeError("account was not synchronized before startup reconciliation")
        ensured = configured
        self._reconcile_or_pause(
            configured,
            self._broker_records(self.broker.query_orders, "orders"),
            self._broker_records(self.broker.query_trades, "trades"),
        )
        if (
            configured is not None
            and _status(configured["status"], AccountStatus) is AccountStatus.PAUSED
        ):
            self.repository.resume_account(account_id)
            ensured = self.repository.get_account(account_id) or ensured
        return ensured

    # 通过日作业包装生成指定信号日的策略决策。
    def prepare_signal(
        self,
        signal_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> DailyDecisionResult | None:
        return self._run_job(
            "prepare_signal",
            signal_date,
            trigger_source,
            lambda: self._prepare_signal(signal_date, trigger_source=trigger_source),
            lock_connection,
        )

    # 组织策略信号准备，检查账户状态并推进各策略步骤。
    def _prepare_signal(
        self,
        signal_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
    ) -> DailyDecisionResult | None:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        if signal_date < now.date():
            raise JobSkipped("STALE_SIGNAL_DATE")
        if (
            signal_date > now.date()
            or now.time().replace(tzinfo=None) < self.config.signal.run_time
        ):
            raise JobSkipped("SIGNAL_TIME_NOT_REACHED")
        physical_account = self._active_account()
        self._require_no_unresolved()
        runtime = self.strategy_runtime
        return self._run_strategy_step(
            "strategy_signal", signal_date, runtime.spec.strategy_id, trigger_source,
            lambda: self._prepare_strategy_signal(
                signal_date, now, str(physical_account["account_id"]),
                runtime.spec.strategy_id, runtime,
            ),
        )

    # 结算持仓可卖状态，按虚拟现金与持仓计算信号，并在事务中保存决策及显式目标。
    def _prepare_strategy_signal(
        self,
        signal_date: date,
        now: datetime,
        account_id: str,
        strategy_id: str,
        runtime: StrategyRuntime,
    ) -> DailyDecisionResult:
        self.repository.settle_strategy_positions(
            account_id,
            strategy_id,
            signal_date,
            turnover_rules=runtime.turnover_rules,
        )
        strategy_account = self.repository.get_strategy(account_id, strategy_id)
        if strategy_account is None:
            raise RuntimeError(f"virtual strategy account is missing: {strategy_id}")
        positions = self.repository.load_strategy_positions(
            account_id,
            strategy_id,
            turnover_rules=runtime.turnover_rules,
            captured_at=now,
        )
        result = runtime.signal_evaluator.evaluate(
            signal_date=signal_date,
            symbols=runtime.spec.symbols,
            virtual_cash=Decimal(strategy_account["virtual_cash"]),
            positions=positions,
            turnover_rules=runtime.turnover_rules,
        )
        with self.repository.transaction() as connection:
            saved = self.repository.create_or_get_decision(
                decision_id=self._decision_id(strategy_id, signal_date),
                account_id=account_id,
                strategy_id=strategy_id,
                signal_date=result.signal_date,
                execution_date=result.execution_date,
                schedule_index=result.schedule_index,
                status=result.status,
                data_as_of=signal_date,
                model_id=runtime.spec.model_id,
                connection=connection,
            )
            if result.status is DecisionStatus.TARGET_CREATED:
                assert result.target_portfolio is not None
                self.repository.save_target_positions(
                    str(saved["decision_id"]),
                    result.target_portfolio.weights,
                    connection=connection,
                )
        return result

    # 通过作业包装执行指定交易日已保存的目标，供尾盘调度调用。
    def execute_pending_target(
        self,
        execution_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        return self._run_job(
            "rebalance",
            execution_date,
            trigger_source,
            lambda: self._execute_pending_target(execution_date, trigger_source=trigger_source),
            lock_connection,
        )

    # 检查日期、窗口和账户对账，先执行并等待卖出阶段，再刷新行情执行买入阶段。
    def _execute_pending_target(
        self,
        execution_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
    ) -> tuple[Mapping[str, object], ...]:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        if execution_date < now.date():
            raise JobSkipped("STALE_EXECUTION_DATE")
        if execution_date > now.date():
            raise JobSkipped("FUTURE_EXECUTION_DATE")
        account = self._active_account()
        decision = self.repository.pending_decision_for_date(
            str(account["account_id"]), execution_date
        )
        if decision is None:
            return ()
        if decision["strategy_id"] != self.config.strategy.strategy_id:
            raise AccountSafetyError("pending decision does not match the single strategy")
        local_time = now.time().replace(tzinfo=None)
        if local_time < self.config.execution.submit_start:
            raise JobSkipped("REBALANCE_TIME_NOT_REACHED")
        has_started = bool(
            self.repository.list_order_intents_for_decision(str(decision["decision_id"]))
            or self.repository.load_execution_targets(str(decision["decision_id"])) is not None
        )
        if local_time >= self.config.execution.stop_new_orders and not has_started:
            raise JobSkipped("MISSED_ORDER_WINDOW")
        self._reconcile_or_pause(
            account,
            self._broker_records(self.broker.query_orders, "orders"),
            self._broker_records(self.broker.query_trades, "trades"),
        )
        self._require_no_unresolved()
        quotes, quote_error = self._latest_quotes_or_error()

        self._execute_phase(
            decision=decision, execution_date=execution_date, account=account,
            side=OrderSide.SELL, quotes=quotes, quote_error=quote_error,
            trigger_source=trigger_source,
        )
        buy_now = self.clock().astimezone(MARKET_TIMEZONE)
        buy_quotes, buy_quote_error = self._latest_quotes_or_error()
        self._execute_phase(
            decision=decision, execution_date=execution_date, account=account,
            side=OrderSide.BUY, quotes=buy_quotes, quote_error=buy_quote_error,
            trigger_source=trigger_source,
            skip_reason=("MISSED_BUY_WINDOW"
                         if buy_now.time().replace(tzinfo=None) >= self.config.execution.stop_new_orders
                         else None),
        )
        return self.repository.list_order_intents_for_decision(str(decision["decision_id"]))

    # 把单方向执行封装为可记录的策略步骤，供卖出和买入阶段共用。
    def _execute_phase(
        self, *, decision: Mapping[str, object], execution_date: date,
        account: Mapping[str, object], side: OrderSide, quotes: Mapping[str, LiveQuote],
        quote_error: Exception | None, trigger_source: JobTriggerSource,
        skip_reason: str | None = None,
    ) -> None:
        # 在策略阶段内调用该方向的委托规划与提交，传递本次行情或查询错误。
        def execute() -> None:
            if skip_reason is not None:
                raise JobSkipped(skip_reason)
            self._execute_strategy_side(
                decision=decision, runtime=self.strategy_runtime, execution_date=execution_date,
                side=side, quotes=quotes, quote_error=quote_error,
            )
            self._wait_for_phase(
                execution_date, account, side=side,
                deadline_time=(self.config.execution.sell_phase_deadline
                               if side is OrderSide.SELL else self.config.execution.cancel_open_orders),
            )
            if side is OrderSide.SELL:
                self._reconcile_or_pause(
                    account,
                    self._broker_records(self.broker.query_orders, "orders"),
                    self._broker_records(self.broker.query_trades, "trades"),
                )

        self._run_strategy_step(
            "strategy_sell" if side is OrderSide.SELL else "strategy_buy",
            execution_date, self.config.strategy.strategy_id, trigger_source, execute,
        )

    # 恢复已有意图或按虚拟账户规划新单；卖出阶段首次冻结目标数量，扣除委托预留后风控、落库并提交。
    def _execute_strategy_side(
        self,
        *,
        decision: Mapping[str, object],
        runtime: StrategyRuntime,
        execution_date: date,
        side: OrderSide,
        quotes: Mapping[str, LiveQuote],
        quote_error: Exception | None,
    ) -> tuple[Mapping[str, object], ...]:
        if quote_error is not None:
            raise quote_error
        account_id = self.config.account.account_id()
        strategy_id = runtime.spec.strategy_id
        decision_id = str(decision["decision_id"])
        target = TargetPortfolio(self.repository.load_target_positions(decision_id))
        existing = tuple(
            row
            for row in self.repository.list_order_intents_for_decision(decision_id)
            if _status(row["side"], OrderSide) is side
        )
        if existing:
            submit_allowed = not (
                side is OrderSide.SELL
                and self.clock().astimezone(MARKET_TIMEZONE).time().replace(tzinfo=None)
                >= self.config.execution.sell_phase_deadline
            )
            self._resume_persisted_intents(
                rows=existing,
                target=target,
                execution_date=execution_date,
                submit_allowed=submit_allowed,
                closed_reason="MISSED_SELL_WINDOW",
            )
            return existing

        now = self.clock().astimezone(MARKET_TIMEZONE)
        valuation, limits = self._prices_for_strategy(runtime, quotes, side, now)
        self.repository.settle_strategy_positions(
            account_id,
            strategy_id,
            execution_date,
            turnover_rules=runtime.turnover_rules,
        )
        strategy_account = self.repository.get_strategy(account_id, strategy_id)
        if strategy_account is None:
            raise RuntimeError(f"virtual strategy account is missing: {strategy_id}")
        positions = self.repository.load_strategy_positions(
            account_id,
            strategy_id,
            turnover_rules=runtime.turnover_rules,
            captured_at=now,
        )
        position_map = {
            symbol: next((row for row in positions if row.symbol == symbol), None)
            or BrokerPositionSnapshot(
                symbol=symbol,
                total_quantity=0,
                available_quantity=0,
                today_buy_quantity=0,
                market_value=Decimal("0"),
                turnover_rule=runtime.turnover_rules[symbol],
                captured_at=now,
            )
            for symbol in runtime.spec.symbols
        }
        total_asset = Decimal(strategy_account["virtual_cash"]) + sum(
            valuation[symbol] * position_map[symbol].total_quantity
            for symbol in runtime.spec.symbols
        )
        frozen = self.repository.load_execution_targets(decision_id)
        if frozen is None:
            if side is OrderSide.BUY:
                raise RuntimeError("buy phase cannot start before execution targets are frozen")
            frozen = self.repository.freeze_execution_targets(
                decision_id,
                total_asset=total_asset,
                valuation_prices=valuation,
                lot_size=self.config.execution.lot_size,
            )
        frozen_prices = {symbol: value[0] for symbol, value in frozen.items()}
        target_quantities = {symbol: value[1] for symbol, value in frozen.items()}
        reserved_cash, reserved_sells, active_rows = self.repository.strategy_reservations(
            account_id, strategy_id
        )
        reserved_cash += sum(
            self.fee_model.calculate(
                trade_amount=Decimal(row["limit_price"]) * int(row["remaining_quantity"]),
                side=OrderSide.BUY,
            ).total
            for row in active_rows
            if _status(row["side"], OrderSide) is OrderSide.BUY
            and int(row["remaining_quantity"]) > 0
        )
        available_cash = Decimal(strategy_account["virtual_cash"]) - reserved_cash
        if available_cash < 0:
            self.repository.pause_account(account_id, "VIRTUAL_CASH_NEGATIVE_AFTER_FEES")
            raise AccountSafetyError(
                f"strategy BUY reservations exceed virtual cash: {strategy_id}"
            )
        for symbol, reserved in reserved_sells.items():
            row = position_map[symbol]
            if reserved > row.available_quantity:
                self.repository.pause_account(account_id, "SELL_RESERVATION_EXCEEDS_POSITION")
                raise AccountSafetyError(
                    f"strategy SELL reservations exceed available quantity: {strategy_id}/{symbol}"
                )
            position_map[symbol] = BrokerPositionSnapshot(
                symbol=symbol,
                total_quantity=row.total_quantity,
                available_quantity=row.available_quantity - reserved,
                today_buy_quantity=row.today_buy_quantity,
                market_value=valuation[symbol] * row.total_quantity,
                turnover_rule=row.turnover_rule,
                captured_at=now,
                average_cost=row.average_cost,
            )
        active_orders = tuple(
            BrokerOrderSnapshot(
                broker_order_id=f"intent:{row['intent_id']}",
                symbol=str(row["symbol"]),
                side=_status(row["side"], OrderSide),
                requested_quantity=int(row["remaining_quantity"]),
                filled_quantity=0,
                limit_price=Decimal(row["limit_price"]),
                status=BrokerOrderStatus.PENDING,
                captured_at=now,
                remark_token=str(row["remark_token"]),
            )
            for row in active_rows
            if int(row["remaining_quantity"]) > 0
        )
        intents = self.planner.plan(
            account_id=account_id,
            strategy_id=strategy_id,
            decision_id=decision_id,
            execution_date=execution_date,
            symbols=runtime.spec.symbols,
            target=target,
            total_asset=total_asset,
            available_cash=available_cash,
            positions=position_map,
            active_orders=active_orders,
            valuation_prices=frozen_prices,
            limit_prices=limits,
            lot_size=self.config.execution.lot_size,
            target_quantities=target_quantities,
            side=side,
        )
        try:
            effective_weights = self._projected_weights(
                symbols=runtime.spec.symbols,
                positions=position_map,
                active_orders=active_orders,
                intents=intents,
                valuation_prices=valuation,
                total_asset=total_asset,
            )
        except AccountSafetyError:
            self.repository.pause_account(account_id, "PROJECTED_POSITION_NEGATIVE")
            raise
        daily_notional = self.repository.strategy_order_notional_for_date(
            account_id, strategy_id, execution_date
        )
        approved: set[str] = set()
        rejected: dict[str, str] = {}
        sell_window_closed = (
            side is OrderSide.SELL
            and now.time().replace(tzinfo=None) >= self.config.execution.sell_phase_deadline
        )
        for intent in intents:
            if sell_window_closed:
                rejected[intent.intent_key] = "MISSED_SELL_WINDOW"
                continue
            available = position_map[intent.symbol]
            risk = self.risk.check(
                intent,
                symbols=runtime.spec.symbols,
                effective_target_weights=effective_weights,
                max_total_target_weight=self.config.risk.max_total_target_weight,
                available_cash=available_cash,
                available_quantity=available.available_quantity,
                lot_size=self.config.execution.lot_size,
                max_single_order_notional=self.config.risk.max_single_order_notional,
                max_daily_order_notional=self.config.risk.max_daily_order_notional,
                daily_planned_notional=daily_notional,
                min_order_notional=self.config.risk.min_order_notional,
                quote_valid=True,
            )
            if risk.approved:
                approved.add(intent.intent_key)
                daily_notional += intent.requested_quantity * intent.limit_price
            else:
                rejected[intent.intent_key] = risk.reason or "RISK_REJECTED"
        saved: list[Mapping[str, object]] = []
        with self.repository.transaction() as connection:
            for intent in intents:
                saved_row = self.repository.create_order_intent(intent, connection=connection)
                saved.append(saved_row)
                reason = rejected.get(intent.intent_key)
                if reason is not None:
                    self.repository.mark_intent_rejected(
                        str(saved_row["intent_id"]), reason, connection=connection
                    )
        for intent, persisted_row in zip(intents, saved, strict=True):
            if (
                intent.intent_key in approved
                and _status(persisted_row["status"], OrderIntentStatus) is OrderIntentStatus.PLANNED
            ):
                self._submit_intent(intent, str(persisted_row["intent_id"]), account_id)
        return tuple(saved)

    # 逐证券计算用于当前方向的估值价和限价，拒绝缺失或失效行情。
    def _prices_for_strategy(
        self,
        runtime: StrategyRuntime,
        quotes: Mapping[str, LiveQuote],
        side: OrderSide,
        now: datetime,
    ) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        valuation: dict[str, Decimal] = {}
        limits: dict[str, Decimal] = {}
        for symbol in runtime.spec.symbols:
            quote = quotes.get(symbol)
            if quote is None:
                raise RuntimeError(f"quote is missing: {symbol}")
            result = self.price_policy.calculate(
                side=side,
                quote=quote,
                tick_size=quote.price_tick,
                price_offset_ticks=self.config.execution.price_offset_ticks,
                now=now,
                quote_stale_seconds=self.config.execution.quote_stale_seconds,
            )
            if result is None:
                raise RuntimeError(f"valid near-close price is unavailable: {symbol}")
            valuation[symbol] = result.valuation_price
            limits[symbol] = result.limit_price
        return valuation, limits

    def _latest_quotes_or_error(self) -> tuple[dict[str, LiveQuote], Exception | None]:
        """把行情源失败作为确定性的策略输入错误，而不是券商账户事实错误。"""

        try:
            result = self.quote_provider.latest_quotes(self.all_symbols)
        except Exception as error:
            return {}, RuntimeError(f"quote query raised an exception: {error}")
        if not result.success:
            return {}, RuntimeError(f"quote query failed: {result.error}")
        return {quote.symbol: quote for quote in result.records}, None

    # 把持仓、活动委托和本次意图合并，按当前估值计算风控使用的预计权重。
    @staticmethod
    def _projected_weights(
        *,
        symbols: Sequence[str],
        positions: Mapping[str, BrokerPositionSnapshot],
        active_orders: Sequence[BrokerOrderSnapshot],
        intents: Sequence[OrderIntent],
        valuation_prices: Mapping[str, Decimal],
        total_asset: Decimal,
    ) -> dict[str, Decimal]:
        quantities = {symbol: positions[symbol].total_quantity for symbol in symbols}
        for order in active_orders:
            if order.status.is_active:
                quantities[order.symbol] += (
                    order.remaining_quantity
                    if order.side is OrderSide.BUY
                    else -order.remaining_quantity
                )
        for intent in intents:
            quantities[intent.symbol] += (
                intent.requested_quantity
                if intent.side is OrderSide.BUY
                else -intent.requested_quantity
            )
        if any(quantity < 0 for quantity in quantities.values()):
            raise AccountSafetyError("projected strategy quantity is negative")
        return {
            symbol: Decimal(quantity) * valuation_prices[symbol] / total_asset
            for symbol, quantity in quantities.items()
        }

    # 只恢复尚未提交的 PLANNED 意图；提交结果未知时暂停账户，窗口关闭时拒绝剩余计划。
    def _resume_persisted_intents(
        self,
        *,
        rows: Sequence[Mapping[str, object]],
        target: TargetPortfolio,
        execution_date: date,
        submit_allowed: bool = True,
        closed_reason: str = "ORDER_WINDOW_CLOSED",
    ) -> None:
        account_id = self.config.account.account_id()
        for row in rows:
            status = _status(row["status"], OrderIntentStatus)
            if status in {OrderIntentStatus.SUBMITTING, OrderIntentStatus.SUBMIT_UNKNOWN}:
                self.repository.pause_account(account_id, "SUBMIT_RESULT_UNKNOWN")
                raise AccountSafetyError("persisted order submission result is unknown")
            if status is not OrderIntentStatus.PLANNED:
                continue
            if not submit_allowed:
                self.repository.mark_intent_rejected(str(row["intent_id"]), closed_reason)
                continue
            symbol = str(row["symbol"])
            weight = target.weight_for(symbol)
            if weight is None:
                raise RuntimeError("persisted intent symbol is absent from its target")
            intent = OrderIntent(
                intent_key=str(row["intent_key"]),
                remark_token=str(row["remark_token"]),
                account_id=str(row["account_id"]),
                strategy_id=str(row["strategy_id"]),
                decision_id=str(row["decision_id"]),
                execution_date=execution_date,
                symbol=symbol,
                side=_status(row["side"], OrderSide),
                requested_quantity=int(cast(int | str, row["requested_quantity"])),
                target_weight=weight,
                valuation_price=Decimal(cast(Decimal | int | str, row["valuation_price"])),
                limit_price=Decimal(cast(Decimal | int | str, row["limit_price"])),
            )
            self._submit_intent(intent, str(row["intent_id"]), account_id)

    # 等待当前买卖阶段订单结束，到截止时间仍有活动订单则进入撤单流程。
    def _wait_for_phase(
        self,
        execution_date: date,
        account: Mapping[str, object],
        *,
        side: OrderSide,
        deadline_time: time,
    ) -> None:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        deadline = time_module.monotonic() + max(
            0.0,
            (
                datetime.combine(execution_date, deadline_time, MARKET_TIMEZONE) - now
            ).total_seconds(),
        )
        if not self._wait_for_order_closure(account, side=side, deadline=deadline):
            self._cancel_open_orders(side=side)

    def _wait_for_order_closure(
        self, account: Mapping[str, object], *, side: OrderSide | None, deadline: float,
    ) -> bool:
        """查询并对账直到没有本地活动订单；超时返回 False，由调用方决定下一步。"""
        while True:
            orders = self._broker_records(self.broker.query_orders, "orders")
            self._reconcile_or_pause(
                account,
                orders,
                self._broker_records(self.broker.query_trades, "trades"),
            )
            if not self._associated_active_orders(orders, side=side):
                return True
            remaining = deadline - time_module.monotonic()
            if remaining <= 0:
                return False
            self.sleep(min(float(self.config.miniqmt.reconnect_interval_seconds), remaining))
            self._require_broker_healthy()

    # 先持久化 SUBMITTING 再调用券商；按受理、拒绝或未知结果更新状态，未知时暂停账户。
    def _submit_intent(self, intent: OrderIntent, intent_id: str, account_id: str) -> None:
        self._require_broker_healthy()
        self.repository.mark_intent_submitting(intent_id)
        try:
            result = self.broker.submit_order(intent)
        except Exception as error:
            self._mark_submission_unknown(intent_id, account_id, str(error))
            raise AccountSafetyError(str(error)) from error
        self._require_broker_healthy()
        if result.status is SubmitOrderStatus.ACCEPTED:
            assert result.broker_order_id
            try:
                self.repository.bind_broker_order(
                    account_id=account_id,
                    intent_id=intent_id,
                    remark_token=intent.remark_token,
                    order=BrokerOrderSnapshot(
                        broker_order_id=result.broker_order_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        requested_quantity=intent.requested_quantity,
                        filled_quantity=0,
                        limit_price=intent.limit_price,
                        status=BrokerOrderStatus.PENDING,
                        captured_at=self.clock(),
                        remark_token=intent.remark_token,
                    ),
                )
            except Exception as error:
                self._mark_submission_unknown(intent_id, account_id, str(error))
                raise AccountSafetyError(
                    "broker accepted the order but its local binding failed"
                ) from error
        elif result.status is SubmitOrderStatus.REJECTED:
            self.repository.mark_intent_rejected(intent_id, result.error or "BROKER_REJECTED")
        else:
            self._mark_submission_unknown(
                intent_id, account_id, result.error or "SUBMIT_RESULT_UNKNOWN"
            )
            raise AccountSafetyError(result.error or "SUBMIT_RESULT_UNKNOWN")

    def _mark_submission_unknown(self, intent_id: str, account_id: str, reason: str) -> None:
        """先保存提交不确定状态，再暂停账户；禁止在此重试下单。"""
        self.repository.mark_intent_submit_unknown(intent_id, reason)
        self.repository.pause_account(account_id, "SUBMIT_RESULT_UNKNOWN")

    # 从券商订单中筛选能关联本地意图的活动订单，可限定买卖方向。
    def _associated_active_orders(
        self,
        orders: Sequence[BrokerOrderSnapshot],
        *,
        side: OrderSide | None = None,
    ) -> tuple[BrokerOrderSnapshot, ...]:
        account = self._active_account()
        account_id = self.config.account.account_id()
        associated: list[BrokerOrderSnapshot] = []
        unknown_local: list[str] = []
        for order in orders:
            if not order.status.is_active:
                continue
            saved = self.repository.get_broker_order(account_id, order.broker_order_id)
            intent = self.repository.get_intent(str(saved["intent_id"])) if saved else None
            if intent is None and is_local_remark_token(order.remark_token):
                intent = self.repository.get_intent_by_remark_token(
                    order.remark_token or "", account_id=account_id
                )
            if intent is None:
                if is_local_remark_token(order.remark_token):
                    unknown_local.append(order.broker_order_id)
                continue
            if str(intent["account_id"]) == str(account["account_id"]) and (
                side is None or _status(intent["side"], OrderSide) is side
            ):
                associated.append(order)
        if unknown_local:
            self.repository.pause_account(str(account["account_id"]), "UNKNOWN_LOCAL_ACTIVE_ORDER")
            raise AccountSafetyError(f"unknown local active broker orders: {unknown_local}")
        return tuple(associated)

    # 对本地活动订单发起撤单并等待对账确认；未能确认结束时暂停账户。
    def _cancel_open_orders(self, *, side: OrderSide | None = None) -> None:
        account = self._active_account()
        account_id = self.config.account.account_id()
        associated = self._associated_active_orders(
            self._broker_records(self.broker.query_orders, "orders"),
            side=side,
        )
        for order in associated:
            self._require_broker_healthy()
            try:
                accepted = self.broker.cancel_order(order.broker_order_id)
            except Exception as error:
                self.repository.pause_account(account_id, "CANCEL_RESULT_UNKNOWN")
                raise AccountSafetyError(
                    f"cancel result is unknown: {order.broker_order_id}"
                ) from error
            if not accepted:
                self.repository.pause_account(account_id, "CANCEL_RESULT_UNKNOWN")
                raise AccountSafetyError(
                    f"cancel request was not accepted: {order.broker_order_id}"
                )
            self._require_broker_healthy()
        deadline = time_module.monotonic() + self.config.execution.cancel_confirm_timeout_seconds
        if not self._wait_for_order_closure(account, side=side, deadline=deadline):
            self.repository.pause_account(str(account["account_id"]), "CANCEL_CONFIRM_TIMEOUT")
            raise AccountSafetyError(
                "broker orders remained active after the cancel confirmation timeout"
            )

    # 通过统一作业包装执行收盘对账与虚拟账户快照。
    def eod(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> ReconciliationReport:
        # 顺序执行收盘对账和快照，确保先处理订单成交再记录资产。
        def reconcile_and_snapshot() -> ReconciliationReport:
            # 日终对账通过后才保存虚拟策略快照，避免把未确认的账本写成日终结果。
            report = self._reconcile_eod(trade_date)
            self._snapshot_eod(trade_date, trigger_source=trigger_source)
            return report

        return self._run_job(
            "eod", trade_date, trigger_source, reconcile_and_snapshot, lock_connection
        )

    # 核对券商收盘订单和成交，要求没有未解决状态或仍活动的本地订单。
    def _reconcile_eod(self, trade_date: date) -> ReconciliationReport:
        self._require_current_trade_date(trade_date)
        self._require_time_at_or_after(self.config.eod.run_time, "EOD_TIME_NOT_REACHED")
        account = self._active_account(allow_paused=True)
        try:
            report = self.reconciliation.reconcile(
                account_id=self.config.account.account_id(),
                broker_orders=self._broker_records(self.broker.query_orders, "orders"),
                broker_trades=self._broker_records(self.broker.query_trades, "trades"),
                repository=self.repository,
            )
        except Exception:
            self.repository.pause_account(str(account["account_id"]), "EOD_QUERY_FAILED")
            raise
        if report.has_unresolved or report.active_broker_order_ids:
            self.repository.pause_account(str(account["account_id"]), "EOD_UNRESOLVED")
            raise RuntimeError("end-of-day reconciliation is unresolved")
        return report


    # 组织各策略的收盘资产快照写入。
    def _snapshot_eod(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
    ) -> None:
        self._require_current_trade_date(trade_date)
        self._require_time_at_or_after(self.config.eod.run_time, "EOD_TIME_NOT_REACHED")
        account = self._active_account(allow_paused=True)
        account_id = str(account["account_id"])
        runtime = self.strategy_runtime
        self._run_strategy_step(
            "strategy_snapshot", trade_date, runtime.spec.strategy_id, trigger_source,
            lambda: self._snapshot_strategy(trade_date, account_id, runtime.spec.strategy_id, runtime),
        )

    # 读取虚拟现金与持仓，用收盘价估值并保存策略日终快照。
    def _snapshot_strategy(
        self,
        trade_date: date,
        account_id: str,
        strategy_id: str,
        runtime: StrategyRuntime,
    ) -> None:
        self.repository.settle_strategy_positions(
            account_id,
            strategy_id,
            trade_date,
            turnover_rules=runtime.turnover_rules,
        )
        strategy_account = self.repository.get_strategy(account_id, strategy_id)
        if strategy_account is None:
            raise RuntimeError(f"virtual strategy account is missing: {strategy_id}")
        positions = self.repository.load_strategy_positions(
            account_id,
            strategy_id,
            turnover_rules=runtime.turnover_rules,
            captured_at=self.clock(),
        )
        held_symbols = tuple(row.symbol for row in positions if row.total_quantity > 0)
        prices = runtime.close_price_provider(trade_date, held_symbols) if held_symbols else {}
        rows = [
            {
                "symbol": row.symbol,
                "total_quantity": row.total_quantity,
                "available_quantity": row.available_quantity,
                "today_buy_quantity": row.today_buy_quantity,
                "average_cost": row.average_cost or Decimal("0"),
                "close_price": prices[row.symbol],
                "market_value": prices[row.symbol] * row.total_quantity,
            }
            for row in positions
            if row.total_quantity > 0
        ]
        self.repository.save_strategy_daily_snapshot(
            account_id=account_id,
            strategy_id=strategy_id,
            trading_date=trade_date,
            virtual_cash=Decimal(strategy_account["virtual_cash"]),
            positions=rows,
        )

    # 取得可以继续运行的账户；缺失或暂停状态按作业约定拒绝。
    def _active_account(self, *, allow_paused: bool = False) -> Mapping[str, object]:
        account = self.repository.get_account(self.config.account.account_id())
        if account is None:
            raise RuntimeError("account does not exist")
        status = _status(account["status"], AccountStatus)
        if status is not AccountStatus.ACTIVE and not (
            allow_paused and status is AccountStatus.PAUSED
        ):
            raise RuntimeError("account is not ACTIVE")
        return account

    # 要求账户不存在尚未解决的委托／对账状态。
    def _require_no_unresolved(self) -> None:
        if self.repository.current_unresolved(self.config.account.account_id()):
            account_id = self.config.account.account_id()
            self.repository.pause_account(account_id, "UNRESOLVED_LOCAL_INTENT")
            raise AccountSafetyError("account has unresolved intents")

    # 执行订单成交对账，发现未解决结果时暂停账户并阻止继续执行。
    def _reconcile_or_pause(
        self,
        account: Mapping[str, object],
        orders: Sequence[BrokerOrderSnapshot],
        trades: Sequence[BrokerTradeSnapshot],
    ) -> ReconciliationReport:
        try:
            report = self.reconciliation.reconcile(
                account_id=self.config.account.account_id(),
                broker_orders=orders,
                broker_trades=trades,
                repository=self.repository,
            )
        except Exception as error:
            self.repository.pause_account(str(account["account_id"]), "RECONCILIATION_FAILED")
            raise AccountSafetyError("broker reconciliation failed") from error
        if report.has_unresolved:
            reason = (
                "VIRTUAL_CASH_NEGATIVE_AFTER_FEES"
                if report.virtual_cash_breach_strategy_ids
                else "RECONCILIATION_UNRESOLVED"
            )
            self.repository.pause_account(str(account["account_id"]), reason)
            raise AccountSafetyError("broker reconciliation is unresolved")
        return report

    # 要求作业日期等于当前市场日期，避免补发过期交易。
    def _require_current_trade_date(self, trade_date: date) -> None:
        current = self.clock().astimezone(MARKET_TIMEZONE).date()
        if trade_date < current:
            raise JobSkipped("STALE_TRADE_DATE")
        if trade_date > current:
            raise JobSkipped("FUTURE_TRADE_DATE")

    # 要求当前市场时间已到指定作业起点。
    def _require_time_at_or_after(self, run_time: time, reason: str) -> None:
        if self.clock().astimezone(MARKET_TIMEZONE).time().replace(tzinfo=None) < run_time:
            raise JobSkipped(reason)


    # 按策略和信号／执行日期生成稳定决策标识。
    def _decision_id(self, strategy_id: str, signal_date: date) -> str:
        value = f"{self.config.account.account_id()}|{strategy_id}|{signal_date.isoformat()}"
        return hashlib.sha256(value.encode()).hexdigest()




__all__ = [
    "AccountSafetyError", "JobAlreadySucceeded", "JobSkipped",
    "LiveDailyJobs",
]
