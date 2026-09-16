"""模拟盘自动任务使用的事务型 SQLAlchemy Core 仓储。"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, delete, insert, or_, select, text, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.base import Executable

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE, normalize_symbol
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.core.sizing import calculate_target_quantities
from etf_backtest.live.persistence.schema import (
    live_account,
    live_broker_order,
    live_broker_trade,
    live_decision,
    live_job_run,
    live_order_intent,
    live_strategy,
    live_strategy_account_snapshot,
    live_strategy_position,
    live_strategy_position_snapshot,
    live_target_position,
)
from etf_backtest.live.state import (
    AccountStatus,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    JobStatus,
    JobTriggerSource,
    OrderIntent,
    OrderIntentStatus,
    StrategyStatus,
    TradeApplyResult,
)

StateRow = dict[str, Any]


# 取得市场时区当前时间，供状态表记录创建和更新时间。
def _now() -> datetime:
    return datetime.now(MARKET_TIMEZONE)


# 为数据库返回的无时区时间补上市场时区；已有时区的时间原样返回。
def _market_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=MARKET_TIMEZONE)


# 将查询结果读取为至多一条映射记录。
def _one(connection: Connection, statement: Executable) -> StateRow | None:
    row = connection.execute(statement).mappings().first()
    return None if row is None else dict(row)


# 将查询结果读取为映射记录序列。
def _many(connection: Connection, statement: Executable) -> tuple[StateRow, ...]:
    rows = connection.execute(statement).mappings().all()
    return tuple(dict(row) for row in rows)


# 生成长度可控、身份稳定的 MySQL advisory lock 名称。
def _lock_name(kind: str, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
    return f"qmt:{kind}:{digest}"


# 在指定数据库连接上申请命名锁，并返回是否成功。
def _acquire_lock(connection: Connection, lock_name: str) -> bool:
    result = connection.execute(
        text("SELECT GET_LOCK(:lock_name, 0)"), {"lock_name": lock_name}
    ).scalar_one_or_none()
    return result == 1


# 在持锁连接上释放指定 MySQL 命名锁。
def _release_lock(connection: Connection, lock_name: str) -> None:
    connection.execute(
        text("SELECT RELEASE_LOCK(:lock_name)"), {"lock_name": lock_name}
    ).scalar_one_or_none()


# 申请账户级进程独占锁，避免多个执行器操作同一账户。
def acquire_account_lock(connection: Connection, account_id: str) -> bool:
    return _acquire_lock(connection, _lock_name("account", account_id))


# 释放账户级独占锁。
def release_account_lock(connection: Connection, account_id: str) -> None:
    _release_lock(connection, _lock_name("account", account_id))


# 申请账户、作业类型与日期对应的执行锁。
def acquire_job_lock(
    connection: Connection, account_id: str, job_type: str, trade_date: date
) -> bool:
    identity = f"{account_id}|{job_type}|{trade_date.isoformat()}"
    return _acquire_lock(connection, _lock_name("job", identity))


# 释放指定日期的作业锁。
def release_job_lock(
    connection: Connection, account_id: str, job_type: str, trade_date: date
) -> None:
    identity = f"{account_id}|{job_type}|{trade_date.isoformat()}"
    _release_lock(connection, _lock_name("job", identity))


# 集中管理模拟盘账户、决策、意图、订单、成交和快照的 SQL 持久化及事务一致性。
class LiveStateRepository:
    # 保存状态数据库引擎和费用模型，供委托状态及成交账本更新使用。
    def __init__(self, engine: Engine, *, fee_model: FeeModel) -> None:
        self._engine = engine
        self._fee_model = fee_model

    # 复用调用方事务连接或自行创建连接，统一连接生命周期。
    @contextmanager
    def _connection(self, connection: Connection | None, *, write: bool) -> Iterator[Connection]:
        if connection is not None:
            yield connection
            return
        manager = self._engine.begin() if write else self._engine.connect()
        with manager as owned:
            yield owned

    # 提供数据库事务上下文，供跨表写入一起提交或回滚。
    @contextmanager
    def transaction(self, connection: Connection | None = None) -> Iterator[Connection]:
        with self._connection(connection, write=True) as active:
            yield active

    # 创建作业运行记录并返回运行 ID，后续写入最终状态。
    def start_job_run(
        self,
        *,
        account_id: str,
        job_type: str,
        trade_date: date,
        trigger_source: JobTriggerSource,
        strategy_id: str | None = None,
        connection: Connection | None = None,
        job_run_id: str | None = None,
    ) -> str:
        run_id = job_run_id or uuid.uuid4().hex
        with self._connection(connection, write=True) as active:
            active.execute(
                insert(live_job_run).values(
                    job_run_id=run_id,
                    account_id=account_id,
                    strategy_id=strategy_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                    status=JobStatus.RUNNING,
                    started_at=_now(),
                )
            )
        return run_id

    def finish_job_run(
        self, job_run_id: str, *, error: Exception | None = None,
        skip_reason: str | None = None, connection: Connection | None = None,
    ) -> None:
        """统一记录成功、失败或跳过，沿用原状态值和错误字段。"""
        details: dict[str, object] = {}
        if skip_reason is not None:
            status = JobStatus.SKIPPED
            details = {"error_type": "SKIP_REASON", "error_message": skip_reason}
        elif error is not None:
            status = JobStatus.FAILED
            details = {"error_type": type(error).__name__, "error_message": str(error)}
        else:
            status = JobStatus.SUCCEEDED
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_job_run)
                .where(live_job_run.c.job_run_id == job_run_id)
                .values(status=status, finished_at=_now(), **details)
            )


    # 查询作业是否已有可去重的终态记录，可限定策略。
    def has_terminal_job(
        self,
        account_id: str,
        job_type: str,
        trade_date: date,
        *,
        strategy_id: str | None = None,
        connection: Connection | None = None,
    ) -> bool:
        with self._connection(connection, write=False) as active:
            row = _one(
                active,
                select(live_job_run.c.job_run_id)
                .where(
                    live_job_run.c.account_id == account_id,
                    live_job_run.c.job_type == job_type,
                    live_job_run.c.trade_date == trade_date,
                    live_job_run.c.strategy_id.is_(None)
                    if strategy_id is None
                    else live_job_run.c.strategy_id == strategy_id,
                    or_(
                        live_job_run.c.status == JobStatus.SUCCEEDED,
                        and_(
                            live_job_run.c.status == JobStatus.SKIPPED,
                            ~live_job_run.c.error_message.like("FUTURE_%"),
                            live_job_run.c.error_message.not_in(
                                (
                                    "SIGNAL_TIME_NOT_REACHED",
                                    "REBALANCE_TIME_NOT_REACHED",
                                    "CANCEL_TIME_NOT_REACHED",
                                    "EOD_TIME_NOT_REACHED",
                                )
                            ),
                        ),
                    ),
                )
                .limit(1),
            )
        return row is not None

    # 读取指定券商账户的本地状态记录。
    def get_account(
        self, account_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_account).where(live_account.c.account_id == account_id),
            )

    # 创建或核对账户与单策略虚拟账本；保留已有资金状态，拒绝冲突策略和初始资金改写。
    def sync_account(
        self,
        *,
        account_id: str,
        mode: str,
        account_type: str,
        strategy: Mapping[str, object],
        connection: Connection | None = None,
    ) -> StateRow:
        strategy_id = str(strategy["strategy_id"])
        initial_capital = Decimal(str(strategy["initial_capital"]))
        if not initial_capital.is_finite() or initial_capital <= 0:
            raise ValueError("strategy initial_capital must be finite and positive")
        with self._connection(connection, write=True) as active:
            now = _now()
            account = _one(
                active,
                select(live_account)
                .where(live_account.c.account_id == account_id)
                .with_for_update(),
            )
            if account is None:
                active.execute(
                    insert(live_account).values(
                        account_id=account_id,
                        mode=mode,
                        account_type=account_type,
                        capital_pool=initial_capital,
                        status=AccountStatus.ACTIVE,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                immutable = {"mode": mode, "account_type": account_type}
                mismatched = [name for name, value in immutable.items() if account[name] != value]
                if mismatched:
                    raise ValueError("account immutable fields differ: " + ", ".join(mismatched))
                active.execute(
                    update(live_account)
                    .where(live_account.c.account_id == account_id)
                    .values(capital_pool=initial_capital, updated_at=now)
                )

            rows = _many(
                active,
                select(live_strategy)
                .where(live_strategy.c.account_id == account_id)
                .with_for_update(),
            )
            # ponytail: 保留旧表结构以读取历史账本；切换 ID 使用独立状态库，避免重新发放本金。
            if any(str(row["strategy_id"]) != strategy_id for row in rows):
                raise ValueError("state account belongs to another strategy; use a separate state database")
            current = rows[0] if rows else None
            mutable = {
                "case": strategy["case"],
                "experiment_path": str(strategy["experiment_path"]),
                "model_backend": strategy.get("model_backend"),
                "bundle_path": strategy.get("bundle_path"),
                "model_id": strategy.get("model_id"),
                "status": StrategyStatus.ACTIVE,
                "updated_at": now,
            }
            if current is None:
                active.execute(insert(live_strategy).values(
                    account_id=account_id, strategy_id=strategy_id,
                    initial_capital=initial_capital, virtual_cash=initial_capital,
                    created_at=now, **mutable,
                ))
            else:
                if current["status"] != StrategyStatus.ACTIVE:
                    raise ValueError("existing strategy ledger is not active")
                if Decimal(str(current["initial_capital"])) != initial_capital:
                    raise ValueError("strategy initial_capital is immutable")
                active.execute(update(live_strategy).where(
                    live_strategy.c.account_id == account_id,
                    live_strategy.c.strategy_id == strategy_id,
                ).values(**mutable))
            row = self.get_account(account_id, connection=active)
            assert row is not None
            return row

    # 读取指定策略的虚拟现金和运行状态。
    def get_strategy(
        self,
        account_id: str,
        strategy_id: str,
        *,
        connection: Connection | None = None,
        for_update: bool = False,
    ) -> StateRow | None:
        statement = select(live_strategy).where(
            live_strategy.c.account_id == account_id,
            live_strategy.c.strategy_id == strategy_id,
        )
        if for_update:
            statement = statement.with_for_update()
        with self._connection(connection, write=False) as active:
            return _one(active, statement)


    # 把策略持仓表转换为带周转规则的持仓快照。
    def load_strategy_positions(
        self,
        account_id: str,
        strategy_id: str,
        *,
        turnover_rules: Mapping[str, TurnoverRule],
        captured_at: datetime | None = None,
        connection: Connection | None = None,
    ) -> tuple[BrokerPositionSnapshot, ...]:
        with self._connection(connection, write=False) as active:
            rows = _many(
                active,
                select(live_strategy_position)
                .where(
                    live_strategy_position.c.account_id == account_id,
                    live_strategy_position.c.strategy_id == strategy_id,
                )
                .order_by(live_strategy_position.c.symbol),
            )
        when = captured_at or _now()
        result: list[BrokerPositionSnapshot] = []
        for row in rows:
            symbol = str(row["symbol"])
            try:
                rule = turnover_rules[symbol]
            except KeyError:
                raise ValueError(f"missing turnover rule for {symbol}") from None
            result.append(
                BrokerPositionSnapshot(
                    symbol=symbol,
                    total_quantity=int(row["total_quantity"]),
                    available_quantity=int(row["available_quantity"]),
                    today_buy_quantity=int(row["today_buy_quantity"]),
                    market_value=Decimal("0"),
                    turnover_rule=rule,
                    average_cost=Decimal(row["average_cost"]),
                    captured_at=when,
                )
            )
        return tuple(result)

    # 按交易日结算虚拟持仓，将上一交易日 T+1 买入数量释放为可卖数量。
    def settle_strategy_positions(
        self,
        account_id: str,
        strategy_id: str,
        trade_date: date,
        *,
        turnover_rules: Mapping[str, TurnoverRule],
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            rows = _many(
                active,
                select(live_strategy_position)
                .where(
                    live_strategy_position.c.account_id == account_id,
                    live_strategy_position.c.strategy_id == strategy_id,
                )
                .with_for_update(),
            )
            for row in rows:
                if row["last_settlement_date"] >= trade_date:
                    continue
                symbol = str(row["symbol"])
                try:
                    rule = turnover_rules[symbol]
                except KeyError:
                    raise ValueError(f"missing turnover rule for {symbol}") from None
                total = int(row["total_quantity"])
                active.execute(
                    update(live_strategy_position)
                    .where(
                        live_strategy_position.c.account_id == account_id,
                        live_strategy_position.c.strategy_id == strategy_id,
                        live_strategy_position.c.symbol == symbol,
                    )
                    .values(
                        available_quantity=total,
                        today_buy_quantity=0,
                        last_settlement_date=trade_date,
                        updated_at=_now(),
                    )
                )
                if rule is TurnoverRule.T0 and int(row["today_buy_quantity"]) != 0:
                    raise ValueError("persisted T0 position has a non-zero today buy bucket")

    # 按账户、策略和交易日幂等替换日快照，保存虚拟现金、持仓市值与总资产。
    def save_strategy_daily_snapshot(
        self,
        *,
        account_id: str,
        strategy_id: str,
        trading_date: date,
        virtual_cash: Decimal,
        positions: Sequence[Mapping[str, object]],
        connection: Connection | None = None,
    ) -> None:
        """幂等替换单个策略在某个交易日的快照。"""

        market_value = sum((Decimal(str(row["market_value"])) for row in positions), Decimal("0"))
        account_values = {
            "account_id": account_id,
            "strategy_id": strategy_id,
            "trading_date": trading_date,
            "virtual_cash": virtual_cash,
            "market_value": market_value,
            "total_asset": virtual_cash + market_value,
            "created_at": _now(),
        }
        with self._connection(connection, write=True) as active:
            strategy_account = self.get_strategy(
                account_id,
                strategy_id,
                connection=active,
                for_update=True,
            )
            if strategy_account is None:
                raise ValueError("snapshot strategy account does not exist")
            existing = _one(
                active,
                select(live_strategy_account_snapshot)
                .where(
                    live_strategy_account_snapshot.c.account_id == account_id,
                    live_strategy_account_snapshot.c.strategy_id == strategy_id,
                    live_strategy_account_snapshot.c.trading_date == trading_date,
                )
                .with_for_update(),
            )
            if existing is None:
                active.execute(insert(live_strategy_account_snapshot).values(**account_values))
            else:
                active.execute(
                    update(live_strategy_account_snapshot)
                    .where(
                        live_strategy_account_snapshot.c.account_id == account_id,
                        live_strategy_account_snapshot.c.strategy_id == strategy_id,
                        live_strategy_account_snapshot.c.trading_date == trading_date,
                    )
                    .values(
                        virtual_cash=virtual_cash,
                        market_value=market_value,
                        total_asset=virtual_cash + market_value,
                        created_at=_now(),
                    )
                )
            active.execute(
                delete(live_strategy_position_snapshot).where(
                    live_strategy_position_snapshot.c.account_id == account_id,
                    live_strategy_position_snapshot.c.strategy_id == strategy_id,
                    live_strategy_position_snapshot.c.trading_date == trading_date,
                )
            )
            if positions:
                active.execute(
                    insert(live_strategy_position_snapshot),
                    [
                        {
                            "account_id": account_id,
                            "strategy_id": strategy_id,
                            "trading_date": trading_date,
                            "symbol": row["symbol"],
                            "total_quantity": row["total_quantity"],
                            "available_quantity": row["available_quantity"],
                            "today_buy_quantity": row["today_buy_quantity"],
                            "average_cost": row["average_cost"],
                            "close_price": row["close_price"],
                            "market_value": row["market_value"],
                            "created_at": _now(),
                        }
                        for row in positions
                    ],
                )


    # 将账户置为暂停并记录原因，后续执行检查会阻止新交易。
    def pause_account(
        self,
        account_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_account)
                .where(live_account.c.account_id == account_id)
                .values(
                    status=AccountStatus.PAUSED,
                    pause_reason=reason,
                    paused_at=_now(),
                    updated_at=_now(),
                )
            )

    # 将账户恢复为可运行状态，并更新暂停相关信息。
    def resume_account(self, account_id: str, *, connection: Connection | None = None) -> None:
        with self._connection(connection, write=True) as active:
            result = active.execute(
                update(live_account)
                .where(
                    live_account.c.account_id == account_id,
                    live_account.c.status == AccountStatus.PAUSED,
                )
                .values(
                    status=AccountStatus.ACTIVE,
                    pause_reason=None,
                    paused_at=None,
                    updated_at=_now(),
                )
            )
            if result.rowcount == 0:
                raise ValueError("only a PAUSED account can be resumed")

    # 列出账户当前尚未解决的订单意图，供执行前检查。
    def current_unresolved(
        self, account_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        return self.list_unresolved_intents(account_id=account_id, connection=connection)

    # 按稳定决策身份创建或复用信号记录，避免同日重复决策写入。
    def create_or_get_decision(
        self,
        *,
        decision_id: str,
        account_id: str,
        strategy_id: str,
        signal_date: date,
        execution_date: date,
        schedule_index: int,
        status: DecisionStatus,
        data_as_of: date,
        model_id: str | None = None,
        connection: Connection | None = None,
    ) -> StateRow:
        with self._connection(connection, write=True) as active:
            existing = _one(
                active,
                select(live_decision).where(
                    live_decision.c.account_id == account_id,
                    live_decision.c.strategy_id == strategy_id,
                    live_decision.c.signal_date == signal_date,
                ),
            )
            if existing is not None:
                expected = {
                    "execution_date": execution_date,
                    "schedule_index": schedule_index,
                    "status": status,
                    "data_as_of": data_as_of,
                    "model_id": model_id,
                }
                if any(existing.get(name) != value for name, value in expected.items()):
                    raise ValueError("existing strategy decision has different content")
                return existing
            active.execute(
                insert(live_decision).values(
                    decision_id=decision_id,
                    account_id=account_id,
                    strategy_id=strategy_id,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    schedule_index=schedule_index,
                    status=status,
                    data_as_of=data_as_of,
                    model_id=model_id,
                    created_at=_now(),
                )
            )
            row = _one(
                active,
                select(live_decision).where(live_decision.c.decision_id == decision_id),
            )
            assert row is not None
            return row

    # 保存决策明确给出的证券目标权重。
    def save_target_positions(
        self,
        decision_id: str,
        target_weights: Mapping[str, Decimal],
        *,
        connection: Connection | None = None,
    ) -> None:
        canonical = {normalize_symbol(symbol): weight for symbol, weight in target_weights.items()}
        with self._connection(connection, write=True) as active:
            decision = _one(
                active,
                select(live_decision).where(live_decision.c.decision_id == decision_id),
            )
            if decision is None:
                raise ValueError("target decision does not exist")
            rows = _many(
                active,
                select(live_target_position).where(
                    live_target_position.c.decision_id == decision_id
                ),
            )
            existing = {str(row["symbol"]): row["target_weight"] for row in rows}
            if existing:
                if existing != canonical:
                    raise ValueError("saved target positions differ from the requested target")
                return
            if canonical:
                active.execute(
                    insert(live_target_position),
                    [
                        {
                            "decision_id": decision_id,
                            "account_id": decision["account_id"],
                            "strategy_id": decision["strategy_id"],
                            "symbol": symbol,
                            "target_weight": weight,
                        }
                        for symbol, weight in sorted(canonical.items())
                    ],
                )

    # 读取指定执行日的已保存目标决策，供尾盘执行。
    def pending_decision_for_date(
        self, account_id: str, execution_date: date, *, connection: Connection | None = None,
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            row = active.execute(select(live_decision).where(
                live_decision.c.account_id == account_id,
                live_decision.c.execution_date == execution_date,
                live_decision.c.status == DecisionStatus.TARGET_CREATED,
            )).mappings().one_or_none()
            return None if row is None else dict(row)

    # 读取决策的显式目标权重映射，省略证券不自动补零。
    def load_target_positions(
        self, decision_id: str, *, connection: Connection | None = None
    ) -> dict[str, Decimal]:
        with self._connection(connection, write=False) as active:
            rows = _many(
                active,
                select(live_target_position).where(
                    live_target_position.c.decision_id == decision_id
                ),
            )
        return {str(row["symbol"]): row["target_weight"] for row in rows}

    # 读取决策已冻结的估值价和目标数量；未冻结时返回 None，部分冻结视为损坏。
    def load_execution_targets(
        self, decision_id: str, *, connection: Connection | None = None
    ) -> dict[str, tuple[Decimal, int]] | None:
        """读取完整冻结目标；尚未冻结返回 ``None``，部分冻结视为损坏。"""

        with self._connection(connection, write=False) as active:
            rows = _many(
                active,
                select(live_target_position).where(
                    live_target_position.c.decision_id == decision_id
                ),
            )
        if not rows:
            return {}
        frozen = [
            row["execution_valuation_price"] is not None and row["target_quantity"] is not None
            for row in rows
        ]
        if not any(frozen):
            return None
        if not all(frozen):
            raise RuntimeError("execution targets are only partially frozen")
        return {
            str(row["symbol"]): (
                Decimal(row["execution_valuation_price"]),
                int(row["target_quantity"]),
            )
            for row in rows
        }

    # 首次执行时按估值和整手原子固定显式目标数量，后续买入阶段与重启恢复复用，避免价格变化重新放大目标。
    def freeze_execution_targets(
        self,
        decision_id: str,
        *,
        total_asset: Decimal,
        valuation_prices: Mapping[str, Decimal],
        lot_size: int,
        connection: Connection | None = None,
    ) -> dict[str, tuple[Decimal, int]]:
        """第一次调用原子冻结执行价格和目标股数，后续调用只返回原值。"""

        with self._connection(connection, write=True) as active:
            rows = _many(
                active,
                select(live_target_position)
                .where(live_target_position.c.decision_id == decision_id)
                .with_for_update(),
            )
            if not rows:
                return {}
            frozen = [
                row["execution_valuation_price"] is not None and row["target_quantity"] is not None
                for row in rows
            ]
            if any(frozen):
                if not all(frozen):
                    raise RuntimeError("execution targets are only partially frozen")
                return {
                    str(row["symbol"]): (
                        Decimal(row["execution_valuation_price"]),
                        int(row["target_quantity"]),
                    )
                    for row in rows
                }
            weights = {str(row["symbol"]): Decimal(row["target_weight"]) for row in rows}
            quantities = calculate_target_quantities(
                target_weights=weights,
                total_asset=total_asset,
                valuation_prices=valuation_prices,
                lot_size=lot_size,
            )
            canonical_prices = {
                normalize_symbol(symbol): price for symbol, price in valuation_prices.items()
            }
            for symbol, quantity in quantities.items():
                active.execute(
                    update(live_target_position)
                    .where(
                        live_target_position.c.decision_id == decision_id,
                        live_target_position.c.symbol == symbol,
                        live_target_position.c.execution_valuation_price.is_(None),
                        live_target_position.c.target_quantity.is_(None),
                    )
                    .values(
                        execution_valuation_price=canonical_prices[symbol],
                        target_quantity=quantity,
                    )
                )
            return {
                symbol: (canonical_prices[symbol], quantity)
                for symbol, quantity in quantities.items()
            }

    # 列出一个决策已经生成的订单意图，供恢复和重复执行检查。
    def list_order_intents_for_decision(
        self, decision_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_order_intent).where(live_order_intent.c.decision_id == decision_id),
            )

    # 按意图键幂等创建委托计划，先保存计划再向券商提交。
    def create_order_intent(
        self,
        intent: OrderIntent,
        *,
        connection: Connection | None = None,
        intent_id: str | None = None,
    ) -> StateRow:
        with self._connection(connection, write=True) as active:
            decision = _one(
                active,
                select(live_decision).where(live_decision.c.decision_id == intent.decision_id),
            )
            if decision is None:
                raise ValueError("intent decision does not exist")
            if (
                str(decision["account_id"]) != intent.account_id
                or str(decision["strategy_id"]) != intent.strategy_id
            ):
                raise ValueError("intent strategy identity does not match decision")
            existing = self.get_intent_by_key(intent.intent_key, connection=active)
            expected = {
                "strategy_id": intent.strategy_id,
                "symbol": intent.symbol,
                "side": intent.side,
                "requested_quantity": intent.requested_quantity,
                "valuation_price": intent.valuation_price,
                "limit_price": intent.limit_price,
                "remark_token": intent.remark_token,
            }
            if existing is not None:
                if any(existing[name] != value for name, value in expected.items()):
                    raise ValueError("existing intent_key has different economic content")
                return existing
            token_owner = self.get_intent_by_remark_token(intent.remark_token, connection=active)
            if token_owner is not None:
                raise ValueError("remark_token is already bound to another intent")
            identifier = intent_id or uuid.uuid4().hex
            now = _now()
            active.execute(
                insert(live_order_intent).values(
                    intent_id=identifier,
                    decision_id=intent.decision_id,
                    account_id=intent.account_id,
                    strategy_id=intent.strategy_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    requested_quantity=intent.requested_quantity,
                    valuation_price=intent.valuation_price,
                    limit_price=intent.limit_price,
                    intent_key=intent.intent_key,
                    remark_token=intent.remark_token,
                    status=intent.status,
                    created_at=now,
                    updated_at=now,
                )
            )
            row = self.get_intent_by_key(intent.intent_key, connection=active)
            assert row is not None
            return row

    # 按稳定意图键查询已有计划。
    def get_intent_by_key(
        self, intent_key: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_order_intent).where(live_order_intent.c.intent_key == intent_key),
            )

    # 按券商备注令牌查找本地意图，供回报恢复关联。
    def get_intent_by_remark_token(
        self,
        remark_token: str,
        *,
        account_id: str | None = None,
        connection: Connection | None = None,
    ) -> StateRow | None:
        statement = select(live_order_intent).where(
            live_order_intent.c.remark_token == remark_token
        )
        if account_id is not None:
            statement = statement.where(live_order_intent.c.account_id == account_id)
        with self._connection(connection, write=False) as active:
            return _one(active, statement)

    # 在调用券商前将意图标记为正在提交。
    def mark_intent_submitting(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> None:
        self._set_intent_status(intent_id, OrderIntentStatus.SUBMITTING, connection=connection)

    # 记录意图被明确拒绝及其原因。
    def mark_intent_rejected(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id, OrderIntentStatus.REJECTED, connection=connection, reject_reason=reason
        )

    # 记录提交结果未知，要求后续对账确认而非直接重发。
    def mark_intent_submit_unknown(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id,
            OrderIntentStatus.SUBMIT_UNKNOWN,
            connection=connection,
            reject_reason=reason,
        )

    # 把已满足完成条件的意图标记为完成。
    def mark_intent_completed(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> None:
        self._set_intent_status(intent_id, OrderIntentStatus.COMPLETED, connection=connection)

    # 记录已终结但未完成全部目标数量的意图。
    def mark_intent_incomplete(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id,
            OrderIntentStatus.INCOMPLETE,
            connection=connection,
            reject_reason=reason,
        )


    # 按账户及可选条件列出订单意图。
    def list_order_intents(
        self,
        *,
        account_id: str | None = None,
        strategy_id: str | None = None,
        statuses: Sequence[OrderIntentStatus] | None = None,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        statement = select(live_order_intent)
        if account_id is not None:
            statement = statement.where(live_order_intent.c.account_id == account_id)
        if strategy_id is not None:
            statement = statement.where(live_order_intent.c.strategy_id == strategy_id)
        if statuses:
            statement = statement.where(live_order_intent.c.status.in_(tuple(statuses)))
        statement = statement.order_by(live_order_intent.c.updated_at.desc())
        with self._connection(connection, write=False) as active:
            return _many(active, statement)

    # 统一更新意图状态及相关错误说明，供各状态转换方法复用。
    def _set_intent_status(
        self,
        intent_id: str,
        status: OrderIntentStatus,
        *,
        connection: Connection | None,
        reject_reason: str | None = None,
    ) -> None:
        allowed = {
            OrderIntentStatus.SUBMITTING: (OrderIntentStatus.PLANNED,),
            OrderIntentStatus.SUBMITTED: (
                OrderIntentStatus.PLANNED,
                OrderIntentStatus.SUBMITTING,
                OrderIntentStatus.SUBMIT_UNKNOWN,
                OrderIntentStatus.SUBMITTED,
            ),
            OrderIntentStatus.SUBMIT_UNKNOWN: (
                OrderIntentStatus.SUBMITTING,
                OrderIntentStatus.SUBMIT_UNKNOWN,
            ),
            OrderIntentStatus.REJECTED: (
                OrderIntentStatus.PLANNED,
                OrderIntentStatus.SUBMITTING,
                OrderIntentStatus.REJECTED,
            ),
            OrderIntentStatus.COMPLETED: (
                OrderIntentStatus.SUBMITTED,
                OrderIntentStatus.COMPLETED,
            ),
            OrderIntentStatus.INCOMPLETE: (
                OrderIntentStatus.SUBMITTED,
                OrderIntentStatus.INCOMPLETE,
            ),
        }.get(status)
        if allowed is None:
            raise ValueError(f"unsupported intent status transition target: {status}")
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_order_intent)
                .where(
                    live_order_intent.c.intent_id == intent_id,
                    live_order_intent.c.status.in_(allowed),
                )
                .values(status=status, reject_reason=reject_reason, updated_at=_now())
            )

    # 列出仍需确认提交或执行结果的意图。
    def list_unresolved_intents(
        self,
        *,
        account_id: str | None = None,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        statement = select(live_order_intent).where(
            live_order_intent.c.status.in_(
                (OrderIntentStatus.SUBMITTING, OrderIntentStatus.SUBMIT_UNKNOWN)
            )
        )
        if account_id is not None:
            statement = statement.where(live_order_intent.c.account_id == account_id)
        with self._connection(connection, write=False) as active:
            return _many(active, statement)

    # 将券商订单标识绑定到本地意图，建立后续订单与成交关联。
    def bind_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            self.upsert_broker_order(
                account_id=account_id,
                intent_id=intent_id,
                remark_token=remark_token,
                order=order,
                connection=active,
                order_sysid=order.broker_order_sysid,
                average_fill_price=order.traded_price,
            )
            self._set_intent_status(intent_id, OrderIntentStatus.SUBMITTED, connection=active)

    # 插入或合并券商订单回报，保护已成交数量与终态不被较旧回报倒退覆盖。
    def upsert_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: Connection | None = None,
        order_sysid: str | None = None,
        average_fill_price: Decimal | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            intent = self.get_intent(intent_id, connection=active)
            if intent is None or str(intent["account_id"]) != account_id:
                raise ValueError("broker order intent does not belong to account")
            existing = _one(
                active,
                select(live_broker_order)
                .where(
                    live_broker_order.c.account_id == account_id,
                    live_broker_order.c.broker_order_id == order.broker_order_id,
                )
                .with_for_update(),
            )
            values = {
                "account_id": account_id,
                "strategy_id": intent["strategy_id"],
                "broker_order_id": order.broker_order_id,
                "order_sysid": order_sysid,
                "intent_id": intent_id,
                "requested_quantity": order.requested_quantity,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": average_fill_price,
                "status": order.status,
                "remark_token": remark_token,
                "updated_at": order.captured_at,
            }
            if existing is None:
                active.execute(insert(live_broker_order).values(**values))
                return
            if str(existing["intent_id"]) != intent_id:
                raise ValueError("broker order cannot be rebound to another intent")
            existing_status = BrokerOrderStatus(str(existing["status"]))
            existing_time = _market_datetime(existing["updated_at"])
            incoming_time = _market_datetime(order.captured_at)
            incoming_is_newer = incoming_time >= existing_time
            existing_filled = int(existing["filled_quantity"])
            incoming_has_more_fills = order.filled_quantity > existing_filled
            merged_filled = max(existing_filled, order.filled_quantity)
            existing_is_terminal = not existing_status.is_active and (
                existing_status is not BrokerOrderStatus.UNKNOWN
            )
            incoming_is_terminal = not order.status.is_active and (
                order.status is not BrokerOrderStatus.UNKNOWN
            )
            if existing_is_terminal:
                if order.status is BrokerOrderStatus.FILLED and merged_filled >= int(
                    existing["requested_quantity"]
                ):
                    merged_status = BrokerOrderStatus.FILLED
                elif incoming_is_terminal and incoming_is_newer:
                    merged_status = order.status
                else:
                    merged_status = existing_status
            elif incoming_is_terminal:
                merged_status = order.status
            elif (
                existing_status is BrokerOrderStatus.PARTIALLY_FILLED
                or order.status is BrokerOrderStatus.PARTIALLY_FILLED
                or merged_filled > 0
            ):
                merged_status = BrokerOrderStatus.PARTIALLY_FILLED
            elif order.status is BrokerOrderStatus.PENDING:
                merged_status = BrokerOrderStatus.PENDING
            else:
                merged_status = existing_status
            active.execute(
                update(live_broker_order)
                .where(
                    live_broker_order.c.account_id == account_id,
                    live_broker_order.c.broker_order_id == order.broker_order_id,
                )
                .values(
                    order_sysid=(
                        order_sysid if order_sysid is not None else existing["order_sysid"]
                    ),
                    requested_quantity=(
                        order.requested_quantity
                        if incoming_is_newer
                        else existing["requested_quantity"]
                    ),
                    filled_quantity=merged_filled,
                    average_fill_price=(
                        average_fill_price
                        if average_fill_price is not None
                        and (incoming_is_newer or incoming_has_more_fills)
                        else existing["average_fill_price"]
                    ),
                    status=merged_status,
                    remark_token=existing["remark_token"],
                    updated_at=max(existing_time, incoming_time),
                )
            )

    # 按券商订单标识读取本地订单记录。
    def get_broker_order(
        self,
        account_id: str,
        broker_order_id: str,
        *,
        connection: Connection | None = None,
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_broker_order).where(
                    live_broker_order.c.account_id == account_id,
                    live_broker_order.c.broker_order_id == broker_order_id,
                ),
            )


    # 按账户与成交 ID 幂等写入成交，并在同一事务更新意图和虚拟现金／持仓；现金越界记录事实后报告异常。
    def record_strategy_trade_if_absent(
        self,
        *,
        account_id: str,
        intent_id: str,
        trade: BrokerTradeSnapshot,
        turnover_rule: TurnoverRule,
        connection: Connection | None = None,
    ) -> TradeApplyResult:
        """以原子方式持久化单笔本地成交，并将其应用到策略账本。"""

        if trade.quantity <= 0 or trade.price <= 0 or not trade.price.is_finite():
            raise ValueError("trade quantity and price must be positive")
        with self._connection(connection, write=True) as active:
            intent = self.get_intent(intent_id, connection=active)
            if intent is None:
                raise ValueError("trade intent does not exist")
            side = intent["side"]
            intent_side = side if isinstance(side, OrderSide) else OrderSide(str(side))
            if str(intent["symbol"]) != trade.symbol or intent_side is not trade.side:
                raise ValueError("trade identity does not match intent")
            if str(intent["account_id"]) != account_id:
                raise ValueError("trade intent does not belong to account")
            strategy_id = str(intent["strategy_id"])
            strategy_account = self.get_strategy(
                account_id,
                strategy_id,
                connection=active,
                for_update=True,
            )
            if strategy_account is None:
                raise ValueError("trade strategy account does not exist")
            duplicate = _one(
                active,
                select(live_broker_trade)
                .where(
                    live_broker_trade.c.account_id == account_id,
                    live_broker_trade.c.broker_trade_id == trade.broker_trade_id,
                )
                .with_for_update(),
            )
            if duplicate is not None:
                return TradeApplyResult(
                    inserted=False,
                    virtual_cash=Decimal(strategy_account["virtual_cash"]),
                )

            position = _one(
                active,
                select(live_strategy_position)
                .where(
                    live_strategy_position.c.account_id == account_id,
                    live_strategy_position.c.strategy_id == strategy_id,
                    live_strategy_position.c.symbol == trade.symbol,
                )
                .with_for_update(),
            )
            trade_date = trade.traded_at.astimezone(MARKET_TIMEZONE).date()
            if position is None:
                total = available = today = 0
                average_cost = Decimal("0")
                last_settlement_date = trade_date
            else:
                total = int(position["total_quantity"])
                available = int(position["available_quantity"])
                today = int(position["today_buy_quantity"])
                average_cost = Decimal(position["average_cost"])
                last_settlement_date = position["last_settlement_date"]
                if last_settlement_date < trade_date:
                    available = total
                    today = 0
                    last_settlement_date = trade_date

            notional = trade.price * trade.quantity
            fee = self._fee_model.calculate(trade_amount=notional, side=trade.side)
            cash = Decimal(strategy_account["virtual_cash"])
            if trade.side is OrderSide.BUY:
                new_total = total + trade.quantity
                average_cost = (
                    (average_cost * total + notional + fee.total) / new_total
                    if new_total
                    else Decimal("0")
                )
                total = new_total
                if turnover_rule is TurnoverRule.T0:
                    available += trade.quantity
                    today = 0
                else:
                    today += trade.quantity
                cash -= notional + fee.total
            else:
                if trade.quantity > available or trade.quantity > total:
                    raise ValueError("trade exceeds strategy virtual available quantity")
                total -= trade.quantity
                available -= trade.quantity
                cash += notional - fee.total

            if turnover_rule is TurnoverRule.T0:
                if available != total:
                    raise ValueError("T0 virtual position buckets are inconsistent")
                today = 0
            elif available + today != total:
                raise ValueError("T1 virtual position buckets are inconsistent")

            active.execute(
                update(live_strategy)
                .where(
                    live_strategy.c.account_id == account_id,
                    live_strategy.c.strategy_id == strategy_id,
                )
                .values(virtual_cash=cash, updated_at=_now())
            )
            if total == 0:
                active.execute(
                    delete(live_strategy_position).where(
                        live_strategy_position.c.account_id == account_id,
                        live_strategy_position.c.strategy_id == strategy_id,
                        live_strategy_position.c.symbol == trade.symbol,
                    )
                )
            elif position is None:
                active.execute(
                    insert(live_strategy_position).values(
                        account_id=account_id,
                        strategy_id=strategy_id,
                        symbol=trade.symbol,
                        total_quantity=total,
                        available_quantity=available,
                        today_buy_quantity=today,
                        average_cost=average_cost,
                        last_settlement_date=last_settlement_date,
                        updated_at=_now(),
                    )
                )
            else:
                active.execute(
                    update(live_strategy_position)
                    .where(
                        live_strategy_position.c.account_id == account_id,
                        live_strategy_position.c.strategy_id == strategy_id,
                        live_strategy_position.c.symbol == trade.symbol,
                    )
                    .values(
                        total_quantity=total,
                        available_quantity=available,
                        today_buy_quantity=today,
                        average_cost=average_cost,
                        last_settlement_date=last_settlement_date,
                        updated_at=_now(),
                    )
                )
            active.execute(
                insert(live_broker_trade).values(
                    account_id=account_id,
                    strategy_id=strategy_id,
                    broker_trade_id=trade.broker_trade_id,
                    broker_order_id=trade.broker_order_id,
                    intent_id=intent_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=fee.commission,
                    stamp_duty=fee.stamp_duty,
                    total_fee=fee.total,
                    trade_time=trade.traded_at,
                )
            )
            cash_breach = cash < 0
            if cash_breach:
                active.execute(
                    update(live_account)
                    .where(live_account.c.account_id == account_id)
                    .values(
                        status=AccountStatus.PAUSED,
                        pause_reason="VIRTUAL_CASH_NEGATIVE_AFTER_FEES",
                        paused_at=_now(),
                        updated_at=_now(),
                    )
                )
            return TradeApplyResult(
                inserted=True,
                virtual_cash=cash,
                cash_breach=cash_breach,
            )


    # 按数据库意图 ID 读取一条委托计划。
    def get_intent(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        statement = select(live_order_intent).where(live_order_intent.c.intent_id == intent_id)
        with self._connection(connection, write=False) as active:
            return _one(active, statement)

    # 列出已关联某意图的成交，供完成数量和终态判断。
    def list_broker_trades_for_intent(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_broker_trade).where(live_broker_trade.c.intent_id == intent_id),
            )

    # 统计尚未完成意图占用的买入资金和卖出数量，防止重复使用同一份现金或可卖持仓。
    def strategy_reservations(
        self,
        account_id: str,
        strategy_id: str,
        *,
        connection: Connection | None = None,
    ) -> tuple[Decimal, dict[str, int], tuple[StateRow, ...]]:
        """返回单个策略的 BUY 现金占用、SELL 数量占用和活动本地意图。"""

        statuses = (
            OrderIntentStatus.PLANNED,
            OrderIntentStatus.SUBMITTING,
            OrderIntentStatus.SUBMITTED,
            OrderIntentStatus.SUBMIT_UNKNOWN,
        )
        with self._connection(connection, write=False) as active:
            rows = self.list_order_intents(
                account_id=account_id,
                strategy_id=strategy_id,
                statuses=statuses,
                connection=active,
            )
            buy_cash = Decimal("0")
            sell_quantities: dict[str, int] = {}
            intents: list[StateRow] = []
            for intent in rows:
                trades = self.list_broker_trades_for_intent(
                    str(intent["intent_id"]), connection=active
                )
                filled = sum(int(row["quantity"]) for row in trades)
                remaining = max(0, int(intent["requested_quantity"]) - filled)
                intent = dict(intent)
                intent["remaining_quantity"] = remaining
                intents.append(intent)
                side = intent["side"]
                order_side = side if isinstance(side, OrderSide) else OrderSide(str(side))
                if order_side is OrderSide.BUY:
                    buy_cash += Decimal(intent["limit_price"]) * remaining
                else:
                    symbol = str(intent["symbol"])
                    sell_quantities[symbol] = sell_quantities.get(symbol, 0) + remaining
        return buy_cash, sell_quantities, tuple(intents)

    # 统计策略指定交易日的委托名义金额，供单日金额风控。
    def strategy_order_notional_for_date(
        self,
        account_id: str,
        strategy_id: str,
        execution_date: date,
        *,
        connection: Connection | None = None,
    ) -> Decimal:
        statement = (
            select(live_order_intent)
            .join(live_decision)
            .where(
                live_decision.c.account_id == account_id,
                live_decision.c.execution_date == execution_date,
                live_order_intent.c.strategy_id == strategy_id,
                live_order_intent.c.status.not_in(
                    (OrderIntentStatus.REJECTED, OrderIntentStatus.ABANDONED)
                ),
            )
        )
        with self._connection(connection, write=False) as active:
            rows = _many(active, statement)
        return sum(
            (Decimal(str(row["limit_price"])) * int(row["requested_quantity"]) for row in rows),
            Decimal("0"),
        )


__all__ = [
    "LiveStateRepository",
    "acquire_account_lock",
    "acquire_job_lock",
    "release_account_lock",
    "release_job_lock",
]
