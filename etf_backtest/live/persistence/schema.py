"""Live 状态数据库拥有的 13 张 SQLAlchemy Core 表。"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import (
    AccountStatus,
    BrokerOrderStatus,
    JobStatus,
    JobTriggerSource,
    OrderIntentStatus,
    SnapshotType,
    StrategyStatus,
)

metadata = MetaData()

live_account = Table(
    "live_account",
    metadata,
    Column("account_id", String(64), primary_key=True),
    Column("mode", String(16), nullable=False),
    Column("account_type", String(16), nullable=False),
    Column("capital_pool", Numeric(24, 8), nullable=False),
    Column("status", Enum(AccountStatus), nullable=False),
    Column("pause_reason", Text),
    Column("paused_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_live_account_status", "status"),
)

live_strategy = Table(
    "live_strategy",
    metadata,
    Column("account_id", ForeignKey("live_account.account_id"), primary_key=True),
    Column("strategy_id", String(64), primary_key=True),
    Column("case", String(16), nullable=False),
    Column("initial_capital", Numeric(24, 8), nullable=False),
    Column("virtual_cash", Numeric(24, 8), nullable=False),
    Column("experiment_path", String(512), nullable=False),
    Column("model_backend", String(16)),
    Column("bundle_path", String(512)),
    Column("model_id", String(128)),
    Column("status", Enum(StrategyStatus), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_live_strategy_status", "account_id", "status"),
)

live_strategy_position = Table(
    "live_strategy_position",
    metadata,
    Column("account_id", String(64), primary_key=True),
    Column("strategy_id", String(64), primary_key=True),
    Column("symbol", String(16), primary_key=True),
    Column("total_quantity", BigInteger, nullable=False),
    Column("available_quantity", BigInteger, nullable=False),
    Column("today_buy_quantity", BigInteger, nullable=False),
    Column("average_cost", Numeric(24, 8), nullable=False),
    Column("last_settlement_date", Date, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["account_id", "strategy_id"],
        ["live_strategy.account_id", "live_strategy.strategy_id"],
    ),
)

live_strategy_account_snapshot = Table(
    "live_strategy_account_snapshot",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("trading_date", Date, nullable=False),
    Column("virtual_cash", Numeric(24, 8), nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("total_asset", Numeric(24, 8), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "account_id",
        "strategy_id",
        "trading_date",
        name="uq_live_strategy_account_snapshot_key",
    ),
    ForeignKeyConstraint(
        ["account_id", "strategy_id"],
        ["live_strategy.account_id", "live_strategy.strategy_id"],
    ),
)

live_strategy_position_snapshot = Table(
    "live_strategy_position_snapshot",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("trading_date", Date, nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("total_quantity", BigInteger, nullable=False),
    Column("available_quantity", BigInteger, nullable=False),
    Column("today_buy_quantity", BigInteger, nullable=False),
    Column("average_cost", Numeric(24, 8), nullable=False),
    Column("close_price", Numeric(24, 8), nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint(
        "account_id",
        "strategy_id",
        "trading_date",
        "symbol",
        name="uq_live_strategy_position_snapshot_key",
    ),
    ForeignKeyConstraint(
        ["account_id", "strategy_id"],
        ["live_strategy.account_id", "live_strategy.strategy_id"],
    ),
)

live_job_run = Table(
    "live_job_run",
    metadata,
    Column("job_run_id", String(64), primary_key=True),
    Column("account_id", ForeignKey("live_account.account_id"), nullable=False),
    Column("strategy_id", String(64)),
    Column("job_type", String(64), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("trigger_source", Enum(JobTriggerSource), nullable=False),
    Column("status", Enum(JobStatus), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("error_type", String(128)),
    Column("error_message", Text),
    Index("ix_live_job_run_account_job_date", "account_id", "job_type", "trade_date"),
    Index(
        "ix_live_job_run_strategy_job_date",
        "account_id",
        "strategy_id",
        "job_type",
        "trade_date",
    ),
    ForeignKeyConstraint(
        ["account_id", "strategy_id"],
        ["live_strategy.account_id", "live_strategy.strategy_id"],
        name="fk_live_job_run_strategy",
    ),
)

live_decision = Table(
    "live_decision",
    metadata,
    Column("decision_id", String(64), primary_key=True),
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("signal_date", Date, nullable=False),
    Column("execution_date", Date, nullable=False),
    Column("schedule_index", Integer, nullable=False),
    Column("status", Enum(DecisionStatus), nullable=False),
    Column("data_as_of", Date, nullable=False),
    Column("model_id", String(128)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("decision_id", "account_id", "strategy_id", name="uq_live_decision_owner"),
    UniqueConstraint(
        "account_id",
        "strategy_id",
        "signal_date",
        name="uq_live_decision_strategy_signal",
    ),
    ForeignKeyConstraint(
        ["account_id", "strategy_id"],
        ["live_strategy.account_id", "live_strategy.strategy_id"],
    ),
)

live_target_position = Table(
    "live_target_position",
    metadata,
    Column("decision_id", String(64), nullable=False),
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("target_weight", Numeric(18, 10), nullable=False),
    Column("execution_valuation_price", Numeric(24, 8)),
    Column("target_quantity", BigInteger),
    UniqueConstraint("decision_id", "symbol", name="uq_live_target_decision_symbol"),
    ForeignKeyConstraint(
        ["decision_id", "account_id", "strategy_id"],
        ["live_decision.decision_id", "live_decision.account_id", "live_decision.strategy_id"],
    ),
)

live_order_intent = Table(
    "live_order_intent",
    metadata,
    Column("intent_id", String(64), primary_key=True),
    Column("decision_id", String(64), nullable=False),
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("side", Enum(OrderSide), nullable=False),
    Column("requested_quantity", BigInteger, nullable=False),
    Column("valuation_price", Numeric(24, 8), nullable=False),
    Column("limit_price", Numeric(24, 8), nullable=False),
    Column("intent_key", String(64), nullable=False),
    Column("remark_token", String(24), nullable=False),
    Column("status", Enum(OrderIntentStatus), nullable=False),
    Column("reject_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("intent_id", "account_id", "strategy_id", name="uq_live_intent_owner"),
    UniqueConstraint("intent_key", name="uq_live_order_intent_key"),
    UniqueConstraint("remark_token", name="uq_live_order_remark_token"),
    ForeignKeyConstraint(
        ["decision_id", "account_id", "strategy_id"],
        ["live_decision.decision_id", "live_decision.account_id", "live_decision.strategy_id"],
    ),
)

live_broker_order = Table(
    "live_broker_order",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("broker_order_id", String(128), nullable=False),
    Column("order_sysid", String(128)),
    Column("intent_id", String(64), nullable=False),
    Column("requested_quantity", BigInteger, nullable=False),
    Column("filled_quantity", BigInteger, nullable=False),
    Column("average_fill_price", Numeric(24, 8)),
    Column("status", Enum(BrokerOrderStatus), nullable=False),
    Column("remark_token", String(24), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("account_id", "broker_order_id", name="uq_live_broker_order_account_id"),
    ForeignKeyConstraint(
        ["intent_id", "account_id", "strategy_id"],
        [
            "live_order_intent.intent_id",
            "live_order_intent.account_id",
            "live_order_intent.strategy_id",
        ],
    ),
)

live_broker_trade = Table(
    "live_broker_trade",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("strategy_id", String(64), nullable=False),
    Column("broker_trade_id", String(128), nullable=False),
    Column("broker_order_id", String(128), nullable=False),
    Column("intent_id", String(64), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("side", Enum(OrderSide), nullable=False),
    Column("quantity", BigInteger, nullable=False),
    Column("price", Numeric(24, 8), nullable=False),
    Column("commission", Numeric(24, 8), nullable=False),
    Column("stamp_duty", Numeric(24, 8), nullable=False),
    Column("total_fee", Numeric(24, 8), nullable=False),
    Column("trade_time", DateTime(timezone=True), nullable=False),
    UniqueConstraint("account_id", "broker_trade_id", name="uq_live_broker_trade_account_id"),
    ForeignKeyConstraint(
        ["intent_id", "account_id", "strategy_id"],
        [
            "live_order_intent.intent_id",
            "live_order_intent.account_id",
            "live_order_intent.strategy_id",
        ],
    ),
)

live_account_snapshot = Table(
    "live_account_snapshot",
    metadata,
    Column("account_id", ForeignKey("live_account.account_id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("snapshot_type", Enum(SnapshotType), nullable=False),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("cash", Numeric(24, 8), nullable=False),
    Column("available_cash", Numeric(24, 8), nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("total_asset", Numeric(24, 8), nullable=False),
    Column("frozen_cash", Numeric(24, 8), nullable=False),
    UniqueConstraint(
        "account_id", "trade_date", "snapshot_type", name="uq_live_account_snapshot_key"
    ),
)

live_position_snapshot = Table(
    "live_position_snapshot",
    metadata,
    Column("account_id", ForeignKey("live_account.account_id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("snapshot_type", Enum(SnapshotType), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("total_quantity", BigInteger, nullable=False),
    Column("available_quantity", BigInteger, nullable=False),
    Column("frozen_quantity", BigInteger, nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("last_price", Numeric(24, 8), nullable=False),
    UniqueConstraint(
        "account_id",
        "trade_date",
        "snapshot_type",
        "symbol",
        name="uq_live_position_snapshot_key",
    ),
)

LIVE_TABLES = (
    live_account,
    live_strategy,
    live_strategy_position,
    live_strategy_account_snapshot,
    live_strategy_position_snapshot,
    live_job_run,
    live_decision,
    live_target_position,
    live_order_intent,
    live_broker_order,
    live_broker_trade,
    live_account_snapshot,
    live_position_snapshot,
)

__all__ = ["LIVE_TABLES", "metadata"]
