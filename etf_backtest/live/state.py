"""内部实盘交易边界共用的精简不可变值对象。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.live.broker.symbols import normalize_broker_symbol


# 检查状态对象时间包含有效时区，避免无时区回报参与时序比较。
def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


# 定义本地意图从计划、提交、受理到完成／未知等阶段的状态。
class OrderIntentStatus(StrEnum):
    PLANNED = "PLANNED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"
    ABANDONED = "ABANDONED"
    REJECTED = "REJECTED"


# 定义券商订单的活动、成交、撤销、拒绝等标准状态。
class BrokerOrderStatus(StrEnum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

    # 判断该订单状态是否仍可能继续成交。
    @property
    def is_active(self) -> bool:
        return self in {BrokerOrderStatus.PENDING, BrokerOrderStatus.PARTIALLY_FILLED}


# 定义账户可运行或暂停等状态，供执行入口检查。
class AccountStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"


# 定义策略账本的运行状态。
class StrategyStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    RETIRED = "RETIRED"


# 定义作业开始、成功、失败或跳过等记录状态。
class JobStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# 标记作业由调度或其他入口触发，供运行日志溯源。
class JobTriggerSource(StrEnum):
    SCHEDULED = "SCHEDULED"
    MANUAL = "MANUAL"
    RECOVERY = "RECOVERY"


# 定义资产快照的业务类型。
class SnapshotType(StrEnum):
    CURRENT = "CURRENT"
    EOD = "EOD"


# 区分券商明确受理、明确拒绝和无法确认提交结果。
class SubmitOrderStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


# 保存券商资产查询结果及采集时间。
@dataclass(frozen=True, slots=True)
class BrokerAssetSnapshot:
    total_asset: Decimal
    available_cash: Decimal
    captured_at: datetime
    account_id: str | None = None
    frozen_cash: Decimal = Decimal("0")
    market_value: Decimal = Decimal("0")

    # 要求券商资产快照的采集时间包含时区。
    def __post_init__(self) -> None:
        _aware(self.captured_at, "captured_at")


# 保存券商持仓数量、可卖量、市值及周转规则。
@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    symbol: str
    total_quantity: int
    available_quantity: int
    today_buy_quantity: int
    market_value: Decimal
    turnover_rule: TurnoverRule
    captured_at: datetime
    account_id: str | None = None
    frozen_quantity: int = 0
    on_road_quantity: int = 0
    yesterday_quantity: int = 0
    average_cost: Decimal | None = None

    # 规范券商持仓证券代码，并检查采集时间包含时区。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_broker_symbol(self.symbol))
        _aware(self.captured_at, "captured_at")


# 保存券商订单身份、数量、限价、状态与本地备注关联。
@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    symbol: str
    side: OrderSide
    requested_quantity: int
    filled_quantity: int
    limit_price: Decimal
    status: BrokerOrderStatus
    captured_at: datetime
    remark_token: str | None = None
    account_id: str | None = None
    broker_order_sysid: str | None = None
    traded_price: Decimal | None = None

    # 规范券商订单证券代码，并检查采集时间包含时区。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_broker_symbol(self.symbol))
        _aware(self.captured_at, "captured_at")

    # 计算委托数量减去已成交数量得到的剩余数量。
    @property
    def remaining_quantity(self) -> int:
        return max(0, self.requested_quantity - self.filled_quantity)


# 保存一笔券商成交的唯一标识、证券、方向、价格和数量。
@dataclass(frozen=True, slots=True)
class BrokerTradeSnapshot:
    broker_trade_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    traded_at: datetime
    account_id: str | None = None
    broker_order_sysid: str | None = None
    remark_token: str | None = None

    # 规范券商成交证券代码，并检查成交时间包含时区。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_broker_symbol(self.symbol))
        _aware(self.traded_at, "traded_at")


# 保存最新价、盘口、涨跌停、停牌与报价时间，供尾盘限价计算。
@dataclass(frozen=True, slots=True)
class LiveQuote:
    symbol: str
    last_price: Decimal
    bid1: Decimal | None
    ask1: Decimal | None
    lower_limit: Decimal | None
    upper_limit: Decimal | None
    suspended: bool
    quoted_at: datetime
    price_tick: Decimal = Decimal("0.001")

    # 规范实时行情证券代码，并检查报价时间包含时区。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.quoted_at, "quoted_at")


# 保存尚待执行的标准委托意图，包括稳定键、目标权重、估值价和限价。
@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_key: str
    remark_token: str
    account_id: str
    strategy_id: str
    decision_id: str
    execution_date: date
    symbol: str
    side: OrderSide
    requested_quantity: int
    target_weight: Decimal
    valuation_price: Decimal
    limit_price: Decimal
    status: OrderIntentStatus = OrderIntentStatus.PLANNED

    # 将订单意图中的证券代码规范为内部格式；交易约束由规划与风控步骤检查。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))


# 封装一次券商下单调用的结果及可选订单 ID／错误信息。
@dataclass(frozen=True, slots=True)
class SubmitOrderResult:
    status: SubmitOrderStatus
    broker_order_id: str | None = None
    error: str | None = None

    # 要求受理结果携带券商订单 ID，非受理结果不能携带该 ID。
    def __post_init__(self) -> None:
        if self.status is SubmitOrderStatus.ACCEPTED and not self.broker_order_id:
            raise ValueError("accepted submission requires broker_order_id")
        if self.status is not SubmitOrderStatus.ACCEPTED and self.broker_order_id is not None:
            raise ValueError("only accepted submission may contain broker_order_id")


# 封装券商或行情查询结果，区分成功空列表与查询失败。
@dataclass(frozen=True, slots=True)
class QueryResult[RecordT]:
    success: bool
    records: tuple[RecordT, ...] = ()
    error: str | None = None

    # 要求成功查询没有错误；失败查询必须包含错误且不能同时携带记录。
    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("successful query cannot contain an error")
        if not self.success and (not self.error or self.records):
            raise ValueError("failed query requires an error and no records")


# 汇总主动对账已处理记录、未解决问题与未完成意图。
@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    matched_order_count: int
    inserted_trade_count: int
    unresolved_intent_ids: tuple[str, ...] = ()
    active_broker_order_ids: tuple[str, ...] = ()
    incomplete_intent_ids: tuple[str, ...] = ()
    order_trade_mismatch_ids: tuple[str, ...] = ()
    order_identity_mismatch_ids: tuple[str, ...] = ()
    trade_identity_mismatch_ids: tuple[str, ...] = ()
    unknown_order_status_ids: tuple[str, ...] = ()
    unknown_broker_order_ids: tuple[str, ...] = ()
    unknown_broker_trade_ids: tuple[str, ...] = ()
    virtual_cash_breach_strategy_ids: tuple[str, ...] = ()

    # 判断本次对账是否存在需要暂停执行的未解决问题。
    @property
    def has_unresolved(self) -> bool:
        return bool(
            self.unresolved_intent_ids
            or self.order_trade_mismatch_ids
            or self.order_identity_mismatch_ids
            or self.trade_identity_mismatch_ids
            or self.unknown_order_status_ids
            or self.unknown_broker_order_ids
            or self.unknown_broker_trade_ids
            or self.virtual_cash_breach_strategy_ids
        )

    # 判断本次对账是否存在已终结但目标未完整成交的意图。
    @property
    def has_incomplete(self) -> bool:
        return bool(self.incomplete_intent_ids)


# 保存成交应用结果，标明是否重复及账本处理情况。
@dataclass(frozen=True, slots=True)
class TradeApplyResult:
    inserted: bool
    virtual_cash: Decimal
    cash_breach: bool = False


__all__ = [
    "AccountStatus",
    "BrokerAssetSnapshot",
    "BrokerOrderSnapshot",
    "BrokerOrderStatus",
    "BrokerPositionSnapshot",
    "BrokerTradeSnapshot",
    "JobStatus",
    "JobTriggerSource",
    "LiveQuote",
    "OrderIntent",
    "OrderIntentStatus",
    "QueryResult",
    "ReconciliationReport",
    "SnapshotType",
    "StrategyStatus",
    "SubmitOrderResult",
    "SubmitOrderStatus",
    "TradeApplyResult",
]
