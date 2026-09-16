"""xtquant 值与实盘领域值之间唯一的转换边界。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from types import ModuleType
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.live.broker.symbols import normalize_broker_symbol
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    LiveQuote,
    SubmitOrderResult,
    SubmitOrderStatus,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")

# 以下为 xtquant 已公布的值；运行时提交订单仍读取已安装 SDK 的命名常量，防止 SDK
# 发生变化后静默使用错误值提交。
STOCK_BUY = 23
STOCK_SELL = 24
OFFSET_BUY = 48
OFFSET_SELL = 49

_ORDER_STATUSES = {
    48: BrokerOrderStatus.PENDING,  # 未报
    49: BrokerOrderStatus.PENDING,  # 待报
    50: BrokerOrderStatus.PENDING,  # 已报
    51: BrokerOrderStatus.PENDING,  # 已报待撤
    52: BrokerOrderStatus.PARTIALLY_FILLED,  # 部成待撤
    53: BrokerOrderStatus.CANCELED,  # 部撤
    54: BrokerOrderStatus.CANCELED,  # 已撤
    55: BrokerOrderStatus.PARTIALLY_FILLED,  # 部成
    56: BrokerOrderStatus.FILLED,  # 已成
    57: BrokerOrderStatus.REJECTED,  # 废单
    255: BrokerOrderStatus.UNKNOWN,
}


def decimal_value(value: object) -> Decimal:
    """转换外部数值，避免从 float 直接构造 Decimal。"""

    return Decimal(str(value))


# 调用 int() 将 SDK 字段转换为整数，供数量和状态映射复用。
def integer_value(value: object) -> int:
    return int(cast(Any, value))


# 把框架证券代码转换为券商 SDK 使用的代码格式。
def internal_to_external_symbol(symbol: str) -> str:
    exchange, _, code = normalize_symbol(symbol).partition(".")
    return f"{code}.{exchange}"


# 把券商证券代码转换为框架统一格式。
def external_to_internal_symbol(symbol: str) -> str:
    return normalize_broker_symbol(symbol)


# 将内部买卖方向映射为 xtquant 常量。
def side_to_xt(side: OrderSide, constants: ModuleType | Any) -> int:
    if side is OrderSide.BUY:
        return int(constants.STOCK_BUY)
    if side is OrderSide.SELL:
        return int(constants.STOCK_SELL)
    raise ValueError(f"unsupported order side: {side}")


# 将 xtquant 买卖方向常量还原为内部枚举。
def side_from_xt(
    order_type: object | None,
    *,
    offset_flag: object | None = None,
    constants: ModuleType | Any | None = None,
) -> OrderSide:
    buys = {STOCK_BUY, OFFSET_BUY}
    sells = {STOCK_SELL, OFFSET_SELL}
    if constants is not None:
        buys.add(int(constants.STOCK_BUY))
        sells.add(int(constants.STOCK_SELL))
    values = {integer_value(value) for value in (order_type, offset_flag) if value is not None}
    if values & buys:
        return OrderSide.BUY
    if values & sells:
        return OrderSide.SELL
    raise ValueError(f"unsupported stock order side: {values}")


# 将 SDK 订单状态码转换为应用订单状态，未知状态按映射约定处理。
def order_status_from_xt(value: object) -> BrokerOrderStatus:
    try:
        return _ORDER_STATUSES.get(integer_value(value), BrokerOrderStatus.UNKNOWN)
    except (TypeError, ValueError):
        return BrokerOrderStatus.UNKNOWN


def market_datetime(
    value: object,
    *,
    unit: Literal["seconds", "milliseconds"],
) -> datetime:
    """使用明确单位转换 SDK 文档规定的 epoch 值。

    XtOrder/XtTrade 时间按秒处理，xtdata tick 时间按毫秒处理；目标 SDK 版本必须确认
    这些单位。
    """

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=SHANGHAI)
        return value.astimezone(SHANGHAI)
    divisor = Decimal("1000") if unit == "milliseconds" else Decimal("1")
    seconds = decimal_value(value) / divisor
    return datetime.fromtimestamp(float(seconds), tz=SHANGHAI)


# 从 SDK 对象读取字段，并按调用方约定处理缺失值。
def _attribute(source: object, name: str, default: object | None = None) -> object:
    return getattr(source, name, default)


# 把 SDK 资产对象转换成带采集时间的 BrokerAssetSnapshot。
def map_asset(source: object, *, captured_at: datetime) -> BrokerAssetSnapshot:
    return BrokerAssetSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        available_cash=decimal_value(_attribute(source, "cash", 0)),
        frozen_cash=decimal_value(_attribute(source, "frozen_cash", 0)),
        market_value=decimal_value(_attribute(source, "market_value", 0)),
        total_asset=decimal_value(_attribute(source, "total_asset", 0)),
        captured_at=captured_at,
    )


# 把 SDK 持仓对象转换成标准持仓快照，并携带 T+0/T+1 规则。
def map_position(source: object, *, captured_at: datetime) -> BrokerPositionSnapshot:
    on_road = integer_value(_attribute(source, "on_road_volume", 0))
    average_cost = _attribute(source, "avg_price", _attribute(source, "open_price"))
    return BrokerPositionSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        total_quantity=integer_value(_attribute(source, "volume", 0)),
        available_quantity=integer_value(_attribute(source, "can_use_volume", 0)),
        # 直接使用 SDK 明确提供的在途字段，不根据 total_quantity - available_quantity 推断。
        today_buy_quantity=max(0, on_road),
        market_value=decimal_value(_attribute(source, "market_value", 0)),
        turnover_rule=TurnoverRule.T1,
        captured_at=captured_at,
        frozen_quantity=integer_value(_attribute(source, "frozen_volume", 0)),
        on_road_quantity=on_road,
        yesterday_quantity=integer_value(_attribute(source, "yesterday_volume", 0)),
        average_cost=None if average_cost is None else decimal_value(average_cost),
    )


# 把 SDK 订单对象转换成统一订单快照，保留券商 ID 与备注关联。
def map_order(source: object, *, constants: ModuleType | Any | None = None) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        broker_order_id=str(_attribute(source, "order_id")),
        broker_order_sysid=str(_attribute(source, "order_sysid", "")) or None,
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        side=side_from_xt(
            _attribute(source, "order_type"),
            offset_flag=_attribute(source, "offset_flag"),
            constants=constants,
        ),
        requested_quantity=integer_value(_attribute(source, "order_volume", 0)),
        filled_quantity=integer_value(_attribute(source, "traded_volume", 0)),
        limit_price=decimal_value(_attribute(source, "price", 0)),
        traded_price=decimal_value(_attribute(source, "traded_price", 0)),
        status=order_status_from_xt(_attribute(source, "order_status")),
        remark_token=str(_attribute(source, "order_remark", "")) or None,
        captured_at=market_datetime(_attribute(source, "order_time"), unit="seconds"),
    )


# 把 SDK 成交对象转换成标准成交快照，供仓库去重入账。
def map_trade(source: object, *, constants: ModuleType | Any | None = None) -> BrokerTradeSnapshot:
    return BrokerTradeSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        broker_trade_id=str(_attribute(source, "traded_id")),
        broker_order_id=str(_attribute(source, "order_id")),
        broker_order_sysid=str(_attribute(source, "order_sysid", "")) or None,
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        side=side_from_xt(
            _attribute(source, "order_type"),
            offset_flag=_attribute(source, "offset_flag"),
            constants=constants,
        ),
        quantity=integer_value(_attribute(source, "traded_volume", 0)),
        price=decimal_value(_attribute(source, "traded_price", 0)),
        remark_token=str(_attribute(source, "order_remark", "")) or None,
        traded_at=market_datetime(_attribute(source, "traded_time"), unit="seconds"),
    )


# 将同步下单返回值解释为受理、拒绝或未知；未知结果不可直接重发。
def map_submit_result(value: object) -> SubmitOrderResult:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return SubmitOrderResult(SubmitOrderStatus.ACCEPTED, broker_order_id=str(value))
    if value == -1:
        return SubmitOrderResult(
            SubmitOrderStatus.REJECTED, error="MiniQMT order_stock returned -1"
        )
    return SubmitOrderResult(
        SubmitOrderStatus.UNKNOWN,
        error=f"MiniQMT order_stock returned unexpected result: {value!r}",
    )


# 从行情盘口数组中取得第一档可用价格。
def _first_price(values: object) -> Decimal | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return None
    value = decimal_value(values[0])
    return value if value > 0 else None


# 结合实时 tick 与证券资料生成 LiveQuote，包含最新价、买卖一档和涨跌停。
def map_quote(
    external_symbol: str,
    tick: Mapping[str, object],
    instrument: Mapping[str, object],
) -> LiveQuote:
    stock_status = integer_value(tick.get("stockStatus", 0))
    raw_instrument_status = instrument.get("InstrumentStatus", 0)
    instrument_status = str(raw_instrument_status).casefold()
    try:
        instrument_status_code = integer_value(raw_instrument_status)
    except (TypeError, ValueError):
        instrument_status_code = 0
    # XtQuant 250516 的实时 tick 和 SDK 自带示例在正常连续交易时使用状态 3；
    # 旧运行环境也观察到状态 5。IsTrading 在目标 MiniQMT 对正常 ETF 返回 False，
    # 因此停牌判断使用 tick 状态和 InstrumentStatus，不依赖该不稳定字段。
    tradable_tick_status = stock_status in {3, 5, 11, 12, 13, 18, 19, 22}
    suspended = (
        not tradable_tick_status
        or instrument_status_code >= 1
        or "停牌" in instrument_status
        or "suspend" in instrument_status
    )
    return LiveQuote(
        symbol=external_to_internal_symbol(external_symbol),
        last_price=decimal_value(tick.get("lastPrice", 0)),
        bid1=_first_price(tick.get("bidPrice")),
        ask1=_first_price(tick.get("askPrice")),
        lower_limit=_optional_positive(instrument.get("DownStopPrice")),
        upper_limit=_optional_positive(instrument.get("UpStopPrice")),
        price_tick=_optional_positive(instrument.get("PriceTick")) or Decimal("0.001"),
        suspended=suspended,
        quoted_at=market_datetime(tick["time"], unit="milliseconds"),
    )


# 把可选价格转换为正 Decimal；缺失或不可用价格保留为空。
def _optional_positive(value: object | None) -> Decimal | None:
    if value is None:
        return None
    converted = decimal_value(value)
    return converted if converted > 0 else None


__all__ = [
    "decimal_value",
    "external_to_internal_symbol",
    "internal_to_external_symbol",
    "map_asset",
    "map_order",
    "map_position",
    "map_quote",
    "map_submit_result",
    "map_trade",
    "market_datetime",
    "order_status_from_xt",
    "side_from_xt",
    "side_to_xt",
]
