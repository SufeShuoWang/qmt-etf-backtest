"""纯日频行情领域值对象。

原始 QMT 行情只用于执行和账户估值；独立的前复权视图是唯一向策略或模型公开的价格对象。
本模块中的对象都不负责筛选数据或打开 MySQL 连接。
"""

from __future__ import annotations


from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from types import MappingProxyType

from etf_backtest.validation import non_blank as _non_blank, plain_date as _plain_date
from etf_backtest.config.schema import (
    CALENDAR_POLICY,
    MARKET_TIMEZONE,
    normalize_index_code,
    normalize_symbol,
)


class Exchange(StrEnum):
    """受支持证券代码所代表的境内交易所。"""

    SSE = "SSE"
    SZSE = "SZSE"


class EtfCategory(StrEnum):
    """单次运行允许的两种 ETF 类别。"""

    DOMESTIC_STOCK_ETF = "DOMESTIC_STOCK_ETF"
    GOLD_ETF = "GOLD_ETF"


class TurnoverRule(StrEnum):
    """当日卖出可用性。"""

    T0 = "T0"
    T1 = "T1"


class PriceLimitSource(StrEnum):
    """执行所用法定日价格边界的来源。"""

    TUSHARE_EXPLICIT = "TUSHARE_EXPLICIT"
    DERIVED_RULE_FALLBACK = "DERIVED_RULE_FALLBACK"


# 校验行情 Decimal 字段的有限性与数值范围。
def _decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


# 严格检查整数字段，避免布尔值被当作数量。
def _strict_int(value: object, field_name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


# 检查开高低收价格关系和正值约束，拒绝内部矛盾的行情。
def _ohlc(
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
) -> None:
    for field_name, value in (
        ("open", open_price),
        ("high", high_price),
        ("low", low_price),
        ("close", close_price),
    ):
        _decimal(value, field_name, positive=True)
    if high_price < max(open_price, close_price, low_price):
        raise ValueError("high must be at least open, close and low")
    if low_price > min(open_price, close_price, high_price):
        raise ValueError("low must be at most open, close and high")


@dataclass(frozen=True, slots=True)
class FrameKey:
    """单个上交所日历收盘行情帧的身份信息。"""

    trade_date: date
    calendar_version: str
    calendar_policy: str = CALENDAR_POLICY

    # 校验交易日期、日历版本及固定 SSE 日历政策，确定日行情帧身份。
    def __post_init__(self) -> None:
        _plain_date(self.trade_date, "trade_date")
        object.__setattr__(
            self, "calendar_version", _non_blank(self.calendar_version, "calendar_version")
        )
        if self.calendar_policy != CALENDAR_POLICY:
            raise ValueError(f"calendar_policy must be {CALENDAR_POLICY}")

    @property
    def close_time(self) -> datetime:
        """返回固定的可观察、可执行日频收盘时间戳。"""

        return datetime.combine(self.trade_date, time(15, 0), tzinfo=MARKET_TIMEZONE)


@dataclass(frozen=True, slots=True)
class MarketBar:
    """用于执行和估值的单条未复权 QMT 日频行情。"""

    source_record_key: str  #
    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    pre_close: Decimal
    volume: int
    amount: Decimal
    suspended: bool
    price_limit_down: Decimal | None = None
    price_limit_up: Decimal | None = None
    price_limit_source: PriceLimitSource | None = None

    # 校验原始日行情的证券、价格、成交量和来源信息。
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_record_key",
            _non_blank(self.source_record_key, "source_record_key"),
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        _decimal(self.pre_close, "pre_close", positive=True)
        _strict_int(self.volume, "volume")
        _decimal(self.amount, "amount", non_negative=True)
        if not isinstance(self.suspended, bool):
            raise TypeError("suspended must be bool")
        lower = self.price_limit_down
        upper = self.price_limit_up
        if (lower is None) != (upper is None):
            raise ValueError("explicit price limits must be supplied as a complete pair")
        if lower is None:
            if self.price_limit_source is not None:
                raise ValueError("price_limit_source requires explicit price limits")
            return
        if self.price_limit_source is not PriceLimitSource.TUSHARE_EXPLICIT:
            raise ValueError("explicit price limits must use TUSHARE_EXPLICIT source")
        lower = _decimal(lower, "price_limit_down", positive=True)
        upper = _decimal(upper, "price_limit_up", positive=True)
        if not lower <= self.close <= upper:
            raise ValueError("raw close must stay inside explicit legal price limits")


@dataclass(frozen=True, slots=True)
class MarketBarView:
    """独立的前复权日频策略视图。"""

    source_record_key: str
    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    suspended: bool

    # 校验供策略使用的前复权行情视图及其来源身份。
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_record_key",
            _non_blank(self.source_record_key, "source_record_key"),
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        _strict_int(self.volume, "volume")
        if not isinstance(self.suspended, bool):
            raise TypeError("suspended must be bool")

    @property
    def signal_time(self) -> datetime:
        """返回该视图首次进入策略历史的时间戳。"""

        return datetime.combine(self.trade_date, time(15, 0), tzinfo=MARKET_TIMEZONE)


@dataclass(frozen=True, slots=True)
class IndexBarView:
    """仅 Rule 策略可见的单条数据源原生 PRICE 指数行情。"""

    index_code: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    pre_close: Decimal | None
    pct_change: Decimal | None
    source_system: str

    # 校验指数日行情代码、日期、价格和来源字段。
    def __post_init__(self) -> None:
        object.__setattr__(self, "index_code", normalize_index_code(self.index_code))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        if self.pre_close is not None:
            _decimal(self.pre_close, "pre_close", positive=True)
        if self.pct_change is not None:
            _decimal(self.pct_change, "pct_change")
        if _non_blank(self.source_system, "source_system") != "TUSHARE":
            raise ValueError("index source_system must be TUSHARE")


@dataclass(frozen=True, slots=True)
class MarketFrame:
    """单个上交所交易日的完整原始日频行情。"""

    frame_key: FrameKey
    bars_by_symbol: Mapping[str, MarketBar]

    # 检查同一行情帧内证券唯一且时间一致，并冻结行情映射。
    def __post_init__(self) -> None:
        if not isinstance(self.frame_key, FrameKey):
            raise TypeError("frame_key must be FrameKey")
        if not isinstance(self.bars_by_symbol, Mapping) or not self.bars_by_symbol:
            raise ValueError("bars_by_symbol must be a non-empty mapping")
        canonical: dict[str, MarketBar] = {}  # 行情数据
        source_keys: set[str] = set()  # 源记录键集合
        for supplied_symbol, bar in self.bars_by_symbol.items():
            if not isinstance(bar, MarketBar):
                raise TypeError("MarketFrame may contain only MarketBar")
            symbol = normalize_symbol(supplied_symbol)
            if bar.symbol != symbol:
                raise ValueError("bar mapping key must equal MarketBar.symbol")
            if bar.trade_date != self.frame_key.trade_date:
                raise ValueError("all bars must match the frame trade_date")
            if symbol in canonical or bar.source_record_key in source_keys:
                raise ValueError("duplicate symbol or source record in MarketFrame")
            canonical[symbol] = bar
            source_keys.add(bar.source_record_key)
        object.__setattr__(
            self, "bars_by_symbol", MappingProxyType(dict(sorted(canonical.items())))
        )

    @classmethod
    def from_bars(cls, frame_key: FrameKey, bars: Iterable[MarketBar]) -> MarketFrame:
        """根据原始行情构建确定性行情帧。"""

        indexed: dict[str, MarketBar] = {}
        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise TypeError("bars must contain MarketBar")
            if bar.symbol in indexed:
                raise ValueError(f"duplicate symbol in frame: {bar.symbol}")
            indexed[bar.symbol] = bar
        return cls(frame_key=frame_key, bars_by_symbol=indexed)

    # 从行情帧标识读取交易日期。
    @property
    def trade_date(self) -> date:
        return self.frame_key.trade_date

    # 返回当前行情帧中排序后的标准证券代码。
    @property
    def canonical_symbols(self) -> tuple[str, ...]:
        return tuple(self.bars_by_symbol)

    # 按标准证券代码取得该帧的原始行情。
    def bar_for(self, symbol: str) -> MarketBar:
        return self.bars_by_symbol[normalize_symbol(symbol)]


@dataclass(frozen=True, slots=True)
class EtfInfo:
    """用于解析运行证券范围的冻结当前 ``dim_etf`` 元数据。"""

    symbol: str
    exchange: Exchange
    name: str
    primary_category: str
    fund_type: str
    list_date: date
    delist_date: date | None
    current_status: str
    delist_date_approximated: bool = False

    # 校验 ETF 主信息中的代码、分类及上市／退市日期。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.exchange, Exchange):
            raise TypeError("exchange must be Exchange")
        object.__setattr__(self, "name", _non_blank(self.name, "name"))
        object.__setattr__(
            self,
            "primary_category",
            _non_blank(self.primary_category, "primary_category"),
        )
        object.__setattr__(self, "fund_type", _non_blank(self.fund_type, "fund_type"))
        list_date = _plain_date(self.list_date, "list_date")
        if self.delist_date is not None:
            delist_date = _plain_date(self.delist_date, "delist_date")
            if delist_date < list_date:
                raise ValueError("delist_date must not precede list_date")
        object.__setattr__(
            self, "current_status", _non_blank(self.current_status, "current_status")
        )
        if not isinstance(self.delist_date_approximated, bool):
            raise TypeError("delist_date_approximated must be bool")
        expected_exchange = Exchange.SSE if self.symbol.startswith("SH.") else Exchange.SZSE
        if self.exchange is not expected_exchange:
            raise ValueError("exchange does not match normalized symbol")

    def is_active(self, trade_date: date) -> bool:
        """返回冻结生命周期区间是否包含指定日期。"""

        value = _plain_date(trade_date, "trade_date")
        return self.list_date <= value and (self.delist_date is None or value <= self.delist_date)


@dataclass(frozen=True, slots=True)
class EtfTradingRule:
    """带生效日期的规则结果；类别与涨跌幅限制相互独立。"""

    symbol: str
    etf_category: EtfCategory
    turnover_rule: TurnoverRule
    price_limit_ratio: Decimal
    lot_size: int = 100
    tick_size: Decimal = Decimal("0.001")

    # 检查最小报价单位、交易整手、涨跌幅和 T+0/T+1 等交易参数。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.etf_category, EtfCategory):
            raise TypeError("etf_category must be EtfCategory")
        if not isinstance(self.turnover_rule, TurnoverRule):
            raise TypeError("turnover_rule must be TurnoverRule")
        ratio = _decimal(self.price_limit_ratio, "price_limit_ratio", positive=True)
        if ratio >= 1:
            raise ValueError("price_limit_ratio must be less than one")
        _strict_int(self.lot_size, "lot_size", positive=True)
        _decimal(self.tick_size, "tick_size", positive=True)
        expected_turnover = (
            TurnoverRule.T0 if self.etf_category is EtfCategory.GOLD_ETF else TurnoverRule.T1
        )
        if self.turnover_rule is not expected_turnover:
            raise ValueError("turnover rule conflicts with the supported ETF category")


def resolve_legal_price_limits(
    *,
    execution_bar: MarketBar,
    trading_rule: EtfTradingRule,
) -> tuple[Decimal, Decimal, PriceLimitSource]:
    """优先解析显式法定价格，再使用有效比例回退规则。"""

    if not isinstance(execution_bar, MarketBar):
        raise TypeError("execution_bar must be MarketBar")
    if not isinstance(trading_rule, EtfTradingRule):
        raise TypeError("trading_rule must be EtfTradingRule")
    if trading_rule.symbol != execution_bar.symbol:
        raise ValueError("trading rule symbol does not match the execution bar")

    lower = execution_bar.price_limit_down
    upper = execution_bar.price_limit_up
    if lower is not None and upper is not None:
        for value in (lower, upper):
            ticks = value / trading_rule.tick_size
            if ticks != ticks.to_integral_value():
                raise ValueError("explicit legal price limit is not aligned to the ETF tick")
        return lower, upper, PriceLimitSource.TUSHARE_EXPLICIT

    # 把涨跌停参考价按证券最小报价单位进行十进制取整。
    def round_to_tick(value: Decimal) -> Decimal:
        ticks = (value / trading_rule.tick_size).to_integral_value(rounding=ROUND_HALF_UP)
        rounded = ticks * trading_rule.tick_size
        if not rounded.is_finite() or rounded <= 0:
            raise ValueError("calculated legal price limit must be positive")
        return rounded

    ratio = trading_rule.price_limit_ratio
    return (
        round_to_tick(execution_bar.pre_close * (Decimal("1") - ratio)),
        round_to_tick(execution_bar.pre_close * (Decimal("1") + ratio)),
        PriceLimitSource.DERIVED_RULE_FALLBACK,
    )


__all__ = [
    "EtfCategory",
    "EtfInfo",
    "EtfTradingRule",
    "Exchange",
    "FrameKey",
    "IndexBarView",
    "MarketBar",
    "MarketBarView",
    "MarketFrame",
    "PriceLimitSource",
    "TurnoverRule",
    "resolve_legal_price_limits",
]
