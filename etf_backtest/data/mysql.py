"""统一 QMT 日频 MySQL 数据集的只读适配器。"""

from __future__ import annotations


import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Final

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.base import Executable

from etf_backtest.validation import non_blank as _non_blank, plain_date as _plain_date
from etf_backtest.config.schema import (
    MARKET_TIMEZONE,
    etf_code,
    normalize_index_code,
    normalize_symbol,
)
from etf_backtest.core.market import (
    EtfInfo,
    Exchange,
    FrameKey,
    IndexBarView,
    MarketBar,
    MarketBarView,
    MarketFrame,
    PriceLimitSource,
)

_QMT_SOURCE: Final = "QMT"
_SSE_SOURCE: Final = "AKSHARE_SINA"
_OPEN: Final = time(9, 30)
_MORNING_CLOSE: Final = time(11, 30)
_AFTERNOON_OPEN: Final = time(13)
_CLOSE: Final = time(15)
_TUSHARE_LIMIT_ROUNDING_TOLERANCE: Final = Decimal("0.005")
_SQL_IDENTIFIER: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
HUIJIN_ENTITIES: Final = (
    "中央汇金投资有限责任公司",
    "中央汇金资产管理有限责任公司",
)


class QmtDataQualityError(ValueError):
    """冻结的 QMT 数据切片无法满足日频回测契约。"""


# 检查 SQL 标识符格式，避免把任意文本拼入表名或列名。
def _sql_identifier(value: object, field_name: str) -> str:
    normalized = _non_blank(value, field_name)
    if _SQL_IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a plain MySQL identifier")
    return normalized


# 生成包含起止端点的自然日期序列，供日历完整性检查。
def _closed_dates(start_date: date, end_date: date) -> tuple[date, date]:
    start = _plain_date(start_date, "start_date")
    end = _plain_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must not follow end_date")
    return start, end


# 将数据库数值转换为满足约束的 Decimal。
def _decimal(value: object, field_name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


# 转换允许带正负号的数据库 Decimal 字段，同时拒绝无效数值。
def _signed_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return value


# 把数据库状态值转换为布尔值，拒绝无法识别的表示。
def _bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{field_name} must be bool or database 0/1")


# 从数据库 ETF 代码与市场信息构造内部标准证券代码。
def _canonical_symbol(etf_code: object, qmt_symbol: object, exchange: object) -> str:
    code = _non_blank(etf_code, "etf_code")
    if len(code) != 6 or not code.isdigit():
        raise QmtDataQualityError(f"invalid six-digit etf_code: {code!r}")
    exchange_name = _non_blank(exchange, "exchange")
    if exchange_name not in {Exchange.SSE.value, Exchange.SZSE.value}:
        raise QmtDataQualityError(f"unsupported ETF exchange: {exchange_name}")
    inferred = normalize_symbol(code)
    expected_prefix = "SH." if exchange_name == Exchange.SSE.value else "SZ."
    if not inferred.startswith(expected_prefix):
        raise QmtDataQualityError(f"etf_code {code} conflicts with master exchange {exchange_name}")
    if qmt_symbol is not None:
        normalized_qmt = normalize_symbol(_non_blank(qmt_symbol, "qmt_symbol"))
        if normalized_qmt != inferred:
            raise QmtDataQualityError(f"qmt_symbol {qmt_symbol!r} conflicts with etf_code {code}")
    return inferred


# 提取标准证券代码中的六位基金代码，供数据库查询使用。
def _code_for_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).partition(".")[2]


# 规范、去重并排序待查询的证券代码。
def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
        raise TypeError("symbols must be a sequence of strings")
    canonical = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
    if not canonical:
        raise ValueError("symbols must not be empty")
    return canonical


# 把交易日期与指定日内时刻组合成带市场时区的时间戳。
def _local_datetime(trade_date: date, local_time: time) -> datetime:
    return datetime.combine(trade_date, local_time, tzinfo=MARKET_TIMEZONE)


# 组合数据库行的来源主键，供行情与结果溯源。
def _source_record_key(*, symbol: str, trade_date: date, dataset_version: str) -> str:
    return f"QMT:{dataset_version}:{etf_code(symbol)}:{trade_date.isoformat()}"


# 为停牌缺口承接行情生成来源键，明确其引用了之前的数据。
def _carry_source_record_key(*, symbol: str, trade_date: date, dataset_version: str) -> str:
    return f"CARRY:{dataset_version}:{etf_code(symbol)}:{trade_date.isoformat()}"


@dataclass(frozen=True, slots=True)
class QmtEtfMasterRecord:
    """仅用于校验显式证券范围的当前 ETF 主表记录。"""

    symbol: str
    etf_code: str
    qmt_symbol: str
    exchange: Exchange
    name: str
    list_date: date
    delist_date: date | None
    current_status: str
    primary_category: str | None
    fund_type: str | None
    etf_type: str | None
    source_system: str


@dataclass(frozen=True, slots=True)
class QmtSseCalendarDay:
    """来自 ``dim_trading_calendar`` 且带数据源版本的单个上交所自然日。"""

    cal_date: date
    is_open: bool
    previous_open_date: date | None
    next_open_date: date | None
    source_system: str
    calendar_version: str

    # 检查 SSE 自然日记录的日期、开市状态及相邻交易日信息。
    def __post_init__(self) -> None:
        _plain_date(self.cal_date, "cal_date")
        if not isinstance(self.is_open, bool):
            raise TypeError("is_open must be bool")
        if self.previous_open_date is not None and self.previous_open_date >= self.cal_date:
            raise QmtDataQualityError("previous_open_date must precede cal_date")
        if self.next_open_date is not None and self.next_open_date <= self.cal_date:
            raise QmtDataQualityError("next_open_date must follow cal_date")


@dataclass(frozen=True, slots=True)
class QmtRawDailyBar:
    """未复权 QMT 日频记录及其无损字符串来源键。"""

    source_record_key: str
    symbol: str
    trade_date: date
    bar_start_time: datetime
    bar_end_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    pre_close: Decimal
    volume: int
    amount: Decimal
    source_system: str

    # 校验数据库原始日线记录及其来源字段。
    def __post_init__(self) -> None:
        _non_blank(self.source_record_key, "source_record_key")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        for field_name in ("open", "high", "low", "close", "pre_close"):
            _decimal(getattr(self, field_name), field_name, positive=True)
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise QmtDataQualityError("raw daily OHLC ordering is invalid")
        if self.high < self.low:
            raise QmtDataQualityError("raw daily high must not be below low")
        if isinstance(self.volume, bool) or not isinstance(self.volume, int) or self.volume < 0:
            raise ValueError("volume must be a non-negative integer number of shares")
        _decimal(self.amount, "amount")
        if self.source_system != _QMT_SOURCE:
            raise QmtDataQualityError("raw source_system must be QMT")

    # 将数据库原始日线记录转换为执行与估值使用的 MarketBar。
    def to_market_bar(
        self,
        *,
        suspended: bool,
        explicit_price_limit: QmtExplicitPriceLimit | None = None,
    ) -> MarketBar:
        return MarketBar(
            source_record_key=self.source_record_key,
            symbol=self.symbol,
            trade_date=self.trade_date,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            pre_close=self.pre_close,
            volume=self.volume,
            amount=self.amount,
            suspended=suspended,
            price_limit_down=(
                None if explicit_price_limit is None else explicit_price_limit.price_limit_down
            ),
            price_limit_up=(
                None if explicit_price_limit is None else explicit_price_limit.price_limit_up
            ),
            price_limit_source=(
                None if explicit_price_limit is None else PriceLimitSource.TUSHARE_EXPLICIT
            ),
        )


@dataclass(frozen=True, slots=True)
class QmtFrontDailyBar:
    """按 ETF 和交易日期索引的单条前复权 QMT 日频行情。"""

    source_record_key: str
    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    source_system: str

    # 校验数据库前复权日线记录，供策略历史视图使用。
    def __post_init__(self) -> None:
        _non_blank(self.source_record_key, "source_record_key")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        for field_name in ("open", "high", "low", "close"):
            _decimal(getattr(self, field_name), field_name, positive=True)
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise QmtDataQualityError("front daily OHLC ordering is invalid")
        if self.high < self.low:
            raise QmtDataQualityError("front daily high must not be below low")
        if self.source_system != _QMT_SOURCE:
            raise QmtDataQualityError("front source_system must be QMT")


@dataclass(frozen=True, slots=True)
class QmtTradeStatusRecord:
    """QMT 数据源原生状态（含未知状态）及可选法定价格上下限。"""

    symbol: str
    trade_date: date
    qmt_suspend_flag: int | None
    price_limit_down: Decimal | None
    price_limit_up: Decimal | None
    qmt_source_system: str | None
    price_limit_source_system: str | None

    # 校验交易状态记录中的日期、停牌及涨跌停字段。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        flag = self.qmt_suspend_flag
        if flag is not None and (isinstance(flag, bool) or flag not in (-1, 0, 1)):
            raise QmtDataQualityError("qmt_suspend_flag must be -1, 0, 1, or NULL")
        if flag is None:
            if self.qmt_source_system is not None:
                raise QmtDataQualityError("NULL qmt_suspend_flag requires NULL qmt_source_system")
        elif _non_blank(self.qmt_source_system, "qmt_source_system") != _QMT_SOURCE:
            raise QmtDataQualityError("qmt_source_system must be QMT")
        lower = self.price_limit_down
        upper = self.price_limit_up
        if (lower is None) != (upper is None):
            raise QmtDataQualityError(
                "explicit upper and lower price limits must both be present or both be NULL"
            )
        if lower is None:
            if self.price_limit_source_system is not None:
                raise QmtDataQualityError("NULL price limits require NULL source")
        else:
            lower = _decimal(lower, "down_limit_price_cny", positive=True)
            upper = _decimal(upper, "up_limit_price_cny", positive=True)
            if lower > upper:
                raise QmtDataQualityError("explicit lower price limit exceeds upper price limit")
            if _non_blank(self.price_limit_source_system, "price_limit_source_system") != "TUSHARE":
                raise QmtDataQualityError("explicit price-limit source_system must be TUSHARE")

    # 根据数据库交易状态记录判断是否停牌。
    @property
    def suspended(self) -> bool:
        return self.qmt_suspend_flag == 1

    # 从状态记录提取显式涨跌停价格；未提供时留给后续规则推导。
    def explicit_price_limit(self) -> QmtExplicitPriceLimit | None:
        if self.price_limit_down is None or self.price_limit_up is None:
            return None
        return QmtExplicitPriceLimit(
            symbol=self.symbol,
            trade_date=self.trade_date,
            price_limit_down=self.price_limit_down,
            price_limit_up=self.price_limit_up,
            source_system="TUSHARE",
        )


@dataclass(frozen=True, slots=True)
class QmtExplicitPriceLimit:
    """来自辅助日表的一组完整 Tushare 法定价格上下限。"""

    symbol: str
    trade_date: date
    price_limit_down: Decimal
    price_limit_up: Decimal
    source_system: str

    # 检查显式涨跌停价格为合法且有序的价格区间。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        lower = _decimal(self.price_limit_down, "down_limit_price_cny", positive=True)
        upper = _decimal(self.price_limit_up, "up_limit_price_cny", positive=True)
        if lower > upper:
            raise QmtDataQualityError("explicit lower price limit exceeds upper price limit")
        if _non_blank(self.source_system, "price_limit_source_system") != "TUSHARE":
            raise QmtDataQualityError("explicit price-limit source_system must be TUSHARE")


@dataclass(frozen=True, slots=True)
class QmtEtfShareRecord:
    """单条 ``etf_share_daily`` 业务日期记录。"""

    symbol: str
    asof_date: date
    total_share: Decimal
    source_system: str

    # 校验某只 ETF 在指定日期的精确份额观察值。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.asof_date, "asof_date")
        _decimal(self.total_share, "total_share")
        if _non_blank(self.source_system, "source_system") != "TUSHARE":
            raise QmtDataQualityError("ETF share source_system must be TUSHARE")


@dataclass(frozen=True, slots=True)
class HuijinHolderRatioRecord:
    """以 Decimal 小数表示的单个公司、ETF 和报告期 HolderOfListing 比例。"""

    symbol: str
    end_date: date
    entity: str
    ratio: Decimal

    # 校验汇金主体、报告期和持有比例；报告期与披露日的区别需在数据来源处确认。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.end_date, "end_date")
        entity = _non_blank(self.entity, "entity")
        if entity not in HUIJIN_ENTITIES:
            raise QmtDataQualityError("unsupported Huijin entity")
        ratio = _decimal(self.ratio, "ratio")
        if ratio > Decimal("1"):
            raise QmtDataQualityError("aggregated HolderOfListing ratio exceeds one")
        object.__setattr__(self, "entity", entity)


@dataclass(frozen=True, slots=True)
class QmtDailyFrame:
    """单个上交所开市日，原始记录和前复权记录分别存于不可变映射。"""

    trade_date: date
    bar_start_time: datetime
    bar_end_time: datetime
    raw_by_symbol: Mapping[str, QmtRawDailyBar]
    front_by_symbol: Mapping[str, QmtFrontDailyBar]
    trade_status_by_symbol: Mapping[str, QmtTradeStatusRecord] = field(default_factory=dict)
    price_limits_by_symbol: Mapping[str, QmtExplicitPriceLimit] = field(default_factory=dict)
    carried_symbols: frozenset[str] = frozenset()

    # 校验同一日数据库行情帧中原始与前复权记录的对应关系。
    def __post_init__(self) -> None:
        raw = dict(sorted(self.raw_by_symbol.items()))
        front = dict(sorted(self.front_by_symbol.items()))
        statuses = dict(sorted(self.trade_status_by_symbol.items()))
        limits = dict(sorted(self.price_limits_by_symbol.items()))
        carried = frozenset(normalize_symbol(symbol) for symbol in self.carried_symbols)
        if not raw:
            raise ValueError("QmtDailyFrame must contain at least one active symbol")
        if set(raw) != set(front):
            raise QmtDataQualityError("raw/front frame symbols are not one-to-one")
        for symbol, bar in raw.items():
            if normalize_symbol(symbol) != bar.symbol or bar.trade_date != self.trade_date:
                raise QmtDataQualityError("raw frame key/date mismatch")
            factor = front[symbol]
            if factor.symbol != bar.symbol or factor.trade_date != bar.trade_date:
                raise QmtDataQualityError("front frame key/date mismatch")
            if factor.source_record_key != bar.source_record_key:
                raise QmtDataQualityError("raw/front frame source_record_key mismatch")
        if not set(statuses).issubset(raw):
            raise QmtDataQualityError("trade status has no paired raw/front row")
        for symbol, status in statuses.items():
            if status.symbol != symbol or status.trade_date != self.trade_date:
                raise QmtDataQualityError("trade-status frame key/date mismatch")
        if not set(limits).issubset(raw):
            raise QmtDataQualityError("explicit price limit has no paired raw/front row")
        if not carried.issubset(raw):
            raise QmtDataQualityError("carried symbol has no paired raw/front row")
        for symbol, limit in limits.items():
            if (
                normalize_symbol(symbol) != limit.symbol
                or limit.symbol != raw[symbol].symbol
                or limit.trade_date != self.trade_date
            ):
                raise QmtDataQualityError("explicit price-limit frame key/date mismatch")
            for value in (limit.price_limit_down, limit.price_limit_up):
                ticks = value / Decimal("0.001")
                if ticks != ticks.to_integral_value():
                    raise QmtDataQualityError(
                        "explicit price limit is not aligned to the 0.001 ETF tick"
                    )
            if not limit.price_limit_down <= raw[symbol].close <= limit.price_limit_up:
                raise QmtDataQualityError("raw close is outside explicit legal price limits")
        object.__setattr__(self, "raw_by_symbol", MappingProxyType(raw))
        object.__setattr__(self, "front_by_symbol", MappingProxyType(front))
        object.__setattr__(self, "trade_status_by_symbol", MappingProxyType(statuses))
        object.__setattr__(self, "price_limits_by_symbol", MappingProxyType(limits))
        object.__setattr__(self, "carried_symbols", carried)

    # 将数据库日帧转换为回测执行用的 MarketFrame。
    def to_market_frame(self, *, calendar_version: str) -> MarketFrame:
        frame_key = FrameKey(
            trade_date=self.trade_date,
            calendar_version=calendar_version,
        )
        bars = (
            raw.to_market_bar(
                suspended=(
                    symbol in self.carried_symbols
                    or (
                        self.trade_status_by_symbol[symbol].suspended
                        if symbol in self.trade_status_by_symbol
                        else False
                    )
                ),
                explicit_price_limit=self.price_limits_by_symbol.get(symbol),
            )
            for symbol, raw in self.raw_by_symbol.items()
        )
        return MarketFrame.from_bars(frame_key, bars)


@dataclass(frozen=True, slots=True)
class QmtDailyDataset:
    """经完整预检并冻结的单次上交所日频运行数据契约。"""

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    dataset_version: str
    calendar_source: str
    calendar: tuple[QmtSseCalendarDay, ...]
    etf_master: tuple[QmtEtfMasterRecord, ...]
    etf_infos: tuple[EtfInfo, ...]
    frames: tuple[QmtDailyFrame, ...]
    share_records: tuple[QmtEtfShareRecord, ...] = ()
    huijin_ratio_records: tuple[HuijinHolderRatioRecord, ...] = ()
    index_records: tuple[IndexBarView, ...] = ()
    suspension_carry_keys: tuple[tuple[str, date], ...] = ()

    @property
    def explicit_price_limit_count(self) -> int:
        """具备完整显式法定价格上下限的执行行情数量。"""

        return sum(len(frame.price_limits_by_symbol) for frame in self.frames)

    @property
    def derived_price_limit_fallback_count(self) -> int:
        """必须使用有效比例回退规则的执行行情数量。"""

        return (
            sum(len(frame.raw_by_symbol) for frame in self.frames) - self.explicit_price_limit_count
        )

    def market_frames(self) -> tuple[MarketFrame, ...]:
        """返回带稳定数据集来源键的原始领域行情帧。"""

        frames = tuple(
            frame.to_market_frame(
                calendar_version=self.dataset_version,
            )
            for frame in self.frames
        )
        keys = [bar.source_record_key for frame in frames for bar in frame.bars_by_symbol.values()]
        if len(keys) != len(set(keys)):
            raise QmtDataQualityError("duplicate paired source_record_key")
        return frames

    def front_market_bar_views(self) -> tuple[MarketBarView, ...]:
        """返回与原始执行记录逐一配对的前复权策略视图。"""

        views: list[MarketBarView] = []
        for frame in self.frames:
            for symbol, raw in frame.raw_by_symbol.items():
                front = frame.front_by_symbol[symbol]
                status = frame.trade_status_by_symbol.get(symbol)
                views.append(
                    MarketBarView(
                        source_record_key=raw.source_record_key,
                        symbol=symbol,
                        trade_date=raw.trade_date,
                        open=front.open,
                        high=front.high,
                        low=front.low,
                        close=front.close,
                        volume=raw.volume,
                        suspended=(
                            symbol in frame.carried_symbols
                            or (status.suspended if status is not None else False)
                        ),
                    )
                )
        return tuple(views)


class QmtDailyRepository:
    """基于单个统一 QMT 数据库连接的只读仓储。"""

    # 绑定 SQLAlchemy 引擎、数据版本和数据来源设置；查询通过只读方法完成。
    def __init__(
        self,
        engine: Engine,
        *,
        connection: Connection | None = None,
        dataset_version: str,
        calendar_source: str = _SSE_SOURCE,
        trade_status_table: str = "etf_trade_status_daily",
        share_table: str | None = None,
        index_table: str | None = None,
        index_codes: Sequence[str] = (),
        huijin_holders_csv: str | Path | None = None,
        huijin_holders_csv_sha256: str | None = None,
    ) -> None:
        self._engine = engine
        self._connection = connection
        self._dataset_version = _non_blank(dataset_version, "dataset_version")
        self._calendar_source = _non_blank(calendar_source, "calendar_source")
        self._trade_status_table = _sql_identifier(trade_status_table, "trade_status_table")
        self._share_table = (
            None if share_table is None else _sql_identifier(share_table, "share_table")
        )
        self._index_table = (
            None if index_table is None else _sql_identifier(index_table, "index_table")
        )
        if isinstance(index_codes, (str, bytes)) or not isinstance(index_codes, Sequence):
            raise TypeError("index_codes must be a sequence")
        self._index_codes = tuple(sorted({normalize_index_code(code) for code in index_codes}))
        self._huijin_holders_csv = (
            None if huijin_holders_csv is None else Path(huijin_holders_csv).resolve(strict=True)
        )
        if self._huijin_holders_csv is None:
            if huijin_holders_csv_sha256 is not None:
                raise ValueError("Huijin CSV hash requires a configured CSV path")
            self._huijin_holders_csv_sha256 = None
        else:
            if not isinstance(huijin_holders_csv_sha256, str):
                raise TypeError("huijin_holders_csv_sha256 must be a string")
            digest = huijin_holders_csv_sha256.strip().lower()
            if _SHA256.fullmatch(digest) is None:
                raise ValueError("huijin_holders_csv_sha256 must be a lowercase SHA-256 digest")
            self._huijin_holders_csv_sha256 = digest

    # 返回仓库当前使用的数据集版本。
    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # 返回仓库读取 SSE 日历时使用的来源标识。
    @property
    def calendar_source(self) -> str:
        return self._calendar_source

    # 执行参数化 SQL 查询并返回映射行，统一管理查询连接。
    def _fetch(
        self, statement: Executable, parameters: Mapping[str, object]
    ) -> Sequence[Mapping[str, object]]:
        # 借用已有连接时不关闭它；查询结果在连接退出前全部取回。
        scope = (
            nullcontext(self._connection)
            if self._connection is not None
            else self._engine.connect()
        )
        with scope as connection:
            result = connection.execute(statement, dict(parameters))
            return result.mappings().all()

    # 按显式证券读取 ETF 当前主表记录，供资产池冻结和生命周期解析。
    def load_etf_master(self, symbols: Sequence[str]) -> tuple[QmtEtfMasterRecord, ...]:
        requested = _normalize_symbols(symbols)
        requested_by_code = {_code_for_symbol(symbol): symbol for symbol in requested}
        statement = text(
            """
            SELECT etf_code, qmt_symbol, exchange, fund_name, list_date, delist_date,
                   current_status, primary_category, fund_type, etf_type, source_system
            FROM dim_etf
            WHERE etf_code IN :etf_codes
            ORDER BY etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        records: dict[str, QmtEtfMasterRecord] = {}
        for row in self._fetch(statement, {"etf_codes": sorted(requested_by_code)}):
            raw_code = _non_blank(row["etf_code"], "etf_code")
            if raw_code not in requested_by_code:
                raise QmtDataQualityError(f"unexpected dim_etf code: {raw_code}")
            record = self._master_record_from_row(row)
            symbol = record.symbol
            if symbol != requested_by_code[raw_code]:
                raise QmtDataQualityError(f"dim_etf symbol mismatch for {raw_code}")
            if symbol in records:
                raise QmtDataQualityError(f"duplicate dim_etf row for {symbol}")
            records[symbol] = record
        missing = tuple(sorted(set(requested) - set(records)))
        if missing:
            raise QmtDataQualityError(f"missing dim_etf rows: {missing!r}")
        return tuple(records[symbol] for symbol in requested)

    def load_pool_etf_master(self, pool_name: str) -> tuple[QmtEtfMasterRecord, ...]:
        """解析当前主表中的单个受支持资产池，不使用状态过滤。"""

        pool = _non_blank(pool_name, "pool_name")
        conditions = {
            "domestic_stock_etf": "primary_category = '纯境内' AND fund_type = '股票型'",
            "gold_etf": "primary_category = '\u7eaf\u5883\u5185' AND LEFT(etf_code, 3) = '518'",
            "all_supported_etf": (
                "((primary_category = '纯境内' AND fund_type = '股票型') "
                "OR (primary_category = '\u7eaf\u5883\u5185' AND LEFT(etf_code, 3) = '518'))"
            ),
        }
        try:
            condition = conditions[pool]
        except KeyError:
            raise ValueError(f"unsupported ETF pool: {pool}") from None
        rows = self._fetch(
            text(
                f"""
                SELECT etf_code, qmt_symbol, exchange, fund_name, list_date, delist_date,
                       current_status, primary_category, fund_type, etf_type, source_system
                FROM dim_etf
                WHERE {condition}
                ORDER BY etf_code
                """
            ),
            {},
        )
        indexed: dict[str, QmtEtfMasterRecord] = {}
        for row in rows:
            record = self._master_record_from_row(row)
            if record.symbol in indexed:
                raise QmtDataQualityError(f"duplicate pool master row for {record.symbol}")
            indexed[record.symbol] = record
        return tuple(indexed[symbol] for symbol in sorted(indexed))

    def load_last_raw_trade_dates(self, symbols: Sequence[str]) -> Mapping[str, date]:
        """加载统一数据集中的最后一个原始业务日期。"""

        requested = _normalize_symbols(symbols)
        codes = [_code_for_symbol(symbol) for symbol in requested]
        requested_by_code = dict(zip(codes, requested, strict=True))
        statement = text(
            """
            SELECT etf_code, MAX(trade_date) AS last_trade_date
            FROM etf_quote_qmt_unadjusted_daily
            WHERE etf_code IN :etf_codes
              AND source_system = 'QMT'
            GROUP BY etf_code
            ORDER BY etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        result: dict[str, date] = {}
        for row in self._fetch(
            statement,
            {"etf_codes": codes},
        ):
            code = _non_blank(row["etf_code"], "etf_code")
            symbol = requested_by_code.get(code)
            if symbol is None:
                raise QmtDataQualityError(f"unexpected last-date etf_code: {code}")
            if symbol in result:
                raise QmtDataQualityError(f"duplicate last-date row for {symbol}")
            result[symbol] = _plain_date(row["last_trade_date"], "last_trade_date")
        return MappingProxyType(dict(sorted(result.items())))


    # 将 ETF 主表查询行转换为统一的主信息记录。
    @classmethod
    def _master_record_from_row(cls, row: Mapping[str, object]) -> QmtEtfMasterRecord:
        raw_code = _non_blank(row["etf_code"], "etf_code")
        symbol = _canonical_symbol(row["etf_code"], row["qmt_symbol"], row["exchange"])
        list_date = _plain_date(row["list_date"], "list_date")
        raw_delist = row["delist_date"]
        delist_date = None if raw_delist is None else _plain_date(raw_delist, "delist_date")
        if delist_date is not None and delist_date < list_date:
            raise QmtDataQualityError(f"delist date precedes list date for {symbol}")
        return QmtEtfMasterRecord(
            symbol=symbol,
            etf_code=raw_code,
            qmt_symbol=_non_blank(row["qmt_symbol"], "qmt_symbol"),
            exchange=Exchange(_non_blank(row["exchange"], "exchange")),
            name=_non_blank(row["fund_name"], "fund_name"),
            list_date=list_date,
            delist_date=delist_date,
            current_status=_non_blank(row["current_status"], "current_status"),
            primary_category=cls._optional_string(row["primary_category"]),
            fund_type=cls._optional_string(row["fund_type"]),
            etf_type=cls._optional_string(row["etf_type"]),
            source_system=_non_blank(row["source_system"], "source_system"),
        )

    # 读取指定日期区间的 SSE 日历，并校验自然日覆盖及交易日关联。
    def load_sse_calendar(self, start_date: date, end_date: date) -> tuple[QmtSseCalendarDay, ...]:
        start, end = _closed_dates(start_date, end_date)
        rows = self._fetch(
            text(
                """
                SELECT exchange, cal_date, is_open, previous_open_date, next_open_date,
                       source_system
                FROM dim_trading_calendar
                WHERE exchange = 'SSE'
                  AND cal_date BETWEEN :start_date AND :end_date
                ORDER BY cal_date
                """
            ),
            {"start_date": start, "end_date": end},
        )
        days: list[QmtSseCalendarDay] = []
        seen: set[date] = set()
        for row in rows:
            if _non_blank(row["exchange"], "exchange") != Exchange.SSE.value:
                raise QmtDataQualityError("calendar query returned a non-SSE row")
            cal_date = _plain_date(row["cal_date"], "cal_date")
            if cal_date in seen:
                raise QmtDataQualityError(f"duplicate SSE calendar date: {cal_date}")
            seen.add(cal_date)
            is_open = _bool(row["is_open"], "is_open")
            source = _non_blank(row["source_system"], "source_system")
            if source != self._calendar_source:
                raise QmtDataQualityError(
                    f"calendar source {source!r} does not match {self._calendar_source!r}"
                )
            days.append(
                QmtSseCalendarDay(
                    cal_date=cal_date,
                    is_open=is_open,
                    previous_open_date=self._optional_date(
                        row["previous_open_date"], "previous_open_date"
                    ),
                    next_open_date=self._optional_date(row["next_open_date"], "next_open_date"),
                    source_system=source,
                    calendar_version=self._dataset_version,
                )
            )
        expected = tuple(
            date.fromordinal(ordinal) for ordinal in range(start.toordinal(), end.toordinal() + 1)
        )
        actual = tuple(day.cal_date for day in days)
        if actual != expected:
            missing = tuple(item for item in expected if item not in seen)
            raise QmtDataQualityError(f"incomplete SSE natural-date coverage: {missing!r}")
        self._validate_calendar_links(days)
        return tuple(days)


    def load_etf_share_records(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> tuple[QmtEtfShareRecord, ...]:
        """加载精确的每日总份额观察值，不进行前向填充。"""

        if self._share_table is None:
            return ()
        requested = _normalize_symbols(symbols)
        start, end = _closed_dates(start_date, end_date)
        codes = [_code_for_symbol(symbol) for symbol in requested]
        statement = text(
            f"""
            SELECT etf_code, asof_date, total_share, source_system
            FROM {self._share_table}
            WHERE etf_code IN :etf_codes
              AND asof_date BETWEEN :start_date AND :end_date
              AND total_share IS NOT NULL
              AND source_system = 'TUSHARE'
            ORDER BY asof_date, etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        rows = self._fetch(
            statement,
            {
                "etf_codes": codes,
                "start_date": start,
                "end_date": end,
            },
        )
        requested_by_code = dict(zip(codes, requested, strict=True))
        records: list[QmtEtfShareRecord] = []
        selected_keys: set[tuple[str, date]] = set()
        for row in rows:
            code = _non_blank(row["etf_code"], "etf_code")
            symbol = requested_by_code.get(code)
            if symbol is None:
                raise QmtDataQualityError(f"unexpected share etf_code: {code}")
            asof_date = _plain_date(row["asof_date"], "asof_date")
            key = (symbol, asof_date)
            if key in selected_keys:
                raise QmtDataQualityError(f"duplicate selected share business key: {key!r}")
            selected_keys.add(key)
            records.append(
                QmtEtfShareRecord(
                    symbol=symbol,
                    asof_date=asof_date,
                    total_share=_decimal(row["total_share"], "total_share"),
                    source_system=_non_blank(row["source_system"], "source_system"),
                )
            )
        return tuple(records)

    def load_index_records(
        self,
        index_codes: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> tuple[IndexBarView, ...]:
        """加载供 Rule 使用的已配置 PRICE 指数行情，不执行 ETF 代码规范化。"""

        if self._index_table is None:
            return ()
        if isinstance(index_codes, (str, bytes)) or not isinstance(index_codes, Sequence):
            raise TypeError("index_codes must be a sequence")
        requested = tuple(sorted({normalize_index_code(code) for code in index_codes}))
        if not requested:
            return ()
        start, end = _closed_dates(start_date, end_date)
        statement = text(
            f"""
            SELECT index_code, trade_date, price_series_type,
                   open_value, high_value, low_value, close_value,
                   pre_close_value, pct_chg, source_system
            FROM {self._index_table}
            WHERE index_code IN :index_codes
              AND trade_date BETWEEN :start_date AND :end_date
              AND price_series_type = 'PRICE'
              AND source_system = 'TUSHARE'
            ORDER BY trade_date, index_code
            """
        ).bindparams(bindparam("index_codes", expanding=True))
        rows = self._fetch(
            statement,
            {"index_codes": list(requested), "start_date": start, "end_date": end},
        )
        records: list[IndexBarView] = []
        seen: set[tuple[str, date]] = set()
        for row in rows:
            index_code = normalize_index_code(_non_blank(row["index_code"], "index_code"))
            if index_code not in requested:
                raise QmtDataQualityError(f"unexpected index_code: {index_code}")
            trade_date = _plain_date(row["trade_date"], "trade_date")
            key = (index_code, trade_date)
            if key in seen:
                raise QmtDataQualityError(f"duplicate selected index business key: {key!r}")
            seen.add(key)
            if _non_blank(row["price_series_type"], "price_series_type") != "PRICE":
                raise QmtDataQualityError("index price_series_type must be PRICE")
            records.append(
                IndexBarView(
                    index_code=index_code,
                    trade_date=trade_date,
                    open=_decimal(row["open_value"], "open_value", positive=True),
                    high=_decimal(row["high_value"], "high_value", positive=True),
                    low=_decimal(row["low_value"], "low_value", positive=True),
                    close=_decimal(row["close_value"], "close_value", positive=True),
                    pre_close=(
                        None
                        if row["pre_close_value"] is None
                        else _decimal(row["pre_close_value"], "pre_close_value", positive=True)
                    ),
                    pct_change=(
                        None
                        if row["pct_chg"] is None
                        else _signed_decimal(row["pct_chg"], "pct_chg")
                    ),
                    source_system=_non_blank(row["source_system"], "source_system"),
                )
            )
        return tuple(records)

    def load_huijin_ratio_records(
        self,
        symbols: Sequence[str],
    ) -> tuple[HuijinHolderRatioRecord, ...]:
        """读取并汇总两个已配置汇金主体的 HolderOfListing 百分比。"""

        if self._huijin_holders_csv is None:
            return ()
        assert self._huijin_holders_csv_sha256 is not None
        content = self._huijin_holders_csv.read_bytes()
        actual_digest = hashlib.sha256(content).hexdigest()
        if actual_digest != self._huijin_holders_csv_sha256:
            raise QmtDataQualityError("Huijin holders CSV SHA-256 mismatch")
        requested = _normalize_symbols(symbols)
        requested_by_code = {_code_for_symbol(symbol): symbol for symbol in requested}
        aggregated: dict[tuple[str, date, str], Decimal] = {}
        with self._huijin_holders_csv.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            for raw_row in reader:
                row = {
                    str(key).lstrip("\ufeff"): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                code = _non_blank(row.get("Symbol"), "Symbol")
                symbol = requested_by_code.get(code)
                if symbol is None:
                    continue
                entity = _non_blank(row.get("HuijinEntity"), "HuijinEntity")
                if entity not in HUIJIN_ENTITIES:
                    continue
                try:
                    end_date = date.fromisoformat(_non_blank(row.get("EndDate"), "EndDate"))
                except ValueError as exc:
                    raise QmtDataQualityError("invalid Huijin EndDate") from exc
                percentage_text = _non_blank(row.get("HolderOfListing"), "HolderOfListing")
                try:
                    percentage = Decimal(percentage_text)
                except InvalidOperation as exc:
                    raise QmtDataQualityError("invalid HolderOfListing decimal") from exc
                if not percentage.is_finite() or percentage < 0:
                    raise QmtDataQualityError("HolderOfListing must be finite and non-negative")
                key = (symbol, end_date, entity)
                aggregated[key] = aggregated.get(key, Decimal("0")) + percentage / Decimal("100")
        return tuple(
            HuijinHolderRatioRecord(
                symbol=symbol,
                end_date=end_date,
                entity=entity,
                ratio=ratio,
            )
            for (symbol, end_date, entity), ratio in sorted(aggregated.items())
        )

    # 数据准备主入口：读取日历、成对日线与状态，处理允许的孤立停牌缺口并核对生命周期覆盖，返回冻结数据集。
    def load_daily_dataset(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
        *,
        etf_infos: Sequence[EtfInfo] | None = None,
    ) -> QmtDailyDataset:
        """加载并预检完整的上交所日频数据切片。

        原始记录和前复权记录均按统一业务键直接读取，两类业务键必须一一对应，并精确覆盖
        每只请求证券在其闭区间上市期内的每个开市日。
        """

        requested = _normalize_symbols(symbols)
        start, end = _closed_dates(start_date, end_date)
        master = self.load_etf_master(requested)
        lifecycle_infos = (
            self._infos_from_master(master)
            if etf_infos is None
            else self._validate_lifecycle_infos(requested, master, etf_infos)
        )
        calendar = self.load_sse_calendar(start, end)
        open_calendar_days = tuple(day for day in calendar if day.is_open)
        query_start = start
        query_end = end
        if open_calendar_days:
            previous_open = open_calendar_days[0].previous_open_date
            next_open = open_calendar_days[-1].next_open_date
            if previous_open is not None:
                query_start = min(query_start, previous_open)
            if next_open is not None:
                query_end = max(query_end, next_open)
        buffered_raw_by_key, buffered_incomplete_raw_keys = self._raw_records(
            requested, query_start, query_end
        )
        buffered_front_by_key = self._front_records(requested, query_start, query_end)
        raw_by_key = {
            key: value for key, value in buffered_raw_by_key.items() if start <= key[1] <= end
        }
        front_by_key = {
            key: value for key, value in buffered_front_by_key.items() if start <= key[1] <= end
        }
        incomplete_raw_keys = {
            key for key in buffered_incomplete_raw_keys if start <= key[1] <= end
        }
        status_by_key = self._trade_status_records(requested, start, end)
        price_limits_by_key = {
            key: limit
            for key, status in status_by_key.items()
            if (limit := status.explicit_price_limit()) is not None
        }
        share_records = self.load_etf_share_records(requested, start, end)
        huijin_ratio_records = self.load_huijin_ratio_records(requested)
        index_records = self.load_index_records(self._index_codes, start, end)
        raw_keys = set(raw_by_key)
        front_keys = set(front_by_key)
        carry_candidates = {
            key
            for key, status in status_by_key.items()
            if status.qmt_suspend_flag in (1, None)
            and key not in raw_keys
            and key not in front_keys
            and key not in incomplete_raw_keys
        }
        raw_only = raw_keys - front_keys - carry_candidates
        front_only = front_keys - raw_keys - carry_candidates
        if raw_only or front_only:
            raise QmtDataQualityError(
                "raw/front one-to-one conflict; "
                f"raw_only={tuple(sorted(raw_only))!r}, front_only={tuple(sorted(front_only))!r}"
            )
        price_limits_by_key = self._reconcile_explicit_price_limit_rounding(
            raw_by_key=raw_by_key,
            price_limits_by_key=price_limits_by_key,
        )
        raw_by_key, front_by_key, suspension_carry_keys = (
            self._materialize_isolated_suspension_carries(
                raw_by_key=buffered_raw_by_key,
                front_by_key=buffered_front_by_key,
                carry_keys=tuple(sorted(carry_candidates)),
                calendar=calendar,
                dataset_version=self._dataset_version,
            )
        )
        raw_by_key = {key: value for key, value in raw_by_key.items() if start <= key[1] <= end}
        front_by_key = {key: value for key, value in front_by_key.items() if start <= key[1] <= end}
        raw_keys = set(raw_by_key)
        front_keys = set(front_by_key)
        info_by_symbol = {record.symbol: record for record in lifecycle_infos}
        open_dates = {day.cal_date for day in calendar if day.is_open}
        expected_keys = {
            (symbol, trade_date)
            for symbol, info in info_by_symbol.items()
            for trade_date in open_dates
            if info.list_date <= trade_date
            and (info.delist_date is None or trade_date <= info.delist_date)
        }
        missing = tuple(sorted(expected_keys - raw_keys))
        unexpected = tuple(sorted(raw_keys - expected_keys))
        if missing or unexpected:
            raise QmtDataQualityError(
                f"daily listing/calendar coverage conflict; missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )

        raw_source_keys: set[str] = set()
        front_source_keys: set[str] = set()
        by_date: dict[date, tuple[dict[str, QmtRawDailyBar], dict[str, QmtFrontDailyBar]]] = {}
        for key in sorted(raw_keys, key=lambda item: (item[1], item[0])):
            raw = raw_by_key[key]
            front = front_by_key[key]
            if raw.source_record_key in raw_source_keys:
                raise QmtDataQualityError(f"duplicate raw source key: {raw.source_record_key}")
            if front.source_record_key in front_source_keys:
                raise QmtDataQualityError(f"duplicate front source key: {front.source_record_key}")
            raw_source_keys.add(raw.source_record_key)
            front_source_keys.add(front.source_record_key)
            self._validate_raw_front_pair(raw, front)
            if raw.source_record_key != front.source_record_key:
                raise QmtDataQualityError("raw/front source_record_key mismatch")
            raw_frame, front_frame = by_date.setdefault(key[1], ({}, {}))
            raw_frame[key[0]] = raw
            front_frame[key[0]] = front

        frames = tuple(
            QmtDailyFrame(
                trade_date=trade_date,
                bar_start_time=_local_datetime(trade_date, _OPEN),
                bar_end_time=_local_datetime(trade_date, _CLOSE),
                raw_by_symbol=raw_frame,
                front_by_symbol=front_frame,
                trade_status_by_symbol={
                    symbol: status_by_key[(symbol, trade_date)]
                    for symbol in raw_frame
                    if (symbol, trade_date) in status_by_key
                },
                price_limits_by_symbol={
                    symbol: price_limits_by_key[(symbol, trade_date)]
                    for symbol in raw_frame
                    if (symbol, trade_date) in price_limits_by_key
                },
                carried_symbols=frozenset(
                    symbol for symbol in raw_frame if (symbol, trade_date) in suspension_carry_keys
                ),
            )
            for trade_date, (raw_frame, front_frame) in sorted(by_date.items())
        )
        return QmtDailyDataset(
            symbols=requested,
            start_date=start,
            end_date=end,
            dataset_version=self._dataset_version,
            calendar_source=self._calendar_source,
            calendar=calendar,
            etf_master=master,
            etf_infos=lifecycle_infos,
            frames=frames,
            share_records=share_records,
            huijin_ratio_records=huijin_ratio_records,
            index_records=index_records,
            suspension_carry_keys=suspension_carry_keys,
        )

    @staticmethod
    def _reconcile_explicit_price_limit_rounding(
        *,
        raw_by_key: Mapping[tuple[str, date], QmtRawDailyBar],
        price_limits_by_key: Mapping[tuple[str, date], QmtExplicitPriceLimit],
    ) -> dict[tuple[str, date], QmtExplicitPriceLimit]:
        """仅在 QMT 收盘价触及按分取整的 Tushare 边界时扩展该边界。

        部分历史 Tushare 涨跌停价按人民币 0.01 元取整，而 QMT ETF 行情保留法定的
        0.001 元最小变动单位。仅当收盘价同时为当日最高价或最低价，且差异不超过半分时
        才进行协调；差异更大或并非边界冲突时，行情帧预检仍会失败。
        """

        reconciled = dict(price_limits_by_key)
        for key, limit in price_limits_by_key.items():
            raw = raw_by_key.get(key)
            if raw is None:
                continue
            lower = limit.price_limit_down
            upper = limit.price_limit_up
            if (
                raw.close > upper
                and raw.close == raw.high
                and raw.close - upper <= _TUSHARE_LIMIT_ROUNDING_TOLERANCE
            ):
                upper = raw.close
            if (
                raw.close < lower
                and raw.close == raw.low
                and lower - raw.close <= _TUSHARE_LIMIT_ROUNDING_TOLERANCE
            ):
                lower = raw.close
            if lower != limit.price_limit_down or upper != limit.price_limit_up:
                reconciled[key] = replace(
                    limit,
                    price_limit_down=lower,
                    price_limit_up=upper,
                )
        return reconciled

    @staticmethod
    def _materialize_isolated_suspension_carries(
        *,
        raw_by_key: Mapping[tuple[str, date], QmtRawDailyBar],
        front_by_key: Mapping[tuple[str, date], QmtFrontDailyBar],
        carry_keys: Sequence[tuple[str, date]],
        calendar: Sequence[QmtSseCalendarDay],
        dataset_version: str,
    ) -> tuple[
        dict[tuple[str, date], QmtRawDailyBar],
        dict[tuple[str, date], QmtFrontDailyBar],
        tuple[tuple[str, date], ...],
    ]:
        """对已记录状态的孤立数据源缺口，分别延续原始价和前复权价。"""

        raw = dict(raw_by_key)
        front = dict(front_by_key)
        days = {day.cal_date: day for day in calendar}
        carried: list[tuple[str, date]] = []
        for symbol, trade_date in sorted(carry_keys):
            key = (symbol, trade_date)
            if key in raw or key in front:
                continue
            day = days.get(trade_date)
            if (
                day is None
                or not day.is_open
                or day.previous_open_date is None
                or day.next_open_date is None
            ):
                continue
            previous_key = (symbol, day.previous_open_date)
            next_key = (symbol, day.next_open_date)
            if (
                previous_key not in raw
                or previous_key not in front
                or next_key not in raw
                or next_key not in front
            ):
                continue
            previous_raw = raw[previous_key]
            previous_front = front[previous_key]
            source_key = _carry_source_record_key(
                symbol=symbol,
                trade_date=trade_date,
                dataset_version=dataset_version,
            )
            raw[(symbol, trade_date)] = replace(
                previous_raw,
                source_record_key=source_key,
                trade_date=trade_date,
                bar_start_time=_local_datetime(trade_date, _OPEN),
                bar_end_time=_local_datetime(trade_date, _CLOSE),
                open=previous_raw.close,
                high=previous_raw.close,
                low=previous_raw.close,
                pre_close=previous_raw.close,
                volume=0,
                amount=Decimal("0"),
            )
            front[(symbol, trade_date)] = replace(
                previous_front,
                source_record_key=source_key,
                trade_date=trade_date,
                open=previous_front.close,
                high=previous_front.close,
                low=previous_front.close,
            )
            carried.append((symbol, trade_date))
        return raw, front, tuple(carried)


    # 将原始日线查询结果转换为按证券、日期索引的记录，并检查重复。
    def _raw_records(
        self, symbols: tuple[str, ...], start_date: date, end_date: date
    ) -> tuple[dict[tuple[str, date], QmtRawDailyBar], set[tuple[str, date]]]:
        codes = [_code_for_symbol(symbol) for symbol in symbols]
        statement = text(
            """
            SELECT etf_code, trade_date,
                   open_price_cny, high_price_cny, low_price_cny,
                   close_price_cny, pre_close_price_cny,
                   volume_share, amount_cny, source_system
            FROM etf_quote_qmt_unadjusted_daily
            WHERE etf_code IN :etf_codes
              AND trade_date BETWEEN :start_date AND :end_date
              AND source_system = 'QMT'
            ORDER BY trade_date, etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        rows = self._fetch(
            statement,
            {
                "etf_codes": codes,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        requested_by_code = dict(zip(codes, symbols, strict=True))
        indexed: dict[tuple[str, date], QmtRawDailyBar] = {}
        incomplete: set[tuple[str, date]] = set()
        for row in rows:
            code = _non_blank(row["etf_code"], "etf_code")
            symbol = requested_by_code.get(code)
            if symbol is None:
                raise QmtDataQualityError(f"unexpected raw etf_code: {code}")
            trade_date = _plain_date(row["trade_date"], "trade_date")
            key = (symbol, trade_date)
            if key in indexed or key in incomplete:
                raise QmtDataQualityError(f"duplicate raw business key: {key!r}")
            required_fields = (
                "open_price_cny",
                "high_price_cny",
                "low_price_cny",
                "close_price_cny",
                "pre_close_price_cny",
                "volume_share",
                "amount_cny",
            )
            if any(row[field_name] is None for field_name in required_fields):
                incomplete.add(key)
                continue
            volume_decimal = _decimal(row["volume_share"], "volume_share")
            if volume_decimal != volume_decimal.to_integral_value():
                raise QmtDataQualityError(
                    f"volume_share cannot be converted losslessly to shares for {key!r}"
                )
            indexed[key] = QmtRawDailyBar(
                source_record_key=_source_record_key(
                    symbol=symbol,
                    trade_date=trade_date,
                    dataset_version=self._dataset_version,
                ),
                symbol=symbol,
                trade_date=trade_date,
                bar_start_time=_local_datetime(trade_date, _OPEN),
                bar_end_time=_local_datetime(trade_date, _CLOSE),
                open=_decimal(row["open_price_cny"], "open_price_cny", positive=True),
                high=_decimal(row["high_price_cny"], "high_price_cny", positive=True),
                low=_decimal(row["low_price_cny"], "low_price_cny", positive=True),
                close=_decimal(row["close_price_cny"], "close_price_cny", positive=True),
                pre_close=_decimal(
                    row["pre_close_price_cny"], "pre_close_price_cny", positive=True
                ),
                volume=int(volume_decimal),
                amount=_decimal(row["amount_cny"], "amount_cny"),
                source_system=_non_blank(row["source_system"], "source_system"),
            )
        return indexed, incomplete

    # 将交易状态查询结果转换为索引记录，供停牌与显式价格限制处理。
    def _trade_status_records(
        self, symbols: tuple[str, ...], start_date: date, end_date: date
    ) -> dict[tuple[str, date], QmtTradeStatusRecord]:
        codes = [_code_for_symbol(symbol) for symbol in symbols]
        requested_by_code = dict(zip(codes, symbols, strict=True))
        statement = text(
            f"""
            SELECT etf_code, trade_date, qmt_suspend_flag,
                   up_limit_price_cny, down_limit_price_cny,
                   qmt_source_system, price_limit_source_system
            FROM {self._trade_status_table}
            WHERE etf_code IN :etf_codes
              AND trade_date BETWEEN :start_date AND :end_date
            ORDER BY trade_date, etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        rows = self._fetch(
            statement,
            {
                "etf_codes": codes,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        indexed: dict[tuple[str, date], QmtTradeStatusRecord] = {}
        for row in rows:
            code = _non_blank(row["etf_code"], "etf_code")
            symbol = requested_by_code.get(code)
            if symbol is None:
                raise QmtDataQualityError(f"unexpected trade-status etf_code: {code}")
            trade_date = _plain_date(row["trade_date"], "trade_date")
            key = (symbol, trade_date)
            if key in indexed:
                raise QmtDataQualityError(f"duplicate trade-status business key: {key!r}")
            lower = row["down_limit_price_cny"]
            upper = row["up_limit_price_cny"]
            raw_flag = row["qmt_suspend_flag"]
            if raw_flag is not None and (
                isinstance(raw_flag, bool) or not isinstance(raw_flag, int)
            ):
                raise QmtDataQualityError("qmt_suspend_flag must be an integer or NULL")
            indexed[key] = QmtTradeStatusRecord(
                symbol=symbol,
                trade_date=trade_date,
                qmt_suspend_flag=raw_flag,
                price_limit_down=(
                    None
                    if lower is None
                    else _decimal(lower, "down_limit_price_cny", positive=True)
                ),
                price_limit_up=(
                    None if upper is None else _decimal(upper, "up_limit_price_cny", positive=True)
                ),
                qmt_source_system=(
                    None if row["qmt_source_system"] is None else str(row["qmt_source_system"])
                ),
                price_limit_source_system=(
                    None
                    if row["price_limit_source_system"] is None
                    else str(row["price_limit_source_system"])
                ),
            )
        return indexed

    # 将前复权日线查询结果转换为索引记录，并检查重复。
    def _front_records(
        self, symbols: tuple[str, ...], start_date: date, end_date: date
    ) -> dict[tuple[str, date], QmtFrontDailyBar]:
        codes = [_code_for_symbol(symbol) for symbol in symbols]
        statement = text(
            """
            SELECT etf_code, trade_date,
                   open_price_cny, high_price_cny, low_price_cny,
                   close_price_cny, source_system
            FROM etf_quote_qmt_front_ratio_daily
            WHERE etf_code IN :etf_codes
              AND trade_date BETWEEN :start_date AND :end_date
              AND source_system = 'QMT'
            ORDER BY trade_date, etf_code
            """
        ).bindparams(bindparam("etf_codes", expanding=True))
        rows = self._fetch(
            statement,
            {
                "etf_codes": codes,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
        requested_by_code = dict(zip(codes, symbols, strict=True))
        indexed: dict[tuple[str, date], QmtFrontDailyBar] = {}
        for row in rows:
            code = _non_blank(row["etf_code"], "etf_code")
            symbol = requested_by_code.get(code)
            if symbol is None:
                raise QmtDataQualityError(f"unexpected front etf_code: {code}")
            trade_date = _plain_date(row["trade_date"], "trade_date")
            key = (symbol, trade_date)
            if key in indexed:
                raise QmtDataQualityError(f"duplicate selected front business key: {key!r}")
            indexed[key] = QmtFrontDailyBar(
                source_record_key=_source_record_key(
                    symbol=symbol,
                    trade_date=trade_date,
                    dataset_version=self._dataset_version,
                ),
                symbol=symbol,
                trade_date=trade_date,
                open=_decimal(row["open_price_cny"], "open_price_cny", positive=True),
                high=_decimal(row["high_price_cny"], "high_price_cny", positive=True),
                low=_decimal(row["low_price_cny"], "low_price_cny", positive=True),
                close=_decimal(row["close_price_cny"], "close_price_cny", positive=True),
                source_system=_non_blank(row["source_system"], "source_system"),
            )
        return indexed

    # 核对同证券同日的原始与前复权记录，防止缺失或错配进入数据集。
    @staticmethod
    def _validate_raw_front_pair(raw: QmtRawDailyBar, front: QmtFrontDailyBar) -> None:
        if raw.symbol != front.symbol or raw.trade_date != front.trade_date:
            raise QmtDataQualityError("raw/front business keys differ")

    # 从主表记录生成回测所需的 ETF 生命周期信息。
    @staticmethod
    def _infos_from_master(
        master: Sequence[QmtEtfMasterRecord],
    ) -> tuple[EtfInfo, ...]:
        return tuple(
            EtfInfo(
                symbol=record.symbol,
                exchange=record.exchange,
                name=record.name,
                primary_category=_non_blank(record.primary_category, "primary_category"),
                fund_type=_non_blank(record.fund_type, "fund_type"),
                list_date=record.list_date,
                delist_date=record.delist_date,
                current_status=record.current_status,
            )
            for record in master
        )

    # 检查冻结证券的生命周期信息是否完整且与请求范围一致。
    @classmethod
    def _validate_lifecycle_infos(
        cls,
        requested: tuple[str, ...],
        master: Sequence[QmtEtfMasterRecord],
        etf_infos: Sequence[EtfInfo],
    ) -> tuple[EtfInfo, ...]:
        master_by_symbol = {record.symbol: record for record in master}
        provided: dict[str, EtfInfo] = {}
        for info in etf_infos:
            if not isinstance(info, EtfInfo):
                raise TypeError("etf_infos must contain EtfInfo")
            if info.symbol in provided:
                raise QmtDataQualityError(f"duplicate lifecycle info for {info.symbol}")
            provided[info.symbol] = info
        if set(provided) != set(requested):
            raise QmtDataQualityError("etf_infos must exactly cover requested symbols")
        for symbol, info in provided.items():
            record = master_by_symbol[symbol]
            if (
                info.exchange is not record.exchange
                or info.name != record.name
                or info.list_date != record.list_date
                or info.primary_category != record.primary_category
                or info.fund_type != record.fund_type
            ):
                raise QmtDataQualityError(f"lifecycle info conflicts with dim_etf for {symbol}")
            if record.delist_date is not None and info.delist_date != record.delist_date:
                raise QmtDataQualityError(
                    f"exact dim_etf delist_date cannot be overridden for {symbol}"
                )
            if record.delist_date is None and info.delist_date is not None:
                if not info.delist_date_approximated:
                    raise QmtDataQualityError(
                        f"inferred delist_date must be marked approximated for {symbol}"
                    )
        return tuple(provided[symbol] for symbol in requested)

    # 检查数据库日历中的前后交易日引用是否与实际开市记录相符。
    @staticmethod
    def _validate_calendar_links(days: Sequence[QmtSseCalendarDay]) -> None:
        by_date = {day.cal_date: day for day in days}
        open_dates = tuple(day.cal_date for day in days if day.is_open)
        for index, trade_date in enumerate(open_dates):
            day = by_date[trade_date]
            if index > 0 and day.previous_open_date != open_dates[index - 1]:
                raise QmtDataQualityError(f"SSE previous_open_date chain conflict on {trade_date}")
            if index + 1 < len(open_dates) and day.next_open_date != open_dates[index + 1]:
                raise QmtDataQualityError(f"SSE next_open_date chain conflict on {trade_date}")

    # 将可空数据库日期转换为 date 或 None。
    @staticmethod
    def _optional_date(value: object, field_name: str) -> date | None:
        return None if value is None else _plain_date(value, field_name)

    # 将可空数据库文本转换为规范字符串或 None。
    @staticmethod
    def _optional_string(value: object) -> str | None:
        return None if value is None else _non_blank(value, "optional string")


__all__ = [
    "HUIJIN_ENTITIES",
    "HuijinHolderRatioRecord",
    "QmtDailyDataset",
    "QmtDailyFrame",
    "QmtDailyRepository",
    "QmtDataQualityError",
    "QmtEtfMasterRecord",
    "QmtEtfShareRecord",
    "QmtExplicitPriceLimit",
    "QmtFrontDailyBar",
    "QmtRawDailyBar",
    "QmtSseCalendarDay",
    "QmtTradeStatusRecord",
]
