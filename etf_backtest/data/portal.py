"""原始执行行情帧与前复权策略视图之间的纯日频边界。"""

from __future__ import annotations


from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from types import MappingProxyType

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import EtfInfo, IndexBarView, MarketBarView, MarketFrame
from etf_backtest.data.calendar import SseTradingCalendar
from etf_backtest.data.mysql import (
    HUIJIN_ENTITIES,
    HuijinHolderRatioRecord,
    QmtDailyDataset,
    QmtEtfShareRecord,
)


class DataQualityError(ValueError):
    """数据集违反冻结日频数据入口契约。"""


class DailyDataPortal:
    """不可变原始行情帧及无未来泄漏的前复权视图历史。

    执行调用方可获取某个上交所交易日的原始行情帧；策略和模型调用方只能通过明确的
    ``as_of_date`` 边界获取前复权视图，不提供包含全部日期的公开视图集合。
    """

    # 将日数据集组织成日期／证券索引，建立策略历史与原始执行行情的分离入口。
    def __init__(self, dataset: QmtDailyDataset) -> None:
        if not isinstance(dataset, QmtDailyDataset):
            raise TypeError("dataset must be QmtDailyDataset")
        calendar = SseTradingCalendar(
            dataset.calendar,
            calendar_source=dataset.calendar_source,
            calendar_version=dataset.dataset_version,
        )
        if calendar.start_date != dataset.start_date or calendar.end_date != dataset.end_date:
            raise DataQualityError("dataset dates must equal SSE calendar coverage")

        infos = self._index_infos(dataset.etf_infos, dataset.symbols)
        raw_frames = dataset.market_frames()
        frames_by_date: dict[date, MarketFrame] = {}
        for frame in raw_frames:
            trade_date = frame.trade_date
            if not calendar.is_trading_day(trade_date):
                raise DataQualityError(f"raw frame exists on closed SSE date: {trade_date}")
            if frame.frame_key.calendar_version != dataset.dataset_version:
                raise DataQualityError("raw FrameKey has the wrong calendar version")
            if trade_date in frames_by_date:
                raise DataQualityError(f"duplicate raw frame date: {trade_date}")
            frames_by_date[trade_date] = frame

        views_by_date: dict[date, dict[str, MarketBarView]] = {}
        for view in dataset.front_market_bar_views():
            if not calendar.is_trading_day(view.trade_date):
                raise DataQualityError(f"front view exists on closed SSE date: {view.trade_date}")
            per_date = views_by_date.setdefault(view.trade_date, {})
            if view.symbol in per_date:
                raise DataQualityError(
                    f"duplicate front view for {view.symbol} on {view.trade_date}"
                )
            per_date[view.symbol] = view

        self._validate_active_coverage(
            calendar=calendar,
            infos=infos,
            frames_by_date=frames_by_date,
            views_by_date=views_by_date,
        )
        frozen_views = {
            trade_date: MappingProxyType(dict(sorted(per_date.items())))
            for trade_date, per_date in sorted(views_by_date.items())
        }
        share_by_symbol: dict[str, dict[date, Decimal]] = {symbol: {} for symbol in dataset.symbols}
        for share_record in dataset.share_records:
            if not isinstance(share_record, QmtEtfShareRecord):
                raise TypeError("share_records may contain only QmtEtfShareRecord")
            if share_record.symbol not in share_by_symbol:
                raise DataQualityError("share record is outside the configured universe")
            if not dataset.start_date <= share_record.asof_date <= dataset.end_date:
                raise DataQualityError("share record is outside dataset date coverage")
            per_symbol = share_by_symbol[share_record.symbol]
            if share_record.asof_date in per_symbol:
                raise DataQualityError("duplicate selected daily share record")
            per_symbol[share_record.asof_date] = share_record.total_share
        frozen_shares = {
            symbol: MappingProxyType(dict(sorted(rows.items())))
            for symbol, rows in sorted(share_by_symbol.items())
        }

        ratio_history: dict[str, dict[str, dict[date, Decimal]]] = {
            symbol: {} for symbol in dataset.symbols
        }
        for ratio_record in dataset.huijin_ratio_records:
            if not isinstance(ratio_record, HuijinHolderRatioRecord):
                raise TypeError("huijin_ratio_records may contain only HuijinHolderRatioRecord")
            if ratio_record.symbol not in ratio_history:
                raise DataQualityError("Huijin ratio is outside the configured universe")
            per_entity = ratio_history[ratio_record.symbol].setdefault(ratio_record.entity, {})
            if ratio_record.end_date in per_entity:
                raise DataQualityError("duplicate aggregated Huijin ratio record")
            per_entity[ratio_record.end_date] = ratio_record.ratio
        frozen_ratios = {
            symbol: MappingProxyType(
                {
                    entity: MappingProxyType(dict(sorted(rows.items())))
                    for entity, rows in sorted(entities.items())
                }
            )
            for symbol, entities in sorted(ratio_history.items())
        }
        index_history: dict[str, dict[date, IndexBarView]] = {}
        for index_bar in dataset.index_records:
            if not isinstance(index_bar, IndexBarView):
                raise TypeError("index_records may contain only IndexBarView")
            if not dataset.start_date <= index_bar.trade_date <= dataset.end_date:
                raise DataQualityError("index record is outside dataset date coverage")
            if not calendar.is_trading_day(index_bar.trade_date):
                raise DataQualityError("index record exists on a closed SSE date")
            per_index = index_history.setdefault(index_bar.index_code, {})
            if index_bar.trade_date in per_index:
                raise DataQualityError("duplicate selected daily index record")
            per_index[index_bar.trade_date] = index_bar
        frozen_index_history = {
            index_code: MappingProxyType(dict(sorted(rows.items())))
            for index_code, rows in sorted(index_history.items())
        }
        self._dataset_version = dataset.dataset_version
        self._symbols = dataset.symbols
        self._infos = MappingProxyType(infos)
        self._calendar = calendar
        self._frames_by_date = MappingProxyType(dict(sorted(frames_by_date.items())))
        self._views_by_date: Mapping[date, Mapping[str, MarketBarView]] = MappingProxyType(
            frozen_views
        )
        self._share_by_symbol = MappingProxyType(frozen_shares)
        self._huijin_ratio_history_by_symbol = MappingProxyType(frozen_ratios)
        self._index_history_by_code = MappingProxyType(frozen_index_history)

    # 返回行情入口使用的数据集版本。
    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # 返回本次数据入口覆盖的标准证券代码。
    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    # 返回冻结证券的主信息，供生命周期和规则检查。
    @property
    def etf_infos(self) -> tuple[EtfInfo, ...]:
        return tuple(self._infos[symbol] for symbol in self._symbols)

    # 返回数据入口对应的 SSE 交易日历。
    @property
    def trading_calendar(self) -> SseTradingCalendar:
        return self._calendar


    # 取得某交易日的原始行情帧，供执行和账户估值。
    def raw_frame(self, trade_date: date) -> MarketFrame | None:
        value = self._calendar.require_trading_day(trade_date)
        return self._frames_by_date.get(value)


    # 按信号日和可选回看窗口截取前复权历史；限制的是行的业务日期，数据快照本身仍属于回顾式快照。
    def views_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
        lookback_trading_days: int | None = None,
    ) -> tuple[MarketBarView, ...]:
        """仅返回业务日期不晚于 ``D`` 的前复权视图。"""

        cutoff = self._checked_cutoff(as_of_date)
        selected = self._selected_symbols(symbols)
        if lookback_trading_days is not None:
            if type(lookback_trading_days) is not int or lookback_trading_days <= 0:
                raise ValueError("lookback_trading_days must be a positive integer")
            right = bisect_right(self._calendar.open_dates, cutoff)
            eligible_dates = self._calendar.open_dates[
                max(0, right - lookback_trading_days) : right
            ]
        else:
            eligible_dates = tuple(
                trade_date for trade_date in self._calendar.open_dates if trade_date <= cutoff
            )
        return tuple(
            view
            for trade_date in eligible_dates
            for symbol, view in self._views_by_date.get(trade_date, {}).items()
            if symbol in selected
        )


    def share_history_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[date, Decimal]]:
        """返回截至信号日 D 的精确每日总份额观察值。"""

        cutoff = self._checked_cutoff(as_of_date)
        selected = self._selected_symbols(symbols)
        return MappingProxyType(
            {
                symbol: MappingProxyType(
                    {
                        asof_date: value
                        for asof_date, value in self._share_by_symbol[symbol].items()
                        if asof_date <= cutoff
                    }
                )
                for symbol in sorted(selected)
            }
        )

    # 实际按报告期 EndDate < 信号日筛选各主体最新比例；不能据此证明该报告在当时已经披露。
    def huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]:
        """返回每个汇金主体严格早于信号日 D 的最新比例。"""

        cutoff = self._checked_cutoff(as_of_date)
        selected = self._selected_symbols(symbols)
        result: dict[str, Mapping[str, tuple[date, Decimal]]] = {}
        for symbol in sorted(selected):
            latest: dict[str, tuple[date, Decimal]] = {}
            for entity, history in self._huijin_ratio_history_by_symbol[symbol].items():
                eligible = tuple(end_date for end_date in history if end_date < cutoff)
                if eligible:
                    end_date = eligible[-1]
                    latest[entity] = (end_date, history[end_date])
            result[symbol] = MappingProxyType(latest)
        return MappingProxyType(result)

    # 选择信号日前最新的共同计算报告期，汇总该期各主体比例；不会把不同报告期各自最新值直接相加。
    def combined_huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, tuple[date, Decimal]]:
        """返回两个汇金主体在 D 日前最新同报告期比例之和。"""

        cutoff = self._checked_cutoff(as_of_date)
        selected = self._selected_symbols(symbols)
        result: dict[str, tuple[date, Decimal]] = {}
        for symbol in sorted(selected):
            histories = self._huijin_ratio_history_by_symbol[symbol]
            eligible_dates = {
                end_date
                for entity in HUIJIN_ENTITIES
                for end_date in histories.get(entity, {})
                if end_date < cutoff
            }
            if not eligible_dates:
                continue
            end_date = max(eligible_dates)
            ratio = Decimal("0")
            for entity in HUIJIN_ENTITIES:
                entity_history = histories.get(entity)
                if entity_history is not None:
                    ratio += entity_history.get(end_date, Decimal("0"))
            if ratio > Decimal("1"):
                raise DataQualityError("combined Huijin ratio exceeds one")
            result[symbol] = (end_date, ratio)
        return MappingProxyType(result)

    def index_history_through(
        self,
        as_of_date: date,
        *,
        lookback_trading_days: int | None = None,
    ) -> Mapping[str, tuple[IndexBarView, ...]]:
        """返回截至信号日 D 的已配置 PRICE 指数行情。"""

        cutoff = self._checked_cutoff(as_of_date)
        if lookback_trading_days is not None:
            if type(lookback_trading_days) is not int or lookback_trading_days <= 0:
                raise ValueError("lookback_trading_days must be a positive integer")
        result: dict[str, tuple[IndexBarView, ...]] = {}
        for index_code, history in self._index_history_by_code.items():
            eligible = tuple(bar for trade_date, bar in history.items() if trade_date <= cutoff)
            if lookback_trading_days is not None:
                eligible = eligible[-lookback_trading_days:]
            result[index_code] = eligible
        return MappingProxyType(result)


    # 按起止区间返回连续有序的执行行情帧。
    def execution_frames(self, start_date: date, end_date: date) -> tuple[MarketFrame, ...]:
        trading_dates = self._calendar.trading_dates(start_date, end_date)
        return tuple(
            frame
            for trade_date in trading_dates
            if (frame := self._frames_by_date.get(trade_date)) is not None
        )

    def _checked_cutoff(self, as_of_date: date) -> date:
        """所有策略历史查询共用的日期类型与日历覆盖检查。"""
        cutoff = _plain_date(as_of_date, "as_of_date")
        self._calendar.day(cutoff)
        return cutoff

    # 校验调用方请求的证券属于已加载范围，并规范返回顺序。
    def _selected_symbols(self, symbols: Sequence[str] | None) -> frozenset[str]:
        if symbols is None:
            return frozenset(self._symbols)
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
            raise TypeError("symbols must be a sequence")
        selected = frozenset(normalize_symbol(symbol) for symbol in symbols)
        unknown = tuple(sorted(selected - set(self._symbols)))
        if unknown:
            raise ValueError(f"strategy history symbols are outside the universe: {unknown!r}")
        return selected

    # 索引并校验 ETF 主信息，避免证券重复或元数据冲突。
    @staticmethod
    def _index_infos(infos: Sequence[EtfInfo], symbols: tuple[str, ...]) -> dict[str, EtfInfo]:
        indexed: dict[str, EtfInfo] = {}
        for info in infos:
            if not isinstance(info, EtfInfo):
                raise TypeError("dataset.etf_infos must contain EtfInfo")
            if info.symbol in indexed:
                raise DataQualityError(f"duplicate EtfInfo for {info.symbol}")
            indexed[info.symbol] = info
        if set(indexed) != set(symbols):
            raise DataQualityError("EtfInfo rows must exactly cover dataset symbols")
        return dict(sorted(indexed.items()))

    # 检查每只证券在有效上市区间内的行情覆盖，避免静默跳过缺失交易日。
    @staticmethod
    def _validate_active_coverage(
        *,
        calendar: SseTradingCalendar,
        infos: Mapping[str, EtfInfo],
        frames_by_date: Mapping[date, MarketFrame],
        views_by_date: Mapping[date, Mapping[str, MarketBarView]],
    ) -> None:
        unexpected_frame_dates = tuple(sorted(set(frames_by_date) - set(calendar.open_dates)))
        if unexpected_frame_dates:
            raise DataQualityError(
                f"frames exist outside open SSE dates: {unexpected_frame_dates!r}"
            )
        for trade_date in calendar.open_dates:
            expected = {symbol for symbol, info in infos.items() if info.is_active(trade_date)}
            frame = frames_by_date.get(trade_date)
            raw_symbols = set() if frame is None else set(frame.bars_by_symbol)
            view_symbols = set(views_by_date.get(trade_date, {}))
            if raw_symbols != expected or view_symbols != expected:
                raise DataQualityError(
                    f"active daily coverage conflict on {trade_date}; "
                    f"expected={tuple(sorted(expected))!r}, "
                    f"raw={tuple(sorted(raw_symbols))!r}, "
                    f"front={tuple(sorted(view_symbols))!r}"
                )
            if frame is None:
                continue
            for symbol in expected:
                raw = frame.bars_by_symbol[symbol]
                view = views_by_date[trade_date][symbol]
                if (
                    raw.source_record_key != view.source_record_key
                    or raw.volume != view.volume
                    or raw.suspended != view.suspended
                ):
                    raise DataQualityError(
                        f"raw/front domain pair conflict for {symbol} on {trade_date}"
                    )


DataPortal = DailyDataPortal

__all__ = ["DailyDataPortal", "DataPortal", "DataQualityError"]
