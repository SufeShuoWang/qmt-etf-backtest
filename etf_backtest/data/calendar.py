"""所有受支持 ETF 共用的纯日频上交所交易日历。"""

from __future__ import annotations


from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import date, datetime, time
from types import MappingProxyType

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.config.schema import CALENDAR_POLICY, MARKET_TIMEZONE
from etf_backtest.core.market import FrameKey
from etf_backtest.data.mysql import QmtSseCalendarDay


class CalendarCoverageError(LookupError):
    """日期或相邻交易日超出冻结日历范围。"""


class SseTradingCalendar:
    """不可变的上交所自然日历，不根据星期推断交易日。

    日历策略固定为 ``SSE_FOR_ALL``，因此深交所 ETF 也按同一组冻结上交所日期推进，
    实现不会虚构第二套交易所日历数据。
    """

    # 建立 SSE 自然日与开市日索引，检查日期覆盖和前后交易日关联。
    def __init__(
        self,
        days: Sequence[QmtSseCalendarDay],
        *,
        calendar_source: str | None = None,
        calendar_version: str | None = None,
    ) -> None:
        if not days:
            raise ValueError("calendar days must not be empty")
        by_date: dict[date, QmtSseCalendarDay] = {}
        sources: set[str] = set()
        versions: set[str] = set()
        for day in days:
            if not isinstance(day, QmtSseCalendarDay):
                raise TypeError("days must contain QmtSseCalendarDay")
            if day.cal_date in by_date:
                raise ValueError(f"duplicate SSE calendar date: {day.cal_date}")
            by_date[day.cal_date] = day
            sources.add(day.source_system)
            versions.add(day.calendar_version)
        ordered_dates = tuple(sorted(by_date))
        expected = tuple(
            date.fromordinal(ordinal)
            for ordinal in range(ordered_dates[0].toordinal(), ordered_dates[-1].toordinal() + 1)
        )
        if ordered_dates != expected:
            missing = tuple(item for item in expected if item not in by_date)
            raise ValueError(f"incomplete SSE natural-date calendar: {missing!r}")
        if len(sources) != 1 or len(versions) != 1:
            raise ValueError("calendar source and version must be uniform")
        source = next(iter(sources))
        version = next(iter(versions))
        if calendar_source is not None and calendar_source.strip() != source:
            raise ValueError("calendar_source does not match loaded SSE records")
        if calendar_version is not None and calendar_version.strip() != version:
            raise ValueError("calendar_version does not match loaded SSE records")
        open_dates = tuple(day for day in ordered_dates if by_date[day].is_open)
        if not open_dates:
            raise ValueError("calendar coverage contains no open SSE date")
        self._validate_links(by_date, open_dates)
        self._days_by_date = MappingProxyType(by_date)
        self._natural_dates = ordered_dates
        self._open_dates = open_dates
        self._calendar_source = source
        self._calendar_version = version

    # 读取日历来源名称。
    @property
    def calendar_source(self) -> str:
        return self._calendar_source

    # 读取日历版本，供行情帧身份和结果溯源使用。
    @property
    def calendar_version(self) -> str:
        return self._calendar_version

    # 返回本项目统一使用的 SSE 日历政策。
    @property
    def calendar_policy(self) -> str:
        return CALENDAR_POLICY

    # 返回日历覆盖的起始自然日。
    @property
    def start_date(self) -> date:
        return self._natural_dates[0]

    # 返回日历覆盖的结束自然日。
    @property
    def end_date(self) -> date:
        return self._natural_dates[-1]

    # 返回覆盖区间内有序的开市日期。
    @property
    def open_dates(self) -> tuple[date, ...]:
        return self._open_dates

    # 判断给定自然日是否位于已加载日历中。
    def contains(self, trade_date: date) -> bool:
        return _plain_date(trade_date, "trade_date") in self._days_by_date

    # 按日期取得日历记录；缺失日期按接口约定报错。
    def day(self, trade_date: date) -> QmtSseCalendarDay:
        value = _plain_date(trade_date, "trade_date")
        try:
            return self._days_by_date[value]
        except KeyError:
            raise CalendarCoverageError(f"date is outside SSE calendar: {value}") from None

    # 判断给定日期是否为 SSE 开市日。
    def is_trading_day(self, trade_date: date) -> bool:
        return self.day(trade_date).is_open

    # 要求日期为已覆盖的交易日，否则阻止回测或信号计算继续。
    def require_trading_day(self, trade_date: date) -> date:
        value = _plain_date(trade_date, "trade_date")
        if not self.day(value).is_open:
            raise ValueError(f"date is not an SSE trading date: {value}")
        return value

    # 提取指定起止区间内有序的交易日期。
    def trading_dates(self, start_date: date, end_date: date) -> tuple[date, ...]:
        start = _plain_date(start_date, "start_date")
        end = _plain_date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date must not follow end_date")
        self.day(start)
        self.day(end)
        left = bisect_left(self._open_dates, start)
        right = bisect_right(self._open_dates, end)
        return self._open_dates[left:right]

    # 查询指定日期之后的下一交易日，供 D+1 执行绑定使用。
    def next_trading_day(self, trade_date: date) -> date:
        value = _plain_date(trade_date, "trade_date")
        self.day(value)
        index = bisect_right(self._open_dates, value)
        if index >= len(self._open_dates):
            raise CalendarCoverageError(f"no next SSE trading date is loaded after {value}")
        return self._open_dates[index]

    # 查询指定日期之前的上一交易日。
    def previous_trading_day(self, trade_date: date) -> date:
        value = _plain_date(trade_date, "trade_date")
        self.day(value)
        index = bisect_left(self._open_dates, value) - 1
        if index < 0:
            raise CalendarCoverageError(f"no previous SSE trading date is loaded before {value}")
        return self._open_dates[index]

    # 为一个交易日生成标准日频行情帧标识。
    def frame_key(self, trade_date: date) -> FrameKey:
        value = self.require_trading_day(trade_date)
        return FrameKey(trade_date=value, calendar_version=self._calendar_version)

    # 为日期区间生成有序的日频行情帧标识序列。
    def frame_keys(self, start_date: date, end_date: date) -> tuple[FrameKey, ...]:
        return tuple(
            FrameKey(trade_date=value, calendar_version=self._calendar_version)
            for value in self.trading_dates(start_date, end_date)
        )

    # 取得下一交易日的行情帧标识。
    def next_frame_key(self, frame_key: FrameKey) -> FrameKey:
        if not isinstance(frame_key, FrameKey):
            raise TypeError("frame_key must be FrameKey")
        if frame_key.calendar_version != self._calendar_version:
            raise ValueError("FrameKey calendar version does not match frozen calendar")
        return self.frame_key(self.next_trading_day(frame_key.trade_date))

    # 生成交易日收盘时刻，带市场时区。
    def close_time(self, trade_date: date) -> datetime:
        value = self.require_trading_day(trade_date)
        return datetime.combine(value, time(15), tzinfo=MARKET_TIMEZONE)

    # 校验日历记录声明的前后交易日与开市日期索引相符。
    @staticmethod
    def _validate_links(
        by_date: dict[date, QmtSseCalendarDay], open_dates: tuple[date, ...]
    ) -> None:
        open_set = set(open_dates)
        for index, trade_date in enumerate(open_dates):
            day = by_date[trade_date]
            if index > 0 and day.previous_open_date != open_dates[index - 1]:
                raise ValueError(f"previous_open_date conflict on {trade_date}")
            if index + 1 < len(open_dates) and day.next_open_date != open_dates[index + 1]:
                raise ValueError(f"next_open_date conflict on {trade_date}")
        for natural_date, day in by_date.items():
            if day.previous_open_date in by_date and day.previous_open_date not in open_set:
                raise ValueError(f"previous_open_date is closed on {natural_date}")
            if day.next_open_date in by_date and day.next_open_date not in open_set:
                raise ValueError(f"next_open_date is closed on {natural_date}")


TradingCalendar = SseTradingCalendar

__all__ = ["CalendarCoverageError", "SseTradingCalendar", "TradingCalendar"]
