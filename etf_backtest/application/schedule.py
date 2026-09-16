"""根据上交所交易日历生成稳定的策略调度索引。"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from etf_backtest.data.calendar import SseTradingCalendar


def trading_day_index(
    *, calendar: SseTradingCalendar, anchor_date: date, signal_date: date,
) -> int:
    """以固定交易日起点计算零基索引，并确认下一执行日存在。"""
    anchor = calendar.require_trading_day(anchor_date)
    signal = calendar.require_trading_day(signal_date)
    calendar.next_trading_day(signal)
    return len(calendar.trading_dates(anchor, signal)) - 1
