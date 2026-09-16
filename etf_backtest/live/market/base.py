"""获取已规范化实时报价的协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from etf_backtest.live.state import LiveQuote, QueryResult


# 规定行情订阅和最新报价查询接口，使执行流程不直接依赖 xtdata。
class QuoteProvider(Protocol):
    # 订阅指定证券实时行情，由具体提供者实现。
    def subscribe(self, symbols: Sequence[str]) -> None: ...

    # 查询指定证券的最新报价，明确区分查询失败与成功结果。
    def latest_quotes(self, symbols: Sequence[str]) -> QueryResult[LiveQuote]: ...


__all__ = ["QuoteProvider"]
