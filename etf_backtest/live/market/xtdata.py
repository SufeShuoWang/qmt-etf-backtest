"""基于 xtdata、面向单个冻结 ETF 证券范围的精简当前报价提供器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType

from etf_backtest.live.broker.mapper import internal_to_external_symbol, map_quote
from etf_backtest.live.state import LiveQuote, QueryResult


# 按需加载 xtquant.xtdata，未安装时报告行情依赖缺失。
def _load_xtdata() -> ModuleType:
    try:
        return import_module("xtquant.xtdata")
    except ImportError as error:
        raise RuntimeError("当前环境未安装 xtquant, 无法创建交易运行时。") from error


# 用 xtdata 实现行情接口，结合 tick 与缓存证券资料生成标准报价。
class XtDataQuoteProvider:
    # 绑定或加载 xtdata 模块，初始化订阅与证券资料缓存。
    def __init__(self) -> None:
        self._xtdata = _load_xtdata()
        self._subscription_ids: dict[str, int] = {}
        self._instrument_cache: dict[str, Mapping[str, object]] = {}

    # 为尚未订阅的证券建立行情订阅。
    def subscribe(self, symbols: Sequence[str]) -> None:
        external_symbols = tuple(internal_to_external_symbol(symbol) for symbol in symbols)
        for symbol in external_symbols:
            if symbol in self._subscription_ids:
                continue
            subscription_id = self._xtdata.subscribe_quote(symbol, period="tick", count=0)
            if not isinstance(subscription_id, int) or subscription_id <= 0:
                raise RuntimeError(f"xtdata quote subscription failed: {symbol}")
            self._subscription_ids[symbol] = subscription_id

    # 获取最新 tick 和所需证券资料，映射为带时刻、盘口及价格限制的 LiveQuote。
    def latest_quotes(self, symbols: Sequence[str]) -> QueryResult[LiveQuote]:
        external_symbols = tuple(internal_to_external_symbol(symbol) for symbol in symbols)
        try:
            ticks = self._xtdata.get_full_tick(list(external_symbols))
            if not isinstance(ticks, Mapping):
                return QueryResult(success=False, error="xtdata get_full_tick returned no mapping")
            quotes: list[LiveQuote] = []
            for symbol in external_symbols:
                tick = ticks.get(symbol)
                if not isinstance(tick, Mapping):
                    return QueryResult(success=False, error=f"xtdata tick is missing: {symbol}")
                detail = self._instrument_cache.get(symbol)
                if detail is None:
                    loaded = self._xtdata.get_instrument_detail(symbol)
                    if not isinstance(loaded, Mapping):
                        return QueryResult(
                            success=False,
                            error=f"xtdata instrument detail is missing: {symbol}",
                        )
                    detail = dict(loaded)
                    self._instrument_cache[symbol] = detail
                quotes.append(map_quote(symbol, tick, detail))
            return QueryResult(success=True, records=tuple(quotes))
        except Exception as error:
            return QueryResult(success=False, error=str(error))


__all__ = ["XtDataQuoteProvider"]
