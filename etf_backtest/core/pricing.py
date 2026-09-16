"""唯一生产执行价格模型：未复权日频收盘价。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import (
    EtfTradingRule,
    MarketFrame,
    resolve_legal_price_limits,
)
from etf_backtest.core.order import TradePriceQuote


class TradePriceQuoteCache:
    """为完整日频行情帧中的每只证券缓存一个收盘报价。"""

    __slots__ = ("_frame", "_quotes", "_rules")

    # 为一个执行行情帧建立原始报价缓存，保存当日交易规则。
    def __init__(
        self,
        *,
        frame: MarketFrame,
        trading_rules: Mapping[str, EtfTradingRule],
    ) -> None:
        if not isinstance(frame, MarketFrame):
            raise TypeError("frame must be MarketFrame")
        if not isinstance(trading_rules, Mapping):
            raise TypeError("trading_rules must be a mapping")
        resolved: dict[str, EtfTradingRule] = {}
        for supplied_symbol, rule in trading_rules.items():
            symbol = normalize_symbol(supplied_symbol)
            if not isinstance(rule, EtfTradingRule):
                raise TypeError("trading_rules may contain only EtfTradingRule")
            if symbol != rule.symbol or symbol in resolved:
                raise ValueError("trading rule key mismatch or duplicate")
            resolved[symbol] = rule
        self._frame = frame
        self._rules = MappingProxyType(dict(sorted(resolved.items())))
        self._quotes: dict[str, TradePriceQuote] = {}

    # 按证券生成或复用合法报价，统一涨跌停价格与来源信息。
    def quote_for(self, symbol: str) -> TradePriceQuote:
        canonical = normalize_symbol(symbol)
        cached = self._quotes.get(canonical)
        if cached is not None:
            return cached
        try:
            rule = self._rules[canonical]
        except KeyError:
            raise KeyError(f"no effective trading rule for {canonical}") from None
        bar = self._frame.bar_for(canonical)
        lower, upper, source = resolve_legal_price_limits(
            execution_bar=bar, trading_rule=rule,
        )
        quote = TradePriceQuote(
            source_record_key=bar.source_record_key,
            symbol=bar.symbol,
            trade_date=bar.trade_date,
            base_trade_price=bar.close,
            price_limit_down=lower,
            price_limit_up=upper,
            price_limit_source=source,
        )
        self._quotes[canonical] = quote
        return quote

    # 一次取得指定证券的报价集合，避免审批和成交重复解析报价。
    def resolve_all(self, symbols: Iterable[str]) -> Mapping[str, TradePriceQuote]:
        canonical = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
        return MappingProxyType({symbol: self.quote_for(symbol) for symbol in canonical})


__all__ = ["TradePriceQuoteCache"]
