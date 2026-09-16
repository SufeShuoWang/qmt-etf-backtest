"""原始收盘价和有效法定价格边界测试。"""

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.market import (
    EtfCategory,
    EtfTradingRule,
    FrameKey,
    MarketBar,
    MarketFrame,
    PriceLimitSource,
    TurnoverRule,
)
from etf_backtest.core.pricing import TradePriceQuoteCache


def _bar() -> MarketBar:
    return MarketBar(
        source_record_key="raw:SH.510300:2024-01-03",
        symbol="SH.510300",
        trade_date=date(2024, 1, 3),
        open=Decimal("10.000"),
        high=Decimal("10.200"),
        low=Decimal("9.900"),
        close=Decimal("10.100"),
        pre_close=Decimal("10.005"),
        volume=100000,
        amount=Decimal("1000000"),
        suspended=False,
    )


def _rule() -> EtfTradingRule:
    return EtfTradingRule(
        symbol="SH.510300",
        etf_category=EtfCategory.DOMESTIC_STOCK_ETF,
        turnover_rule=TurnoverRule.T1,
        price_limit_ratio=Decimal("0.10"),
    )


def _cache(bar, rule):
    return TradePriceQuoteCache(
        frame=MarketFrame.from_bars(FrameKey(trade_date=bar.trade_date, calendar_version="test"), (bar,)),
        trading_rules={bar.symbol: rule},
    )


@pytest.mark.unit
def test_close_model_uses_raw_close_and_tick_rounded_legal_limits() -> None:
    quote = _cache(_bar(), _rule()).quote_for(_bar().symbol)

    assert quote.base_trade_price == Decimal("10.100")
    assert quote.price_limit_down == Decimal("9.005")
    assert quote.price_limit_up == Decimal("11.006")
    assert quote.price_source == "CLOSE"
    assert quote.price_limit_source is PriceLimitSource.DERIVED_RULE_FALLBACK
    assert quote.source_record_key == _bar().source_record_key


@pytest.mark.unit
def test_close_model_prefers_explicit_daily_price_limits() -> None:
    bar = replace(
        _bar(),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("10.500"),
        price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
    )

    quote = _cache(bar, _rule()).quote_for(bar.symbol)

    assert quote.price_limit_down == Decimal("9.000")
    assert quote.price_limit_up == Decimal("10.500")
    assert quote.price_limit_source is PriceLimitSource.TUSHARE_EXPLICIT


@pytest.mark.unit
def test_quote_cache_resolves_a_frame_symbol_once() -> None:
    bar = _bar()
    frame = MarketFrame.from_bars(
        FrameKey(trade_date=bar.trade_date, calendar_version="qmt-v1"),
        (bar,),
    )
    cache = TradePriceQuoteCache(
        frame=frame,
        trading_rules={bar.symbol: _rule()},
    )

    first = cache.quote_for("510300")
    second = cache.quote_for("SH.510300")

    assert first is second


@pytest.mark.unit
def test_rule_identity_must_match_bar() -> None:
    wrong = EtfTradingRule(
        symbol="SH.518880",
        etf_category=EtfCategory.GOLD_ETF,
        turnover_rule=TurnoverRule.T0,
        price_limit_ratio=Decimal("0.10"),
    )
    with pytest.raises(ValueError, match="key mismatch"):
        _cache(_bar(), wrong)
