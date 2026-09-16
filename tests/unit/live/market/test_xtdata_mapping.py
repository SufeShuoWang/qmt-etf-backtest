from decimal import Decimal

from etf_backtest.live.broker.mapper import map_quote


def test_tick_and_instrument_detail_map_without_fabricating_bid_or_ask() -> None:
    quote = map_quote(
        "510300.SH",
        {
            "time": 1724049000000,
            "lastPrice": 4.012,
            "bidPrice": [0, 4.01],
            "askPrice": [],
            "stockStatus": 13,
        },
        {
            "UpStopPrice": 4.4,
            "DownStopPrice": 3.6,
            "PriceTick": 0.001,
            "IsTrading": True,
            "InstrumentStatus": "Trading",
        },
    )
    assert quote.last_price == Decimal("4.012")
    assert quote.bid1 is None and quote.ask1 is None
    assert quote.lower_limit == Decimal("3.6")
    assert quote.upper_limit == Decimal("4.4")
    assert quote.price_tick == Decimal("0.001")
    assert quote.suspended is False
    assert quote.quoted_at.tzinfo is not None

    suspended = map_quote(
        "159915.SZ",
        {"time": 1724049000000, "lastPrice": 1, "stockStatus": 17},
        {"IsTrading": False, "InstrumentStatus": "停牌"},
    )
    assert suspended.suspended is True


def test_target_xtquant_active_etf_status_is_not_misclassified_as_suspended() -> None:
    quote = map_quote(
        "510300.SH",
        {
            "time": 1787813521000,
            "lastPrice": 4.692,
            "bidPrice": [4.692, 4.691],
            "askPrice": [4.693, 4.694],
            "stockStatus": 3,
        },
        {
            "UpStopPrice": 5.117,
            "DownStopPrice": 4.187,
            "PriceTick": 0.001,
            "IsTrading": False,
            "InstrumentStatus": 0,
        },
    )

    assert quote.suspended is False
    assert quote.last_price == Decimal("4.692")
    assert quote.bid1 == Decimal("4.692")
    assert quote.ask1 == Decimal("4.693")
