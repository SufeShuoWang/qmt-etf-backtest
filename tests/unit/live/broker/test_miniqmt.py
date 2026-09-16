from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import etf_backtest.live.broker.miniqmt as miniqmt
from etf_backtest.live.broker.miniqmt import MiniQmtBrokerGateway


def _gateway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    connect_results: list[int],
    *,
    session_id: int | None = 123,
) -> tuple[MiniQmtBrokerGateway, Mock, Mock]:
    trader = Mock()
    trader.connect.side_effect = connect_results
    trader_factory = Mock(return_value=trader)
    account_factory = Mock(return_value=object())
    monkeypatch.setattr(
        miniqmt,
        "_load_xtquant",
        lambda: (
            SimpleNamespace(XtQuantTrader=trader_factory),
            SimpleNamespace(StockAccount=account_factory),
            SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24),
        ),
    )
    monkeypatch.setattr(
        miniqmt,
        "create_xtquant_callback",
        lambda events, **kwargs: object(),
    )
    monkeypatch.setattr(miniqmt.time, "sleep", Mock())
    return (
        MiniQmtBrokerGateway(
            userdata_path=tmp_path,
            session_id=session_id,
            account_id="paper-1",
            event_queue=Queue(),
        ),
        trader,
        trader_factory,
    )


def test_connect_succeeds_on_first_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gateway, trader, _ = _gateway(monkeypatch, tmp_path, [0])

    gateway.connect()

    trader.connect.assert_called_once_with()
    trader.stop.assert_not_called()


def test_connect_retries_until_third_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway, trader, _ = _gateway(monkeypatch, tmp_path, [-1, -1, 0])

    gateway.connect()

    assert trader.connect.call_count == 3
    assert miniqmt.time.sleep.call_count == 2
    trader.stop.assert_not_called()


def test_connect_stops_and_never_constructs_account_after_five_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway, trader, _ = _gateway(monkeypatch, tmp_path, [-1] * 5)
    account_factory = gateway._xttype.StockAccount

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        gateway.connect()

    assert trader.connect.call_count == 5
    trader.stop.assert_called_once_with()
    trader.subscribe.assert_not_called()
    account_factory.assert_not_called()


def test_separate_production_gateway_lifetimes_use_distinct_session_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(miniqmt.time, "time", lambda: 1_800_000_000)
    first, _, first_factory = _gateway(monkeypatch, tmp_path, [0], session_id=None)
    first.connect()
    second, _, second_factory = _gateway(monkeypatch, tmp_path, [0], session_id=None)
    second.connect()

    first_session = first_factory.call_args.args[1]
    second_session = second_factory.call_args.args[1]
    assert first_session != second_session


def test_reconnecting_same_gateway_creates_new_trader_and_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(miniqmt.time, "time", lambda: 1_800_000_000)
    first_trader = Mock()
    first_trader.connect.return_value = 0
    second_trader = Mock()
    second_trader.connect.return_value = 0
    trader_factory = Mock(side_effect=(first_trader, second_trader))
    monkeypatch.setattr(
        miniqmt,
        "_load_xtquant",
        lambda: (
            SimpleNamespace(XtQuantTrader=trader_factory),
            SimpleNamespace(StockAccount=Mock(return_value=object())),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        miniqmt,
        "create_xtquant_callback",
        lambda events, **kwargs: object(),
    )
    gateway = MiniQmtBrokerGateway(
        userdata_path=tmp_path,
        session_id=None,
        account_id="paper-1",
        event_queue=Queue(),
    )

    gateway.connect()
    gateway.disconnect()
    gateway.connect()
    gateway.disconnect()

    first_session = trader_factory.call_args_list[0].args[1]
    second_session = trader_factory.call_args_list[1].args[1]
    assert first_session != second_session
    first_trader.stop.assert_called_once_with()
    second_trader.stop.assert_called_once_with()


def test_mixed_physical_account_queries_accept_external_stocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway, trader, _ = _gateway(monkeypatch, tmp_path, [0])
    gateway.connect()
    trader.query_stock_positions.return_value = [
        SimpleNamespace(stock_code="510300.SH", volume=100, can_use_volume=100),
        SimpleNamespace(stock_code="600000.SH", volume=200, can_use_volume=200),
        SimpleNamespace(stock_code="000001.SZ", volume=300, can_use_volume=300),
    ]
    trader.query_stock_orders.return_value = [
        SimpleNamespace(
            order_id=1,
            stock_code="510300.SH",
            order_type=23,
            order_volume=100,
            traded_volume=0,
            price=4,
            order_status=50,
            order_remark="external-etf",
            order_time=1_724_049_000,
        ),
        SimpleNamespace(
            order_id=2,
            stock_code="600000.SH",
            order_type=23,
            order_volume=100,
            traded_volume=0,
            price=10,
            order_status=50,
            order_remark="external-stock",
            order_time=1_724_049_000,
        ),
    ]
    trader.query_stock_trades.return_value = [
        SimpleNamespace(
            traded_id="trade-1",
            order_id=2,
            stock_code="300750.SZ",
            order_type=23,
            traded_volume=100,
            traded_price=10,
            traded_time=1_724_049_001,
            order_remark="external-stock",
        )
    ]

    positions = gateway.query_positions()
    orders = gateway.query_orders()
    trades = gateway.query_trades()

    assert positions.success and [row.symbol for row in positions.records] == [
        "SH.510300",
        "SH.600000",
        "SZ.000001",
    ]
    assert orders.success and [row.symbol for row in orders.records] == [
        "SH.510300",
        "SH.600000",
    ]
    assert trades.success and [row.symbol for row in trades.records] == ["SZ.300750"]


def test_account_with_only_external_stocks_still_returns_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway, trader, _ = _gateway(monkeypatch, tmp_path, [0])
    gateway.connect()
    trader.query_stock_positions.return_value = [
        SimpleNamespace(stock_code="600000.SH", volume=200, can_use_volume=200),
        SimpleNamespace(stock_code="300750.SZ", volume=300, can_use_volume=300),
    ]

    result = gateway.query_positions()

    assert result.success
    assert [row.symbol for row in result.records] == ["SH.600000", "SZ.300750"]
