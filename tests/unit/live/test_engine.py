from datetime import date
from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.engine import LiveTradingEngine
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.scheduler import BrokerJobRunner, LiveScheduler
from etf_backtest.live.state import AccountStatus, JobTriggerSource

ROOT = Path(__file__).parents[3]


def _engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveTradingEngine, Mock, Mock, Mock, Mock, Mock]:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    config = load_live_config(ROOT / "qmt_example/configs/live/beginner_example_paper.yaml")
    connection = Mock(spec=Connection)
    state_engine = Mock(spec=Engine)
    state_engine.connect.return_value = connection
    repository = Mock(spec=LiveStateRepository)
    broker = Mock(spec=BrokerGateway)
    quotes = Mock(spec=QuoteProvider)
    jobs = Mock(spec=LiveDailyJobs)
    jobs.all_symbols = ("SH.510300",)
    jobs.startup_reconcile.return_value = {
        "status": AccountStatus.ACTIVE,
        "universe_json": '["SH.510300"]',
    }
    scheduler = Mock(spec=LiveScheduler)
    engine = LiveTradingEngine(
        config=config,
        state_engine=cast(Engine, state_engine),
        repository=cast(LiveStateRepository, repository),
        broker=cast(BrokerGateway, broker),
        quote_provider=cast(QuoteProvider, quotes),
        jobs=cast(LiveDailyJobs, jobs),
        scheduler=cast(LiveScheduler, scheduler),
    )
    return engine, connection, broker, quotes, jobs, scheduler


def _runner(scheduler: Mock) -> BrokerJobRunner:
    return cast(BrokerJobRunner, scheduler.set_broker_job_runner.call_args.args[0])


def test_engine_start_only_locks_starts_consumer_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    engine, connection, broker, quotes, jobs, scheduler = _engine(monkeypatch)
    consumer = Mock()
    engine.set_event_consumer(consumer)
    monkeypatch.setattr(
        "etf_backtest.live.engine.acquire_account_lock",
        lambda *args: events.append("lock") or True,
    )
    monkeypatch.setattr(
        "etf_backtest.live.engine.release_account_lock",
        lambda *args: events.append("unlock"),
    )
    consumer.start.side_effect = lambda: events.append("consumer_start")
    scheduler.start.side_effect = lambda: events.append("scheduler_start")
    scheduler.stop.side_effect = lambda: events.append("scheduler_stop")
    consumer.stop.side_effect = lambda: events.append("consumer_stop")

    engine.start(date(2026, 8, 19))

    assert events == ["lock", "consumer_start", "scheduler_start"]
    broker.connect.assert_not_called()
    broker.subscribe_account.assert_not_called()
    jobs.startup_reconcile.assert_not_called()
    quotes.subscribe.assert_not_called()

    engine.stop()
    assert events == [
        "lock",
        "consumer_start",
        "scheduler_start",
        "scheduler_stop",
        "consumer_stop",
        "unlock",
    ]
    broker.disconnect.assert_not_called()
    connection.close.assert_called_once()


def test_account_lock_failure_does_not_connect_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, connection, broker, _, _, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: False)
    with pytest.raises(RuntimeError, match="account lock"):
        engine.start(date(2026, 8, 19))
    broker.connect.assert_not_called()
    scheduler.start.assert_not_called()
    connection.close.assert_called_once()


def test_run_forever_heartbeats_account_lock_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, connection, _, _, _, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    monkeypatch.setattr("etf_backtest.live.engine.monotonic", Mock(side_effect=[0.0, 301.0]))
    shutdown = Mock(spec=Event)
    shutdown.wait.side_effect = [False, True]
    engine._shutdown = shutdown

    engine.run_forever()

    connection.exec_driver_sql.assert_called_once_with("SELECT 1")
    scheduler.tick.assert_called_once()
    connection.close.assert_called_once()


def test_account_lock_heartbeat_failure_stops_without_release_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, connection, _, _, _, scheduler = _engine(monkeypatch)
    release = Mock()
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", release)
    monkeypatch.setattr("etf_backtest.live.engine.monotonic", Mock(side_effect=[0.0, 301.0]))
    shutdown = Mock(spec=Event)
    shutdown.wait.return_value = False
    engine._shutdown = shutdown
    connection.exec_driver_sql.side_effect = RuntimeError("database connection lost")

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        engine.run_forever()

    release.assert_not_called()
    scheduler.tick.assert_not_called()
    connection.close.assert_called_once()


def test_rebalance_and_eod_each_use_one_complete_broker_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    engine, _, broker, quotes, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    broker.connect.side_effect = lambda: events.append("connect")
    broker.subscribe_account.side_effect = lambda account: events.append("subscribe")
    jobs.startup_reconcile.side_effect = lambda *args, **kwargs: (
        events.append("startup") or {"status": AccountStatus.ACTIVE, "universe_json": "[]"}
    )
    quotes.subscribe.side_effect = lambda symbols: events.append("quotes")
    jobs.execute_pending_target.side_effect = lambda *args, **kwargs: events.append("rebalance")
    jobs.eod.side_effect = lambda *args, **kwargs: events.append("eod_snapshot")
    broker.disconnect.side_effect = lambda: events.append("disconnect")
    engine.start(date(2026, 8, 19))
    runner = _runner(scheduler)

    runner("rebalance", date(2026, 8, 19), JobTriggerSource.SCHEDULED)
    assert events == [
        "connect",
        "subscribe",
        "startup",
        "quotes",
        "rebalance",
        "disconnect",
    ]

    events.clear()
    runner("eod", date(2026, 8, 19), JobTriggerSource.SCHEDULED)
    assert events == ["connect", "subscribe", "startup", "eod_snapshot", "disconnect"]
    assert broker.connect.call_count == 2
    assert broker.disconnect.call_count == 2
    engine.stop()


@pytest.mark.parametrize(
    ("failure_at", "message"),
    [
        ("connect", "connect failed"),
        ("subscribe", "subscribe failed"),
        ("startup", "reconcile failed"),
        ("rebalance", "rebalance failed"),
        ("eod", "eod failed"),
    ],
)
def test_broker_session_failure_always_disconnects_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
    message: str,
) -> None:
    engine, _, broker, _, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    target = {
        "connect": broker.connect,
        "subscribe": broker.subscribe_account,
        "startup": jobs.startup_reconcile,
        "rebalance": jobs.execute_pending_target,
        "eod": jobs.eod,
    }[failure_at]
    target.side_effect = RuntimeError(message)
    engine.start(date(2026, 8, 19))
    runner = _runner(scheduler)
    job_name = "eod" if failure_at == "eod" else "rebalance"

    with pytest.raises(RuntimeError, match=message):
        runner(job_name, date(2026, 8, 19), JobTriggerSource.SCHEDULED)

    broker.disconnect.assert_called_once()
    assert engine._broker_connected is False
    engine.stop()
    broker.disconnect.assert_called_once()


def test_one_multi_strategy_rebalance_uses_one_broker_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, broker, _, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    jobs.strategy_runtime = Mock()
    engine.start(date(2026, 8, 19))

    _runner(scheduler)("rebalance", date(2026, 8, 19), JobTriggerSource.SCHEDULED)

    broker.connect.assert_called_once()
    broker.subscribe_account.assert_called_once_with("account-1")
    broker.disconnect.assert_called_once()
    jobs.execute_pending_target.assert_called_once()
    engine.stop()


def test_broker_unhealthy_callback_pauses_current_job_but_keeps_service_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, broker, _, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    broker.subscribe_account.side_effect = lambda account_id: engine.notify_broker_unhealthy(
        "BROKER_DISCONNECTED"
    )
    engine.start(date(2026, 8, 19))

    with pytest.raises(RuntimeError, match="BROKER_DISCONNECTED"):
        _runner(scheduler)("rebalance", date(2026, 8, 19), JobTriggerSource.SCHEDULED)

    engine.repository.pause_account.assert_called_once_with("account-1", "BROKER_DISCONNECTED")
    jobs.startup_reconcile.assert_not_called()
    broker.disconnect.assert_called_once()
    scheduler.stop.assert_not_called()
    assert engine._connection is not None
    engine.stop()
