from contextlib import nullcontext
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Engine

from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.core.sizing import calculate_target_quantities
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.jobs import JobSkipped, LiveDailyJobs
from etf_backtest.live.signals import StrategyRuntime, StrategySpec
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.state import (
    AccountStatus,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    JobTriggerSource,
    LiveQuote,
    OrderIntentStatus,
    QueryResult,
    ReconciliationReport,
    SubmitOrderResult,
    SubmitOrderStatus,
)

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"
NOW = datetime(2026, 8, 19, 20, 41, tzinfo=MARKET_TIMEZONE)
SYMBOLS = ("SH.510300", "SH.518880", "SH.588000")


def _runtime(
    strategy_id: str, case: str, evaluator: Mock, close: Decimal = Decimal("10")
) -> StrategyRuntime:
    return StrategyRuntime(
        spec=StrategySpec(
            account_id="account-1",
            strategy_id=strategy_id,
            case=case,
            initial_capital=Decimal("500000"),
            experiment_path=f"{strategy_id}.yaml",
            schedule_anchor_date=date(2021, 1, 4),
            symbols=SYMBOLS,
            model_backend=None if case == "rule" else "xgboost",
            model_bundle_path=None if case == "rule" else "model.ubj",
            model_id=None if case == "rule" else "xgboost",
        ),
        signal_evaluator=evaluator,
        turnover_rules={
            symbol: (TurnoverRule.T0 if symbol == "SH.518880" else TurnoverRule.T1)
            for symbol in SYMBOLS
        },
        close_price_provider=lambda trading_date, symbols: {symbol: close for symbol in symbols},
    )


def _jobs(
    monkeypatch: pytest.MonkeyPatch,
    repository: Mock,
    broker: Mock,
    evaluator: Mock | None = None,
) -> LiveDailyJobs:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    config = load_live_config(CONFIG)
    repository.strategy_order_notional_for_date.return_value = Decimal("0")
    repository.current_unresolved.return_value = ()
    repository.load_execution_targets.return_value = None
    repository.has_terminal_job.return_value = False
    if repository.start_job_run.side_effect is None:
        repository.start_job_run.side_effect = lambda **kwargs: (
            f"{kwargs['job_type']}:{kwargs.get('strategy_id') or 'account'}"
        )
    rule_eval = evaluator or Mock()
    quote_provider = Mock(spec=QuoteProvider)
    quote_provider.latest_quotes.return_value = QueryResult(
        success=True,
        records=tuple(
            LiveQuote(
                symbol=symbol,
                last_price=Decimal("10"),
                bid1=Decimal("9.99"),
                ask1=Decimal("10.01"),
                lower_limit=Decimal("9"),
                upper_limit=Decimal("11"),
                suspended=False,
                quoted_at=NOW.replace(hour=14, minute=52),
                price_tick=Decimal("0.01"),
            )
            for symbol in SYMBOLS
        ),
    )
    state_engine = Mock(spec=Engine)
    return LiveDailyJobs(
        config=config,
        broker=cast(BrokerGateway, broker),
        quote_provider=cast(QuoteProvider, quote_provider),
        state_repository=cast(LiveStateRepository, repository),
        state_engine=cast(Engine, state_engine),
        strategy_runtime=_runtime("beginner_rule", "rule", rule_eval),
        reconciliation_service=cast(ReconciliationService, Mock()),
        clock=lambda: NOW,
    )


def _broker() -> Mock:
    broker = Mock(spec=BrokerGateway)
    broker.query_orders.return_value = QueryResult(success=True)
    broker.query_trades.return_value = QueryResult(success=True)
    broker.submit_order.return_value = SubmitOrderResult(
        SubmitOrderStatus.ACCEPTED, broker_order_id="broker-1"
    )
    return broker


def _account() -> dict[str, object]:
    return {"account_id": "account-1", "status": AccountStatus.ACTIVE}


def _configure_targets(repository: Mock, targets: dict[str, dict[str, Decimal]]) -> None:
    frozen: dict[str, dict[str, tuple[Decimal, int]]] = {}
    repository.load_target_positions.side_effect = lambda decision_id: targets[decision_id]
    repository.load_execution_targets.side_effect = lambda decision_id: frozen.get(decision_id)

    def freeze(
        decision_id: str,
        *,
        total_asset: Decimal,
        valuation_prices: dict[str, Decimal],
        lot_size: int,
    ) -> dict[str, tuple[Decimal, int]]:
        quantities = calculate_target_quantities(
            target_weights=targets[decision_id],
            total_asset=total_asset,
            valuation_prices=valuation_prices,
            lot_size=lot_size,
        )
        result = {
            symbol: (valuation_prices[symbol], quantity) for symbol, quantity in quantities.items()
        }
        frozen[decision_id] = result
        return result

    repository.freeze_execution_targets.side_effect = freeze


def test_job_syncs_account_before_creating_job_run(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Mock(spec=LiveStateRepository)
    events: list[str] = []

    def sync_account(**kwargs: object) -> dict[str, object]:
        events.append("sync_account")
        return _account()

    def start_job_run(**kwargs: object) -> str:
        events.append("start_job_run")
        return "run-1"

    repository.sync_account.side_effect = sync_account
    repository.start_job_run.side_effect = start_job_run
    repository.has_terminal_job.return_value = False
    repository.finish_job_run.side_effect = lambda run_id: events.append("finish_job_run")
    jobs = _jobs(monkeypatch, repository, _broker())
    jobs.state_engine.connect.return_value = nullcontext(Mock())
    monkeypatch.setattr("etf_backtest.live.jobs.acquire_job_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.jobs.release_job_lock", lambda *args: None)

    result = jobs._run_job(
        "startup_reconcile",
        date(2026, 8, 19),
        JobTriggerSource.RECOVERY,
        lambda: events.append("body") or "ok",
        None,
        deduplicate=False,
    )

    assert result == "ok"
    assert events == ["sync_account", "start_job_run", "body", "finish_job_run"]
    synced = repository.sync_account.call_args.kwargs
    assert synced["account_id"] == "account-1"
    assert synced["strategy"]["strategy_id"] == "beginner_rule"


def test_buy_phase_uses_reconciled_virtual_cash_not_unfilled_sell_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.pending_decision_for_date.return_value = {"decision_id": "rule-d", "strategy_id": "beginner_rule"}
    repository.list_order_intents_for_decision.return_value = ()
    repository.get_strategy.side_effect = [
        {"virtual_cash": Decimal("0")},
        {"virtual_cash": Decimal("600")},
    ]
    before = BrokerPositionSnapshot(
        symbol="SH.510300",
        total_quantity=100,
        available_quantity=100,
        today_buy_quantity=0,
        market_value=Decimal("1000"),
        turnover_rule=TurnoverRule.T1,
        captured_at=NOW,
    )
    after_partial_sell = BrokerPositionSnapshot(
        symbol="SH.510300",
        total_quantity=50,
        available_quantity=50,
        today_buy_quantity=0,
        market_value=Decimal("500"),
        turnover_rule=TurnoverRule.T1,
        captured_at=NOW,
    )
    repository.load_strategy_positions.side_effect = [(before,), (after_partial_sell,)]
    repository.strategy_reservations.return_value = (Decimal("0"), {}, ())
    _configure_targets(
        repository,
        {
            "rule-d": {
                "SH.510300": Decimal("0"),
                "SH.518880": Decimal("1"),
            }
        },
    )
    repository.transaction.return_value = nullcontext(Mock())
    repository.create_order_intent.side_effect = lambda intent, **kwargs: {
        "intent_id": intent.intent_key,
        "status": OrderIntentStatus.PLANNED,
    }
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    broker = _broker()
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation = reconciliation
    jobs.clock = lambda: NOW.replace(hour=14, minute=52)

    jobs._execute_pending_target(date(2026, 8, 19))

    submitted = [call.args[0] for call in broker.submit_order.call_args_list]
    assert [(intent.side, intent.symbol) for intent in submitted] == [(OrderSide.SELL, "SH.510300")]
    assert repository.get_strategy.call_count == 2


def test_buy_phase_is_skipped_when_sell_completion_crosses_stop_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.pending_decision_for_date.return_value = {"decision_id": "rule-d", "strategy_id": "beginner_rule"}
    repository.list_order_intents_for_decision.return_value = ()
    repository.get_strategy.return_value = {"virtual_cash": Decimal("1000")}
    repository.load_strategy_positions.return_value = ()
    repository.strategy_reservations.return_value = (Decimal("0"), {}, ())
    repository.transaction.return_value = nullcontext(Mock())
    _configure_targets(
        repository,
        {"rule-d": {"SH.510300": Decimal("0")}},
    )
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    jobs = _jobs(monkeypatch, repository, _broker())
    jobs.reconciliation = reconciliation
    clock_values = [
        NOW.replace(hour=14, minute=52),
        NOW.replace(hour=14, minute=52),
        NOW.replace(hour=14, minute=52),
        NOW.replace(hour=14, minute=56),
        NOW.replace(hour=14, minute=56),
    ]
    jobs.clock = Mock(side_effect=clock_values)

    jobs._execute_pending_target(date(2026, 8, 19))

    skipped = [call.kwargs["skip_reason"] for call in repository.finish_job_run.call_args_list if "skip_reason" in call.kwargs]
    assert skipped == ["MISSED_BUY_WINDOW"]
    jobs.broker.submit_order.assert_not_called()
    repository.pause_account.assert_not_called()


def test_external_active_order_is_not_cancelled_but_unknown_local_token_halts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_broker_order.return_value = None
    repository.get_intent_by_remark_token.return_value = None
    broker = _broker()
    external = BrokerOrderSnapshot(
        broker_order_id="external",
        symbol="SH.510300",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        remark_token="external",
    )
    broker.query_orders.return_value = QueryResult(success=True, records=(external,))
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation = reconciliation
    jobs._cancel_open_orders()
    broker.cancel_order.assert_not_called()
    local_unknown = BrokerOrderSnapshot(
        broker_order_id="local",
        symbol="SH.510300",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        remark_token="L" + "A" * 20,
    )
    broker.query_orders.return_value = QueryResult(success=True, records=(local_unknown,))
    with pytest.raises(RuntimeError, match="unknown local"):
        jobs._cancel_open_orders()
    repository.pause_account.assert_called()


def test_cancel_waits_until_the_broker_confirms_no_active_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_broker_order.return_value = {"intent_id": "intent-1"}
    repository.get_intent.return_value = {"account_id": "account-1"}
    broker = _broker()
    active = BrokerOrderSnapshot(
        broker_order_id="local-order",
        symbol="SH.510300",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        remark_token="L" + "A" * 20,
    )
    broker.query_orders.return_value = QueryResult(success=True, records=(active,))
    broker.query_orders.side_effect = [
        QueryResult(success=True, records=(active,)),
        QueryResult(success=True, records=(active,)),
        QueryResult(success=True),
    ]
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.side_effect = [
        ReconciliationReport(1, 0, active_broker_order_ids=("local-order",)),
        ReconciliationReport(1, 0),
    ]
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation = reconciliation
    jobs.sleep = Mock()

    jobs._cancel_open_orders()

    broker.cancel_order.assert_called_once_with("local-order")
    assert reconciliation.reconcile.call_count == 2
    jobs.sleep.assert_called_once()


def test_cancel_false_result_pauses_account(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_broker_order.return_value = {"intent_id": "intent-1"}
    repository.get_intent.return_value = {"account_id": "account-1"}
    broker = _broker()
    broker.cancel_order.return_value = False
    broker.query_orders.return_value = QueryResult(
        success=True,
        records=(
            BrokerOrderSnapshot(
                broker_order_id="local-order",
                symbol="SH.510300",
                side=OrderSide.BUY,
                requested_quantity=100,
                filled_quantity=0,
                limit_price=Decimal("10"),
                status=BrokerOrderStatus.PENDING,
                captured_at=NOW,
                remark_token="L" + "A" * 20,
            ),
        ),
    )
    jobs = _jobs(monkeypatch, repository, broker)

    with pytest.raises(RuntimeError, match="cancel request was not accepted"):
        jobs._cancel_open_orders()

    repository.pause_account.assert_called_once_with("account-1", "CANCEL_RESULT_UNKNOWN")


def test_cancel_timeout_pauses_account_and_fails_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_broker_order.return_value = {"intent_id": "intent-1"}
    repository.get_intent.return_value = {"account_id": "account-1"}
    broker = _broker()
    active = BrokerOrderSnapshot(
        broker_order_id="local-order",
        symbol="SH.510300",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        remark_token="L" + "A" * 20,
    )
    broker.query_orders.return_value = QueryResult(success=True, records=(active,))
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(
        1, 0, active_broker_order_ids=("local-order",)
    )
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation = reconciliation
    monkeypatch.setattr(
        "etf_backtest.live.jobs.time_module.monotonic",
        Mock(side_effect=[0.0, 61.0]),
    )

    with pytest.raises(RuntimeError, match="cancel confirmation timeout"):
        jobs._cancel_open_orders()

    repository.pause_account.assert_called_once_with("account-1", "CANCEL_CONFIRM_TIMEOUT")


def test_broker_query_exception_pauses_account(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Mock(spec=LiveStateRepository)
    jobs = _jobs(monkeypatch, repository, _broker())

    with pytest.raises(RuntimeError, match="query raised an exception"):
        jobs._broker_records(Mock(side_effect=OSError("socket closed")), "orders")

    repository.pause_account.assert_called_once_with("account-1", "BROKER_ORDERS_QUERY_FAILED")


def test_date_window_and_submit_unknown_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = Mock(spec=LiveStateRepository)
    jobs = _jobs(monkeypatch, repository, _broker())
    with pytest.raises(JobSkipped, match="STALE_SIGNAL_DATE"):
        jobs._prepare_signal(date(2026, 8, 18))
    intent = jobs.planner.plan(
        account_id="rule-xgboost-paper-v1",
        strategy_id="beginner_rule",
        decision_id="decision",
        execution_date=date(2026, 8, 19),
        symbols=("SH.510300",),
        target=TargetPortfolio({"SH.510300": Decimal("0.5")}),
        total_asset=Decimal("10000"),
        available_cash=Decimal("10000"),
        positions={},
        active_orders=(),
        valuation_prices={"SH.510300": Decimal("10")},
        limit_prices={"SH.510300": Decimal("10")},
        lot_size=100,
    )[0]
    jobs.broker.submit_order.return_value = SubmitOrderResult(
        SubmitOrderStatus.UNKNOWN, error="unknown"
    )
    with pytest.raises(RuntimeError, match="unknown"):
        jobs._submit_intent(intent, "intent-1", "account-1")
    repository.mark_intent_submit_unknown.assert_called_once_with("intent-1", "unknown")


def test_unknown_first_submission_stops_remaining_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.pending_decision_for_date.return_value = {"decision_id": "rule-d", "strategy_id": "beginner_rule"}
    repository.list_order_intents_for_decision.return_value = ()
    repository.get_strategy.return_value = {"virtual_cash": Decimal("50000")}
    repository.load_strategy_positions.return_value = ()
    repository.strategy_reservations.return_value = (Decimal("0"), {}, ())
    _configure_targets(
        repository,
        {
            "rule-d": {"SH.510300": Decimal("0.1"), "SH.518880": Decimal("0.1")},
        },
    )
    repository.transaction.return_value = nullcontext(Mock())
    repository.create_order_intent.side_effect = lambda intent, **kwargs: {
        "intent_id": intent.strategy_id,
        "status": OrderIntentStatus.PLANNED,
    }
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    broker = _broker()
    broker.submit_order.return_value = SubmitOrderResult(
        SubmitOrderStatus.UNKNOWN, error="network response lost"
    )
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation = reconciliation
    jobs.clock = lambda: NOW.replace(hour=14, minute=52)

    with pytest.raises(RuntimeError, match="network response lost"):
        jobs._execute_pending_target(date(2026, 8, 19))

    assert broker.submit_order.call_count == 1
    assert repository.create_order_intent.call_count == 2
    repository.pause_account.assert_called_once_with("account-1", "SUBMIT_RESULT_UNKNOWN")


def test_successful_strategy_step_is_skipped_on_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.has_terminal_job.side_effect = [False, True]
    jobs = _jobs(monkeypatch, repository, _broker())
    body = Mock(return_value="done")

    first = jobs._run_strategy_step(
        "strategy_buy",
        date(2026, 8, 19),
        "beginner_rule",
        JobTriggerSource.MANUAL,
        body,
    )
    second = jobs._run_strategy_step(
        "strategy_buy",
        date(2026, 8, 19),
        "beginner_rule",
        JobTriggerSource.RECOVERY,
        body,
    )

    assert first == "done"
    assert second is None
    body.assert_called_once()


def test_single_signal_persists_ledger_cash_and_failure_is_recorded(monkeypatch):
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_strategy.return_value = {"virtual_cash": Decimal("12345")}
    repository.load_strategy_positions.return_value = ()
    repository.transaction.return_value = nullcontext(Mock())
    repository.create_or_get_decision.return_value = {"decision_id": "decision-1"}
    evaluator = Mock()
    result = DailyDecisionResult(
        signal_date=NOW.date(), execution_date=date(2026, 8, 20), schedule_index=1,
        status=DecisionStatus.TARGET_CREATED,
        target_portfolio=TargetPortfolio({"SH.510300": Decimal("0.5")}),
    )
    evaluator.evaluate.return_value = result
    broker = _broker()
    jobs = _jobs(monkeypatch, repository, broker, evaluator)
    assert jobs._prepare_signal(NOW.date()) is result
    assert evaluator.evaluate.call_args.kwargs["virtual_cash"] == Decimal("12345")
    repository.save_target_positions.assert_called_once()
    assert repository.save_target_positions.call_args.args == ("decision-1", result.target_portfolio.weights)
    broker.query_asset.assert_not_called()
    evaluator.evaluate.side_effect = ValueError("signal failed")
    with pytest.raises(ValueError, match="signal failed"):
        jobs._prepare_signal(NOW.date())
    assert isinstance(repository.finish_job_run.call_args.kwargs["error"], ValueError)


def test_single_snapshot_and_failure_are_recorded(monkeypatch):
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_strategy.return_value = {"virtual_cash": Decimal("12345")}
    repository.load_strategy_positions.return_value = ()
    jobs = _jobs(monkeypatch, repository, _broker())
    jobs._snapshot_eod(NOW.date())
    saved = repository.save_strategy_daily_snapshot.call_args.kwargs
    assert saved["strategy_id"] == "beginner_rule"
    assert saved["virtual_cash"] == Decimal("12345")
    assert saved["positions"] == []
    repository.save_strategy_daily_snapshot.side_effect = ValueError("snapshot failed")
    with pytest.raises(ValueError, match="snapshot failed"):
        jobs._snapshot_eod(NOW.date())
    assert isinstance(repository.finish_job_run.call_args.kwargs["error"], ValueError)


def test_orders_use_only_ledger_cash_after_reservations_and_fees(monkeypatch):
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.get_strategy.return_value = {"virtual_cash": Decimal("3000")}
    repository.load_strategy_positions.return_value = ()
    repository.pending_decision_for_date.return_value = {"decision_id": "rule-d", "strategy_id": "beginner_rule"}
    repository.list_order_intents_for_decision.return_value = ()
    repository.strategy_reservations.return_value = (
        Decimal("1000"), {}, ({"limit_price": Decimal("10"), "remaining_quantity": 100,
                               "symbol": "SH.518880", "side": OrderSide.BUY,
                               "intent_id": "reserved", "remark_token": "L" + "A" * 20},),
    )
    _configure_targets(repository, {"rule-d": {"SH.510300": Decimal("1")}})
    repository.transaction.return_value = nullcontext(Mock())
    repository.create_order_intent.side_effect = lambda intent, **kwargs: {
        "intent_id": intent.intent_key, "status": OrderIntentStatus.PLANNED,
    }
    broker = _broker()
    broker.query_asset.side_effect = AssertionError("must not query broker cash")
    jobs = _jobs(monkeypatch, repository, broker)
    jobs.reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    jobs.clock = lambda: NOW.replace(hour=14, minute=52)
    jobs._execute_pending_target(NOW.date())
    intent = broker.submit_order.call_args.args[0]
    # 3000 - 1000 reserved - 5 fee leaves 1995; only one lot fits at the buy limit.
    assert intent.side is OrderSide.BUY and intent.requested_quantity == 100
    broker.submit_order.assert_called_once()
    broker.query_asset.assert_not_called()


def test_failed_sell_phase_never_starts_buy_and_restart_skips_completed_sell(monkeypatch):
    repository = Mock(spec=LiveStateRepository)
    repository.get_account.return_value = _account()
    repository.pending_decision_for_date.return_value = {"decision_id": "rule-d", "strategy_id": "beginner_rule"}
    repository.list_order_intents_for_decision.return_value = ()
    jobs = _jobs(monkeypatch, repository, _broker())
    jobs.clock = lambda: NOW.replace(hour=14, minute=52)
    jobs.reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    jobs._execute_strategy_side = Mock(side_effect=ValueError("sell failed"))
    with pytest.raises(ValueError, match="sell failed"):
        jobs._execute_pending_target(NOW.date())
    assert jobs._execute_strategy_side.call_count == 1
    assert jobs._execute_strategy_side.call_args.kwargs["side"] is OrderSide.SELL
    jobs._execute_strategy_side = Mock()
    jobs._wait_for_phase = Mock()
    repository.has_terminal_job.side_effect = lambda account, job, day, **kw: job == "strategy_sell"
    jobs._execute_pending_target(NOW.date())
    jobs._execute_strategy_side.assert_called_once()
    assert jobs._execute_strategy_side.call_args.kwargs["side"] is OrderSide.BUY
