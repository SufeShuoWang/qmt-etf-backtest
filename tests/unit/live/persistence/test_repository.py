from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE, FeeConfig
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.persistence.schema import metadata
from etf_backtest.live.state import (
    AccountStatus,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerTradeSnapshot,
    JobTriggerSource,
    OrderIntent,
    OrderIntentStatus,
)

NOW = datetime(2026, 8, 19, 15, 0, tzinfo=MARKET_TIMEZONE)


@pytest.fixture
def repository() -> LiveStateRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return LiveStateRepository(engine, fee_model=FeeModel(FeeConfig()))


def _activate(repository: LiveStateRepository) -> None:
    repository.sync_account(
        account_id="account-1", mode="PAPER", account_type="STOCK",
        strategy={
                "strategy_id": "rule",
                "case": "rule",
                "initial_capital": Decimal("5000"),
                "experiment_path": "rule.yaml",
                "model_backend": None,
                "bundle_path": None,
                "model_id": None,
            },
    )


def _intent(
    repository: LiveStateRepository, strategy_id: str, side: OrderSide, symbol: str = "SH.510300"
) -> tuple[str, OrderIntent]:
    decision_id = f"decision-{strategy_id}-{side.value}"
    decision = repository.create_or_get_decision(
        decision_id=decision_id,
        account_id="account-1",
        strategy_id=strategy_id,
        signal_date=date(2026, 8, 18),
        execution_date=date(2026, 8, 19),
        schedule_index=1,
        status=DecisionStatus.TARGET_CREATED,
        data_as_of=date(2026, 8, 18),
    )
    decision_id = str(decision["decision_id"])
    intent = OrderIntent(
        intent_key=(strategy_id + side.value).ljust(64, "0"),
        remark_token=("L" + strategy_id.upper() + side.value).ljust(21, "A"),
        account_id="account-1",
        strategy_id=strategy_id,
        decision_id=decision_id,
        execution_date=date(2026, 8, 19),
        symbol=symbol,
        side=side,
        requested_quantity=100,
        target_weight=Decimal("0.2"),
        valuation_price=Decimal("10"),
        limit_price=Decimal("10"),
    )
    saved = repository.create_order_intent(intent)
    return str(saved["intent_id"]), intent


def _trade(
    identifier: str, side: OrderSide, quantity: int, *, symbol: str = "SH.510300", day: int = 19
) -> BrokerTradeSnapshot:
    return BrokerTradeSnapshot(
        broker_trade_id=identifier,
        broker_order_id="order-1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=Decimal("10"),
        traded_at=NOW.replace(day=day),
    )


def test_unresolved_intent_query_has_unambiguous_join(repository: LiveStateRepository) -> None:
    _activate(repository)

    assert repository.list_unresolved_intents(account_id="account-1") == ()


def test_strategy_job_identity_and_execution_target_freeze_are_idempotent(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    trade_date = date(2026, 8, 19)
    run_id = repository.start_job_run(
        account_id="account-1",
        strategy_id="rule",
        job_type="strategy_sell",
        trade_date=trade_date,
        trigger_source=JobTriggerSource.MANUAL,
    )
    repository.finish_job_run(run_id)
    assert repository.has_terminal_job("account-1", "strategy_sell", trade_date, strategy_id="rule")
    assert not repository.has_terminal_job("account-1", "strategy_sell", trade_date)

    repository.create_or_get_decision(
        decision_id="freeze-decision",
        account_id="account-1",
        strategy_id="rule",
        signal_date=date(2026, 8, 18),
        execution_date=trade_date,
        schedule_index=1,
        status=DecisionStatus.TARGET_CREATED,
        data_as_of=date(2026, 8, 18),
    )
    repository.save_target_positions("freeze-decision", {"SH.510300": Decimal("0.35")})
    first = repository.freeze_execution_targets(
        "freeze-decision",
        total_asset=Decimal("5000"),
        valuation_prices={"SH.510300": Decimal("5")},
        lot_size=100,
    )
    second = repository.freeze_execution_targets(
        "freeze-decision",
        total_asset=Decimal("9999"),
        valuation_prices={"SH.510300": Decimal("9")},
        lot_size=100,
    )

    assert first == second == {"SH.510300": (Decimal("5"), 300)}


def test_strategy_initial_capital_is_immutable(repository: LiveStateRepository) -> None:
    _activate(repository)
    with pytest.raises(ValueError, match="initial_capital is immutable"):
        repository.sync_account(
        account_id="account-1", mode="PAPER", account_type="STOCK",
        strategy={
                    "strategy_id": "rule",
                    "case": "rule",
                    "initial_capital": Decimal("4000"),
                    "experiment_path": "updated-rule.yaml",
                    "model_backend": None,
                    "bundle_path": None,
                    "model_id": None,
                    },
    )


def test_duplicate_trade_is_idempotent_and_restart_preserves_cash(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    intent_id, _ = _intent(repository, "rule", OrderSide.BUY)
    trade = _trade("trade-1", OrderSide.BUY, 100)
    assert repository.record_strategy_trade_if_absent(
        account_id="account-1", intent_id=intent_id, trade=trade, turnover_rule=TurnoverRule.T1
    ).inserted
    assert not repository.record_strategy_trade_if_absent(
        account_id="account-1", intent_id=intent_id, trade=trade, turnover_rule=TurnoverRule.T1
    ).inserted
    assert repository.get_strategy("account-1", "rule")["virtual_cash"] == Decimal("3995.00000000")  # type: ignore[index]
    _activate(repository)
    assert repository.get_strategy("account-1", "rule")["virtual_cash"] == Decimal("3995.00000000")
    saved_trade = repository.list_broker_trades_for_intent(intent_id)[0]
    assert (saved_trade["account_id"], saved_trade["strategy_id"]) == (
        "account-1",
        "rule",
    )
    assert (
        saved_trade["commission"],
        saved_trade["stamp_duty"],
        saved_trade["total_fee"],
    ) == (Decimal("5.00000000"), Decimal("0E-8"), Decimal("5.00000000"))


def test_partial_t1_buys_settle_next_day_and_sell_cannot_borrow(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    buy_id, _ = _intent(repository, "rule", OrderSide.BUY)
    repository.record_strategy_trade_if_absent(
        account_id="account-1",
        intent_id=buy_id,
        trade=_trade("t1", OrderSide.BUY, 40),
        turnover_rule=TurnoverRule.T1,
    )
    repository.record_strategy_trade_if_absent(
        account_id="account-1",
        intent_id=buy_id,
        trade=_trade("t2", OrderSide.BUY, 60),
        turnover_rule=TurnoverRule.T1,
    )
    strategy = repository.get_strategy("account-1", "rule")
    assert strategy is not None
    assert strategy["virtual_cash"] == Decimal("3990.00000000")
    position = repository.load_strategy_positions(
        "account-1", "rule", turnover_rules={"SH.510300": TurnoverRule.T1}
    )[0]
    assert (position.total_quantity, position.available_quantity, position.today_buy_quantity) == (
        100,
        0,
        100,
    )
    assert position.average_cost == Decimal("10.10000000")
    repository.settle_strategy_positions(
        "account-1", "rule", date(2026, 8, 20), turnover_rules={"SH.510300": TurnoverRule.T1}
    )
    position = repository.load_strategy_positions(
        "account-1", "rule", turnover_rules={"SH.510300": TurnoverRule.T1}
    )[0]
    assert (position.available_quantity, position.today_buy_quantity) == (100, 0)
    sell_id, _ = _intent(repository, "rule", OrderSide.SELL)
    with pytest.raises(ValueError, match="available"):
        repository.record_strategy_trade_if_absent(
            account_id="account-1",
            intent_id=sell_id,
            trade=_trade("sell", OrderSide.SELL, 101),
            turnover_rule=TurnoverRule.T1,
        )


def test_t0_buy_is_immediately_available(repository: LiveStateRepository) -> None:
    _activate(repository)
    intent_id, _ = _intent(repository, "rule", OrderSide.BUY, symbol="SH.518880")
    repository.record_strategy_trade_if_absent(
        account_id="account-1",
        intent_id=intent_id,
        trade=_trade("gold", OrderSide.BUY, 100, symbol="SH.518880"),
        turnover_rule=TurnoverRule.T0,
    )
    position = repository.load_strategy_positions(
        "account-1", "rule", turnover_rules={"SH.518880": TurnoverRule.T0}
    )[0]
    assert (position.total_quantity, position.available_quantity, position.today_buy_quantity) == (
        100,
        100,
        0,
    )


def test_fee_breach_persists_broker_fact_and_pauses_account(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    intent_id, _ = _intent(repository, "rule", OrderSide.BUY)

    applied = repository.record_strategy_trade_if_absent(
        account_id="account-1",
        intent_id=intent_id,
        trade=_trade("too-large", OrderSide.BUY, 500),
        turnover_rule=TurnoverRule.T1,
    )

    assert applied.inserted and applied.cash_breach
    assert applied.virtual_cash == Decimal("-5.000")
    account = repository.get_account("account-1")
    assert account is not None
    assert account["status"] is AccountStatus.PAUSED
    assert account["pause_reason"] == "VIRTUAL_CASH_NEGATIVE_AFTER_FEES"
    assert len(repository.list_broker_trades_for_intent(intent_id)) == 1


def test_late_order_callback_cannot_regress_terminal_order_or_intent(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    intent_id, intent = _intent(repository, "rule", OrderSide.BUY)
    repository.mark_intent_submitting(intent_id)
    filled = BrokerOrderSnapshot(
        broker_order_id="order-terminal",
        symbol=intent.symbol,
        side=intent.side,
        requested_quantity=intent.requested_quantity,
        filled_quantity=intent.requested_quantity,
        limit_price=intent.limit_price,
        status=BrokerOrderStatus.FILLED,
        captured_at=NOW,
        remark_token=intent.remark_token,
        broker_order_sysid="sys-1",
        traded_price=Decimal("9.99"),
    )
    repository.bind_broker_order(
        account_id="account-1",
        intent_id=intent_id,
        remark_token=intent.remark_token,
        order=filled,
    )
    repository.mark_intent_completed(intent_id)

    late_pending = BrokerOrderSnapshot(
        broker_order_id="order-terminal",
        symbol=intent.symbol,
        side=intent.side,
        requested_quantity=intent.requested_quantity,
        filled_quantity=0,
        limit_price=intent.limit_price,
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW - timedelta(seconds=1),
        remark_token=intent.remark_token,
    )
    repository.bind_broker_order(
        account_id="account-1",
        intent_id=intent_id,
        remark_token=intent.remark_token,
        order=late_pending,
    )

    saved_intent = repository.get_intent(intent_id)
    saved_order = repository.get_broker_order("account-1", "order-terminal")
    assert saved_intent is not None and saved_order is not None
    assert saved_intent["status"] is OrderIntentStatus.COMPLETED
    assert saved_order["status"] is BrokerOrderStatus.FILLED
    assert saved_order["filled_quantity"] == 100
    assert saved_order["order_sysid"] == "sys-1"
    assert saved_order["average_fill_price"] == Decimal("9.99000000")


def test_daily_snapshots_keep_dates_and_replace_same_day_without_duplicates(
    repository: LiveStateRepository,
) -> None:
    _activate(repository)
    rows = (
        {
            "symbol": "SH.510300",
            "total_quantity": 100,
            "available_quantity": 100,
            "today_buy_quantity": 0,
            "average_cost": Decimal("9"),
            "close_price": Decimal("10"),
            "market_value": Decimal("1000"),
        },
    )
    repository.save_strategy_daily_snapshot(
        account_id="account-1",
        strategy_id="rule",
        trading_date=date(2026, 8, 19),
        virtual_cash=Decimal("4000"),
        positions=rows,
    )
    repository.save_strategy_daily_snapshot(
        account_id="account-1",
        strategy_id="rule",
        trading_date=date(2026, 8, 19),
        virtual_cash=Decimal("4000"),
        positions=rows,
    )
    repository.save_strategy_daily_snapshot(
        account_id="account-1",
        strategy_id="rule",
        trading_date=date(2026, 8, 20),
        virtual_cash=Decimal("4000"),
        positions=rows,
    )
    from sqlalchemy import select
    from etf_backtest.live.persistence.schema import live_strategy_account_snapshot, live_strategy_position_snapshot
    with repository.transaction() as connection:
        snapshots = connection.execute(select(live_strategy_account_snapshot)).mappings().all()
        positions = connection.execute(select(live_strategy_position_snapshot)).mappings().all()
    assert len(snapshots) == len(positions) == 2
    assert all(row["total_asset"] == Decimal("5000") for row in snapshots)


def test_second_strategy_is_rejected_without_changing_existing_cash(repository):
    _activate(repository)
    with pytest.raises(ValueError, match="another strategy"):
        repository.sync_account(
            account_id="account-1", mode="PAPER", account_type="STOCK",
            strategy={"strategy_id": "other", "case": "rule", "initial_capital": Decimal("1000"),
                      "experiment_path": "other.yaml"},
        )
    assert repository.get_strategy("account-1", "other") is None
    assert repository.get_strategy("account-1", "rule")["virtual_cash"] == Decimal("5000")


def test_pending_decision_returns_one_row_and_rejects_ambiguous_history(repository):
    from sqlalchemy.exc import MultipleResultsFound

    _activate(repository)
    assert repository.pending_decision_for_date("account-1", NOW.date()) is None
    _intent(repository, "rule", OrderSide.BUY)
    row = repository.pending_decision_for_date("account-1", NOW.date())
    assert row["strategy_id"] == "rule"
    repository.create_or_get_decision(
        decision_id="another-date", account_id="account-1", strategy_id="rule",
        signal_date=date(2026, 8, 17), execution_date=NOW.date(), schedule_index=0,
        status=DecisionStatus.TARGET_CREATED, data_as_of=date(2026, 8, 17),
    )
    with pytest.raises(MultipleResultsFound):
        repository.pending_decision_for_date("account-1", NOW.date())
