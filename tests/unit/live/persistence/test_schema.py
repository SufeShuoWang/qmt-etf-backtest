from etf_backtest.live.persistence.schema import LIVE_TABLES



def test_current_schema_contains_account_strategy_ledgers_fees_and_direct_identity() -> None:
    tables = {table.name: table for table in LIVE_TABLES}
    assert len(tables) == 13
    assert {
        "live_account",
        "live_strategy",
        "live_strategy_position",
        "live_strategy_account_snapshot",
        "live_strategy_position_snapshot",
    } <= set(tables)
    assert "strategy_id" in tables["live_decision"].c
    assert "strategy_id" in tables["live_order_intent"].c
    for name in (
        "live_decision",
        "live_target_position",
        "live_order_intent",
        "live_broker_order",
        "live_broker_trade",
    ):
        assert {"account_id", "strategy_id"} <= set(tables[name].c.keys())
    assert "live_deployment" not in tables
    assert all("deployment_id" not in table.c for table in tables.values())
    assert all(
        name not in table.c
        for table in tables.values()
        for name in ("config_hash", "source_hash", "model_bundle_hash")
    )
    assert {"commission", "stamp_duty", "total_fee"} <= set(tables["live_broker_trade"].c.keys())
    assert tables["live_job_run"].c.strategy_id.nullable
    assert {"execution_valuation_price", "target_quantity"} <= set(
        tables["live_target_position"].c.keys()
    )
    assert any(
        set(constraint.columns.keys()) == {"account_id", "strategy_id", "signal_date"}
        for constraint in tables["live_decision"].constraints
    )
    assert any(
        tuple(index.columns.keys()) == ("account_id", "strategy_id", "job_type", "trade_date")
        for index in tables["live_job_run"].indexes
    )
    assert any(
        set(constraint.columns.keys()) == {"account_id", "strategy_id"}
        for constraint in tables["live_job_run"].constraints
    )


def test_snapshot_unique_keys_match_required_daily_identity() -> None:
    tables = {table.name: table for table in LIVE_TABLES}
    expected = {
        "live_strategy_account_snapshot": {"account_id", "strategy_id", "trading_date"},
        "live_strategy_position_snapshot": {
            "account_id",
            "strategy_id",
            "trading_date",
            "symbol",
        },
    }
    for name, columns in expected.items():
        assert any(
            set(constraint.columns.keys()) == columns for constraint in tables[name].constraints
        )
