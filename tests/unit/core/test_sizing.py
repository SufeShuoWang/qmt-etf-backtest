from decimal import Decimal

from etf_backtest.core.sizing import calculate_order_deltas, calculate_target_quantities


def test_target_quantity_is_rounded_before_subtracting_current_quantity() -> None:
    targets = calculate_target_quantities(
        target_weights={"510300.SH": Decimal("0.35")},
        total_asset=Decimal("5000"),
        valuation_prices={"SH.510300": Decimal("5")},
        lot_size=100,
    )

    assert targets == {"SH.510300": 300}
    assert calculate_order_deltas(
        target_quantities=targets,
        current_quantities={"SH.510300": 500},
    ) == {"SH.510300": -200}


def test_zero_target_liquidates_and_omitted_holding_is_unchanged() -> None:
    targets = calculate_target_quantities(
        target_weights={"510300.SH": Decimal("0")},
        total_asset=Decimal("5000"),
        valuation_prices={"SH.510300": Decimal("5")},
        lot_size=100,
    )

    assert calculate_order_deltas(
        target_quantities=targets,
        current_quantities={"SH.510300": 500, "SH.518880": 200},
    ) == {"SH.510300": -500}
