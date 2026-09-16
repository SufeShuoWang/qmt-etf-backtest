"""回测与 Live 共用的整手目标股数计算。"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal

from etf_backtest.config.schema import normalize_symbol


class TargetSizingError(ValueError):
    """目标权重无法转换为确定的整手股数。"""


# 数量公式为 floor(总资产 × 目标权重 ÷ 估值价 ÷ 整手) × 整手；只处理显式列出的证券。
def calculate_target_quantities(
    *,
    target_weights: Mapping[str, Decimal],
    total_asset: Decimal,
    valuation_prices: Mapping[str, Decimal],
    lot_size: int,
) -> dict[str, int]:
    """仅为显式目标计算整手股数；目标中未出现的证券不参与调仓。"""

    if not isinstance(total_asset, Decimal):
        raise TypeError("total_asset must be Decimal")
    if not total_asset.is_finite() or total_asset < 0:
        raise TargetSizingError("total_asset must be finite and non-negative")
    if type(lot_size) is not int or lot_size <= 0:
        raise TargetSizingError("lot_size must be a positive integer")

    prices = {normalize_symbol(symbol): price for symbol, price in valuation_prices.items()}
    quantities: dict[str, int] = {}
    for supplied_symbol, weight in target_weights.items():
        symbol = normalize_symbol(supplied_symbol)
        if symbol in quantities:
            raise TargetSizingError(f"duplicate target symbol: {symbol}")
        if not isinstance(weight, Decimal):
            raise TypeError("target weights must be Decimal")
        if not weight.is_finite() or weight < 0:
            raise TargetSizingError("target weights must be finite and non-negative")
        price = prices.get(symbol)
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            raise TargetSizingError(f"valuation_price is unavailable for {symbol}")
        if weight == 0 or total_asset == 0:
            quantities[symbol] = 0
            continue
        raw_lots = total_asset * weight / price / lot_size
        quantities[symbol] = int(raw_lots.to_integral_value(rounding=ROUND_FLOOR)) * lot_size
    return dict(sorted(quantities.items()))


def calculate_order_deltas(
    *,
    target_quantities: Mapping[str, int],
    current_quantities: Mapping[str, int],
) -> dict[str, int]:
    """返回显式目标股数减当前股数；未出现在目标中的持仓保持不变。"""

    current = {
        normalize_symbol(symbol): quantity for symbol, quantity in current_quantities.items()
    }
    deltas: dict[str, int] = {}
    for supplied_symbol, target_quantity in target_quantities.items():
        symbol = normalize_symbol(supplied_symbol)
        if symbol in deltas:
            raise TargetSizingError(f"duplicate target symbol: {symbol}")
        if type(target_quantity) is not int or target_quantity < 0:
            raise TargetSizingError("target quantities must be non-negative integers")
        current_quantity = current.get(symbol, 0)
        if type(current_quantity) is not int or current_quantity < 0:
            raise TargetSizingError("current quantities must be non-negative integers")
        deltas[symbol] = target_quantity - current_quantity
    return dict(sorted(deltas.items()))


__all__ = ["TargetSizingError", "calculate_order_deltas", "calculate_target_quantities"]
