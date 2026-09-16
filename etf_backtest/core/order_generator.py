"""将显式调仓目标转换为下一收盘时点的数量请求。"""

from __future__ import annotations


import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.core.account import AccountSnapshot
from etf_backtest.core.order import Order, OrderSide
from etf_backtest.core.sizing import calculate_order_deltas, calculate_target_quantities
from etf_backtest.core.target import TargetPortfolio


class OrderGenerationError(ValueError):
    """目标转换为订单时缺少估值输入。"""


# 根据目标决策和证券方向生成稳定订单 ID，使同一决策的订单可追溯。
def _order_id(*, signal_date: date, execution_date: date, symbol: str, side: OrderSide) -> str:
    payload = f"{signal_date.isoformat()}|{execution_date.isoformat()}|{symbol}|{side.value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class OrderGenerator:
    """生成确定性数量请求；只有规则引擎可以调整其数量。"""

    lot_size: int = 100

    # 校验订单生成器的整手数量为正整数。
    def __post_init__(self) -> None:
        if type(self.lot_size) is not int or self.lot_size <= 0:
            raise ValueError("lot_size must be a positive integer")

    # 按执行日总资产、价格与整手计算显式目标证券的数量差，生成先卖后买的订单；省略证券保持数量。
    def generate(
        self,
        *,
        target_portfolio: TargetPortfolio,
        valuation_snapshot: AccountSnapshot,
        signal_date: date,
        execution_date: date,
    ) -> tuple[Order, ...]:
        if not isinstance(target_portfolio, TargetPortfolio):
            raise TypeError("target_portfolio must be TargetPortfolio")
        if not isinstance(valuation_snapshot, AccountSnapshot):
            raise TypeError("valuation_snapshot must be AccountSnapshot")
        signal = _plain_date(signal_date, "signal_date")
        execution = _plain_date(execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")

        symbols = set(target_portfolio.weights)
        missing = sorted(symbols - set(valuation_snapshot.mark_close_prices))
        if missing:
            raise OrderGenerationError("valuation snapshot lacks prices for: " + ", ".join(missing))

        target_quantities = calculate_target_quantities(
            target_weights=target_portfolio.weights,
            total_asset=valuation_snapshot.total_asset,
            valuation_prices=valuation_snapshot.mark_close_prices,
            lot_size=self.lot_size,
        )
        deltas = calculate_order_deltas(
            target_quantities=target_quantities,
            current_quantities={
                symbol: position.total_quantity
                for symbol, position in valuation_snapshot.positions.items()
            },
        )

        orders: list[Order] = []
        for symbol, delta in deltas.items():
            current_value = valuation_snapshot.position_values.get(symbol, Decimal("0"))
            target_weight = target_portfolio.weight_for(symbol)
            if target_weight is None:  # pragma: no cover - symbols 直接来自显式目标映射
                raise AssertionError("explicit target weight unexpectedly missing")
            target_value = valuation_snapshot.total_asset * target_weight
            gap = target_value - current_value
            requested = abs(delta)
            if requested == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(
                Order(
                    order_id=_order_id(
                        signal_date=signal,
                        execution_date=execution,
                        symbol=symbol,
                        side=side,
                    ),
                    signal_date=signal,
                    execution_date=execution,
                    symbol=symbol,
                    side=side,
                    requested_quantity=requested,
                    target_value_gap=gap,
                )
            )
        return tuple(sorted(orders, key=lambda item: (item.side is OrderSide.BUY, item.symbol)))


__all__ = ["OrderGenerationError", "OrderGenerator"]
