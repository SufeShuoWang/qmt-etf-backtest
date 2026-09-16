"""根据显式调仓目标规划一批确定性的实盘订单意图。"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_FLOOR, Decimal

from etf_backtest.config.schema import FeeConfig, normalize_symbol
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.order import OrderSide
from etf_backtest.core.sizing import calculate_order_deltas, calculate_target_quantities
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.state import BrokerOrderSnapshot, BrokerPositionSnapshot, OrderIntent


# 根据账户、策略、决策、日期、证券和方向生成稳定意图键，用于重复调用去重。
def generate_intent_key(
    *,
    account_id: str,
    strategy_id: str,
    execution_date: date,
    decision_id: str,
    symbol: str,
    side: OrderSide,
) -> str:
    canonical_symbol = normalize_symbol(symbol)
    fields = (
        account_id,
        strategy_id,
        execution_date.strftime("%Y%m%d"),
        decision_id,
        canonical_symbol,
        side.value,
    )
    if any(not field or "|" in field for field in fields):
        raise ValueError("intent key fields must be non-empty and cannot contain '|'")
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


# 把意图键编码成券商备注令牌，便于回报与本地订单重新关联。
def generate_remark_token(intent_key: str) -> str:
    try:
        digest = bytes.fromhex(intent_key)
    except ValueError as error:
        raise ValueError("intent_key must be a SHA-256 hex digest") from error
    if len(digest) != 32:
        raise ValueError("intent_key must be a SHA-256 hex digest")
    return "L" + base64.b32encode(digest).decode("ascii")[:20]


# 校验执行计划使用的估值价或限价为有限正值。
def _positive_price(prices: Mapping[str, Decimal], symbol: str, name: str) -> Decimal:
    price = prices.get(symbol)
    if price is None or not price.is_finite() or price <= 0:
        raise ValueError(f"{name} is unavailable for {symbol}")
    return price


# 把数量向下约束到指定整手倍数。
def _whole_lots(quantity: int, lot_size: int) -> int:
    return max(0, quantity // lot_size * lot_size)


# 把现有持仓与活动订单剩余数量合并成预计持仓，避免重复买卖同一目标。
def _projected_quantities(
    *,
    symbols: tuple[str, ...],
    positions: Mapping[str, BrokerPositionSnapshot],
    active_orders: Sequence[BrokerOrderSnapshot],
) -> dict[str, int]:
    projected = {
        symbol: positions[symbol].total_quantity if symbol in positions else 0 for symbol in symbols
    }
    for order in active_orders:
        if order.symbol not in projected or not order.status.is_active:
            continue
        direction = 1 if order.side is OrderSide.BUY else -1
        projected[order.symbol] += direction * order.remaining_quantity
        if projected[order.symbol] < 0:
            raise ValueError("active orders project a negative position quantity")
    return projected


def effective_target_weights(
    *,
    symbols: Sequence[str],
    target: TargetPortfolio,
    total_asset: Decimal,
    positions: Mapping[str, BrokerPositionSnapshot],
    active_orders: Sequence[BrokerOrderSnapshot],
    valuation_prices: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    """合并显式目标与未调仓持仓，供 PAPER 总仓位风控使用。"""

    frozen_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
    if not frozen_symbols or total_asset <= 0:
        raise ValueError("symbols and total_asset must be positive")
    if any(symbol not in frozen_symbols for symbol in target.weights):
        raise ValueError("target contains a symbol outside the frozen universe")
    projected = _projected_quantities(
        symbols=frozen_symbols,
        positions=positions,
        active_orders=active_orders,
    )
    effective: dict[str, Decimal] = {}
    for symbol in frozen_symbols:
        explicit_weight = target.weight_for(symbol)
        if explicit_weight is not None:
            effective[symbol] = explicit_weight
            continue
        quantity = projected[symbol]
        if quantity == 0:
            effective[symbol] = Decimal("0")
            continue
        effective[symbol] = (
            Decimal(quantity)
            * _positive_price(valuation_prices, symbol, "valuation_price")
            / total_asset
        )
    return effective


class LiveRebalancePlanner:
    """创建单批先卖后买的订单意图，不负责提交或持久化。"""

    # 保存费用模型，供买入可负担数量和现金占用测算。
    def __init__(self, fee_model: FeeModel | None = None) -> None:
        self._fee_model = fee_model or FeeModel(FeeConfig())

    # 依据显式目标或冻结目标数量、持仓和活动委托生成分方向订单意图，考虑整手、可卖数量及费用后的现金。
    def plan(
        self,
        *,
        account_id: str,
        strategy_id: str,
        decision_id: str,
        execution_date: date,
        symbols: Sequence[str],
        target: TargetPortfolio,
        total_asset: Decimal,
        available_cash: Decimal,
        positions: Mapping[str, BrokerPositionSnapshot],
        active_orders: Sequence[BrokerOrderSnapshot],
        valuation_prices: Mapping[str, Decimal],
        limit_prices: Mapping[str, Decimal],
        lot_size: int,
        target_quantities: Mapping[str, int] | None = None,
        side: OrderSide | None = None,
    ) -> tuple[OrderIntent, ...]:
        frozen_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
        if not frozen_symbols:
            raise ValueError("symbols must not be empty")
        if any(symbol not in frozen_symbols for symbol in target.weights):
            raise ValueError("target contains a symbol outside the frozen universe")
        if lot_size <= 0 or total_asset <= 0 or available_cash < 0:
            raise ValueError("lot_size and total_asset must be positive; cash cannot be negative")

        projected = _projected_quantities(
            symbols=frozen_symbols,
            positions=positions,
            active_orders=active_orders,
        )

        calculated_targets = (
            calculate_target_quantities(
                target_weights=target.weights,
                total_asset=total_asset,
                valuation_prices=valuation_prices,
                lot_size=lot_size,
            )
            if target_quantities is None
            else {
                normalize_symbol(symbol): quantity for symbol, quantity in target_quantities.items()
            }
        )
        if set(calculated_targets) != set(target.weights):
            raise ValueError("target quantities must exactly cover explicit target weights")
        deltas = calculate_order_deltas(
            target_quantities=calculated_targets,
            current_quantities=projected,
        )

        intents: list[OrderIntent] = []
        requested_sides = (side,) if side is not None else (OrderSide.SELL, OrderSide.BUY)
        for requested_side in requested_sides:
            for symbol in sorted(deltas):
                delta = deltas[symbol]
                desired = -delta if requested_side is OrderSide.SELL else delta
                if desired <= 0:
                    continue
                valuation_price = _positive_price(valuation_prices, symbol, "valuation_price")
                limit_price = _positive_price(limit_prices, symbol, "limit_price")
                quantity = _whole_lots(desired, lot_size)
                if requested_side is OrderSide.SELL:
                    available = positions[symbol].available_quantity if symbol in positions else 0
                    quantity = _whole_lots(min(quantity, available), lot_size)
                else:
                    affordable = (
                        int(
                            (available_cash / limit_price / lot_size).to_integral_value(
                                rounding=ROUND_FLOOR
                            )
                        )
                        * lot_size
                    )
                    while affordable > 0:
                        amount = limit_price * affordable
                        fee = self._fee_model.calculate(
                            trade_amount=amount, side=OrderSide.BUY
                        ).total
                        if amount + fee <= available_cash:
                            break
                        affordable -= lot_size
                    quantity = min(quantity, affordable)
                if quantity <= 0:
                    continue
                intent_key = generate_intent_key(
                    account_id=account_id,
                    strategy_id=strategy_id,
                    execution_date=execution_date,
                    decision_id=decision_id,
                    symbol=symbol,
                    side=requested_side,
                )
                intents.append(
                    OrderIntent(
                        intent_key=intent_key,
                        remark_token=generate_remark_token(intent_key),
                        account_id=account_id,
                        strategy_id=strategy_id,
                        decision_id=decision_id,
                        execution_date=execution_date,
                        symbol=symbol,
                        side=requested_side,
                        requested_quantity=quantity,
                        target_weight=target.weights[symbol],
                        valuation_price=valuation_price,
                        limit_price=limit_price,
                    )
                )
                if requested_side is OrderSide.BUY:
                    amount = quantity * limit_price
                    available_cash -= (
                        amount
                        + self._fee_model.calculate(trade_amount=amount, side=requested_side).total
                    )
        return tuple(intents)


__all__ = [
    "LiveRebalancePlanner",
    "effective_target_weights",
    "generate_intent_key",
    "generate_remark_token",
]
