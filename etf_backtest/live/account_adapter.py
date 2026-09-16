"""将券商或虚拟账户转换为不含价格的只读策略账户视图。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.live.state import BrokerAssetSnapshot, BrokerPositionSnapshot
from etf_backtest.strategy.context import AccountPositionView, AccountView


# 携带适配后的只读账户视图、当前权重、总资产和券商持仓索引，供信号计算使用。
@dataclass(frozen=True, slots=True)
class AdaptedAccountState:
    account_view: AccountView
    current_weights_by_symbol: Mapping[str, Decimal]
    total_asset: Decimal
    positions_by_symbol: Mapping[str, BrokerPositionSnapshot]


# 将券商资产与持仓适配为策略只读输入，按给定价格重算当前权重。
def adapt_broker_account(
    *,
    asset: BrokerAssetSnapshot,
    positions: Sequence[BrokerPositionSnapshot],
    symbols: Sequence[str],
) -> AdaptedAccountState:
    """构建策略视图，同时保留完整券商持仓快照。"""

    frozen_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
    if not frozen_symbols:
        raise ValueError("symbols must not be empty")
    if asset.total_asset <= 0:
        raise ValueError("total_asset must be positive")
    position_map: dict[str, BrokerPositionSnapshot] = {}
    for position in positions:
        if position.symbol in position_map:
            raise ValueError(f"duplicate broker position: {position.symbol}")
        position_map[position.symbol] = position
    for symbol in frozen_symbols:
        if symbol not in position_map:
            position_map[symbol] = BrokerPositionSnapshot(
                symbol=symbol, total_quantity=0, available_quantity=0, today_buy_quantity=0,
                market_value=Decimal("0"), turnover_rule=TurnoverRule.T1,
                captured_at=asset.captured_at,
            )
    return _account_state(
        cash=asset.available_cash, total_asset=asset.total_asset,
        symbols=frozen_symbols, positions=position_map,
    )


# 把现金、完整持仓与估值价转换为只读账户视图、当前权重和总资产，完整券商字段留在适配结果中。
def _account_state(
    *, cash: Decimal, total_asset: Decimal, symbols: tuple[str, ...],
    positions: Mapping[str, BrokerPositionSnapshot],
) -> AdaptedAccountState:
    """直接生成只读策略视图；完整持仓快照留在结果中，不进入策略。"""
    strategy_positions = {
        symbol: AccountPositionView(
            symbol=symbol,
            turnover_rule=positions[symbol].turnover_rule,
            total_quantity=positions[symbol].total_quantity,
            available_quantity=positions[symbol].available_quantity,
            today_buy_quantity=positions[symbol].today_buy_quantity,
        )
        for symbol in symbols
    }
    weights = {
        symbol: positions[symbol].market_value / total_asset for symbol in symbols
    }
    return AdaptedAccountState(
        account_view=AccountView(cash=cash, positions=strategy_positions),
        current_weights_by_symbol=MappingProxyType(weights),
        total_asset=total_asset,
        positions_by_symbol=MappingProxyType(dict(sorted(positions.items()))),
    )


def adapt_virtual_account(
    *,
    virtual_cash: Decimal,
    positions: Sequence[BrokerPositionSnapshot],
    symbols: Sequence[str],
    prices: Mapping[str, Decimal],
    turnover_rules: Mapping[str, TurnoverRule],
    captured_at: datetime,
) -> AdaptedAccountState:
    """只根据单个已持久化虚拟子账户构建策略视图。"""

    if virtual_cash < 0 or not virtual_cash.is_finite():
        raise ValueError("virtual_cash must be finite and non-negative")
    normalized_prices = {
        normalize_symbol(symbol): Decimal(price) for symbol, price in prices.items()
    }
    position_map = {position.symbol: position for position in positions}
    frozen_symbols = tuple(sorted({normalize_symbol(value) for value in symbols}))
    valued: dict[str, BrokerPositionSnapshot] = {}
    for symbol in frozen_symbols:
        position = position_map.get(symbol)
        turnover_rule = turnover_rules[symbol] if position is None else position.turnover_rule
        price = normalized_prices.get(symbol)
        if price is None or not price.is_finite() or price <= 0:
            raise ValueError(f"virtual valuation price is unavailable for {symbol}")
        quantity = 0 if position is None else position.total_quantity
        valued[symbol] = BrokerPositionSnapshot(
            symbol=symbol,
            total_quantity=quantity,
            available_quantity=0 if position is None else position.available_quantity,
            today_buy_quantity=0 if position is None else position.today_buy_quantity,
            market_value=price * quantity, turnover_rule=turnover_rule,
            captured_at=captured_at,
            average_cost=None if position is None else position.average_cost,
        )
    market_value = sum((row.market_value for row in valued.values()), Decimal("0"))
    total_asset = virtual_cash + market_value
    if total_asset <= 0:
        raise ValueError("virtual total_asset must be positive")
    if not frozen_symbols:
        raise ValueError("symbols must not be empty")
    return _account_state(
        cash=virtual_cash, total_asset=total_asset, symbols=frozen_symbols, positions=valued,
    )


__all__ = ["AdaptedAccountState", "adapt_broker_account", "adapt_virtual_account"]
