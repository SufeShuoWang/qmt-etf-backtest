"""现金、持仓以及按原始收盘价估值的不可变快照。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.order import FillResult, OrderSide
from etf_backtest.core.position import Position

_ZERO = Decimal("0")
_MONEY_QUANTUM = Decimal("0.001")


# 校验账户现金为合法 Decimal，供账户初始化和快照构造使用。
def _cash(value: object, field_name: str = "cash") -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


# 将金额按账户使用的精度取整，使现金与资产统计保持一致。
def _money(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


# 检查持仓映射中的证券与 Position 对象，生成账户内部使用的持仓副本。
def _positions(values: Mapping[str, Position]) -> dict[str, Position]:
    if not isinstance(values, Mapping):
        raise TypeError("positions must be a mapping")
    result: dict[str, Position] = {}
    for supplied_symbol, position in values.items():
        if not isinstance(position, Position):
            raise TypeError("positions may contain only Position")
        symbol = normalize_symbol(supplied_symbol)
        if symbol != position.symbol or symbol in result:
            raise ValueError("position key mismatch or duplicate")
        result[symbol] = position
    return dict(sorted(result.items()))


# 规范估值价格映射，拒绝未知／重复证券以及非有限或非正价格；持仓缺价由快照构造时检查。
def _prices(values: Mapping[str, Decimal], *, registered: frozenset[str]) -> dict[str, Decimal]:
    if not isinstance(values, Mapping):
        raise TypeError("mark_close_prices must be a mapping")
    result: dict[str, Decimal] = {}
    for supplied_symbol, price in values.items():
        symbol = normalize_symbol(supplied_symbol)
        if symbol not in registered or symbol in result:
            raise ValueError("unknown or duplicate price symbol")
        if not isinstance(price, Decimal):
            raise TypeError("mark close prices must be Decimal")
        if not price.is_finite() or price <= 0:
            raise ValueError("mark close prices must be finite and positive")
        result[symbol] = price
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """仅按未复权收盘价估值的只读账户状态。"""

    cash: Decimal
    positions: Mapping[str, Position]
    mark_close_prices: Mapping[str, Decimal]
    position_values: Mapping[str, Decimal] = field(init=False)
    total_position_value: Decimal = field(init=False)
    total_asset: Decimal = field(init=False)

    # 校验并冻结账户快照中的现金、持仓与估值结果，防止历史快照被后续交易改写。
    def __post_init__(self) -> None:
        cash = _money(_cash(self.cash))
        positions = _positions(self.positions)
        prices = _prices(self.mark_close_prices, registered=frozenset(positions))
        missing = sorted(
            symbol
            for symbol, position in positions.items()
            if position.total_quantity > 0 and symbol not in prices
        )
        if missing:
            raise ValueError("missing prices for held positions: " + ", ".join(missing))
        values = {
            symbol: _money(position.market_value(prices[symbol]))
            for symbol, position in positions.items()
            if position.total_quantity > 0
        }
        total_position = sum(values.values(), start=_ZERO)
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "positions", MappingProxyType(positions))
        object.__setattr__(self, "mark_close_prices", MappingProxyType(prices))
        object.__setattr__(self, "position_values", MappingProxyType(values))
        object.__setattr__(self, "total_position_value", total_position)
        object.__setattr__(self, "total_asset", cash + total_position)


@dataclass(frozen=True, slots=True)
class DailySnapshot:
    """可直接用于本地输出的单个日终快照。"""

    trade_date: date
    account_snapshot: AccountSnapshot

    # 校验日快照日期与账户快照，形成逐日结果记录。
    def __post_init__(self) -> None:
        if isinstance(self.trade_date, datetime) or not isinstance(self.trade_date, date):
            raise TypeError("trade_date must be datetime.date")
        if not isinstance(self.account_snapshot, AccountSnapshot):
            raise TypeError("account_snapshot must be AccountSnapshot")

    # 读取日快照中的现金余额。
    @property
    def cash(self) -> Decimal:
        return self.account_snapshot.cash

    # 读取日快照中的持仓总市值。
    @property
    def market_value(self) -> Decimal:
        return self.account_snapshot.total_position_value

    # 读取日快照中的总资产，即现金与持仓市值之和。
    @property
    def total_asset(self) -> Decimal:
        return self.account_snapshot.total_asset


class Account:
    """仅允许正式 ``FillResult`` 改变的可变经济状态。"""

    __slots__ = ("_cash", "_positions")

    # 用初始现金和各证券持仓建立回测账户，并校验输入。
    def __init__(self, *, cash: Decimal, positions: Mapping[str, Position] | None = None) -> None:
        self._cash = _cash(cash)
        self._positions = _positions({} if positions is None else positions)

    # 返回账户当前现金；现金变动由成交入账流程统一处理。
    @property
    def cash(self) -> Decimal:
        return self._cash

    # 提供当前证券持仓映射，供策略快照、订单生成和交易检查读取。
    @property
    def positions(self) -> Mapping[str, Position]:
        return MappingProxyType(self._positions)

    # 按规范证券代码取得对应持仓，供成交和数量检查使用。
    def position_for(self, symbol: str) -> Position:
        canonical = normalize_symbol(symbol)
        try:
            return self._positions[canonical]
        except KeyError:
            raise KeyError(f"position is not registered: {canonical}") from None

    # 将成交金额与费用计入现金，并替换对应持仓；只有实际成交会改变账户。
    def apply_fill(self, fill: FillResult) -> None:
        if not isinstance(fill, FillResult):
            raise TypeError("Account.apply_fill requires a formal FillResult")
        try:
            current = self._positions[fill.symbol]
        except KeyError:
            raise KeyError(f"fill symbol is not registered: {fill.symbol}") from None
        if fill.side is OrderSide.BUY:
            next_cash = self._cash - fill.trade_amount - fill.fee
            if next_cash < 0:
                raise ValueError("BUY fill would make cash negative")
            next_position = current.apply_buy(fill)
        else:
            next_position = current.apply_sell(fill)
            next_cash = self._cash + fill.trade_amount - fill.fee
            if next_cash < 0:
                raise ValueError("SELL fee would make cash negative")
        next_positions = dict(self._positions)
        next_positions[fill.symbol] = next_position
        self._cash = next_cash
        self._positions = next_positions

    # 推进持仓到新交易日，释放上一交易日买入的 T+1 不可卖数量。
    def on_new_trade_date(self) -> None:
        self._positions = {
            symbol: position.on_new_trade_date() for symbol, position in self._positions.items()
        }

    # 用指定原始行情价格计算持仓市值，生成与后续账户修改隔离的估值快照。
    def snapshot(self, mark_close_prices: Mapping[str, Decimal]) -> AccountSnapshot:
        return AccountSnapshot(
            cash=self._cash,
            positions=self._positions,
            mark_close_prices=mark_close_prices,
        )

    # 按输入价格生成账户快照并返回总资产，包含现金和持仓市值。
    def mark_to_market(self, mark_close_prices: Mapping[str, Decimal]) -> Decimal:
        return self.snapshot(mark_close_prices).total_asset


__all__ = ["Account", "AccountSnapshot", "DailySnapshot"]
