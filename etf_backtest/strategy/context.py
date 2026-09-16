"""供日频引擎使用的不含价格信息的精简策略上下文。"""

from __future__ import annotations


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from etf_backtest.validation import plain_date as _plain_date, quantity as _quantity
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import Account
from etf_backtest.core.market import IndexBarView, TurnoverRule
from etf_backtest.core.position import Position

if TYPE_CHECKING:
    from etf_backtest.data.portal import DailyDataPortal


@dataclass(frozen=True, slots=True)
class AccountPositionView:
    """不含价格和账户修改器的不可变份额分桶。"""

    symbol: str
    turnover_rule: TurnoverRule
    total_quantity: int
    available_quantity: int
    today_buy_quantity: int

    # 校验只读持仓视图的数量及周转规则。
    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.turnover_rule, TurnoverRule):
            raise TypeError("turnover_rule must be TurnoverRule")
        total = _quantity(self.total_quantity, "total_quantity")
        available = _quantity(self.available_quantity, "available_quantity")
        today = _quantity(self.today_buy_quantity, "today_buy_quantity")
        if available > total or today > total:
            raise ValueError("position view buckets cannot exceed total quantity")
        if self.turnover_rule is TurnoverRule.T0:
            if available != total or today != 0:
                raise ValueError("T0 position view requires immediate availability")
        elif available + today != total:
            raise ValueError("T1 position view buckets must sum to total")

    # 从账户 Position 提取策略允许读取的不可变持仓字段。
    @classmethod
    def from_position(cls, position: Position) -> AccountPositionView:
        if not isinstance(position, Position):
            raise TypeError("position must be Position")
        return cls(
            symbol=position.symbol,
            turnover_rule=position.turnover_rule,
            total_quantity=position.total_quantity,
            available_quantity=position.available_quantity,
            today_buy_quantity=position.today_buy_quantity,
        )


@dataclass(frozen=True, slots=True)
class AccountView:
    """有意不包含价格的不可变现金与数量状态。"""

    cash: Decimal
    positions: Mapping[str, AccountPositionView]

    # 校验现金和持仓映射并冻结副本，避免策略修改实际账户。
    def __post_init__(self) -> None:
        if not isinstance(self.cash, Decimal):
            raise TypeError("cash must be Decimal")
        if not self.cash.is_finite() or self.cash < 0:
            raise ValueError("cash must be finite and non-negative")
        if not isinstance(self.positions, Mapping):
            raise TypeError("positions must be a mapping")
        canonical: dict[str, AccountPositionView] = {}
        for supplied_symbol, value in self.positions.items():
            symbol = normalize_symbol(supplied_symbol)
            if not isinstance(value, AccountPositionView):
                raise TypeError("positions may contain only AccountPositionView")
            if value.symbol != symbol or symbol in canonical:
                raise ValueError("position view key mismatch or duplicate")
            canonical[symbol] = value
        object.__setattr__(self, "positions", MappingProxyType(dict(sorted(canonical.items()))))

    # 把回测账户转换为只读账户视图，交给信号上下文。
    @classmethod
    def from_account(cls, account: Account) -> AccountView:
        if not isinstance(account, Account):
            raise TypeError("account must be Account")
        return cls(
            cash=account.cash,
            positions={
                symbol: AccountPositionView.from_position(position)
                for symbol, position in account.positions.items()
            },
        )


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """已知的 D/D+1 身份信息，以及不含价格的账户和持仓权重视图。"""

    signal_date: date
    execution_date: date
    frame_index: int
    symbols: tuple[str, ...]
    account_view: AccountView
    current_weights_by_symbol: Mapping[str, Decimal]
    share_history_by_symbol: Mapping[str, Mapping[date, Decimal]] = field(default_factory=dict)
    huijin_ratios_by_symbol: Mapping[str, Mapping[str, tuple[date, Decimal]]] = field(
        default_factory=dict
    )
    index_history_by_code: Mapping[str, tuple[IndexBarView, ...]] = field(default_factory=dict)
    combined_huijin_ratio_by_symbol: Mapping[str, tuple[date, Decimal]] = field(
        default_factory=dict
    )

    @classmethod
    def from_portal(
        cls, *, portal: DailyDataPortal, signal_date: date, execution_date: date,
        frame_index: int, symbols: tuple[str, ...], account_view: AccountView,
        current_weights_by_symbol: Mapping[str, Decimal], lookback_trading_days: int,
    ) -> StrategyContext:
        """辅助数据在此组装并交给构造器校验，回测与模拟盘共用。"""
        return cls(
            signal_date=signal_date, execution_date=execution_date,
            frame_index=frame_index, symbols=symbols, account_view=account_view,
            current_weights_by_symbol=current_weights_by_symbol,
            share_history_by_symbol=portal.share_history_through(signal_date, symbols=symbols),
            huijin_ratios_by_symbol=portal.huijin_ratios_as_of(signal_date, symbols=symbols),
            index_history_by_code=portal.index_history_through(
                signal_date, lookback_trading_days=lookback_trading_days,
            ),
            combined_huijin_ratio_by_symbol=portal.combined_huijin_ratios_as_of(
                signal_date, symbols=symbols,
            ),
        )

    # 校验信号／执行日期、调度序号、证券与账户一致性，并冻结各类辅助数据。
    def __post_init__(self) -> None:
        signal = _plain_date(self.signal_date, "signal_date")
        execution = _plain_date(self.execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        symbols = cast(object, self.symbols)
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
            raise TypeError("symbols must be a sequence")
        canonical = tuple(
            sorted(normalize_symbol(symbol) for symbol in cast(Sequence[str], symbols))
        )
        if not canonical or len(canonical) != len(set(canonical)):
            raise ValueError("symbols must be non-empty and unique")
        if not isinstance(self.account_view, AccountView):
            raise TypeError("account_view must be AccountView")
        if set(canonical) != set(self.account_view.positions):
            raise ValueError("symbols must exactly cover registered account positions")
        object.__setattr__(self, "symbols", canonical)
        validated = {
            "current_weights_by_symbol": self._freeze_current_weights(
                self.current_weights_by_symbol, symbols=canonical),
            "share_history_by_symbol": self._freeze_share_history(
                self.share_history_by_symbol, symbols=canonical, signal_date=signal),
            "huijin_ratios_by_symbol": self._freeze_huijin_ratios(
                self.huijin_ratios_by_symbol, symbols=canonical, signal_date=signal),
            "index_history_by_code": self._freeze_index_history(
                self.index_history_by_code, signal_date=signal),
            "combined_huijin_ratio_by_symbol": self._freeze_combined_huijin_ratios(
                self.combined_huijin_ratio_by_symbol, symbols=canonical, signal_date=signal),
        }
        for name, value in validated.items():
            object.__setattr__(self, name, value)

    # 校验证券当前权重并冻结映射；权重由调用方按原始价估值提供。
    @staticmethod
    def _freeze_current_weights(
        supplied: Mapping[str, Decimal],
        *,
        symbols: tuple[str, ...],
    ) -> Mapping[str, Decimal]:
        if not isinstance(supplied, Mapping):
            raise TypeError("current_weights_by_symbol must be a mapping")
        allowed = frozenset(symbols)
        result: dict[str, Decimal] = {}
        total = Decimal("0")
        for supplied_symbol, weight in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError("current weights contain a symbol outside the universe")
            if symbol in result:
                raise ValueError("current weight symbols must be unique after normalization")
            if not isinstance(weight, Decimal):
                raise TypeError("current weights must be Decimal")
            if not weight.is_finite() or not Decimal("0") <= weight <= Decimal("1"):
                raise ValueError("current weights must be finite and in [0, 1]")
            result[symbol] = weight
            total += weight
        if set(result) != allowed:
            raise ValueError("current weights must exactly cover symbols")
        if total > Decimal("1"):
            raise ValueError("current weights must not sum to more than one")
        return MappingProxyType(dict(sorted(result.items())))

    # 检查同报告期汇金比例汇总的证券、日期及数值，形成只读映射。
    @staticmethod
    def _freeze_combined_huijin_ratios(
        supplied: Mapping[str, tuple[date, Decimal]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, tuple[date, Decimal]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("combined_huijin_ratio_by_symbol must be a mapping")
        allowed = frozenset(symbols)
        result: dict[str, tuple[date, Decimal]] = {}
        for supplied_symbol, supplied_value in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError("combined Huijin ratio contains a symbol outside the universe")
            if symbol in result:
                raise ValueError("combined Huijin ratio symbols must be unique")
            if not isinstance(supplied_value, tuple) or len(supplied_value) != 2:
                raise TypeError("combined Huijin ratio must be an (end_date, Decimal) tuple")
            end_date = _plain_date(supplied_value[0], "combined Huijin end_date")
            ratio = supplied_value[1]
            if end_date >= signal_date:
                raise ValueError("combined Huijin ratio must be strictly earlier than signal_date")
            if not isinstance(ratio, Decimal):
                raise TypeError("combined Huijin ratio must be Decimal")
            if not ratio.is_finite() or not Decimal("0") <= ratio <= Decimal("1"):
                raise ValueError("combined Huijin ratio must be finite and in [0, 1]")
            result[symbol] = (end_date, ratio)
        return MappingProxyType(dict(sorted(result.items())))

    # 检查指数历史代码、日期顺序与信号日边界，形成只读行情序列。
    @staticmethod
    def _freeze_index_history(
        supplied: Mapping[str, tuple[IndexBarView, ...]],
        *,
        signal_date: date,
    ) -> Mapping[str, tuple[IndexBarView, ...]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("index_history_by_code must be a mapping")
        histories: dict[str, tuple[IndexBarView, ...]] = {}
        for index_code, supplied_bars in supplied.items():
            supplied_bars_value = cast(object, supplied_bars)
            if isinstance(supplied_bars_value, (str, bytes)) or not isinstance(
                supplied_bars_value, Sequence
            ):
                raise TypeError("each index history must be a sequence")
            bars = tuple(cast(Sequence[IndexBarView], supplied_bars_value))
            previous_date: date | None = None
            for bar in bars:
                if not isinstance(bar, IndexBarView):
                    raise TypeError("index history may contain only IndexBarView")
                if bar.index_code != index_code:
                    raise ValueError("index history key and IndexBarView code disagree")
                if bar.trade_date > signal_date:
                    raise ValueError("index history contains a future view")
                if previous_date is not None and bar.trade_date <= previous_date:
                    raise ValueError("index history must be strictly chronological")
                previous_date = bar.trade_date
            histories[index_code] = bars
        return MappingProxyType(dict(sorted(histories.items())))

    # 检查 ETF 份额历史的精确日期和数值，冻结供策略查询的数据。
    @staticmethod
    def _freeze_share_history(
        supplied: Mapping[str, Mapping[date, Decimal]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, Mapping[date, Decimal]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("share_history_by_symbol must be a mapping")
        histories: dict[str, Mapping[date, Decimal]] = {
            symbol: MappingProxyType({}) for symbol in symbols
        }
        seen: set[str] = set()
        for supplied_symbol, supplied_rows in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in histories:
                raise ValueError("share history contains a symbol outside the universe")
            if symbol in seen:
                raise ValueError("share history symbols must be unique after normalization")
            seen.add(symbol)
            if not isinstance(supplied_rows, Mapping):
                raise TypeError("each share history must be a mapping")
            rows: dict[date, Decimal] = {}
            for supplied_date, total_share in supplied_rows.items():
                asof_date = _plain_date(supplied_date, "share asof_date")
                if asof_date > signal_date:
                    raise ValueError("share history contains a future business date")
                if not isinstance(total_share, Decimal):
                    raise TypeError("total_share must be Decimal")
                if not total_share.is_finite() or total_share < 0:
                    raise ValueError("total_share must be finite and non-negative")
                rows[asof_date] = total_share
            histories[symbol] = MappingProxyType(dict(sorted(rows.items())))
        return MappingProxyType(histories)

    # 检查各汇金主体报告期比例并冻结；此层不建立独立的披露日期档案。
    @staticmethod
    def _freeze_huijin_ratios(
        supplied: Mapping[str, Mapping[str, tuple[date, Decimal]]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("huijin_ratios_by_symbol must be a mapping")
        snapshots: dict[str, Mapping[str, tuple[date, Decimal]]] = {
            symbol: MappingProxyType({}) for symbol in symbols
        }
        seen: set[str] = set()
        for supplied_symbol, supplied_entities in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in snapshots:
                raise ValueError("Huijin ratios contain a symbol outside the universe")
            if symbol in seen:
                raise ValueError("Huijin ratio symbols must be unique after normalization")
            seen.add(symbol)
            if not isinstance(supplied_entities, Mapping):
                raise TypeError("each Huijin ratio snapshot must be a mapping")
            entities: dict[str, tuple[date, Decimal]] = {}
            for supplied_entity, supplied_value in supplied_entities.items():
                if not isinstance(supplied_entity, str) or not supplied_entity.strip():
                    raise ValueError("Huijin entity must be a nonblank string")
                entity = supplied_entity.strip()
                if not isinstance(supplied_value, tuple) or len(supplied_value) != 2:
                    raise TypeError("Huijin ratio must be an (end_date, Decimal) tuple")
                end_date = _plain_date(supplied_value[0], "Huijin end_date")
                ratio = supplied_value[1]
                if end_date >= signal_date:
                    raise ValueError("Huijin ratio must be strictly earlier than signal_date")
                if not isinstance(ratio, Decimal):
                    raise TypeError("Huijin ratio must be Decimal")
                if not ratio.is_finite() or not Decimal("0") <= ratio <= Decimal("1"):
                    raise ValueError("Huijin ratio must be finite and in [0, 1]")
                if entity in entities:
                    raise ValueError("Huijin entities must be unique")
                entities[entity] = (end_date, ratio)
            snapshots[symbol] = MappingProxyType(dict(sorted(entities.items())))
        return MappingProxyType(snapshots)


__all__ = ["AccountPositionView", "AccountView", "StrategyContext"]
