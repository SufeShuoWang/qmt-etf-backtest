"""用户编写日频 Rule 策略时使用的精简类型化边界。"""

from __future__ import annotations


import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import ClassVar, cast

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import IndexBarView, MarketBarView
from etf_backtest.core.target import (
    NO_REBALANCE,
    NoRebalance,
    StrategyTarget,
    TargetPortfolio,
)
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountPositionView, AccountView, StrategyContext
from etf_backtest.strategy.scheduler import PeriodicDecisionScheduler

WeightInput = Decimal | str | int | float
RuleOutput = Mapping[str, WeightInput] | NoRebalance

_SYSTEM_PARAMETER_KEYS = frozenset(
    {
        "connection",
        "credentials",
        "database",
        "dsn",
        "host",
        "password",
        "password_env",
        "port",
        "snapshot",
        "user",
        "username",
    }
)


# 要求周期或观察数量为严格正整数，拒绝布尔值。
def _strict_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


# 把用户权重转换为 Decimal，拒绝布尔、非法文本和非有限浮点数；范围由后续目标对象检查。
def _decimal_weight(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("user weights must not be boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("user weight float must be finite")
        return Decimal(str(value))
    if type(value) is int:
        return Decimal(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("user weight text must not be blank")
        try:
            return Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError("user weight text must be a valid decimal") from exc
    raise TypeError("user weights must be Decimal, integer, float, or decimal text")


def _freeze_parameter(value: object, field_name: str) -> object:
    """冻结代码自有的 Rule 参数，不接收系统设置。"""

    if value is None or isinstance(value, str | bool):
        return value
    if type(value) is int:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip():
                raise ValueError(f"{field_name} keys must be nonblank strings")
            if key.casefold() in _SYSTEM_PARAMETER_KEYS:
                raise ValueError(f"{field_name} must not contain system setting {key!r}")
            frozen[key] = _freeze_parameter(item, f"{field_name}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_parameter(item, f"{field_name}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{field_name} contains unsupported value {type(value).__qualname__}")


# 递归将 Decimal、只读映射和元组转换成可序列化的参数结构。
def _resolved_parameter(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _resolved_parameter(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_resolved_parameter(item) for item in value]
    return value


# 用户策略独立创建此设置对象；回看和调度由框架读取，target_weight 仅在用户逻辑引用时影响实际目标。
@dataclass(frozen=True, slots=True)
class RuleSettings:
    """在用户 ``rule.py`` 中集中声明的全部 Rule 专属控制项。

    用户可以在这里修改调度、目标仓位和任意策略常量；数据库、日历、费用和执行规则
    有意禁止通过此对象配置。
    """

    lookback_trading_days: int = 20
    rebalance_every_trading_days: int = 20
    target_weight: WeightInput = "0.90"
    parameters: Mapping[str, object] = field(default_factory=dict)

    # 校验回看与调仓设置，转换默认目标权重并递归冻结自定义参数。
    def __post_init__(self) -> None:
        lookback = _strict_positive_int(self.lookback_trading_days, "lookback_trading_days")
        if lookback < 2:
            raise ValueError("lookback_trading_days must be at least two")
        rebalance = _strict_positive_int(
            self.rebalance_every_trading_days,
            "rebalance_every_trading_days",
        )
        weight = _decimal_weight(self.target_weight)
        if not weight.is_finite() or not Decimal("0") < weight <= Decimal("1"):
            raise ValueError("target_weight must be in (0, 1]")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        parameters = _freeze_parameter(self.parameters, "parameters")
        if not isinstance(parameters, Mapping):  # pragma: no cover - 上方已完成检查
            raise AssertionError("Rule parameter freezing failed")
        object.__setattr__(self, "lookback_trading_days", lookback)
        object.__setattr__(self, "rebalance_every_trading_days", rebalance)
        object.__setattr__(self, "target_weight", weight)
        object.__setattr__(self, "parameters", parameters)

    def resolved_dict(self) -> dict[str, object]:
        """返回稳定且可转为 JSON 的实验来源设置。"""

        return {
            "lookback_trading_days": self.lookback_trading_days,
            "rebalance_every_trading_days": self.rebalance_every_trading_days,
            "target_weight": str(self.target_weight),
            "parameters": _resolved_parameter(self.parameters),
        }


@dataclass(frozen=True, slots=True, init=False)
class RuleMarketData:
    """策略只读输入：账户及辅助数据共用上下文，行情仅提供前复权值。"""

    _context: StrategyContext = field(repr=False)
    _bars_by_symbol: Mapping[str, tuple[MarketBarView, ...]] = field(repr=False)

    # 构造用户规则的只读输入；已有 StrategyContext 时复用并核对其日期、账户等身份。
    def __init__(
        self, signal_date: date, execution_date: date, frame_index: int,
        symbols: tuple[str, ...], cash: Decimal,
        positions: Mapping[str, AccountPositionView],
        _bars_by_symbol: Mapping[str, tuple[MarketBarView, ...]],
        _current_weights_by_symbol: Mapping[str, Decimal],
        _share_history_by_symbol: Mapping[str, Mapping[date, Decimal]] = MappingProxyType({}),
        _huijin_ratios_by_symbol: Mapping[str, Mapping[str, tuple[date, Decimal]]] = MappingProxyType({}),
        _index_history_by_code: Mapping[str, tuple[IndexBarView, ...]] = MappingProxyType({}),
        _combined_huijin_ratio_by_symbol: Mapping[str, tuple[date, Decimal]] = MappingProxyType({}),
        _context: StrategyContext | None = None,
    ) -> None:
        # 保留直接构造接口；引擎通过 from_strategy_inputs 直接引用已有上下文。
        if _context is not None and not isinstance(_context, StrategyContext):
            raise TypeError("context must be StrategyContext")
        context = _context or StrategyContext(
            signal_date=signal_date, execution_date=execution_date,
            frame_index=frame_index, symbols=symbols,
            account_view=AccountView(cash=cash, positions=positions),
            current_weights_by_symbol=_current_weights_by_symbol,
            share_history_by_symbol=_share_history_by_symbol,
            huijin_ratios_by_symbol=_huijin_ratios_by_symbol,
            index_history_by_code=_index_history_by_code,
            combined_huijin_ratio_by_symbol=_combined_huijin_ratio_by_symbol,
        )
        if _context is not None and (
            (signal_date, execution_date, frame_index, symbols, cash)
            != (context.signal_date, context.execution_date, context.frame_index,
                context.symbols, context.account_view.cash)
            or positions is not context.account_view.positions
        ):
            raise ValueError("Rule inputs must match the supplied context")
        self._initialize(context, _bars_by_symbol)

    # 检查证券历史的类型、日期顺序和信号日边界，冻结按证券分组的行情。
    def _initialize(
        self, context: StrategyContext,
        bars_by_symbol: Mapping[str, tuple[MarketBarView, ...]],
    ) -> None:
        symbols, signal_date = context.symbols, context.signal_date
        if not isinstance(bars_by_symbol, Mapping):
            raise TypeError("bars_by_symbol must be a mapping")
        histories: dict[str, tuple[MarketBarView, ...]] = {symbol: () for symbol in symbols}
        seen_history_symbols: set[str] = set()
        for supplied_symbol, supplied_bars in bars_by_symbol.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in histories:
                raise ValueError("history contains a symbol outside the universe")
            if symbol in seen_history_symbols:
                raise ValueError("history symbols must be unique after normalization")
            seen_history_symbols.add(symbol)
            bars_value = cast(object, supplied_bars)
            if isinstance(bars_value, (str, bytes)) or not isinstance(bars_value, Sequence):
                raise TypeError("each symbol history must be a sequence")
            bars = tuple(cast(Sequence[MarketBarView], bars_value))
            previous_date: date | None = None
            for bar in bars:
                if not isinstance(bar, MarketBarView):
                    raise TypeError("history may contain only MarketBarView")
                if bar.symbol != symbol:
                    raise ValueError("history key and MarketBarView symbol disagree")
                if bar.trade_date > signal_date:
                    raise ValueError("history contains a future adjusted view")
                if previous_date is not None and bar.trade_date <= previous_date:
                    raise ValueError("symbol history must be strictly chronological")
                previous_date = bar.trade_date
            histories[symbol] = bars

        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_bars_by_symbol", MappingProxyType(dict(sorted(histories.items()))))

    # 返回本次策略计算使用的信号日期 D。
    @property
    def signal_date(self) -> date:
        return self._context.signal_date

    # 返回目标绑定的执行日期，日频回测中为下一 SSE 交易日。
    @property
    def execution_date(self) -> date:
        return self._context.execution_date

    # 返回用于周期调度的行情帧序号。
    @property
    def frame_index(self) -> int:
        return self._context.frame_index

    # 返回本次策略允许查询和返回目标的证券范围。
    @property
    def symbols(self) -> tuple[str, ...]:
        return self._context.symbols

    # 读取信号上下文中的账户现金快照。
    @property
    def cash(self) -> Decimal:
        return self._context.account_view.cash

    # 读取不可变持仓视图，策略不能通过它修改账户。
    @property
    def positions(self) -> Mapping[str, AccountPositionView]:
        return self._context.account_view.positions

    @classmethod
    def from_strategy_inputs(
        cls,
        *,
        market_history: Sequence[MarketBarView],
        account_view: AccountView,
        context: StrategyContext,
    ) -> RuleMarketData:
        """根据引擎已经限制日期边界的输入构建友好视图。"""

        if not isinstance(account_view, AccountView):
            raise TypeError("account_view must be AccountView")
        if not isinstance(context, StrategyContext):
            raise TypeError("context must be StrategyContext")
        if account_view is not context.account_view:
            raise ValueError("account_view must be the one stored in context")
        history_value = cast(object, market_history)
        if isinstance(history_value, (str, bytes)) or not isinstance(history_value, Sequence):
            raise TypeError("market_history must be a sequence")

        grouped: defaultdict[str, list[MarketBarView]] = defaultdict(list)
        for bar in cast(Sequence[MarketBarView], history_value):
            if not isinstance(bar, MarketBarView):
                raise TypeError("market_history may contain only MarketBarView")
            if bar.symbol not in context.symbols:
                raise ValueError("market_history contains a symbol outside the universe")
            grouped[bar.symbol].append(bar)
        ordered = {
            symbol: tuple(sorted(grouped.get(symbol, ()), key=lambda bar: bar.trade_date))
            for symbol in context.symbols
        }
        data = object.__new__(cls)
        data._initialize(context, ordered)
        return data

    def bars(self, symbol: str) -> tuple[MarketBarView, ...]:
        """返回按时间排序的前复权行情；没有数据时返回空元组。"""

        return self._bars_by_symbol[self._known_symbol(symbol)]

    # 从该证券的有序前复权行情中提取收盘价序列。
    def closes(self, symbol: str) -> tuple[Decimal, ...]:
        return tuple(bar.close for bar in self.bars(symbol))

    # 从该证券历史行情中提取成交量序列。
    def volumes(self, symbol: str) -> tuple[int, ...]:
        return tuple(bar.volume for bar in self.bars(symbol))

    # 取得已有历史的最后一条行情；无记录返回 None，调用方仍需检查是否属于信号日。
    def latest(self, symbol: str) -> MarketBarView | None:
        bars = self.bars(symbol)
        return bars[-1] if bars else None

    def current_weight(self, symbol: str) -> Decimal:
        """返回证券范围内某只证券在信号日按原始收盘价计算的组合权重。"""

        return self._context.current_weights_by_symbol[self._known_symbol(symbol)]

    def share_on(self, symbol: str, asof_date: date) -> Decimal | None:
        """返回精确的 ETF 每日份额观察值，不进行前向填充。"""

        value = _plain_date(asof_date, "asof_date")
        if value > self.signal_date:
            raise ValueError("cannot query a future share date")
        return self._context.share_history_by_symbol[self._known_symbol(symbol)].get(value)

    def share_history(self, symbol: str) -> tuple[tuple[date, Decimal], ...]:
        """返回截至信号日 D 可见且按时间排序的 ETF 每日份额。"""

        rows = self._context.share_history_by_symbol[self._known_symbol(symbol)]
        return tuple(rows.items())

    def latest_huijin_ratio(self, symbol: str, company: str) -> Decimal | None:
        """返回某个汇金主体在 D 日前最新的 HolderOfListing 比例。"""

        if not isinstance(company, str) or not company.strip():
            raise ValueError("company must be a nonblank string")
        value = self._context.huijin_ratios_by_symbol[self._known_symbol(symbol)].get(company.strip())
        return None if value is None else value[1]

    def index_bars(self, index_code: str) -> tuple[IndexBarView, ...]:
        """返回截至信号日 D 可见且按时间排序的已配置 PRICE 指数行情。"""

        if not isinstance(index_code, str):
            raise TypeError("index_code must be a string")
        normalized = index_code.strip().upper()
        try:
            return self._context.index_history_by_code[normalized]
        except KeyError:
            raise ValueError(f"index code is not configured: {normalized}") from None

    def latest_combined_huijin_ratio(self, symbol: str) -> tuple[date, Decimal] | None:
        """返回 D 日前最新同报告期比例之和及其 EndDate。"""

        return self._context.combined_huijin_ratio_by_symbol.get(self._known_symbol(symbol))

    # 只按历史记录条数判断是否足够，不额外过滤历史停牌记录。
    def has_history(self, symbol: str, observations: int) -> bool:
        required = _strict_positive_int(observations, "observations")
        return len(self.bars(symbol)) >= required

    def close_return(self, symbol: str, periods: int) -> Decimal | None:
        """数据充足时返回 ``close[D] / close[D-periods] - 1``。"""

        distance = _strict_positive_int(periods, "periods")
        closes = self.closes(symbol)
        if len(closes) <= distance:
            return None
        return closes[-1] / closes[-distance - 1] - Decimal("1")

    # 读取证券当前总持仓数量。
    def position_quantity(self, symbol: str) -> int:
        return self.positions[self._known_symbol(symbol)].total_quantity

    # 读取证券当前可卖数量，已考虑 T+0/T+1 的账户状态。
    def available_quantity(self, symbol: str) -> int:
        return self.positions[self._known_symbol(symbol)].available_quantity

    # 规范证券代码并要求其属于本次策略证券范围。
    def _known_symbol(self, symbol: str) -> str:
        canonical = normalize_symbol(symbol)
        if canonical not in self._bars_by_symbol:
            raise ValueError(f"symbol is outside the configured universe: {canonical}")
        return canonical


class UserRule(ABC):
    """实现单个 Rule，并将设置与策略代码放在同一文件。"""

    settings: ClassVar[RuleSettings] = RuleSettings()

    @property
    def target_weight(self) -> Decimal:
        """返回代码自有的默认目标仓位。"""

        return cast(Decimal, self.settings.target_weight)

    @property
    def parameters(self) -> Mapping[str, object]:
        """返回 ``rule.py`` 中声明的不可变策略参数。"""

        return self.settings.parameters

    # 读取当前用户策略 settings 中的历史窗口，子类自定义值优先于继承默认值。
    @property
    def lookback_trading_days(self) -> int:
        return self.settings.lookback_trading_days

    # 读取当前用户策略 settings 中的调仓间隔，供包装器创建调度器。
    @property
    def rebalance_every_trading_days(self) -> int:
        return self.settings.rebalance_every_trading_days

    @abstractmethod
    def generate_weights(self, data: RuleMarketData) -> RuleOutput:
        """返回显式调仓目标；省略证券保持数量，空映射或 ``NO_REBALANCE`` 不调仓。"""


class SimpleRuleStrategy(BaseStrategy):
    """将单个 :class:`UserRule` 适配到已校验的日频策略引擎。"""

    __slots__ = ("_lookback_trading_days", "_rule", "_scheduler")

    # 保存本次用户规则和回看窗口，并按它自己的设置创建周期调度器。
    def __init__(self, *, rule: UserRule) -> None:
        if not isinstance(rule, UserRule):
            raise TypeError("rule must be UserRule")
        if not isinstance(rule.settings, RuleSettings):
            raise TypeError("UserRule.settings must be RuleSettings")
        self._lookback_trading_days = rule.lookback_trading_days
        self._rule = rule
        self._scheduler = PeriodicDecisionScheduler(
            every_trading_days=rule.rebalance_every_trading_days
        )

    # 返回包装器持有的用户规则实例。
    @property
    def rule(self) -> UserRule:
        return self._rule

    # 向框架声明当前规则需要的历史窗口。
    @property
    def required_history_trading_days(self) -> int:
        return self._lookback_trading_days

    # 把行情序号交给周期调度器，决定是否调用用户策略计算目标。
    def should_generate_target(self, frame_index: int) -> bool:
        if type(frame_index) is not int:
            raise TypeError("frame_index must be an integer")
        return self._scheduler.should_decide(frame_index)

    # 整理 RuleMarketData 后调用用户 generate_weights()；空映射或 NoRebalance 转成不调仓，其余转换为目标组合。
    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> StrategyTarget:
        del signal_date
        data = RuleMarketData.from_strategy_inputs(
            market_history=market_history,
            account_view=account_view,
            context=context,
        )
        supplied_weights = self._rule.generate_weights(data)
        if isinstance(supplied_weights, NoRebalance):
            return NO_REBALANCE
        if isinstance(supplied_weights, Mapping) and not supplied_weights:
            return NO_REBALANCE
        return self._target_from_user_weights(supplied_weights, symbols=data.symbols)

    # 校验用户输出是资产池内不重复证券到权重的映射，统一 Decimal 后交给 TargetPortfolio 检查范围与总和。
    @staticmethod
    def _target_from_user_weights(
        supplied_weights: Mapping[str, WeightInput],
        *,
        symbols: tuple[str, ...],
    ) -> TargetPortfolio:
        if not isinstance(supplied_weights, Mapping):
            raise TypeError("UserRule.generate_weights must return a mapping")
        allowed = frozenset(symbols)
        converted: dict[str, Decimal] = {}
        for supplied_symbol, supplied_weight in supplied_weights.items():
            if not isinstance(supplied_symbol, str):
                raise TypeError("user target symbols must be strings")
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError(f"user target symbol is outside the universe: {symbol}")
            if symbol in converted:
                raise ValueError("user target symbols must be unique after normalization")
            converted[symbol] = _decimal_weight(supplied_weight)
        return TargetPortfolio(weights=converted)


__all__ = [
    "NO_REBALANCE",
    "NoRebalance",
    "RuleMarketData",
    "RuleOutput",
    "RuleSettings",
    "SimpleRuleStrategy",
    "UserRule",
    "WeightInput",
]
