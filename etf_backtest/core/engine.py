"""从 D 日收盘信号到 D+1 日收盘执行的纯日频引擎。"""

from __future__ import annotations


from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.application.daily_decision import evaluate_daily_decision
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import Account, DailySnapshot
from etf_backtest.core.fill import FillModel
from etf_backtest.core.market import (
    EtfInfo,
    EtfTradingRule,
    MarketFrame,
    PriceLimitSource,
)
from etf_backtest.core.order import (
    FillResult,
    Order,
    RuleCheckResult,
)
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.pricing import TradePriceQuoteCache
from etf_backtest.core.rule_resolver import RuleResolver
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView

if TYPE_CHECKING:
    from etf_backtest.core.etf_rules import EtfRuleEngine
    from etf_backtest.data.portal import DailyDataPortal


# 记录信号日期、执行日期和目标组合，供逐日引擎暂存下一交易日要执行的目标。
@dataclass(frozen=True, slots=True)
class TargetDecision:  # 目标决策
    signal_date: date
    execution_date: date
    target_portfolio: TargetPortfolio

    # 要求执行日期严格晚于信号日期，且目标为 TargetPortfolio。
    def __post_init__(self) -> None:
        signal = _plain_date(self.signal_date, "signal_date")
        execution = _plain_date(self.execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")
        if not isinstance(self.target_portfolio, TargetPortfolio):
            raise TypeError("target_portfolio must be TargetPortfolio")


# 汇总一次回测的证券信息、每日账户快照、目标决策、订单、审批和成交。
@dataclass(frozen=True, slots=True)
class BacktestResult:
    etf_infos: tuple[EtfInfo, ...]  # etf信息
    daily_snapshots: tuple[DailySnapshot, ...]  # 每日账户快照
    orders: tuple[Order, ...]  # 每日生成的订单
    fills: tuple[FillResult, ...]  # 每日成交结果
    decisions: tuple[TargetDecision, ...]  # 每日生成的目标组合
    approvals: tuple[RuleCheckResult, ...]  # 每日订单审批结果


class BacktestEngine:
    """按唯一允许的事件顺序推进完整上交所日频行情帧。"""

    __slots__ = (  # BacktestEngine 实例可以拥有的属性
        "_account",
        "_etf_infos",
        "_fill_model",
        "_order_generator",
        "_portal",
        "_rule_engine",
        "_rule_resolver",  # 每日规则解析服务
        "_strategy",
    )

    # 组装数据入口、账户、策略、规则解析器、订单生成器和成交模型，供 run() 推进日期。
    def __init__(
        self,
        *,
        portal: DailyDataPortal,
        account: Account,
        strategy: BaseStrategy,
        rule_resolver: RuleResolver,
        rule_engine: EtfRuleEngine,
        order_generator: OrderGenerator,
        fill_model: FillModel,
    ) -> None:
        for method_name in (
            "execution_frames",
            "views_through",
            "share_history_through",
            "huijin_ratios_as_of",
            "index_history_through",
            "combined_huijin_ratios_as_of",
        ):
            if not callable(getattr(portal, method_name, None)):
                raise TypeError("portal must satisfy the daily data boundary")
        if not isinstance(account, Account):
            raise TypeError("account must be Account")
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must be BaseStrategy")
        if not isinstance(rule_resolver, RuleResolver):
            raise TypeError("rule_resolver must satisfy RuleResolver")
        if not callable(getattr(rule_engine, "approve_batch", None)):
            raise TypeError("rule_engine must provide approve_batch")
        if not isinstance(order_generator, OrderGenerator):
            raise TypeError("order_generator must be OrderGenerator")
        if not isinstance(fill_model, FillModel):
            raise TypeError("fill_model must be FillModel")

        symbols = tuple(sorted(normalize_symbol(symbol) for symbol in portal.symbols))
        if set(symbols) != set(account.positions):
            raise ValueError("account positions must exactly cover portal symbols")
        infos: dict[str, EtfInfo] = {}
        for info in portal.etf_infos:
            if not isinstance(info, EtfInfo):
                raise TypeError("portal.etf_infos may contain only EtfInfo")
            if info.symbol in infos:
                raise ValueError("duplicate EtfInfo")
            infos[info.symbol] = info
        if set(infos) != set(symbols):
            raise ValueError("EtfInfo must exactly cover portal symbols")

        self._portal = portal
        self._account = account
        self._strategy = strategy
        self._rule_resolver = rule_resolver
        self._rule_engine = rule_engine
        self._order_generator = order_generator
        self._fill_model = fill_model
        self._etf_infos = MappingProxyType(dict(sorted(infos.items())))

    # 返回引擎持有的账户，便于读取运行后的现金与持仓。
    @property
    def account(self) -> Account:
        return self._account

    # 回测主循环：每日先释放 T+1 持仓并执行前一日目标，再记录净值、计算下一交易日目标；最后一帧不新建目标。
    def run(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        start = _plain_date(start_date, "start_date")
        end = _plain_date(end_date, "end_date")
        if end < start:
            raise ValueError("end_date must not precede start_date")
        frames = tuple(self._portal.execution_frames(start, end))
        if not frames:
            raise ValueError("backtest interval contains no complete execution frame")
        self._validate_frame_sequence(frames)

        pending: TargetDecision | None = None
        daily_values: list[DailySnapshot] = []
        all_fills: list[FillResult] = []
        all_orders: list[Order] = []
        decisions: list[TargetDecision] = []
        approvals: list[RuleCheckResult] = []

        for frame_index, frame in enumerate(frames):
            # 1. 日期切换时先释放 T+1 持仓，再处理任何订单。
            self._account.on_new_trade_date()  # 更新下一天账户的信息

            # 每个证券和日期都重新解析规则，禁止静态映射跨越规则生效日边界。
            trading_rules = self._resolve_rules(frame)
            raw_closes = {symbol: bar.close for symbol, bar in frame.bars_by_symbol.items()}
            # 2. 上一交易日生成的目标只在绑定的相邻 D+1 收盘行情帧执行。
            if pending is not None:
                if pending.execution_date != frame.trade_date:
                    raise RuntimeError("pending target did not reach its bound D+1 frame")
                frame_fills, frame_approvals, frame_orders = self._execute_pending(
                    decision=pending,
                    frame=frame,
                    raw_closes=raw_closes,
                    trading_rules=trading_rules,
                )
                all_fills.extend(frame_fills)
                all_orders.extend(frame_orders)
                approvals.extend(frame_approvals)
                pending = None

            # 3. 当日正式成交完成后，才按同一原始收盘价记录 NAV。
            daily = DailySnapshot(
                trade_date=frame.trade_date,
                account_snapshot=self._account.snapshot(raw_closes),
            )
            daily_values.append(daily)

            # 4. 最后一个行情帧不再创建无法执行的悬空目标。
            if frame_index == len(frames) - 1:
                continue
            next_frame = frames[frame_index + 1]
            account_view = AccountView.from_account(self._account)
            signal_snapshot = daily.account_snapshot
            current_weights = {
                symbol: (
                    signal_snapshot.position_values.get(symbol, Decimal("0"))
                    / signal_snapshot.total_asset
                    if signal_snapshot.total_asset > Decimal("0")
                    else Decimal("0")
                )
                for symbol in self._account.positions
            }
            decision_result = evaluate_daily_decision(
                strategy=self._strategy,
                portal=self._portal,
                signal_date=frame.trade_date,
                execution_date=next_frame.trade_date,
                schedule_index=frame_index,
                symbols=tuple(self._account.positions),
                account_view=account_view,
                current_weights_by_symbol=current_weights,
            )
            if decision_result.status is not DecisionStatus.TARGET_CREATED:
                continue
            target = decision_result.target_portfolio
            assert target is not None
            pending = TargetDecision(
                signal_date=decision_result.signal_date,
                execution_date=decision_result.execution_date,
                target_portfolio=target,
            )
            decisions.append(pending)

        return BacktestResult(
            etf_infos=tuple(self._etf_infos.values()),
            daily_snapshots=tuple(daily_values),
            orders=tuple(all_orders),
            fills=tuple(all_fills),
            decisions=tuple(decisions),
            approvals=tuple(approvals),
        )

    # 按执行日原始收盘价估值并生成订单，统一估计成交价、审批后把成交计入账户；未成交部分不自动顺延。
    def _execute_pending(
        self,
        *,
        decision: TargetDecision,
        frame: MarketFrame,
        raw_closes: Mapping[str, Decimal],
        trading_rules: Mapping[str, EtfTradingRule],
    ) -> tuple[
        tuple[FillResult, ...],
        tuple[RuleCheckResult, ...],
        tuple[Order, ...],
    ]:
        valuation = self._account.snapshot(raw_closes)
        orders = self._order_generator.generate(
            target_portfolio=decision.target_portfolio,
            valuation_snapshot=valuation,
            signal_date=decision.signal_date,
            execution_date=frame.trade_date,
        )
        if not orders:
            return (), (), ()
        quote_cache = TradePriceQuoteCache(frame=frame, trading_rules=trading_rules)
        quotes = quote_cache.resolve_all(order.symbol for order in orders)
        estimates = {
            order.order_id: self._fill_model.create_estimate(
                order=order,
                quote=quotes[order.symbol],
                tick_size=trading_rules[order.symbol].tick_size,
            )
            for order in orders
        }
        raw_approvals = tuple(
            self._rule_engine.approve_batch(
                frame=frame,
                orders=orders,
                quotes=quotes,
                estimates=estimates,
                account=self._account,
                trading_rules=trading_rules,
                etf_infos=self._etf_infos,
            )
        )
        raw_by_order = self._validate_approvals(orders=orders, approvals=raw_approvals)
        evidenced_approvals = tuple(
            replace(
                raw_by_order[order.order_id],
                base_trade_price=quotes[order.symbol].base_trade_price,
                price_limit_down=quotes[order.symbol].price_limit_down,
                price_limit_up=quotes[order.symbol].price_limit_up,
                price_limit_source=quotes[order.symbol].price_limit_source,
                price_limit_fallback_reason=(
                    "NO_EXPLICIT_PRICE_LIMIT"
                    if quotes[order.symbol].price_limit_source
                    is PriceLimitSource.DERIVED_RULE_FALLBACK
                    else None
                ),
            )
            for order in orders
        )
        by_order = {approval.order_id: approval for approval in evidenced_approvals}
        fills: list[FillResult] = []
        for order in orders:
            fill = self._fill_model.create_fill(
                order=order,
                quote=quotes[order.symbol],
                estimate=estimates[order.order_id],
                approval=by_order[order.order_id],
            )
            if fill is not None:
                self._account.apply_fill(fill)
                fills.append(fill)
        return tuple(fills), evidenced_approvals, orders

    # 为当前行情帧中的证券解析当日生效交易规则。
    def _resolve_rules(self, frame: MarketFrame) -> Mapping[str, EtfTradingRule]:
        resolved: dict[str, EtfTradingRule] = {}
        for symbol in frame.canonical_symbols:
            rule = self._rule_resolver.resolve(symbol, frame.trade_date)
            if not isinstance(rule, EtfTradingRule) or rule.symbol != symbol:
                raise TypeError("RuleResolver returned an invalid EtfTradingRule")
            registered = self._account.position_for(symbol)
            if registered.turnover_rule is not rule.turnover_rule:
                raise ValueError("effective turnover rule conflicts with registered position")
            resolved[symbol] = rule
        return MappingProxyType(resolved)

    # 检查执行帧日期、证券及相邻关系，确保逐日推进对应完整且有序的日历。
    def _validate_frame_sequence(self, frames: tuple[MarketFrame, ...]) -> None:
        if any(not isinstance(frame, MarketFrame) for frame in frames):
            raise TypeError("portal returned a non-MarketFrame value")
        if any(left.trade_date >= right.trade_date for left, right in pairwise(frames)):
            raise ValueError("execution frames must be strictly chronological")
        calendar = self._portal.trading_calendar
        next_trading_day = getattr(calendar, "next_trading_day", None)
        if not callable(next_trading_day):
            raise TypeError("portal.trading_calendar must provide next_trading_day")
        for left, right in pairwise(frames):
            if next_trading_day(left.trade_date) != right.trade_date:
                raise ValueError("execution frames must be adjacent SSE trading dates")

    # 检查审批结果与输入订单一一对应，防止错单或数量不一致进入成交。
    @staticmethod
    def _validate_approvals(
        *,
        orders: Sequence[Order],
        approvals: Sequence[RuleCheckResult],
    ) -> Mapping[str, RuleCheckResult]:
        if any(not isinstance(value, RuleCheckResult) for value in approvals):
            raise TypeError("rule_engine returned a non-RuleCheckResult value")
        by_order = {value.order_id: value for value in approvals}
        order_ids = {order.order_id for order in orders}
        if len(by_order) != len(approvals) or set(by_order) != order_ids:
            raise ValueError("rule_engine must return exactly one approval per order")
        return MappingProxyType(by_order)


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "TargetDecision",
]
