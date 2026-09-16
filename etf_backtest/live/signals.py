"""模拟盘策略信息、历史收盘价和每日信号；交易执行位于 jobs.py。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from sqlalchemy.engine import Engine

from etf_backtest.application.contracts import DailyDecisionResult
from etf_backtest.application.daily_decision import evaluate_daily_decision
from etf_backtest.application.runtime_factory import (
    build_signal_runtime, canonical_symbols, create_repository, resolve_universe,
)
from etf_backtest.application.schedule import trading_day_index
from etf_backtest.application.strategy_source import ModelStrategySource, StrategySource
from etf_backtest.config.schema import MARKET_TIMEZONE, BacktestConfig
from etf_backtest.core.market import TurnoverRule
from etf_backtest.data.calendar import SseTradingCalendar
from etf_backtest.live.account_adapter import AdaptedAccountState, adapt_virtual_account
from etf_backtest.live.config import LiveConfig, LiveStrategyConfig
from etf_backtest.live.state import BrokerPositionSnapshot
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.model_contracts import PredictorBundle

# 保存策略 ID、类型、证券范围等运行规格，供模拟盘信号与执行作业引用。
@dataclass(frozen=True, slots=True)
class StrategySpec:
    account_id: str
    strategy_id: str
    case: str
    initial_capital: Decimal
    experiment_path: str
    schedule_anchor_date: date
    symbols: tuple[str, ...]
    model_backend: str | None = None
    model_bundle_path: str | None = None
    model_id: str | None = None


# 声明已加载推理产物必须提供 bundle 属性，兼容不同模型后端的加载结果。
class LoadedPredictorBundle(Protocol):
    # 返回已加载的模型预测包，供 DailyModelStrategy 使用。
    @property
    def bundle(self) -> PredictorBundle: ...


class DatabaseClosePriceProvider:
    """从既有策略数据仓储读取原始收盘价。"""

    # 绑定数据库或策略运行时依赖，准备按日期读取原始收盘价。
    def __init__(self, *, backtest_config: BacktestConfig, engine: Engine) -> None:
        self.backtest_config, self.engine = backtest_config, engine

    # 查询指定日期和证券的原始收盘价映射，供虚拟账户收盘估值。
    def __call__(self, trading_date: date, symbols: tuple[str, ...]) -> Mapping[str, Decimal]:
        with self.engine.connect() as connection:
            repository = create_repository(self.backtest_config, self.engine, connection=connection)
            dataset = repository.load_daily_dataset(symbols, trading_date, trading_date)
        frames = dataset.market_frames()
        if len(frames) != 1 or frames[0].trade_date != trading_date:
            raise ValueError(f"raw close frame is unavailable for {trading_date.isoformat()}")
        frame = frames[0]
        prices = {
            symbol: frame.bars_by_symbol[symbol].close
            for symbol in symbols
            if symbol in frame.bars_by_symbol
        }
        if set(prices) != set(symbols):
            missing = sorted(set(symbols) - set(prices))
            raise ValueError(f"raw close prices are missing: {missing}")
        return prices


# 携带单策略规格、信号服务、周转规则映射和原始收盘价提供者，供日作业调用。
@dataclass(frozen=True, slots=True)
class StrategyRuntime:
    spec: StrategySpec
    signal_evaluator: SignalService
    turnover_rules: Mapping[str, TurnoverRule]
    close_price_provider: DatabaseClosePriceProvider

    # 要求周转规则映射恰好覆盖策略证券范围。
    def __post_init__(self) -> None:
        if set(self.turnover_rules) != set(self.spec.symbols):
            raise ValueError("turnover rules must exactly cover the strategy Universe")


# 用数据入口的原始收盘价适配账户现金和持仓，生成策略信号需要的账户状态。
def _account_from_portal(
    *,
    portal: object,
    signal_date: date,
    symbols: tuple[str, ...],
    virtual_cash: Decimal,
    positions: tuple[BrokerPositionSnapshot, ...],
    turnover_rules: Mapping[str, TurnoverRule],
) -> AdaptedAccountState:
    frame = portal.raw_frame(signal_date)  # type: ignore[attr-defined]
    if frame is None:
        raise ValueError("signal date raw market frame is unavailable")
    prices: dict[str, Decimal] = {}
    for symbol in symbols:
        bar = frame.bars_by_symbol.get(symbol)
        if bar is None:
            raise ValueError(f"signal date raw close is unavailable for {symbol}")
        prices[symbol] = bar.close
    return adapt_virtual_account(
        virtual_cash=virtual_cash,
        positions=positions,
        symbols=symbols,
        prices=prices,
        turnover_rules=turnover_rules,
        captured_at=datetime.combine(signal_date, time(15), MARKET_TIMEZONE),
    )


# 启动时解析证券范围，生成策略身份、初始资金、调度锚点及可选模型产物信息。
def build_strategy_spec(
    *, live_config: LiveConfig, strategy_config: LiveStrategyConfig,
    source: StrategySource, backtest_config: BacktestConfig, engine: Engine,
    bundle_path: Path | None = None, bundle: PredictorBundle | None = None,
) -> StrategySpec:
    """启动时解析证券范围，并记录规则策略或模型策略的身份。"""
    with engine.connect() as connection:
        repository = create_repository(backtest_config, engine, connection=connection)
        symbols = resolve_universe(backtest_config, repository).symbols
    model = strategy_config.model
    if isinstance(source, ModelStrategySource) and (model is None or bundle is None or bundle_path is None):
        raise ValueError("model strategy requires model configuration and a loaded bundle")
    return StrategySpec(
        account_id=live_config.account.account_id(),
        strategy_id=strategy_config.strategy_id,
        case=strategy_config.case,
        initial_capital=strategy_config.initial_capital,
        experiment_path=str(source.experiment_path),
        schedule_anchor_date=strategy_config.schedule_anchor_date,
        symbols=canonical_symbols(symbols),
        model_backend=model.backend if bundle is not None and model is not None else None,
        model_bundle_path=str(bundle_path) if bundle_path is not None else None,
        model_id=str(bundle.metadata.model_id) if bundle is not None else None,
    )


# 模拟盘信号服务：复用回测的只读账户与日决策接口，同时保留规则和模型各自的历史加载范围。
class SignalService:
    """共用账户适配和决策流程，保留两类策略各自的历史加载范围。"""

    # 保存策略运行时及数据准备依赖，供日信号任务调用。
    def __init__(
        self, *, strategy_config: LiveStrategyConfig, source: StrategySource,
        strategy: BaseStrategy, backtest_config: BacktestConfig, strategy_engine: Engine,
    ) -> None:
        self.strategy_config, self.source = strategy_config, source
        self.strategy, self.engine = strategy, strategy_engine
        self.backtest_config = backtest_config

    # 为指定信号日准备行情、账户和调度序号，调用与回测共用的 evaluate_daily_decision()。
    def evaluate(
        self, *, signal_date: date, symbols: tuple[str, ...], virtual_cash: Decimal,
        positions: tuple[BrokerPositionSnapshot, ...],
        turnover_rules: Mapping[str, TurnoverRule],
    ) -> DailyDecisionResult:
        model = isinstance(self.source, ModelStrategySource)
        if model:
            lookback = self.strategy.required_history_trading_days
            load_start = signal_date - timedelta(days=max(90, lookback * 3))
            load_end = signal_date
        else:
            load_start = min(self.source.experiment.start_date,
                             self.strategy_config.schedule_anchor_date, signal_date)
            load_end = signal_date + timedelta(days=10)
        with self.engine.connect() as connection, connection.begin():
            runtime = build_signal_runtime(
                config=self.backtest_config, engine=self.engine, frozen_symbols=symbols,
                load_start=load_start, load_end=load_end, connection=connection,
            )
            calendar = runtime.portal.trading_calendar
            if model:
                calendar = SseTradingCalendar(runtime.repository.load_sse_calendar(
                    min(load_start, self.strategy_config.schedule_anchor_date),
                    signal_date + timedelta(days=10),
                ))
            execution_date = calendar.next_trading_day(signal_date)
            schedule_index = trading_day_index(
                calendar=calendar, anchor_date=self.strategy_config.schedule_anchor_date,
                signal_date=signal_date,
            )
            account = _account_from_portal(
                portal=runtime.portal, signal_date=signal_date, symbols=symbols,
                virtual_cash=virtual_cash, positions=positions, turnover_rules=turnover_rules,
            )
            return evaluate_daily_decision(
                strategy=self.strategy, portal=runtime.portal, signal_date=signal_date,
                execution_date=execution_date, schedule_index=schedule_index, symbols=symbols,
                account_view=account.account_view,
                current_weights_by_symbol=account.current_weights_by_symbol,
            )
