"""组装 Rule/Model 模拟盘自动运行所需的数据、策略、券商和调度器。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from queue import Queue

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

from etf_backtest.application.runtime_factory import create_database_engine, create_repository
from etf_backtest.application.strategy_source import (
    RuleStrategySource,
    load_strategy_source,
    build_backtest_config,
)
from etf_backtest.config.schema import DatabaseConfig
from etf_backtest.core.fee import FeeModel
from etf_backtest.data.mysql import QmtDailyRepository
from etf_backtest.experiments.config import load_system_settings
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.broker.callbacks import (
    BrokerEvent,
    BrokerEventConsumer,
)
from etf_backtest.live.broker.miniqmt import MiniQmtBrokerGateway
from etf_backtest.live.config import LiveConfig, LiveStateDatabaseConfig
from etf_backtest.live.engine import LiveTradingEngine
from etf_backtest.live.execution.near_close_limit import NearCloseLimitPolicy
from etf_backtest.live.execution.planner import LiveRebalancePlanner
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.signals import (
    DatabaseClosePriceProvider, LoadedPredictorBundle, SignalService,
    StrategyRuntime, build_strategy_spec,
)

from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.market.xtdata import XtDataQuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import ReconciliationService, default_turnover_rule
from etf_backtest.live.risk import LiveRiskManager
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.strategy.model_runtime import DailyModelStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 选择显式状态库配置；未提供时从系统数据库配置构造状态库连接设置。
def resolve_state_database(config: LiveConfig) -> DatabaseConfig | LiveStateDatabaseConfig:
    if config.state_database is not None:
        return config.state_database
    system_path = config.project_path(config.account.system_path, PROJECT_ROOT)
    return load_system_settings(system_path).database


# 依据状态库配置创建 SQLAlchemy 引擎，供持久化和命名锁使用。
def create_state_engine(config: LiveConfig) -> Engine:
    database = resolve_state_database(config)
    url = URL.create(
        "mysql+pymysql",
        username=database.user,
        password=database.resolved_password(),
        host=database.host,
        port=database.port,
        database=database.database,
        query={"charset": database.charset},
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": database.connect_timeout_seconds},
    )


# 通过已准备的策略数据来源实现交易日查询，供模拟盘调度使用。
class _StrategyTradingDaySource:
    # 保存用于加载或查询策略日历的依赖。
    def __init__(self, repository: QmtDailyRepository) -> None:
        self._repository = repository
        self._cache: dict[date, bool] = {}

    # 使用策略数据来源判断给定日期是否开市。
    def is_trading_day(self, trade_date: date) -> bool:
        if trade_date not in self._cache:
            days = self._repository.load_sse_calendar(trade_date, trade_date)
            self._cache[trade_date] = bool(days[0].is_open)
        return self._cache[trade_date]


# 模拟盘装配入口：加载单策略与虚拟资金配置，构建券商、行情、状态库、信号、规划、风控、对账、调度及引擎。
def build_production_runtime(
    config: LiveConfig,
    *,
    broker_factory: Callable[..., BrokerGateway] = MiniQmtBrokerGateway,
    quote_factory: Callable[[], QuoteProvider] = XtDataQuoteProvider,
) -> LiveTradingEngine:
    system = config.project_path(config.account.system_path, PROJECT_ROOT)
    system_settings = load_system_settings(system)
    fee_model = FeeModel(system_settings.fee)
    account_id = config.account.account_id()
    events: Queue[BrokerEvent] = Queue()
    broker = broker_factory(
        userdata_path=config.miniqmt.userdata_path,
        account_id=account_id,
        account_type=config.account.account_type,
        event_queue=events,
    )
    quote_provider = quote_factory()
    state_engine = create_state_engine(config)
    repository = LiveStateRepository(state_engine, fee_model=fee_model)
    strategy_config = config.strategy
    experiment = config.project_path(strategy_config.experiment_path, PROJECT_ROOT)
    source = load_strategy_source(
        experiment, system_path=system, case=strategy_config.case,
        system_settings=system_settings,
    )
    strategy_engine = create_database_engine(source.system.database)
    backtest_config = build_backtest_config(source)
    bundle_path = None
    loaded: LoadedPredictorBundle | None = None
    if isinstance(source, RuleStrategySource):
        strategy = source.strategy
    else:
        assert strategy_config.model is not None
        bundle_path = config.project_path(strategy_config.model.bundle_path, PROJECT_ROOT)
        loaded = source.components.load_inference_bundle(
            bundle_path, backend=strategy_config.model.backend,
            signal_date=source.experiment.start_date,
            device=strategy_config.model.device,
        )
        strategy = DailyModelStrategy(
            feature_builder=source.components.feature_builder,
            bundle=loaded.bundle,
            portfolio=source.components.settings.portfolio,
        )
    spec = build_strategy_spec(
        live_config=config, strategy_config=strategy_config, source=source,
        backtest_config=backtest_config, engine=strategy_engine,
        bundle_path=bundle_path, bundle=loaded.bundle if loaded is not None else None,
    )
    signal_service = SignalService(
        strategy_config=strategy_config, source=source, strategy=strategy,
        backtest_config=backtest_config, strategy_engine=strategy_engine,
    )
    rules = {symbol: default_turnover_rule(symbol) for symbol in spec.symbols}
    runtime = StrategyRuntime(
        spec=spec,
        signal_evaluator=signal_service,
        turnover_rules=rules,
        close_price_provider=DatabaseClosePriceProvider(
            backtest_config=backtest_config,
            engine=strategy_engine,
        ),
    )
    calendar_repository = create_repository(backtest_config, strategy_engine)
    reconciliation = ReconciliationService(rules)
    jobs = LiveDailyJobs(
        config=config,
        broker=broker,
        quote_provider=quote_provider,
        state_repository=repository,
        state_engine=state_engine,
        strategy_runtime=runtime,
        reconciliation_service=reconciliation,
        planner=LiveRebalancePlanner(fee_model),
        risk_manager=LiveRiskManager(fee_model),
        price_policy=NearCloseLimitPolicy(),
        fee_model=fee_model,
    )
    scheduler = LiveScheduler(
        jobs=jobs,
        calendar=_StrategyTradingDaySource(calendar_repository),
        config=config,
    )
    engine = LiveTradingEngine(
        config=config,
        state_engine=state_engine,
        repository=repository,
        broker=broker,
        quote_provider=quote_provider,
        jobs=jobs,
        scheduler=scheduler,
    )
    consumer = BrokerEventConsumer(
        events=events,
        repository=repository,
        account_id=account_id,
        on_unhealthy=engine.notify_broker_unhealthy,
        turnover_rules=rules,
    )
    engine.set_event_consumer(consumer)
    return engine
