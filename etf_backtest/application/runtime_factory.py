"""组装回测及日频 Rule/Model 信号计算所需的共享数据组件。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Connection, Engine

from etf_backtest.config.schema import (
    BacktestConfig,
    DatabaseConfig,
    normalize_symbol,
)
from etf_backtest.core.effective_rules import (
    EffectiveDatedEtfRuleResolver,
    load_effective_rule_resolver,
)
from etf_backtest.data.mysql import QmtDailyDataset, QmtDailyRepository
from etf_backtest.data.portal import DailyDataPortal
from etf_backtest.universe.resolver import FrozenUniverse, FrozenUniverseResolver


@dataclass(frozen=True, slots=True)
class BacktestRuntime:
    """规则与模型回测共用的数据和交易规则。"""

    repository: QmtDailyRepository
    universe: FrozenUniverse
    dataset: QmtDailyDataset
    portal: DailyDataPortal
    rule_resolver: EffectiveDatedEtfRuleResolver


# 这里只创建连接引擎；数据读取由仓库的 SELECT 查询完成，创建引擎不等于已经验证数据库连通。
def create_database_engine(database: DatabaseConfig) -> Engine:
    """创建项目既有的只读 MySQL SQLAlchemy 引擎。"""

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


def create_repository(
    config: BacktestConfig,
    engine: Engine,
    *,
    connection: Connection | None = None,
) -> QmtDailyRepository:
    """创建数据仓储，并可选择绑定调用方持有的连接。"""

    return QmtDailyRepository(
        engine,
        connection=connection,
        dataset_version=config.data_snapshot.dataset_version,
        trade_status_table=config.data_snapshot.trade_status_table,
        share_table=config.data_snapshot.share_table,
        index_table=config.data_snapshot.index_table,
        index_codes=config.data_snapshot.rule_index_codes,
        huijin_holders_csv=config.data_snapshot.huijin_holders_csv,
        huijin_holders_csv_sha256=config.data_snapshot.huijin_holders_csv_sha256,
    )


def resolve_universe(config: BacktestConfig, repository: QmtDailyRepository) -> FrozenUniverse:
    """在当前运行时中一次性解析显式证券与资产池的并集。"""

    return FrozenUniverseResolver(repository).resolve(
        explicit_symbols=config.universe.symbols,
        pools=config.universe.pools,
        start_date=config.start_date,
        end_date=config.end_date,
    )


def required_project_resource(path: Path, project_root: Path, label: str) -> Path:
    """解析配置资源，并确保其位于项目目录内。"""

    root = Path(project_root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(f"{label} is missing: {target}")
    return target


# 按配置建立仓库、冻结证券范围并加载原始／前复权日数据，随后创建数据入口和有效期交易规则解析器。
def build_backtest_runtime(
    *,
    config: BacktestConfig,
    project_root: Path,
    engine: Engine,
    load_start: date,
    load_end: date,
    connection: Connection | None = None,
) -> BacktestRuntime:
    """组装数据仓储、冻结证券范围、数据入口和有效 Rule 资源。"""

    repository = create_repository(config, engine, connection=connection)
    universe = resolve_universe(config, repository)
    dataset = repository.load_daily_dataset(
        universe.symbols,
        load_start,
        load_end,
        etf_infos=universe.etf_infos,
    )
    portal = DailyDataPortal(dataset)
    rule_resolver = load_effective_rule_resolver(
        universe.etf_infos,
        required_project_resource(config.limit_rules_csv, project_root, "limit rule CSV"),
        required_project_resource(
            config.limit_rules_manifest,
            project_root,
            "limit rule manifest",
        ),
    )
    return BacktestRuntime(
        repository=repository,
        universe=universe,
        dataset=dataset,
        portal=portal,
        rule_resolver=rule_resolver,
    )


# 规范证券代码、去重并排序，保证数据查询和后续遍历使用一致的证券顺序。
def canonical_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))


# 携带信号计算所需的数据库仓库、已加载数据集和行情入口。
@dataclass(frozen=True, slots=True)
class SignalRuntime:
    repository: QmtDailyRepository
    dataset: QmtDailyDataset
    portal: DailyDataPortal


def build_signal_runtime(
    *, config: BacktestConfig, engine: Engine, frozen_symbols: Sequence[str],
    load_start: date, load_end: date, connection: Connection,
) -> SignalRuntime:
    """Rule 与 Model 共用的数据准备；沿用各自的日期范围与事务。"""
    repository = create_repository(config, engine, connection=connection)
    dataset = repository.load_daily_dataset(canonical_symbols(frozen_symbols), load_start, load_end)
    return SignalRuntime(repository, dataset, DailyDataPortal(dataset))
