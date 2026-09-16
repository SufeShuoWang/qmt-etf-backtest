"""准备、校验并运行单个可信本地 Rule 或 Model 实验。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy.engine import Engine

from etf_backtest.application.runtime_factory import (
    BacktestRuntime,
    build_backtest_runtime,
    create_database_engine,
    required_project_resource,
)
from etf_backtest.application.strategy_source import (
    RuleStrategySource,
    ModelStrategySource,
    StrategySource,
    load_strategy_source,
    build_backtest_config,
)
from etf_backtest.config.schema import (
    BacktestConfig,
    ModelStrategyConfig,
)
from etf_backtest.core.account import Account
from etf_backtest.core.effective_rules import (
    EffectiveDatedEtfRuleResolver,
)
from etf_backtest.core.engine import BacktestEngine, BacktestResult
from etf_backtest.core.etf_rules import EtfRuleEngine
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.fill import FillModel, SlippageModel
from etf_backtest.core.market import MarketBarView
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.position import Position
from etf_backtest.data.mysql import QmtDailyDataset
from etf_backtest.data.portal import DailyDataPortal
from etf_backtest.evaluation.backtest_metrics import (
    BacktestMetricResult,
    BacktestMetrics,
    DailyMetricRow,
    TradeMetricRow,
)
from etf_backtest.evaluation.backtest_plots import render_backtest_plots
from etf_backtest.file_utils import sha256_file
from etf_backtest.output.writer import BacktestOutputWriter, ModelArtifacts
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.model import LoadedModelComponents
from etf_backtest.strategy.model_contracts import (
    DateRange,
    ModelDataIdentity,
    ModelWorkflow,
    ModelWorkflowResult,
    prediction_rows,
)
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_data import DailyModelDatasetBuilder
from etf_backtest.universe.resolver import FrozenUniverse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SYSTEM_CONFIG = _PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"


# 保存训练后策略、训练工作流、训练结果与各数据切分样本数，供回测和模型文件输出。
@dataclass(slots=True)
class _ModelRuntime:
    strategy: DailyModelStrategy
    workflow: ModelWorkflow
    result: ModelWorkflowResult
    sample_counts: tuple[int, int, int]


# 一次回测的总入口：准备配置、数据与策略，运行逐日引擎并输出结果；try 之前的准备异常由外层入口处理。
def run_experiment(
    experiment_path: Path,
    *,
    system_path: Path = _DEFAULT_SYSTEM_CONFIG,
    project_root: Path = _PROJECT_ROOT,
) -> dict[str, object]:
    """只运行一个策略，并以原子方式发布其固定结果集。"""

    root = Path(project_root).resolve()
    prepared = prepare_experiment(
        experiment_path,
        system_path=system_path,
        project_root=root,
    )
    config = build_backtest_config(prepared)
    runs_dir = (root / config.runs_dir).resolve()
    if not runs_dir.is_relative_to(root):
        raise ValueError("runs_dir must stay inside the project")
    run_id = _run_id(prepared.experiment.case)
    writer = BacktestOutputWriter(runs_dir)
    metadata: dict[str, object] = {
        "name": prepared.experiment.name,
        "case": prepared.experiment.case,
        "experiment_path": prepared.experiment_path,
        "experiment_sha256": sha256_file(prepared.experiment_path),
        "system_path": Path(system_path).resolve(strict=True),
        "resolved_config": config.resolved_dict(),
        "rule_source_sha256": (prepared.strategy_source_sha256 if isinstance(prepared, RuleStrategySource) else None),
    }
    sql_engine: Engine | None = None
    try:
        sql_engine = create_database_engine(config.database)
        model_runtime: _ModelRuntime | None = None
        strategy: BaseStrategy
        runtime = build_backtest_runtime(
            config=config, project_root=root, engine=sql_engine,
            load_start=_load_start(prepared), load_end=config.end_date,
        )
        # 规则直接计算目标；模型先训练一次，再交给同一个逐日回测引擎。
        if isinstance(prepared, RuleStrategySource):
            strategy = prepared.strategy
        else:
            model_runtime = _build_model(
                config, prepared.components, runtime.portal.views_through(config.end_date)
            )
            strategy = model_runtime.strategy
        result = _run_backtest(config, runtime, strategy)
        run_dir = _write_results(
            config, prepared, runtime, result, model_runtime,
            writer=writer, runs_dir=runs_dir, run_id=run_id, metadata=metadata,
        )
        return {
            "status": "success",
            "experiment": prepared.experiment.name,
            "case": prepared.experiment.case,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "daily_count": len(result.daily_snapshots),
            "order_count": len(result.orders),
            "trade_count": len(result.fills),
        }
    except Exception as exc:
        writer.write_failure(run_id=run_id, run_metadata=metadata, error=exc)
        raise
    finally:
        if sql_engine is not None:
            sql_engine.dispose()


# 加载策略来源，检查规则资源路径和密码可解析性；此处不验证数据库网络连接。
def prepare_experiment(
    experiment_path: Path, *, system_path: Path = _DEFAULT_SYSTEM_CONFIG,
    project_root: Path = _PROJECT_ROOT,
) -> StrategySource:
    source = load_strategy_source(experiment_path, system_path=system_path)
    root = Path(project_root).resolve()
    required_project_resource(source.system.limit_rules_csv, root, "limit rule CSV")
    required_project_resource(source.system.limit_rules_manifest, root, "limit rule manifest")
    source.system.database.resolved_password()
    return source


# 向回测开始日或模型训练开始日前预留历史窗口；使用至少 90 个日历日的估算回溯。
def _load_start(prepared: StrategySource) -> date:
    if isinstance(prepared, RuleStrategySource):
        return prepared.experiment.start_date - timedelta(
            days=max(90, prepared.rule.lookback_trading_days * 3)
        )
    assert isinstance(prepared, ModelStrategySource)
    lookback = prepared.components.feature_builder.required_history_trading_days
    return prepared.components.settings.train_range.start_date - timedelta(days=max(90, lookback * 3))


# 从快照配置提取模型数据身份，供训练、保存与推理校验一致性。
def _model_identity(config: BacktestConfig) -> ModelDataIdentity:
    return ModelDataIdentity(
        dataset_version=config.data_snapshot.dataset_version,
        manifest_sha256=config.data_snapshot.manifest_sha256,
        snapshot_started_at_utc=config.data_snapshot.snapshot_started_at_utc,
    )


# 构造训练／验证／测试数据集并训练一次，检查产物元数据后包装成 DailyModelStrategy。
def _build_model(
    config: BacktestConfig,
    components: LoadedModelComponents,
    market_views: Sequence[MarketBarView],
) -> _ModelRuntime:
    assert isinstance(config.strategy, ModelStrategyConfig)
    ranges = (
        DateRange(config.strategy.train_start, config.strategy.train_end),
        DateRange(config.strategy.valid_start, config.strategy.valid_end),
        DateRange(config.strategy.test_start, config.strategy.test_end),
    )
    dataset = DailyModelDatasetBuilder(components.feature_builder).build(
        market_views=market_views,
        train_range=ranges[0],
        valid_range=ranges[1],
        test_range=ranges[2],
    )
    data_identity = _model_identity(config)
    workflow = components.create_workflow(data_identity)
    result = workflow.fit(dataset)
    metadata = result.bundle.metadata
    if (metadata.train_range, metadata.valid_range, metadata.test_range) != ranges:
        raise RuntimeError("fitted model split metadata does not match configured ranges")
    if metadata.data_identity != data_identity:
        raise RuntimeError("fitted model data identity does not match the backtest snapshot")
    if metadata.trained_through >= config.start_date:
        raise ValueError("fitted model must be trained before the first backtest signal")
    return _ModelRuntime(
        strategy=DailyModelStrategy(
            feature_builder=components.feature_builder,
            bundle=result.bundle,
            portfolio=components.settings.portfolio,
        ),
        workflow=workflow,
        result=result,
        sample_counts=(len(dataset.train), len(dataset.valid), len(dataset.test)),
    )


# 用初始资金建立零持仓账户，组合费用、滑点、规则审批和订单生成组件，再运行 BacktestEngine 并核对日快照覆盖。
def _run_backtest(
    config: BacktestConfig, runtime: BacktestRuntime, strategy: BaseStrategy,
) -> BacktestResult:
    """创建账户与成交组件，执行逐日回测并检查净值日期覆盖。"""
    account = Account(
        cash=config.initial_cash,
        positions={
            info.symbol: Position(
                symbol=info.symbol,
                turnover_rule=runtime.rule_resolver.resolve(
                    info.symbol,
                    max(config.start_date, info.list_date),
                ).turnover_rule,
            )
            for info in runtime.universe.etf_infos
        },
    )
    fill_model = FillModel(
        fee_model=FeeModel(config.fee),
        slippage_model=SlippageModel(config.slippage),
    )
    result = BacktestEngine(
        portal=runtime.portal,
        account=account,
        strategy=strategy,
        rule_resolver=runtime.rule_resolver,
        rule_engine=EtfRuleEngine(
            fill_model=fill_model,
            volume_participation_rate=config.volume_participation_rate,
        ),
        order_generator=OrderGenerator(),
        fill_model=fill_model,
    ).run(start_date=config.start_date, end_date=config.end_date)
    expected_dates = runtime.portal.trading_calendar.trading_dates(config.start_date, config.end_date)
    if tuple(row.trade_date for row in result.daily_snapshots) != expected_dates:
        raise RuntimeError("daily NAV does not exactly cover configured SSE dates")
    return result


# 把回测快照和成交交给指标、图表及写入器，模型产物在临时目录有效期间一并发布。
def _write_results(
    config: BacktestConfig, prepared: StrategySource, runtime: BacktestRuntime,
    result: BacktestResult, model_runtime: _ModelRuntime | None,
    *, writer: BacktestOutputWriter, runs_dir: Path, run_id: str,
    metadata: dict[str, object],
) -> Path:
    """整理指标、来源和图表，并在模型临时文件有效期内写完所有结果。"""
    daily_rows = tuple(
        DailyMetricRow(row.trade_date, row.cash, row.market_value, row.total_asset)
        for row in result.daily_snapshots
    )
    calculated_metrics = _metrics(config.initial_cash, result, daily_rows)
    metadata["provenance"] = _provenance(
        config,
        runtime.dataset,
        runtime.portal,
        runtime.universe,
        runtime.rule_resolver,
    )
    metadata["rule_settings"] = (
        prepared.rule.settings.resolved_dict() if isinstance(prepared, RuleStrategySource) else None
    )
    plots = render_backtest_plots(initial_cash=config.initial_cash, daily_rows=daily_rows)
    # 规则与模型共用一次写出；模型附件写完后才清理临时目录。
    with ExitStack() as resources:
        model_artifacts = None
        if model_runtime is not None:
            assert isinstance(prepared, ModelStrategySource)
            metadata["model"] = _model_metadata(model_runtime, prepared.components)
            temp_directory = resources.enter_context(TemporaryDirectory(prefix="qmt-model-bundle-"))
            bundle_path = model_runtime.workflow.save(
                Path(temp_directory) / model_runtime.workflow.bundle_filename,
                source_run_dir=runs_dir / run_id,
            )
            model_artifacts = ModelArtifacts(
                bundle_path=bundle_path,
                bundle_filename=model_runtime.workflow.bundle_filename,
                predictions=prediction_rows(model_runtime.strategy.predictions),
            )
        return writer.write_success(
            run_id=run_id,
            run_metadata=metadata,
            result=result,
            metrics=calculated_metrics,
            plots=plots,
            model_artifacts=model_artifacts,
        )



# 将账户快照与成交记录投影为绩效输入，调用统一指标计算器。
def _metrics(
    initial_cash: Decimal, result: BacktestResult, daily_rows: tuple[DailyMetricRow, ...],
) -> BacktestMetricResult:
    return BacktestMetrics.calculate(
        initial_cash=initial_cash,
        daily_rows=daily_rows,
        trade_rows=tuple(
            TradeMetricRow(
                trade_amount=fill.trade_amount,
                fee=fill.fee,
                base_trade_price=fill.base_trade_price,
                fill_price=fill.fill_price,
                fill_quantity=fill.fill_quantity,
            )
            for fill in result.fills
        ),
    )


# 汇总数据库快照、日历、证券范围和规则／辅助数据来源，记录近似或补齐数据的身份。
def _provenance(
    config: BacktestConfig,
    dataset: QmtDailyDataset,
    portal: DailyDataPortal,
    universe: FrozenUniverse,
    rule_resolver: EffectiveDatedEtfRuleResolver,
) -> Mapping[str, object]:
    resource_identity = rule_resolver.resource_identity
    if resource_identity is None:
        raise RuntimeError("effective rule resource identity is missing")
    return {
        "data_mode": config.data_snapshot.data_mode,
        "pit_compliant": False,
        "dataset_version": portal.dataset_version,
        "snapshot_started_at_utc": config.data_snapshot.snapshot_started_at_utc,
        "calendar_source": portal.trading_calendar.calendar_source,
        "calendar_version": portal.trading_calendar.calendar_version,
        "calendar_policy": portal.trading_calendar.calendar_policy,
        "input_manifest_sha256": config.data_snapshot.manifest_sha256,
        "rule_resource": asdict(resource_identity),
        "universe": universe.csv_rows(),
        "universe_approximation_flags": universe.approximation_flags,
        "loaded_share_record_count": len(dataset.share_records),
        "loaded_huijin_ratio_record_count": len(dataset.huijin_ratio_records),
        "loaded_index_record_count": len(dataset.index_records),
        "loaded_explicit_price_limit_count": dataset.explicit_price_limit_count,
        "loaded_derived_price_limit_count": dataset.derived_price_limit_fallback_count,
        "status_only_suspension_carry_keys": dataset.suspension_carry_keys,
    }


# 提取模型训练与预测相关元数据，供实验结果留档。
def _model_metadata(
    runtime: _ModelRuntime, components: LoadedModelComponents
) -> Mapping[str, object]:
    fit_summary = dict(runtime.result.fit_summary)
    payload: dict[str, object] = {
        "backend": components.settings.backend,
        "bundle": runtime.result.bundle.metadata.to_dict(),
        "source_path": components.source_path,
        "source_sha256": components.source_sha256,
        "settings": components.settings.resolved_dict(),
        "train_sample_count": runtime.sample_counts[0],
        "validation_sample_count": runtime.sample_counts[1],
        "test_sample_count": runtime.sample_counts[2],
        "fit_summary": fit_summary,
        "validation_metrics": asdict(runtime.result.validation_metrics),
        "test_metrics": asdict(runtime.result.test_metrics),
    }
    if components.settings.backend == "torch":
        payload.update(
            {
                "best_epoch": fit_summary["best_epoch"],
                "epochs_trained": fit_summary["epochs_trained"],
                "best_validation_loss": fit_summary["best_validation_loss"],
                "scaler_fit_scope": "TRAIN_ONLY",
            }
        )
    return payload


# 用策略类型、UTC 时间和随机后缀生成本次运行标识。
def _run_id(case: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{case}-{uuid4().hex[:8]}"




__all__ = [
    "StrategySource",
    "prepare_experiment",
    "run_experiment",
]
