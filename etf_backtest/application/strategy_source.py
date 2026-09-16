"""一次读取实验、系统配置和用户策略，供回测与模拟盘共用。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etf_backtest.config.schema import BacktestConfig, ModelStrategyConfig, RuleStrategyConfig
from etf_backtest.experiments.config import (
    SystemSettings, UserExperimentConfig, load_system_settings, load_user_experiment_config,
)
from etf_backtest.file_utils import sha256_file
from etf_backtest.strategy.loader import load_user_rule
from etf_backtest.strategy.model import LoadedModelComponents, load_user_model_components
from etf_backtest.strategy.rule import SimpleRuleStrategy, UserRule


# 保存规则实验的来源路径、两类配置、用户规则实例及其适配器；每个实验各自持有这些对象。
@dataclass(frozen=True, slots=True)
class RuleStrategySource:
    experiment_path: Path
    system_path: Path
    experiment: UserExperimentConfig
    system: SystemSettings
    rule: UserRule
    strategy: SimpleRuleStrategy
    strategy_source_sha256: str


# 保存模型实验的来源路径、配置和已加载的特征／模型组件，训练在后续流程进行。
@dataclass(frozen=True, slots=True)
class ModelStrategySource:
    experiment_path: Path
    system_path: Path
    experiment: UserExperimentConfig
    system: SystemSettings
    components: LoadedModelComponents
    strategy_source_sha256: str


StrategySource = RuleStrategySource | ModelStrategySource


# 读取实验和系统配置；根据 case 加载同目录的 rule.py 或 model.py，返回对应来源对象。
def load_strategy_source(
    experiment_path: Path, *, system_path: Path, case: str | None = None,
    system_settings: SystemSettings | None = None,
) -> StrategySource:
    source = Path(experiment_path).resolve(strict=True)
    if not source.is_file() or source.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("experiment must be one existing YAML file")
    experiment = load_user_experiment_config(source)
    if case is not None and experiment.case != case:
        raise ValueError(f"strategy source requires experiment case: {case}")
    system_path = Path(system_path).resolve(strict=True)
    # 多策略启动复用同一份已校验配置；独立回测仍由这里读取。
    if system_settings is not None and not isinstance(system_settings, SystemSettings):
        raise TypeError("system_settings must be SystemSettings")
    system = load_system_settings(system_path) if system_settings is None else system_settings
    if experiment.case == "rule":
        rule_path = source.parent / "rule.py"
        rule = load_user_rule(rule_path, allowed_root=source.parent)
        return RuleStrategySource(
            source, system_path, experiment, system, rule,
            SimpleRuleStrategy(rule=rule), sha256_file(rule_path),
        )
    components = load_user_model_components(source.parent / "model.py", allowed_root=source.parent)
    return ModelStrategySource(
        source, system_path, experiment, system, components, components.source_sha256,
    )


# 从当前策略提取已生效设置，再通过实验对象 build_case() 显式合并系统字段；不会覆盖为另一策略的默认值。
def build_backtest_config(source: StrategySource) -> BacktestConfig:
    """统一合并策略 Python 设置、实验 YAML 和系统配置，供回测与信号计算使用。"""
    strategy: RuleStrategyConfig | ModelStrategyConfig
    if isinstance(source, RuleStrategySource):
        strategy = RuleStrategyConfig(
            lookback_trading_days=source.rule.lookback_trading_days,
            rebalance_every_trading_days=source.rule.rebalance_every_trading_days,
            target_weight=source.rule.target_weight,
        )
    else:
        settings = source.components.settings
        if settings.valid_range.end_date >= source.experiment.start_date:
            raise ValueError("model validation range must end before the backtest starts")
        strategy = ModelStrategyConfig(
            max_total_weight=settings.portfolio.max_total_weight,
            train_start=settings.train_range.start_date,
            train_end=settings.train_range.end_date,
            valid_start=settings.valid_range.start_date,
            valid_end=settings.valid_range.end_date,
            test_start=source.experiment.start_date,
            test_end=source.experiment.end_date,
        )
    return source.experiment.build_case(source.system, strategy=strategy)
