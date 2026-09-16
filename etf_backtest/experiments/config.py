"""严格且不含密码的用户实验配置。

公开实验文档只包含策略作者负责的选项。数据库凭据、快照身份和有效规则资源保存在
:class:`SystemSettings` 中，仅在构建具体 ``BacktestConfig`` 时合并。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from etf_backtest.config.schema import (
    CALENDAR_POLICY,
    BacktestConfig,
    DatabaseConfig,
    DataSnapshotConfig,
    FeeConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
    SlippageConfig,
    UniverseConfig,
)

_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_FORBIDDEN_NAME_CHARACTERS = frozenset('<>:"/\\|?*')


class _StrictExperimentModel(BaseModel):
    """实验文档共用的不可变、拒绝未知字段配置基类。"""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# 拒绝资金和比例的浮点输入，避免配置阶段引入二进制精度误差。
def _reject_float(value: object, field_name: str) -> object:
    if isinstance(value, float):
        raise TypeError(f"{field_name} must be supplied as decimal text")
    return value


# 检查名称可作为单个 Windows 路径组件，拒绝非法字符和保留名。
def _safe_component(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if normalized != value or not normalized:
        raise ValueError(f"{field_name} must be nonblank without surrounding whitespace")
    if len(normalized) > 80:
        raise ValueError(f"{field_name} must not exceed 80 characters")
    if normalized in {".", ".."} or any(
        character in _FORBIDDEN_NAME_CHARACTERS or ord(character) < 32 for character in normalized
    ):
        raise ValueError(f"{field_name} must be one safe path component")
    if normalized.endswith((".", " ")):
        raise ValueError(f"{field_name} must not end with a dot or space")
    if normalized.partition(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{field_name} uses a reserved Windows filename")
    return normalized


def safe_relative_path(
    value: object,
    field_name: str,
    *,
    suffix: str | None = None,
) -> Path:
    """返回规范化且不能越出项目根目录的相对路径。"""

    if not isinstance(value, str | Path):
        raise TypeError(f"{field_name} must be a path string")
    normalized = Path(str(value).replace("\\", "/"))
    if (
        not normalized.parts
        or normalized.is_absolute()
        or normalized.drive
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    for part in normalized.parts:
        _safe_component(part, field_name)
    if suffix is not None and normalized.suffix.casefold() != suffix.casefold():
        raise ValueError(f"{field_name} must end in {suffix}")
    return normalized


class SystemSettings(_StrictExperimentModel):
    """由维护者管理的数据、执行默认值、输出根目录和规则资源。"""

    database: DatabaseConfig
    data_snapshot: DataSnapshotConfig = Field(default_factory=DataSnapshotConfig)
    limit_rules_csv: Path = Path("resources/limit_rules/etf_price_limit_20pct.csv")
    limit_rules_manifest: Path = Path("resources/limit_rules/manifest.json")
    calendar_policy: Literal["SSE_FOR_ALL"] = CALENDAR_POLICY
    fee: FeeConfig = Field(default_factory=FeeConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    volume_participation_rate: Decimal = Decimal("0.20")
    runs_dir: Path = Path("runs")

    # 校验规则 CSV 为项目内安全相对路径且扩展名正确。
    @field_validator("limit_rules_csv", mode="before")
    @classmethod
    def _rule_csv_path(cls, value: object) -> Path:
        return safe_relative_path(value, "limit_rules_csv", suffix=".csv")

    # 校验规则清单为项目内安全 JSON 相对路径。
    @field_validator("limit_rules_manifest", mode="before")
    @classmethod
    def _rule_manifest_path(cls, value: object) -> Path:
        return safe_relative_path(value, "limit_rules_manifest", suffix=".json")

    # 在转换前拒绝浮点成交量参与比例。
    @field_validator("volume_participation_rate", mode="before")
    @classmethod
    def _volume_rate(cls, value: object) -> object:
        return _reject_float(value, "volume_participation_rate")

    # 校验结果根目录为安全的项目相对路径。
    @field_validator("runs_dir", mode="before")
    @classmethod
    def _runs_path(cls, value: object) -> Path:
        return safe_relative_path(value, "runs_dir")

    # 检查系统配置中的成交量参与比例位于 (0, 1]。
    @model_validator(mode="after")
    def _valid_execution_defaults(self) -> Self:
        if not self.volume_participation_rate.is_finite() or not Decimal(
            "0"
        ) < self.volume_participation_rate <= Decimal("1"):
            raise ValueError("volume_participation_rate must be in (0, 1]")
        return self

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> Self:
        """从已校验的运行配置提取项目系统维护的值。"""

        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be BacktestConfig")
        return cls(
            database=config.database,
            data_snapshot=config.data_snapshot,
            limit_rules_csv=config.limit_rules_csv,
            limit_rules_manifest=config.limit_rules_manifest,
            calendar_policy=config.calendar_policy,
            fee=config.fee,
            slippage=config.slippage,
            volume_participation_rate=config.volume_participation_rate,
            runs_dir=config.runs_dir,
        )


class UserExperimentConfig(_StrictExperimentModel):
    """实验共用输入；所有策略专属设置均在 Python 中维护。"""

    name: str
    start_date: date
    end_date: date
    initial_cash: Decimal = Decimal("1000000")
    universe: UniverseConfig
    case: Literal["rule", "model"]

    # 校验实验名称非空且可以安全用于文件与结果标识。
    @field_validator("name", mode="before")
    @classmethod
    def _experiment_name(cls, value: object) -> str:
        return _safe_component(value, "name")

    # 在类型转换前检查初始资金输入格式。
    @field_validator("initial_cash", mode="before")
    @classmethod
    def _decimal_core_values(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", "Decimal")
        return _reject_float(value, field_name)

    # 检查实验日期顺序与初始资金正值约束。
    @model_validator(mode="after")
    def _valid_experiment(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        return self

    def build_case(
        self,
        system: SystemSettings,
        *,
        strategy: RuleStrategyConfig | ModelStrategyConfig,
    ) -> BacktestConfig:
        """合并代码解析的策略设置、实验共用输入和系统输入。"""

        if not isinstance(system, SystemSettings):
            raise TypeError("system must be SystemSettings")
        if self.case == "rule" and not isinstance(strategy, RuleStrategyConfig):
            raise TypeError("rule case requires RuleStrategyConfig")
        if self.case == "model" and not isinstance(strategy, ModelStrategyConfig):
            raise TypeError("model case requires ModelStrategyConfig")
        return BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_cash=self.initial_cash,
            database=system.database,
            data_snapshot=system.data_snapshot,
            universe=self.universe,
            strategy=strategy,
            fee=system.fee,
            slippage=system.slippage,
            volume_participation_rate=system.volume_participation_rate,
            runs_dir=system.runs_dir,
            limit_rules_csv=system.limit_rules_csv,
            limit_rules_manifest=system.limit_rules_manifest,
            calendar_policy=system.calendar_policy,
        )


def load_user_experiment_config(path: Path) -> UserExperimentConfig:
    """加载单个 UTF-8 用户 YAML 文档，不转换系统设置。"""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration root must be a mapping")
    return UserExperimentConfig.model_validate(payload)


# 读取系统 YAML 的同时校验路径、数据库与执行参数，返回 SystemSettings；不在此执行交易或连接券商。
def load_system_settings(path: Path) -> SystemSettings:
    """加载单个 UTF-8 系统维护者设置 YAML 文档。"""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("system settings root must be a mapping")
    return SystemSettings.model_validate(payload)


__all__ = [
    "SystemSettings",
    "UserExperimentConfig",
    "load_system_settings",
    "load_user_experiment_config",
    "safe_relative_path",
]
