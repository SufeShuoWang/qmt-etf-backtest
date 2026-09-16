"""以物理账户和稳定策略 ID 为边界的严格 PAPER 配置。"""

from __future__ import annotations

import os
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    field_validator,
    model_validator,
)

from etf_backtest.experiments.config import safe_relative_path


# 模拟盘配置共同基类：字段不可重新赋值、拒绝未知字段，并校验默认值。
class _LiveModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# 声明 PAPER 账户模式、系统配置路径及账户标识来源。
class LiveAccountConfig(_LiveModel):
    mode: Literal["PAPER"]
    system_path: Path
    configured_account_id: str | None = Field(default=None, alias="account_id")
    account_id_env: str | None = None
    account_type: Literal["STOCK"] = "STOCK"

    # 校验账户引用的系统配置路径位于项目相对路径范围。
    @field_validator("system_path", mode="before")
    @classmethod
    def _project_path(cls, value: object, info: object) -> Path:
        return safe_relative_path(value, str(getattr(info, "field_name", "path")))

    # 规范可选账户文本字段，拒绝仅含空白的值。
    @field_validator("configured_account_id", "account_id_env")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("account identity fields must not be blank")
        return normalized

    # 要求直接账户 ID 和环境变量来源恰好配置一个。
    @model_validator(mode="after")
    def _one_account_source(self) -> Self:
        if (self.configured_account_id is None) == (self.account_id_env is None):
            raise ValueError("account requires exactly one of account_id or account_id_env")
        return self

    # 从配置或指定环境变量取得账户 ID，缺失时明确报错。
    def account_id(self) -> str:
        if self.configured_account_id is not None:
            return self.configured_account_id
        assert self.account_id_env is not None
        try:
            value = os.environ[self.account_id_env].strip()
        except KeyError:
            raise ValueError(
                f"account environment variable is not set: {self.account_id_env}"
            ) from None
        if not value:
            raise ValueError("configured account environment variable is blank")
        return value


# 保存 MiniQMT 数据目录、会话编号和重连／轮询间隔设置。
class MiniQmtConfig(_LiveModel):
    userdata_path: Path
    session_id: StrictInt
    reconnect_interval_seconds: StrictInt

    # 检查会话编号和重连间隔为正整数。
    @model_validator(mode="after")
    def _positive_values(self) -> Self:
        if self.session_id <= 0 or self.reconnect_interval_seconds <= 0:
            raise ValueError("MiniQMT session and reconnect interval must be positive")
        return self


# 声明每日信号计算的触发时刻。
class SignalConfig(_LiveModel):
    run_time: time


# 声明收盘对账与快照作业的触发时刻。
class EodConfig(_LiveModel):
    run_time: time = time(20, 30)


# 声明尾盘限价执行政策、卖买阶段截止时间、报价有效期及整手约束。
class ExecutionConfig(_LiveModel):
    policy: Literal["NEAR_CLOSE_LIMIT"]
    submit_start: time
    sell_phase_deadline: time
    stop_new_orders: time
    cancel_open_orders: time
    cancel_confirm_timeout_seconds: StrictInt = 60
    price_offset_ticks: StrictInt
    quote_stale_seconds: StrictInt
    lot_size: StrictInt
    order_type: Literal["FIX_PRICE"]

    # 检查执行时刻严格递增，且报价偏移、有效期、整手和撤单等待参数合法。
    @model_validator(mode="after")
    def _validate_execution(self) -> Self:
        if not (
            self.submit_start
            < self.sell_phase_deadline
            < self.stop_new_orders
            < self.cancel_open_orders
        ):
            raise ValueError(
                "execution times must satisfy submit_start < sell_phase_deadline < "
                "stop_new_orders < cancel_open_orders"
            )
        if (
            self.price_offset_ticks < 0
            or self.quote_stale_seconds <= 0
            or self.lot_size <= 0
            or self.cancel_confirm_timeout_seconds <= 0
        ):
            raise ValueError("execution offsets, staleness and lot size are invalid")
        return self


# 声明模拟盘状态数据库连接信息，与历史行情数据库配置可分别提供。
class LiveStateDatabaseConfig(_LiveModel):
    host: str = "127.0.0.1"
    port: StrictInt = 3306
    database: str
    user: str
    password: SecretStr | None = Field(default=None, exclude=True)
    password_env: str | None = None
    charset: Literal["utf8mb4"] = "utf8mb4"
    connect_timeout_seconds: StrictInt = 10

    # 检查状态数据库连接字段、端口、密码来源和超时设置。
    @model_validator(mode="after")
    def _validate_database(self) -> Self:
        if not 1 <= self.port <= 65535:
            raise ValueError("state database port must be between 1 and 65535")
        if not all(value.strip() for value in (self.host, self.database, self.user)):
            raise ValueError("state database text fields must not be blank")
        if self.password is None and self.password_env is None:
            raise ValueError("state database password or password_env is required")
        if self.password_env is not None and not self.password_env.strip():
            raise ValueError("state database password_env must not be blank")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("state database connect timeout must be positive")
        return self

    # 优先使用显式密码，否则从指定环境变量解析状态库密码。
    def resolved_password(self) -> str:
        if self.password is not None:
            return self.password.get_secret_value()
        assert self.password_env is not None
        try:
            return os.environ[self.password_env]
        except KeyError:
            raise ValueError(
                f"state database password environment variable is not set: {self.password_env}"
            ) from None


# 声明单笔／单日委托金额、最小委托金额与组合目标仓位上限。
class LiveRiskConfig(_LiveModel):
    max_single_order_notional: Decimal
    max_daily_order_notional: Decimal
    min_order_notional: Decimal
    max_total_target_weight: Decimal

    # 拒绝用浮点数输入风控金额与比例。
    @field_validator("*", mode="before")
    @classmethod
    def _reject_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise TypeError("risk Decimal values must be supplied as decimal text")
        return value

    # 检查风控阈值有限且为正，组合仓位上限不超过 1。
    @model_validator(mode="after")
    def _positive_risk(self) -> Self:
        values = (
            self.max_single_order_notional,
            self.max_daily_order_notional,
            self.min_order_notional,
            self.max_total_target_weight,
        )
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError("risk limits must be finite and positive")
        if self.max_total_target_weight > 1:
            raise ValueError("max_total_target_weight must not exceed one")
        return self


# 声明模拟盘推理使用的模型后端和已训练产物路径。
class ModelLiveConfig(_LiveModel):
    backend: Literal["torch", "xgboost"]
    bundle_path: Path
    device: str = "cpu"

    @field_validator("device", mode="before")
    @classmethod
    def _device(cls, value: object) -> str:
        """独立选择模拟盘推理设备，无需与训练设备相同。"""
        from etf_backtest.strategy.model_device import normalize_model_device

        return normalize_model_device(value)

    # 校验模型产物位于项目内的安全相对路径。
    @field_validator("bundle_path", mode="before")
    @classmethod
    def _bundle_path(cls, value: object) -> Path:
        path = safe_relative_path(value, "model.bundle_path")
        return path

    # 要求 Torch 使用 .pt、XGBoost 使用 .ubj 产物后缀。
    @model_validator(mode="after")
    def _matching_suffix(self) -> Self:
        expected = {
            "torch": ".pt",
            "xgboost": ".ubj",
        }[self.backend]
        if self.bundle_path.suffix.casefold() != expected:
            raise ValueError(f"{self.backend} bundle_path must end with {expected}")
        return self


# 声明单个模拟盘策略的身份、实验路径、初始虚拟资金、调度锚点及可选模型产物。
class LiveStrategyConfig(_LiveModel):
    strategy_id: str
    case: Literal["rule", "model"]
    experiment_path: Path
    initial_capital: Decimal
    schedule_anchor_date: date
    model: ModelLiveConfig | None = None

    # 规范策略 ID，拒绝空值与用于组合标识分隔的竖线。
    @field_validator("strategy_id")
    @classmethod
    def _strategy_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "|" in normalized:
            raise ValueError("strategy_id must be non-blank and cannot contain '|'")
        return normalized

    # 校验策略实验文件为项目内的 YAML 相对路径。
    @field_validator("experiment_path", mode="before")
    @classmethod
    def _experiment_path(cls, value: object) -> Path:
        return safe_relative_path(value, "strategy.experiment_path", suffix=".yaml")

    # 拒绝浮点初始资金输入，保留十进制精度。
    @field_validator("initial_capital", mode="before")
    @classmethod
    def _initial_capital(cls, value: object) -> object:
        if isinstance(value, float):
            raise TypeError("strategy initial_capital must be decimal text")
        return value

    # 检查初始资金为正，并要求模型策略提供模型配置、规则策略不混入模型配置。
    @model_validator(mode="after")
    def _validate_strategy(self) -> Self:
        if not self.initial_capital.is_finite() or self.initial_capital <= 0:
            raise ValueError("strategy initial_capital must be finite and positive")
        if self.case == "rule" and self.model is not None:
            raise ValueError("Rule strategy must not contain model")
        if self.case == "model" and self.model is None:
            raise ValueError("Model strategy requires model backend and bundle_path")
        return self


# 组织模拟盘账户、单策略、MiniQMT、信号、执行、收盘、状态库与风控配置。
class LiveConfig(_LiveModel):
    config_version: Literal["4.0"]
    account: LiveAccountConfig
    strategy: LiveStrategyConfig
    miniqmt: MiniQmtConfig
    signal: SignalConfig
    execution: ExecutionConfig
    eod: EodConfig = Field(default_factory=EodConfig)
    state_database: LiveStateDatabaseConfig | None = None
    risk: LiveRiskConfig

    # 要求尾盘执行、撤单、收盘对账和信号计算时刻按流程严格递增。
    @model_validator(mode="after")
    def _validate_live(self) -> Self:
        if not (
            self.execution.submit_start
            < self.execution.sell_phase_deadline
            < self.execution.stop_new_orders
            < self.execution.cancel_open_orders
            < self.eod.run_time
            < self.signal.run_time
        ):
            raise ValueError(
                "live times must satisfy submit_start < sell_phase_deadline < "
                "stop_new_orders < cancel_open_orders < eod.run_time < signal.run_time"
            )
        return self

    # 将项目相对配置路径解析为绝对路径，并再次检查没有越出项目根目录。
    def project_path(self, path: Path, project_root: Path) -> Path:
        root = Path(project_root).resolve()
        result = (root / path).resolve()
        if not result.is_relative_to(root):
            raise ValueError("configured project path escapes the project root")
        return result


# 读取模拟盘 YAML，校验结构并创建 LiveConfig。
def load_live_config(path: Path) -> LiveConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("live configuration root must be a mapping")
    return LiveConfig.model_validate(payload)


__all__ = [
    "EodConfig",
    "LiveAccountConfig",
    "LiveConfig",
    "LiveStateDatabaseConfig",
    "LiveStrategyConfig",
    "ModelLiveConfig",
    "load_live_config",
]
