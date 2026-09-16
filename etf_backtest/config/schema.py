"""QMT 日频 ETF 回测器的封闭配置契约。

0.2 版有意不开放频率、复权模式、执行价格或日内调度开关。系统在上交所日线收盘时
观察信号，对应目标只能在下一个上交所交易日收盘执行。
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    field_validator,
    model_validator,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")
CALENDAR_POLICY: Literal["SSE_FOR_ALL"] = "SSE_FOR_ALL"
DATA_MODE: Literal["RETROSPECTIVE_SNAPSHOT"] = "RETROSPECTIVE_SNAPSHOT"
RULE_MODE: Literal["LATEST_SNAPSHOT_WITH_2020_SEED"] = "LATEST_SNAPSHOT_WITH_2020_SEED"
PACKAGE_MANIFEST_SHA256 = "39502872c5544cbf4dc1671ea67488479021e39db97f34aecdedf1bc56cc62a7"

_STANDARD_SYMBOL = re.compile(r"^(?:(SH|SZ)\.)?(\d{6})(?:\.(SH|SZ))?$")
_INDEX_CODE = re.compile(r"^[0-9A-Z][0-9A-Z.]{1,31}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POOL_NAMES = frozenset({"domestic_stock_etf", "gold_etf", "all_supported_etf"})


class _FrozenConfig(BaseModel):
    """不可变且拒绝未知字段的配置基类。"""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


# 去除配置文本首尾空白，并拒绝空字符串或非文本输入。
def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


# 拒绝浮点输入，要求金额或比例使用十进制文本等可控精度的表示。
def _reject_float(value: object, field_name: str) -> object:
    if isinstance(value, float):
        raise TypeError(f"{field_name} must be supplied as decimal text")
    return value


# 检查 Decimal 数值有限且非负，供费用、滑点等配置复用。
def _finite_non_negative(value: Decimal, field_name: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def normalize_symbol(value: str) -> str:
    """将六位 ETF 代码规范化为 ``SH.######`` 或 ``SZ.######``。"""

    normalized = _non_blank(value, "symbol").upper()
    match = _STANDARD_SYMBOL.fullmatch(normalized)
    if match is None:
        raise ValueError("symbol must be a six-digit SH/SZ ETF code")
    leading_exchange, code, trailing_exchange = match.groups()
    if leading_exchange and trailing_exchange:
        raise ValueError("symbol cannot contain two exchange qualifiers")
    inferred = "SH" if code.startswith("5") else "SZ" if code.startswith("1") else None
    if inferred is None:
        raise ValueError("supported ETF codes must start with 5 (SSE) or 1 (SZSE)")
    supplied = leading_exchange or trailing_exchange
    if supplied is not None and supplied != inferred:
        raise ValueError("symbol exchange qualifier conflicts with its code")
    return f"{inferred}.{code}"


def etf_code(value: str) -> str:
    """返回内部证券代码对应的六位 MySQL ``etf_code``。"""

    return normalize_symbol(value).partition(".")[2]


def normalize_index_code(value: str) -> str:
    """规范化数据源原生指数代码，不套用 ETF 交易所推断。"""

    normalized = _non_blank(value, "index_code").upper()
    if "." not in normalized or _INDEX_CODE.fullmatch(normalized) is None:
        raise ValueError("index_code must be a source-native dotted code")
    return normalized


class DatabaseConfig(_FrozenConfig):
    """仅具 SELECT 权限的 MySQL 账户连接设置。"""

    host: str = "127.0.0.1"
    port: StrictInt = 3306
    database: Literal["qmt_etf_quant"] = "qmt_etf_quant"
    user: str
    password: SecretStr | None = Field(default=None, exclude=True)
    password_env: str | None = "QMT_MYSQL_PASSWORD"
    charset: Literal["utf8mb4"] = "utf8mb4"
    connect_timeout_seconds: StrictInt = 10

    # 规范数据库连接字段中的文本，去除首尾空白并拒绝空值。
    @field_validator("host", "database", "user")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _non_blank(value, "database connection field")

    # 校验用于读取密码的环境变量名；允许未配置该可选来源。
    @field_validator("password_env")
    @classmethod
    def _validate_password_env(cls, value: str | None) -> str | None:
        return None if value is None else _non_blank(value, "password_env")

    # 检查端口范围、连接超时为正，并要求提供密码或密码环境变量名。
    @model_validator(mode="after")
    def _validate_connection(self) -> Self:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if self.password is None and self.password_env is None:
            raise ValueError("password or password_env is required")
        return self

    def resolved_password(self) -> str:
        """解析密码，同时避免在序列化配置中暴露。"""

        if self.password is not None:
            return self.password.get_secret_value()
        assert self.password_env is not None
        try:
            return os.environ[self.password_env]
        except KeyError:
            raise ValueError(
                f"database password environment variable is not set: {self.password_env}"
            ) from None


class DataSnapshotConfig(_FrozenConfig):
    """冻结回溯式 QMT 数据包的身份信息。"""

    dataset_version: str = "qmt_etf_quant_20260812"
    snapshot_date: date = date(2026, 8, 12)
    snapshot_started_at_utc: datetime = datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC)
    manifest_sha256: str = PACKAGE_MANIFEST_SHA256
    raw_table: Literal["etf_quote_qmt_unadjusted_daily"] = "etf_quote_qmt_unadjusted_daily"
    front_table: Literal["etf_quote_qmt_front_ratio_daily"] = "etf_quote_qmt_front_ratio_daily"
    share_table: Literal["etf_share_daily"] | None = None
    huijin_holders_csv: Path | None = None
    huijin_holders_csv_sha256: str | None = None
    index_table: Literal["index_quote_daily"] | None = None
    trade_status_table: Literal["etf_trade_status_daily"] = "etf_trade_status_daily"
    rule_index_codes: tuple[str, ...] = ()
    data_mode: Literal["RETROSPECTIVE_SNAPSHOT"] = DATA_MODE

    # 校验数据集版本字符串，确保结果可追溯到明确版本。
    @field_validator("dataset_version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return _non_blank(value, "dataset_version")

    # 检查快照开始时间包含时区，并按配置约定规范时间。
    @field_validator("snapshot_started_at_utc")
    @classmethod
    def _validate_snapshot_started_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_started_at_utc must be timezone-aware")
        return value.astimezone(UTC)

    # 校验清单 SHA-256 的格式，避免把无效摘要当作数据身份。
    @field_validator(
        "manifest_sha256",
        "huijin_holders_csv_sha256",
    )
    @classmethod
    def _validate_manifest_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if _SHA256.fullmatch(normalized) is None:
            raise ValueError("snapshot hashes must be lowercase SHA-256 digests")
        return normalized

    # 校验可选汇金 CSV 路径配置，供后续资源读取使用。
    @field_validator("huijin_holders_csv")
    @classmethod
    def _validate_huijin_csv(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        normalized = Path(str(value))
        if normalized.suffix.casefold() != ".csv":
            raise ValueError("huijin_holders_csv must be a CSV path")
        return normalized

    # 规范规则策略需要的指数代码，检查重复与格式。
    @field_validator("rule_index_codes")
    @classmethod
    def _validate_rule_index_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(normalize_index_code(code) for code in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("rule_index_codes must be unique")
        return tuple(sorted(normalized))

    # 检查基金辅助数据的路径、摘要等身份字段是否配套。
    @model_validator(mode="after")
    def _validate_rule_fund_data_identity(self) -> Self:
        if (self.huijin_holders_csv is None) != (self.huijin_holders_csv_sha256 is None):
            raise ValueError("Huijin CSV path and hash must be configured together")
        configured_index_identity = (
            self.index_table is not None,
            bool(self.rule_index_codes),
        )
        if len(set(configured_index_identity)) != 1:
            raise ValueError("index table and rule index codes must be configured together")
        return self


class UniverseConfig(_FrozenConfig):
    """显式 ETF 证券和命名资产池，解析结果取并集。"""

    symbols: tuple[str, ...] = ()
    pools: tuple[Literal["domestic_stock_etf", "gold_etf", "all_supported_etf"], ...] = ()

    # 将显式 ETF 代码转换为内部标准格式，并处理重复输入。
    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(normalize_symbol(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("symbols must be unique after normalization")
        return tuple(sorted(normalized))

    # 校验并规范资产池名称，拒绝未知或重复的池配置。
    @field_validator("pools")
    @classmethod
    def _normalize_pools(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(value not in _POOL_NAMES for value in values):
            raise ValueError("unsupported ETF pool")
        if len(values) != len(set(values)):
            raise ValueError("pools must be unique")
        return tuple(sorted(values))

    # 要求至少配置显式证券或资产池，避免生成空的证券来源。
    @model_validator(mode="after")
    def _require_source(self) -> Self:
        if not self.symbols and not self.pools:
            raise ValueError("universe requires at least one symbol or pool")
        return self


class RuleStrategyConfig(_FrozenConfig):
    """单个用户 Rule 策略已解析的调度控制项。"""

    kind: Literal["rule"] = "rule"
    lookback_trading_days: StrictInt = 20
    rebalance_every_trading_days: StrictInt = 20
    target_weight: Decimal = Decimal("0.90")

    # 拒绝未经控制的浮点目标权重输入。
    @field_validator("target_weight", mode="before")
    @classmethod
    def _reject_weight_float(cls, value: object) -> object:
        return _reject_float(value, "target_weight")

    # 检查规则配置的回看长度、调仓间隔和默认目标权重范围。
    @model_validator(mode="after")
    def _validate_weight(self) -> Self:
        if self.lookback_trading_days < 2:
            raise ValueError("lookback_trading_days must be at least two")
        if self.rebalance_every_trading_days <= 0:
            raise ValueError("rebalance_every_trading_days must be positive")
        if not Decimal("0") < self.target_weight <= Decimal("1"):
            raise ValueError("target_weight must be in (0, 1]")
        return self


class ModelStrategyConfig(_FrozenConfig):
    """单个用户 Model 工作流已解析的数据切分和仓位上限。"""

    kind: Literal["model"] = "model"
    max_total_weight: Decimal = Decimal("0.90")
    train_start: date = date(2021, 1, 1)
    train_end: date = date(2022, 12, 31)
    valid_start: date = date(2023, 1, 1)
    valid_end: date = date(2023, 12, 31)
    test_start: date = date(2024, 1, 1)
    test_end: date = date(2024, 12, 31)

    # 要求模型仓位上限采用十进制文本等非浮点输入。
    @field_validator("max_total_weight", mode="before")
    @classmethod
    def _reject_decimal_float(cls, value: object) -> object:
        return _reject_float(value, "model Decimal parameter")

    # 校验模型仓位上限及训练、验证、测试区间的先后顺序与不重叠约束。
    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if not Decimal("0") < self.max_total_weight <= Decimal("1"):
            raise ValueError("max_total_weight must be in (0, 1]")
        if self.train_end < self.train_start:
            raise ValueError("train_end must not precede train_start")
        if self.valid_end < self.valid_start:
            raise ValueError("valid_end must not precede valid_start")
        if self.test_end < self.test_start:
            raise ValueError("test_end must not precede test_start")
        if self.train_end >= self.valid_start or self.valid_end >= self.test_start:
            raise ValueError("model splits must be chronological and non-overlapping")
        return self


StrategyConfig = Annotated[
    RuleStrategyConfig | ModelStrategyConfig,
    Field(discriminator="kind"),
]


class FeeConfig(_FrozenConfig):
    """逐笔成交计算的 ETF 费用参数。"""

    commission_rate: Decimal = Decimal("0.0003")
    minimum_commission: Decimal = Decimal("5")
    stamp_duty_rate: Decimal = Decimal("0")

    # 在类型转换前拒绝浮点费率和费用输入。
    @field_validator("commission_rate", "minimum_commission", "stamp_duty_rate", mode="before")
    @classmethod
    def _reject_fee_float(cls, value: object) -> object:
        return _reject_float(value, "fee")

    # 检查费率与最低佣金非负，并要求 ETF 印花税配置为零。
    @model_validator(mode="after")
    def _validate_fee(self) -> Self:
        _finite_non_negative(self.commission_rate, "commission_rate")
        _finite_non_negative(self.minimum_commission, "minimum_commission")
        if self.stamp_duty_rate != 0:
            raise ValueError("ETF stamp duty is fixed to zero")
        return self


class SlippageConfig(_FrozenConfig):
    """区分买卖方向、按收盘价比例计算的滑点。"""

    rate: Decimal = Decimal("0.0005")

    # 在类型转换前拒绝浮点滑点比例输入。
    @field_validator("rate", mode="before")
    @classmethod
    def _reject_rate_float(cls, value: object) -> object:
        return _reject_float(value, "slippage rate")

    # 校验滑点比例为有限非负值。
    @model_validator(mode="after")
    def _validate_rate(self) -> Self:
        _finite_non_negative(self.rate, "rate")
        if self.rate >= 1:
            raise ValueError("slippage rate must be less than one")
        return self


class BacktestConfig(_FrozenConfig):
    """完整的 0.2 版运行配置。"""

    config_version: Literal["0.2.0"] = "0.2.0"
    start_date: date
    end_date: date
    initial_cash: Decimal = Decimal("1000000")
    database: DatabaseConfig
    data_snapshot: DataSnapshotConfig = Field(default_factory=DataSnapshotConfig)
    universe: UniverseConfig
    strategy: StrategyConfig
    fee: FeeConfig = Field(default_factory=FeeConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    volume_participation_rate: Decimal = Decimal("0.20")
    runs_dir: Path = Path("runs")
    limit_rules_csv: Path = Path("resources/limit_rules/etf_price_limit_20pct.csv")
    limit_rules_manifest: Path = Path("resources/limit_rules/manifest.json")
    calendar_policy: Literal["SSE_FOR_ALL"] = CALENDAR_POLICY

    # 在类型转换前检查初始资金与成交量参与比例的精度输入。
    @field_validator("initial_cash", "volume_participation_rate", mode="before")
    @classmethod
    def _reject_money_float(cls, value: object) -> object:
        return _reject_float(value, "backtest Decimal")

    # 规范项目相对路径，拒绝绝对路径和向上越界的路径。
    @field_validator("runs_dir", "limit_rules_csv", "limit_rules_manifest")
    @classmethod
    def _relative_paths(cls, value: Path) -> Path:
        normalized = Path(str(value).replace("\\", "/"))
        if normalized.is_absolute() or normalized.drive or ".." in normalized.parts:
            raise ValueError("project paths must be relative and stay inside the project")
        return normalized

    # 检查回测日期、初始资金、成交量参与比例；模型回测区间必须等于测试区间。
    @model_validator(mode="after")
    def _validate_run(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not Decimal("0") < self.volume_participation_rate <= Decimal("1"):
            raise ValueError("volume_participation_rate must be in (0, 1]")
        if isinstance(self.strategy, ModelStrategyConfig) and (
            self.start_date != self.strategy.test_start or self.end_date != self.strategy.test_end
        ):
            raise ValueError("model backtest dates must equal the configured test interval")
        return self

    def resolved_dict(self) -> dict[str, object]:
        """返回可安全转为 JSON 且明确不含密码的表示。"""

        payload = self.model_dump(mode="json", exclude={"database": {"password"}})
        database = dict(payload["database"])
        database["password"] = "***"
        payload["database"] = database
        payload["fixed_frequency"] = "1d"
        payload["fixed_execution_price"] = "CLOSE"
        payload["signal_execution_lag"] = "NEXT_SSE_TRADING_DAY_CLOSE"
        payload["pit_compliant"] = False
        payload["rule_mode"] = RULE_MODE
        return payload

    def to_resolved_json(self) -> str:
        """以确定性格式序列化已解析配置。"""

        return json.dumps(
            self.resolved_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


__all__ = [
    "CALENDAR_POLICY",
    "DATA_MODE",
    "MARKET_TIMEZONE",
    "PACKAGE_MANIFEST_SHA256",
    "RULE_MODE",
    "BacktestConfig",
    "DataSnapshotConfig",
    "DatabaseConfig",
    "FeeConfig",
    "ModelStrategyConfig",
    "RuleStrategyConfig",
    "SlippageConfig",
    "StrategyConfig",
    "UniverseConfig",
    "etf_code",
    "normalize_index_code",
    "normalize_symbol",
]
