"""与具体框架无关的日频监督模型工作流契约。

执行引擎有意只认识 :class:`BaseStrategy`。本模块定义可插拔模型所需的研究侧边界，
同时不引入对数值计算或深度学习框架的依赖。
"""

from __future__ import annotations


import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from etf_backtest.validation import plain_date as _plain_date
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import MarketBarView

if TYPE_CHECKING:
    from etf_backtest.strategy.portfolio import ModelPortfolioPolicy

DAILY_FORWARD_RETURN_LABEL = "front_close[D+2]/front_close[D+1]-1"
MODEL_BUNDLE_SCHEMA_VERSION = "DAILY_MODEL_BUNDLE_V2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


# 规范模型标识等非空文本字段。
def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


# 检查特征、标签等 Decimal 数值有限，避免 NaN 或无穷参与训练。
def _finite_decimal(value: object, field_name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def canonical_json(value: Mapping[str, object]) -> str:
    """返回模型或训练参数的稳定 JSON 身份。"""

    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata must be finite and JSON-serializable") from exc


# 要求特征名序列非空、名称非空且无重复，并保留特征顺序。
def validate_feature_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("feature_names must be a sequence")
    names = tuple(_non_blank(value, "feature_name") for value in values)
    if not names:
        raise ValueError("feature_names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique")
    return names


# 对特征名称顺序与标签名称生成 SHA-256 指纹，识别训练／推理模式是否一致。
def feature_fingerprint(feature_names: Sequence[str], label_name: str) -> str:
    names = validate_feature_names(feature_names)
    label = _non_blank(label_name, "label_name")
    encoded = json.dumps(
        {"feature_names": names, "label_name": label},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DateRange:
    """单个时间顺序切分使用的信号日期闭区间。"""

    start_date: date
    end_date: date

    # 检查日期区间端点是 date 且起点不晚于终点。
    def __post_init__(self) -> None:
        start = _plain_date(self.start_date, "start_date")
        end = _plain_date(self.end_date, "end_date")
        if end < start:
            raise ValueError("end_date must not precede start_date")

    # 判断信号日期是否落在包含端点的区间内。
    def contains(self, value: date) -> bool:
        return self.start_date <= value <= self.end_date

    # 把日期区间转换为可记录的字典。
    def to_dict(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    # 从字典还原并校验日期区间。
    @classmethod
    def from_mapping(cls, value: Mapping[str, object], field_name: str) -> DateRange:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        try:
            start = date.fromisoformat(_non_blank(value["start_date"], f"{field_name}.start_date"))
            end = date.fromisoformat(_non_blank(value["end_date"], f"{field_name}.end_date"))
        except KeyError as exc:
            raise ValueError(f"{field_name} is incomplete") from exc
        return cls(start_date=start, end_date=end)


@dataclass(frozen=True, slots=True, order=True)
class SampleKey:
    """与信号日 ``D`` 对齐的模型记录稳定身份。"""

    signal_date: date
    symbol: str

    # 校验样本主键的证券和信号日期，供标签与预测精确对齐。
    def __post_init__(self) -> None:
        _plain_date(self.signal_date, "signal_date")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """单个样本键对应的框架无关 Decimal 特征向量。"""

    key: SampleKey
    features: tuple[Decimal, ...]

    # 校验并冻结一个样本的有限特征向量。
    def __post_init__(self) -> None:
        if not isinstance(self.key, SampleKey):
            raise TypeError("key must be SampleKey")
        values = tuple(self.features)
        if not values:
            raise ValueError("features must not be empty")
        for index, value in enumerate(values):
            _finite_decimal(value, f"features[{index}]")
        object.__setattr__(self, "features", values)


@dataclass(frozen=True, slots=True)
class LabeledRecord(FeatureRecord):
    """特征向量及 D+1 收盘到 D+2 收盘的收益标签。"""

    label: Decimal

    # 校验有标签样本的主键、特征和未来收益标签。
    def __post_init__(self) -> None:
        super(LabeledRecord, self).__post_init__()
        _finite_decimal(self.label, "label")


# 校验一组有标签记录的唯一性与特征宽度，并整理为稳定序列。
def _freeze_labeled_records(
    records: Sequence[LabeledRecord],
    *,
    field_name: str,
    feature_count: int,
    date_range: DateRange,
) -> tuple[LabeledRecord, ...]:
    supplied_records = cast(object, records)
    if isinstance(supplied_records, (str, bytes)) or not isinstance(supplied_records, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    values = tuple(records)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(record, LabeledRecord) for record in values):
        raise TypeError(f"{field_name} may contain only LabeledRecord values")
    ordered = tuple(sorted(values, key=lambda record: record.key))
    keys = tuple(record.key for record in ordered)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} contains duplicate sample keys")
    for record in ordered:
        if len(record.features) != feature_count:
            raise ValueError(f"{field_name} has a feature-width mismatch")
        if not date_range.contains(record.key.signal_date):
            raise ValueError(f"{field_name} contains a signal outside its date range")
    return ordered


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Qlib 风格的不可变 train/valid/test 数据集契约。"""

    feature_names: tuple[str, ...]
    label_name: str
    train_range: DateRange
    valid_range: DateRange
    test_range: DateRange
    train: tuple[LabeledRecord, ...]
    valid: tuple[LabeledRecord, ...]
    test: tuple[LabeledRecord, ...]

    # 检查训练／验证／测试区间、非空样本及模式一致性，冻结数据切分。
    def __post_init__(self) -> None:
        names = validate_feature_names(self.feature_names)
        label = _non_blank(self.label_name, "label_name")
        for field_name in ("train_range", "valid_range", "test_range"):
            if not isinstance(getattr(self, field_name), DateRange):
                raise TypeError(f"{field_name} must be DateRange")
        if not self.train_range.end_date < self.valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not self.valid_range.end_date < self.test_range.start_date:
            raise ValueError("valid_range must precede test_range")
        train = _freeze_labeled_records(
            self.train,
            field_name="train",
            feature_count=len(names),
            date_range=self.train_range,
        )
        valid = _freeze_labeled_records(
            self.valid,
            field_name="valid",
            feature_count=len(names),
            date_range=self.valid_range,
        )
        test = _freeze_labeled_records(
            self.test,
            field_name="test",
            feature_count=len(names),
            date_range=self.test_range,
        )
        keys = tuple(record.key for split in (train, valid, test) for record in split)
        if len(keys) != len(set(keys)):
            raise ValueError("dataset splits contain overlapping sample keys")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "label_name", label)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "test", test)

    # 返回训练样本中最大的信号日期；它表示信号截止日，不是标签收益实现日。
    @property
    def trained_through(self) -> date:
        return max(record.key.signal_date for record in self.train)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """与样本键对齐的单个有限模型分数。"""

    key: SampleKey
    score: float

    # 检查预测主键与得分有效，供逐日组合分配和离线评估。
    def __post_init__(self) -> None:
        if not isinstance(self.key, SampleKey):
            raise TypeError("key must be SampleKey")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be numeric")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class RegressionMetricReport:
    """针对一次精确预测对齐计算的有限回归指标。"""

    sample_count: int
    mean_squared_error: float
    mean_absolute_error: float
    prediction_correlation: float

    # 检查回归评估中的样本数及误差、相关性指标数值。
    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        for field_name in (
            "mean_squared_error",
            "mean_absolute_error",
            "prediction_correlation",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite float")


@dataclass(frozen=True, slots=True)
class ModelDataIdentity:
    """加载已保存模型 bundle 时必须匹配的冻结数据身份。"""

    dataset_version: str
    manifest_sha256: str
    snapshot_started_at_utc: datetime

    # 校验模型使用的数据集版本、清单摘要与快照时间身份。
    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_version", _non_blank(self.dataset_version, "dataset_version")
        )
        manifest = _non_blank(self.manifest_sha256, "manifest_sha256").lower()
        if _SHA256.fullmatch(manifest) is None:
            raise ValueError("manifest_sha256 must contain 64 lowercase hex characters")
        snapshot_started = self.snapshot_started_at_utc
        if not isinstance(snapshot_started, datetime):
            raise TypeError("snapshot_started_at_utc must be datetime")
        if snapshot_started.tzinfo is None or snapshot_started.utcoffset() is None:
            raise ValueError("snapshot_started_at_utc must be timezone-aware")
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(self, "snapshot_started_at_utc", snapshot_started.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """嵌入每个 bundle 的自描述兼容性身份。"""

    schema_version: str
    model_id: str
    model_class_name: str
    model_parameters_json: str
    training_parameters_json: str
    feature_names: tuple[str, ...]
    feature_fingerprint: str
    label_name: str
    train_range: DateRange
    valid_range: DateRange
    test_range: DateRange
    random_seed: int
    data_identity: ModelDataIdentity
    trained_through: date
    framework_name: str
    framework_version: str

    # 校验模型元数据版本、特征指纹、参数 JSON、日期切分和训练截止日等一致性。
    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported model bundle schema_version")
        for field_name in (
            "model_id",
            "model_class_name",
            "model_parameters_json",
            "training_parameters_json",
            "label_name",
            "framework_name",
            "framework_version",
        ):
            object.__setattr__(self, field_name, _non_blank(getattr(self, field_name), field_name))
        names = validate_feature_names(self.feature_names)
        fingerprint = _non_blank(self.feature_fingerprint, "feature_fingerprint").lower()
        if _SHA256.fullmatch(fingerprint) is None:
            raise ValueError("feature_fingerprint must be a SHA-256 hex string")
        if fingerprint != feature_fingerprint(names, self.label_name):
            raise ValueError("feature_fingerprint does not match feature/label schema")
        for json_field in ("model_parameters_json", "training_parameters_json"):
            try:
                parsed = json.loads(getattr(self, json_field))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{json_field} must contain valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{json_field} must encode a JSON object")
            if canonical_json(parsed) != getattr(self, json_field):
                raise ValueError(f"{json_field} must use canonical JSON encoding")
        for range_name in ("train_range", "valid_range", "test_range"):
            if not isinstance(getattr(self, range_name), DateRange):
                raise TypeError(f"{range_name} must be DateRange")
        if not self.train_range.end_date < self.valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not self.valid_range.end_date < self.test_range.start_date:
            raise ValueError("valid_range must precede test_range")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not isinstance(self.data_identity, ModelDataIdentity):
            raise TypeError("data_identity must be ModelDataIdentity")
        trained = _plain_date(self.trained_through, "trained_through")
        if not self.train_range.contains(trained):
            raise ValueError("trained_through must be within train_range")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_fingerprint", fingerprint)

    # 把模型身份、特征模式和日期切分转换为可保存字典。
    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_class_name": self.model_class_name,
            "model_parameters_json": self.model_parameters_json,
            "training_parameters_json": self.training_parameters_json,
            "feature_names": self.feature_names,
            "feature_fingerprint": self.feature_fingerprint,
            "label_name": self.label_name,
            "train_range": self.train_range.to_dict(),
            "valid_range": self.valid_range.to_dict(),
            "test_range": self.test_range.to_dict(),
            "random_seed": self.random_seed,
            "data_identity": {
                "dataset_version": self.data_identity.dataset_version,
                "manifest_sha256": self.data_identity.manifest_sha256,
                "snapshot_started_at_utc": (self.data_identity.snapshot_started_at_utc.isoformat()),
            },
            "trained_through": self.trained_through.isoformat(),
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
        }

    # 从产物字典恢复模型元数据并触发完整校验。
    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelMetadata:
        if not isinstance(value, Mapping):
            raise TypeError("model metadata must be a mapping")
        try:
            raw_names = value["feature_names"]
            raw_identity = value["data_identity"]
            if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
                raise TypeError("feature_names must be a sequence")
            if any(not isinstance(item, str) for item in raw_names):
                raise TypeError("feature_names may contain only strings")
            if not isinstance(raw_identity, Mapping):
                raise TypeError("data_identity must be a mapping")
            snapshot_started = datetime.fromisoformat(
                _non_blank(
                    raw_identity["snapshot_started_at_utc"],
                    "snapshot_started_at_utc",
                )
            )
            return cls(
                schema_version=_non_blank(value["schema_version"], "schema_version"),
                model_id=_non_blank(value["model_id"], "model_id"),
                model_class_name=_non_blank(value["model_class_name"], "model_class_name"),
                model_parameters_json=_non_blank(
                    value["model_parameters_json"], "model_parameters_json"
                ),
                training_parameters_json=_non_blank(
                    value["training_parameters_json"], "training_parameters_json"
                ),
                feature_names=tuple(raw_names),
                feature_fingerprint=_non_blank(value["feature_fingerprint"], "feature_fingerprint"),
                label_name=_non_blank(value["label_name"], "label_name"),
                train_range=DateRange.from_mapping(
                    _mapping_value(value["train_range"], "train_range"), "train_range"
                ),
                valid_range=DateRange.from_mapping(
                    _mapping_value(value["valid_range"], "valid_range"), "valid_range"
                ),
                test_range=DateRange.from_mapping(
                    _mapping_value(value["test_range"], "test_range"), "test_range"
                ),
                random_seed=_integer_value(value["random_seed"], "random_seed"),
                data_identity=ModelDataIdentity(
                    dataset_version=_non_blank(raw_identity["dataset_version"], "dataset_version"),
                    manifest_sha256=_non_blank(raw_identity["manifest_sha256"], "manifest_sha256"),
                    snapshot_started_at_utc=snapshot_started,
                ),
                trained_through=date.fromisoformat(
                    _non_blank(value["trained_through"], "trained_through")
                ),
                framework_name=_non_blank(value["framework_name"], "framework_name"),
                framework_version=_non_blank(value["framework_version"], "framework_version"),
            )
        except KeyError as exc:
            raise ValueError("model metadata is incomplete") from exc


# 读取并检查元数据中的映射字段。
def _mapping_value(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


# 读取并检查元数据中的严格整数字段。
def _integer_value(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    return value


@runtime_checkable
class FeatureBuilder(Protocol):
    """用户可插拔、无标签且只使用截至 D 日可见数据的特征计算。"""

    # 声明特征向量对应的有序名称，训练和推理必须一致。
    @property
    def feature_names(self) -> tuple[str, ...]: ...

    # 声明构建一个信号样本需要的历史交易日窗口长度。
    @property
    def required_history_trading_days(self) -> int: ...

    # 依据单证券截至信号日的历史生成特征序列；历史不足等情况下可返回 None 跳过样本。
    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> Sequence[Decimal] | None: ...


@runtime_checkable
class ModelSpec(Protocol):
    """与具体学习框架无关的模型身份与参数。"""

    # 提供稳定模型标识，供产物元数据核对。
    @property
    def model_id(self) -> str: ...

    # 提供模型类名称标识，供训练与加载一致性检查。
    @property
    def model_class_name(self) -> str: ...

    # 提供模型构造／训练算法参数的可序列化映射。
    @property
    def model_parameters(self) -> Mapping[str, object]: ...


@runtime_checkable
class TorchModelFactory(ModelSpec, Protocol):
    """延迟依赖具体框架的单个 torch.nn.Module 架构工厂。"""

    # 按输入特征宽度创建 Torch 网络，由用户模型工厂实现。
    def create(self, *, input_dim: int, seed: int) -> object: ...


@runtime_checkable
class PredictorBundle(Protocol):
    """通用日频模型策略使用的纯推理 bundle。"""

    # 提供已训练模型的身份、日期与特征模式元数据。
    @property
    def metadata(self) -> ModelMetadata: ...

    # 对带主键的特征样本批量预测，返回相同主键对应的得分。
    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]: ...


@runtime_checkable
class ModelWorkflowResult(Protocol):
    """不同学习后端共用的拟合结果边界。"""

    # 提供训练完成后的可推理模型包。
    @property
    def bundle(self) -> PredictorBundle: ...

    # 提供验证集回归评估结果。
    @property
    def validation_metrics(self) -> RegressionMetricReport: ...

    # 提供测试集回归评估结果。
    @property
    def test_metrics(self) -> RegressionMetricReport: ...

    # 提供最佳轮次、损失等后端训练摘要。
    @property
    def fit_summary(self) -> Mapping[str, object]: ...


@runtime_checkable
class ModelWorkflow(Protocol):
    """回测编排器使用的通用训练与 bundle 持久化接口。"""

    # 提供工作流当前已训练或已加载的预测包。
    @property
    def bundle(self) -> PredictorBundle | None: ...

    # 声明该后端保存模型产物时使用的文件名。
    @property
    def bundle_filename(self) -> str: ...

    # 使用既定数据切分训练模型，并返回预测包和评估结果。
    def fit(self, dataset: DatasetSplits) -> ModelWorkflowResult: ...

    # 把已训练模型与复现所需元数据保存到指定产物路径。
    def save(self, path: Path, *, source_run_dir: Path | None = None) -> Path: ...


# 检查特征构建器满足接口、名称模式和回看长度约束。
def validate_feature_builder(builder: FeatureBuilder) -> tuple[tuple[str, ...], int]:
    if not isinstance(builder, FeatureBuilder):
        raise TypeError("feature_builder must satisfy FeatureBuilder")
    names = validate_feature_names(builder.feature_names)
    lookback = builder.required_history_trading_days
    if type(lookback) is not int or lookback <= 0:
        raise ValueError("required_history_trading_days must be a positive integer")
    return names, lookback


def build_feature_record(
    *,
    builder: FeatureBuilder,
    symbol: str,
    signal_date: date,
    history: Sequence[MarketBarView],
) -> FeatureRecord | None:
    """校验单次插件调用，并在不含未来视图的前提下冻结结果。"""

    names, lookback = validate_feature_builder(builder)
    canonical = normalize_symbol(symbol)
    signal = _plain_date(signal_date, "signal_date")
    supplied_history = cast(object, history)
    if isinstance(supplied_history, (str, bytes)) or not isinstance(supplied_history, Sequence):
        raise TypeError("history must be a sequence")
    ordered = tuple(sorted(history, key=lambda view: view.trade_date))
    if any(not isinstance(view, MarketBarView) for view in ordered):
        raise TypeError("history may contain only MarketBarView values")
    if not ordered or ordered[-1].trade_date != signal:
        return None
    if any(view.symbol != canonical for view in ordered):
        raise ValueError("history contains a different symbol")
    if any(view.trade_date > signal for view in ordered):
        raise ValueError("history contains a future view")
    dates = tuple(view.trade_date for view in ordered)
    if len(dates) != len(set(dates)):
        raise ValueError("history contains duplicate daily views")
    visible = ordered[-lookback:]
    values = builder.build_features(symbol=canonical, signal_date=signal, history=visible)
    if values is None:
        return None
    supplied_values = cast(object, values)
    if isinstance(supplied_values, (str, bytes)) or not isinstance(supplied_values, Sequence):
        raise TypeError("FeatureBuilder must return a sequence or None")
    features = tuple(values)
    if len(features) != len(names):
        raise ValueError("FeatureBuilder returned the wrong feature width")
    return FeatureRecord(key=SampleKey(signal_date=signal, symbol=canonical), features=features)


def feature_records_for_signal(
    *,
    builder: FeatureBuilder,
    market_views: Sequence[MarketBarView],
    signal_date: date,
) -> tuple[FeatureRecord, ...]:
    """为截至信号日 D 可见的每只证券构建无标签记录。"""

    signal = _plain_date(signal_date, "signal_date")
    supplied_views = cast(object, market_views)
    if isinstance(supplied_views, (str, bytes)) or not isinstance(supplied_views, Sequence):
        raise TypeError("market_views must be a sequence")
    by_symbol: dict[str, dict[date, MarketBarView]] = {}
    for view in market_views:
        if not isinstance(view, MarketBarView):
            raise TypeError("market_views may contain only MarketBarView values")
        if view.trade_date > signal:
            raise ValueError("market_views contains data after signal_date")
        rows = by_symbol.setdefault(view.symbol, {})
        if view.trade_date in rows:
            raise ValueError(f"duplicate daily view for {view.symbol} on {view.trade_date}")
        rows[view.trade_date] = view
    records: list[FeatureRecord] = []
    for symbol in sorted(by_symbol):
        history = tuple(by_symbol[symbol][day] for day in sorted(by_symbol[symbol]))
        record = build_feature_record(
            builder=builder,
            symbol=symbol,
            signal_date=signal,
            history=history,
        )
        if record is not None:
            records.append(record)
    return tuple(records)


# 将预测记录投影为可导出的行数据，保留样本日期与证券。
def prediction_rows(
    predictions: Sequence[PredictionRecord],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    seen: set[SampleKey] = set()
    for prediction in sorted(predictions, key=lambda value: value.key):
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predictions may contain only PredictionRecord values")
        if prediction.key in seen:
            raise ValueError("predictions contain duplicate keys")
        seen.add(prediction.key)
        rows.append(
            MappingProxyType(
                {
                    "signal_date": prediction.key.signal_date.isoformat(),
                    "symbol": prediction.key.symbol,
                    "score": prediction.score,
                }
            )
        )
    return tuple(rows)


def build_model_metadata(
    *, model: ModelSpec, training_parameters: Mapping[str, object],
    feature_names: tuple[str, ...], dataset: DatasetSplits,
    random_seed: int, data_identity: ModelDataIdentity,
    framework_name: str, framework_version: str,
) -> ModelMetadata:
    """两个训练后端使用同一份特征、数据区间和模型身份说明。"""
    return ModelMetadata(
        schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
        model_id=model.model_id,
        model_class_name=model.model_class_name,
        model_parameters_json=canonical_json(model.model_parameters),
        training_parameters_json=canonical_json(training_parameters),
        feature_names=feature_names,
        feature_fingerprint=feature_fingerprint(feature_names, DAILY_FORWARD_RETURN_LABEL),
        label_name=DAILY_FORWARD_RETURN_LABEL,
        train_range=dataset.train_range, valid_range=dataset.valid_range,
        test_range=dataset.test_range, random_seed=random_seed,
        data_identity=data_identity, trained_through=dataset.trained_through,
        framework_name=framework_name, framework_version=framework_version,
    )


def build_inference_payload(
    *, metadata: ModelMetadata, feature_names: tuple[str, ...],
    required_history: int, portfolio: ModelPortfolioPolicy, source_run_dir: Path,
) -> dict[str, object]:
    """保存格式保持原样；XGBoost 写入 JSON 时自然将 tuple 转成 list。"""
    return {
        "input_dim": len(feature_names),
        "feature_order": feature_names,
        "required_history_trading_days": required_history,
        "portfolio_json": canonical_json(portfolio.resolved_dict()),
        "source_run_dir": str(source_run_dir),
        "factor_schema": {
            "feature_names": feature_names,
            "label_name": metadata.label_name,
            "feature_fingerprint": metadata.feature_fingerprint,
        },
    }


# 要求数据集特征名与构建器相同，并使用项目规定的日频未来收益标签。
def validate_dataset_schema(dataset: DatasetSplits, feature_names: tuple[str, ...]) -> None:
    if not isinstance(dataset, DatasetSplits):
        raise TypeError("dataset must be DatasetSplits")
    if dataset.feature_names != feature_names:
        raise ValueError("dataset feature_names do not match FeatureBuilder")
    if dataset.label_name != DAILY_FORWARD_RETURN_LABEL:
        raise ValueError("dataset label does not match the daily forward-return contract")


# 逐项核对产物元数据与期望值；提供信号日时要求训练信号截止日严格更早。
def validate_model_metadata(
    metadata: ModelMetadata,
    expected: Mapping[str, object],
    error_type: type[ValueError],
    *, signal_date: date | None = None,
) -> None:
    for field_name, expected_value in expected.items():
        if getattr(metadata, field_name) != expected_value:
            raise error_type(f"bundle metadata mismatch for {field_name}")
    if signal_date is not None and metadata.trained_through >= signal_date:
        raise error_type("bundle trained_through must precede signal_date")


def validate_inference_payload(
    inference: Mapping[str, object],
    *, feature_names: tuple[str, ...], required_history: int,
    portfolio: ModelPortfolioPolicy, metadata: ModelMetadata,
    error_type: type[ValueError], json_schema: bool = False,
) -> tuple[str, str]:
    """共用推理检查；Torch 特征列表存为 tuple，XGBoost JSON 中存为 list。"""
    input_dim = inference.get("input_dim")
    if type(input_dim) is not int:
        raise error_type("input_dim must be int")
    if input_dim != len(feature_names):
        raise error_type("bundle input_dim does not match feature count")
    raw_order = inference.get("feature_order")
    if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, Sequence):
        raise error_type("bundle feature_order must be a sequence")
    if tuple(raw_order) != feature_names:
        raise error_type("bundle feature_order does not match Model source")
    history = inference.get("required_history_trading_days")
    if type(history) is not int:
        raise error_type("required_history_trading_days must be int")
    if history != required_history:
        raise error_type("bundle history requirement does not match Model source")
    portfolio_json = canonical_json(portfolio.resolved_dict())
    if inference.get("portfolio_json") != portfolio_json:
        raise error_type("bundle portfolio does not match Model source")
    source_run_dir = inference.get("source_run_dir")
    if not isinstance(source_run_dir, str) or not source_run_dir.strip():
        raise error_type("bundle source_run_dir is missing")
    schema = inference.get("factor_schema")
    if not isinstance(schema, Mapping):
        raise error_type("factor_schema must be a mapping")
    expected_schema = {
        "feature_names": list(feature_names) if json_schema else feature_names,
        "label_name": metadata.label_name,
        "feature_fingerprint": metadata.feature_fingerprint,
    }
    if dict(schema) != expected_schema:
        raise error_type("bundle factor schema does not match metadata")
    return portfolio_json, source_run_dir


__all__ = [
    "DAILY_FORWARD_RETURN_LABEL",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "DatasetSplits",
    "DateRange",
    "FeatureBuilder",
    "FeatureRecord",
    "LabeledRecord",
    "ModelDataIdentity",
    "ModelMetadata",
    "ModelSpec",
    "ModelWorkflow",
    "ModelWorkflowResult",
    "PredictionRecord",
    "PredictorBundle",
    "RegressionMetricReport",
    "SampleKey",
    "TorchModelFactory",
    "build_feature_record",
    "build_model_metadata",
    "build_inference_payload",
    "canonical_json",
    "feature_fingerprint",
    "feature_records_for_signal",
    "prediction_rows",
    "validate_feature_builder",
    "validate_feature_names",
    "validate_dataset_schema",
    "validate_model_metadata",
    "validate_inference_payload",
]
