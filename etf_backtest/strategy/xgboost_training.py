"""Deterministic XGBoost training and immutable inference bundles."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from uuid import uuid4

import numpy as np

from etf_backtest.file_utils import sha256_file as xgboost_bundle_sha256

from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    FeatureBuilder,
    FeatureRecord,
    ModelDataIdentity,
    ModelMetadata,
    ModelSpec,
    PredictionRecord,
    RegressionMetricReport,
    canonical_json,
    feature_fingerprint,
    validate_feature_builder,
    validate_dataset_schema,
    build_model_metadata,
    build_inference_payload,
    validate_model_metadata,
    validate_inference_payload,
)
from etf_backtest.strategy.model_data import _feature_matrix, _target_vector, evaluate_predictions
from etf_backtest.strategy.portfolio import ModelPortfolioPolicy, TopKPortfolio
from etf_backtest.strategy.model_device import normalize_model_device

XGBOOST_BUNDLE_FORMAT: Final = "ETF_DAILY_XGBOOST_UBJ_V1"
XGBOOST_BUNDLE_FORMAT_VERSION: Final = 1

_ATTR_FORMAT: Final = "etf_backtest_bundle_format"
_ATTR_FORMAT_VERSION: Final = "etf_backtest_bundle_format_version"
_ATTR_METADATA: Final = "etf_backtest_metadata"
_ATTR_INFERENCE: Final = "etf_backtest_inference"
_ATTR_FIT_SUMMARY: Final = "etf_backtest_fit_summary"

_RESERVED_MODEL_PARAMETERS: Final = frozenset(
    {
        "booster",
        "device",
        "eval_metric",
        "n_jobs",
        "nthread",
        "objective",
        "random_state",
        "seed",
        "tree_method",
    }
)
_FRAMEWORK_PARAMETERS: Final = MappingProxyType(
    {
        "booster": "gbtree",
        "device": "cpu",
        "eval_metric": "rmse",
        "nthread": 1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
)


# 表示当前环境缺少可选 XGBoost 依赖，只有选择该后端时才需要它。
class XGBoostUnavailableError(ImportError):
    """The optional XGBoost runtime is not installed."""


# 表示保存的 XGBoost 产物与当前策略模式或参数不兼容。
class XGBoostBundleCompatibilityError(ValueError):
    """A saved XGBoost bundle is incompatible with its strategy source."""


# 按需导入 XGBoost，使规则策略或 Torch 路径不强制依赖该后端。
def require_xgboost() -> Any:
    """Import XGBoost only for an explicitly selected XGBoost backend."""

    try:
        return importlib.import_module("xgboost")
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            raise XGBoostUnavailableError(
                "XGBoost is required for the daily xgboost model workflow; "
                'install it with python -m pip install -e ".[xgboost]"'
            ) from exc
        raise


# 声明每个策略的训练设备、轮数、验证早停和随机种子设置。
@dataclass(frozen=True, slots=True)
class XGBoostTrainingConfig:
    """Framework-owned training device and validation early-stopping settings."""

    seed: int = 42
    num_boost_round: int = 500
    early_stopping_rounds: int = 30
    min_delta: float = 0.0
    device: str = "cpu"

    # 校验 XGBoost 最大轮数、早停轮数与随机种子等训练设置。
    def __post_init__(self) -> None:
        object.__setattr__(self, "device", normalize_model_device(self.device))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.num_boost_round) is not int or self.num_boost_round <= 0:
            raise ValueError("num_boost_round must be a positive integer")
        if type(self.early_stopping_rounds) is not int or self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, int | float):
            raise TypeError("min_delta must be numeric")
        if not math.isfinite(float(self.min_delta)) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")

    # 导出 XGBoost 训练控制参数，供模型元数据记录。
    def to_parameters(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_delta": float(self.min_delta),
            **dict(_FRAMEWORK_PARAMETERS),
            "device": self.device,
            "early_stopping_metric": "valid/rmse",
            "early_stopping_save_best": True,
        }


# 汇总 XGBoost 模型包、验证／测试预测与指标以及训练摘要。
@dataclass(frozen=True, slots=True)
class DailyXGBoostWorkflowResult:
    bundle: DailyXGBoostBundle
    validation_metrics: RegressionMetricReport
    test_metrics: RegressionMetricReport
    validation_predictions: tuple[PredictionRecord, ...]
    test_predictions: tuple[PredictionRecord, ...]
    fit_summary: Mapping[str, object]

    # 检查 XGBoost 结果包和指标类型，并校验、冻结训练摘要。
    def __post_init__(self) -> None:
        if not isinstance(self.bundle, DailyXGBoostBundle):
            raise TypeError("bundle must be DailyXGBoostBundle")
        for field_name in ("validation_metrics", "test_metrics"):
            if not isinstance(getattr(self, field_name), RegressionMetricReport):
                raise TypeError(f"{field_name} must be RegressionMetricReport")
        summary = _validate_fit_summary(self.fit_summary)
        object.__setattr__(self, "fit_summary", MappingProxyType(summary))


# 保存加载后的 XGBoost 包、文件摘要、组合政策 JSON 与来源运行目录。
@dataclass(frozen=True, slots=True)
class LoadedXGBoostInferenceBundle:
    bundle: DailyXGBoostBundle
    file_sha256: str
    portfolio_json: str
    source_run_dir: str


# 将已校验 Booster、模式元数据和训练摘要包装为统一预测接口。
class DailyXGBoostBundle:
    """A validated Booster implementing the framework-neutral predictor contract."""

    __slots__ = ("_booster", "_fit_summary", "_metadata", "_device")

    # 绑定模型元数据、Booster 与训练摘要，并校验特征模式兼容。
    def __init__(
        self,
        *,
        booster: Any,
        metadata: ModelMetadata,
        fit_summary: Mapping[str, object],
        device: str = "cpu",
    ) -> None:
        if not isinstance(metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if metadata.framework_name != "xgboost":
            raise XGBoostBundleCompatibilityError("bundle framework_name must be xgboost")
        _validate_booster(booster, expected_features=metadata.feature_names)
        self._device = _configure_xgboost_device(booster, device)
        self._booster = booster
        self._metadata = metadata
        self._fit_summary = MappingProxyType(_validate_fit_summary(fit_summary))

    @property
    def device(self) -> str:
        """返回本次推理使用的设备，训练时的设备另存于元数据。"""
        return self._device

    # 返回 XGBoost 模型包元数据。
    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # 返回 XGBoost 最佳迭代和验证误差等训练摘要。
    @property
    def fit_summary(self) -> Mapping[str, object]:
        return self._fit_summary

    # 提供包内已训练的 Booster 对象。
    @property
    def booster(self) -> Any:
        return self._booster

    # 按固定特征名和顺序构建预测输入，调用 Booster 并返回带样本主键的得分。
    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]:
        ordered = tuple(records)
        if not ordered:
            return ()
        width = len(self._metadata.feature_names)
        if any(len(record.features) != width for record in ordered):
            raise ValueError("xgboost bundle feature width mismatch")
        matrix = np.asarray(
            [[float(value) for value in record.features] for record in ordered],
            dtype=np.float64,
        )
        if matrix.shape != (len(ordered), width) or not np.isfinite(matrix).all():
            raise ValueError("xgboost bundle input contains invalid values")
        xgboost = require_xgboost()
        data = xgboost.DMatrix(matrix, feature_names=list(self._metadata.feature_names))
        scores = np.asarray(self._booster.predict(data), dtype=np.float64).reshape(-1)
        if len(scores) != len(ordered) or not np.isfinite(scores).all():
            raise ValueError("xgboost bundle produced invalid scores")
        return tuple(
            PredictionRecord(key=record.key, score=float(score))
            for record, score in zip(ordered, scores, strict=True)
        )


# 在训练集拟合一次，通过验证集选取最佳模型，再保存一个 UBJ 产物供回测或模拟盘推理。
class DailyXGBoostWorkflow:
    """Fit once on train, select on validation, then persist one UBJSON bundle."""

    __slots__ = (
        "_bundle",
        "_data_identity",
        "_feature_builder",
        "_feature_names",
        "_model_spec",
        "_portfolio",
        "_required_history",
        "_training_config",
    )

    # 保存特征构建器、模型规格、训练控制项与数据身份，准备 XGBoost 工作流。
    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder,
        model_spec: ModelSpec,
        data_identity: ModelDataIdentity,
        training_config: XGBoostTrainingConfig | None = None,
        portfolio: ModelPortfolioPolicy | None = None,
    ) -> None:
        names, required_history = validate_feature_builder(feature_builder)
        _validate_model_spec(model_spec)
        validate_xgboost_model_parameters(model_spec.model_parameters)
        if not isinstance(data_identity, ModelDataIdentity):
            raise TypeError("data_identity must be ModelDataIdentity")
        config = training_config or XGBoostTrainingConfig()
        if not isinstance(config, XGBoostTrainingConfig):
            raise TypeError("training_config must be XGBoostTrainingConfig")
        selected_portfolio = portfolio or TopKPortfolio()
        if not isinstance(selected_portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        self._feature_builder = feature_builder
        self._feature_names = names
        self._required_history = required_history
        self._model_spec = model_spec
        self._data_identity = data_identity
        self._training_config = config
        self._portfolio = selected_portfolio
        self._bundle: DailyXGBoostBundle | None = None

    # 返回已训练或已加载的 XGBoost 模型包。
    @property
    def bundle(self) -> DailyXGBoostBundle | None:
        return self._bundle

    # 返回 XGBoost 默认 UBJ 产物文件名。
    @property
    def bundle_filename(self) -> str:
        return "model_bundle.ubj"

    # 在所选设备训练树模型，在验证集上早停并保留最佳模型，再计算验证／测试预测和指标。
    def fit(self, dataset: DatasetSplits) -> DailyXGBoostWorkflowResult:
        if self._bundle is not None:
            raise RuntimeError("DailyXGBoostWorkflow may fit or load only once")
        validate_dataset_schema(dataset, self._feature_names)
        xgboost = require_xgboost()
        train_matrix = _feature_matrix(dataset.train, len(self._feature_names))
        valid_matrix = _feature_matrix(dataset.valid, len(self._feature_names))
        train_targets = _target_vector(dataset.train)
        valid_targets = _target_vector(dataset.valid)
        train_data = xgboost.DMatrix(
            train_matrix,
            label=train_targets,
            feature_names=list(self._feature_names),
        )
        valid_data = xgboost.DMatrix(
            valid_matrix,
            label=valid_targets,
            feature_names=list(self._feature_names),
        )
        parameters = {
            **dict(_FRAMEWORK_PARAMETERS),
            **dict(self._model_spec.model_parameters),
            "seed": self._training_config.seed,
            "device": self._training_config.device,
        }
        if self._training_config.device != "cpu":
            # 正式训练前检查后端实际设备，阻止 XGBoost 自动回退 CPU。
            probe = xgboost.Booster(params={"nthread": 1}, cache=[train_data])
            _configure_xgboost_device(probe, self._training_config.device)
        evaluation_history: dict[str, dict[str, list[float]]] = {}
        callback = xgboost.callback.EarlyStopping(
            rounds=self._training_config.early_stopping_rounds,
            metric_name="rmse",
            data_name="valid",
            maximize=False,
            save_best=True,
            min_delta=float(self._training_config.min_delta),
        )
        booster = xgboost.train(
            params=parameters,
            dtrain=train_data,
            num_boost_round=self._training_config.num_boost_round,
            evals=[(train_data, "train"), (valid_data, "valid")],
            evals_result=evaluation_history,
            callbacks=[callback],
            verbose_eval=False,
        )
        valid_history = evaluation_history.get("valid", {}).get("rmse", [])
        if not valid_history:
            raise RuntimeError("xgboost training produced no validation history")
        best_iteration = getattr(booster, "best_iteration", None)
        best_score = getattr(booster, "best_score", None)
        if type(best_iteration) is not int or best_iteration < 0:
            raise RuntimeError("xgboost training produced no valid best_iteration")
        if best_score is None:
            raise RuntimeError("xgboost training produced no best_score")
        best_validation_rmse = float(best_score)
        if not math.isfinite(best_validation_rmse) or best_validation_rmse < 0:
            raise RuntimeError("xgboost training produced no finite best_score")
        fit_summary = {
            "best_iteration": best_iteration,
            "rounds_evaluated": len(valid_history),
            "best_validation_rmse": best_validation_rmse,
            "preprocessing": "NONE",
        }
        metadata = build_model_metadata(
            model=self._model_spec, training_parameters=self._training_config.to_parameters(),
            feature_names=self._feature_names, dataset=dataset,
            random_seed=self._training_config.seed, data_identity=self._data_identity,
            framework_name="xgboost", framework_version=str(xgboost.__version__),
        )
        bundle = DailyXGBoostBundle(
            booster=booster,
            metadata=metadata,
            fit_summary=fit_summary,
            device=self._training_config.device,
        )
        self._bundle = bundle
        validation_predictions = bundle.predict(dataset.valid)
        test_predictions = bundle.predict(dataset.test)
        return DailyXGBoostWorkflowResult(
            bundle=bundle,
            validation_metrics=evaluate_predictions(
                samples=dataset.valid,
                predictions=validation_predictions,
            ),
            test_metrics=evaluate_predictions(
                samples=dataset.test,
                predictions=test_predictions,
            ),
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            fit_summary=fit_summary,
        )

    # 把 Booster 与模型、训练和推理元数据保存到 UBJ 文件。
    def save(self, path: Path, *, source_run_dir: Path | None = None) -> Path:
        if self._bundle is None:
            raise RuntimeError("DailyXGBoostWorkflow has no bundle to save")
        target = Path(path)
        if target.suffix.casefold() != ".ubj":
            raise ValueError("xgboost bundle path must end with .ubj")
        target.parent.mkdir(parents=True, exist_ok=True)
        run_directory = Path(source_run_dir or target.parent).resolve()
        inference = build_inference_payload(
            metadata=self._bundle.metadata, feature_names=self._feature_names,
            required_history=self._required_history, portfolio=self._portfolio,
            source_run_dir=run_directory,
        )
        booster = self._bundle.booster
        booster.set_attr(
            **{
                _ATTR_FORMAT: XGBOOST_BUNDLE_FORMAT,
                _ATTR_FORMAT_VERSION: str(XGBOOST_BUNDLE_FORMAT_VERSION),
                _ATTR_METADATA: canonical_json(self._bundle.metadata.to_dict()),
                _ATTR_INFERENCE: canonical_json(inference),
                _ATTR_FIT_SUMMARY: canonical_json(self._bundle.fit_summary),
            }
        )
        temporary = target.with_name(f".{target.stem}.{uuid4().hex}.ubj")
        try:
            booster.save_model(temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    # 读取 UBJ 产物，校验工作流兼容性并恢复可推理包。
    def load(
        self, path: Path, *, dataset: DatasetSplits, device: str | None = None,
    ) -> DailyXGBoostBundle:
        if self._bundle is not None:
            raise RuntimeError("DailyXGBoostWorkflow may fit or load only once")
        validate_dataset_schema(dataset, self._feature_names)
        loaded = _load_bundle_file(
            Path(path), device=self._training_config.device if device is None else device,
        )
        metadata = loaded.bundle.metadata
        expected = {
            "model_id": self._model_spec.model_id,
            "model_class_name": self._model_spec.model_class_name,
            "model_parameters_json": canonical_json(self._model_spec.model_parameters),
            "training_parameters_json": canonical_json(self._training_config.to_parameters()),
            "feature_names": self._feature_names,
            "label_name": DAILY_FORWARD_RETURN_LABEL,
            "train_range": dataset.train_range,
            "valid_range": dataset.valid_range,
            "test_range": dataset.test_range,
            "random_seed": self._training_config.seed,
            "data_identity": self._data_identity,
            "trained_through": dataset.trained_through,
        }
        validate_model_metadata(metadata, expected, XGBoostBundleCompatibilityError)
        validate_inference_payload(
            loaded.inference,
            feature_names=self._feature_names,
            required_history=self._required_history,
            portfolio=self._portfolio,
            metadata=metadata,
            error_type=XGBoostBundleCompatibilityError, json_schema=True,
        )
        self._bundle = loaded.bundle
        return loaded.bundle


# 加载已有 XGBoost 产物，检查特征、模型、组合政策和来源身份；此入口不重新训练。
def load_daily_xgboost_bundle_for_inference(
    path: Path,
    *,
    feature_builder: FeatureBuilder,
    model_spec: ModelSpec,
    portfolio: ModelPortfolioPolicy,
    signal_date: date,
    expected_sha256: str | None = None,
    expected_model_id: str | None = None,
    device: str = "cpu",
) -> LoadedXGBoostInferenceBundle:
    """Load and validate one immutable XGBoost bundle without fitting."""

    source = Path(path)
    actual_sha256 = xgboost_bundle_sha256(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise XGBoostBundleCompatibilityError("model bundle SHA-256 does not match deployment")
    feature_names, required_history = validate_feature_builder(feature_builder)
    _validate_model_spec(model_spec)
    validate_xgboost_model_parameters(model_spec.model_parameters)
    if not isinstance(portfolio, ModelPortfolioPolicy):
        raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
    loaded = _load_bundle_file(source, device=device)
    metadata = loaded.bundle.metadata
    if expected_model_id is not None and metadata.model_id != expected_model_id:
        raise XGBoostBundleCompatibilityError("bundle model_id does not match deployment")
    expected = {
        "model_id": model_spec.model_id,
        "model_class_name": model_spec.model_class_name,
        "model_parameters_json": canonical_json(model_spec.model_parameters),
        "feature_names": feature_names,
        "feature_fingerprint": feature_fingerprint(feature_names, DAILY_FORWARD_RETURN_LABEL),
        "label_name": DAILY_FORWARD_RETURN_LABEL,
    }
    validate_model_metadata(
        metadata, expected, XGBoostBundleCompatibilityError, signal_date=signal_date,
    )
    expected_portfolio_json, source_run_dir = validate_inference_payload(
        loaded.inference,
        feature_names=feature_names,
        required_history=required_history,
        portfolio=portfolio,
        metadata=metadata,
        error_type=XGBoostBundleCompatibilityError, json_schema=True,
    )
    return LoadedXGBoostInferenceBundle(
        bundle=loaded.bundle,
        file_sha256=actual_sha256,
        portfolio_json=expected_portfolio_json,
        source_run_dir=source_run_dir,
    )


# 保存文件解析得到的预测包与额外推理元数据。
@dataclass(frozen=True, slots=True)
class _LoadedBundleFile:
    bundle: DailyXGBoostBundle
    inference: Mapping[str, object]


# 加载 UBJ Booster 及其 JSON 属性，校验后还原模型包和推理信息。
def _load_bundle_file(path: Path, *, device: str = "cpu") -> _LoadedBundleFile:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() != ".ubj":
        raise XGBoostBundleCompatibilityError("xgboost bundle path must end with .ubj")
    xgboost = require_xgboost()
    booster = xgboost.Booster(params={"device": "cpu"})
    try:
        booster.load_model(source)
    except Exception as exc:
        raise XGBoostBundleCompatibilityError(f"invalid xgboost bundle: {source}") from exc
    if booster.attr(_ATTR_FORMAT) != XGBOOST_BUNDLE_FORMAT:
        raise XGBoostBundleCompatibilityError("unsupported xgboost bundle format")
    if booster.attr(_ATTR_FORMAT_VERSION) != str(XGBOOST_BUNDLE_FORMAT_VERSION):
        raise XGBoostBundleCompatibilityError("unsupported xgboost bundle format version")
    metadata = ModelMetadata.from_mapping(_json_attribute(booster, _ATTR_METADATA))
    if metadata.framework_name != "xgboost":
        raise XGBoostBundleCompatibilityError("bundle framework_name must be xgboost")
    inference = _json_attribute(booster, _ATTR_INFERENCE)
    fit_summary = _json_attribute(booster, _ATTR_FIT_SUMMARY)
    bundle = DailyXGBoostBundle(
        booster=booster,
        metadata=metadata,
        fit_summary=fit_summary,
        device=device,
    )
    return _LoadedBundleFile(bundle=bundle, inference=MappingProxyType(dict(inference)))


def _configure_xgboost_device(booster: Any, value: str) -> str:
    """设置并核对 Booster 的实际设备，拒绝 CUDA 不可用或设备编号被替换。"""
    device = normalize_model_device(value)
    booster.set_param({"device": device})
    actual = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    matches = actual == device or (device == "cuda" and actual.startswith("cuda:"))
    if not matches:
        raise RuntimeError(
            f"XGBoost device {device!r} requested, but backend selected {actual!r}; "
            "check the GPU index, CUDA-enabled XGBoost build and driver, or use device='cpu'"
        )
    return device


# 从 Booster 属性中读取并解析指定 JSON 字段。
def _json_attribute(booster: Any, key: str) -> Mapping[str, object]:
    raw_value = booster.attr(key)
    if not isinstance(raw_value, str) or not raw_value:
        raise XGBoostBundleCompatibilityError(f"bundle attribute {key} is missing")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise XGBoostBundleCompatibilityError(f"bundle attribute {key} is invalid") from exc
    if not isinstance(value, dict):
        raise XGBoostBundleCompatibilityError(f"bundle attribute {key} must be an object")
    return value


# 检查模型规格接口、模型标识与参数映射可序列化。
def _validate_model_spec(model_spec: ModelSpec) -> None:
    if not isinstance(model_spec, ModelSpec):
        raise TypeError("model_spec must satisfy ModelSpec")
    for field_name in ("model_id", "model_class_name"):
        value = getattr(model_spec, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"model_spec.{field_name} must be nonblank")
    canonical_json(model_spec.model_parameters)


# 拒绝用户模型参数覆盖由框架固定管理的 XGBoost 参数。
def validate_xgboost_model_parameters(value: Mapping[str, object]) -> None:
    canonical_json(value)
    reserved = sorted(_RESERVED_MODEL_PARAMETERS.intersection(value))
    if reserved:
        raise ValueError(
            "xgboost model_parameters contain framework-owned keys: " + ", ".join(reserved)
        )


# 检查 Booster 可预测／保存，并要求特征宽度和名称顺序与元数据一致。
def _validate_booster(booster: Any, *, expected_features: tuple[str, ...]) -> None:
    if not callable(getattr(booster, "predict", None)) or not callable(
        getattr(booster, "save_model", None)
    ):
        raise TypeError("booster must be an xgboost Booster")
    try:
        width = int(booster.num_features())
    except Exception as exc:
        raise XGBoostBundleCompatibilityError("cannot inspect xgboost feature width") from exc
    if width != len(expected_features):
        raise XGBoostBundleCompatibilityError("xgboost feature width does not match metadata")
    raw_names = getattr(booster, "feature_names", None)
    if raw_names is None or tuple(raw_names) != expected_features:
        raise XGBoostBundleCompatibilityError("xgboost feature names do not match metadata")


# 校验最佳迭代、已评估轮数和验证 RMSE，并要求 XGBoost 预处理标记为 NONE。
def _validate_fit_summary(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("fit_summary must be a mapping")
    best_iteration = _required_int(value, "best_iteration")
    rounds_evaluated = _required_int(value, "rounds_evaluated")
    if best_iteration < 0:
        raise ValueError("best_iteration must be non-negative")
    if rounds_evaluated <= best_iteration:
        raise ValueError("rounds_evaluated must follow best_iteration")
    score = _required_float(value, "best_validation_rmse")
    if score < 0:
        raise ValueError("best_validation_rmse must be non-negative")
    if value.get("preprocessing") != "NONE":
        raise ValueError("xgboost preprocessing must be NONE")
    return {
        "best_iteration": best_iteration,
        "rounds_evaluated": rounds_evaluated,
        "best_validation_rmse": score,
        "preprocessing": "NONE",
    }


# 从 XGBoost 产物字段中读取严格整数。
def _required_int(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if type(result) is not int:
        raise XGBoostBundleCompatibilityError(f"{key} must be int")
    return result


# 从 XGBoost 产物字段中读取有限浮点数。
def _required_float(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise XGBoostBundleCompatibilityError(f"{key} must be numeric")
    converted = float(result)
    if not math.isfinite(converted):
        raise XGBoostBundleCompatibilityError(f"{key} must be finite")
    return converted


__all__ = [
    "XGBOOST_BUNDLE_FORMAT",
    "XGBOOST_BUNDLE_FORMAT_VERSION",
    "DailyXGBoostBundle",
    "DailyXGBoostWorkflow",
    "DailyXGBoostWorkflowResult",
    "LoadedXGBoostInferenceBundle",
    "XGBoostBundleCompatibilityError",
    "XGBoostTrainingConfig",
    "XGBoostUnavailableError",
    "load_daily_xgboost_bundle_for_inference",
    "require_xgboost",
    "validate_xgboost_model_parameters",
    "xgboost_bundle_sha256",
]
