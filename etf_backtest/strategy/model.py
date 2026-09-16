"""固定的用户 Model 扩展契约及其共用具体工作流。"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Literal

from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    DateRange,
    FeatureBuilder,
    FeatureRecord,
    LabeledRecord,
    ModelDataIdentity,
    ModelMetadata,
    ModelSpec,
    ModelWorkflow,
    ModelWorkflowResult,
    PredictionRecord,
    PredictorBundle,
    SampleKey,
    TorchModelFactory,
    canonical_json,
    validate_feature_builder,
)
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_training import (
    DailyModelDatasetBuilder,
    DailyTorchBundle,
    DailyTorchDatasetBuilder,
    DailyTorchWorkflow,
    DailyTorchWorkflowResult,
    TorchTrainingConfig,
    TorchUnavailableError,
    LoadedInferenceBundle,
    load_daily_torch_bundle_for_inference,
)
from etf_backtest.strategy.portfolio import (
    AllocationFunction,
    CustomPortfolio,
    ModelPortfolioPolicy,
    PortfolioWeightInput,
    TopKPortfolio,
    WeightingMode,
)
from etf_backtest.strategy.xgboost_training import (
    XGBoostTrainingConfig,
    DailyXGBoostWorkflow,
    LoadedXGBoostInferenceBundle,
    load_daily_xgboost_bundle_for_inference,
    validate_xgboost_model_parameters,
)

_MODEL_MODULE_PREFIX = "_etf_backtest_user_model_"


class UserModelLoadError(ValueError):
    """可信本地模型文件违反受控加载契约。"""


# 检查特征／模型构造参数键可作为 Python 参数名，内容可序列化且有限，再冻结映射。
def _settings_kwargs(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    parameters = dict(value)
    if any(not isinstance(key, str) or not key.isidentifier() for key in parameters):
        raise ValueError(f"{field_name} keys must be Python identifiers")
    try:
        canonical_json(parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and JSON-compatible") from exc
    return MappingProxyType(dict(sorted(parameters.items())))


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """由 ``model.py`` 管理的模型专属数据、训练和构造设置。

    回测测试区间仍来自实验共用 YAML。用户可以修改这里的全部字段；共用 Workflow
    继续统一管理标签、训练集预处理、优化、early stopping 和日频推理。
    """

    backend: Literal["torch", "xgboost"] = "torch"
    train_range: DateRange = field(
        default_factory=lambda: DateRange(date(2021, 1, 1), date(2022, 12, 31))
    )
    valid_range: DateRange = field(
        default_factory=lambda: DateRange(date(2023, 1, 1), date(2023, 12, 31))
    )
    portfolio: ModelPortfolioPolicy = field(default_factory=TopKPortfolio)
    training: TorchTrainingConfig | XGBoostTrainingConfig = field(
        default_factory=TorchTrainingConfig
    )
    feature_kwargs: Mapping[str, object] = field(default_factory=dict)
    model_kwargs: Mapping[str, object] = field(default_factory=dict)

    # 校验模型后端、训练配置、日期切分和组合政策是否匹配，规范特征及模型构造参数。
    def __post_init__(self) -> None:
        if self.backend not in {"torch", "xgboost"}:
            raise ValueError("backend must be torch or xgboost")
        if not isinstance(self.train_range, DateRange):
            raise TypeError("train_range must be DateRange")
        if not isinstance(self.valid_range, DateRange):
            raise TypeError("valid_range must be DateRange")
        if self.train_range.end_date >= self.valid_range.start_date:
            raise ValueError(
                "train_range and valid_range must be chronological and non-overlapping"
            )
        expected_training = (
            TorchTrainingConfig if self.backend == "torch" else XGBoostTrainingConfig
        )
        if not isinstance(self.training, expected_training):
            raise TypeError(f"{self.backend} backend requires {expected_training.__name__}")
        if not isinstance(self.portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        if (
            not isinstance(self.portfolio.max_total_weight, Decimal)
            or not self.portfolio.max_total_weight.is_finite()
            or not Decimal("0") < self.portfolio.max_total_weight <= Decimal("1")
        ):
            raise ValueError("portfolio max_total_weight must be in (0, 1]")
        try:
            canonical_json(self.portfolio.resolved_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError("portfolio provenance must be finite and JSON-compatible") from exc
        object.__setattr__(
            self,
            "feature_kwargs",
            _settings_kwargs(self.feature_kwargs, "feature_kwargs"),
        )
        object.__setattr__(
            self,
            "model_kwargs",
            _settings_kwargs(self.model_kwargs, "model_kwargs"),
        )

    def resolved_dict(self) -> dict[str, object]:
        """返回写入工作流来源信息的精确代码自有值。"""

        return {
            "backend": self.backend,
            "train_range": self.train_range.to_dict(),
            "valid_range": self.valid_range.to_dict(),
            "portfolio": self.portfolio.resolved_dict(),
            "training": self.training.to_parameters(),
            "feature_kwargs": dict(self.feature_kwargs),
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(frozen=True, slots=True)
class LoadedModelComponents:
    """:func:`load_user_model_components` 返回的可审计组件集合。"""

    settings: ModelSettings
    feature_builder: FeatureBuilder
    model_factory: ModelSpec
    source_path: Path
    source_sha256: str

    def create_workflow(self, data_identity: ModelDataIdentity) -> ModelWorkflow:
        """回测：按用户模型设置选择训练后端，不在这里执行训练。"""
        if self.settings.backend == "torch":
            if not isinstance(self.model_factory, TorchModelFactory):
                raise TypeError("torch backend requires TorchModelFactory")
            if not isinstance(self.settings.training, TorchTrainingConfig):
                raise TypeError("torch backend requires TorchTrainingConfig")
            return DailyTorchWorkflow(
                feature_builder=self.feature_builder, model_factory=self.model_factory,
                data_identity=data_identity, training_config=self.settings.training,
                portfolio=self.settings.portfolio,
            )
        if not isinstance(self.settings.training, XGBoostTrainingConfig):
            raise TypeError("xgboost backend requires XGBoostTrainingConfig")
        return DailyXGBoostWorkflow(
            feature_builder=self.feature_builder, model_spec=self.model_factory,
            data_identity=data_identity, training_config=self.settings.training,
            portfolio=self.settings.portfolio,
        )

    def load_inference_bundle(
        self, path: Path, *, backend: Literal["torch", "xgboost"], signal_date: date,
        device: str = "cpu",
    ) -> LoadedInferenceBundle | LoadedXGBoostInferenceBundle:
        """模拟盘：按部署配置加载已有模型，保持原有校验，不触发训练。"""
        if backend == "torch":
            if not isinstance(self.model_factory, TorchModelFactory):
                raise TypeError("torch backend requires TorchModelFactory")
            return load_daily_torch_bundle_for_inference(
                path, feature_builder=self.feature_builder, model_factory=self.model_factory,
                portfolio=self.settings.portfolio, signal_date=signal_date,
                device=device,
            )
        return load_daily_xgboost_bundle_for_inference(
            path, feature_builder=self.feature_builder, model_spec=self.model_factory,
            portfolio=self.settings.portfolio, signal_date=signal_date,
            device=device,
        )


def load_user_model_components(
    path: str | Path,
    *,
    allowed_root: str | Path,
) -> LoadedModelComponents:
    """从单个可信本地 Python 文件加载明确命名的组件。

    所选文件必须定义模块级 ``MODEL_SETTINGS`` 值，其构造参数会传给选定的特征类和
    模型类。这是受控加载器而非沙箱：执行获准文件时，该文件拥有正常 Python 进程权限。
    """

    root = _resolved_model_root(allowed_root)
    source_path = _resolved_model_source(path, root=root)
    feature_name, model_name = "Features", "Model"
    source = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    module_name = (
        f"{_MODEL_MODULE_PREFIX}"
        f"{hashlib.sha256(str(source_path).encode('utf-8') + b'\0' + source).hexdigest()[:24]}"
    )
    module = _execute_model_module(
        module_name=module_name,
        source_path=source_path,
        source=source,
    )
    feature_type = _selected_component_class(module, feature_name)
    model_type = _selected_component_class(module, model_name)
    settings = vars(module).get("MODEL_SETTINGS")
    if not isinstance(settings, ModelSettings):
        raise UserModelLoadError("model file must define MODEL_SETTINGS as ModelSettings")
    feature_instance = _instantiate_component(
        feature_type,
        settings.feature_kwargs,
        component_name=feature_name,
    )
    model_instance = _instantiate_component(
        model_type,
        settings.model_kwargs,
        component_name=model_name,
    )
    if not isinstance(feature_instance, FeatureBuilder):
        raise UserModelLoadError(f"{feature_name} must satisfy FeatureBuilder")
    if not isinstance(model_instance, ModelSpec):
        raise UserModelLoadError(f"{model_name} must satisfy ModelSpec")
    if settings.backend == "torch" and not isinstance(model_instance, TorchModelFactory):
        raise UserModelLoadError(f"{model_name} must satisfy TorchModelFactory")
    try:
        validate_feature_builder(feature_instance)
        canonical_json(model_instance.model_parameters)
        if settings.backend == "xgboost":
            validate_xgboost_model_parameters(model_instance.model_parameters)
    except (TypeError, ValueError) as exc:
        raise UserModelLoadError(f"loaded model component validation failed: {exc}") from exc
    return LoadedModelComponents(
        settings=settings,
        feature_builder=feature_instance,
        model_factory=model_instance,
        source_path=source_path,
        source_sha256=source_sha256,
    )


# 解析并检查允许加载模型代码的根目录。
def _resolved_model_root(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UserModelLoadError("allowed_root must be an existing directory") from exc
    if not root.is_dir():
        raise UserModelLoadError("allowed_root must be an existing directory")
    return root


# 解析模型文件并检查其位置及文件类型。
def _resolved_model_source(value: str | Path, *, root: Path) -> Path:
    supplied = Path(value)
    unresolved = supplied if supplied.is_absolute() else root / supplied
    try:
        source_path = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UserModelLoadError("user Model file does not exist") from exc
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise UserModelLoadError("user Model file must stay inside allowed_root") from exc
    if source_path.suffix.casefold() != ".py" or not source_path.is_file():
        raise UserModelLoadError("user Model source must be one regular .py file")
    return source_path


# 执行模型 Python 模块并处理加载异常，取得用户声明的组件。
def _execute_model_module(*, module_name: str, source_path: Path, source: bytes) -> ModuleType:
    del source
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise UserModelLoadError(f"cannot load model module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise UserModelLoadError(
            f"user Model module execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
    return module


# 按约定名称取得模型或特征类，并检查它符合所需接口。
def _selected_component_class(module: ModuleType, class_name: str) -> type[object]:
    value = vars(module).get(class_name)
    if not isinstance(value, type):
        raise UserModelLoadError(f"model.py must define class {class_name}")
    return value


# 用配置中的关键字参数实例化组件，给构造错误补充组件上下文。
def _instantiate_component(
    component_type: type[object],
    parameters: Mapping[str, object],
    *,
    component_name: str,
) -> object:
    try:
        return component_type(**parameters)
    except Exception as exc:
        raise UserModelLoadError(
            f"{component_name} initialization failed: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "DAILY_FORWARD_RETURN_LABEL",
    "AllocationFunction",
    "CustomPortfolio",
    "DailyModelDatasetBuilder",
    "DailyModelStrategy",
    "DailyTorchBundle",
    "DailyTorchDatasetBuilder",
    "DailyTorchWorkflow",
    "DailyTorchWorkflowResult",
    "DatasetSplits",
    "DateRange",
    "FeatureBuilder",
    "FeatureRecord",
    "LabeledRecord",
    "LoadedModelComponents",
    "ModelDataIdentity",
    "ModelMetadata",
    "ModelPortfolioPolicy",
    "ModelSettings",
    "ModelSpec",
    "ModelWorkflow",
    "ModelWorkflowResult",
    "PortfolioWeightInput",
    "PredictionRecord",
    "PredictorBundle",
    "SampleKey",
    "TopKPortfolio",
    "TorchModelFactory",
    "TorchTrainingConfig",
    "TorchUnavailableError",
    "UserModelLoadError",
    "WeightingMode",
    "XGBoostTrainingConfig",
    "load_user_model_components",
]
