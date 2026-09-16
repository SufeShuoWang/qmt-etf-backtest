"""用于日频 ETF 预测的可插拔、确定性 PyTorch 工作流。

导入本模块时不会导入 PyTorch；只有在拟合、预测、保存或加载 torch state-dict bundle
时才需要该框架。
"""

from __future__ import annotations

import importlib
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np

from etf_backtest.file_utils import sha256_file as bundle_sha256
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    FeatureBuilder,
    FeatureRecord,
    ModelDataIdentity,
    ModelMetadata,
    PredictionRecord,
    RegressionMetricReport,
    TorchModelFactory,
    canonical_json,
    feature_fingerprint,
    validate_feature_builder,
    validate_dataset_schema,
    build_model_metadata,
    build_inference_payload,
    validate_model_metadata,
    validate_inference_payload,
)
from etf_backtest.strategy.model_data import (
    DailyModelDatasetBuilder,
    _feature_matrix,
    _target_vector,
    evaluate_predictions,
)
from etf_backtest.strategy.portfolio import ModelPortfolioPolicy, TopKPortfolio
from etf_backtest.strategy.model_device import normalize_model_device

TORCH_BUNDLE_FORMAT: Final = "ETF_DAILY_TORCH_STATE_DICT_V2"
INFERENCE_BUNDLE_FORMAT_VERSION: Final = 1


class TorchUnavailableError(ImportError):
    """未安装可选的 PyTorch 运行时。"""


class TorchBundleCompatibilityError(ValueError):
    """已保存 bundle 与请求的工作流身份不匹配。"""


def require_torch() -> Any:
    """延迟导入 PyTorch；缺少依赖时抛出可操作的提示。"""

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise TorchUnavailableError(
                "PyTorch is required for the daily torch model workflow; "
                "install a Python 3.12 compatible torch build before fitting, "
                "predicting, saving, or loading a torch bundle"
            ) from exc
        raise


@dataclass(frozen=True, slots=True)
class TorchTrainingConfig:
    """确定性的顺序 mini-batch 与验证集 early stopping 策略。"""

    seed: int = 20260803
    max_epochs: int = 500
    patience: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    min_delta: float = 0.0
    device: str = "cpu"

    # 检查 Torch 批大小、学习率、训练轮数、早停耐心值和随机种子等设置。
    def __post_init__(self) -> None:
        object.__setattr__(self, "device", normalize_model_device(self.device))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for field_name in ("learning_rate", "weight_decay", "min_delta"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("weight_decay and min_delta must be non-negative")

    # 把 Torch 训练设置转换为稳定参数映射，保存到模型元数据。
    def to_parameters(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "min_delta": float(self.min_delta),
            "optimizer": "Adam",
            "loss": "MSELoss",
            "device": self.device,
            "batch_mode": "DETERMINISTIC_SEQUENTIAL_MINI_BATCH",
        }


# 汇总 Torch 模型包、验证／测试预测与指标，以及最佳轮次和验证损失。
@dataclass(frozen=True, slots=True)
class DailyTorchWorkflowResult:
    bundle: DailyTorchBundle
    validation_metrics: RegressionMetricReport
    test_metrics: RegressionMetricReport
    validation_predictions: tuple[PredictionRecord, ...]
    test_predictions: tuple[PredictionRecord, ...]
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float

    # 检查训练结果包、指标类型与最佳轮次、实际轮数和损失的关系。
    def __post_init__(self) -> None:
        if not isinstance(self.bundle, DailyTorchBundle):
            raise TypeError("bundle must be DailyTorchBundle")
        for field_name in ("validation_metrics", "test_metrics"):
            if not isinstance(getattr(self, field_name), RegressionMetricReport):
                raise TypeError(f"{field_name} must be RegressionMetricReport")
        if type(self.best_epoch) is not int or self.best_epoch <= 0:
            raise ValueError("best_epoch must be a positive integer")
        if type(self.epochs_trained) is not int or self.epochs_trained < self.best_epoch:
            raise ValueError("epochs_trained must not precede best_epoch")
        if not isinstance(self.best_validation_loss, float) or not math.isfinite(
            self.best_validation_loss
        ):
            raise ValueError("best_validation_loss must be finite")

    # 提供 Torch 最佳轮次、实际训练轮数与最佳验证损失的只读摘要。
    @property
    def fit_summary(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "best_epoch": self.best_epoch,
                "epochs_trained": self.epochs_trained,
                "best_validation_loss": self.best_validation_loss,
                "preprocessing": "STANDARD_SCALER_TRAIN_ONLY",
            }
        )


@dataclass(frozen=True, slots=True)
class LoadedInferenceBundle:
    """经过哈希校验、可供 DailyModelStrategy 使用的固定 bundle。"""

    bundle: DailyTorchBundle
    file_sha256: str
    portfolio_json: str
    source_run_dir: str


class DailyTorchBundle:
    """权重以 CPU 数组保存，推理可独立选择 CPU 或 CUDA 的模型包。"""

    __slots__ = (
        "_best_epoch",
        "_best_validation_loss",
        "_epochs_trained",
        "_factory",
        "_metadata",
        "_scaler",
        "_state_dict",
        "_device",
    )

    # 保存模型元数据、网络工厂、训练集标准化器和最佳权重，校验推理所需状态。
    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        scaler: StandardScaler,
        factory: TorchModelFactory,
        state_dict: Mapping[str, np.ndarray[Any, Any]],
        best_epoch: int,
        epochs_trained: int,
        best_validation_loss: float,
        device: str = "cpu",
    ) -> None:
        if not isinstance(metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if not isinstance(scaler, StandardScaler):
            raise TypeError("scaler must be StandardScaler")
        if getattr(scaler, "n_features_in_", None) != len(metadata.feature_names):
            raise ValueError("scaler feature width does not match model metadata")
        _validate_factory(factory)
        if metadata.model_id != factory.model_id:
            raise TorchBundleCompatibilityError("factory model_id does not match metadata")
        if metadata.model_class_name != factory.model_class_name:
            raise TorchBundleCompatibilityError("factory model_class_name does not match metadata")
        if metadata.model_parameters_json != canonical_json(factory.model_parameters):
            raise TorchBundleCompatibilityError("factory parameters do not match metadata")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("state_dict must be a non-empty mapping")
        frozen_state: dict[str, np.ndarray[Any, Any]] = {}
        for raw_name, raw_value in state_dict.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("state_dict keys must be nonblank strings")
            value = np.asarray(raw_value).copy()
            if value.dtype == object:
                raise ValueError("state_dict arrays may not use object dtype")
            value.setflags(write=False)
            frozen_state[raw_name] = value
        if type(best_epoch) is not int or best_epoch <= 0:
            raise ValueError("best_epoch must be a positive integer")
        if type(epochs_trained) is not int or epochs_trained < best_epoch:
            raise ValueError("epochs_trained must not precede best_epoch")
        best_loss = float(best_validation_loss)
        if not math.isfinite(best_loss) or best_loss < 0:
            raise ValueError("best_validation_loss must be finite and non-negative")
        self._metadata = metadata
        self._scaler = scaler
        self._factory = factory
        self._state_dict = MappingProxyType(dict(sorted(frozen_state.items())))
        self._best_epoch = best_epoch
        self._epochs_trained = epochs_trained
        self._best_validation_loss = best_loss
        self._device = _resolve_torch_device(require_torch(), device)
        self._validate_state_dict()

    @property
    def device(self) -> str:
        """返回当前推理设备；它可以不同于元数据中记录的历史训练设备。"""
        return self._device

    # 返回 Torch 模型包元数据。
    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    # 返回推理使用的训练集标准化器。
    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    # 返回验证表现最佳的训练轮次。
    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    # 返回实际执行的训练轮数，可能因早停少于上限。
    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    # 返回最佳轮次的验证损失。
    @property
    def best_validation_loss(self) -> float:
        return self._best_validation_loss

    # 提供保存的网络参数状态，供保存或重建推理网络。
    @property
    def state_dict(self) -> Mapping[str, np.ndarray[Any, Any]]:
        return MappingProxyType({name: value.copy() for name, value in self._state_dict.items()})

    # 用训练集标准化器转换特征，在所选设备加载网络并推理，得分转回 CPU 后返回。
    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]:
        supplied_records = cast(object, records)
        if isinstance(supplied_records, (str, bytes)) or not isinstance(supplied_records, Sequence):
            raise TypeError("records must be a sequence")
        ordered = tuple(sorted(records, key=lambda record: record.key))
        if not ordered:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, FeatureRecord) for record in ordered):
            raise TypeError("records may contain only FeatureRecord values")
        keys = tuple(record.key for record in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("records contain duplicate sample keys")
        if any(len(record.features) != len(self._metadata.feature_names) for record in ordered):
            raise ValueError("record feature width does not match bundle metadata")
        matrix = _feature_matrix(ordered, len(self._metadata.feature_names))
        normalized = self._scaler.transform(matrix)
        torch = require_torch()
        model = self._new_model(torch)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self._device)
        model.eval()
        with torch.no_grad():
            output = _flat_output(model(tensor), expected_rows=len(ordered))
        scores = output.detach().cpu().numpy()
        return tuple(
            PredictionRecord(key=record.key, score=float(score))
            for record, score in zip(ordered, scores, strict=True)
        )

    # 通过用户工厂创建符合特征宽度的网络，供权重校验与推理使用。
    def _new_model(self, torch: Any) -> Any:
        _seed_everything(torch, self._metadata.random_seed)
        model = self._factory.create(
            input_dim=len(self._metadata.feature_names),
            seed=self._metadata.random_seed,
        )
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchModelFactory.create must return torch.nn.Module")
        model.to(self._device)
        tensors = {name: torch.as_tensor(value.copy()) for name, value in self._state_dict.items()}
        model.load_state_dict(tensors, strict=True)
        return model

    # 检查保存权重与网络参数名称、形状等兼容性。
    def _validate_state_dict(self) -> None:
        torch = require_torch()
        self._new_model(torch)


class DailyTorchWorkflow:
    """只拟合一次，在验证集上 early stopping，并持久化 state_dict bundle。"""

    __slots__ = (
        "_bundle",
        "_data_identity",
        "_factory",
        "_feature_builder",
        "_feature_names",
        "_portfolio",
        "_required_history",
        "_training_config",
    )

    # 保存特征构建器、模型工厂、训练设置和数据身份，准备 Torch 工作流。
    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder,
        model_factory: TorchModelFactory,
        data_identity: ModelDataIdentity,
        training_config: TorchTrainingConfig | None = None,
        portfolio: ModelPortfolioPolicy | None = None,
    ) -> None:
        names, required_history = validate_feature_builder(feature_builder)
        _validate_factory(model_factory)
        if not isinstance(data_identity, ModelDataIdentity):
            raise TypeError("data_identity must be ModelDataIdentity")
        if training_config is None:
            training_config = TorchTrainingConfig()
        if not isinstance(training_config, TorchTrainingConfig):
            raise TypeError("training_config must be TorchTrainingConfig")
        portfolio = portfolio or TopKPortfolio()
        if not isinstance(portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        self._feature_builder = feature_builder
        self._feature_names = names
        self._required_history = required_history
        self._portfolio = portfolio
        self._factory = model_factory
        self._data_identity = data_identity
        self._training_config = training_config
        self._bundle: DailyTorchBundle | None = None

    # 返回已训练或已加载的 Torch 包，尚未准备好时按接口约定报错。
    @property
    def bundle(self) -> DailyTorchBundle | None:
        return self._bundle

    # 返回 Torch 工作流默认产物文件名。
    @property
    def bundle_filename(self) -> str:
        return "model_bundle.pt"

    # 标准化器仅拟合训练集；网络和数据移到指定设备，早停后保存 CPU 权重副本。
    def fit(self, dataset: DatasetSplits) -> DailyTorchWorkflowResult:
        if self._bundle is not None:
            raise RuntimeError("DailyTorchWorkflow may fit or load only once")
        validate_dataset_schema(dataset, self._feature_names)
        torch = require_torch()
        device = _resolve_torch_device(torch, self._training_config.device)
        seed = self._training_config.seed
        _seed_everything(torch, seed)

        train_matrix = _feature_matrix(dataset.train, len(self._feature_names))
        valid_matrix = _feature_matrix(dataset.valid, len(self._feature_names))
        scaler = StandardScaler()
        scaler.fit(train_matrix)
        normalized_train = scaler.transform(train_matrix)
        normalized_valid = scaler.transform(valid_matrix)
        train_targets = _target_vector(dataset.train)
        valid_targets = _target_vector(dataset.valid)

        model = self._factory.create(input_dim=len(self._feature_names), seed=seed)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchModelFactory.create must return torch.nn.Module")
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self._training_config.learning_rate),
            weight_decay=float(self._training_config.weight_decay),
        )
        loss_function = torch.nn.MSELoss()
        train_x = torch.as_tensor(normalized_train, dtype=torch.float32, device=device)
        valid_x = torch.as_tensor(normalized_valid, dtype=torch.float32, device=device)
        train_y = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
        valid_y = torch.as_tensor(valid_targets, dtype=torch.float32, device=device)

        best_loss = math.inf
        best_epoch = 0
        epochs_trained = 0
        stale_epochs = 0
        best_state: dict[str, Any] | None = None
        for epoch in range(1, self._training_config.max_epochs + 1):
            epochs_trained = epoch
            model.train()
            for batch_start in range(0, len(dataset.train), self._training_config.batch_size):
                batch_end = min(
                    batch_start + self._training_config.batch_size,
                    len(dataset.train),
                )
                batch_x = train_x[batch_start:batch_end]
                batch_y = train_y[batch_start:batch_end]
                optimizer.zero_grad(set_to_none=True)
                prediction = _flat_output(model(batch_x), expected_rows=batch_end - batch_start)
                loss = loss_function(prediction, batch_y)
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("training loss became non-finite")
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_prediction = _flat_output(model(valid_x), expected_rows=len(dataset.valid))
                validation_loss = float(loss_function(valid_prediction, valid_y).item())
            if not math.isfinite(validation_loss):
                raise ValueError("validation loss became non-finite")
            if validation_loss < best_loss - float(self._training_config.min_delta):
                best_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= self._training_config.patience:
                    break
        if best_state is None or best_epoch <= 0 or not math.isfinite(best_loss):
            raise RuntimeError("validation early stopping did not produce a finite model")
        model.load_state_dict(best_state, strict=True)
        state_arrays = {
            name: value.detach().cpu().numpy().copy() for name, value in best_state.items()
        }
        metadata = build_model_metadata(
            model=self._factory, training_parameters=self._training_config.to_parameters(),
            feature_names=self._feature_names, dataset=dataset,
            random_seed=self._training_config.seed, data_identity=self._data_identity,
            framework_name="torch", framework_version=str(torch.__version__),
        )
        bundle = DailyTorchBundle(
            metadata=metadata,
            scaler=scaler,
            factory=self._factory,
            state_dict=state_arrays,
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            best_validation_loss=best_loss,
            device=device,
        )
        self._bundle = bundle
        validation_predictions = bundle.predict(dataset.valid)
        test_predictions = bundle.predict(dataset.test)
        return DailyTorchWorkflowResult(
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
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            best_validation_loss=best_loss,
        )

    # 将网络权重、标准化器、元数据与推理所需信息保存为 Torch 产物。
    def save(self, path: Path, *, source_run_dir: Path | None = None) -> Path:
        if self._bundle is None:
            raise RuntimeError("DailyTorchWorkflow has no bundle to save")
        torch = require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        run_directory = Path(source_run_dir or target.parent).resolve()
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        payload = {
            "bundle_format": TORCH_BUNDLE_FORMAT,
            "bundle_format_version": INFERENCE_BUNDLE_FORMAT_VERSION,
            "metadata": self._bundle.metadata.to_dict(),
            "inference": build_inference_payload(
                metadata=self._bundle.metadata, feature_names=self._feature_names,
                required_history=self._required_history, portfolio=self._portfolio,
                source_run_dir=run_directory,
            ),
            "scaler": _scaler_payload(self._bundle.scaler),
            "state_dict": {
                name: torch.as_tensor(value.copy())
                for name, value in self._bundle.state_dict.items()
            },
            "fit_summary": {
                "best_epoch": self._bundle.best_epoch,
                "epochs_trained": self._bundle.epochs_trained,
                "best_validation_loss": self._bundle.best_validation_loss,
            },
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    # 加载 Torch 产物并检查与当前特征、模型工厂和配置兼容，恢复可推理包。
    def load(
        self, path: Path, *, dataset: DatasetSplits, device: str | None = None,
    ) -> DailyTorchBundle:
        if self._bundle is not None:
            raise RuntimeError("DailyTorchWorkflow may fit or load only once")
        validate_dataset_schema(dataset, self._feature_names)
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        torch = require_torch()
        raw_payload = _load_payload(source, torch)
        if raw_payload.get("bundle_format") != TORCH_BUNDLE_FORMAT:
            raise TorchBundleCompatibilityError("unsupported torch bundle format")
        metadata = ModelMetadata.from_mapping(
            _required_mapping(raw_payload, "metadata")
        )
        self._validate_metadata(metadata, dataset)
        bundle = _restore_bundle(
            raw_payload, metadata, self._factory,
            device=self._training_config.device if device is None else device,
        )
        self._bundle = bundle
        return bundle


    # 核对 Torch 产物元数据与当前工作流期望的身份及模式。
    def _validate_metadata(self, metadata: ModelMetadata, dataset: DatasetSplits) -> None:
        expected = {
            "model_id": self._factory.model_id,
            "model_class_name": self._factory.model_class_name,
            "model_parameters_json": canonical_json(self._factory.model_parameters),
            "training_parameters_json": canonical_json(self._training_config.to_parameters()),
            "feature_names": self._feature_names,
            "feature_fingerprint": feature_fingerprint(
                self._feature_names, DAILY_FORWARD_RETURN_LABEL
            ),
            "label_name": DAILY_FORWARD_RETURN_LABEL,
            "train_range": dataset.train_range,
            "valid_range": dataset.valid_range,
            "test_range": dataset.test_range,
            "random_seed": self._training_config.seed,
            "data_identity": self._data_identity,
            "trained_through": dataset.trained_through,
        }
        validate_model_metadata(metadata, expected, TorchBundleCompatibilityError)
        if metadata.framework_name != "torch":
            raise TorchBundleCompatibilityError("bundle framework_name must be torch")
        current_torch = require_torch()
        saved_major = metadata.framework_version.split(".", maxsplit=1)[0]
        current_major = str(current_torch.__version__).split(".", maxsplit=1)[0]
        if saved_major != current_major:
            raise TorchBundleCompatibilityError("bundle torch major version is incompatible")


def load_daily_torch_bundle_for_inference(
    path: Path,
    *,
    feature_builder: FeatureBuilder,
    model_factory: TorchModelFactory,
    portfolio: ModelPortfolioPolicy,
    signal_date: date,
    expected_sha256: str | None = None,
    expected_model_id: str | None = None,
    device: str = "cpu",
) -> LoadedInferenceBundle:
    """加载并严格校验单个固定 bundle，不拟合任何模型。"""

    source = Path(path)
    actual_sha256 = bundle_sha256(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise TorchBundleCompatibilityError("model bundle SHA-256 does not match deployment")
    feature_names, required_history = validate_feature_builder(feature_builder)
    _validate_factory(model_factory)
    if not isinstance(portfolio, ModelPortfolioPolicy):
        raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
    if expected_model_id is not None and model_factory.model_id != expected_model_id:
        raise TorchBundleCompatibilityError("Model source model_id does not match deployment")
    torch = require_torch()
    raw_payload = _load_payload(source, torch)
    if raw_payload.get("bundle_format") != TORCH_BUNDLE_FORMAT:
        raise TorchBundleCompatibilityError(
            "unsupported torch bundle format; retrain the model to create a Live bundle"
        )
    if raw_payload.get("bundle_format_version") != INFERENCE_BUNDLE_FORMAT_VERSION:
        raise TorchBundleCompatibilityError("unsupported inference bundle format version")
    metadata = ModelMetadata.from_mapping(_required_mapping(raw_payload, "metadata"))
    expected = {
        "model_id": model_factory.model_id,
        "model_class_name": model_factory.model_class_name,
        "model_parameters_json": canonical_json(model_factory.model_parameters),
        "feature_names": feature_names,
        "feature_fingerprint": feature_fingerprint(feature_names, DAILY_FORWARD_RETURN_LABEL),
    }
    validate_model_metadata(
        metadata, expected, TorchBundleCompatibilityError, signal_date=signal_date,
    )

    inference = _required_mapping(raw_payload, "inference")
    expected_portfolio_json, source_run_dir = validate_inference_payload(
        inference, feature_names=feature_names, required_history=required_history,
        portfolio=portfolio, metadata=metadata, error_type=TorchBundleCompatibilityError,
    )

    bundle = _restore_bundle(raw_payload, metadata, model_factory, device=device)
    saved_major = metadata.framework_version.split(".", maxsplit=1)[0]
    current_major = str(torch.__version__).split(".", maxsplit=1)[0]
    if metadata.framework_name != "torch" or saved_major != current_major:
        raise TorchBundleCompatibilityError("bundle torch runtime is incompatible")
    return LoadedInferenceBundle(
        bundle=bundle,
        file_sha256=actual_sha256,
        portfolio_json=expected_portfolio_json,
        source_run_dir=source_run_dir,
    )


def _load_payload(source: Path, torch: Any) -> Mapping[str, object]:
    """两个加载入口共用文件读取；各自的兼容性要求仍在调用处检查。"""
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - 兼容旧版 torch
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TorchBundleCompatibilityError("torch bundle payload must be a mapping")
    return payload


# 从序列化字段恢复元数据、标准化器和网络权重，构造可推理 Torch 包。
def _restore_bundle(
    payload: Mapping[str, object], metadata: ModelMetadata, factory: TorchModelFactory,
    *, device: str = "cpu",
) -> DailyTorchBundle:
    scaler = _restore_scaler(_required_mapping(payload, "scaler"))
    raw_state = _required_mapping(payload, "state_dict")
    state_arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not name:
            raise TorchBundleCompatibilityError("state_dict keys must be nonblank strings")
        if not hasattr(value, "detach"):
            raise TorchBundleCompatibilityError("state_dict values must be torch tensors")
        state_arrays[name] = value.detach().cpu().numpy().copy()
    summary = _required_mapping(payload, "fit_summary")
    return DailyTorchBundle(
        metadata=metadata, scaler=scaler, factory=factory, state_dict=state_arrays,
        best_epoch=_required_int(summary, "best_epoch"),
        epochs_trained=_required_int(summary, "epochs_trained"),
        best_validation_loss=_required_float(summary, "best_validation_loss"),
        device=device,
    )


def _resolve_torch_device(torch: Any, value: str) -> str:
    """校验 CUDA 可用性与设备编号；显式选择 GPU 时不允许回退到 CPU。"""
    device = normalize_model_device(value)
    if device == "cpu":
        return device
    # 在 CUDA 运算前配置 cuBLAS 工作空间，配合已有的确定性算法开关。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Torch device {device!r} requested, but CUDA is unavailable; "
            "use device='cpu' or install a CUDA-enabled PyTorch build with a compatible GPU/driver"
        )
    if ":" in device and int(device.split(":", 1)[1]) >= torch.cuda.device_count():
        raise RuntimeError(f"Torch device {device!r} does not exist")
    return device


# 检查用户工厂满足 Torch 模型创建与身份参数接口。
def _validate_factory(factory: TorchModelFactory) -> None:
    if not isinstance(factory, TorchModelFactory):
        raise TypeError("model_factory must satisfy TorchModelFactory")
    if not isinstance(factory.model_id, str) or not factory.model_id.strip():
        raise ValueError("model_factory.model_id must be nonblank")
    if not isinstance(factory.model_class_name, str) or not factory.model_class_name.strip():
        raise ValueError("model_factory.model_class_name must be nonblank")
    canonical_json(factory.model_parameters)


# 设置训练使用的随机源与确定性选项，减少重复运行差异。
def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 把网络输出整理为每个样本一个得分，并检查输出形状。
def _flat_output(value: Any, *, expected_rows: int) -> Any:
    flattened = value.reshape(-1)
    if int(flattened.numel()) != expected_rows:
        raise ValueError("torch model must return exactly one score per input row")
    return flattened


# 提取标准化器的均值、缩放和样本数等可保存状态。
def _scaler_payload(scaler: StandardScaler) -> dict[str, object]:
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "var": np.asarray(scaler.var_, dtype=np.float64),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": np.asarray(scaler.n_samples_seen_),
    }


# 从保存状态重建标准化器，供推理使用训练时的变换。
def _restore_scaler(value: Mapping[str, object]) -> StandardScaler:
    try:
        mean = np.asarray(value["mean"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
        variance = np.asarray(value["var"], dtype=np.float64)
        feature_count = _required_int(value, "n_features_in")
        samples_seen = np.asarray(value["n_samples_seen"])
    except KeyError as exc:
        raise TorchBundleCompatibilityError("scaler payload is incomplete") from exc
    if mean.shape != (feature_count,) or scale.shape != mean.shape or variance.shape != mean.shape:
        raise TorchBundleCompatibilityError("scaler arrays have incompatible shapes")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(variance).all()
    ):
        raise TorchBundleCompatibilityError("scaler arrays must be finite")
    if (scale <= 0).any() or (variance < 0).any():
        raise TorchBundleCompatibilityError("scaler scale/variance are invalid")
    scaler = StandardScaler()
    scaler.mean_ = mean.copy()
    scaler.scale_ = scale.copy()
    scaler.var_ = variance.copy()
    scaler.n_features_in_ = feature_count
    scaler.n_samples_seen_ = samples_seen.item() if samples_seen.ndim == 0 else samples_seen.copy()
    return scaler


# 读取模型产物中必需的映射字段。
def _required_mapping(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{key} is missing") from exc
    if not isinstance(result, Mapping):
        raise TorchBundleCompatibilityError(f"{key} must be a mapping")
    return result


# 读取模型产物中必需的整数字段。
def _required_int(value: Mapping[str, object], key: str) -> int:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{key} is missing") from exc
    if type(result) is not int:
        raise TorchBundleCompatibilityError(f"{key} must be int")
    return result


# 读取模型产物中必需的有限浮点字段。
def _required_float(value: Mapping[str, object], key: str) -> float:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{key} is missing") from exc
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise TorchBundleCompatibilityError(f"{key} must be numeric")
    converted = float(result)
    if not math.isfinite(converted):
        raise TorchBundleCompatibilityError(f"{key} must be finite")
    return converted


# Backward-compatible import alias for existing user models and tests.
DailyTorchDatasetBuilder = DailyModelDatasetBuilder


__all__ = [
    "INFERENCE_BUNDLE_FORMAT_VERSION",
    "TORCH_BUNDLE_FORMAT",
    "DailyModelDatasetBuilder",
    "DailyTorchBundle",
    "DailyTorchDatasetBuilder",
    "DailyTorchWorkflow",
    "DailyTorchWorkflowResult",
    "LoadedInferenceBundle",
    "TorchBundleCompatibilityError",
    "TorchTrainingConfig",
    "TorchUnavailableError",
    "bundle_sha256",
    "evaluate_predictions",
    "load_daily_torch_bundle_for_inference",
    "require_torch",
]
