import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from etf_backtest.live.config import ModelLiveConfig
from etf_backtest.strategy.model_device import normalize_model_device
from etf_backtest.strategy.model_training import TorchTrainingConfig, _resolve_torch_device
from etf_backtest.strategy.xgboost_training import (
    XGBoostTrainingConfig,
    _configure_xgboost_device,
)


@pytest.mark.parametrize("value,expected", [
    ("cpu", "cpu"), (" CPU ", "cpu"), ("cuda", "cuda"), ("cuda:1", "cuda:1"),
    ("gpu", "cuda"), ("GPU:0", "cuda:0"),
])
def test_device_config_normalization(value, expected):
    assert normalize_model_device(value) == expected
    for config_type in (TorchTrainingConfig, XGBoostTrainingConfig):
        config = config_type(device=value)
        assert config.device == expected
        assert config.to_parameters()["device"] == expected
    assert ModelLiveConfig(backend="torch", bundle_path="model.pt", device=value).device == expected


@pytest.mark.parametrize("value", ["auto", "mps", "cuda:-1", "cuda:01", "cuda:", "cpu:0", ""])
def test_invalid_device_is_rejected(value):
    for config_type in (TorchTrainingConfig, XGBoostTrainingConfig):
        with pytest.raises(ValueError, match="device"):
            config_type(device=value)
    with pytest.raises(ValueError, match="device"):
        ModelLiveConfig(backend="xgboost", bundle_path="model.ubj", device=value)


def test_device_defaults_remain_cpu():
    assert TorchTrainingConfig().to_parameters()["device"] == "cpu"
    assert XGBoostTrainingConfig().to_parameters()["device"] == "cpu"
    assert ModelLiveConfig(backend="torch", bundle_path="model.pt").device == "cpu"


def test_torch_rejects_unavailable_cuda_and_invalid_index():
    backend = SimpleNamespace(cuda=SimpleNamespace(
        is_available=Mock(return_value=False), device_count=Mock(return_value=1),
    ))
    assert _resolve_torch_device(backend, "cpu") == "cpu"
    backend.cuda.is_available.assert_not_called()
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_torch_device(backend, "cuda")
    backend.cuda.is_available.return_value = True
    assert _resolve_torch_device(backend, "gpu:0") == "cuda:0"
    with pytest.raises(RuntimeError, match="does not exist"):
        _resolve_torch_device(backend, "cuda:1")


@pytest.mark.parametrize("requested,actual", [("cuda", "cpu"), ("cuda:1", "cuda:0")])
def test_xgboost_rejects_silent_device_replacement(requested, actual):
    booster = Mock()
    booster.save_config.return_value = json.dumps({
        "learner": {"generic_param": {"device": actual}},
    })
    with pytest.raises(RuntimeError, match="backend selected"):
        _configure_xgboost_device(booster, requested)
    booster.set_param.assert_called_once_with({"device": requested})


def test_torch_gpu_unavailable_stops_before_training(monkeypatch):
    torch = pytest.importorskip("torch")
    from etf_backtest.strategy.model_training import DailyTorchWorkflow
    from tests.unit.strategy import test_daily_torch_runtime as fixtures

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    factory = fixtures._TinyMlpFactory()
    factory.create = Mock()
    workflow = DailyTorchWorkflow(
        feature_builder=fixtures._OneFeatureBuilder(), model_factory=factory,
        data_identity=fixtures._identity(), training_config=TorchTrainingConfig(device="cuda"),
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        workflow.fit(fixtures._dataset())
    factory.create.assert_not_called()
    assert workflow.bundle is None


def test_xgboost_gpu_fallback_stops_before_training(monkeypatch):
    xgboost = pytest.importorskip("xgboost")
    from etf_backtest.strategy.xgboost_training import DailyXGBoostWorkflow
    from tests.unit.strategy import test_xgboost_workflow as fixtures

    probe = Mock()
    probe.save_config.return_value = json.dumps({
        "learner": {"generic_param": {"device": "cpu"}},
    })
    monkeypatch.setattr(xgboost, "Booster", Mock(return_value=probe))
    train = Mock()
    monkeypatch.setattr(xgboost, "train", train)
    workflow = DailyXGBoostWorkflow(
        feature_builder=fixtures._OneFeatureBuilder(), model_spec=fixtures._TreeSpec(),
        data_identity=fixtures._identity(), training_config=XGBoostTrainingConfig(device="cuda"),
    )
    with pytest.raises(RuntimeError, match="backend selected 'cpu'"):
        workflow.fit(fixtures._dataset())
    train.assert_not_called()
    assert workflow.bundle is None


@pytest.mark.parametrize("backend", ["torch", "xgboost"])
def test_components_forward_independent_inference_device(backend, monkeypatch):
    from pathlib import Path

    import etf_backtest.strategy.model as module

    project = Path(__file__).resolve().parents[3]
    sample = "beginner_example" if backend == "torch" else "xgboost_example"
    components = module.load_user_model_components(
        project / "private_strategy" / sample / "model.py",
        allowed_root=project / "private_strategy",
    )
    loader = Mock()
    monkeypatch.setattr(module, f"load_daily_{backend}_bundle_for_inference", loader)
    components.load_inference_bundle(Path("unused"), backend=backend,
                                     signal_date=date(2025, 1, 1), device="cuda:1")
    assert loader.call_args.kwargs["device"] == "cuda:1"
    assert components.settings.training.device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_torch_device_training_and_portable_cpu_reload(device, tmp_path):
    torch = pytest.importorskip("torch")
    from etf_backtest.strategy.model_training import (
        DailyTorchWorkflow,
        load_daily_torch_bundle_for_inference,
    )
    from etf_backtest.strategy.portfolio import TopKPortfolio
    from tests.unit.strategy import test_daily_torch_runtime as fixtures

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch and GPU required")
    config = replace(fixtures._training(), device=device)
    dataset = fixtures._dataset()
    workflow = DailyTorchWorkflow(
        feature_builder=fixtures._OneFeatureBuilder(), model_factory=fixtures._TinyMlpFactory(),
        data_identity=fixtures._identity(), training_config=config,
    )
    result = workflow.fit(dataset)
    assert result.bundle.device == device
    assert next(result.bundle._new_model(torch).parameters()).device == torch.device(device)
    path = workflow.save(tmp_path / "bundle.pt")
    loaded = load_daily_torch_bundle_for_inference(
        path, feature_builder=fixtures._OneFeatureBuilder(), model_factory=fixtures._TinyMlpFactory(),
        portfolio=TopKPortfolio(), signal_date=date(2021, 3, 1), device="cpu",
    )
    assert loaded.bundle.device == "cpu"
    assert json.loads(loaded.bundle.metadata.training_parameters_json)["device"] == device
    np.testing.assert_allclose(
        [p.score for p in loaded.bundle.predict(dataset.test)],
        [p.score for p in result.test_predictions], rtol=1e-5, atol=1e-6,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_xgboost_device_training_and_portable_cpu_reload(device, tmp_path):
    xgboost = pytest.importorskip("xgboost")
    from etf_backtest.strategy.portfolio import TopKPortfolio
    from etf_backtest.strategy.xgboost_training import (
        DailyXGBoostWorkflow,
        load_daily_xgboost_bundle_for_inference,
    )
    from tests.unit.strategy import test_xgboost_workflow as fixtures

    if device.startswith("cuda"):
        data = xgboost.DMatrix(np.ones((2, 1)), label=np.ones(2))
        probe = xgboost.Booster(params={"nthread": 1}, cache=[data])
        try:
            _configure_xgboost_device(probe, device)
        except RuntimeError as exc:
            pytest.skip(str(exc))
    dataset = fixtures._dataset()
    workflow = DailyXGBoostWorkflow(
        feature_builder=fixtures._OneFeatureBuilder(), model_spec=fixtures._TreeSpec(),
        data_identity=fixtures._identity(),
        training_config=XGBoostTrainingConfig(device=device, num_boost_round=5),
    )
    result = workflow.fit(dataset)
    assert result.bundle.device == device
    path = workflow.save(tmp_path / "bundle.ubj")
    loaded = load_daily_xgboost_bundle_for_inference(
        path, feature_builder=fixtures._OneFeatureBuilder(), model_spec=fixtures._TreeSpec(),
        portfolio=TopKPortfolio(), signal_date=date(2021, 3, 1), device="cpu",
    )
    assert loaded.bundle.device == "cpu"
    assert json.loads(loaded.bundle.metadata.training_parameters_json)["device"] == device
    np.testing.assert_allclose(
        [p.score for p in loaded.bundle.predict(dataset.test)],
        [p.score for p in result.test_predictions], rtol=1e-5, atol=1e-6,
    )
