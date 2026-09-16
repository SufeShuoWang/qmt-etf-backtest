"""XGBoost fit-once workflow and immutable UBJSON bundle tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("xgboost")

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    DateRange,
    LabeledRecord,
    ModelDataIdentity,
    SampleKey,
)
from etf_backtest.strategy.portfolio import TopKPortfolio
from etf_backtest.strategy.xgboost_training import (
    DailyXGBoostWorkflow,
    XGBoostBundleCompatibilityError,
    XGBoostTrainingConfig,
    XGBoostUnavailableError,
    load_daily_xgboost_bundle_for_inference,
    require_xgboost,
)


class _OneFeatureBuilder:
    feature_names = ("x",)
    required_history_trading_days = 1

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> tuple[Decimal, ...] | None:
        del symbol, signal_date
        if not history:
            return None
        return (history[-1].close,)


class _TreeSpec:
    model_id = "tests.xgboost"
    model_class_name = "TreeSpec"
    model_parameters: Mapping[str, object] = {
        "eta": 0.1,
        "max_depth": 2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }


def _records(start: date, values: tuple[int, ...]) -> tuple[LabeledRecord, ...]:
    return tuple(
        LabeledRecord(
            key=SampleKey(signal_date=start + timedelta(days=index), symbol="SH.510300"),
            features=(Decimal(value),),
            label=Decimal(value * value) / Decimal("1000"),
        )
        for index, value in enumerate(values)
    )


def _dataset() -> DatasetSplits:
    return DatasetSplits(
        feature_names=("x",),
        label_name=DAILY_FORWARD_RETURN_LABEL,
        train_range=DateRange(date(2021, 1, 1), date(2021, 1, 12)),
        valid_range=DateRange(date(2021, 2, 1), date(2021, 2, 4)),
        test_range=DateRange(date(2021, 3, 1), date(2021, 3, 4)),
        train=_records(date(2021, 1, 1), tuple(range(1, 13))),
        valid=_records(date(2021, 2, 1), (13, 14, 15, 16)),
        test=_records(date(2021, 3, 1), (17, 18, 19, 20)),
    )


def _identity() -> ModelDataIdentity:
    return ModelDataIdentity(
        dataset_version="test-snapshot",
        manifest_sha256="a" * 64,
        snapshot_started_at_utc=datetime(2026, 8, 26, tzinfo=UTC),
    )


def _workflow() -> DailyXGBoostWorkflow:
    return DailyXGBoostWorkflow(
        feature_builder=_OneFeatureBuilder(),
        model_spec=_TreeSpec(),
        data_identity=_identity(),
        training_config=XGBoostTrainingConfig(
            seed=17,
            num_boost_round=40,
            early_stopping_rounds=5,
            min_delta=0.0,
        ),
        portfolio=TopKPortfolio(),
    )


@pytest.mark.unit
def test_xgboost_workflow_is_deterministic_and_test_labels_do_not_train() -> None:
    dataset = _dataset()
    first = _workflow().fit(dataset)
    repeated = _workflow().fit(dataset)
    changed_test = tuple(
        replace(record, label=record.label + Decimal("100")) for record in dataset.test
    )
    changed = replace(dataset, test=changed_test)
    changed_result = _workflow().fit(changed)

    first_scores = [value.score for value in first.test_predictions]
    np.testing.assert_allclose(
        first_scores,
        [value.score for value in repeated.test_predictions],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        first_scores,
        [value.score for value in changed_result.test_predictions],
        rtol=0,
        atol=0,
    )
    assert first.fit_summary["preprocessing"] == "NONE"
    assert first.fit_summary["rounds_evaluated"] <= 40


@pytest.mark.unit
def test_xgboost_bundle_round_trip_and_live_validation(tmp_path) -> None:
    dataset = _dataset()
    workflow = _workflow()
    result = workflow.fit(dataset)
    path = workflow.save(tmp_path / "model_bundle.ubj", source_run_dir=tmp_path / "run-1")

    loaded = _workflow().load(path, dataset=dataset)
    np.testing.assert_allclose(
        [value.score for value in loaded.predict(dataset.test)],
        [value.score for value in result.test_predictions],
        rtol=0,
        atol=0,
    )
    live = load_daily_xgboost_bundle_for_inference(
        path,
        feature_builder=_OneFeatureBuilder(),
        model_spec=_TreeSpec(),
        portfolio=TopKPortfolio(),
        signal_date=date(2021, 3, 1),
    )
    assert live.bundle.metadata.model_id == "tests.xgboost"
    assert live.source_run_dir == str((tmp_path / "run-1").resolve())
    assert len(live.file_sha256) == 64

    with pytest.raises(XGBoostBundleCompatibilityError, match="SHA-256"):
        load_daily_xgboost_bundle_for_inference(
            path,
            feature_builder=_OneFeatureBuilder(),
            model_spec=_TreeSpec(),
            portfolio=TopKPortfolio(),
            signal_date=date(2021, 3, 1),
            expected_sha256="0" * 64,
        )


@pytest.mark.unit
def test_xgboost_rejects_framework_owned_model_parameters() -> None:
    class _InvalidSpec(_TreeSpec):
        model_parameters: ClassVar[Mapping[str, object]] = {"objective": "binary:logistic"}

    with pytest.raises(ValueError, match="framework-owned"):
        DailyXGBoostWorkflow(
            feature_builder=_OneFeatureBuilder(),
            model_spec=_InvalidSpec(),
            data_identity=_identity(),
        )


@pytest.mark.unit
def test_xgboost_bundle_rejects_feature_order_and_training_cutoff(tmp_path) -> None:
    workflow = _workflow()
    workflow.fit(_dataset())
    path = workflow.save(tmp_path / "model_bundle.ubj")

    class _WrongFeatureBuilder(_OneFeatureBuilder):
        feature_names = ("different_x",)

    with pytest.raises(XGBoostBundleCompatibilityError, match="feature_names"):
        load_daily_xgboost_bundle_for_inference(
            path,
            feature_builder=_WrongFeatureBuilder(),
            model_spec=_TreeSpec(),
            portfolio=TopKPortfolio(),
            signal_date=date(2021, 3, 1),
        )
    with pytest.raises(XGBoostBundleCompatibilityError, match="trained_through"):
        load_daily_xgboost_bundle_for_inference(
            path,
            feature_builder=_OneFeatureBuilder(),
            model_spec=_TreeSpec(),
            portfolio=TopKPortfolio(),
            signal_date=date(2021, 1, 1),
        )


@pytest.mark.unit
def test_xgboost_dependency_error_is_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import etf_backtest.strategy.xgboost_training as module

    def missing(name: str):
        raise ModuleNotFoundError("No module named 'xgboost'", name=name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(XGBoostUnavailableError, match=r"\[xgboost\]"):
        require_xgboost()
