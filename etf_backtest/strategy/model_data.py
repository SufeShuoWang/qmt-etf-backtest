"""模型共用的特征、训练样本和评估；不依赖 Torch 或 XGBoost。"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL, DatasetSplits, DateRange, FeatureBuilder,
    FeatureRecord, LabeledRecord, PredictionRecord, RegressionMetricReport,
    build_feature_record, feature_records_for_signal, validate_feature_builder,
)

class DailyModelDatasetBuilder:
    """构建与 D 日对齐的样本，并统一生成固定前瞻标签。"""

    __slots__ = ("_feature_builder", "_feature_names", "_lookback")

    # 校验并保存特征构建器、特征名称及回看长度。
    def __init__(self, feature_builder: FeatureBuilder) -> None:
        names, lookback = validate_feature_builder(feature_builder)
        self._feature_builder = feature_builder
        self._feature_names = names
        self._lookback = lookback

    # 返回数据集使用的特征构建器。
    @property
    def feature_builder(self) -> FeatureBuilder:
        return self._feature_builder

    # 返回数据集特征列的固定顺序。
    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    # 按信号日划分样本，特征只取当日及以前历史；标签使用后两条行情的收盘价比值，当前未按标签日期隔离切分边界。
    def build(
        self,
        *,
        market_views: Iterable[MarketBarView],
        train_range: DateRange,
        valid_range: DateRange,
        test_range: DateRange,
    ) -> DatasetSplits:
        for value, field_name in (
            (train_range, "train_range"),
            (valid_range, "valid_range"),
            (test_range, "test_range"),
        ):
            if not isinstance(value, DateRange):
                raise TypeError(f"{field_name} must be DateRange")
        if not train_range.end_date < valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not valid_range.end_date < test_range.start_date:
            raise ValueError("valid_range must precede test_range")

        by_symbol: dict[str, dict[date, MarketBarView]] = {}
        for view in market_views:
            if not isinstance(view, MarketBarView):
                raise TypeError("market_views may contain only MarketBarView values")
            rows = by_symbol.setdefault(view.symbol, {})
            if view.trade_date in rows:
                raise ValueError(f"duplicate daily view for {view.symbol} on {view.trade_date}")
            rows[view.trade_date] = view
        if not by_symbol:
            raise ValueError("market_views must not be empty")

        split_rows: dict[str, list[LabeledRecord]] = {"train": [], "valid": [], "test": []}
        for symbol in sorted(by_symbol):
            ordered = tuple(by_symbol[symbol][day] for day in sorted(by_symbol[symbol]))
            for index in range(len(ordered) - 2):
                signal_date = ordered[index].trade_date
                split_name = _split_name(
                    signal_date,
                    train_range=train_range,
                    valid_range=valid_range,
                    test_range=test_range,
                )
                if split_name is None:
                    continue
                history = ordered[max(0, index - self._lookback + 1) : index + 1]
                feature_record = build_feature_record(
                    builder=self._feature_builder,
                    symbol=symbol,
                    signal_date=signal_date,
                    history=history,
                )
                if feature_record is None:
                    continue
                label = ordered[index + 2].close / ordered[index + 1].close - Decimal("1")
                split_rows[split_name].append(
                    LabeledRecord(
                        key=feature_record.key,
                        features=feature_record.features,
                        label=label,
                    )
                )
        return DatasetSplits(
            feature_names=self._feature_names,
            label_name=DAILY_FORWARD_RETURN_LABEL,
            train_range=train_range,
            valid_range=valid_range,
            test_range=test_range,
            train=tuple(split_rows["train"]),
            valid=tuple(split_rows["valid"]),
            test=tuple(split_rows["test"]),
        )

    # 为某个信号日构造各证券特征记录，供逐日模型策略推理。
    def features_for_signal(
        self,
        *,
        market_views: Sequence[MarketBarView],
        signal_date: date,
    ) -> tuple[FeatureRecord, ...]:
        return feature_records_for_signal(
            builder=self._feature_builder,
            market_views=market_views,
            signal_date=signal_date,
        )


# 根据样本信号日期选择训练、验证或测试分组，区间外返回空结果。
def _split_name(
    signal_date: date,
    *,
    train_range: DateRange,
    valid_range: DateRange,
    test_range: DateRange,
) -> str | None:
    if train_range.contains(signal_date):
        return "train"
    if valid_range.contains(signal_date):
        return "valid"
    if test_range.contains(signal_date):
        return "test"
    return None


# 按样本特征顺序构造 NumPy 二维特征矩阵。
def _feature_matrix(
    records: Sequence[FeatureRecord],
    feature_count: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    matrix = np.asarray(
        [[float(value) for value in record.features] for record in records],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape != (len(records), feature_count):
        raise ValueError("feature matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must be finite")
    return matrix


# 从有标签样本提取 NumPy 目标向量。
def _target_vector(
    records: Sequence[LabeledRecord],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    target = np.asarray([float(record.label) for record in records], dtype=np.float64)
    if target.shape != (len(records),) or not np.isfinite(target).all():
        raise ValueError("target vector must be finite and one-dimensional")
    return target


# 先按样本键精确对齐预测，再计算均方误差、平均绝对误差和预测相关系数。
def evaluate_predictions(
    *,
    samples: Sequence[LabeledRecord],
    predictions: Sequence[PredictionRecord],
) -> RegressionMetricReport:
    ordered = tuple(sorted(samples, key=lambda sample: sample.key))
    if not ordered or any(not isinstance(sample, LabeledRecord) for sample in ordered):
        raise ValueError("samples must contain labeled records")
    by_key: dict[object, PredictionRecord] = {}
    for prediction in predictions:
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predictions may contain only PredictionRecord values")
        if prediction.key in by_key:
            raise ValueError("predictions contain duplicate keys")
        by_key[prediction.key] = prediction
    expected_keys = tuple(sample.key for sample in ordered)
    if frozenset(by_key) != frozenset(expected_keys):
        raise ValueError("predictions must exactly match sample keys")
    target = _target_vector(ordered)
    score = np.asarray([by_key[key].score for key in expected_keys], dtype=np.float64)
    error = score - target
    if np.std(score) == 0.0 or np.std(target) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(score, target)[0, 1])
    return RegressionMetricReport(
        sample_count=len(ordered),
        mean_squared_error=float(np.mean(error**2)),
        mean_absolute_error=float(np.mean(np.abs(error))),
        prediction_correlation=correlation,
    )
