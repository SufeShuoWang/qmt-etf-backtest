"""由预测器 bundle 驱动且与具体框架无关的日频策略。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from etf_backtest.core.market import MarketBarView
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView, StrategyContext
from etf_backtest.strategy.model_contracts import (
    FeatureBuilder,
    PredictionRecord,
    PredictorBundle,
    feature_fingerprint,
    feature_records_for_signal,
    validate_feature_builder,
)
from etf_backtest.strategy.portfolio import (
    ModelPortfolioPolicy,
    validate_model_allocation,
)
from etf_backtest.strategy.scheduler import EveryTradingDayScheduler


@dataclass(frozen=True, slots=True)
class ModelAllocationRecord:
    """单条预测在信号日的排名及实际采用的目标权重。"""

    signal_date: date
    symbol: str
    score: float
    rank: int
    selected: bool
    target_weight: Decimal


class DailyModelStrategy(BaseStrategy):
    """将日频模型分数转换为经过校验的多资产目标组合。"""

    __slots__ = (
        "_allocations",
        "_bundle",
        "_feature_builder",
        "_lookback",
        "_portfolio",
        "_predictions",
        "_scheduler",
    )

    # 绑定特征构建器、已训练预测包与组合政策，初始化逐日预测和分配记录。
    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder,
        bundle: PredictorBundle,
        portfolio: ModelPortfolioPolicy,
    ) -> None:
        feature_names, lookback = validate_feature_builder(feature_builder)
        if not isinstance(bundle, PredictorBundle):
            raise TypeError("bundle must satisfy PredictorBundle")
        metadata = bundle.metadata
        if metadata.feature_names != feature_names:
            raise ValueError("bundle feature_names do not match FeatureBuilder")
        if metadata.feature_fingerprint != feature_fingerprint(feature_names, metadata.label_name):
            raise ValueError("bundle feature fingerprint is invalid")
        if not isinstance(portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        self._feature_builder = feature_builder
        self._bundle = bundle
        self._lookback = lookback
        self._portfolio = portfolio
        self._predictions: list[PredictionRecord] = []
        self._allocations: list[ModelAllocationRecord] = []
        self._scheduler = EveryTradingDayScheduler()

    # 返回已经生成的预测记录，供结果文件与评估使用。
    @property
    def predictions(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._predictions)

    # 返回逐日组合分配记录，供审计模型得分如何变成仓位。
    @property
    def allocations(self) -> tuple[ModelAllocationRecord, ...]:
        return tuple(self._allocations)

    # 返回特征构建器要求的历史窗口长度。
    @property
    def required_history_trading_days(self) -> int:
        return self._lookback

    # 使用每日调度器判断是否生成模型目标。
    def should_generate_target(self, frame_index: int) -> bool:
        return self._scheduler.should_decide(frame_index)

    # 为信号日构建特征并预测，核对预测主键后分配权重；输出覆盖证券范围的目标，当前空／无正权重分配会报错。
    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> TargetPortfolio:
        del account_view
        if signal_date <= self._bundle.metadata.trained_through:
            raise ValueError("model signal_date must follow the training interval")
        records = feature_records_for_signal(
            builder=self._feature_builder,
            market_views=market_history,
            signal_date=signal_date,
        )
        if not records:
            raise ValueError("model signal date produced no feature records")
        predictions = self._bundle.predict(records)
        expected_keys = tuple(record.key for record in records)
        actual_keys = tuple(prediction.key for prediction in predictions)
        if len(actual_keys) != len(set(actual_keys)) or frozenset(actual_keys) != frozenset(
            expected_keys
        ):
            raise ValueError("bundle predictions must exactly match feature record keys")
        self._predictions.extend(predictions)
        weights = validate_model_allocation(
            self._portfolio.allocate(predictions),
            predictions=predictions,
            exposure_cap=self._portfolio.max_total_weight,
        )
        if not weights or not any(weight > Decimal("0") for weight in weights.values()):
            raise ValueError("model portfolio must select at least one positive-weight symbol")
        complete_weights = {symbol: weights.get(symbol, Decimal("0")) for symbol in context.symbols}
        ranked = sorted(
            predictions,
            key=lambda prediction: (prediction.score, prediction.key.symbol),
            reverse=True,
        )
        self._allocations.extend(
            ModelAllocationRecord(
                signal_date=prediction.key.signal_date,
                symbol=prediction.key.symbol,
                score=prediction.score,
                rank=rank,
                selected=weights.get(prediction.key.symbol, Decimal("0")) > Decimal("0"),
                target_weight=weights.get(prediction.key.symbol, Decimal("0")),
            )
            for rank, prediction in enumerate(ranked, start=1)
        )
        return TargetPortfolio(weights=complete_weights)


__all__ = ["DailyModelStrategy", "ModelAllocationRecord"]
