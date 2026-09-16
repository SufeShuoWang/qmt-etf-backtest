"""Three-factor deterministic XGBoost regression strategy example."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model import (
    DateRange,
    FeatureBuilder,
    ModelSettings,
    TopKPortfolio,
    XGBoostTrainingConfig,
)

MODEL_SETTINGS = ModelSettings(
    backend="xgboost",
    train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
    valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
    portfolio=TopKPortfolio(
        top_k=2,
        total_weight="0.90",
        min_score=-1.0,
        weighting="equal",
    ),
    training=XGBoostTrainingConfig(
        device="cpu",  # 改为 "cuda" 使用 GPU，"cuda:0" 指定第一张显卡。
        seed=42,
        num_boost_round=500,
        early_stopping_rounds=30,
        min_delta=0.0,
    ),
    feature_kwargs={},
    model_kwargs={},
)


# XGBoost 示例特征构建器：只使用证券截至信号日的历史价格和成交量。
class Features(FeatureBuilder):
    feature_names: tuple[str, ...] = (
        "return_5d",
        "return_20d",
        "volume_ratio_5_20",
    )
    required_history_trading_days: int = 21

    # 计算 5／20 日收益及短长均量比；窗口不足、包含停牌或均量无效时不生成样本。
    def build_features(
        self, *, symbol: str, signal_date: date, history: Sequence[MarketBarView]
    ) -> Sequence[Decimal] | None:
        del symbol, signal_date
        if len(history) < 21 or any(bar.suspended or bar.close <= 0 for bar in history[-21:]):
            return None
        recent = history[-21:]
        volume_5 = Decimal(sum(bar.volume for bar in recent[-5:])) / Decimal("5")
        volume_20 = Decimal(sum(bar.volume for bar in recent[-20:])) / Decimal("20")
        if volume_20 <= 0:
            return None
        return (
            recent[-1].close / recent[-6].close - Decimal("1"),
            recent[-1].close / recent[0].close - Decimal("1"),
            volume_5 / volume_20,
        )


# 声明示例 XGBoost 模型身份及用户可设的树参数，训练过程由框架工作流执行。
class Model:
    # 返回示例 XGBoost 模型稳定标识。
    model_id = "my_xgboost_v2_equal_weight"
    model_class_name = "Model"
    model_parameters = {
        "eta": 0.03,
        "max_depth": 8,
    }
