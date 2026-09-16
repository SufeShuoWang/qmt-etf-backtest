"""120 日日收益率序列 → LSTM → 每日 Top3 等权。"""

from datetime import date

from etf_backtest.strategy.model import (
    DateRange,
    ModelSettings,
    TopKPortfolio,
    TorchTrainingConfig,
)


MODEL_SETTINGS = ModelSettings(
    train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
    valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
    portfolio=TopKPortfolio(top_k=3, total_weight="1", weighting="equal"),
    training=TorchTrainingConfig(
        seed=42, max_epochs=50, patience=5, batch_size=32, learning_rate=0.001, device="cuda",
    ),
)


class Features:
    feature_names = tuple(f"return_lag_{lag}" for lag in range(119, -1, -1))
    required_history_trading_days = 121

    def build_features(self, symbol, signal_date, history):
        # 新上市基金历史不足时不生成样本；121 个收盘价形成 120 个日收益率。
        if len(history) < 121:
            return None
        return tuple(b.close / a.close - 1 for a, b in zip(history, history[1:]))


class Model:
    model_id = "lstm_120_v1"
    model_class_name = "Model"
    model_parameters = {"hidden_dim": 50, "num_layers": 2, "fc_dim": 25}

    def create(self, input_dim, seed):
        from torch import nn

        class LSTM(nn.Module):
            def __init__(self, hidden_dim, num_layers, fc_dim):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, 1),
                )

            def forward(self, x):
                _, (hidden, _) = self.lstm(x.unsqueeze(-1))
                return self.head(hidden[-1])

        return LSTM(**self.model_parameters)
