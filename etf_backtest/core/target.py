"""不可变的策略目标组合值对象。"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol


@dataclass(frozen=True, slots=True)
class NoRebalance:
    """明确要求保持现有持仓且不创建目标的策略指令。"""


NO_REBALANCE = NoRebalance()


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """策略生成的显式调仓目标权重映射。

    ``weights`` 中未出现的证券不参与本次调仓，其持有数量保持不变；显式零权重表示将对应
    证券清仓。输入映射会被防御性复制并按证券排序，随后以不可变映射公开，避免调用方事后
    改变策略决策。
    """

    weights: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        """校验 Decimal 权重并冻结规范化的防御性副本。"""
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must be a mapping")
        if not self.weights:
            raise ValueError("weights must contain at least one explicit target")

        canonical_weights: dict[str, Decimal] = {}
        total = Decimal("0")

        for symbol, weight in self.weights.items():
            if not isinstance(symbol, str):
                raise TypeError("target symbol must be a string")
            canonical_symbol = normalize_symbol(symbol)
            if not isinstance(weight, Decimal):
                raise TypeError("target weight must be Decimal")
            if not weight.is_finite():
                raise ValueError("target weight must be finite")
            if weight < Decimal("0") or weight > Decimal("1"):
                raise ValueError("target weight must be between zero and one")

            if canonical_symbol in canonical_weights:
                raise ValueError("target symbols must be unique after normalization")
            canonical_weights[canonical_symbol] = weight
            total += weight

        if total > Decimal("1"):
            raise ValueError("target weights must not sum to more than one")

        sorted_weights = dict(sorted(canonical_weights.items()))
        object.__setattr__(self, "weights", MappingProxyType(sorted_weights))

    def weight_for(self, symbol: str) -> Decimal | None:
        """返回证券的显式目标权重；未参与本次调仓时返回 ``None``。"""
        return self.weights.get(normalize_symbol(symbol))


StrategyTarget = TargetPortfolio | NoRebalance

__all__ = ["NO_REBALANCE", "NoRebalance", "StrategyTarget", "TargetPortfolio"]
