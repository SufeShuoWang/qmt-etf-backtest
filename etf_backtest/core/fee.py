"""对每笔正式成交独立执行的纯 Decimal 费用计算。"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from etf_backtest.config.schema import FeeConfig
from etf_backtest.core.order import OrderSide

_ZERO: Final = Decimal("0")
_MONEY_QUANTUM: Final = Decimal("0.001")


def _quantize_money(value: Decimal) -> Decimal:
    """按统一财务结果精度返回费用金额。"""

    return value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


# 检查费用计算所用金额是有限非负 Decimal。
def _require_non_negative_money(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value < _ZERO:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """不可变的逐笔成交费用明细。"""

    commission: Decimal
    stamp_duty: Decimal
    total: Decimal

    # 校验佣金、印花税及总费用之间的数值关系。
    def __post_init__(self) -> None:
        commission = _require_non_negative_money(self.commission, "commission")
        stamp_duty = _require_non_negative_money(self.stamp_duty, "stamp_duty")
        total = _require_non_negative_money(self.total, "total")
        if total != commission + stamp_duty:
            raise ValueError("fee total is inconsistent with its components")


class FeeModel:
    """逐笔计算佣金及区分买卖方向的费用。"""

    __slots__ = ("_config",)

    # 保存佣金比例、最低佣金及印花税参数，供每笔成交统一计费。
    def __init__(self, config: FeeConfig) -> None:
        if not isinstance(config, FeeConfig):
            raise TypeError("config must be FeeConfig")
        self._config = config

    def calculate(
        self,
        *,
        trade_amount: Decimal,
        side: OrderSide,
    ) -> FeeBreakdown:
        """返回单笔候选成交的独立费用明细。"""
        amount = _require_non_negative_money(trade_amount, "trade_amount")
        if not isinstance(side, OrderSide):
            raise TypeError("side must be OrderSide")
        if amount == _ZERO:
            return FeeBreakdown(
                commission=_ZERO,
                stamp_duty=_ZERO,
                total=_ZERO,
            )

        commission = _quantize_money(
            max(
                amount * self._config.commission_rate,
                self._config.minimum_commission,
            )
        )
        stamp_duty = (
            _quantize_money(amount * self._config.stamp_duty_rate)
            if side is OrderSide.SELL
            else _ZERO
        )
        return FeeBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            total=commission + stamp_duty,
        )


__all__ = ["FeeBreakdown", "FeeModel"]
