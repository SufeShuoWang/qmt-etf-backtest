"""执行估算、资金可承受性检查和正式成交创建。"""

from __future__ import annotations


from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from etf_backtest.validation import quantity as _quantity
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.order import (
    ExecutionEstimate,
    FillResult,
    Order,
    OrderSide,
    RuleCheckResult,
    TradePriceQuote,
)
from etf_backtest.config.schema import SlippageConfig


# 校验成交成本金额为有限非负 Decimal。
def _money(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


# 携带指定价格和数量对应的成交金额、费用及现金影响，供审批测算。
@dataclass(frozen=True, slots=True)
class ExecutionCost:
    quantity: int
    trade_amount: Decimal
    fee: Decimal
    total_cash_required: Decimal

    # 检查成交成本各项为合法金额且彼此一致。
    def __post_init__(self) -> None:
        quantity = _quantity(self.quantity, "quantity")
        amount = _money(self.trade_amount, "trade_amount")
        fee = _money(self.fee, "fee")
        total = _money(self.total_cash_required, "total_cash_required")
        if quantity == 0 and any(value != 0 for value in (amount, fee, total)):
            raise ValueError("zero quantity requires zero cost")
        if quantity > 0 and amount <= 0:
            raise ValueError("positive quantity requires positive trade amount")
        if total != amount + fee:
            raise ValueError("total cash required is inconsistent")


# 要求价格等输入为有限正 Decimal。
def _positive_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return value


# 判断价格是否对齐证券最小报价单位。
def _is_tick_aligned(value: Decimal, tick_size: Decimal) -> bool:
    return value / tick_size == (value / tick_size).to_integral_value()


class SlippageModel:
    """沿订单不利方向调整原始收盘价，并限制在法定日价格边界内。"""

    __slots__ = ("_rate",)

    # 校验并保存按买卖方向施加的滑点比例。
    def __init__(self, config: SlippageConfig) -> None:
        if not isinstance(config, SlippageConfig):
            raise TypeError("config must be SlippageConfig")
        self._rate = config.rate

    # 返回当前滑点比例。
    @property
    def rate(self) -> Decimal:
        return self._rate

    def apply(
        self,
        *,
        base_trade_price: Decimal,
        side: OrderSide,
        tick_size: Decimal,
        price_limit_down: Decimal,
        price_limit_up: Decimal,
    ) -> Decimal:
        """返回只受法定价格边界限制的保守最小变动单位价格。

        本接口有意不接收原始行情的最高价和最低价；二者只是观察值，并非法定执行边界，
        不能用来限制按收盘价比例计算的滑点。
        """

        price = _positive_decimal(base_trade_price, "base_trade_price")
        tick = _positive_decimal(tick_size, "tick_size")
        lower = _positive_decimal(price_limit_down, "price_limit_down")
        upper = _positive_decimal(price_limit_up, "price_limit_up")
        if not isinstance(side, OrderSide):
            raise TypeError("side must be OrderSide")
        if not lower <= price <= upper:
            raise ValueError("base_trade_price must be inside the legal price limits")
        if not _is_tick_aligned(lower, tick) or not _is_tick_aligned(upper, tick):
            raise ValueError("legal price limits must be tick aligned")

        if side is OrderSide.BUY:
            adjusted = price * (Decimal("1") + self._rate)
            rounded = (adjusted / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            return min(rounded, upper)

        adjusted = price * (Decimal("1") - self._rate)
        rounded = (adjusted / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        return max(_positive_decimal(rounded, "fill_price"), lower)


class FillModel:
    """只计算一次滑点，随后所有数量计算复用该价格。"""

    __slots__ = ("_fee_model", "_slippage_model")

    # 组合费用和滑点模型，使审批测算与最终成交使用同一套成本规则。
    def __init__(self, *, fee_model: FeeModel, slippage_model: SlippageModel) -> None:
        if not isinstance(fee_model, FeeModel):
            raise TypeError("fee_model must be FeeModel")
        if not isinstance(slippage_model, SlippageModel):
            raise TypeError("slippage_model must be SlippageModel")
        self._fee_model = fee_model
        self._slippage_model = slippage_model

    # 由原始报价、方向和交易规则计算固定的成交价格估计，供后续复用。
    def create_estimate(
        self,
        *,
        order: Order,
        quote: TradePriceQuote,
        tick_size: Decimal,
    ) -> ExecutionEstimate:
        if not isinstance(order, Order):
            raise TypeError("order must be Order")
        if not isinstance(quote, TradePriceQuote):
            raise TypeError("quote must be TradePriceQuote")
        if order.symbol != quote.symbol or order.execution_date != quote.trade_date:
            raise ValueError("quote identity does not match order symbol and execution date")
        fill_price = self._slippage_model.apply(
            base_trade_price=quote.base_trade_price,
            side=order.side,
            tick_size=tick_size,
            price_limit_down=quote.price_limit_down,
            price_limit_up=quote.price_limit_up,
        )
        cost = self._cost(side=order.side, fill_price=fill_price, quantity=order.requested_quantity)
        return ExecutionEstimate(
            order_id=order.order_id,
            requested_quantity=order.requested_quantity,
            base_trade_price=quote.base_trade_price,
            fill_price=fill_price,
            estimated_trade_amount=cost.trade_amount,
            estimated_fee=cost.fee,
            estimated_total_cash_required=cost.total_cash_required,
        )

    # 按成交估计和数量计算金额与费用，不修改账户。
    def estimate_cost(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        quantity: int,
    ) -> ExecutionCost:
        self._validate_estimate(order=order, estimate=estimate)
        requested = _quantity(quantity, "quantity")
        if requested > order.requested_quantity:
            raise ValueError("quantity exceeds the original order request")
        return self._cost(side=order.side, fill_price=estimate.fill_price, quantity=requested)

    # 在整手约束下搜索现金足以覆盖成交金额及费用的最大买入数量。
    def max_affordable_buy_quantity(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        available_cash: Decimal,
        lot_size: int,
        upper_quantity: int | None = None,
    ) -> int:
        self._validate_estimate(order=order, estimate=estimate)
        if order.side is not OrderSide.BUY:
            raise ValueError("affordability search is only defined for BUY")
        cash = _money(available_cash, "available_cash")
        lot = _quantity(lot_size, "lot_size")
        if lot <= 0:
            raise ValueError("lot_size must be positive")
        upper = (
            order.requested_quantity
            if upper_quantity is None
            else _quantity(upper_quantity, "upper_quantity")
        )
        if upper > order.requested_quantity:
            raise ValueError("upper_quantity exceeds the original order request")
        upper = (upper // lot) * lot

        low_lots = 0
        high_lots = upper // lot
        while low_lots < high_lots:
            middle = (low_lots + high_lots + 1) // 2
            candidate = middle * lot
            cost = self.estimate_cost(order=order, estimate=estimate, quantity=candidate)
            if cost.total_cash_required <= cash:
                low_lots = middle
            else:
                high_lots = middle - 1
        return low_lots * lot

    # 把订单审批结果转换为成交结果，复核执行估计并计算正式成交费用。
    def create_fill(
        self,
        *,
        order: Order,
        quote: TradePriceQuote,
        estimate: ExecutionEstimate,
        approval: RuleCheckResult,
    ) -> FillResult | None:
        if not isinstance(approval, RuleCheckResult):
            raise TypeError("approval must be RuleCheckResult")
        self._validate_estimate(order=order, estimate=estimate)
        if approval.order_id != order.order_id:
            raise ValueError("approval identity does not match order")
        if not approval.passed:
            return None
        cost = self.estimate_cost(
            order=order,
            estimate=estimate,
            quantity=approval.approved_quantity,
        )
        return FillResult.from_approved(
            order=order,
            quote=quote,
            estimate=estimate,
            approval=approval,
            trade_amount=cost.trade_amount,
            fee=cost.fee,
        )

    # 汇总指定价格和数量的成交金额、费用与现金变动。
    def _cost(self, *, side: OrderSide, fill_price: Decimal, quantity: int) -> ExecutionCost:
        requested = _quantity(quantity, "quantity")
        amount = fill_price * requested
        fee = self._fee_model.calculate(trade_amount=amount, side=side).total
        return ExecutionCost(
            quantity=requested,
            trade_amount=amount,
            fee=fee,
            total_cash_required=amount + fee,
        )

    # 要求输入为订单和成交估计对象，并核对两者的订单 ID 与请求数量一致。
    @staticmethod
    def _validate_estimate(*, order: Order, estimate: ExecutionEstimate) -> None:
        if not isinstance(order, Order):
            raise TypeError("order must be Order")
        if not isinstance(estimate, ExecutionEstimate):
            raise TypeError("estimate must be ExecutionEstimate")
        if (
            order.order_id != estimate.order_id
            or order.requested_quantity != estimate.requested_quantity
        ):
            raise ValueError("execution estimate identity does not match order")


__all__ = ["ExecutionCost", "FillModel", "SlippageModel"]
