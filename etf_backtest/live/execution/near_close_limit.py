"""根据单个临近收盘报价选择估值价格和法定价格边界。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import LiveQuote


# 携带尾盘执行的估值价和限价；估值使用最新价，委托价由盘口与偏移计算。
@dataclass(frozen=True, slots=True)
class NearClosePrice:
    valuation_price: Decimal
    limit_price: Decimal


# 判断可选价格为有限正数，供报价可用性检查。
def _is_positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0


class NearCloseLimitPolicy:
    """生成不区分方向的估值价格和区分买卖方向的订单价格。"""

    # 拒绝过期、未来或停牌报价；买单用卖一价加偏移、卖单用买一价减偏移，缺盘口时回退最新价并约束于涨跌停。
    def calculate(
        self,
        *,
        side: OrderSide,
        quote: LiveQuote,
        tick_size: Decimal,
        price_offset_ticks: int,
        now: datetime,
        quote_stale_seconds: int,
    ) -> NearClosePrice | None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        if quote_stale_seconds < 0 or price_offset_ticks < 0:
            raise ValueError("staleness and price offset must be non-negative")
        age = (now - quote.quoted_at).total_seconds()
        if age < 0 or age > quote_stale_seconds or quote.suspended:
            return None
        if not _is_positive(quote.last_price) or not _is_positive(tick_size):
            return None
        if not _is_positive(quote.lower_limit) or not _is_positive(quote.upper_limit):
            return None
        assert quote.lower_limit is not None
        assert quote.upper_limit is not None
        if quote.lower_limit > quote.upper_limit:
            return None

        offset = tick_size * price_offset_ticks
        if side is OrderSide.BUY:
            base = quote.ask1 if _is_positive(quote.ask1) else quote.last_price
            assert base is not None
            raw_price = min(base + offset, quote.upper_limit)
            limit_price = (raw_price / tick_size).to_integral_value(
                rounding=ROUND_CEILING
            ) * tick_size
        else:
            base = quote.bid1 if _is_positive(quote.bid1) else quote.last_price
            assert base is not None
            raw_price = max(base - offset, quote.lower_limit)
            limit_price = (raw_price / tick_size).to_integral_value(
                rounding=ROUND_FLOOR
            ) * tick_size
        if not _is_positive(limit_price):
            return None
        if not quote.lower_limit <= limit_price <= quote.upper_limit:
            return None
        return NearClosePrice(valuation_price=quote.last_price, limit_price=limit_price)


__all__ = ["NearCloseLimitPolicy", "NearClosePrice"]
