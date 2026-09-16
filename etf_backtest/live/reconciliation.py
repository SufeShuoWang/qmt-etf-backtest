"""将已规范化券商事实与持久化实盘状态进行对账。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal

from sqlalchemy.engine import Connection

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerTradeSnapshot,
    ReconciliationReport,
)

_LOCAL_REMARK = re.compile(r"^L[A-Z2-7]{20}$")


# 检查备注是否符合本程序生成的本地令牌格式。
def is_local_remark_token(value: str | None) -> bool:
    return bool(value and _LOCAL_REMARK.fullmatch(value))


# 提供当前模拟盘默认周转映射：SH.518880 为 T+0，其他证券为 T+1；调用方可提供具体映射。
def default_turnover_rule(symbol: str) -> TurnoverRule:
    return TurnoverRule.T0 if normalize_symbol(symbol) == "SH.518880" else TurnoverRule.T1


def order_terms_match(order: BrokerOrderSnapshot, intent: Mapping[str, object]) -> bool:
    """回报与主动对账共用的数量、限价和本地备注核验。"""
    return (
        int(str(intent["requested_quantity"])) == order.requested_quantity
        and Decimal(str(intent["limit_price"])) == order.limit_price
        and str(intent["remark_token"]) == str(order.remark_token)
    )


# 把券商查询得到的订单／成交与本地意图核对、关联和入账，汇总未解决及未完成项。
class ReconciliationService:
    # 绑定状态仓库和周转规则解析方式。
    def __init__(self, turnover_rules: Mapping[str, TurnoverRule] | None = None) -> None:
        self._turnover_rules = {
            normalize_symbol(symbol): rule for symbol, rule in (turnover_rules or {}).items()
        }

    # 取得指定证券的周转规则，供成交更新可卖数量。
    def _turnover_rule(self, symbol: str) -> TurnoverRule:
        return self._turnover_rules.get(symbol, default_turnover_rule(symbol))

    # 执行主动对账：核对本地关联、更新订单、幂等应用成交，再报告未知或不完整的执行状态。
    def reconcile(
        self,
        *,
        account_id: str,
        broker_orders: Sequence[BrokerOrderSnapshot],
        broker_trades: Sequence[BrokerTradeSnapshot],
        repository: LiveStateRepository,
        connection: Connection | None = None,
    ) -> ReconciliationReport:
        matched_orders = 0
        inserted_trades = 0
        unknown_orders: list[str] = []
        unknown_trades: list[str] = []
        active_orders: list[str] = []
        incomplete_intents: list[str] = []
        quantity_mismatches: list[str] = []
        identity_mismatches: list[str] = []
        trade_identity_mismatches: list[str] = []
        unknown_statuses: list[str] = []
        matched: dict[str, tuple[BrokerOrderSnapshot, str]] = {}
        accepted_trades: list[BrokerTradeSnapshot] = []
        cash_breaches: list[str] = []

        with repository.transaction(connection) as active:
            for order in broker_orders:
                if order.account_id not in {None, account_id}:
                    continue
                saved_order = repository.get_broker_order(
                    account_id, order.broker_order_id, connection=active
                )
                intent = None
                if saved_order is not None:
                    intent_id = str(saved_order["intent_id"])
                    remark_token = str(saved_order["remark_token"])
                elif is_local_remark_token(order.remark_token):
                    assert order.remark_token is not None
                    intent = repository.get_intent_by_remark_token(
                        order.remark_token, account_id=account_id, connection=active
                    )
                    if intent is None:
                        unknown_orders.append(order.broker_order_id)
                        continue
                    intent_id = str(intent["intent_id"])
                    remark_token = order.remark_token
                else:
                    continue
                intent_row = repository.get_intent(intent_id, connection=active)
                if intent_row is None or not self._order_matches_intent(order, intent_row):
                    identity_mismatches.append(order.broker_order_id)
                    continue
                repository.bind_broker_order(
                    account_id=account_id,
                    intent_id=intent_id,
                    remark_token=remark_token,
                    order=order,
                    connection=active,
                )
                matched_orders += 1
                matched[order.broker_order_id] = (order, intent_id)

            for trade in broker_trades:
                if trade.account_id not in {None, account_id}:
                    continue
                saved_order = repository.get_broker_order(
                    account_id, trade.broker_order_id, connection=active
                )
                if saved_order is None:
                    if is_local_remark_token(trade.remark_token):
                        assert trade.remark_token is not None
                        intent_row = repository.get_intent_by_remark_token(
                            trade.remark_token,
                            account_id=account_id,
                            connection=active,
                        )
                        if intent_row is None:
                            unknown_trades.append(trade.broker_trade_id)
                            continue
                        intent_id = str(intent_row["intent_id"])
                    else:
                        continue
                else:
                    intent_id = str(saved_order["intent_id"])
                    intent_row = repository.get_intent(intent_id, connection=active)
                if intent_row is None or not self._trade_matches_intent(trade, intent_row):
                    trade_identity_mismatches.append(trade.broker_trade_id)
                    continue
                applied = repository.record_strategy_trade_if_absent(
                    account_id=account_id,
                    intent_id=intent_id,
                    trade=trade,
                    turnover_rule=self._turnover_rule(trade.symbol),
                    connection=active,
                )
                if applied.inserted:
                    inserted_trades += 1
                if applied.cash_breach:
                    cash_breaches.append(str(intent_row["strategy_id"]))
                accepted_trades.append(trade)

            traded_by_order: dict[str, int] = {}
            for trade in accepted_trades:
                traded_by_order[trade.broker_order_id] = (
                    traded_by_order.get(trade.broker_order_id, 0) + trade.quantity
                )
            for broker_order_id, (order, intent_id) in matched.items():
                traded_quantity = traded_by_order.get(broker_order_id, 0)
                if traded_quantity != order.filled_quantity:
                    quantity_mismatches.append(broker_order_id)
                    continue
                if order.status is BrokerOrderStatus.UNKNOWN:
                    unknown_statuses.append(broker_order_id)
                    continue
                if order.status.is_active:
                    active_orders.append(broker_order_id)
                elif (
                    order.status is BrokerOrderStatus.FILLED
                    and order.filled_quantity == order.requested_quantity
                ):
                    repository.mark_intent_completed(intent_id, connection=active)
                else:
                    repository.mark_intent_incomplete(
                        intent_id,
                        f"BROKER_TERMINAL_{order.status.value}",
                        connection=active,
                    )
                    incomplete_intents.append(intent_id)

            unresolved = repository.list_unresolved_intents(
                account_id=account_id, connection=active
            )

        return ReconciliationReport(
            matched_order_count=matched_orders,
            inserted_trade_count=inserted_trades,
            unresolved_intent_ids=tuple(sorted(str(row["intent_id"]) for row in unresolved)),
            active_broker_order_ids=tuple(sorted(set(active_orders))),
            incomplete_intent_ids=tuple(sorted(set(incomplete_intents))),
            order_trade_mismatch_ids=tuple(sorted(set(quantity_mismatches))),
            order_identity_mismatch_ids=tuple(sorted(set(identity_mismatches))),
            trade_identity_mismatch_ids=tuple(sorted(set(trade_identity_mismatches))),
            unknown_order_status_ids=tuple(sorted(set(unknown_statuses))),
            unknown_broker_order_ids=tuple(sorted(set(unknown_orders))),
            unknown_broker_trade_ids=tuple(sorted(set(unknown_trades))),
            virtual_cash_breach_strategy_ids=tuple(sorted(set(cash_breaches))),
        )

    # 比较券商订单与本地意图的交易身份及数量／价格约束。
    @staticmethod
    def _order_matches_intent(order: BrokerOrderSnapshot, intent: dict[str, object]) -> bool:
        return (
            str(intent["symbol"]) == order.symbol
            and str(intent["side"]) == order.side.value
            and order_terms_match(order, intent)
        )

    # 对已经关联到意图的成交复核证券和买卖方向。
    @staticmethod
    def _trade_matches_intent(trade: BrokerTradeSnapshot, intent: dict[str, object]) -> bool:
        return str(intent["symbol"]) == trade.symbol and str(intent["side"]) == trade.side.value


__all__ = [
    "ReconciliationService",
    "default_turnover_rule",
    "is_local_remark_token",
    "order_terms_match",
]
