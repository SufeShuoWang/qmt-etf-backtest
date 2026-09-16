"""精简的 xtquant 回调到队列桥接层和单工作线程消费者。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from queue import Empty, Queue
from threading import Event, Thread

from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.live.broker.mapper import map_order, map_trade
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import (
    default_turnover_rule, is_local_remark_token, order_terms_match,
)
from etf_backtest.live.state import BrokerOrderSnapshot, BrokerTradeSnapshot, TradeApplyResult

LOGGER = logging.getLogger(__name__)


# 枚举断线、账户状态、订单、成交与错误等券商回调事件类别。
class BrokerEventType(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    ACCOUNT_STATUS = "ACCOUNT_STATUS"
    ORDER = "ORDER"
    TRADE = "TRADE"
    ORDER_ERROR = "ORDER_ERROR"
    CANCEL_ERROR = "CANCEL_ERROR"


# 封装回调事件类型、载荷和账户／错误信息，供消费线程统一处理。
@dataclass(frozen=True, slots=True)
class BrokerEvent:
    event_type: BrokerEventType
    payload: object | None = None
    account_id: str | None = None
    status: int | None = None
    error: str | None = None


# 将券商断线通知放入事件队列，交给消费者处理健康状态。
def enqueue_disconnected(events: Queue[BrokerEvent]) -> None:
    events.put(BrokerEvent(BrokerEventType.DISCONNECTED))


# 将账户状态变化放入事件队列。
def enqueue_account_status(events: Queue[BrokerEvent], status: object) -> None:
    events.put(
        BrokerEvent(
            BrokerEventType.ACCOUNT_STATUS,
            account_id=str(getattr(status, "account_id", "")) or None,
            status=int(getattr(status, "status", -1)),
        )
    )


# 将订单回报对象放入队列，避免在 SDK 回调线程直接写数据库。
def enqueue_order(events: Queue[BrokerEvent], order: BrokerOrderSnapshot) -> None:
    events.put(BrokerEvent(BrokerEventType.ORDER, payload=order, account_id=order.account_id))


# 将成交回报对象放入队列，交给消费者执行幂等入账。
def enqueue_trade(events: Queue[BrokerEvent], trade: BrokerTradeSnapshot) -> None:
    events.put(BrokerEvent(BrokerEventType.TRADE, payload=trade, account_id=trade.account_id))


# 将委托或撤单错误封装成事件入队。
def enqueue_error(events: Queue[BrokerEvent], event_type: BrokerEventType, error: object) -> None:
    events.put(
        BrokerEvent(
            event_type,
            account_id=str(getattr(error, "account_id", "")) or None,
            error=(
                f"order_id={getattr(error, 'order_id', '')} "
                f"error_id={getattr(error, 'error_id', '')} "
                f"error_msg={getattr(error, 'error_msg', '')}"
            ).strip(),
        )
    )


def create_xtquant_callback(
    events: Queue[BrokerEvent], *, expected_disconnect: Event | None = None
) -> object:
    """仅在实际构造生产回调时导入 SDK。"""

    try:
        module = import_module("xtquant.xttrader")
        constants = import_module("xtquant.xtconstant")
    except ImportError as error:
        raise RuntimeError("当前环境未安装 xtquant, 无法创建交易运行时。") from error

    # 实现 xtquant 回调接口，把 SDK 通知转交给应用事件队列。
    class QueueingCallback(module.XtQuantTraderCallback):  # type: ignore[name-defined,misc]
        # 处理 SDK 断线通知；主动断连时根据标记避免误报异常。
        def on_disconnected(self) -> None:
            if expected_disconnect is None or not expected_disconnect.is_set():
                enqueue_disconnected(events)

        # 把 SDK 账户状态通知转换为队列事件。
        def on_account_status(self, status: object) -> None:
            enqueue_account_status(events, status)

        # 把 SDK 订单状态回报转入队列。
        def on_stock_order(self, order: object) -> None:
            enqueue_order(events, map_order(order, constants=constants))

        # 把 SDK 成交回报转入队列。
        def on_stock_trade(self, trade: object) -> None:
            enqueue_trade(events, map_trade(trade, constants=constants))

        # 把 SDK 下单错误转入队列。
        def on_order_error(self, error: object) -> None:
            enqueue_error(events, BrokerEventType.ORDER_ERROR, error)

        # 把 SDK 撤单错误转入队列。
        def on_cancel_error(self, error: object) -> None:
            enqueue_error(events, BrokerEventType.CANCEL_ERROR, error)

    return QueueingCallback()


class BrokerEventConsumer:
    """幂等持久化单条回调；主动查询结果仍是权威事实来源。"""

    # 绑定事件队列、账户、状态仓库、费用模型和异常处理回调，准备消费线程。
    def __init__(
        self,
        *,
        events: Queue[BrokerEvent],
        repository: LiveStateRepository,
        account_id: str,
        on_unhealthy: Callable[[str], None],
        turnover_rules: Mapping[str, TurnoverRule] | None = None,
    ) -> None:
        self._events = events
        self._repository = repository
        self._account_id = account_id
        self._on_unhealthy = on_unhealthy
        self._turnover_rules = dict(turnover_rules or {})
        self._stop = Event()
        self._thread: Thread | None = None

    # 启动后台消费者线程，串行处理券商事件。
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="qmt-broker-events", daemon=True)
        self._thread.start()

    # 通知消费线程停止并等待退出。
    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # 按事件类型处理断线／账户异常、订单更新和成交入账，异常通过健康回调上报。
    def process(self, event: BrokerEvent) -> None:
        if event.event_type is BrokerEventType.DISCONNECTED:
            self._on_unhealthy("BROKER_DISCONNECTED")
            return
        if event.event_type is BrokerEventType.ACCOUNT_STATUS:
            if event.account_id not in {None, self._account_id} or event.status != 0:
                self._on_unhealthy(f"BROKER_ACCOUNT_STATUS_{event.status}")
            return
        if event.event_type in {BrokerEventType.ORDER_ERROR, BrokerEventType.CANCEL_ERROR}:
            LOGGER.error("MiniQMT %s: %s", event.event_type, event.error)
            self._on_unhealthy(f"BROKER_{event.event_type.value}")
            return
        if event.event_type is BrokerEventType.ORDER:
            self._persist_order(event.payload)
        elif event.event_type is BrokerEventType.TRADE:
            self._persist_trade(event.payload)

    # 持续从队列取出事件；处理失败时记录日志并将券商状态标为异常。
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.process(event)
            except Exception:
                LOGGER.exception("failed to consume MiniQMT callback event")
                self._on_unhealthy("BROKER_CALLBACK_PERSISTENCE_ERROR")

    # 映射并核对订单回报与本地意图，再持久化券商订单状态。
    def _persist_order(self, payload: object | None) -> None:
        if not isinstance(payload, BrokerOrderSnapshot):
            raise TypeError("ORDER event requires BrokerOrderSnapshot")
        account_id = payload.account_id or self._account_id
        with self._repository.transaction() as connection:
            saved = self._repository.get_broker_order(
                account_id, payload.broker_order_id, connection=connection
            )
            if saved is not None:
                intent_id = str(saved["intent_id"])
                remark = str(saved["remark_token"])
            elif is_local_remark_token(payload.remark_token):
                assert payload.remark_token is not None
                intent = self._repository.get_intent_by_remark_token(
                    payload.remark_token, account_id=account_id, connection=connection
                )
                if intent is None:
                    LOGGER.error("unknown local callback order: %s", payload.broker_order_id)
                    self._pause_for_unknown_local("UNKNOWN_LOCAL_CALLBACK_ORDER")
                    return
                intent_id = str(intent["intent_id"])
                remark = payload.remark_token
            else:
                LOGGER.debug("ignored external callback order: %s", payload.broker_order_id)
                return
            intent = self._repository.get_intent(intent_id, connection=connection)
            if intent is None or not _order_matches_intent(payload, intent):
                LOGGER.error("local callback order identity mismatch: %s", payload.broker_order_id)
                self._pause_for_unknown_local("LOCAL_CALLBACK_ORDER_IDENTITY_MISMATCH")
                return
            self._repository.bind_broker_order(
                account_id=account_id,
                intent_id=intent_id,
                remark_token=remark,
                order=payload,
                connection=connection,
            )

    # 映射并核对成交回报，调用仓库以成交标识去重并更新虚拟账户。
    def _persist_trade(self, payload: object | None) -> None:
        if not isinstance(payload, BrokerTradeSnapshot):
            raise TypeError("TRADE event requires BrokerTradeSnapshot")
        account_id = payload.account_id or self._account_id
        with self._repository.transaction() as connection:
            order = self._repository.get_broker_order(
                account_id, payload.broker_order_id, connection=connection
            )
            if order is None:
                if is_local_remark_token(payload.remark_token):
                    assert payload.remark_token is not None
                    intent = self._repository.get_intent_by_remark_token(
                        payload.remark_token,
                        account_id=account_id,
                        connection=connection,
                    )
                    if intent is None:
                        LOGGER.error("unknown local callback trade: %s", payload.broker_trade_id)
                        self._pause_for_unknown_local("UNKNOWN_LOCAL_CALLBACK_TRADE")
                        return
                    intent_id = str(intent["intent_id"])
                else:
                    LOGGER.debug("ignored external callback trade: %s", payload.broker_trade_id)
                    return
            else:
                intent_id = str(order["intent_id"])
                intent = self._repository.get_intent(intent_id, connection=connection)
            if intent is None or not _trade_matches_intent(payload, intent):
                LOGGER.error("local callback trade identity mismatch: %s", payload.broker_trade_id)
                self._pause_for_unknown_local("LOCAL_CALLBACK_TRADE_IDENTITY_MISMATCH")
                return
            applied = self._repository.record_strategy_trade_if_absent(
                account_id=account_id,
                intent_id=intent_id,
                trade=payload,
                turnover_rule=self._turnover_rules.get(
                    payload.symbol, default_turnover_rule(payload.symbol)
                ),
                connection=connection,
            )
        if isinstance(applied, TradeApplyResult) and applied.cash_breach:
            self._on_unhealthy("VIRTUAL_CASH_NEGATIVE_AFTER_FEES")

    # 遇到无法关联的本地回报时暂停已有账户，并通知引擎停止交易。
    def _pause_for_unknown_local(self, reason: str) -> None:
        account = self._repository.get_account(self._account_id)
        if account is not None:
            self._repository.pause_account(self._account_id, reason)
        self._on_unhealthy(reason)


# 检查已关联订单的证券、方向及委托数量／限价／备注条款是否符合本地意图。
def _order_matches_intent(order: BrokerOrderSnapshot, intent: Mapping[str, object]) -> bool:
    return (
        str(intent["symbol"]) == order.symbol
        and _intent_side(intent) is order.side
        and order_terms_match(order, intent)
    )


# 对已经关联到意图的成交复核证券和买卖方向。
def _trade_matches_intent(trade: BrokerTradeSnapshot, intent: Mapping[str, object]) -> bool:
    return str(intent["symbol"]) == trade.symbol and _intent_side(intent) is trade.side


# 把持久化意图中的方向字段还原为 OrderSide。
def _intent_side(intent: Mapping[str, object]) -> OrderSide:
    value = intent["side"]
    return value if isinstance(value, OrderSide) else OrderSide(str(value))


__all__ = [
    "BrokerEvent",
    "BrokerEventConsumer",
    "BrokerEventType",
    "create_xtquant_callback",
    "enqueue_account_status",
    "enqueue_disconnected",
    "enqueue_error",
    "enqueue_order",
    "enqueue_trade",
]
