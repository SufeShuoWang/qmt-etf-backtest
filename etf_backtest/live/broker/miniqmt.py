"""既有 BrokerGateway 协议的同步 MiniQMT 实现。"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from types import ModuleType
from typing import Any, TypeVar

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.live.broker.callbacks import BrokerEvent, create_xtquant_callback
from etf_backtest.live.broker.mapper import (
    internal_to_external_symbol,
    map_asset,
    map_order,
    map_position,
    map_submit_result,
    map_trade,
    side_to_xt,
)
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    OrderIntent,
    QueryResult,
    SubmitOrderResult,
    SubmitOrderStatus,
)

RecordT = TypeVar("RecordT")

_SESSION_ID_LOCK = Lock()
_LAST_SESSION_ID = 0


def _fresh_session_id() -> int:
    """返回当前进程内唯一的基于时间的 session id。"""

    global _LAST_SESSION_ID
    with _SESSION_ID_LOCK:
        session_id = max(int(time.time()), _LAST_SESSION_ID + 1)
        _LAST_SESSION_ID = session_id
        return session_id


# 按需导入 xtquant 交易、账户与常量模块；未安装时给出运行错误。
def _load_xtquant() -> tuple[ModuleType, ModuleType, ModuleType]:
    try:
        return (
            import_module("xtquant.xttrader"),
            import_module("xtquant.xttype"),
            import_module("xtquant.xtconstant"),
        )
    except ImportError as error:
        raise RuntimeError("当前环境未安装 xtquant, 无法创建交易运行时。") from error


# 用 xtquant 实现券商网关，把 SDK 会话和原始对象转换成应用接口与状态对象。
class MiniQmtBrokerGateway:
    # 保存 MiniQMT 目录、账户和回调队列，加载 SDK；实际连接由 connect() 建立。
    def __init__(
        self,
        *,
        userdata_path: Path,
        session_id: int | None = None,
        account_id: str,
        event_queue: Queue[BrokerEvent],
        account_type: str = "STOCK",
        strategy_name: str = "qmt-etf-live",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        xttrader, xttype, constants = _load_xtquant()
        self._userdata_path = Path(userdata_path)
        self._session_id_override = session_id
        self._account_id = account_id
        self._account_type = account_type
        self._strategy_name = strategy_name
        self._events = event_queue
        self._clock = clock or (lambda: datetime.now(MARKET_TIMEZONE))
        self._xttrader = xttrader
        self._xttype = xttype
        self._constants = constants
        self._trader: Any | None = None
        self._account: Any | None = None
        self._subscribed = False
        self._trading_enabled = False
        self._expected_disconnect: Event | None = None

    # 创建交易会话、注册回调并启动 SDK，有限次重试连接，失败时清理会话。
    def connect(self) -> None:
        if not self._userdata_path.is_dir():
            raise FileNotFoundError(f"userdata_mini directory is missing: {self._userdata_path}")
        session_id = (
            self._session_id_override
            if self._session_id_override is not None
            else _fresh_session_id()
        )
        trader = self._xttrader.XtQuantTrader(str(self._userdata_path), session_id)
        expected_disconnect = Event()
        trader.register_callback(
            create_xtquant_callback(self._events, expected_disconnect=expected_disconnect)
        )
        trader.start()
        try:
            result = -1
            for attempt in range(1, 6):
                result = trader.connect()
                if result == 0:
                    break
                if attempt < 5:
                    time.sleep(1.0)
        except Exception:
            expected_disconnect.set()
            trader.stop()
            raise
        if result != 0:
            expected_disconnect.set()
            trader.stop()
            raise RuntimeError(f"MiniQMT connect failed after 5 attempts: {result}")
        try:
            account = self._xttype.StockAccount(self._account_id, self._account_type)
        except Exception:
            expected_disconnect.set()
            trader.stop()
            raise
        self._trader = trader
        self._account = account
        self._expected_disconnect = expected_disconnect

    # 订阅当前账户并检查 SDK 返回状态，成功后启用委托通道。
    def subscribe_account(self, account_id: str) -> None:
        if account_id != self._account_id:
            raise ValueError("subscribed account does not match configured account_id")
        trader, account = self._connected()
        result = trader.subscribe(account)
        if result != 0:
            raise RuntimeError(f"MiniQMT account subscribe failed: {result}")
        self._subscribed = True
        self._trading_enabled = True

    # 标记主动断连，停止 SDK 会话并清空本地连接状态。
    def disconnect(self) -> None:
        trader, account = self._trader, self._account
        self._trading_enabled = False
        if self._expected_disconnect is not None:
            self._expected_disconnect.set()
        try:
            if trader is not None and account is not None and self._subscribed:
                result = trader.unsubscribe(account)
                if result not in {None, 0}:
                    raise RuntimeError(f"MiniQMT account unsubscribe failed: {result}")
        finally:
            try:
                if trader is not None:
                    trader.stop()
            finally:
                self._subscribed = False
                self._trader = None
                self._account = None
                self._expected_disconnect = None

    # 关闭本地委托开关，阻止后续新订单提交。
    def disable_trading(self) -> None:
        self._trading_enabled = False

    # 查询并映射账户资产；查询异常保留为失败结果。
    def query_asset(self) -> QueryResult[BrokerAssetSnapshot]:
        try:
            trader, account = self._connected()
            value = trader.query_stock_asset(account)
            if value is None:
                return QueryResult(success=False, error="MiniQMT asset query returned None")
            return QueryResult(success=True, records=(map_asset(value, captured_at=self._clock()),))
        except Exception as error:
            return QueryResult(success=False, error=str(error))

    # 查询并映射证券持仓，向调用方返回标准快照序列。
    def query_positions(self) -> QueryResult[BrokerPositionSnapshot]:
        return self._query_list("positions", "query_stock_positions", map_position)

    # 查询并映射券商订单，供回调补偿和主动对账。
    def query_orders(self) -> QueryResult[BrokerOrderSnapshot]:
        return self._query_list(
            "orders",
            "query_stock_orders",
            lambda value, **_: map_order(value, constants=self._constants),
        )

    # 查询并映射券商成交，供幂等账本更新。
    def query_trades(self) -> QueryResult[BrokerTradeSnapshot]:
        return self._query_list(
            "trades",
            "query_stock_trades",
            lambda value, **_: map_trade(value, constants=self._constants),
        )

    # 检查连接和交易开关后，以固定限价提交订单意图并映射受理结果。
    def submit_order(self, intent: OrderIntent) -> SubmitOrderResult:
        if not self._trading_enabled:
            return SubmitOrderResult(
                SubmitOrderStatus.UNKNOWN, error="MiniQMT trading is not enabled"
            )
        if len(intent.remark_token.encode("ascii")) > 24:
            return SubmitOrderResult(
                SubmitOrderStatus.REJECTED, error="order_remark exceeds 24 ASCII bytes"
            )
        try:
            trader, account = self._connected()
            order_id = trader.order_stock(
                account,
                internal_to_external_symbol(intent.symbol),
                side_to_xt(intent.side, self._constants),
                int(intent.requested_quantity),
                int(self._constants.FIX_PRICE),
                float(intent.limit_price),
                self._strategy_name,
                intent.remark_token,
            )
        except Exception as error:
            return SubmitOrderResult(SubmitOrderStatus.UNKNOWN, error=str(error))
        return map_submit_result(order_id)

    # 调用 SDK 撤销指定券商订单，返回请求是否成功。
    def cancel_order(self, broker_order_id: str) -> bool:
        if not self._trading_enabled:
            raise RuntimeError("MiniQMT trading is not enabled")
        trader, account = self._connected()
        try:
            result = trader.cancel_order_stock(account, int(broker_order_id))
        except Exception as error:
            raise RuntimeError(f"MiniQMT cancel failed: {error}") from error
        if result == 0:
            return True
        raise RuntimeError(f"MiniQMT cancel failed: {result}")

    # 要求交易会话与账户对象已建立，并返回这两个 SDK 对象。
    def _connected(self) -> tuple[Any, Any]:
        if self._trader is None or self._account is None:
            raise RuntimeError("MiniQMT broker is not connected")
        return self._trader, self._account

    # 统一执行券商列表查询和记录映射，区分失败与成功但无记录。
    def _query_list(
        self,
        label: str,
        method_name: str,
        mapper: Callable[..., RecordT],
    ) -> QueryResult[RecordT]:
        try:
            trader, account = self._connected()
            values = getattr(trader, method_name)(account)
            if values is None:
                return QueryResult(success=False, error=f"MiniQMT {label} query returned None")
            if not isinstance(values, (list, tuple)):
                return QueryResult(
                    success=False,
                    error=f"MiniQMT {label} query returned an invalid collection",
                )
            captured_at = self._clock()
            return QueryResult(
                success=True,
                records=tuple(mapper(value, captured_at=captured_at) for value in values),
            )
        except Exception as error:
            return QueryResult(success=False, error=str(error))


__all__ = ["MiniQmtBrokerGateway"]
