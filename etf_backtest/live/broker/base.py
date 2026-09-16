"""实盘任务使用的券商协议，业务层无需导入外部券商 SDK。"""

from __future__ import annotations

from typing import Protocol

from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    OrderIntent,
    QueryResult,
    SubmitOrderResult,
)


# 规定模拟盘券商连接、查询、下单和撤单接口，具体 SDK 由网关实现。
class BrokerGateway(Protocol):
    # 建立券商连接；具体会话创建逻辑由实现类提供。
    def connect(self) -> None: ...

    # 关闭券商连接并释放会话资源。
    def disconnect(self) -> None: ...

    # 订阅指定账户，准备接收订单与成交回报。
    def subscribe_account(self, account_id: str) -> None: ...

    # 查询券商资产快照，返回明确区分成功与失败的结果。
    def query_asset(self) -> QueryResult[BrokerAssetSnapshot]: ...

    # 查询券商当前持仓及可卖数量。
    def query_positions(self) -> QueryResult[BrokerPositionSnapshot]: ...

    # 查询券商订单快照，供恢复和对账使用。
    def query_orders(self) -> QueryResult[BrokerOrderSnapshot]: ...

    # 查询券商成交记录，供幂等入账使用。
    def query_trades(self) -> QueryResult[BrokerTradeSnapshot]: ...

    # 提交持久化订单意图，返回受理、拒绝或结果未知的状态。
    def submit_order(self, intent: OrderIntent) -> SubmitOrderResult: ...

    # 按券商订单标识发起撤单，并报告请求结果。
    def cancel_order(self, broker_order_id: str) -> bool: ...


__all__ = ["BrokerGateway"]
