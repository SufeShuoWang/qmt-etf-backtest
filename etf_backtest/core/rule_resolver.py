"""日频引擎所需的最小带生效日期规则边界。"""

from datetime import date
from typing import Protocol, runtime_checkable

from etf_backtest.core.market import EtfTradingRule


@runtime_checkable
class RuleResolver(Protocol):
    """解析对单个证券和交易日期生效的规则。"""

    # 定义按证券与日期取得交易规则的接口，由具体有效期解析器实现。
    def resolve(self, symbol: str, trade_date: date) -> EtfTradingRule: ...


__all__ = ["RuleResolver"]
