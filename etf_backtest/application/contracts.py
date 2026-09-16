"""单次日频策略决策的状态和结果。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from etf_backtest.core.target import TargetPortfolio


class DecisionStatus(StrEnum):
    """策略日期评估的三种结果：未到计划日、不调仓或已生成目标。"""

    NOT_SCHEDULED = "NOT_SCHEDULED"
    NO_REBALANCE = "NO_REBALANCE"
    TARGET_CREATED = "TARGET_CREATED"


@dataclass(frozen=True, slots=True)
class DailyDecisionResult:
    """某个信号日的调度结果，以及可选的非空显式调仓目标。"""

    signal_date: date
    execution_date: date
    schedule_index: int
    status: DecisionStatus
    target_portfolio: TargetPortfolio | None

    # 校验调度序号非负，且只有 TARGET_CREATED 状态可以携带 TargetPortfolio。
    def __post_init__(self) -> None:
        if type(self.schedule_index) is not int or self.schedule_index < 0:
            raise ValueError("schedule_index must be a non-negative integer")
        if not isinstance(self.status, DecisionStatus):
            raise TypeError("status must be DecisionStatus")
        if self.status is DecisionStatus.TARGET_CREATED:
            from etf_backtest.core.target import TargetPortfolio

            if not isinstance(self.target_portfolio, TargetPortfolio):
                raise TypeError("TARGET_CREATED requires a TargetPortfolio")
        elif self.target_portfolio is not None:
            raise ValueError("only TARGET_CREATED may contain a target portfolio")


__all__ = ["DailyDecisionResult", "DecisionStatus"]
