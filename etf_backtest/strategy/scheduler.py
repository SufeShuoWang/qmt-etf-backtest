"""基于上交所行情帧的确定性决策调度。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PeriodicDecisionScheduler:
    """在第 0 个行情帧及其后每隔固定交易行情帧数执行决策。"""

    every_trading_days: int

    # 校验周期调度间隔为严格正整数。
    def __post_init__(self) -> None:
        if type(self.every_trading_days) is not int or self.every_trading_days <= 0:
            raise ValueError("every_trading_days must be a positive integer")

    # 当行情序号可被间隔整除时决策，因此序号 0 及之后每隔 N 帧满足条件。
    def should_decide(self, frame_index: int) -> bool:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        return frame_index % self.every_trading_days == 0


# 固定间隔为 1 的调度器，让每个有效行情帧都满足决策条件。
class EveryTradingDayScheduler(PeriodicDecisionScheduler):
    # 调用周期调度器构造函数，将间隔设为一个交易日。
    def __init__(self) -> None:
        super().__init__(every_trading_days=1)


__all__ = ["EveryTradingDayScheduler", "PeriodicDecisionScheduler"]
