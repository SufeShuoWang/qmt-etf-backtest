"""各模块共用的基础值检查；交易规则仍由业务模块负责。"""
from datetime import date, datetime


# 检查输入为纯 date，拒绝把带时分秒的 datetime 当作交易日。
def plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


# 检查文本字段非空并按公共约定规范其内容。
def non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


# 要求数量是严格非负整数，拒绝布尔值和负数。
def quantity(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value

