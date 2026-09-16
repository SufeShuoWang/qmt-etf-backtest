"""原始沪深券商事实的证券代码规范化，不依赖策略支持范围。"""

from __future__ import annotations

import re

_BROKER_SYMBOL = re.compile(r"^(?:(SH|SZ)\.)?(\d{6})(?:\.(SH|SZ))?$")


def normalize_broker_symbol(value: str) -> str:
    """规范化普通六位沪深证券代码，不对资产分类。"""

    if not isinstance(value, str):
        raise TypeError("broker symbol must be a string")
    normalized = value.strip().upper()
    match = _BROKER_SYMBOL.fullmatch(normalized)
    if match is None:
        raise ValueError("broker symbol must be a six-digit SH/SZ security code")
    leading_exchange, code, trailing_exchange = match.groups()
    if leading_exchange and trailing_exchange:
        raise ValueError("broker symbol cannot contain two exchange qualifiers")
    supplied = leading_exchange or trailing_exchange
    if supplied is not None:
        return f"{supplied}.{code}"
    inferred = (
        "SH"
        if code.startswith(("5", "6", "9"))
        else "SZ"
        if code.startswith(("0", "1", "2", "3"))
        else None
    )
    if inferred is None:
        raise ValueError("broker symbol is not an ordinary SH/SZ security code")
    return f"{inferred}.{code}"


__all__ = ["normalize_broker_symbol"]
