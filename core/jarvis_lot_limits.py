"""Strict entry quantity policy; protective reduce-only exits are unconstrained.

The Delta product ``contract_size`` field describes the underlying amount in a
contract and is deliberately *not* treated as an order-lot increment.  Order
quantity is an integer contract count unless the venue metadata explicitly
provides an order-size increment.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Optional


# Explicit order-quantity metadata only.  In particular, ``contract_size`` is
# excluded: it is contract value/underlying amount, not order-lot granularity.
_ORDER_STEP_KEYS = (
    "order_size_increment",
    "size_increment",
    "order_size_step",
    "size_step",
    "lot_step",
    "contract_step",
    "quantity_step",
)


def _configured_positive_int(name: str) -> Optional[int]:
    if name not in os.environ:
        return None
    raw = os.environ[name]
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = int(raw, 10)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def entry_lot_limits() -> tuple[Optional[int], Optional[int]]:
    """Return configured (minimum, maximum); absent settings impose no bound."""
    minimum = _configured_positive_int("JARVIS_MIN_ENTRY_LOTS")
    maximum = _configured_positive_int("JARVIS_MAX_ENTRY_LOTS")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("JARVIS_MIN_ENTRY_LOTS must not exceed JARVIS_MAX_ENTRY_LOTS")
    return minimum, maximum


def _finite_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} is invalid") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _integer_step(value: object, label: str) -> int:
    number = _finite_positive_number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return int(number)


def _order_size_step(metadata: Optional[Mapping[str, object]]) -> Optional[int]:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("product metadata is invalid")
    for key in _ORDER_STEP_KEYS:
        if key in metadata and metadata[key] is not None:
            return _integer_step(metadata[key], f"{key} metadata")
    return None


def enforce_entry_lots(
    safe_qty: object,
    *,
    metadata: Optional[Mapping[str, object]] = None,
    available_balance: object = None,
    reduce_only: bool = False,
) -> int:
    """Apply configured entry limits without ever rounding upward.

    ``safe_qty`` is floored (never rounded up), capped at the configured max,
    and aligned downward to explicit order-size step metadata.  A quantity
    below the configured minimum is rejected, including after step alignment.
    ``reduce_only`` validates only that the requested quantity is positive
    and whole; entry limits, balance checks, and order-step metadata are
    irrelevant to a protective exit and are deliberately skipped.
    """
    qty_number = _finite_positive_number(safe_qty, "safe quantity")
    qty = int(math.floor(qty_number))
    if qty < 1:
        raise ValueError("safe quantity must contain at least one whole lot")
    if reduce_only:
        return qty

    if available_balance is not None:
        _finite_positive_number(available_balance, "available balance")

    step = _order_size_step(metadata)

    minimum, maximum = entry_lot_limits()
    if minimum is not None and qty < minimum:
        raise ValueError(f"risk-safe quantity {qty} is below minimum entry lots {minimum}")
    if maximum is not None:
        qty = min(qty, maximum)
    if step is not None:
        qty -= qty % step
        if qty < 1:
            raise ValueError("order-size increment leaves no positive entry quantity")
    if minimum is not None and qty < minimum:
        raise ValueError("order-size increment leaves entry below minimum")
    return qty
