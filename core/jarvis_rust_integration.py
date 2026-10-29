"""Opt-in Rust acceleration for bounded, pure indicator calculations.

The live candle contract remains owned by :mod:`direct_candle_cache`: only the
500 completed native-interval candles are accepted here.  Rust is never used
for exchange I/O, websocket routing, order placement, sizing, or safety gates.

Set ``JARVIS_RUST_MATH=1`` to request the Rust SMA implementation.  If the
optional extension is unavailable, this module logs one explicit fallback and
uses the equivalent Python calculation.  Invalid or missing data is an error,
not a fallback condition.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Iterable

logger = logging.getLogger(__name__)
_ENABLED = os.getenv("JARVIS_RUST_MATH", "0").strip().lower() in {"1", "true", "yes", "on"}
_warned_unavailable = False


def _values(values: Iterable[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be > 0")
    result = [float(value) for value in values]
    if len(result) < period:
        raise ValueError(f"need at least {period} values, got {len(result)}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("values must contain only finite numbers")
    return result


def calculate_sma(values: Iterable[float], period: int) -> float:
    """Return a latest-window SMA with a semantically identical fallback."""
    data = _values(values, period)
    if _ENABLED:
        try:
            import jarvis_rust
            if not callable(getattr(jarvis_rust, "calculate_sma", None)):
                raise ImportError("jarvis_rust native SMA extension unavailable")

        except (ImportError, ModuleNotFoundError):
            global _warned_unavailable
            if not _warned_unavailable:
                logger.warning("JARVIS_RUST_MATH=1 but jarvis_rust is unavailable; using Python SMA fallback")
                _warned_unavailable = True
        else:
            result = float(jarvis_rust.calculate_sma(data, period))
            if not math.isfinite(result):
                raise ValueError("Rust SMA returned a non-finite result")
            return result
    return sum(data[-period:]) / period


def rust_math_enabled() -> bool:
    """Expose the startup-time feature flag for diagnostics and tests."""
    return _ENABLED
