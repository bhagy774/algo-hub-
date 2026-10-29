#!/usr/bin/env python3
"""
JARVIS Data Validator — Pre-Brain Data Quality Gate
====================================================
Ensures ONLY clean, fresh, sane data reaches the brain (JarvisElite).
Bad data → brain gets "WAIT/NO-DATA" status → no decision on garbage.

Checks (applied to every source every cycle):
  1. COMPLETENESS : OHLC values must all exist, no NaN/None/Inf
  2. PRICE SANITY : price > 0, and no spike > 20% vs last known price
  3. STALENESS    : data timestamp must be < 30 seconds old
  4. CROSS-SOURCE : Delta vs Binance price divergence
                   > 1% → WARNING (logged, trading continues)
                   > 3% → BLOCK  (this cycle returns NO-DATA)

Stats are tracked and exposed via get_stats() / status_line().

Toggle:
  JARVIS_DATA_VALIDATOR=0  →  disable entirely (all data passes through)
  Default: enabled (JARVIS_DATA_VALIDATOR != "0")

Integration:
  from jarvis_data_validator import get_validator
  v = get_validator()
  result = v.validate_candle(candle_dict, source="delta")
  result = v.validate_dataframe(df, source="binance")
  result = v.cross_source_check(delta_price, binance_price)
  # result.ok == True  → safe to use
  # result.ok == False → brain should WAIT

Safety guarantees:
  - FAIL-OPEN on internal error: if the validator itself crashes, data
    passes through (logged). The validator must never break the brain.
  - Thread-safe counters via threading.Lock.
  - Zero network calls. Pure validation logic only.
"""

import logging
import math
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

VALIDATOR_ENABLED = os.environ.get("JARVIS_DATA_VALIDATOR", "1") != "0"

# Price sanity
MAX_SPIKE_PCT = 20.0        # reject if price jumps > 20% from last known

# Staleness
# Candle data: a forming 1m candle can legitimately be up to ~60s old
# (open timestamp) plus fetch latency. 120s default still catches dead feeds.
MAX_STALE_SECONDS = float(os.getenv("JARVIS_MAX_STALE_SEC", "120"))

# Cross-source divergence
CROSS_SOURCE_WARN_PCT = 1.0   # > 1% divergence → warning
CROSS_SOURCE_BLOCK_PCT = 3.0  # > 3% divergence → block this cycle

# ANSI colors for status line
_Y = '\033[93m'; _G = '\033[92m'; _R = '\033[91m'
_DG = '\033[90m'; _W = '\033[97m'; _RST = '\033[0m'


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of a single validation pass."""
    ok: bool
    status: str              # "VALID", "WAIT", "NO-DATA"
    source: str = ""
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self):
        return self.ok


VALID = lambda src="": ValidationResult(ok=True, status="VALID", source=src)


# ── Core Validator ───────────────────────────────────────────────────────────

class JarvisDataValidator:
    """Pre-brain data quality gate. Pure validation, no network calls."""

    def __init__(self):
        self._enabled = VALIDATOR_ENABLED
        self._lock = threading.Lock()

        # Per-source last known good price (for spike detection)
        self._last_prices: Dict[str, float] = {}

        # Stats
        self._stats = {
            "total_checks": 0,
            "passed": 0,
            "rejected": 0,
            "warnings": 0,
            "rejects_by_reason": {
                "completeness": 0,
                "price_sanity": 0,
                "staleness": 0,
                "cross_source": 0,
            },
        }

    # ── Public API ───────────────────────────────────────────────────────

    def validate_candle(self, candle: Dict[str, Any],
                        source: str = "unknown",
                        now: Optional[float] = None) -> ValidationResult:
        """
        Validate a single candle dict.

        Expected keys: open, high, low, close (required)
                       time/timestamp (optional, for staleness)
                       volume (optional)

        Returns ValidationResult with ok=True (safe) or ok=False (WAIT/NO-DATA).
        """
        if not self._enabled:
            return VALID(source)

        try:
            now = now or time.time()
            failures: List[str] = []
            warnings: List[str] = []

            # 1. COMPLETENESS
            comp_fail = self._check_completeness(candle)
            if comp_fail:
                failures.extend(comp_fail)

            # 2. PRICE SANITY (only if completeness passed for close)
            if not comp_fail:
                price_fail = self._check_price_sanity(candle, source)
                if price_fail:
                    failures.extend(price_fail)

            # 3. STALENESS
            stale_fail = self._check_staleness(candle, now)
            if stale_fail:
                failures.extend(stale_fail)

            ok = len(failures) == 0
            status = "VALID" if ok else "WAIT"

            result = ValidationResult(
                ok=ok, status=status, source=source,
                failures=failures, warnings=warnings,
            )

            self._record(result)

            if not ok:
                logger.warning(
                    "[DATA-VALIDATOR] %s REJECTED from %s: %s",
                    status, source, "; ".join(failures)
                )
            return result

        except Exception as e:
            # FAIL-OPEN: validator crash must never block the brain
            logger.error("[DATA-VALIDATOR] Internal error (fail-open): %s", e)
            return VALID(source)

    def validate_dataframe(self, df, source: str = "unknown",
                           now: Optional[float] = None,
                           symbol: Optional[str] = None) -> ValidationResult:
        """
        Validate a pandas DataFrame with OHLCV columns.
        Checks the LAST row (most recent candle) for completeness/sanity/staleness.

        Returns ValidationResult.
        """
        if not self._enabled:
            return VALID(source)

        try:
            if df is None or len(df) == 0:
                result = ValidationResult(
                    ok=False, status="NO-DATA", source=source,
                    failures=["Empty or None DataFrame"],
                )
                self._record(result)
                logger.warning("[DATA-VALIDATOR] NO-DATA from %s: empty DataFrame", source)
                return result

            now = now or time.time()
            last_row = df.iloc[-1]

            # Build candle dict from DataFrame row
            candle: Dict[str, Any] = {}
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    val = last_row.get(col) if hasattr(last_row, 'get') else getattr(last_row, col, None)
                    candle[col] = val

            # --- FIX: inject symbol so price spike key is source:symbol not just source ---
            if symbol:
                candle["symbol"] = symbol
            elif "symbol" in df.columns:
                _sym = last_row.get("symbol") if hasattr(last_row, 'get') else getattr(last_row, "symbol", None)
                if _sym:
                    candle["symbol"] = str(_sym)

            # Try to extract timestamp from index or column
            ts = None
            if hasattr(df.index, 'dtype') and hasattr(last_row, 'name'):
                try:
                    import pandas as pd
                    if isinstance(df.index, pd.DatetimeIndex):
                        ts = last_row.name.timestamp()
                except Exception:
                    pass
            if ts is None:
                for ts_col in ("time", "timestamp", "date"):
                    if ts_col in df.columns:
                        raw = last_row.get(ts_col) if hasattr(last_row, 'get') else getattr(last_row, ts_col, None)
                        if raw is not None:
                            try:
                                ts = float(raw)
                            except (TypeError, ValueError):
                                try:
                                    import pandas as pd
                                    ts = pd.Timestamp(raw).timestamp()
                                except Exception:
                                    pass
                            break
            if ts is not None:
                # Frame holds CLOSED candles: index is the OPEN time, so the
                # latest closed bar opened one interval before it closed, and may
                # legitimately stay latest until the current forming bar closes
                # (up to 2 intervals from open time, matching DirectCandleCache).
                try:
                    if len(df) >= 2:
                        _idx = df.index
                        _step = (_idx[-1] - _idx[-2]).total_seconds()
                        if 0 < _step <= 40 * 86400:
                            candle['_interval_s'] = _step
                            candle['_closed_bar'] = True
                except Exception:
                    pass
                candle["time"] = ts

            return self.validate_candle(candle, source=source, now=now)

        except Exception as e:
            logger.error("[DATA-VALIDATOR] DataFrame validation error (fail-open): %s", e)
            return VALID(source)

    def cross_source_check(self, price_a: float, price_b: float,
                           source_a: str = "delta",
                           source_b: str = "binance") -> ValidationResult:
        """
        Cross-source price divergence check.

        > 1% → warning (trading continues)
        > 3% → block (NO-DATA for this cycle)

        Returns ValidationResult.
        """
        if not self._enabled:
            return VALID(f"{source_a}_vs_{source_b}")

        try:
            tag = f"{source_a}_vs_{source_b}"
            warnings: List[str] = []
            failures: List[str] = []

            # Skip if either price is invalid
            if not _is_valid_number(price_a) or price_a <= 0:
                return VALID(tag)  # Can't compare, pass through
            if not _is_valid_number(price_b) or price_b <= 0:
                return VALID(tag)  # Can't compare, pass through

            mid = (price_a + price_b) / 2.0
            divergence_pct = abs(price_a - price_b) / mid * 100.0

            if divergence_pct > CROSS_SOURCE_BLOCK_PCT:
                failures.append(
                    f"Cross-source BLOCK: {source_a}=${price_a:.2f} vs "
                    f"{source_b}=${price_b:.2f} ({divergence_pct:.2f}% > {CROSS_SOURCE_BLOCK_PCT}%)"
                )
            elif divergence_pct > CROSS_SOURCE_WARN_PCT:
                warnings.append(
                    f"Cross-source WARNING: {source_a}=${price_a:.2f} vs "
                    f"{source_b}=${price_b:.2f} ({divergence_pct:.2f}% > {CROSS_SOURCE_WARN_PCT}%)"
                )

            ok = len(failures) == 0
            status = "VALID" if ok else "NO-DATA"

            result = ValidationResult(
                ok=ok, status=status, source=tag,
                failures=failures, warnings=warnings,
            )

            # Record stats
            with self._lock:
                self._stats["total_checks"] += 1
                if ok:
                    self._stats["passed"] += 1
                else:
                    self._stats["rejected"] += 1
                    self._stats["rejects_by_reason"]["cross_source"] += 1
                if warnings:
                    self._stats["warnings"] += 1

            if warnings:
                for w in warnings:
                    logger.warning("[DATA-VALIDATOR] %s", w)
            if failures:
                for f in failures:
                    logger.warning("[DATA-VALIDATOR] %s", f)

            return result

        except Exception as e:
            logger.error("[DATA-VALIDATOR] Cross-source check error (fail-open): %s", e)
            return VALID(f"{source_a}_vs_{source_b}")

    def validate_multi_source(self, sources: Dict[str, Dict],
                              now: Optional[float] = None) -> ValidationResult:
        """
        Validate multiple source candles AND run cross-source checks.

        sources: {"delta": candle_dict, "binance": candle_dict, ...}

        Returns combined ValidationResult.
        """
        if not self._enabled:
            return VALID("multi")

        try:
            now = now or time.time()
            all_failures: List[str] = []
            all_warnings: List[str] = []

            prices: Dict[str, float] = {}

            for src_name, candle in sources.items():
                if candle is None:
                    continue
                r = self.validate_candle(candle, source=src_name, now=now)
                all_failures.extend(r.failures)
                all_warnings.extend(r.warnings)
                if r.ok and "close" in candle:
                    close_val = candle["close"]
                    if _is_valid_number(close_val) and float(close_val) > 0:
                        prices[src_name] = float(close_val)

            # Cross-source: Delta vs Binance
            if "delta" in prices and "binance" in prices:
                xr = self.cross_source_check(
                    prices["delta"], prices["binance"],
                    "delta", "binance"
                )
                all_failures.extend(xr.failures)
                all_warnings.extend(xr.warnings)

            ok = len(all_failures) == 0
            status = "VALID" if ok else "WAIT"

            return ValidationResult(
                ok=ok, status=status, source="multi",
                failures=all_failures, warnings=all_warnings,
            )

        except Exception as e:
            logger.error("[DATA-VALIDATOR] Multi-source validation error (fail-open): %s", e)
            return VALID("multi")

    # ── Stats & Status ───────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Thread-safe stats snapshot."""
        with self._lock:
            return dict(self._stats)

    def status_line(self) -> str:
        """One-line status for terminal display."""
        with self._lock:
            total = self._stats["total_checks"]
            passed = self._stats["passed"]
            rejected = self._stats["rejected"]
            warns = self._stats["warnings"]

        if not self._enabled:
            return f"  {_DG}DATA-VALIDATOR:{_RST} {_Y}DISABLED{_RST}"

        rej_col = _R if rejected > 0 else _G
        return (
            f"  {_DG}DATA-VALIDATOR:{_RST} {_G}ON{_RST} "
            f"{_DG}│{_RST} Checks: {_W}{total}{_RST} "
            f"{_DG}│{_RST} Pass: {_G}{passed}{_RST} "
            f"{_DG}│{_RST} Reject: {rej_col}{rejected}{_RST} "
            f"{_DG}│{_RST} Warn: {_Y}{warns}{_RST}"
        )

    def is_enabled(self) -> bool:
        return self._enabled

    def reset_stats(self):
        """Reset counters (useful for tests)."""
        with self._lock:
            self._stats = {
                "total_checks": 0,
                "passed": 0,
                "rejected": 0,
                "warnings": 0,
                "rejects_by_reason": {
                    "completeness": 0,
                    "price_sanity": 0,
                    "staleness": 0,
                    "cross_source": 0,
                },
            }
            self._last_prices.clear()

    # ── Internal checks ──────────────────────────────────────────────────

    def _check_completeness(self, candle: Dict[str, Any]) -> List[str]:
        """OHLC must all exist and be valid numbers (no NaN/None/Inf)."""
        failures = []
        for key in ("open", "high", "low", "close"):
            val = candle.get(key)
            if val is None:
                failures.append(f"Missing {key}")
            elif not _is_valid_number(val):
                failures.append(f"Invalid {key}={val} (NaN/Inf/non-numeric)")
        return failures

    def _check_price_sanity(self, candle: Dict[str, Any],
                            source: str) -> List[str]:
        """
        Price > 0 and no spike > MAX_SPIKE_PCT from last known price.
        Updates last known price on success.
        """
        failures = []
        close = float(candle["close"])

        # Zero / negative
        for key in ("open", "high", "low", "close"):
            val = float(candle[key])
            if val <= 0:
                failures.append(f"{key}={val} <= 0")

        if failures:
            return failures

        # --- FIX: include symbol in cache key so BTC price doesn't pollute NEAR etc ---
        symbol = candle.get("symbol", candle.get("instrument", ""))
        price_key = f"{source}:{symbol}" if symbol else source

        # Spike detection vs last known price
        with self._lock:
            last = self._last_prices.get(price_key)

        if last is not None and last > 0:
            change_pct = abs(close - last) / last * 100.0
            if change_pct > MAX_SPIKE_PCT:
                failures.append(
                    f"Price spike: {source} close={close:.2f} vs last={last:.2f} "
                    f"({change_pct:.1f}% > {MAX_SPIKE_PCT}%)"
                )
                # Do NOT update last_price on spike — keep the last good one
                return failures

        # Update last known good price
        with self._lock:
            self._last_prices[price_key] = close

        return failures

    def _check_staleness(self, candle: Dict[str, Any],
                         now: float) -> List[str]:
        """Data timestamp must be within MAX_STALE_SECONDS of now."""
        ts = candle.get("time", candle.get("timestamp"))
        if ts is None:
            # No timestamp in candle — can't check staleness, pass through
            return []

        try:
            ts_float = float(ts)
        except (TypeError, ValueError):
            return []  # unparseable timestamp — skip check, don't block

        # Handle millisecond timestamps
        if ts_float > 1e12:
            ts_float = ts_float / 1000.0

        age = now - ts_float
        _allow = MAX_STALE_SECONDS
        try:
            _iv = float(candle.get("_interval_s") or 0.0)
            if _iv > 0:
                _allow += (2.0 * _iv) if candle.get("_closed_bar") else _iv
        except (TypeError, ValueError):
            pass
        if age > _allow:
            return [f"Stale data: age={age:.1f}s > {_allow}s"]
        if age < -MAX_STALE_SECONDS:
            # Future timestamp — also suspicious
            return [f"Future timestamp: age={age:.1f}s (clock skew?)"]

        return []

    def _record(self, result: ValidationResult):
        """Thread-safe stats update."""
        with self._lock:
            self._stats["total_checks"] += 1
            if result.ok:
                self._stats["passed"] += 1
            else:
                self._stats["rejected"] += 1
                for f in result.failures:
                    fl = f.lower()
                    if "missing" in fl or "invalid" in fl or "nan" in fl:
                        self._stats["rejects_by_reason"]["completeness"] += 1
                    elif "spike" in fl or "<= 0" in fl:
                        self._stats["rejects_by_reason"]["price_sanity"] += 1
                    elif "stale" in fl or "future" in fl:
                        self._stats["rejects_by_reason"]["staleness"] += 1
                    elif "cross" in fl:
                        self._stats["rejects_by_reason"]["cross_source"] += 1
            if result.warnings:
                self._stats["warnings"] += 1


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_valid_number(val: Any) -> bool:
    """True if val is a finite number (not NaN, not Inf, not None)."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[JarvisDataValidator] = None
_instance_lock = threading.Lock()


def get_validator() -> JarvisDataValidator:
    """Thread-safe singleton accessor."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = JarvisDataValidator()
                if _instance._enabled:
                    logger.info("[DATA-VALIDATOR] ✅ Initialized — pre-brain data quality gate ONLINE")
                else:
                    logger.info("[DATA-VALIDATOR] ⚠️ DISABLED via JARVIS_DATA_VALIDATOR=0")
    return _instance
