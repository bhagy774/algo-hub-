"""Shared, fail-closed Part 7 volatility analysis.

Part 7 is an analysis consumer of the canonical native exchange candle snapshot;
it is not a candle builder.  Pandas validates native frames; bounded standard-
library quantitative math computes robust return distributions, state evidence,
and price-unit risk metrics on CPU. Optional neural GPU inference is separate.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from binance_timeframes import (
    BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_CORRELATION_GROUPS,
    TIMEFRAME_GROUP_WEIGHTS, next_candle_open, stale_deadline,
    validate_closed_candle_timestamps,
    validate_interval,
)

DEFAULT_TIMEFRAMES = BINANCE_SPOT_TIMEFRAMES
_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _symbol(value: Any) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").replace("_", "")


def _base_result(
    *,
    symbol: str,
    timeframe: str,
    signal: int = 0,
    status: str,
    reason: str,
    data_status: str,
    volatility_status: str = "unknown",
    entry_blocked: bool = False,
    risk_veto: bool = False,
    telemetry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    direction = "CALL" if signal > 0 else "PUT" if signal < 0 else "NEUTRAL"
    return {
        "signal": int(signal),
        "signal_identity": f"part7:{_symbol(symbol) or 'UNKNOWN'}:{timeframe}",
        "signal_name": direction,
        "symbol": _symbol(symbol) or "UNKNOWN",
        "timeframe": timeframe,
        "status": status,
        "reason": reason,
        "thought": reason,
        "data_status": data_status,
        "volatility_status": volatility_status,
        "entry_blocked": bool(entry_blocked),
        "risk_veto": bool(risk_veto),
        "computation_backend": "stdlib_cpu",
        "computation_engine": "quantitative_math",
        "accelerator": "cpu",
        "telemetry": dict(telemetry or {}),
    }


def _blocked(symbol: str, timeframe: str, status: str, reason: str, data_status: str) -> Dict[str, Any]:
    return _base_result(
        symbol=symbol,
        timeframe=timeframe,
        status=status,
        reason=reason,
        data_status=data_status,
        entry_blocked=True,
    )


def _base_asset(value: str) -> str:
    for quote in ("USDT", "USDC", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[:-len(quote)]
    return value


def _validate_frame(data: Any, symbol: str, timeframe: str, context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, pd.DataFrame):
        return _blocked(symbol, timeframe, "invalid", "Part7 data invalid: expected Pandas DataFrame", "invalid")
    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        return _blocked(symbol, timeframe, "invalid", f"Part7 data invalid: missing columns {','.join(missing)}", "invalid")
    expected_symbol = _symbol(context.get("selected_symbol") or context.get("symbol") or symbol)
    frame_symbol = _symbol(data.attrs.get("symbol"))
    if not context.get("is_backtest_mode", False) and not expected_symbol:
        return _blocked(symbol, timeframe, "invalid", "Part7 selected symbol missing", "invalid")
    if expected_symbol and not frame_symbol and not context.get("is_backtest_mode", False) and not context.get("allow_missing_frame_symbol", False):
        return _blocked(symbol, timeframe, "symbol_mismatch", "Part7 data symbol identity missing", "invalid")
    if expected_symbol and frame_symbol and expected_symbol != frame_symbol and _base_asset(expected_symbol) != _base_asset(frame_symbol):
        return _blocked(symbol, timeframe, "symbol_mismatch", f"Part7 data symbol mismatch: {frame_symbol} != {expected_symbol}", "invalid")
    try:
        validate_interval(timeframe)
    except ValueError:
        return _blocked(symbol, timeframe, "invalid", f"Part7 unsupported timeframe: {timeframe}", "invalid")
    if len(data) < 30:
        return _blocked(symbol, timeframe, "invalid", f"Part7 data invalid: insufficient candles ({len(data)} < 30)", "invalid")
    try:
        numeric = data.loc[:, _REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
        # pandas >= 2.1 deprecated applymap in favour of map
        _applymap = getattr(numeric, "map", None) or getattr(numeric, "applymap", None)
        finite = _applymap(math.isfinite)
        if numeric.isna().any().any() or not finite.all().all():
            return _blocked(symbol, timeframe, "invalid", "Part7 data invalid: non-finite OHLCV", "invalid")
        if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
            return _blocked(symbol, timeframe, "invalid", "Part7 data invalid: non-positive price", "invalid")
        if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any() or (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any():
            return _blocked(symbol, timeframe, "invalid", "Part7 data invalid: OHLC bounds", "invalid")
    except Exception as exc:
        return _blocked(symbol, timeframe, "error", f"Part7 data validation error: {type(exc).__name__}", "error")

    # Direct live frames are closed native candles and must carry a real
    # timestamp index so stale/future data cannot enter the analysis path.
    if not context.get("is_backtest_mode", False):
        if not isinstance(data.index, pd.DatetimeIndex):
            return _blocked(symbol, timeframe, "invalid", "Part7 data timestamp index missing", "invalid")
        try:
            index = data.index
            if index.tz is None:
                index = index.tz_localize("UTC")
            else:
                index = index.tz_convert("UTC")
            stamps = [float(stamp.timestamp()) for stamp in index]
            now = time.time()
            validate_closed_candle_timestamps(timeframe, stamps, now)
            last_open = stamps[-1]
            age = now - float(next_candle_open(timeframe, last_open))
            # Apply the same calendar-aware freshness rule to every supported
            # interval. Tiny bars get a 10s transport allowance so REST polling
            # at the configured cadence does not make each short frame instantly stale.
            bar_seconds = next_candle_open(timeframe, last_open) - last_open
            freshness_intervals = max(3.0, 10.0 / bar_seconds) if bar_seconds > 0 else 3.0
            stale = now > stale_deadline(timeframe, last_open, freshness_intervals)
            if stale:
                return _blocked(symbol, timeframe, "stale", f"Part7 data stale: closed {int(age)}s ago", "stale")
            if age < -5:
                return _blocked(symbol, timeframe, "invalid", "Part7 data timestamp is in the future or not closed", "invalid")
        except ValueError as exc:
            return _blocked(symbol, timeframe, "invalid", f"Part7 timestamp validation failed: {str(exc)[:120]}", "invalid")
        except Exception as exc:
            return _blocked(symbol, timeframe, "error", f"Part7 timestamp validation error: {type(exc).__name__}", "error")
    return None


def analyze_timeframe(data: Any, *, symbol: str, timeframe: str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Analyze one already-selected native exchange timeframe.

    Neutral market conditions remain neutral.  Missing, malformed, stale,
    wrong-symbol, and exceptional inputs are distinct fail-closed statuses and
    block new entries while leaving protective exit code outside this analyzer.
    """
    context = dict(context or {})
    symbol = _symbol(symbol or context.get("selected_symbol")) or "UNKNOWN"
    timeframe = str(timeframe)
    invalid = _validate_frame(data, symbol, timeframe, context)
    if invalid is not None:
        return invalid
    try:
        import quantitative_math as qm
        advisory = qm.part_signal("7", data.tail(128))
        telemetry = dict(advisory.get("telemetry") or {})
        state = telemetry.get("volatility_state", {})
        high_p = float(state.get("high_state_posterior", 0.5)) if isinstance(state, Mapping) else 0.5
        ratio = float(state.get("short_long_ratio", 1.0)) if isinstance(state, Mapping) else 1.0
        blocked = bool(telemetry.get("risk_veto"))
        signal = int(advisory.get("signal", 0))
        if blocked:
            status, reason, vol_status = "veto", str(advisory.get("thought", "Part7 volatility risk veto")), "extreme"
        elif signal:
            status, reason, vol_status = "ok", str(advisory.get("thought", "Part7 realized-variance expansion")), "expanding"
        else:
            status = "neutral"
            reason = str(advisory.get("thought", "Part7 realized-variance state neutral"))
            vol_status = "high" if high_p >= 0.80 else "compressed" if ratio <= 0.5 else "normal"
        telemetry.update({"high_state_posterior_assumption": high_p,
                          "short_long_realized_vol_ratio": ratio,
                          "entry_blocked_by_part7": blocked,
                          "confidence": float(advisory.get("confidence", 5.0))})
        # PHASE4: Part 7 volatility is advisory only on valid data; never blocks or vetoes entry.
        # Missing/stale/symbol-mismatched data still returns _blocked(..., entry_blocked=True) above.
        telemetry["advisory_volatility_warning"] = bool(blocked)
        return _base_result(symbol=symbol, timeframe=timeframe, signal=signal,
                            status=status, reason=reason, data_status="valid",
                            volatility_status=vol_status, entry_blocked=False,
                            risk_veto=False, telemetry=telemetry)
    except Exception as exc:
        return _blocked(symbol, timeframe, "error", f"Part7 analysis exception: {type(exc).__name__}", "error")


def aggregate_results(
    results: Mapping[str, Mapping[str, Any]],
    *,
    symbol: str,
    required_timeframes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate explicit timeframe results into one entry gate."""
    required = tuple(required_timeframes or DEFAULT_TIMEFRAMES)
    normalized_symbol = _symbol(symbol)
    per_timeframe: dict[str, dict[str, Any]] = {}
    for raw_tf in required:
        tf = str(raw_tf)
        raw = results.get(tf) if isinstance(results, Mapping) else None
        if not isinstance(raw, Mapping):
            per_timeframe[tf] = _blocked(symbol, tf, "invalid", "Part7 timeframe result missing or malformed", "missing")
            continue
        row = dict(raw)
        if row.get("timeframe") != tf or _symbol(row.get("symbol")) != normalized_symbol:
            per_timeframe[tf] = _blocked(symbol, tf, "symbol_mismatch", "Part7 timeframe result identity mismatch", "invalid")
            continue
        if (not isinstance(row.get("entry_blocked"), bool)
                or not isinstance(row.get("risk_veto"), bool)
                or row.get("data_status") != "valid"
                or row.get("status") not in {"ok", "neutral", "veto"}):
            per_timeframe[tf] = _blocked(symbol, tf, "invalid", "Part7 timeframe result is incomplete or not execution-valid [raw %s/%s: %s]" % (row.get("status"), row.get("data_status"), str(row.get("reason"))[:100]), "invalid")
            continue
        per_timeframe[tf] = row
    vetoes = [tf for tf, result in per_timeframe.items() if result.get("risk_veto") is True or result.get("status") == "veto"]
    blocked_data = [tf for tf, result in per_timeframe.items()
                    if result.get("status") not in {"ok", "neutral", "veto"}
                    or result.get("data_status") != "valid"
                    or not isinstance(result.get("entry_blocked"), bool)
                    or not isinstance(result.get("risk_veto"), bool)
                    or result.get("entry_blocked") is True]
    if blocked_data:
        status, reason, entry_blocked = "data_blocked", f"Part7 data blocked on {', '.join(blocked_data)}", True
    elif vetoes:
        status, reason, entry_blocked = "ok", f"Part7 volatility advisory on {', '.join(vetoes)} (non-blocking)", False
    else:
        status, reason, entry_blocked = "ok", "Part7 timeframe data valid", False
    # Reduce correlated resolutions to one horizon value before voting. A
    # 1m/3m/5m cluster must not outvote an independent daily/weekly horizon just
    # because it contains more interval labels.
    groups = {}
    required_set = set(required)
    for group_name, frames in TIMEFRAME_CORRELATION_GROUPS.items():
        observed = []
        for tf in frames:
            if tf not in required_set:
                continue
            try:
                value = float(per_timeframe[tf].get("signal", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                observed.append(value)
        if observed:
            groups[group_name] = sum(observed) / len(observed)
    denominator = sum(TIMEFRAME_GROUP_WEIGHTS[name] for name in groups)
    weighted = (sum(groups[name] * TIMEFRAME_GROUP_WEIGHTS[name] for name in groups) / denominator
                if denominator else 0.0)
    signal = 1 if weighted > 0 else -1 if weighted < 0 else 0
    volatility_statuses = sorted({str(r.get("volatility_status", "unknown")) for r in per_timeframe.values()})
    return {
        "signal": signal,
        "signal_identity": f"part7:{_symbol(symbol) or 'UNKNOWN'}:aggregate",
        "signal_name": "CALL" if signal > 0 else "PUT" if signal < 0 else "NEUTRAL",
        "symbol": _symbol(symbol) or "UNKNOWN",
        "timeframe": "aggregate",
        "status": status,
        "reason": reason,
        "thought": reason,
        "data_status": "blocked" if entry_blocked else "valid",
        "volatility_status": ",".join(volatility_statuses),
        "entry_blocked": entry_blocked,
        "risk_veto": False,
        "advisory_volatility_veto": bool(vetoes),
        "computation_backend": "stdlib_cpu",
        "computation_engine": "quantitative_math",
        "accelerator": "cpu",
        "timeframe_results": per_timeframe,
        "blocked_timeframes": blocked_data,
        "veto_timeframes": vetoes,
    }
