"""Bounded, display-only diagnostics for Jarvis live analysis snapshots.

This module consumes values already produced by the pipeline. It never fetches
market data, evaluates strategy rules, derives confidence, or submits orders.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import math
import re
from numbers import Real
from typing import Any, Iterable

from binance_timeframes import (
    BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_HISTORY_CANDLES,
    next_candle_open, stale_deadline,
)

TIMEFRAMES = BINANCE_SPOT_TIMEFRAMES
INTERVAL_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
                    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
                    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
                    "3d": 259200, "1w": 604800, "1M": 2592000}
PART_KEYS = (
    "part1_breakout", "part2_zone", "part3_psychology", "part4_volume",
    "part5_ml", "part6_trend", "part7_volatility", "part8_structure",
    "part9_orderflow", "part10_candlestats",
)
PART_LABELS = tuple(f"P{i:02d}" for i in range(1, 11))
_MISSING = {"", "N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNAVAILABLE", "MISSING", "—", "-"}
_BAD_STATUS = {"error", "failed", "stale", "invalid", "missing", "unavailable", "not_run", "not run",
               "symbol_mismatch", "timeframe_mismatch", "frame_mismatch", "insufficient", "insufficient_data", "insufficient_candles",
               "data_error", "stale_data", "invalid_data"}
_BLOCKED_STATUS = {"blocked", "veto", "risk_veto", "entry_blocked", "rejected"}
_REDACT = re.compile(r"(?i)(api[_ -]?key|secret|token|credential|authorization|password|account(?:[_ -]?(?:id|identifier|number|no))?|wallet(?:[_ -]?(?:id|identifier))?|(?:client[_ -]?)?order[_ -]?(?:id|identifier)|position[_ -]?(?:id|identifier))\s*[:=]\s*[^\s,;]+")
_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.I)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().upper() not in _MISSING
    return True


def _clean_text(value: Any, limit: int = 120) -> str:
    if value is None:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    text = _URL_CREDENTIALS.sub(r"\1[redacted]@", text)
    text = _REDACT.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit - 1].rstrip() + "…"
    return text


def _text(value: Any, limit: int = 120) -> str:
    return _clean_text(value, limit) if _present(value) else ""


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _signal(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        return _signal(value[-1]) if value else None
    if isinstance(value, Mapping):
        for key in ("signal", "direction", "bias", "verdict", "action", "recommendation"):
            if key in value:
                return _signal(value[key])
        return None
    number = _finite_number(value)
    if number is not None:
        return "BUY" if number > 0 else "SELL" if number < 0 else "NEUTRAL"
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"BUY", "CALL", "LONG", "BULL", "BULLISH", "UP"}:
        return "BUY"
    if text in {"SELL", "PUT", "SHORT", "BEAR", "BEARISH", "DOWN"}:
        return "SELL"
    if text in {"0", "NEUTRAL", "NO_TRADE", "WAIT", "FLAT", "NONE"}:
        return "NEUTRAL"
    return None


def classify_part_output(result: Any = None, *, error_type: str | None = None) -> dict[str, Any]:
    """Normalize one actual part return without conflating neutral and missing."""
    if error_type:
        return {"status": "ERROR", "signal": None, "confidence": None,
                "reason": f"Analyzer exception: {_text(error_type, 60)}"}
    if not isinstance(result, Mapping):
        return {"status": "MISSING", "signal": None, "confidence": None,
                "reason": "Analyzer returned no result mapping"}

    confidence = None
    for key in ("confidence", "confidence_score", "conf"):
        if key in result and _present(result[key]):
            confidence = result[key]
            break

    raw_signal = None
    signal_key_present = False
    for key in ("signal", "direction", "bias", "verdict", "action", "recommendation"):
        if key in result:
            raw_signal = result[key]
            signal_key_present = True
            break
    parsed_signal = _signal(raw_signal) if signal_key_present else None

    raw_status = _clean_text(result.get("status"), 40).lower().replace("-", "_").replace(" ", "_")
    data_status = _clean_text(result.get("data_status"), 40).lower().replace("-", "_").replace(" ", "_")
    reason = ""
    for key in ("reason", "thought", "reasoning", "message", "verdict"):
        candidate = result.get(key)
        if _present(candidate) and not (key == "verdict" and candidate == raw_signal):
            reason = _text(candidate)
            break

    if raw_status in _BAD_STATUS:
        status = raw_status.upper().replace("_", " ")
    elif data_status in _BAD_STATUS:
        status = data_status.upper().replace("_", " ")
    elif raw_status in _BLOCKED_STATUS or data_status in _BLOCKED_STATUS or result.get("risk_veto") is True or result.get("entry_blocked") is True:
        status = "BLOCKED"
    elif raw_status == "neutral" or data_status == "neutral":
        status, parsed_signal = "NEUTRAL", parsed_signal or "NEUTRAL"
    elif not signal_key_present or parsed_signal is None:
        status = "MISSING"
        reason = reason or ("Signal field absent" if not signal_key_present else "Signal value unrecognized")
    elif parsed_signal == "NEUTRAL":
        status = "NEUTRAL"
    else:
        status = "OK"

    return {"status": status, "signal": parsed_signal, "confidence": confidence, "reason": reason}


def _epoch(value: Any) -> float | None:
    number = _finite_number(value)
    if number is not None:
        return number
    try:
        if hasattr(value, "timestamp") and callable(value.timestamp):
            return float(value.timestamp())
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return None


def _timestamp_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        if hasattr(value, "isoformat"):
            rendered = value.isoformat()
        else:
            rendered = str(value)
    except Exception:
        return "UNKNOWN"
    return _text(rendered, 48) or "UNKNOWN"


def _canonical_symbol(value: Any) -> str | None:
    text = _text(value, 80).upper().replace("/", "").replace("-", "").replace("_", "")
    # Delta India routes are XXXUSD; analysis candles are Binance Spot XXXUSDT.
    if text.endswith("USD") and not text.endswith("USDT"):
        text = text[:-3] + "USDT"
    return text or None


def _symbols_match(a: Any, b: Any) -> bool:
    sa = _canonical_symbol(a)
    sb = _canonical_symbol(b)
    if not sa or not sb:
        return False
    if sa == sb:
        return True
    def _base(s: str) -> str:
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if s.endswith(quote) and len(s) > len(quote):
                return s[:-len(quote)]
        return s
    base_a = _base(sa)
    base_b = _base(sb)
    return bool(base_a and base_b and base_a == base_b)



def build_timeframe_inputs(
    frames: Any,
    snapshot: Any = None,
    *,
    timeframes: Iterable[str] = TIMEFRAMES,
    now: float | None = None,
    expected_symbol: Any = None,
    stale_intervals: float | None = 2.0,
    fetch_ttl_seconds: float | None = 45.0,
) -> dict[str, dict[str, Any]]:
    """Create bounded readiness/freshness/identity metadata from actual inputs.

    A timeframe is labelled STALE for display only when its last closed candle
    exceeds the cache's stale-interval policy or its snapshot age exceeds the
    cache refresh TTL. This never gates, changes, or substitutes a strategy result.
    """
    frames = frames if isinstance(frames, Mapping) else {}
    snapshot_frames = _get(snapshot, "frames", {})
    snapshot_frames = snapshot_frames if isinstance(snapshot_frames, Mapping) else {}
    current = float(datetime.now(timezone.utc).timestamp()) if now is None else float(now)
    root_fetch = _epoch(_get(snapshot, "fetched_at"))
    expected = _canonical_symbol(expected_symbol)
    root_symbol = _canonical_symbol(_get(snapshot, "symbol"))
    report: dict[str, dict[str, Any]] = {}
    for tf in timeframes:
        frame = frames.get(tf)
        if frame is None:
            report[str(tf)] = {"status": "MISSING", "bars": None,
                               "expected_bars": TIMEFRAME_HISTORY_CANDLES.get(str(tf)),
                               "last_closed": None,
                               "closed_age_seconds": None, "identity": "UNKNOWN",
                               "source": None, "snapshot_age_seconds": None,
                               "current_candle": "UNKNOWN"}
            continue
        try:
            bars = len(frame)
        except Exception:
            bars = None
        try:
            last_closed = frame.index[-1] if bars else None
        except Exception:
            last_closed = None
        candle_frame = snapshot_frames.get(tf)
        source = _get(candle_frame, "source")
        frame_fetch = _epoch(_get(candle_frame, "fetched_at"))
        fetched = frame_fetch if frame_fetch is not None else root_fetch
        age = current - fetched if fetched is not None and current >= fetched else None
        close_epoch = _epoch(last_closed)
        interval = INTERVAL_SECONDS.get(str(tf))
        if close_epoch is not None and str(tf) == "1M":
            try:
                closed_age = current - next_candle_open(str(tf), close_epoch)
            except (ValueError, OverflowError, OSError):
                closed_age = current - (close_epoch + interval) if interval is not None else None
        else:
            closed_age = current - (close_epoch + interval) if close_epoch is not None and interval is not None else None
        expected_bars = TIMEFRAME_HISTORY_CANDLES.get(str(tf))
        if closed_age is not None and closed_age < -5.0:
            closed_age = None
        current_candle = "FORMING / UNCONFIRMED" if _get(candle_frame, "current") is not None else "not supplied"
        if candle_frame is not None and _get(candle_frame, "current_is_confirmed") is False:
            current_candle = "FORMING / UNCONFIRMED"

        attrs = getattr(frame, "attrs", {})
        attrs = attrs if isinstance(attrs, Mapping) else {}
        frame_symbol = (_canonical_symbol(_get(candle_frame, "symbol"))
                        or _canonical_symbol(attrs.get("symbol")) or root_symbol)
        frame_tf = (_text(_get(candle_frame, "timeframe"), 12)
                    or _text(attrs.get("timeframe"), 12))
        identity = "UNKNOWN"
        if frame_symbol is not None:
            identity = "VERIFIED" if expected is None or _symbols_match(frame_symbol, expected) else "SYMBOL MISMATCH"
        if frame_tf and frame_tf != str(tf):
            identity = "TIMEFRAME MISMATCH"
        if bars is None:
            status = "UNKNOWN"
        elif bars <= 0:
            status = "EMPTY"
        else:
            status = "PRESENT"

        stale_reasons = []
        stale_after_seconds = None
        stale_closed = False
        if closed_age is not None and stale_intervals is not None and close_epoch is not None:
            if str(tf) == "1M":
                try:
                    close_boundary = next_candle_open(str(tf), close_epoch)
                    deadline = stale_deadline(str(tf), close_epoch, 1.0 + float(stale_intervals))
                    stale_after_seconds = deadline - close_boundary
                    stale_closed = current > deadline
                except (ValueError, OverflowError, OSError):
                    stale_after_seconds = None
            elif interval is not None:
                stale_after_seconds = max(interval * float(stale_intervals), 10.0)
                stale_closed = closed_age > stale_after_seconds
        if stale_closed:
            stale_reasons.append(
                f"closed age {closed_age:.1f}s > {float(stale_intervals):g} intervals"
            )
        if (age is not None and fetch_ttl_seconds is not None
                and age > float(fetch_ttl_seconds)):
            stale_reasons.append(f"fetch age {age:.1f}s > refresh TTL {float(fetch_ttl_seconds):g}s")
        freshness_reason = "; ".join(stale_reasons) or None
        if identity.endswith("MISMATCH"):
            status = identity
        elif status == "PRESENT" and freshness_reason:
            status = "STALE"
        report[str(tf)] = {
            "status": status, "bars": bars, "expected_bars": expected_bars,
            "last_closed": _timestamp_label(last_closed) if last_closed is not None else None,
            "closed_age_seconds": round(closed_age, 1) if closed_age is not None else None,
            "identity": identity,
            "source": _text(source, 40) or None,
            "snapshot_age_seconds": round(age, 1) if age is not None else None,
            "current_candle": current_candle,
            "freshness_reason": freshness_reason,
            "closed_stale_after_seconds": (
                round(interval * float(stale_intervals), 1)
                if interval is not None and stale_intervals is not None else None
            ),
            "fetch_stale_after_seconds": fetch_ttl_seconds,
        }
    return report


def _cell(item: Any, *, input_status: Any = None) -> str:
    item = item if isinstance(item, Mapping) else {"status": "MISSING"}
    status = _clean_text(item.get("status"), 24).upper() or "UNKNOWN"
    signal = item.get("signal")
    confidence = item.get("confidence")
    if status in {"ERROR", "FAILED", "STALE", "STALE DATA", "INVALID", "INVALID DATA",
                  "MISSING", "UNAVAILABLE", "NOT RUN", "SYMBOL MISMATCH", "TIMEFRAME MISMATCH",
                  "UNKNOWN", "EMPTY", "INSUFFICIENT", "INSUFFICIENT DATA", "INSUFFICIENT CANDLES", "DATA ERROR"}:
        label = f"{signal}/{status}" if signal else status.replace("_", " ")
    elif status == "BLOCKED":
        label = f"BLOCKED/{signal}" if signal else "BLOCKED"
    elif status == "NEUTRAL":
        label = "NEUTRAL"
    elif signal:
        label = signal if status == "OK" else f"{signal}/{status}"
    else:
        label = status
    input_state = _clean_text(input_status, 24).upper()
    if input_state in {"STALE", "SYMBOL MISMATCH", "TIMEFRAME MISMATCH"} and input_state not in label.upper():
        label = f"{label}/{input_state}"

    conf_text = "?" if confidence is None else _text(confidence, 20)
    return f"{label} c={conf_text}"


def _part_reason_groups(per_tf: Mapping[str, Any], name: str, order: tuple[str, ...], verbose: bool) -> list[str]:
    outcomes = []
    for tf in order:
        frame_result = per_tf.get(tf, {})
        item = frame_result.get(name, {}) if isinstance(frame_result, Mapping) else {}
        outcomes.append((tf, item if isinstance(item, Mapping) else {}))
    if not outcomes:
        return []
    def sig(item: Mapping[str, Any]) -> tuple[Any, Any, str]:
        return item.get("status"), item.get("signal"), repr(item.get("confidence"))
    signatures = [sig(item) for _, item in outcomes]
    distinct = len(set(signatures))
    lines: list[str] = []
    if verbose:
        groups: dict[str, list[str]] = {}
        for tf, item in outcomes:
            reason = _text(item.get("reason"), 150)
            if reason:
                groups.setdefault(reason, []).append(tf)
        for reason, tfs in groups.items():
            lines.append(f"  {name}: {','.join(tfs)} reason: {reason}")
        return lines

    if distinct <= 1:
        # One representative reason is enough when all timeframes report the
        # same state; avoid printing an identical paragraph eight times.
        for tf, item in outcomes:
            reason = _text(item.get("reason"), 150)
            if reason:
                lines.append(f"  {name}: same result across {len(outcomes)} TFs; {tf} reason: {reason}")
                break
        return lines

    previous = None
    for tf, item in outcomes:
        current = sig(item)
        if current != previous:
            reason = _text(item.get("reason"), 150)
            status = _clean_text(item.get("status"), 24) or "UNKNOWN"
            side = _text(item.get("signal"), 16) or "—"
            extra = f"; reason: {reason}" if reason else ""
            lines.append(f"  {name} @ {tf}: {side}/{status}{extra}")
        previous = current
    return lines


def _raw_confidence(value: Any) -> str:
    if value is None:
        return "UNKNOWN (field absent)"
    return _text(value, 48)


def _safe_runtime(runtime: Any) -> list[str]:
    if not isinstance(runtime, Mapping):
        return ["Runtime backend: UNKNOWN — no runtime attributes supplied."]
    allowed = ("backend", "device", "device_name", "detected", "torch_available", "fallback_reason")
    fields = [f"{key}={_text(runtime.get(key), 80)}" for key in allowed if _present(runtime.get(key))]
    return ["Runtime backend (reported by process): " + ("; ".join(fields) if fields else "UNKNOWN fields")] 


def _neural_status_text(status: Any) -> str:
    if not isinstance(status, Mapping):
        return "Neural/advisory result: UNKNOWN — no current-cycle neural status supplied; deterministic Part outputs remain separate."
    name = _clean_text(status.get("status"), 40) or "UNKNOWN"
    reason = _text(status.get("reason"), 120)
    availability = _clean_text(status.get("availability"), 40)
    fields = [f"status={name}"]
    if availability:
        fields.append(f"availability={availability}")
    if reason:
        fields.append(f"reason={reason}")
    return "Neural/advisory result (not a rule Part result): " + "; ".join(fields)


def _gate_text(gate: Any) -> str:
    if not isinstance(gate, Mapping):
        return "Risk/Part 7 gate: UNKNOWN — no current-cycle gate record supplied."
    fields = []
    for key in ("status", "entry_blocked", "risk_veto", "data_status", "volatility_status"):
        if key in gate and gate.get(key) is not None:
            fields.append(f"{key}={_clean_text(gate.get(key), 48)}")
    reason = _text(gate.get("reason"), 160)
    if reason:
        fields.append(f"reason={reason}")
    return "Risk/Part 7 gate (reported): " + ("; ".join(fields) if fields else "UNKNOWN fields")


def format_cycle_unavailable(*, symbol: Any, stage: Any, reason: Any = None,
                             timeframes: Iterable[str] = TIMEFRAMES) -> str:
    """Render a compact current-cycle fetch/gate failure without invented state."""
    lines = ["JARVIS | LIVE ANALYSIS DIAGNOSTICS (display only)",
             f"Market: {_text(symbol, 60) or 'UNKNOWN'} | stage={_text(stage, 32) or 'UNKNOWN'}",
             "Current-cycle analyzer evidence: NOT GENERATED / UNAVAILABLE; no prior-cycle evidence reused."]
    safe_reason = _text(reason, 160)
    lines.append("Observed reason: " + (safe_reason if safe_reason else "UNKNOWN — no reason supplied"))
    lines.append("Native timeframe inputs: " + ", ".join(f"{tf}=UNAVAILABLE" for tf in timeframes))
    lines.append("Parts 1–12, decision, risk gate, execution, position, and neural result: UNKNOWN / NOT REPORTED.")
    lines.append("Display only; no trade decision inferred from unavailable input.")
    return "\n".join(lines)


def format_live_diagnostics(
    *,
    symbol: Any,
    snapshot: Any = None,
    timeframe_inputs: Any = None,
    parts_by_timeframe: Any = None,
    fused_parts: Any = None,
    decision: Any = None,
    candidate: Any = None,
    part7_gate: Any = None,
    entry_trigger: Any = None,
    setup_mode: Any = None,
    runtime: Any = None,
    neural_status: Any = None,
    cycle_status: Any = None,
    cycle_reason: Any = None,
    verbose: bool = False,
    timeframes: Iterable[str] = TIMEFRAMES,
) -> str:
    """Render one bounded synthetic/live-cycle diagnostics block, display only."""
    order = tuple(str(tf) for tf in timeframes)
    inputs = timeframe_inputs if isinstance(timeframe_inputs, Mapping) else {}
    matrix = parts_by_timeframe if isinstance(parts_by_timeframe, Mapping) else {}
    fused = fused_parts if isinstance(fused_parts, Mapping) else {}
    decision = decision if isinstance(decision, Mapping) else {}
    candidate = candidate if isinstance(candidate, Mapping) else {}
    cycle = cycle_status if isinstance(cycle_status, Mapping) else {}
    cycle_label = _clean_text(cycle.get("status"), 32).upper() or "UNKNOWN"
    cycle_note = _text(cycle_reason, 160) or _text(cycle.get("reason"), 160)
    lines = ["JARVIS | LIVE ANALYSIS DIAGNOSTICS (display only)",
             f"Market: {_text(symbol, 60) or 'UNKNOWN'} | analysis uses closed candles; forming candle is not a signal input.",
             f"Analysis cycle: {cycle_label}" + (f"; reason={cycle_note}" if cycle_note else "; reason=UNKNOWN — not recorded.")]
    snap_symbol = _text(_get(snapshot, "symbol"), 60)
    snap_venue = _text(_get(snapshot, "venue"), 40)
    snap_fetched = _epoch(_get(snapshot, "fetched_at"))
    if snap_symbol or snap_venue or snap_fetched is not None:
        _snap_delta = None if snap_fetched is None else datetime.now(timezone.utc).timestamp() - snap_fetched
        snap_age = "unknown" if _snap_delta is None or _snap_delta < 0 else f"{_snap_delta:.1f}s"
        # Do not print instrument ids, account identifiers, or arbitrary snapshot mappings.
        selected = _canonical_symbol(symbol)
        observed = _canonical_symbol(snap_symbol)
        identity = "UNKNOWN" if not selected or not observed else "MATCH" if _symbols_match(selected, observed) else "MISMATCH"
        lines.append(f"Snapshot: symbol={snap_symbol or 'UNKNOWN'}; selected-symbol identity={identity}; source venue={snap_venue or 'UNKNOWN'}; fetched age={snap_age}.")
    threshold_example = next((inputs.get(tf) for tf in order
                              if isinstance(inputs.get(tf), Mapping)), {})
    closed_threshold = threshold_example.get("closed_stale_after_seconds")
    fetch_threshold = threshold_example.get("fetch_stale_after_seconds")
    threshold_text = []
    if closed_threshold is not None:
        threshold_text.append(f"closed age > {closed_threshold}s")
    if fetch_threshold is not None:
        threshold_text.append(f"fetch age > {fetch_threshold}s")
    threshold_suffix = (" — STALE is a display label using " + " or ".join(threshold_text)
                        + " (not a strategy gate).") if threshold_text else " — age thresholds unknown."
    lines.append("Timeframe input readiness (freshness per-frame)" + threshold_suffix)
    for tf in order:
        item = inputs.get(tf) if isinstance(inputs.get(tf), Mapping) else {}
        bars = item.get("bars")
        expected_bars = item.get("expected_bars")
        bars_text = "unknown" if bars is None else str(bars)
        if expected_bars is not None:
            bars_text += f"/{expected_bars} closed"
        age = item.get("snapshot_age_seconds")
        age_text = "unknown" if age is None else f"{age}s"
        close_age = item.get("closed_age_seconds")
        close_age_text = "unknown" if close_age is None else f"{close_age}s"
        identity = _text(item.get("identity"), 24) or "UNKNOWN"
        status = _clean_text(item.get("status"), 24).upper() or "UNKNOWN"
        fresh_reason = _text(item.get("freshness_reason"), 96)
        fresh_suffix = f" freshness={fresh_reason}" if fresh_reason else ""
        lines.append(f"  {tf:>3} {status} identity={identity} bars={bars_text} source={_text(item.get('source'), 32) or 'UNKNOWN'} fetch_age={age_text} closed_age={close_age_text} last_closed_index={_text(item.get('last_closed'), 48) or 'UNKNOWN'}; current={_text(item.get('current_candle'), 32) or 'UNKNOWN'}{fresh_suffix}")

    lines.append("Parts 1–10 raw analyzer output by native timeframe (not extra votes):")
    lines.append("  Legend: BUY/SELL/NEUTRAL are reported signal values; c is the exact reported confidence field, not a calibrated probability; c=? means absent; c=0 is a reported zero.")
    header = "Part | " + " | ".join(f"{tf:>5}" for tf in order)
    lines.append("  " + header)
    cycle_has_not_run = cycle_label in {"NOT RUN", "BLOCKED", "UNAVAILABLE", "INSUFFICIENT", "ANALYZING", "ERROR"}
    for name, label in zip(PART_KEYS, PART_LABELS):
        cells = []
        for tf in order:
            tf_parts = matrix.get(tf, {})
            tf_parts = tf_parts if isinstance(tf_parts, Mapping) else {}
            item = tf_parts.get(name)
            if item is None:
                item = ({"status": "NOT RUN"} if cycle_has_not_run
                        else {"status": "MISSING"})
            input_status = (inputs.get(tf, {}).get("status")
                            if isinstance(inputs.get(tf), Mapping) else None)
            cells.append(_cell(item, input_status=input_status))
        lines.append(f"  {label} | " + " | ".join(cells))

    lines.append("Parts 11–12 are fused evidence outputs, not per-candle analyzers:")
    for name, label in (("part11_fusion", "Part 11 fusion"), ("part12_confidence", "Part 12 confidence")):
        item = fused.get(name)
        if not isinstance(item, Mapping):
            lines.append(f"  {label}: NOT REPORTED — no current-cycle fusion return recorded.")
            continue
        outcome = classify_part_output(item)
        if name == "part12_confidence" and "signal" not in item:
            status = _text(item.get("status"), 32) or "reported"
            confidence = _raw_confidence(item.get("confidence", item.get("confidence_score")))
            reason = _text(item.get("thought", item.get("reason", item.get("reasoning"))), 130)
            details = f"confidence(raw)={confidence}; status={status}"
            if reason:
                details += f"; reason={reason}"
            lines.append(f"  {label} over all available Part 1–10 MTF evidence: {details} (not a calibrated probability).")
        else:
            reason = f"; reason={outcome['reason']}" if outcome.get("reason") else ""
            fused_signal = outcome.get('signal')
            fused_signal_text = fused_signal if fused_signal is not None else 'UNKNOWN'
            lines.append(f"  {label}: status={outcome['status']} signal={fused_signal_text} confidence(raw)={_raw_confidence(outcome.get('confidence'))}{reason}.")

    unique_candidate = candidate.get("trade_signal") if isinstance(candidate.get("trade_signal"), Mapping) else candidate
    plan = []
    plan_fields = (("entry", "Entry", ("entry", "entry_price")),
                   ("tp1", "TP1", ("tp1", "take_profit_1")),
                   ("tp2", "TP2", ("tp2", "take_profit_2")),
                   ("sl", "SL", ("sl", "stop_loss")))
    for decision_key, label, aliases in plan_fields:
        value = decision.get(decision_key) if _present(decision.get(decision_key)) else None
        if value is None:
            value = next((unique_candidate[k] for k in aliases if isinstance(unique_candidate, Mapping) and _present(unique_candidate.get(k))), None)
        plan.append(f"{label}={_text(value, 40) if value is not None else 'UNKNOWN'}")
    if isinstance(setup_mode, Mapping):
        mode_text = _text(setup_mode.get("mode"), 32) or "UNKNOWN"
        mode_reason = _text(setup_mode.get("reason"), 140)
    else:
        mode_text = _text(setup_mode, 32) or "UNKNOWN"
        mode_reason = ""
    mode_source = "reported/derived execution mapping" if mode_text != "UNKNOWN" else "not supplied"
    lines.append(f"Setup mode ({mode_source}): {mode_text}" + (f"; reason={mode_reason}" if mode_reason else "") + " | plan values are candidates only (not proof of submission/fill): " + "; ".join(plan) + ".")
    decision_direction = _text(decision.get("direction"), 32) or "UNKNOWN"
    execution_direction = _text(decision.get("execution_direction"), 32)
    reasons = decision.get("reasons")
    if isinstance(reasons, (list, tuple)):
        reason_items = [_text(x, 120) for x in reasons if _present(x)]
    elif _present(reasons):
        reason_items = [_text(reasons, 120)]
    else:
        reason_items = []
    if not reason_items and isinstance(candidate, Mapping) and _present(candidate.get("no_trade_reason")):
        reason_items = [_text(candidate.get("no_trade_reason"), 120)]
    permission = "ALLOWED" if decision.get("execution_allowed") is True else "BLOCKED" if decision.get("execution_allowed") is False else "UNKNOWN"
    decision_status = _clean_text(decision.get("status"), 32).upper() or "UNKNOWN"
    execution_note = f"; execution_direction={execution_direction}" if execution_direction else ""
    lines.append(f"Final decision: {decision_direction}{execution_note}; execution_allowed={permission}; status={decision_status}; reason(s)=" + (" | ".join(reason_items) if reason_items else "UNKNOWN — no reason recorded") + ".")
    if isinstance(entry_trigger, Mapping):
        confirmed = entry_trigger.get("confirmed")
        state = "confirmed" if confirmed is True else "waiting / not confirmed" if confirmed is False else "status unknown"
        trigger_note = _text(entry_trigger.get("note"), 140) or "no trigger note recorded"
        lines.append(f"1m entry trigger: {state}; {trigger_note}.")
    else:
        lines.append("1m entry trigger: UNKNOWN — no current-cycle confirmation record supplied.")
    lines.append(_gate_text(part7_gate))
    lines.extend(_safe_runtime(runtime))
    lines.append(_neural_status_text(neural_status))

    reasons_by_part: list[str] = []
    for name, label in zip(PART_KEYS, PART_LABELS):
        reasons_by_part.extend(_part_reason_groups(matrix, name, order, verbose))
    limit = 80 if verbose else 24
    lines.append("Per-timeframe reasons/status changes:" if reasons_by_part else (
        "Per-timeframe reasons/status changes: analyzers did not run; see analysis-cycle reason above."
        if cycle_has_not_run else
        "Per-timeframe reasons/status changes: no reason text was supplied by the raw part analyzers."
    ))
    lines.extend(reasons_by_part[:limit])
    if len(reasons_by_part) > limit:
        lines.append(f"  {len(reasons_by_part) - limit} reason lines summarized; set JARVIS_DIAGNOSTICS_VERBOSE=1 for all unique supplied reasons (maximum {len(PART_KEYS) * len(order)}).")
    lines.append("Terminal diagnostics are read-only; neutral signals are not promoted, and unavailable/error/stale outputs are not treated as neutral.")
    return "\n".join(lines)
