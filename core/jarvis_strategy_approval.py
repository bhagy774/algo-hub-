"""Pure Jarvis-owned strategy consensus and entry-approval value objects.

Parts 1-12 are evidence producers.  This module has no exchange/network side
 effects and does not place orders; jarvis_FIXED.py is the runtime authority
that calls these functions after analyzing a symbol-scoped candle snapshot.
The approval record is an explicit runtime contract, not a secret credential.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Mapping, Optional

from binance_timeframes import (
    BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_CORRELATION_GROUPS,
    TIMEFRAME_GROUP_WEIGHTS,
)

POLICY_VERSION = "jarvis-central-entry-v1"
APPROVAL_SCHEMA = "jarvis-entry-approval-v1"
DEFAULT_APPROVAL_TTL_SECONDS = 180.0
EXECUTION_PLAN_SCHEMA = "jarvis-execution-plan-v1"
# Existing JarvisAutoTrader swing policy. Scalp levels continue to come from
# ScalpingEngine.calculate_targets in jarvis_FIXED.py.
SWING_STOP_LOSS_PCT = 0.008
SWING_TAKE_PROFIT_PCT = 0.020

PART_WEIGHTS = {
    "part1_breakout": 1.1,
    "part2_zone": 1.3,
    "part3_psychology": 1.0,
    "part4_volume": 1.3,
    "part5_ml": 1.2,
    "part6_trend": 1.5,
    "part7_volatility": 1.1,
    "part8_structure": 1.4,
    "part9_orderflow": 1.4,
    "part10_candlestats": 1.0,
}
ANCHORS = {"part6_trend", "part8_structure", "part9_orderflow"}
INACTIVE_MARKERS = ("error", "offline", "fallback", "missing")
REQUIRED_TIMEFRAMES = BINANCE_SPOT_TIMEFRAMES
# Frames in each horizon bucket are averaged before the horizon receives one
# vote, so adding correlated intervals cannot multiply their influence.
SCALP_DIRECTION_GROUPS = ("fast", "short")
SWING_DIRECTION_GROUPS = ("short", "session", "swing", "macro")
SCALP_HIGHER_TIMEFRAME_GUARD_GROUPS = ("session", "swing", "macro")
ENTRY_TRIGGER_TIMEFRAME = "1m"
CENTRAL_CONSENSUS_THRESHOLD = 0.65  # Retained as the initial confluence threshold.


def _direction(value: Any) -> str:
    value = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if value in {"CALL", "BUY", "LONG", "BULLISH"}:
        return "BUY"
    if value in {"PUT", "SELL", "SHORT", "BEARISH"}:
        return "SELL"
    return "NO_TRADE"


def _clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")


def select_trade_mode(recommended_expiry: Any) -> Optional[str]:
    """Map Jarvis' existing deterministic expiry recommendation to executor modes.

    The live executor supports SCALP and SWING. Its previous call-site mapped
    DAY_TRADE/SWING/15M/30M to SWING and everything else to SCALP; this stricter
    version accepts only the three values emitted by TradeOptimizer.
    """
    value = str(recommended_expiry or "").strip().upper().replace("-", "_").replace(" ", "_")
    if value == "SCALP":
        return "SCALP"
    if value in {"DAY_TRADE", "SWING"}:
        return "SWING"
    return None


def _plan_digest(plan: Mapping[str, Any]) -> str:
    core = {key: plan.get(key) for key in (
        "schema_version", "authority", "trade_mode", "direction", "symbol",
        "snapshot_version", "confidence", "entry_price", "stop_loss",
        "take_profit", "risk_per_unit", "risk_fraction", "reward_risk",
        "sizing_policy",
    )}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


def build_execution_plan(
    *, direction: Any, recommended_expiry: Any, entry_price: Any,
    stop_loss: Any, take_profit: Any, symbol: Any, snapshot_version: Any,
    confidence: Any,
) -> dict[str, Any]:
    """Build Jarvis' only deterministic single-symbol entry plan.

    The SWING policy preserves the existing live trader's 0.8% SL / 2% TP.
    SCALP uses the existing Part 1-12 pipeline's deterministic ScalpingEngine
    SL and TP1; missing/invalid levels are not estimated or backfilled.
    """
    mode = select_trade_mode(recommended_expiry)
    side = _direction(direction)
    clean_symbol = _clean_symbol(symbol)
    version = str(snapshot_version or "").strip()
    try:
        entry = float(entry_price)
        conf = int(float(confidence))
        supplied_stop = float(stop_loss)
        supplied_target = float(take_profit)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("execution plan inputs are malformed")
    if mode is None or side not in {"BUY", "SELL"}:
        raise ValueError("unsupported trade mode or direction")
    if not clean_symbol or not version or not all(math.isfinite(v) and v > 0 for v in (entry, supplied_stop, supplied_target)):
        raise ValueError("execution plan scope or prices are missing/invalid")
    if not 0 <= conf <= 100:
        raise ValueError("execution plan confidence is outside 0..100")
    if mode == "SWING":
        stop = round(entry * (1 - SWING_STOP_LOSS_PCT) if side == "BUY" else entry * (1 + SWING_STOP_LOSS_PCT), 2)
        target = round(entry * (1 + SWING_TAKE_PROFIT_PCT) if side == "BUY" else entry * (1 - SWING_TAKE_PROFIT_PCT), 2)
        sizing_policy = "jarvis_risk.calculate_trade_size+legacy_swing_levels"
    else:
        stop, target = supplied_stop, supplied_target
        sizing_policy = "jarvis_risk.calculate_trade_size+ScalpingEngine.calculate_targets"
    if (side == "BUY" and not stop < entry < target) or (side == "SELL" and not target < entry < stop):
        raise ValueError("entry/stop/target geometry is invalid")
    risk = abs(entry - stop)
    plan = {
        "schema_version": EXECUTION_PLAN_SCHEMA,
        "authority": "jarvis_FIXED",
        "trade_mode": mode,
        "direction": side,
        "symbol": clean_symbol,
        "snapshot_version": version,
        "confidence": conf,
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk_per_unit": risk,
        "risk_fraction": risk / entry,
        "reward_risk": abs(target - entry) / risk,
        "sizing_policy": sizing_policy,
    }
    plan["plan_id"] = _plan_digest(plan)
    return plan


def validate_execution_plan(
    plan: Any, *, direction: Any, symbol: Any, snapshot_version: Any,
    confidence: Any = None, trade_mode: Any = None,
) -> tuple[bool, str]:
    """Reject altered, stale-scope, direction-mismatched or malformed plans."""
    if not isinstance(plan, Mapping):
        return False, "Jarvis execution plan is required"
    if plan.get("schema_version") != EXECUTION_PLAN_SCHEMA or plan.get("authority") != "jarvis_FIXED":
        return False, "Jarvis execution plan schema/authority is invalid"
    expected_direction = _direction(direction)
    if _direction(plan.get("direction")) != expected_direction or expected_direction not in {"BUY", "SELL"}:
        return False, "Jarvis execution plan direction mismatch"
    if _clean_symbol(plan.get("symbol")) != _clean_symbol(symbol):
        return False, "Jarvis execution plan symbol mismatch"
    if not str(snapshot_version or "").strip() or str(plan.get("snapshot_version") or "").strip() != str(snapshot_version).strip():
        return False, "Jarvis execution plan snapshot mismatch"
    mode = str(plan.get("trade_mode") or "").strip().upper()
    if mode not in {"SCALP", "SWING"} or (trade_mode is not None and str(trade_mode).strip().upper() != mode):
        return False, "Jarvis execution plan trade mode mismatch"
    try:
        entry, stop, target = (float(plan[key]) for key in ("entry_price", "stop_loss", "take_profit"))
        conf = int(float(plan["confidence"]))
        risk = float(plan["risk_per_unit"])
        risk_fraction = float(plan["risk_fraction"])
        reward_risk = float(plan["reward_risk"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, "Jarvis execution plan numeric fields are malformed"
    if not all(math.isfinite(x) for x in (entry, stop, target, risk, risk_fraction, reward_risk)) or min(entry, stop, target, risk, risk_fraction, reward_risk) <= 0:
        return False, "Jarvis execution plan contains non-finite/non-positive values"
    if ((expected_direction == "BUY" and not stop < entry < target)
            or (expected_direction == "SELL" and not target < entry < stop)
            or not math.isclose(risk, abs(entry - stop), rel_tol=1e-10, abs_tol=1e-10)
            or not math.isclose(risk_fraction, risk / entry, rel_tol=1e-10, abs_tol=1e-10)
            or not math.isclose(reward_risk, abs(target - entry) / risk, rel_tol=1e-10, abs_tol=1e-10)):
        return False, "Jarvis execution plan geometry is invalid"
    if not 0 <= conf <= 100 or (confidence is not None and conf != int(float(confidence))):
        return False, "Jarvis execution plan confidence mismatch"
    if not str(plan.get("sizing_policy") or "").strip() or plan.get("plan_id") != _plan_digest(plan):
        return False, "Jarvis execution plan integrity check failed"
    return True, "Jarvis execution plan validated"


_REGIME_MULT = {
    "TREND": {"part2_zone": 1.25, "part3_psychology": 1.25, "part6_trend": 1.25, "part8_structure": 1.25, "part9_orderflow": 1.25, "part5_ml": 0.55},
    "MEAN_REVERSION": {"part1_breakout": 1.25, "part4_volume": 1.25, "part5_ml": 1.25, "part10_candlestats": 1.25, "part6_trend": 0.65, "part8_structure": 0.65},
    "VOL_EXPANSION": {"part1_breakout": 1.25, "part4_volume": 1.25, "part8_structure": 1.25, "part9_orderflow": 1.25},
}


def classify_regime(part_results: Any) -> tuple[str, dict[str, Any]]:
    """Regime from the parts' own live telemetry (median across parts). Neutral if unknown."""
    hs, ts, sl = [], [], []
    if isinstance(part_results, Mapping):
        for item in part_results.values():
            tele = item.get("telemetry") if isinstance(item, Mapping) else None
            if not isinstance(tele, Mapping):
                continue
            for key, bucket in (("hurst_proxy", hs), ("trend_score", ts), ("short_long_ratio", sl)):
                v = tele.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    bucket.append(float(v))
    def med(a):
        a = sorted(a)
        return a[len(a) // 2] if a else None
    h, t, r = med(hs), med(ts), med(sl)
    info = {"hurst": h, "trend_score": t, "short_long_ratio": r}
    if r is not None and r >= 1.25:
        return "VOL_EXPANSION", info
    if (h is not None and h >= 0.53) or (t is not None and abs(t) >= 1.2):
        return "TREND", info
    if h is not None and h <= 0.47 and (t is None or abs(t) < 0.8):
        return "MEAN_REVERSION", info
    return "NEUTRAL", info


LAST_REGIME: dict = {}


def evaluate_central_strategy(
    part_results: Any,
    part7_gate: Any,
    *,
    confidence: Any = None,
    minimum_confidence: Optional[float] = None,
    expected_symbol: Any = None,
) -> dict[str, Any]:
    """Apply deterministic Part 1-10 evidence policy centrally in Jarvis.

    Mirrors the canonical Part11 weights, 3-engine / 3.5 quorum, 65% weighted
    consensus, anchor dissent, and Part2 support/resistance veto conditions.
    Part11's return value is intentionally not used as authority: its veto
    findings are re-evaluated here from the underlying part evidence.
    Neutral votes remain neutral, not vetoes.  Part7's explicit block is a
    separate new-entry safety finding and fails closed when absent/malformed.
    """
    blockers: list[str] = []
    if not isinstance(part_results, Mapping):
        return _result("BLOCKED", "NO_TRADE", None, ["Part result mapping is missing"], {})
    if not isinstance(part7_gate, Mapping):
        return _result("BLOCKED", "NO_TRADE", None, ["Part7 gate is missing"], {})
    if part7_gate.get("entry_blocked") is not False or part7_gate.get("risk_veto") is not False:
        reason = str(part7_gate.get("reason") or "Part7 entry gate blocks or is incomplete")
        return _result("BLOCKED", "NO_TRADE", None, [reason], {})
    if expected_symbol is not None:
        expected_clean = _clean_symbol(expected_symbol)
        gate_symbol = _clean_symbol(part7_gate.get("symbol"))
        if not expected_clean or not gate_symbol or gate_symbol != expected_clean:
            return _result("BLOCKED", "NO_TRADE", None, ["Part7 gate symbol identity is missing or mismatched"], {})
        # A bare pair of booleans is not an execution-grade Part7 result. The
        # entry gate must be the complete aggregate from all native intervals,
        # with each frame explicitly valid for the same selected symbol.
        required_timeframes = REQUIRED_TIMEFRAMES
        timeframe_results = part7_gate.get("timeframe_results")
        if (part7_gate.get("status") != "ok" or part7_gate.get("data_status") != "valid"
                or part7_gate.get("timeframe") != "aggregate"
                or not isinstance(timeframe_results, Mapping)
                or set(timeframe_results) != set(required_timeframes)
                or part7_gate.get("blocked_timeframes") != []
                or part7_gate.get("veto_timeframes") != []):
            return _result("BLOCKED", "NO_TRADE", None, ["Part7 aggregate is incomplete or not execution-valid"], {})
        for timeframe in required_timeframes:
            row = timeframe_results.get(timeframe)
            if (not isinstance(row, Mapping) or row.get("timeframe") != timeframe
                    or _clean_symbol(row.get("symbol")) != expected_clean
                    or row.get("data_status") != "valid"
                    or row.get("status") not in {"ok", "neutral", "veto", "blocked"}):
                return _result("BLOCKED", "NO_TRADE", None,
                               [f"Part7 {timeframe} result is missing, stale, mismatched, or blocked"], {})

    required = tuple(PART_WEIGHTS)
    missing = [name for name in required if name not in part_results]
    if missing:
        return _result("BLOCKED", "NO_TRADE", None,
                       ["Part evidence is incomplete: " + ", ".join(missing)], {})

    buy = sell = 0.0
    denominator = 0.0
    buy_engines: list[str] = []
    sell_engines: list[str] = []
    neutral_engines: list[str] = []
    lean_buy = lean_sell = 0.0
    part_scores: dict = {}
    lean_buy_names: list[str] = []
    lean_sell_names: list[str] = []
    regime, regime_info = classify_regime(part_results)
    _mult = dict(_REGIME_MULT.get(regime, {}))
    try:
        from jarvis_self_learning import get_self_learning_engine
        for _pk, _pv in get_self_learning_engine().get_regime_multipliers(regime).items():
            _mult[_pk] = _mult.get(_pk, 1.0) * _pv
    except Exception:
        pass
    for name, base_weight in PART_WEIGHTS.items():
        weight = base_weight * _mult.get(name, 1.0)
        item = part_results.get(name)
        if not isinstance(item, Mapping):
            # Match Part11's offline/errored-engine exclusion rather than
            # letting an unavailable engine manufacture either direction.
            continue
        if name == "part7_volatility":
            continue  # PHASE4: Part7 is advisory/non-directional; excluded from directional denominator
        thought = str(item.get("thought", "")).lower()
        if any(marker in thought for marker in INACTIVE_MARKERS):
            continue
        raw_signal = item.get("signal", 0)
        try:
            signal = float(raw_signal)
        except (TypeError, ValueError, OverflowError):
            return _result("BLOCKED", "NO_TRADE", None,
                           [f"{name} signal is malformed"], {})
        if not math.isfinite(signal):
            return _result("BLOCKED", "NO_TRADE", None,
                           [f"{name} signal is non-finite"], {})

        # Preserve the existing Part11 1m-noise handling when its annotated
        # result format is present. The live MTF adapter emits MTF x/8; those
        # values were already gated by Jarvis' per-part MTF integrity logic.
        import re
        mtf_match = re.search(r"mtf: (\d+)/(\d+)", thought)
        if mtf_match and signal != 0:
            mtf_agree, mtf_total = int(mtf_match.group(1)), int(mtf_match.group(2))
            if mtf_total > 0 and mtf_agree / mtf_total < 0.5:
                signal = 0.0

        denominator += weight
        _tele = item.get("telemetry") if isinstance(item.get("telemetry"), Mapping) else {}
        _cs_raw = item.get("continuous_score", _tele.get("continuous_score"))
        try:
            _cs = float(_cs_raw) if _cs_raw is not None else None
            if _cs is not None and not math.isfinite(_cs):
                _cs = None
        except (TypeError, ValueError, OverflowError):
            _cs = None
        part_scores[name] = [int(signal), None if _cs is None else round(_cs, 3)]
        if signal > 0:
            buy += weight * (max(abs(_cs), 0.75) if (_cs is not None and _cs > 0) else abs(signal))
            buy_engines.append(name)
        elif signal < 0:
            sell += weight * (max(abs(_cs), 0.75) if (_cs is not None and _cs < 0) else abs(signal))
            sell_engines.append(name)
        else:
            neutral_engines.append(name)
            # Neutral parts stay in the denominator; a clear continuous lean earns partial credit.
            if _cs is not None and _cs >= 0.20:
                lean_buy += weight * _cs
                lean_buy_names.append(name)
            elif _cs is not None and _cs <= -0.20:
                lean_sell += weight * abs(_cs)
                lean_sell_names.append(name)

    # Lean credit counts only when >=2 discrete votes already back that direction (never creates a trade alone).
    lean_applied = []
    if len(buy_engines) >= 2 and lean_buy > 0:
        buy += lean_buy
        lean_applied.append("BUY:" + ",".join(lean_buy_names))
    if len(sell_engines) >= 2 and lean_sell > 0:
        sell += lean_sell
        lean_applied.append("SELL:" + ",".join(lean_sell_names))

    active_count = len(buy_engines) + len(sell_engines) + len(neutral_engines)
    directional_count = len(buy_engines) + len(sell_engines)
    active_dir_weight = buy + sell
    effective_den = denominator  # PHASE8: neutral Parts always stay in the confluence denominator
    diagnostic = {
        "weighted_buy": round(buy, 6), "weighted_sell": round(sell, 6),
        "denominator": round(denominator, 6), "effective_denominator": round(effective_den, 6),
        "active_count": active_count, "directional_count": directional_count,
        "buy_engines": list(buy_engines), "sell_engines": list(sell_engines),
        "neutral_engines": list(neutral_engines),
        "regime": regime, "regime_info": regime_info, "lean_applied": lean_applied,
        "part_scores": part_scores, "lean_buy": round(lean_buy, 4), "lean_sell": round(lean_sell, 4),
        "part_scores": part_scores, "lean_buy": round(lean_buy, 4), "lean_sell": round(lean_sell, 4),
    }
    if active_count < 3 or denominator < 3.5:
        return _result("NEUTRAL", "NO_TRADE", None,
                       [f"Insufficient confluence ({active_count}/3 engines, weight {denominator:.1f}/3.5)"], diagnostic)

    buy_ratio = buy / effective_den if effective_den else 0.0
    sell_ratio = sell / effective_den if effective_den else 0.0
    diagnostic.update({"buy_ratio": round(buy_ratio, 6), "sell_ratio": round(sell_ratio, 6)})
    p2 = part_results.get("part2_zone")
    p2 = p2 if isinstance(p2, Mapping) else {}
    p2_thought = str(p2.get("thought", "")).lower()
    try:
        p2_signal = float(p2.get("signal", 0))
    except (TypeError, ValueError, OverflowError):
        return _result("BLOCKED", "NO_TRADE", None, ["Part2 zone signal is malformed"], diagnostic)
    if not math.isfinite(p2_signal):
        return _result("BLOCKED", "NO_TRADE", None, ["Part2 zone signal is non-finite"], diagnostic)

    recommendation = "NO_TRADE"

    def _trend_zone_bypass(side):
        # Trend-continuation: a text-only zone label (Part2 not signalling the opposite way) does not veto
        # when the regime is TREND/VOL_EXPANSION and Part6 trend + another anchor genuinely agree.
        if regime not in ("TREND", "VOL_EXPANSION"):
            return False
        if (side == "BUY" and p2_signal < 0) or (side == "SELL" and p2_signal > 0):
            return False
        agree = buy_engines if side == "BUY" else sell_engines
        return "part6_trend" in agree and any(n in agree for n in ("part8_structure", "part9_orderflow"))

    if buy_ratio >= 0.65:
        if p2_signal < 0 or "resistance zone" in p2_thought:
            if _trend_zone_bypass("BUY"):
                diagnostic["zone_veto_bypassed"] = "BUY"
            else:
                return _result("BLOCKED", "NO_TRADE", None,
                               [f"Central zone veto: BUY blocked at resistance ({p2.get('thought', '')})"], diagnostic)
        strong_opponents = [name for name in sell_engines if name in ANCHORS] + [name for name in ("part5_ml", "part6_trend", "part8_structure", "part9_orderflow") if name not in sell_engines and (part_scores.get(name) or [0, None])[1] is not None and (part_scores.get(name) or [0, 0.0])[1] <= -0.65]
        if len(strong_opponents) >= 2:
            return _result("BLOCKED", "NO_TRADE", None,
                           ["Central anchor-dissent veto: BUY opposed by " + ", ".join(strong_opponents)], diagnostic)
        recommendation = "BUY"
    elif sell_ratio >= 0.65:
        if p2_signal > 0 or "support zone" in p2_thought:
            if _trend_zone_bypass("SELL"):
                diagnostic["zone_veto_bypassed"] = "SELL"
            else:
                return _result("BLOCKED", "NO_TRADE", None,
                               [f"Central zone veto: SELL blocked at support ({p2.get('thought', '')})"], diagnostic)
        strong_opponents = [name for name in buy_engines if name in ANCHORS] + [name for name in ("part5_ml", "part6_trend", "part8_structure", "part9_orderflow") if name not in buy_engines and (part_scores.get(name) or [0, None])[1] is not None and (part_scores.get(name) or [0, 0.0])[1] >= 0.65]
        if len(strong_opponents) >= 2:
            return _result("BLOCKED", "NO_TRADE", None,
                           ["Central anchor-dissent veto: SELL opposed by " + ", ".join(strong_opponents)], diagnostic)
        recommendation = "SELL"
    else:
        return _result("NEUTRAL", "NO_TRADE", None,
                       [f"Split weighted consensus (BUY {buy_ratio:.0%}, SELL {sell_ratio:.0%}; 65% required)"], diagnostic)

    normalized_confidence = None
    if confidence is not None:
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return _result("BLOCKED", "NO_TRADE", recommendation,
                           ["Central confidence is malformed"], diagnostic)
        if not math.isfinite(normalized_confidence) or not 0 <= normalized_confidence <= 100:
            return _result("BLOCKED", "NO_TRADE", recommendation,
                           ["Central confidence is outside 0..100"], diagnostic)
        if minimum_confidence is not None and normalized_confidence < float(minimum_confidence):
            return _result("BLOCKED", "NO_TRADE", recommendation,
                           [f"Central confidence {normalized_confidence:.0f} below minimum {float(minimum_confidence):.0f}"], diagnostic)

    return _result("APPROVED", recommendation, normalized_confidence, [], diagnostic)


def evaluate_mtf_central_strategy(
    timeframe_parts: Any,
    part7_gate: Any,
    *,
    confidence: Any = None,
    minimum_confidence: Optional[float] = None,
    expected_symbol: Any = None,
) -> dict[str, Any]:
    """Select a deterministic SCALP/SWING candidate from complete native evidence.

    All 15 native frames are independently evaluated with the existing Parts
    1â€“11 quorum, Part 2 zone veto and anchor-dissent policy. Frames are then
    averaged within five predeclared correlation horizons before 65% weighted
    confluence is measured. SCALP uses fast+short horizons; SWING uses
    short+session+swing+macro horizons; 1m remains a separate mandatory entry
    trigger. Higher horizons may veto a counter-trend scalp. A neutral 1m is
    always pending, never approval.
    """
    expected_clean = _clean_symbol(expected_symbol)
    if not expected_clean:
        return _result("BLOCKED", "NO_TRADE", None, ["Selected symbol is required for MTF approval"], {})
    if not isinstance(timeframe_parts, Mapping) or set(timeframe_parts) != set(REQUIRED_TIMEFRAMES):
        return _result("BLOCKED", "NO_TRADE", None, ["Complete native 15-timeframe Part evidence is required"], {})
    if not isinstance(part7_gate, Mapping):
        return _result("BLOCKED", "NO_TRADE", None, ["Part7 aggregate gate is missing"], {})
    aggregate_frames = part7_gate.get("timeframe_results")
    if not isinstance(aggregate_frames, Mapping):
        return _result("BLOCKED", "NO_TRADE", None, ["Part7 timeframe evidence is missing"], {})
    if (part7_gate.get("status") != "ok"
            or part7_gate.get("data_status") != "valid"
            or part7_gate.get("timeframe") != "aggregate"
            or _clean_symbol(part7_gate.get("symbol")) != expected_clean
            or set(aggregate_frames) != set(REQUIRED_TIMEFRAMES)
            or part7_gate.get("blocked_timeframes") != []):
        return _result("BLOCKED", "NO_TRADE", None, ["Part7 aggregate is incomplete, blocked, stale, or symbol-mismatched"], {})

    frame_gate = part7_gate["timeframe_results"]
    decisions: dict[str, dict[str, Any]] = {}
    for timeframe in REQUIRED_TIMEFRAMES:
        parts = timeframe_parts.get(timeframe)
        if not isinstance(parts, Mapping):
            return _result("BLOCKED", "NO_TRADE", None, [f"{timeframe} Part evidence is missing"], {})
        missing = [name for name in PART_WEIGHTS if name not in parts]
        if missing:
            return _result("BLOCKED", "NO_TRADE", None,
                           [f"{timeframe} Part evidence is incomplete: " + ", ".join(missing)], {})
        for part_name in PART_WEIGHTS:
            item = parts.get(part_name)
            if not isinstance(item, Mapping):
                return _result("BLOCKED", "NO_TRADE", None,
                               [f"{timeframe} {part_name} evidence is malformed"], {})
            item_symbol = item.get("symbol", item.get("selected_symbol"))
            if item_symbol is not None and _clean_symbol(item_symbol) != expected_clean:
                return _result("BLOCKED", "NO_TRADE", None,
                               [f"{timeframe} {part_name} symbol identity mismatch"], {})
            for identity_key in ("timeframe", "native_timeframe"):
                item_timeframe = item.get(identity_key)
                if item_timeframe is not None and str(item_timeframe) != timeframe:
                    return _result("BLOCKED", "NO_TRADE", None,
                                   [f"{timeframe} {part_name} timeframe identity mismatch"], {})
        part7_item = parts.get("part7_volatility")
        if (part7_item.get("timeframe") != timeframe
                or _clean_symbol(part7_item.get("symbol")) != expected_clean
                or part7_item.get("data_status") != "valid"
                or part7_item.get("entry_blocked") is True
                or part7_item.get("status") not in {"ok", "neutral", "veto"}):
            return _result("BLOCKED", "NO_TRADE", None,
                           [f"{timeframe} Part7 evidence identity/status mismatch"], {})
        gate = frame_gate.get(timeframe)
        if (not isinstance(gate, Mapping) or gate.get("timeframe") != timeframe
                or _clean_symbol(gate.get("symbol")) != expected_clean
                or gate.get("data_status") != "valid"
                or gate.get("entry_blocked") is True
                or gate.get("status") not in {"ok", "neutral", "veto"}):
            return _result("BLOCKED", "NO_TRADE", None,
                           [f"Part7 {timeframe} evidence is missing, stale, blocked, or symbol-mismatched"], {})
        decisions[timeframe] = evaluate_central_strategy(parts, gate)

    try:
        _rc: dict = {}
        _hs = []
        for _d in decisions.values():
            _dg = _d.get("diagnostic") or {}
            _rg = str(_dg.get("regime"))
            _rc[_rg] = _rc.get(_rg, 0) + 1
            _h = (_dg.get("regime_info") or {}).get("hurst")
            if isinstance(_h, (int, float)):
                _hs.append(float(_h))
        _ref = decisions.get("15m") or next(iter(decisions.values()))
        _rd = (_ref.get("diagnostic") or {})
        LAST_REGIME[expected_clean] = {"counts": _rc, "hurst_med": (sorted(_hs)[len(_hs) // 2] if _hs else None),
                                       "summary": {"regime": _rd.get("regime"), "denominator": _rd.get("denominator"),
                                                   "effective_den": _rd.get("effective_denominator"),
                                                   "lean_applied": _rd.get("lean_applied"), "lean_buy": _rd.get("lean_buy"),
                                                   "lean_sell": _rd.get("lean_sell"), "parts[sig,cs]": _rd.get("part_scores"),
                                                   "buy_ratio": _rd.get("buy_ratio"), "sell_ratio": _rd.get("sell_ratio")}}
    except Exception:
        pass

    def group_vote(group_names: tuple[str, ...]) -> dict[str, Any]:
        timeframes = tuple(tf for name in group_names for tf in TIMEFRAME_CORRELATION_GROUPS[name])
        horizon_values: dict[str, float] = {}
        for name in group_names:
            members = TIMEFRAME_CORRELATION_GROUPS[name]
            values = []
            for timeframe in members:
                decision = decisions[timeframe]
                value = (1.0 if decision.get("status") == "APPROVED" and decision.get("direction") == "BUY"
                         else -1.0 if decision.get("status") == "APPROVED" and decision.get("direction") == "SELL"
                         else 0.0)
                values.append(value)
            if values:
                horizon_values[name] = sum(values) / len(values)
        total = sum(TIMEFRAME_GROUP_WEIGHTS[name] for name in group_names)
        buy = sum(TIMEFRAME_GROUP_WEIGHTS[name] * max(horizon_values.get(name, 0.0), 0.0)
                  for name in group_names)
        sell = sum(TIMEFRAME_GROUP_WEIGHTS[name] * max(-horizon_values.get(name, 0.0), 0.0)
                   for name in group_names)
        buy_ratio = buy / total if total else 0.0
        sell_ratio = sell / total if total else 0.0
        direction = "BUY" if buy_ratio >= CENTRAL_CONSENSUS_THRESHOLD else (
            "SELL" if sell_ratio >= CENTRAL_CONSENSUS_THRESHOLD else "NO_TRADE")
        return {"groups": list(group_names), "timeframes": list(timeframes),
                "horizon_values": {key: round(value, 6) for key, value in horizon_values.items()},
                "buy_weight": round(buy, 4), "sell_weight": round(sell, 4),
                "total_weight": round(total, 4), "buy_ratio": round(buy_ratio, 6),
                "sell_ratio": round(sell_ratio, 6), "direction": direction}

    def pooled_track_vote(tfs: tuple[str, ...]) -> dict[str, Any]:
        part_buy = 0.0
        part_sell = 0.0
        active_parts_cnt = 0
        track_group_names = tuple(
            g_name for g_name, g_tfs in TIMEFRAME_CORRELATION_GROUPS.items()
            if any(t in tfs for t in g_tfs)
        )
        for p_name, p_weight in PART_WEIGHTS.items():
            if p_name == "part7_volatility":
                continue
            g_w_sum = 0.0
            g_val_sum = 0.0
            for g_name in track_group_names:
                g_members = [t for t in TIMEFRAME_CORRELATION_GROUPS[g_name] if t in tfs]
                if not g_members:
                    continue
                vals = []
                for tf in g_members:
                    item = (timeframe_parts.get(tf) or {}).get(p_name) or {}
                    try:
                        sig = float(item.get("signal", 0))
                    except (TypeError, ValueError):
                        sig = 0.0
                    _tl = item.get("telemetry") if isinstance(item.get("telemetry"), Mapping) else {}
                    _cs_raw = item.get("continuous_score", _tl.get("continuous_score"))
                    _cs_v = float(_cs_raw) if isinstance(_cs_raw, (int, float)) and math.isfinite(float(_cs_raw)) else 0.0
                    vals.append(1.0 if sig > 0 else (-1.0 if sig < 0 else (0.55 * _cs_v if abs(_cs_v) >= 0.65 else 0.0)))
                g_avg = sum(vals) / len(vals)
                if abs(g_avg) > 1e-6:
                    gw = float(TIMEFRAME_GROUP_WEIGHTS.get(g_name, 1.0))
                    g_w_sum += gw
                    g_val_sum += gw * g_avg
            net_score = (g_val_sum / g_w_sum) if g_w_sum > 0 else 0.0
            if net_score >= 0.25:
                part_buy += p_weight
                active_parts_cnt += 1
            elif net_score <= -0.25:
                part_sell += p_weight
                active_parts_cnt += 1
        active_w = part_buy + part_sell
        # Neutral parts must count: 3 leaning parts out of 9 is NOT a 100% consensus.
        _total_w = sum(w for n, w in PART_WEIGHTS.items() if n != "part7_volatility")
        _enough = active_parts_cnt >= 4 and active_w >= 4.5 and (max(part_buy, part_sell) / max(_total_w, 1e-9)) >= 0.40
        b_ratio = part_buy / active_w if _enough else 0.0
        s_ratio = part_sell / active_w if _enough else 0.0
        p_dir = "BUY" if b_ratio >= CENTRAL_CONSENSUS_THRESHOLD else ("SELL" if s_ratio >= CENTRAL_CONSENSUS_THRESHOLD else "NO_TRADE")
        if p_dir in {"BUY", "SELL"} and sum(1 for _cp in ("part5_ml", "part6_trend", "part8_structure", "part9_orderflow") if ((sum((float((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("signal") or 0) or float((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("continuous_score") or ((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("telemetry") or {}).get("continuous_score") or 0.0)) for _tf in tfs) / max(1, len(tfs))) <= -0.35 if p_dir == "BUY" else (sum((float((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("signal") or 0) or float((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("continuous_score") or ((timeframe_parts.get(_tf) or {}).get(_cp, {}).get("telemetry") or {}).get("continuous_score") or 0.0)) for _tf in tfs) / max(1, len(tfs))) >= 0.35)) >= 2:
            p_dir = "NO_TRADE"
        return {
            "active_parts": active_parts_cnt,
            "buy_weight": round(part_buy, 4),
            "sell_weight": round(part_sell, 4),
            "buy_ratio": round(b_ratio, 6),
            "sell_ratio": round(s_ratio, 6),
            "direction": p_dir,
        }

    swing = group_vote(SWING_DIRECTION_GROUPS)
    scalp = group_vote(SCALP_DIRECTION_GROUPS)
    guard = group_vote(SCALP_HIGHER_TIMEFRAME_GUARD_GROUPS)
    pooled_scalp = pooled_track_vote(("1m", "3m", "5m", "15m"))
    pooled_swing = pooled_track_vote(("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"))

    # SWING requires the canonical horizon-grouped vote; the pooled fallback is diagnostic only for SWING.
    if scalp["direction"] == "NO_TRADE" and pooled_scalp["direction"] in {"BUY", "SELL"}:
        scalp["direction"] = pooled_scalp["direction"]
        scalp["buy_ratio"] = max(scalp["buy_ratio"], pooled_scalp["buy_ratio"])
        scalp["sell_ratio"] = max(scalp["sell_ratio"], pooled_scalp["sell_ratio"])
        scalp["pooled_track_used"] = True

    frame_summary = {
        tf: {"status": decisions[tf].get("status"), "direction": decisions[tf].get("direction"),
             "reasons": list(decisions[tf].get("reasons") or [])}
        for tf in REQUIRED_TIMEFRAMES
    }
    diagnostic = {"swing": swing, "scalp": scalp, "pooled_swing": pooled_swing, "pooled_scalp": pooled_scalp,
                  "scalp_higher_timeframe_guard": guard, "timeframe_decisions": frame_summary}

    if swing["direction"] in {"BUY", "SELL"} and scalp["direction"] in {"BUY", "SELL"} and swing["direction"] != scalp["direction"]:
        result = _result("BLOCKED", "NO_TRADE", None,
                         [f"SCALP_VS_SWING_CONFLICT: SCALP={scalp['direction']} vs SWING={swing['direction']}"], diagnostic)
        result.update({"trade_mode": "CONFLICT", "setup_direction": "NO_TRADE", "entry_trigger": "SCALP_VS_SWING_CONFLICT"})
        return result

    mode = "SWING" if swing["direction"] in {"BUY", "SELL"} else (
        "SCALP" if scalp["direction"] in {"BUY", "SELL"} else None)
    candidate = swing["direction"] if mode == "SWING" else (scalp["direction"] if mode == "SCALP" else "NO_TRADE")
    if mode is None:
        return _result("NEUTRAL", "NO_TRADE", None,
                       ["Neither the conservative SWING nor SCALP timeframe group reaches existing 65% confluence"], diagnostic)

    def _finalize(mode, candidate):
        # A zone or anchor veto in the selected directional group remains a veto;
        # it is not diluted by otherwise-aligned frames.
        selected_groups = SWING_DIRECTION_GROUPS if mode == "SWING" else SCALP_DIRECTION_GROUPS
        selected_timeframes = tuple(tf for name in selected_groups for tf in TIMEFRAME_CORRELATION_GROUPS[name])
        critical_reasons = [reason for tf in selected_timeframes for reason in decisions[tf].get("reasons", [])
                            if ("zone veto" in str(reason).lower() or "anchor-dissent veto" in str(reason).lower())
                            and f"{candidate} " in str(reason)]  # only vetoes against THIS direction (a SELL-at-support veto must not cut a BUY)
        if critical_reasons:
            result = _result("BLOCKED", "NO_TRADE", None, critical_reasons, diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "BLOCKED"})
            return result
        # PHASE4: mode-specific entry trigger (SCALP: 15m/5m/1m, SWING: 1h/4h/30m/15m) so a quiet 1m
        # candle cannot block a valid setup; any approved matching trigger confirms the entry.
        _trig_order = ("15m", "5m", "1m") if mode == "SCALP" else (ENTRY_TRIGGER_TIMEFRAME,)
        if mode == "SWING":
            _d15 = decisions.get("15m") or {}
            if _d15.get("status") == "APPROVED" and _d15.get("direction") in {"BUY", "SELL"} and _d15.get("direction") != candidate:
                result = _result("BLOCKED", "NO_TRADE", None,
                                 [f"SWING {candidate} entry paused: 15m timeframe strongly opposes ({_d15.get('direction')})"], diagnostic)
                result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "15m_OPPOSED"})
                return result
        trigger = None
        for _tf in _trig_order:
            _d = decisions.get(_tf) or {}
            if _d.get("status") == "APPROVED" and _d.get("direction") == candidate:
                trigger = _d
                break
        if trigger is None:
            trigger = decisions.get(_trig_order[0]) or decisions[ENTRY_TRIGGER_TIMEFRAME]
            for _tf in _trig_order:
                _d = decisions.get(_tf) or {}
                if _d.get("status") == "APPROVED":
                    trigger = _d
                    break
        if trigger.get("status") not in {"APPROVED", "NEUTRAL"}:
            result = _result("BLOCKED", "NO_TRADE", None,
                             [f"Entry trigger ({_trig_order[0]}) does not confirm {candidate}"], diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate,
                           "entry_trigger": "TRIGGER_PENDING" if trigger.get("status") != "APPROVED" else "TRIGGER_CONFLICT"})
            return result

        if mode == "SCALP" and guard["direction"] in {"BUY", "SELL"} and guard["direction"] != candidate:
            result = _result("BLOCKED", "NO_TRADE", None,
                             [f"SCALP {candidate} vetoed by higher-timeframe {guard['direction']} confluence"], diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "HTF_VETO"})
            return result

        if mode == "SCALP":
            # A SCALP may never fight the session/swing anchors (ML drift, trend, structure, flow).
            _sgn = 1.0 if candidate == "BUY" else -1.0
            _opp = []
            for _an in ("part5_ml", "part6_trend", "part8_structure", "part9_orderflow"):
                _vals = []
                for _tf in ("30m", "1h", "4h"):
                    _it = (timeframe_parts.get(_tf) or {}).get(_an) or {}
                    try:
                        _s = float(_it.get("signal") or 0)
                    except (TypeError, ValueError):
                        _s = 0.0
                    if _s == 0:
                        _tl = _it.get("telemetry") if isinstance(_it.get("telemetry"), Mapping) else {}
                        _c = _it.get("continuous_score", _tl.get("continuous_score"))
                        _s = float(_c) if isinstance(_c, (int, float)) and math.isfinite(float(_c)) else 0.0
                    _vals.append(max(-1.0, min(1.0, _s)))
                if _vals and (sum(_vals) / len(_vals)) * _sgn <= -0.55:
                    _opp.append(_an)
            if len(_opp) >= 2:
                result = _result("BLOCKED", "NO_TRADE", None,
                                 [f"SCALP {candidate} vetoed: higher-timeframe anchors oppose ({', '.join(_opp)})"], diagnostic)
                result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "HTF_ANCHOR_VETO"})
                return result

        normalized_confidence = None
        if confidence is not None:
            try:
                normalized_confidence = float(confidence)
            except (TypeError, ValueError, OverflowError):
                return _result("BLOCKED", "NO_TRADE", None, ["Central confidence is malformed"], diagnostic)
            if not math.isfinite(normalized_confidence) or not 0 <= normalized_confidence <= 100:
                return _result("BLOCKED", "NO_TRADE", None, ["Central confidence is outside 0..100"], diagnostic)
            if minimum_confidence is not None:
                try:
                    minimum = float(minimum_confidence)
                except (TypeError, ValueError, OverflowError):
                    return _result("BLOCKED", "NO_TRADE", None, ["Minimum confidence is malformed"], diagnostic)
                if not math.isfinite(minimum) or normalized_confidence < minimum:
                    return _result("BLOCKED", "NO_TRADE", candidate,
                                   [f"Central confidence {normalized_confidence:.0f} below minimum {minimum:.0f}"], diagnostic)

        # PHASE4: verify mode-specific entry trigger without forcing 1m on SWING or when track pooled vote already confirmed
        _trig_status = trigger.get("status")
        _trig_dir = trigger.get("direction")
        _track_confirmed = (mode == "SWING" and swing.get("direction") == candidate) or (mode == "SCALP" and scalp.get("direction") == candidate)
        if _trig_status == "BLOCKED":
            result = _result("BLOCKED", "NO_TRADE", normalized_confidence, ["Entry-trigger evidence is blocked or incomplete"], diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "BLOCKED"})
            return result
        if _trig_status == "APPROVED" and _trig_dir in {"BUY", "SELL"} and _trig_dir != candidate:
            result = _result("BLOCKED", "NO_TRADE", normalized_confidence,
                             [f"Entry trigger ({_trig_dir}) opposes {mode} {candidate} setup"], diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "OPPOSITE"})
            return result
        if _trig_status != "APPROVED" and (mode == "SWING" or not _track_confirmed):
            result = _result("PENDING", "NO_TRADE", normalized_confidence, ["Entry trigger is neutral; setup pending"], diagnostic)
            result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "NEUTRAL"})
            return result

        if normalized_confidence is None and minimum_confidence is not None:
            return _result("BLOCKED", "NO_TRADE", None, ["Central confidence is required for entry approval"], diagnostic)
        result = _result("APPROVED", candidate, normalized_confidence, [], diagnostic)
        result.update({"trade_mode": mode, "setup_direction": candidate, "entry_trigger": "ALIGNED"})
        return result

    result = _finalize(mode, candidate)
    # SWING is preferred, but if the SWING candidate is not approved while the SCALP group independently
    # agrees on the same direction and passes its own trigger/vetoes, evaluate it as SCALP.
    if ((not result.get("approved")) and mode == "SWING" and scalp["direction"] == candidate
            and result.get("entry_trigger") not in {"NEUTRAL", "OPPOSITE", "15m_OPPOSED", "TRIGGER_PENDING", "TRIGGER_CONFLICT"}):
        scalp_result = _finalize("SCALP", candidate)
        if scalp_result.get("approved"):
            scalp_result["fallback_from"] = "SWING"
            return scalp_result
    return result


def _result(status: str, direction: str, confidence: Any, reasons: list[str], diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "approved": status == "APPROVED",
        "direction": direction,
        "recommendation": direction if direction != "NO_TRADE" else "NO_TRADE",
        "confidence": confidence,
        "reasons": list(reasons),
        "policy_version": POLICY_VERSION,
        "diagnostic": diagnostic,
    }


@dataclass(frozen=True)
class EntryApproval:
    schema_version: str
    authority: str
    status: str
    policy_version: str
    direction: str
    symbol: str
    exchange: str
    contract: str
    instrument_id: str
    market_type: str
    analysis_symbol: str
    analysis_exchange: str
    snapshot_version: str
    analysis_timestamp: float
    expires_at: float
    confidence: int
    approval_id: str
    execution_plan_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_entry_approval(
    central_decision: Mapping[str, Any], *, direction: Any, symbol: Any,
    exchange: Any, contract: Any, instrument_id: Any = "", market_type: Any = "",
    analysis_symbol: Any = "", analysis_exchange: Any = "", snapshot_version: Any,
    analysis_timestamp: Any, confidence: Any, execution_plan: Any = None,
    ttl_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
) -> dict[str, Any]:
    """Create a scope-bound approval only from an APPROVED Jarvis decision."""
    if not isinstance(central_decision, Mapping) or central_decision.get("approved") is not True:
        raise ValueError("Jarvis central strategy did not approve this entry")
    normalized_direction = _direction(direction)
    if normalized_direction not in {"BUY", "SELL"} or _direction(central_decision.get("direction")) != normalized_direction:
        raise ValueError("approval direction differs from the central decision")
    clean_symbol, clean_contract = _clean_symbol(symbol), _clean_symbol(contract)
    clean_exchange = str(exchange or "").strip().lower()
    clean_id = str(instrument_id or clean_contract).strip()
    clean_market = str(market_type or "").strip().lower()
    version = str(snapshot_version or "").strip()
    try:
        issued = float(analysis_timestamp)
        conf = int(float(confidence))
        ttl = float(ttl_seconds)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("approval timestamps/confidence are malformed")
    if (not clean_symbol or clean_symbol != clean_contract or not clean_exchange
            or not clean_id or not version or not math.isfinite(issued)
            or not math.isfinite(ttl) or ttl <= 0 or not 0 <= conf <= 100):
        raise ValueError("approval scope is incomplete")
    if _clean_symbol(central_decision.get("symbol", clean_symbol)) != clean_symbol:
        raise ValueError("approval symbol differs from central decision scope")
    plan_id = ""
    if execution_plan is not None:
        valid_plan, plan_reason = validate_execution_plan(
            execution_plan, direction=normalized_direction, symbol=clean_symbol,
            snapshot_version=version, confidence=conf,
        )
        if not valid_plan:
            raise ValueError(plan_reason)
        plan_id = str(execution_plan.get("plan_id") or "")
    expires = issued + min(ttl, DEFAULT_APPROVAL_TTL_SECONDS)
    core = {
        "schema_version": APPROVAL_SCHEMA,
        "authority": "jarvis_FIXED",
        "status": "APPROVED",
        "policy_version": POLICY_VERSION,
        "direction": normalized_direction,
        "symbol": clean_symbol,
        "exchange": clean_exchange,
        "contract": clean_contract,
        "instrument_id": clean_id,
        "market_type": clean_market,
        "analysis_symbol": _clean_symbol(analysis_symbol or clean_symbol),
        "analysis_exchange": str(analysis_exchange or clean_exchange).strip().lower(),
        "snapshot_version": version,
        "analysis_timestamp": issued,
        "expires_at": expires,
        "confidence": conf,
        "execution_plan_id": plan_id,
    }
    # The identifier is diagnostic and helps catch accidental field drift; it
    # is deliberately not treated as a secret or as cryptographic authorization.
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**core, "approval_id": digest[:24]}


def validate_entry_approval(
    approval: Any, *, direction: Any, symbol: Any, exchange: Any, contract: Any,
    instrument_id: Any = None, market_type: Any = None, snapshot_version: Any,
    confidence: Any = None, execution_plan: Any = None, now: Optional[float] = None,
    max_age_seconds: float = DEFAULT_APPROVAL_TTL_SECONDS,
) -> tuple[bool, str]:
    """Validate an explicit central approval against the actual entry request."""
    if isinstance(approval, EntryApproval):
        approval = approval.to_dict()
    if not isinstance(approval, Mapping):
        return False, "Central Jarvis entry approval is required"
    if (approval.get("schema_version") != APPROVAL_SCHEMA
            or approval.get("authority") != "jarvis_FIXED"
            or approval.get("status") != "APPROVED"
            or approval.get("policy_version") != POLICY_VERSION):
        return False, "Central Jarvis approval schema/authority is invalid"
    if _direction(approval.get("direction")) != _direction(direction) or _direction(direction) not in {"BUY", "SELL"}:
        return False, "Central Jarvis approval direction mismatch"
    if (_clean_symbol(approval.get("symbol")) != _clean_symbol(symbol)
            or _clean_symbol(approval.get("contract")) != _clean_symbol(contract)
            or str(approval.get("exchange") or "").strip().lower() != str(exchange or "").strip().lower()):
        return False, "Central Jarvis approval instrument scope mismatch"
    if instrument_id is not None and str(approval.get("instrument_id") or "").strip() != str(instrument_id).strip():
        return False, "Central Jarvis approval contract id mismatch"
    if market_type is not None and str(approval.get("market_type") or "").strip().lower() != str(market_type).strip().lower():
        return False, "Central Jarvis approval contract type mismatch"
    version = str(snapshot_version or "").strip()
    if not version or str(approval.get("snapshot_version") or "").strip() != version:
        return False, "Central Jarvis approval snapshot version mismatch"
    try:
        issued = float(approval.get("analysis_timestamp"))
        expires = float(approval.get("expires_at"))
        approved_confidence = int(approval.get("confidence"))
        current = time.time() if now is None else float(now)
        max_age = float(max_age_seconds)
    except (TypeError, ValueError, OverflowError):
        return False, "Central Jarvis approval timestamp/confidence is malformed"
    if (not all(math.isfinite(x) for x in (issued, expires, current, max_age))
            or max_age <= 0 or issued > current + 30 or expires <= current
            or current - issued > max_age or expires - issued > DEFAULT_APPROVAL_TTL_SECONDS + 1e-6):
        return False, "Central Jarvis approval is expired or from the future"
    if confidence is not None:
        try:
            requested_confidence = int(float(confidence))
        except (TypeError, ValueError, OverflowError):
            return False, "Requested confidence is malformed"
        if requested_confidence != approved_confidence:
            return False, "Central Jarvis approval confidence mismatch"
    bound_plan_id = str(approval.get("execution_plan_id") or "")
    if bound_plan_id:
        valid_plan, plan_reason = validate_execution_plan(
            execution_plan, direction=direction, symbol=symbol,
            snapshot_version=snapshot_version, confidence=approved_confidence,
        )
        if not valid_plan or str(execution_plan.get("plan_id") or "") != bound_plan_id:
            return False, plan_reason if not valid_plan else "Central Jarvis approval execution-plan mismatch"
    elif execution_plan is not None:
        valid_plan, plan_reason = validate_execution_plan(
            execution_plan, direction=direction, symbol=symbol,
            snapshot_version=snapshot_version, confidence=approved_confidence,
        )
        if not valid_plan:
            return False, plan_reason
    # Check the diagnostic digest to reject accidental mutation. This is not
    # an authentication mechanism; callers must enforce provenance/scope.
    core = {k: approval.get(k) for k in (
        "schema_version", "authority", "status", "policy_version", "direction", "symbol",
        "exchange", "contract", "instrument_id", "market_type", "analysis_symbol",
        "analysis_exchange", "snapshot_version", "analysis_timestamp", "expires_at", "confidence",
        "execution_plan_id",
    )}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    if approval.get("approval_id") != digest:
        return False, "Central Jarvis approval record was modified"
    return True, "Central Jarvis approval validated"
