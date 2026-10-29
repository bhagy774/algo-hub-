"""Canonical decision and confidence contracts for JARVIS.

The live loop, dashboard, and executor must consume the same immutable-ish
plain snapshot.  This module is intentionally dependency-free for offline
regression tests.
"""
from __future__ import annotations

import re
import numbers
from typing import Any, Dict, Iterable, Optional
from dataclasses import dataclass, field


_NA = {"", "N/A", "NA", "NONE", "NULL", "UNKNOWN", "UNAVAILABLE", "MISSING", "REJECTED"}


def normalize_confidence(raw: Any, default: Optional[int] = None) -> Optional[int]:
    """Normalize common confidence forms to 0..100, preserving absent as None.

    Accepted forms include ``85``, ``85%``, ``85/100`` and
    ``85/100 (Autonomy)``.  Values explicitly marked unavailable/rejected are
    not converted to zero, so callers can fail closed rather than inventing
    confidence.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return default
    if isinstance(raw, numbers.Real):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if 0 <= value <= 1 and value != 0:
            value *= 100
        if value < 0 or value > 100:
            return default
        return int(round(value))
    text = str(raw).strip()
    if text.upper() in _NA:
        return default
    # Prefer numerator/denominator when present; do not accidentally parse a
    # later parenthetical score before the actual percentage.
    fraction = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", text)
    try:
        if fraction:
            numerator = float(fraction.group(1))
            denominator = float(fraction.group(2))
            if denominator <= 0:
                return default
            value = numerator / denominator * 100
        else:
            percent = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%", text)
            number = percent or re.search(r"([-+]?\d+(?:\.\d+)?)", text)
            if not number:
                return default
            value = float(number.group(1))
            if not percent and 0 < value <= 1:
                value *= 100
    except (TypeError, ValueError):
        return default
    if not 0 <= value <= 100:
        return default
    return int(round(value))


def confidence_text(raw: Any, default: Optional[int] = None) -> str:
    value = normalize_confidence(raw, default=default)
    return "N/A" if value is None else str(value)


def canonical_direction(raw: Any) -> str:
    value = str(raw or "").strip().upper().replace("-", "_")
    if value in {"BUY", "CALL", "LONG", "BULLISH"}:
        return "BUY"
    if value in {"SELL", "PUT", "SHORT", "BEARISH"}:
        return "SELL"
    if value in {"WAIT", "PENDING", "STANDBY"}:
        return "WAIT"
    return "NO_TRADE"


def execution_direction(raw: Any) -> Optional[str]:
    direction = canonical_direction(raw)
    return {"BUY": "CALL", "SELL": "PUT"}.get(direction)


def _opinion_direction(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("direction", value.get("signal", value.get("bias", value.get("verdict"))))
    direction = canonical_direction(value)
    return direction if direction in {"BUY", "SELL"} else None


@dataclass
class DecisionContext:
    symbol: str
    price: Any = None
    reasons: Iterable[Any] = field(default_factory=tuple)
    opinions: Iterable[Any] = field(default_factory=tuple)
    options_context: Optional[Dict[str, Any]] = None
    require_options: bool = False
    gate_reason: str = ""
    blocking_reasons: Iterable[Any] = field(default_factory=tuple)
    plan: Optional[Dict[str, Any]] = None


def build_final_decision(
    signal: Dict[str, Any] | None = None,
    context: Optional[DecisionContext] = None,
    *,
    symbol: str = "",
    price: Any = None,
    reasons: Iterable[Any] = (),
    opinions: Iterable[Any] = (),
    options_context: Optional[Dict[str, Any]] = None,
    require_options: bool = False,
    gate_reason: str = "",
    blocking_reasons: Iterable[Any] = (),
    plan: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Produce the one authoritative decision snapshot consumed downstream."""
    if context is None:
        context = DecisionContext(
            symbol=symbol,
            price=price,
            reasons=reasons,
            opinions=opinions,
            options_context=options_context,
            require_options=require_options,
            gate_reason=gate_reason,
            blocking_reasons=blocking_reasons,
            plan=plan,
        )
    signal = signal if isinstance(signal, dict) else {}
    direction = canonical_direction(signal.get("direction"))
    confidence = normalize_confidence(signal.get("confidence_score", signal.get("confidence")))
    clean_reasons = [str(x) for x in context.reasons if x not in (None, "")]
    conflict = False
    votes = {_opinion_direction(op) for op in context.opinions}
    votes.discard(None)
    if len(votes) > 1:
        conflict = True
        direction = "NO_TRADE"
        clean_reasons.append("Unresolved BUY/SELL opinions; consensus required")
    if confidence is None and direction in {"BUY", "SELL"}:
        direction = "NO_TRADE"
        clean_reasons.append("Confidence unavailable")
    options_ctx = context.options_context if isinstance(context.options_context, dict) else {}
    candidate_direction = direction
    # Conservative macro filter: for non-BTC/ETH charts, a unanimous, fresh,
    # complete BTC+ETH options conflict blocks the candidate. It is never a
    # weighted vote or a substitute for selected-asset evidence. Missing or
    # contradictory macro data is not scored. The separate BTC-only Oracle
    # execution guard remains in force for altcoins.
    symbol_base = str(context.symbol or "").upper().replace("-", "").replace("_", "")
    for quote in ("USDT", "USD"):
        if symbol_base.endswith(quote):
            symbol_base = symbol_base[:-len(quote)]
            break
    if direction in {"BUY", "SELL"} and symbol_base not in {"BTC", "ETH"}:
        macros = options_ctx.get("macro_contexts", [])
        qualified = {}
        if isinstance(macros, list):
            for macro in macros:
                if not isinstance(macro, dict) or macro.get("role") != "macro_context" or macro.get("available") is not True:
                    continue
                asset = str(macro.get("source_asset", "")).upper()
                validation = macro.get("validation") if isinstance(macro.get("validation"), dict) else {}
                if asset in {"BTC", "ETH"} and validation.get("usable") is True and validation.get("complete") is True and validation.get("fresh") is True and validation.get("freshness_status") == "fresh" and validation.get("identity_valid") is True and validation.get("underlying") == asset:
                    qualified[asset] = str(macro.get("bias", "NEUTRAL")).upper()
        expected_conflict = "BEARISH" if direction == "BUY" else "BULLISH"
        if qualified.get("BTC") == expected_conflict and qualified.get("ETH") == expected_conflict:
            direction = "NO_TRADE"
            clean_reasons.append("BTC and ETH options context unanimously conflicts with selected-asset direction")
    if context.require_options and candidate_direction in {"BUY", "SELL"} and (
        not options_ctx.get("available", False) or options_ctx.get("role") != "asset_primary"
    ):
        direction = "NO_TRADE"
        clean_reasons.append("Required selected-asset options context unavailable or non-primary")
    blockers = [str(x) for x in context.blocking_reasons if x not in (None, "")]
    if blockers and direction in {"BUY", "SELL"}:
        direction = "NO_TRADE"
    clean_reasons.extend(blockers)
    if context.gate_reason and direction in {"BUY", "SELL"}:
        direction = "NO_TRADE"
        clean_reasons.append(str(context.gate_reason))
    if direction == "NO_TRADE" and context.gate_reason:
        clean_reasons.append(str(context.gate_reason))
    unique_reasons = list(dict.fromkeys(clean_reasons))
    p = context.plan if isinstance(context.plan, dict) else {}
    allowed = direction in {"BUY", "SELL"} and confidence is not None and not conflict
    return {
        "direction": direction,
        "execution_direction": execution_direction(direction) if allowed else None,
        "confidence": confidence,
        "confidence_display": confidence_text(confidence),
        "symbol": str(context.symbol or "").upper().replace("-", "").replace("_", ""),
        "price": float(context.price) if isinstance(context.price, (int, float)) and float(context.price) > 0 else None,
        "entry": p.get("entry", signal.get("entry_price")),
        "tp1": p.get("tp1", signal.get("take_profit_1")),
        "tp2": p.get("tp2", signal.get("take_profit_2")),
        "sl": p.get("sl", signal.get("stop_loss")),
        "expiry": p.get("expiry", signal.get("recommended_expiry", "N/A")),
        "reasons": unique_reasons,
        "options_context": options_ctx,
        "status": "READY" if allowed else ("CONFLICT" if conflict else "WAIT"),
        "execution_allowed": allowed,
    }
