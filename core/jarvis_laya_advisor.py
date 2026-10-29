"""Optional, non-authoritative Laya commentary for a single market snapshot.

No model output may become a trade vote, validation, confidence or order input.
One background worker is allowed; a hung inference cannot spawn more workers.
"""
from __future__ import annotations

import copy
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

ALLOWED = {"BUY", "SELL", "NO_TRADE"}
TIMEOUT_SECONDS = 3.0
RESULT_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class Advisory:
    status: str
    symbol: str
    deterministic_decision: str
    suggestion: Optional[str] = None
    confidence: Optional[float] = None
    rationale: str = ""


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ROUTER_LOCK = threading.Lock()
_ROUTER_INSTANCE = None


def _get_router() -> Any:
    global _ROUTER_INSTANCE
    if _ROUTER_INSTANCE is None:
        with _ROUTER_LOCK:
            if _ROUTER_INSTANCE is None:
                from laya import Router
                _ROUTER_INSTANCE = Router()
    return _ROUTER_INSTANCE


def _predict_with_laya(snapshot: dict[str, Any]) -> Any:
    # Upstream NandhaKishorM/laya: Router.predict(state, questions), answers[id].choice.
    # Checkpoint loading is lazy; invoke only from a bounded daemon worker.
    questions = {
        "suggestion": {
            "type": "choice",
            "instructions": (
                "You are reviewing a crypto futures trade setup for decision support only. "
                "Based on the snapshot: parts agreement, timeframe alignment, options walls, "
                "and volatility — which label best describes this setup?"
            ),
            "criteria": {
                "BUY":      "Bullish setup: majority of analysis parts agree to go long, "
                            "HTF timeframes show uptrend, options walls support price above entry.",
                "SELL":     "Bearish setup: majority of analysis parts agree to go short, "
                            "HTF timeframes show downtrend, price near resistance wall.",
                "NO_TRADE": "Unclear or conflicting setup: parts are split, low confidence, "
                            "extreme volatility, or insufficient evidence to enter a trade.",
            },
        }
    }
    result = _get_router().predict(snapshot, questions)
    answer = result.get("answers", {}).get("suggestion", {})
    # Use answer_confidence (probability of chosen class) — the raw 'confidence'
    # field is uncalibrated in current checkpoints (RuntimeWarning from router).
    answer_conf = answer.get("answer_confidence") or answer.get("confidence")
    return {"suggestion": answer.get("choice"), "confidence": answer_conf,
            "rationale": "Laya non-autoregressive advisory (general model, not crypto-trained)."}


def _enabled() -> bool:
    # Enabled by default when laya package is present; set JARVIS_LAYA_ADVISORY=false to disable.
    env = os.getenv("JARVIS_LAYA_ADVISORY", "true").strip().lower()
    return env not in {"0", "false", "no", "off"}


def advise(snapshot: dict[str, Any], *, symbol: str, deterministic_decision: str, predictor: Optional[Callable] = None, enabled: Optional[bool] = None) -> Advisory:
    symbol = str(symbol or "UNKNOWN").upper()
    deterministic = str(deterministic_decision or "NO_TRADE").upper()
    if enabled is None:
        enabled = _enabled()
    if not enabled:
        return Advisory("disabled", symbol, deterministic)
    # Allow snapshots that either have a matching symbol field OR no symbol field at all
    # (Jarvis holistic engine passes snapshots that may not include a top-level symbol key)
    snap_symbol = str(snapshot.get("symbol", "")).upper()
    if not isinstance(snapshot, dict) or symbol == "UNKNOWN":
        return Advisory("invalid", symbol, deterministic, rationale="missing snapshot or unknown symbol")
    if snap_symbol and snap_symbol != symbol:
        return Advisory("invalid", symbol, deterministic, rationale="mismatched symbol snapshot")
    try:
        result = (predictor or _predict_with_laya)(snapshot)
    except (ImportError, ModuleNotFoundError) as exc:
        return Advisory("unavailable", symbol, deterministic, rationale=type(exc).__name__)
    except Exception as exc:
        return Advisory("error", symbol, deterministic, rationale=type(exc).__name__)
    if (not isinstance(result, dict) or not isinstance(result.get("suggestion"), str)
            or result["suggestion"] not in ALLOWED):
        return Advisory("invalid", symbol, deterministic, rationale="invalid Laya structured choice")
    confidence = result.get("confidence")
    if (confidence is not None and (isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1)):
        return Advisory("invalid", symbol, deterministic, rationale="invalid confidence")
    return Advisory("ok", symbol, deterministic, result["suggestion"], confidence,
                    str(result.get("rationale", "Experimental advisory"))[:360])


_lock = threading.Lock()
# Result is served at most once, and only to its exact same-symbol snapshot and decision.
_state = {"busy": False, "generation": 0, "fingerprint": None, "started": 0.0,
          "completed": 0.0, "result": None}


def _fingerprint(snapshot: dict, symbol: str, decision: str) -> Any:
    # Independent of dict insertion order; an opaque repr is only used for equality,
    # never a model input or security token. Defensive copying happens before inference.
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return repr(value)
    return (symbol, decision, freeze(snapshot))


def request_runtime_advisory(snapshot: dict[str, Any], *, symbol: str, deterministic_decision: str) -> Advisory:
    key = str(symbol or "UNKNOWN").upper()
    decision = str(deterministic_decision or "NO_TRADE").upper()
    if not _enabled():
        return Advisory("disabled", key, decision)
    if not isinstance(snapshot, dict) or str(snapshot.get("symbol", "")).upper() != key or key == "UNKNOWN":
        return Advisory("invalid", key, decision, rationale="missing or mismatched symbol snapshot")
    try:
        frozen = copy.deepcopy(snapshot)
        fingerprint = _fingerprint(frozen, key, decision)
    except Exception:
        return Advisory("invalid", key, decision, rationale="snapshot cannot be copied")
    now = time.monotonic()
    with _lock:
        if _state["busy"]:
            if now - _state["started"] >= TIMEOUT_SECONDS:
                return Advisory("timeout", key, decision, rationale="Laya worker timed out; no confirmation")
            return Advisory("pending", key, decision, rationale="bounded Laya worker busy")
        result = _state["result"]
        if (result is not None and _state["fingerprint"] == fingerprint
                and now - _state["completed"] <= RESULT_TTL_SECONDS):
            _state["result"] = None
            return result
        _state.update(busy=True, fingerprint=fingerprint, started=now, result=None)
        _state["generation"] += 1
        generation = _state["generation"]

    def worker():
        try:
            result = advise(frozen, symbol=key, deterministic_decision=decision, enabled=True)
        except Exception as exc:
            result = Advisory("error", key, decision, rationale=type(exc).__name__)
        with _lock:
            if generation == _state["generation"]:
                late = time.monotonic() - _state["started"] >= TIMEOUT_SECONDS
                _state.update(busy=False, completed=time.monotonic(),
                              result=Advisory("timeout", key, decision, rationale="Laya result arrived after deadline") if late else result)
    try:
        threading.Thread(target=worker, name="jarvis-laya-advisory", daemon=True).start()
    except Exception as exc:
        with _lock:
            if generation == _state["generation"]:
                _state.update(busy=False, result=None, fingerprint=None)
        return Advisory("unavailable", key, decision, rationale=f"worker start failed: {type(exc).__name__}")
    return Advisory("pending", key, decision, rationale="background inference; not a trading gate")
