#!/usr/bin/env python3
"""
Jarvis Pre-Trade Simulator
━━━━━━━━━━━━━━━━━━━━━━━━━━
"Dar signal pehla instant backtest check — aa setup history ma kaam karyu hato?"
(Before every signal: an instant check — did this setup work in history?)

Runs a FAST, fully offline pre-trade simulation BEFORE a paper trade is
opened:

  1. Historical win-rate of similar setups — from recorded trades in
     jarvis_learning.json (same direction + symbol, + regime if available).
  2. R:R sanity — TP/SL from smart_tpsl_calculator; veto when reward:risk
     falls below JARVIS_PRESIM_MIN_RR (default 1.0).
  3. Volatility regime — extreme ATR vs recent norm → 'adjust' with a
     confidence penalty (never a veto).

Hard rules:
  * NO network access anywhere in this module. Pure offline math + local JSON.
  * Bounded time budget (default 500 ms, env JARVIS_PRESIM_BUDGET_MS). When
    the budget is exhausted the simulator abstains: {'action': 'pass'}.
  * Conservative defaults: insufficient history (<5 similar trades) → 'pass'
    with confidence_delta 0. Never veto on missing data.
  * Fail-open: any internal error → 'pass'. Disable with JARVIS_PRESIM=0.
"""

import json
import logging
import os
import time

logger = logging.getLogger("JarvisPreSim")

LEARNING_FILE = os.environ.get("JARVIS_LEARNING_FILE", "jarvis_learning.json")

MIN_SIMILAR_TRADES = 5        # below this, abstain (pass, delta 0)
VETO_WIN_RATE = 0.35          # below → veto (setup historically loses)
LOW_WIN_RATE = 0.50           # below → adjust (confidence penalty)
HIGH_WIN_RATE = 0.60          # at/above → adjust (small confidence boost)
MAX_CONFIDENCE_DELTA = 10

# Volatility extremes: ATR as % of price outside these bounds vs recent norm
VOL_EXTREME_LOW = 0.0005      # ATR < 0.05% of price → dead market
VOL_EXTREME_HIGH = 0.05       # ATR > 5% of price → chaotic market
VOL_PENALTY = -5


def presim_enabled():
    """Pre-trade simulator is on unless JARVIS_PRESIM=0."""
    return os.environ.get("JARVIS_PRESIM", "1") != "0"


def _budget_ms():
    try:
        return max(0, int(os.environ.get("JARVIS_PRESIM_BUDGET_MS", "500")))
    except (TypeError, ValueError):
        return 500


def _min_rr():
    try:
        return float(os.environ.get("JARVIS_PRESIM_MIN_RR", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def _pass(reason=""):
    return {"action": "pass", "confidence_delta": 0, "reason": reason}


def _adjust(delta, reason):
    delta = max(-MAX_CONFIDENCE_DELTA, min(MAX_CONFIDENCE_DELTA, int(delta)))
    return {"action": "adjust", "confidence_delta": delta, "reason": reason}


def _veto(reason):
    return {"action": "veto", "confidence_delta": 0, "reason": reason}


def _load_learning_trades(path=None):
    """Load recorded trades from jarvis_learning.json. Never raises."""
    path = path or LEARNING_FILE
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades = data.get("trades", []) if isinstance(data, dict) else []
        return [t for t in trades if isinstance(t, dict)]
    except Exception as e:
        logger.debug(f"[PRESIM] learning file unreadable ({path}): {e}")
        return []


def _similar_trades(trades, symbol=None, direction=None, regime=None):
    """Filter recorded trades matching direction (+symbol, +regime if stored)."""
    out = []
    for t in trades:
        if direction and t.get("direction") != direction:
            continue
        if symbol and t.get("symbol") and t.get("symbol") != symbol:
            continue
        # Only match regime when BOTH sides carry one — never invent fields.
        if regime and t.get("regime") and t.get("regime") != regime:
            continue
        out.append(t)
    return out


def check_historical_winrate(signal, learning_path=None, deadline=None):
    """Check 1: empirical win-rate of similar setups from jarvis_learning.json.

    Returns a decision dict or None (check not applicable / skipped).
    """
    trades = _load_learning_trades(learning_path)
    similar = _similar_trades(
        trades,
        symbol=signal.get("symbol"),
        direction=signal.get("direction"),
        regime=signal.get("regime"),
    )
    decided = [t for t in similar if t.get("result") in ("WIN", "LOSS")]
    if len(decided) < MIN_SIMILAR_TRADES:
        return None  # insufficient history → abstain (never veto on missing data)

    wins = sum(1 for t in decided if t.get("result") == "WIN")
    wr = wins / len(decided)
    n = len(decided)
    if wr < VETO_WIN_RATE:
        return _veto(f"historical win-rate {wr:.0%} over {n} similar trades (< {VETO_WIN_RATE:.0%})")
    if wr < LOW_WIN_RATE:
        return _adjust(-8, f"weak history: win-rate {wr:.0%} over {n} similar trades")
    if wr >= HIGH_WIN_RATE:
        return _adjust(+5, f"strong history: win-rate {wr:.0%} over {n} similar trades")
    return None  # neutral zone


def check_risk_reward(signal, candles=None, deadline=None):
    """Check 2: R:R sanity via smart_tpsl_calculator (import guarded).

    Veto when reward:risk < JARVIS_PRESIM_MIN_RR (default 1.0).
    Returns a decision dict or None.
    """
    try:
        from smart_tpsl_calculator import SmartTPSLCalculator
    except Exception:
        return None  # calculator unavailable → skip check (fail-open)

    try:
        calc = SmartTPSLCalculator()
        levels = calc.calculate(
            candles or [],
            signal.get("direction", "CALL"),
            entry_price=signal.get("entry_price"),
        )
        rr = float(levels.get("risk_reward", 0.0) or 0.0)
        if rr <= 0:
            return None  # could not compute (bad inputs) → abstain
        min_rr = _min_rr()
        if rr < min_rr:
            return _veto(f"R:R {rr:.2f} below minimum {min_rr:.2f}")
        return None
    except Exception as e:
        logger.debug(f"[PRESIM] R:R check failed (ignored): {e}")
        return None


def check_volatility_regime(signal, candles=None, snapshot=None, deadline=None):
    """Check 3: extreme volatility → 'adjust' with penalty (never veto).

    Uses candles (ATR vs price) when available, else the snapshot's
    volatility label. Returns a decision dict or None.
    """
    snapshot = snapshot or {}
    try:
        if candles is not None:
            try:
                from smart_tpsl_calculator import SmartTPSLCalculator
                atr = SmartTPSLCalculator().calculate_atr(candles)
            except Exception:
                atr = 0.0
            entry = signal.get("entry_price")
            try:
                entry = float(entry)
            except (TypeError, ValueError):
                entry = 0.0
            if atr > 0 and entry > 0:
                atr_pct = atr / entry
                if atr_pct < VOL_EXTREME_LOW:
                    return _adjust(VOL_PENALTY, f"volatility extremely low (ATR {atr_pct:.3%} of price)")
                if atr_pct > VOL_EXTREME_HIGH:
                    return _adjust(VOL_PENALTY, f"volatility extremely high (ATR {atr_pct:.3%} of price)")
                return None

        vol = str(snapshot.get("volatility", "")).upper()
        if vol in ("VERY_LOW", "EXTREME_LOW"):
            return _adjust(VOL_PENALTY, f"volatility regime {vol}")
        if vol in ("VERY_HIGH", "EXTREME_HIGH", "EXTREME"):
            return _adjust(VOL_PENALTY, f"volatility regime {vol}")
        return None
    except Exception as e:
        logger.debug(f"[PRESIM] volatility check failed (ignored): {e}")
        return None


def run_presim(signal, candles=None, snapshot=None, learning_path=None):
    """Run the full pre-trade simulation. Always returns a decision dict.

    signal:  {'symbol', 'direction' ('CALL'|'PUT'), 'entry_price', optional 'regime'}
    candles: recent historical candles from the data pipeline (list of dicts
             or DataFrame) — used for ATR/R:R. Never fetched here.
    snapshot: current market snapshot dict (e.g. market_context with
             'volatility', 'trend').

    Precedence: veto > adjust > pass. Budget guard: if the time budget is
    already exhausted before a check runs, remaining checks are skipped and
    the best decision so far is returned (default 'pass').
    """
    if not presim_enabled():
        return _pass("presim disabled (JARVIS_PRESIM=0)")

    start = time.monotonic()
    deadline = start + _budget_ms() / 1000.0

    def _expired():
        return time.monotonic() >= deadline

    try:
        direction = str(signal.get("direction", "")).upper()
        if direction not in ("CALL", "PUT"):
            return _pass("no directional signal")

        decisions = []
        for check in (
            lambda: check_historical_winrate(signal, learning_path, deadline),
            lambda: check_risk_reward(signal, candles, deadline),
            lambda: check_volatility_regime(signal, candles, snapshot, deadline),
        ):
            if _expired():
                logger.debug("[PRESIM] budget exhausted — abstaining on remaining checks")
                break
            try:
                d = check()
            except Exception as e:
                logger.debug(f"[PRESIM] check raised (ignored): {e}")
                d = None
            if d:
                decisions.append(d)
            if d and d.get("action") == "veto":
                break  # veto short-circuits

        for d in decisions:
            if d.get("action") == "veto":
                return d
        if decisions:
            total_delta = sum(int(d.get("confidence_delta", 0)) for d in decisions)
            reasons = "; ".join(d.get("reason", "") for d in decisions if d.get("reason"))
            if total_delta != 0 or any(d.get("action") == "adjust" for d in decisions):
                return _adjust(total_delta, reasons or "adjust")
        return _pass()
    except Exception as e:
        logger.warning(f"[PRESIM] simulation failed (fail-open → pass): {e}")
        return _pass(f"presim error: {e}")
