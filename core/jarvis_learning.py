#!/usr/bin/env python3
"""
Jarvis Learning Loop
━━━━━━━━━━━━━━━━━━━━
Records every closed paper-trade outcome into jarvis_learning.json and
maintains per-engine running stats + a weight multiplier used to scale
each engine's contribution in the brain's weighted signal fusion.

Fail-safe design: every public function is guarded — any error is logged
and swallowed so the trading loop is never broken.

Disable entirely with env JARVIS_LEARNING=0.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime

logger = logging.getLogger("JarvisLearning")

LEARNING_FILE = os.environ.get("JARVIS_LEARNING_FILE", "jarvis_learning.json")

MIN_TRADES_FOR_WEIGHT = 10   # below this, weight is neutral 1.0
MIN_WEIGHT = 0.5
MAX_WEIGHT = 1.5

_lock = threading.Lock()
_cache = None          # in-process copy of the JSON state
_cache_path = None


def learning_enabled() -> bool:
    """Learning loop is on unless JARVIS_LEARNING=0."""
    return os.environ.get("JARVIS_LEARNING", "1") != "0"


def _empty_state() -> dict:
    return {"trades": [], "engines": {}}


def _load(path=None):
    """Load state from disk; gracefully handle corruption."""
    global _cache, _cache_path
    path = path or LEARNING_FILE
    with _lock:
        if _cache is not None and _cache_path == path:
            return _cache
        state = _empty_state()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    state["trades"] = data.get("trades") if isinstance(data.get("trades"), list) else []
                    state["engines"] = data.get("engines") if isinstance(data.get("engines"), dict) else {}
        except Exception as e:
            logger.warning(f"[Learning] Could not load {path} (corrupt?) — starting fresh: {e}")
            state = _empty_state()
        _cache = state
        _cache_path = path
        return _cache


def _save(state, path=None):
    """Atomic write: write tmp file then rename."""
    path = path or LEARNING_FILE
    try:
        directory = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".jarvis_learning_", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
    except Exception as e:
        logger.warning(f"[Learning] Could not save {path}: {e}")


def _engine_stats(engines, name):
    return engines.setdefault(name, {"trades": 0, "wins": 0, "losses": 0})


def record_trade(trade, engine_signals=None, path=None):
    """Record a closed trade outcome.

    trade: dict with at least direction/entry_price/exit_price/result/pnl_dollar.
    engine_signals: optional {engine_name: signal (-1/0/1)} captured at decision
                    time. Engines whose signal agreed with the trade direction
                    are credited/blamed for the outcome.

    Returns True if recorded, False otherwise. Never raises.
    """
    try:
        if not learning_enabled():
            return False
        if not isinstance(trade, dict):
            return False

        result = trade.get("result")
        if result not in ("WIN", "LOSS", "BREAKEVEN"):
            return False

        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": trade.get("symbol", "BTC/USDT"),
            "direction": trade.get("direction"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "pnl": trade.get("pnl_dollar", 0),
            "result": result,
            "close_reason": trade.get("close_reason"),
            "confidence": trade.get("confidence"),
        }
        # Store per-engine signal snapshot when available (don't invent fields)
        if isinstance(engine_signals, dict) and engine_signals:
            entry["engine_signals"] = {
                str(k): v for k, v in engine_signals.items()
                if isinstance(v, (int, float))
            }

        state = _load(path)
        state["trades"].append(entry)

        # Attribute outcome to engines that voted with the trade direction
        if isinstance(engine_signals, dict):
            direction = trade.get("direction")
            dir_sign = 1 if direction == "CALL" else (-1 if direction == "PUT" else 0)
            if dir_sign != 0:
                for name, sig in engine_signals.items():
                    try:
                        sig = float(sig)
                    except (TypeError, ValueError):
                        continue
                    if sig == 0:
                        continue
                    # Engine participated if its signal matched trade direction
                    if (sig > 0) == (dir_sign > 0):
                        stats = _engine_stats(state["engines"], str(name))
                        stats["trades"] += 1
                        if result == "WIN":
                            stats["wins"] += 1
                        elif result == "LOSS":
                            stats["losses"] += 1

        _save(state, path)
        return True
    except Exception as e:
        logger.warning(f"[Learning] record_trade failed (ignored): {e}")
        return False


def get_engine_stats(name, path=None):
    """Return running stats for an engine: trades/wins/losses/win_rate."""
    try:
        state = _load(path)
        s = state["engines"].get(name, {"trades": 0, "wins": 0, "losses": 0})
        trades = s.get("trades", 0)
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        decided = wins + losses
        win_rate = (wins / decided) if decided > 0 else 0.0
        return {"trades": trades, "wins": wins, "losses": losses, "win_rate": win_rate}
    except Exception as e:
        logger.warning(f"[Learning] get_engine_stats failed (ignored): {e}")
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}


def get_engine_weight(name, path=None):
    """Weight multiplier in [0.5, 1.5] for an engine.

    Neutral 1.0 when learning disabled or insufficient data (<10 trades).
    Linear map of win rate: 0% -> 0.5, 50% -> 1.0, 100% -> 1.5
    (clamped; <=30% win rate sits at/below 0.8, >=70% at/above 1.2).
    """
    try:
        if not learning_enabled():
            return 1.0
        stats = get_engine_stats(name, path)
        if stats["trades"] < MIN_TRADES_FOR_WEIGHT:
            return 1.0
        if stats["wins"] + stats["losses"] == 0:
            return 1.0  # no decided outcomes yet (all breakeven) — stay neutral
        weight = 0.5 + stats["win_rate"]  # 0% wr -> 0.5, 100% wr -> 1.5
        return max(MIN_WEIGHT, min(MAX_WEIGHT, weight))
    except Exception as e:
        logger.warning(f"[Learning] get_engine_weight failed (ignored): {e}")
        return 1.0
