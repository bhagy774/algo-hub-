#!/usr/bin/env python3
"""
Multi-Source Cross-Exchange Data for Jarvis
============================================
Gives the brain (jarvis_FIXED.py) a light-weight "cross_exchange" perspective
alongside Delta Exchange data:

  - Binance  : public spot API (no keys needed). 15m trend + 24h change for BTC.
  - Upstox   : Indian market bias via existing upstox_data.py. Fully OPTIONAL —
               gracefully skipped when no token is configured.

Safety guarantees:
  - FAIL-CLOSED: any source that errors or times out (>5s) is returned with
    ok=False. Data is NEVER fabricated.
  - All network calls use timeout <= 5s, single attempt (no retry storms).
  - Last good result per source is cached for max 60s.
  - This module NEVER blocks trading: callers skip the perspective when a
    source is unavailable. Delta remains the primary data source.

Normalized context dict:
  {
      "source":     "binance" | "upstox",
      "symbol":     str,
      "price":      float,
      "trend":      "BULLISH" | "BEARISH" | "NEUTRAL",
      "change_pct": float,          # % change used for trend strength
      "timestamp":  float (epoch),
      "ok":         bool,
  }
"""

import logging
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"
HTTP_TIMEOUT = 5.0          # hard cap on every network call
CACHE_TTL = 60.0            # cache last good result max 60s

# Trend strength thresholds (15m close-to-close % change)
STRONG_TREND_PCT = 0.5      # |change| above this counts as a "strong" trend
WEAK_TREND_PCT = 0.1        # below this the market is treated as NEUTRAL

# Fusion tuning (weight ~1.1 perspective — small, advisory adjustments)
AGREE_BOOST = 5             # sources agree with Delta signal direction
DISAGREE_PENALTY = 8        # mild disagreement
STRONG_DISAGREE_PENALTY = 15  # strong opposite trend -> heavy cut, may veto
UPSTOX_BIAS = 2             # informational nudge only, never blocks

# ── Per-source last-good-result cache ────────────────────────────────────────
_cache: Dict[str, Dict] = {}


def _cached(source: str) -> Optional[Dict]:
    """Return the cached context for a source if it is still fresh (<=60s)."""
    entry = _cache.get(source)
    if entry and (time.time() - entry["timestamp"]) <= CACHE_TTL:
        return dict(entry)
    return None


def _store(source: str, ctx: Dict) -> None:
    """Cache a good result. Failed results are never cached as 'good'."""
    if ctx.get("ok"):
        _cache[source] = dict(ctx)


def _neutral(source: str, symbol: str, reason: str) -> Dict:
    """Fail-closed empty context."""
    return {
        "source": source,
        "symbol": symbol,
        "price": 0.0,
        "trend": "NEUTRAL",
        "change_pct": 0.0,
        "timestamp": time.time(),
        "ok": False,
        "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE (public, no keys)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_binance_context(symbol: str = "BTCUSDT") -> Dict:
    """
    Fetch BTC 15m trend + live price from Binance public API.
    Single request each for klines and ticker, timeout <= 5s, no retries.
    Falls back to the last good cached result (<60s old) on failure.
    """
    cached = _cached("binance")
    sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
    try:
        klines = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params={"symbol": sym, "interval": "15m", "limit": 10},
            timeout=HTTP_TIMEOUT,
        )
        ticker = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/ticker/price",
            params={"symbol": sym},
            timeout=HTTP_TIMEOUT,
        )
        if klines.status_code != 200 or ticker.status_code != 200:
            raise RuntimeError(f"HTTP {klines.status_code}/{ticker.status_code}")

        candles = klines.json()
        price = float(ticker.json().get("price", 0.0))
        if not candles or price <= 0:
            raise RuntimeError("empty/invalid payload")

        first_close = float(candles[0][4])
        last_close = float(candles[-1][4])
        change_pct = ((last_close - first_close) / first_close * 100.0) if first_close > 0 else 0.0

        if change_pct > WEAK_TREND_PCT:
            trend = "BULLISH"
        elif change_pct < -WEAK_TREND_PCT:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        ctx = {
            "source": "binance",
            "symbol": sym,
            "price": price,
            "trend": trend,
            "change_pct": round(change_pct, 3),
            "timestamp": time.time(),
            "ok": True,
        }
        _store("binance", ctx)
        return ctx
    except Exception as e:
        logger.warning(f"[CROSS-EXCHANGE] Binance context failed: {e}")
        if cached:
            logger.info("[CROSS-EXCHANGE] Using cached Binance context (<60s old)")
            return cached
        return _neutral("binance", sym, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# UPSTOX (optional — skips gracefully without credentials)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_upstox_context() -> Dict:
    """
    Fetch Indian market bias (Nifty 50) via existing upstox_data.py.
    Returns ok=False (silently skipped) when no token is configured or the
    API fails. Never raises.
    """
    cached = _cached("upstox")
    try:
        from upstox_data import get_upstox_data
        ud = get_upstox_data()
        if not getattr(ud, "_enabled", False):
            return _neutral("upstox", "NSE_INDEX|Nifty 50", "no credentials / disabled")

        quote = ud.get_full_quote("NSE_INDEX|Nifty 50") or {}
        ltp = float(quote.get("ltp", 0.0) or 0.0)
        change_pct = float(quote.get("change_pct", 0.0) or 0.0)
        if ltp <= 0:
            raise RuntimeError("no LTP returned")

        if change_pct > WEAK_TREND_PCT:
            trend = "BULLISH"
        elif change_pct < -WEAK_TREND_PCT:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        ctx = {
            "source": "upstox",
            "symbol": "NSE_INDEX|Nifty 50",
            "price": ltp,
            "trend": trend,
            "change_pct": round(change_pct, 3),
            "timestamp": time.time(),
            "ok": True,
        }
        _store("upstox", ctx)
        return ctx
    except Exception as e:
        logger.debug(f"[CROSS-EXCHANGE] Upstox context skipped: {e}")
        if cached:
            return cached
        return _neutral("upstox", "NSE_INDEX|Nifty 50", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# FUSION EVALUATION (pure function — fully unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cross_exchange(logic_signal: int,
                            binance_ctx: Optional[Dict] = None,
                            upstox_ctx: Optional[Dict] = None) -> Dict:
    """
    Compare cross-exchange context with the Delta-driven signal direction.

    logic_signal: 1 (CALL/BUY), -1 (PUT/SELL), 0 (no trade)

    Returns:
      {
        "active":     bool,   # any source contributed
        "adjustment": int,    # score delta (+ boost / - penalty)
        "veto":       bool,   # True only on STRONG Binance disagreement
        "notes":      [str],
      }

    Rules:
      - No signal (0) or no reachable source -> inactive, zero adjustment.
      - Binance agrees  -> +AGREE_BOOST.
      - Binance disagrees mildly -> -DISAGREE_PENALTY.
      - Binance disagrees strongly (|15m change| > STRONG_TREND_PCT) ->
        -STRONG_DISAGREE_PENALTY and veto=True (caller may veto the entry).
      - Binance unreachable (ok=False) -> skipped entirely, never blocks.
      - Upstox -> informational bias +/-UPSTOX_BIAS only, NEVER veto/blocks.
    """
    result = {"active": False, "adjustment": 0, "veto": False, "notes": []}
    if logic_signal == 0:
        return result

    def _dir(ctx):
        t = str(ctx.get("trend", "NEUTRAL")).upper()
        return 1 if t == "BULLISH" else (-1 if t == "BEARISH" else 0)

    # ── Binance (primary cross-exchange check) ──
    if binance_ctx and binance_ctx.get("ok"):
        b_dir = _dir(binance_ctx)
        strength = abs(float(binance_ctx.get("change_pct", 0.0)))
        if b_dir == logic_signal:
            result["adjustment"] += AGREE_BOOST
            result["active"] = True
            result["notes"].append(
                f"Binance 15m {binance_ctx['trend']} ({binance_ctx['change_pct']:+.2f}%) agrees (+{AGREE_BOOST}%)")
        elif b_dir == -logic_signal:
            result["active"] = True
            if strength >= STRONG_TREND_PCT:
                result["adjustment"] -= STRONG_DISAGREE_PENALTY
                result["veto"] = True
                result["notes"].append(
                    f"Binance 15m STRONGLY {binance_ctx['trend']} ({binance_ctx['change_pct']:+.2f}%) "
                    f"opposes signal (-{STRONG_DISAGREE_PENALTY}%, veto)")
            else:
                result["adjustment"] -= DISAGREE_PENALTY
                result["notes"].append(
                    f"Binance 15m {binance_ctx['trend']} ({binance_ctx['change_pct']:+.2f}%) "
                    f"opposes signal (-{DISAGREE_PENALTY}%)")
        else:
            result["notes"].append("Binance 15m NEUTRAL — no impact")
    else:
        result["notes"].append("Binance unreachable — cross-exchange check skipped (Delta primary)")

    # ── Upstox (informational only, never blocking) ──
    if upstox_ctx and upstox_ctx.get("ok"):
        u_dir = _dir(upstox_ctx)
        if u_dir == logic_signal:
            result["adjustment"] += UPSTOX_BIAS
            result["notes"].append(f"Upstox Nifty bias supports signal (+{UPSTOX_BIAS}% info)")
        elif u_dir == -logic_signal:
            result["adjustment"] -= UPSTOX_BIAS
            result["notes"].append(f"Upstox Nifty bias opposes signal (-{UPSTOX_BIAS}% info)")
    # missing creds / failure -> silently ignored (already covered by ok=False)

    result["active"] = result["active"] or bool(upstox_ctx and upstox_ctx.get("ok"))
    return result


def get_cross_exchange_perspective(logic_signal: int, symbol: str = "BTCUSDT") -> Dict:
    """
    One-call helper used by jarvis_FIXED.py:
    fetches Binance + Upstox context (fail-closed, cached) and evaluates
    them against the current Delta-driven signal direction.
    Never raises.
    """
    try:
        binance_ctx = fetch_binance_context(symbol)
    except Exception as e:  # defensive — fetchers already fail closed
        binance_ctx = _neutral("binance", symbol, str(e))
    try:
        upstox_ctx = fetch_upstox_context()
    except Exception as e:
        upstox_ctx = _neutral("upstox", "NSE_INDEX|Nifty 50", str(e))
    res = evaluate_cross_exchange(logic_signal, binance_ctx, upstox_ctx)
    res["binance"] = binance_ctx
    res["upstox"] = upstox_ctx
    return res


# Cross-exchange perspective tuning constant (exposed for the brain)
CROSS_EXCHANGE_WEIGHT = 1.1
