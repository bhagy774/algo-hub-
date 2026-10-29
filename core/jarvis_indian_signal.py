#!/usr/bin/env python3
"""
JARVIS Indian Market Signal Engine
====================================
Dual-Market Intelligence: Indian NSE + Crypto BTC

HOW IT WORKS:
  1. Every 5 min: fetch Nifty50, BankNifty live data (Upstox)
  2. Every 5 min: fetch BTC live data (Binance)
  3. Score both markets: momentum, volatility, trend clarity
  4. Pick the BEST market (highest score)
  5. Generate CALL / PUT signal with entry/target/SL
  6. Print beautiful signal card for manual trade on Upstox app

SIGNAL LOGIC (Indian Market):
  - Nifty/BankNifty 1min + 5min momentum
  - RSI-style overbought/oversold on 5min candles
  - Trend direction from EMA5 vs EMA20 proxy
  - Volatility (ATR) for TP/SL calculation
  - Market hours gate: 09:15–15:30 IST only

.env settings:
  UPSTOX_ENABLED=true
  UPSTOX_ACCESS_TOKEN=<your token>
  INDIAN_SIGNAL_INTERVAL=300     # 5 min
  INDIAN_SIGNAL_ENABLED=true
  INDIAN_INSTRUMENTS=NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank
"""

import os
import sys
import time
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("JarvisIndianSignal")

# ── ANSI Colors ────────────────────────────────────────────────
R   = '\033[91m'; G   = '\033[92m'; Y   = '\033[93m'
C   = '\033[96m'; W   = '\033[97m'; DG  = '\033[90m'
M   = '\033[95m'; BD  = '\033[1m';  RST = '\033[0m'

# ── Config ─────────────────────────────────────────────────────
INDIAN_SIGNAL_ENABLED  = os.environ.get("INDIAN_SIGNAL_ENABLED", "true").lower() == "true"
INDIAN_SIGNAL_INTERVAL = int(os.environ.get("INDIAN_SIGNAL_INTERVAL", "300"))
INDIAN_INSTRUMENTS     = os.environ.get(
    "INDIAN_INSTRUMENTS",
    "NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank"
).split(",")

# Crypto competitor
CRYPTO_SYMBOL = "BTCUSDT"

# Options sizing defaults
NIFTY_LOT_SIZE    = 50    # 1 lot = 50 units
BANKNIFTY_LOT_SIZE = 15   # 1 lot = 15 units


# ══════════════════════════════════════════════════════════════════
#  SECTION 1: TECHNICAL ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════

def _ema(prices: List[float], period: int) -> float:
    """Simple EMA calculation."""
    if not prices or len(prices) < 2:
        return prices[-1] if prices else 0.0
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(prices: List[float], period: int = 14) -> float:
    """RSI calculation from close prices."""
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(candles: List[Dict], period: int = 14) -> float:
    """Average True Range from OHLC candles."""
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i].get("high", 0)
        l = candles[i].get("low", 0)
        pc = candles[i - 1].get("close", 0)
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / min(period, len(trs))


def _momentum_score(prices: List[float]) -> float:
    """
    Score -100 to +100.
    Positive = bullish momentum, Negative = bearish.
    """
    if len(prices) < 10:
        return 0.0
    recent  = prices[-3:]
    earlier = prices[-8:-3]
    avg_recent  = sum(recent) / len(recent)
    avg_earlier = sum(earlier) / len(earlier)
    if avg_earlier == 0:
        return 0.0
    pct_change = (avg_recent - avg_earlier) / avg_earlier * 100
    return max(-100, min(100, pct_change * 50))   # scaled


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: UPSTOX CANDLE FETCHER
# ══════════════════════════════════════════════════════════════════

class UpstoxCandles:
    """Fetch historical candles from Upstox for technical analysis."""

    BASE = "https://api.upstox.com/v2"

    def __init__(self, token: str):
        self._token = token
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        })
        self._cache: Dict[str, Tuple[float, List]] = {}   # key -> (ts, candles)
        self._cache_ttl = 60.0   # 1 min cache for candles

    def get_intraday_candles(self, symbol: str, interval: str = "1minute",
                              limit: int = 30) -> List[Dict]:
        """
        Fetch intraday candles.
        interval: '1minute', '5minute', '15minute', '30minute', '1hour'
        """
        cache_key = f"{symbol}_{interval}"
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            r = self._session.get(
                f"{self.BASE}/historical-candle/intraday/{symbol}/{interval}",
                timeout=10,
            )
            if r.status_code == 200:
                raw = r.json().get("data", {}).get("candles", [])
                # Upstox returns: [timestamp, open, high, low, close, volume, oi]
                candles = []
                for c in raw[-limit:]:
                    if len(c) >= 5:
                        candles.append({
                            "ts":     c[0],
                            "open":   float(c[1]),
                            "high":   float(c[2]),
                            "low":    float(c[3]),
                            "close":  float(c[4]),
                            "volume": float(c[5]) if len(c) > 5 else 0.0,
                        })
                self._cache[cache_key] = (now, candles)
                return candles
            else:
                logger.debug(f"[IndianCandles] {symbol} {interval}: HTTP {r.status_code}")
        except Exception as e:
            logger.debug(f"[IndianCandles] Error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: SINGLE INSTRUMENT ANALYZER
# ══════════════════════════════════════════════════════════════════

class IndianInstrumentAnalyzer:
    """Analyzes one NSE index and generates CALL/PUT signal."""

    def __init__(self, symbol: str, upstox_data, candle_fetcher: UpstoxCandles):
        self.symbol   = symbol
        self.name     = symbol.split("|")[-1] if "|" in symbol else symbol
        self.ud       = upstox_data
        self.candles  = candle_fetcher
        self.lot_size = BANKNIFTY_LOT_SIZE if "Bank" in self.name else NIFTY_LOT_SIZE

    def analyze(self) -> Dict[str, Any]:
        """
        Full technical analysis → signal dict.
        Returns:
          {
            symbol, name, ltp, signal: CALL/PUT/WAIT,
            confidence, entry, target, stop_loss,
            direction_reason, score, lot_size
          }
        """
        result: Dict[str, Any] = {
            "symbol":    self.symbol,
            "name":      self.name,
            "ltp":       0.0,
            "signal":    "WAIT",
            "confidence": 0,
            "entry":     0.0,
            "target":    0.0,
            "stop_loss": 0.0,
            "direction_reason": "Insufficient data",
            "score":     0.0,
            "lot_size":  self.lot_size,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Live price
        ltp = self.ud.get_ltp(self.symbol)
        if ltp <= 0:
            result["direction_reason"] = "No live price (market closed?)"
            return result
        result["ltp"] = ltp

        # 2. Get 1min + 5min candles
        c1m = self.candles.get_intraday_candles(self.symbol, "1minute", 30)
        c5m = self.candles.get_intraday_candles(self.symbol, "5minute", 20)

        closes_1m = [c["close"] for c in c1m] if c1m else [ltp]
        closes_5m = [c["close"] for c in c5m] if c5m else [ltp]

        # 3. Technical indicators
        ema5_1m   = _ema(closes_1m, 5)
        ema20_1m  = _ema(closes_1m, 20)
        ema5_5m   = _ema(closes_5m, 5)
        ema20_5m  = _ema(closes_5m, 20)
        rsi_1m    = _rsi(closes_1m, 14)
        rsi_5m    = _rsi(closes_5m, 14)
        mom_1m    = _momentum_score(closes_1m)
        mom_5m    = _momentum_score(closes_5m)
        atr_val   = _atr(c1m, 14) if c1m else ltp * 0.001

        # 4. Scoring system
        score = 0.0
        reasons = []

        # EMA trend (1min)
        if ema5_1m > ema20_1m * 1.0002:
            score += 25; reasons.append("1m EMA5>EMA20 ↑")
        elif ema5_1m < ema20_1m * 0.9998:
            score -= 25; reasons.append("1m EMA5<EMA20 ↓")

        # EMA trend (5min) — stronger weight
        if ema5_5m > ema20_5m * 1.0003:
            score += 35; reasons.append("5m EMA5>EMA20 ↑ (strong)")
        elif ema5_5m < ema20_5m * 0.9997:
            score -= 35; reasons.append("5m EMA5<EMA20 ↓ (strong)")

        # RSI signals
        if rsi_1m < 35:
            score += 15; reasons.append(f"1m RSI={rsi_1m:.0f} oversold")
        elif rsi_1m > 65:
            score -= 15; reasons.append(f"1m RSI={rsi_1m:.0f} overbought")

        if rsi_5m < 35:
            score += 20; reasons.append(f"5m RSI={rsi_5m:.0f} oversold")
        elif rsi_5m > 65:
            score -= 20; reasons.append(f"5m RSI={rsi_5m:.0f} overbought")

        # Momentum
        if mom_5m > 10:
            score += 15; reasons.append(f"5m Momentum +{mom_5m:.1f}")
        elif mom_5m < -10:
            score -= 15; reasons.append(f"5m Momentum {mom_5m:.1f}")

        # Current price vs EMA20 (5min)
        if ltp > ema20_5m * 1.001:
            score += 10; reasons.append("Price above 5m EMA20")
        elif ltp < ema20_5m * 0.999:
            score -= 10; reasons.append("Price below 5m EMA20")

        result["score"] = round(score, 1)

        # 5. Determine signal
        if score >= 40:
            signal     = "CALL"
            confidence = min(95, 50 + int(score * 0.5))
            entry      = ltp
            target     = round(ltp + atr_val * 1.5, 2)
            sl         = round(ltp - atr_val * 0.8, 2)
        elif score <= -40:
            signal     = "PUT"
            confidence = min(95, 50 + int(abs(score) * 0.5))
            entry      = ltp
            target     = round(ltp - atr_val * 1.5, 2)
            sl         = round(ltp + atr_val * 0.8, 2)
        else:
            signal     = "WAIT"
            confidence = 0
            entry = target = sl = ltp

        result.update({
            "signal":           signal,
            "confidence":       confidence,
            "entry":            entry,
            "target":           target,
            "stop_loss":        sl,
            "direction_reason": " | ".join(reasons) if reasons else "No clear trend",
            "atr":              round(atr_val, 2),
            "rsi_1m":           round(rsi_1m, 1),
            "rsi_5m":           round(rsi_5m, 1),
            "ema5_5m":          round(ema5_5m, 2),
            "ema20_5m":         round(ema20_5m, 2),
        })
        return result


# ══════════════════════════════════════════════════════════════════
#  SECTION 4: CRYPTO SCORER (compare with Indian market)
# ══════════════════════════════════════════════════════════════════

def _score_crypto_market() -> Dict[str, Any]:
    """
    Score the crypto market (BTC) for movement strength.
    Returns score + signal direction.
    """
    try:
        from binance_data import get_binance_data
        bd = get_binance_data()
        btc_price = bd.get_live_price("BTCUSDT")
        candles   = bd.get_candles("BTCUSDT", "5m", limit=25)
        closes    = [c["close"] for c in candles] if candles else [btc_price]

        ema5   = _ema(closes, 5)
        ema20  = _ema(closes, 20)
        rsi    = _rsi(closes, 14)
        mom    = _momentum_score(closes)
        atr_c  = _atr(candles, 14) if candles else btc_price * 0.001

        score = 0.0
        if ema5 > ema20 * 1.001:   score += 35
        elif ema5 < ema20 * 0.999: score -= 35
        if rsi < 35:               score += 20
        elif rsi > 65:             score -= 20
        if mom > 10:               score += 15
        elif mom < -10:            score -= 15
        if btc_price > ema20:      score += 10
        elif btc_price < ema20:    score -= 10

        signal = "CALL" if score >= 40 else ("PUT" if score <= -40 else "WAIT")
        conf   = min(95, 50 + int(abs(score) * 0.5)) if signal != "WAIT" else 0

        return {
            "market":     "CRYPTO",
            "symbol":     "BTC/USDT",
            "ltp":        btc_price,
            "score":      round(abs(score), 1),
            "raw_score":  round(score, 1),
            "signal":     signal,
            "confidence": conf,
            "rsi":        round(rsi, 1),
            "atr":        round(atr_c, 2),
            "entry":      btc_price,
            "target":     round(btc_price + atr_c * 1.5, 2) if signal == "CALL" else round(btc_price - atr_c * 1.5, 2),
            "stop_loss":  round(btc_price - atr_c * 0.8, 2) if signal == "CALL" else round(btc_price + atr_c * 0.8, 2),
        }
    except Exception as e:
        logger.debug(f"[CryptoScore] Error: {e}")
        return {"market": "CRYPTO", "symbol": "BTC/USDT", "score": 0.0, "signal": "WAIT", "confidence": 0, "ltp": 0}


# ══════════════════════════════════════════════════════════════════
#  SECTION 5: DUAL MARKET COMPARATOR
# ══════════════════════════════════════════════════════════════════

class DualMarketComparator:
    """
    Compares Indian NSE vs Crypto market.
    Picks the BEST moving market and returns the winner's signal.
    """

    def __init__(self):
        self._upstox = None
        self._candles = None
        self._analyzers: List[IndianInstrumentAnalyzer] = []
        self._init()

    def _init(self):
        try:
            from upstox_data import get_upstox_data
            ud = get_upstox_data()
            if ud._enabled:
                self._upstox = ud
                token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
                self._candles = UpstoxCandles(token)
                for sym in INDIAN_INSTRUMENTS:
                    self._analyzers.append(
                        IndianInstrumentAnalyzer(sym.strip(), ud, self._candles)
                    )
                logger.info(f"[DualMarket] {len(self._analyzers)} Indian instruments loaded")
            else:
                logger.warning("[DualMarket] Upstox not enabled — Indian signals unavailable")
        except Exception as e:
            logger.warning(f"[DualMarket] Init error: {e}")

    def compare_and_pick(self) -> Dict[str, Any]:
        """
        Run analysis on both markets → return best signal.

        Returns:
        {
          winner: 'INDIAN' | 'CRYPTO' | 'NONE',
          winner_signal: { ... signal dict ... },
          indian_results: [ ... per-instrument dicts ... ],
          crypto_result:  { ... },
          reason: "why this market was picked"
        }
        """
        from upstox_data import is_market_open

        result = {
            "winner":         "NONE",
            "winner_signal":  {},
            "indian_results": [],
            "crypto_result":  {},
            "reason":         "",
            "timestamp":      datetime.now().isoformat(),
        }

        # ── Crypto Analysis ──────────────────────────────────────
        crypto = _score_crypto_market()
        result["crypto_result"] = crypto

        # ── Indian Market Analysis ───────────────────────────────
        indian_results = []
        best_indian: Optional[Dict] = None
        best_indian_score = 0.0

        if self._analyzers and is_market_open():
            for analyzer in self._analyzers:
                try:
                    sig = analyzer.analyze()
                    indian_results.append(sig)
                    abs_score = abs(sig.get("score", 0.0))
                    if abs_score > best_indian_score and sig["signal"] != "WAIT":
                        best_indian_score = abs_score
                        best_indian = sig
                except Exception as e:
                    logger.debug(f"[DualMarket] Analyze error for {analyzer.symbol}: {e}")
        elif not is_market_open():
            result["reason"] = "NSE market CLOSED — Crypto only mode"

        result["indian_results"] = indian_results

        # ── Pick Winner ──────────────────────────────────────────
        crypto_score  = abs(crypto.get("raw_score", 0.0))
        crypto_signal = crypto.get("signal", "WAIT")

        if best_indian and best_indian_score > 0:
            indian_score = best_indian_score

            # Both have signals — pick stronger one
            if crypto_signal != "WAIT" and crypto_score > 0:
                if indian_score >= crypto_score * 0.85:
                    # Indian market equally strong or stronger → prefer Indian
                    result["winner"]        = "INDIAN"
                    result["winner_signal"] = _to_unified_signal(best_indian, "INDIAN")
                    result["reason"]        = (
                        f"Indian market ({best_indian['name']}) score={indian_score:.0f} "
                        f"vs Crypto score={crypto_score:.0f} → Indian preferred"
                    )
                else:
                    result["winner"]        = "CRYPTO"
                    result["winner_signal"] = _to_unified_signal(crypto, "CRYPTO")
                    result["reason"]        = (
                        f"Crypto score={crypto_score:.0f} stronger than "
                        f"Indian score={indian_score:.0f}"
                    )
            else:
                # Only Indian has signal
                result["winner"]        = "INDIAN"
                result["winner_signal"] = _to_unified_signal(best_indian, "INDIAN")
                result["reason"]        = f"Only Indian market ({best_indian['name']}) has signal"

        elif crypto_signal != "WAIT":
            # Only crypto has signal (or Indian market closed)
            result["winner"]        = "CRYPTO"
            result["winner_signal"] = _to_unified_signal(crypto, "CRYPTO")
            result["reason"]        = "Crypto has signal; Indian market no signal or closed"
        else:
            result["winner"]  = "NONE"
            result["reason"]  = "Both markets neutral — WAIT"

        return result


def _to_unified_signal(src: Dict, market: str) -> Dict:
    """Normalize any signal dict to unified format."""
    return {
        "market":     market,
        "symbol":     src.get("symbol", ""),
        "name":       src.get("name", src.get("symbol", "")),
        "ltp":        src.get("ltp", 0.0),
        "signal":     src.get("signal", "WAIT"),
        "confidence": src.get("confidence", 0),
        "entry":      src.get("entry", 0.0),
        "target":     src.get("target", 0.0),
        "stop_loss":  src.get("stop_loss", 0.0),
        "reason":     src.get("direction_reason", ""),
        "rsi":        src.get("rsi_5m", src.get("rsi", 0.0)),
        "score":      src.get("score", 0.0),
        "atr":        src.get("atr", 0.0),
        "lot_size":   src.get("lot_size", None),
        "timestamp":  src.get("timestamp", datetime.now().isoformat()),
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 6: SIGNAL DISPLAY (Beautiful Console Card)
# ══════════════════════════════════════════════════════════════════

def print_dual_market_signal(result: Dict[str, Any]):
    """Print the full dual-market signal card to console."""
    now_str = datetime.now().strftime("%H:%M:%S IST")
    winner  = result.get("winner", "NONE")
    sig     = result.get("winner_signal", {})
    signal  = sig.get("signal", "WAIT")

    sig_col = G if signal == "CALL" else (R if signal == "PUT" else Y)
    mkt_col = M if winner == "INDIAN" else C

    print(f"\n{C}{'═'*80}{RST}")
    print(f"{C}║{RST}  {BD}JARVIS DUAL-MARKET SIGNAL ENGINE{RST}  [{DG}{now_str}{RST}]{' '*28}{C}║{RST}")
    print(f"{C}{'═'*80}{RST}")

    # ── Winner announcement ──────────────────────────────────────
    if winner == "NONE":
        print(f"  {Y}{BD}⏸  BOTH MARKETS NEUTRAL — WAIT. No trade recommended.{RST}")
    else:
        mkt_label = "🇮🇳  INDIAN NSE MARKET" if winner == "INDIAN" else "🪙  CRYPTO MARKET"
        print(f"  {mkt_col}{BD}BEST MARKET: {mkt_label}{RST}")
        print(f"  {DG}Reason: {result.get('reason', '')}{RST}")
        print(f"  {C}{'─'*76}{RST}")

        # ── Main Signal Box ──────────────────────────────────────
        print(f"\n  {sig_col}{BD}  ████  {signal}  ████{RST}  "
              f"{W}{sig.get('name', sig.get('symbol', ''))}{RST}  "
              f"@ {W}₹{sig['ltp']:,.2f}{RST}" if winner == "INDIAN" else
              f"\n  {sig_col}{BD}  ████  {signal}  ████{RST}  "
              f"{W}{sig.get('name', 'BTC/USDT')}{RST}  "
              f"@ {W}${sig['ltp']:,.2f}{RST}")

        print(f"\n  {BD}Confidence  :{RST} {sig_col}{BD}{sig.get('confidence', 0)}%{RST}")
        print(f"  {BD}Score       :{RST} {W}{sig.get('score', 0):.1f}/100{RST}")

        currency = "₹" if winner == "INDIAN" else "$"
        print(f"  {BD}Entry Zone  :{RST} {G}{currency}{sig.get('entry', 0):,.2f}{RST}")
        print(f"  {BD}Target (TP) :{RST} {G}{currency}{sig.get('target', 0):,.2f}{RST}  "
              f"{DG}(+ATR×1.5 = {currency}{sig.get('atr',0)*1.5:,.1f}){RST}")
        print(f"  {BD}Stop Loss   :{RST} {R}{currency}{sig.get('stop_loss', 0):,.2f}{RST}  "
              f"{DG}(-ATR×0.8 = {currency}{sig.get('atr',0)*0.8:,.1f}){RST}")

        if sig.get("lot_size"):
            print(f"  {BD}Lot Size    :{RST} {W}{sig['lot_size']} units/lot{RST}")

        print(f"\n  {BD}RSI         :{RST} {W}{sig.get('rsi', 0):.1f}{RST}")
        if sig.get("reason"):
            print(f"  {BD}Why         :{RST} {DG}{sig.get('reason', '')}{RST}")

        # ── Trade Instructions (for manual Upstox app trade) ─────
        if winner == "INDIAN":
            print(f"\n  {Y}{'─'*76}{RST}")
            print(f"  {Y}{BD}📱 MANUAL TRADE STEPS (Upstox App):{RST}")
            print(f"  {W}  1. Open Upstox → Options → {sig.get('name', 'Nifty')}{RST}")
            print(f"  {W}  2. Buy {sig_col}{BD}{signal}{RST}{W} option near strike ₹{sig.get('entry',0):,.0f}{RST}")
            print(f"  {W}  3. Set Target: ₹{sig.get('target',0):,.2f}  |  SL: ₹{sig.get('stop_loss',0):,.2f}{RST}")
            print(f"  {W}  4. Hold max 15–20 minutes{RST}")
            print(f"  {Y}{'─'*76}{RST}")

    # ── All Markets Scorecard ─────────────────────────────────────
    print(f"\n  {BD}MARKET SCORECARD{RST}")
    print(f"  {'Market':<20} {'Signal':<8} {'Score':>8} {'Confidence':>12} {'RSI':>6}")
    print(f"  {DG}{'─'*56}{RST}")

    for ind in result.get("indian_results", []):
        s_col = G if ind["signal"] == "CALL" else (R if ind["signal"] == "PUT" else DG)
        marker = " ◀ WINNER" if winner == "INDIAN" and ind.get("score") == sig.get("score") else ""
        print(f"  {M}🇮🇳 {ind.get('name',''):<17}{RST} "
              f"{s_col}{ind['signal']:<8}{RST} "
              f"{W}{abs(ind.get('score',0)):>8.1f}{RST} "
              f"{W}{ind.get('confidence',0):>11}%{RST} "
              f"{W}{ind.get('rsi_5m', 0):>6.1f}{RST}"
              f"{Y}{BD}{marker}{RST}")

    cr = result.get("crypto_result", {})
    c_col  = G if cr.get("signal") == "CALL" else (R if cr.get("signal") == "PUT" else DG)
    marker = " ◀ WINNER" if winner == "CRYPTO" else ""
    print(f"  {C}🪙 {'BTC/USDT':<17}{RST} "
          f"{c_col}{cr.get('signal','WAIT'):<8}{RST} "
          f"{W}{abs(cr.get('raw_score',0)):>8.1f}{RST} "
          f"{W}{cr.get('confidence',0):>11}%{RST} "
          f"{W}{cr.get('rsi',0):>6.1f}{RST}"
          f"{Y}{BD}{marker}{RST}")

    print(f"{C}{'═'*80}{RST}\n")


# ══════════════════════════════════════════════════════════════════
#  SECTION 7: MAIN ENGINE (background loop)
# ══════════════════════════════════════════════════════════════════

class IndianSignalEngine:
    """
    Background service that runs every INDIAN_SIGNAL_INTERVAL seconds.
    Compares Indian NSE + Crypto → prints winner signal card.
    Publishes to CognitiveBus if available.
    """

    def __init__(self, bus=None):
        self.bus       = bus
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._comparator = DualMarketComparator()
        self.last_result: Dict[str, Any] = {}
        self.signal_count = 0

    def start(self):
        if not INDIAN_SIGNAL_ENABLED:
            logger.info("[IndianSignal] Disabled via INDIAN_SIGNAL_ENABLED=false")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="JarvisIndianSignal"
        )
        self._thread.start()
        logger.info(f"[IndianSignal] Engine started (interval={INDIAN_SIGNAL_INTERVAL}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def run_now(self) -> Dict[str, Any]:
        """Manually trigger one analysis cycle."""
        return self._run_cycle()

    def get_latest(self) -> Dict[str, Any]:
        return self.last_result

    def _loop(self):
        time.sleep(5)   # brief startup delay
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"[IndianSignal] Loop error: {e}", exc_info=True)
            elapsed = 0
            while self._running and elapsed < INDIAN_SIGNAL_INTERVAL:
                time.sleep(2)
                elapsed += 2

    def _run_cycle(self) -> Dict[str, Any]:
        logger.info("[IndianSignal] Running dual-market comparison...")
        result = self._comparator.compare_and_pick()
        self.last_result = result
        self.signal_count += 1

        # Print to console
        print_dual_market_signal(result)

        # Publish to CognitiveBus
        if self.bus:
            try:
                self.bus.publish("DUAL_MARKET_SIGNAL", "IndianSignalEngine", result)
            except Exception:
                pass

        # Save to disk
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/jarvis_dual_market.json", "w", encoding="utf-8") as f:
                import json
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result


# ── Singleton ──────────────────────────────────────────────────────
_engine: Optional[IndianSignalEngine] = None


def get_indian_signal_engine(bus=None) -> IndianSignalEngine:
    global _engine
    if _engine is None:
        _engine = IndianSignalEngine(bus=bus)
    return _engine


# ── CLI Quick Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  JARVIS DUAL MARKET SIGNAL — LIVE TEST")
    print("=" * 60)
    engine = IndianSignalEngine()
    result = engine.run_now()
    print(f"\nWinner: {result.get('winner')}")
    print(f"Signal: {result.get('winner_signal', {}).get('signal')}")
    print(f"Reason: {result.get('reason')}")
