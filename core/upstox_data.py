#!/usr/bin/env python3
"""
Upstox Indian Market Data Wrapper for Jarvis
=============================================
- Live LTP (Last Traded Price) for NSE/BSE instruments
- Index prices: Nifty 50, Bank Nifty, Nifty IT, etc.
- Equity prices: any NSE stock by ISIN
- Options chain (NSE F&O) data
- Fully OPTIONAL: if token missing or expired, silently skips

Configuration via .env:
  UPSTOX_ACCESS_TOKEN=<your token>
  UPSTOX_ENABLED=true
  UPSTOX_INSTRUMENTS=NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank
  UPSTOX_DEFAULT_STOCK=NSE_EQ|INE002A01018   (RELIANCE)
  NSE_MARKET_OPEN=09:15
  NSE_MARKET_CLOSE=15:30
"""

import os
import time
import logging
import requests
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Any

import sys
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

logger = logging.getLogger("UpstoxData")

UPSTOX_BASE_URL = "https://api.upstox.com/v2"

# ── Config from .env ────────────────────────────────────────────────────────
UPSTOX_ENABLED       = os.environ.get("UPSTOX_ENABLED", "true").lower() == "true"
UPSTOX_ACCESS_TOKEN  = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
UPSTOX_INSTRUMENTS   = os.environ.get(
    "UPSTOX_INSTRUMENTS",
    "NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank"
).split(",")
UPSTOX_DEFAULT_STOCK = os.environ.get("UPSTOX_DEFAULT_STOCK", "NSE_EQ|INE002A01018")
NSE_OPEN_STR         = os.environ.get("NSE_MARKET_OPEN",  "09:15")
NSE_CLOSE_STR        = os.environ.get("NSE_MARKET_CLOSE", "15:30")


def _parse_time(t: str) -> dt_time:
    h, m = t.split(":")
    return dt_time(int(h), int(m))


def is_market_open() -> bool:
    """Check if NSE market hours (Mon-Fri, 09:15–15:30 IST)."""
    now = datetime.now()
    if now.weekday() >= 5:          # Sat/Sun
        return False
    t = now.time()
    return _parse_time(NSE_OPEN_STR) <= t <= _parse_time(NSE_CLOSE_STR)


# ─────────────────────────────────────────────────────────────────────────────

class UpstoxData:
    """
    Upstox Indian market data client.
    All methods return {} / 0.0 gracefully if token is missing or API fails.
    """

    def __init__(self, access_token: str = ""):
        self._token = access_token or UPSTOX_ACCESS_TOKEN
        self._enabled = UPSTOX_ENABLED and bool(self._token)
        self._session = requests.Session()
        if self._token:
            self._session.headers.update({
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "JarvisTradingSystem/3.0",
            })
        # Price cache: symbol -> (price, timestamp)
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 5.0          # seconds

        if self._enabled:
            logger.info("[Upstox] Data client initialized ✅")
        else:
            logger.info("[Upstox] Disabled (UPSTOX_ENABLED=false or no token)")

    # ──────────────────────────────────────────────────────────────────
    #  CORE: Live LTP  (Last Traded Price)
    # ──────────────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        """
        Get Last Traded Price for any Upstox instrument key.
        Examples:
          'NSE_INDEX|Nifty 50'
          'NSE_INDEX|Nifty Bank'
          'NSE_EQ|INE002A01018'    (RELIANCE)
          'NSE_FO|...'             (F&O contracts)

        Returns 0.0 on failure.
        """
        if not self._enabled:
            return 0.0

        # Cache check
        now = time.time()
        if symbol in self._cache:
            price, ts = self._cache[symbol]
            if now - ts < self._cache_ttl:
                return price

        try:
            r = self._session.get(
                f"{UPSTOX_BASE_URL}/market-quote/ltp",
                params={"symbol": symbol},
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json().get("data", {})
                # Key may use ':' instead of '|'
                key = symbol.replace("|", ":")
                price = float(data.get(key, {}).get("last_price", 0.0))
                self._cache[symbol] = (price, now)
                return price
            else:
                logger.debug(f"[Upstox] LTP {symbol}: HTTP {r.status_code} — {r.text[:120]}")
        except Exception as e:
            logger.debug(f"[Upstox] LTP error for {symbol}: {e}")
        return 0.0

    def get_multiple_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch LTP for multiple symbols in one API call (max 500).
        Returns dict: {symbol: price}
        """
        if not self._enabled or not symbols:
            return {}
        try:
            r = self._session.get(
                f"{UPSTOX_BASE_URL}/market-quote/ltp",
                params={"symbol": ",".join(symbols)},
                timeout=10,
            )
            if r.status_code == 200:
                raw = r.json().get("data", {})
                result = {}
                now = time.time()
                for sym in symbols:
                    key = sym.replace("|", ":")
                    price = float(raw.get(key, {}).get("last_price", 0.0))
                    result[sym] = price
                    self._cache[sym] = (price, now)
                return result
        except Exception as e:
            logger.debug(f"[Upstox] Multiple LTP error: {e}")
        return {}

    # ──────────────────────────────────────────────────────────────────
    #  CONVENIENCE: Named Indices
    # ──────────────────────────────────────────────────────────────────

    def get_nifty50(self) -> float:
        """Return Nifty 50 index level."""
        return self.get_ltp("NSE_INDEX|Nifty 50")

    def get_banknifty(self) -> float:
        """Return Bank Nifty index level."""
        return self.get_ltp("NSE_INDEX|Nifty Bank")

    def get_niftyit(self) -> float:
        """Return Nifty IT index level."""
        return self.get_ltp("NSE_INDEX|Nifty IT")

    def get_sensex(self) -> float:
        """Return BSE Sensex level."""
        return self.get_ltp("BSE_INDEX|SENSEX")

    def get_stock(self, symbol: str = None) -> float:
        """Return price of configured default stock (or any NSE_EQ symbol)."""
        return self.get_ltp(symbol or UPSTOX_DEFAULT_STOCK)

    # ──────────────────────────────────────────────────────────────────
    #  FULL QUOTE (OHLC + Volume)
    # ──────────────────────────────────────────────────────────────────

    def get_full_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get full OHLCV quote for a symbol.
        Returns dict with keys: open, high, low, close, volume, oi, ltp
        """
        if not self._enabled:
            return {}
        try:
            r = self._session.get(
                f"{UPSTOX_BASE_URL}/market-quote/quotes",
                params={"symbol": symbol},
                timeout=10,
            )
            if r.status_code == 200:
                raw = r.json().get("data", {})
                key = symbol.replace("|", ":")
                d = raw.get(key, {})
                ohlc = d.get("ohlc", {})
                depth = d.get("depth", {})
                return {
                    "symbol":       symbol,
                    "ltp":          d.get("last_price", 0.0),
                    "open":         ohlc.get("open", 0.0),
                    "high":         ohlc.get("high", 0.0),
                    "low":          ohlc.get("low", 0.0),
                    "close":        ohlc.get("close", 0.0),
                    "volume":       d.get("volume", 0),
                    "oi":           d.get("oi", 0),
                    "change":       d.get("net_change", 0.0),
                    "change_pct":   d.get("net_change", 0.0) / max(ohlc.get("close", 1), 1) * 100,
                    "bid":          depth.get("buy", [{}])[0].get("price", 0.0) if depth.get("buy") else 0.0,
                    "ask":          depth.get("sell", [{}])[0].get("price", 0.0) if depth.get("sell") else 0.0,
                    "market_open":  is_market_open(),
                    "timestamp":    datetime.now().isoformat(),
                }
        except Exception as e:
            logger.debug(f"[Upstox] Full quote error for {symbol}: {e}")
        return {}

    # ──────────────────────────────────────────────────────────────────
    #  OPTIONS CHAIN (NSE F&O)
    # ──────────────────────────────────────────────────────────────────

    def get_option_contracts(self, underlying: str = "NSE_INDEX|Nifty 50",
                              expiry: str = None) -> List[Dict]:
        """
        Fetch NSE F&O option contracts for an underlying.
        underlying: 'NSE_INDEX|Nifty 50' or 'NSE_INDEX|Nifty Bank'
        expiry: 'YYYY-MM-DD' format (None = nearest expiry)
        """
        if not self._enabled:
            return []
        try:
            params = {"instrument_key": underlying}
            if expiry:
                params["expiry_date"] = expiry
            r = self._session.get(
                f"{UPSTOX_BASE_URL}/option/chain",
                params=params,
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception as e:
            logger.debug(f"[Upstox] Options chain error: {e}")
        return []

    def get_option_expiries(self, underlying: str = "NSE_INDEX|Nifty 50") -> List[str]:
        """Get list of available option expiry dates."""
        if not self._enabled:
            return []
        try:
            r = self._session.get(
                f"{UPSTOX_BASE_URL}/option/contract",
                params={"instrument_key": underlying},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                expiries = sorted(set(c.get("expiry", "") for c in data if c.get("expiry")))
                return expiries
        except Exception as e:
            logger.debug(f"[Upstox] Expiries error: {e}")
        return []

    # ──────────────────────────────────────────────────────────────────
    #  MARKET SNAPSHOT (all configured instruments at once)
    # ──────────────────────────────────────────────────────────────────

    def get_market_snapshot(self) -> Dict[str, Any]:
        """
        Returns a comprehensive snapshot of all configured instruments.
        Used by JarvisMarketOracle for Indian market context.
        """
        snap: Dict[str, Any] = {
            "enabled":       self._enabled,
            "market_open":   is_market_open(),
            "timestamp":     datetime.now().isoformat(),
            "nifty50":       0.0,
            "banknifty":     0.0,
            "default_stock": 0.0,
            "instruments":   {},
        }

        if not self._enabled:
            return snap

        # Batch fetch all configured instruments
        all_syms = list(set(
            ["NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank", UPSTOX_DEFAULT_STOCK]
            + UPSTOX_INSTRUMENTS
        ))

        prices = self.get_multiple_ltp(all_syms)

        snap["nifty50"]       = prices.get("NSE_INDEX|Nifty 50", 0.0)
        snap["banknifty"]     = prices.get("NSE_INDEX|Nifty Bank", 0.0)
        snap["default_stock"] = prices.get(UPSTOX_DEFAULT_STOCK, 0.0)
        snap["instruments"]   = prices

        return snap

    # ──────────────────────────────────────────────────────────────────
    #  STATUS CHECK
    # ──────────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """Quick connectivity check — returns True if API is reachable with token."""
        if not self._enabled:
            return False
        price = self.get_nifty50()
        return price > 0.0

    def status(self) -> Dict[str, Any]:
        """Return status dict for display/logging."""
        nifty = self.get_nifty50()
        bank  = self.get_banknifty()
        return {
            "enabled":     self._enabled,
            "token_set":   bool(self._token),
            "market_open": is_market_open(),
            "nifty50":     nifty,
            "banknifty":   bank,
            "active":      nifty > 0,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_instance: Optional[UpstoxData] = None


def get_upstox_data() -> UpstoxData:
    """Return singleton UpstoxData instance."""
    global _instance
    if _instance is None:
        _instance = UpstoxData()
    return _instance


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("=" * 55)
    print("  UPSTOX DATA MODULE — SELF TEST")
    print("=" * 55)
    ud = UpstoxData()
    snap = ud.get_market_snapshot()
    print(f"\n✅ Market Open : {snap['market_open']}")
    print(f"✅ Nifty 50    : ₹{snap['nifty50']:,.2f}")
    print(f"✅ Bank Nifty  : ₹{snap['banknifty']:,.2f}")
    print(f"✅ Default Stock: ₹{snap['default_stock']:,.2f}")
    print(f"\nAll instruments:")
    for sym, price in snap["instruments"].items():
        print(f"   {sym:40s}  ₹{price:,.2f}")
    print("\n" + "=" * 55)
