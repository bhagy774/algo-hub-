#!/usr/bin/env python3
"""
Binance Public Data Wrapper for Jarvis
- No API Key required (public endpoints only)
- Replaces Delta Exchange for price + candle data
- Keeps Delta Exchange for options data + trade execution
- Auto-fallback to Binance mirrors on HTTP 451 (geo-restriction)
- Ultimate fallback to Bybit public API when all Binance endpoints are blocked

Endpoints used:
  Binance: GET /api/v3/ticker/price, /api/v3/klines
  Bybit:   GET /v5/market/tickers, /v5/market/kline  (fallback, no auth needed)
"""

import os
import requests
import logging
import time
import sys
from typing import List, Dict, Optional

from binance_timeframes import BINANCE_SPOT_TIMEFRAMES, history_limit, validate_interval

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Primary + fallback base URLs (tried in order on geo-restriction / failure)
# api.binance.us is the US-compliant endpoint; api1/api2/api3 are Binance CDN mirrors
BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",  # official public market-data mirror (same spot klines/tickers); works when api.binance.com is unreachable
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.us",       # Binance US (geo-unrestricted for most regions)
]
BINANCE_FAPI_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]
BINANCE_BASE_URL = BINANCE_BASE_URLS[0]  # kept for backward compat

# Bybit public API — used as ultimate fallback when Binance is geo-blocked
BYBIT_BASE_URL = "https://api.bybit.com"

# Bybit interval map (Binance format -> Bybit format)
BYBIT_INTERVAL_MAP = {
    # Only native Bybit linear intervals with exact duration equivalence.
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360",
    "12h": "720", "1d": "D", "1w": "W", "1M": "M",
}

# Optional: set BINANCE_PROXY env var to route via proxy
# Example: BINANCE_PROXY=socks5://127.0.0.1:1080  or  http://user:pass@host:port
_PROXY = os.environ.get("BINANCE_PROXY", "").strip()

# Map Jarvis resolution names → Binance interval names
RESOLUTION_MAP = {interval: interval for interval in BINANCE_SPOT_TIMEFRAMES}


class BinanceData:
    """
    Public Binance market data client.
    Drop-in replacement for DeltaExchangeData for price + candles.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (JarvisTradingSystem/2.0)"
        })
        # Apply proxy if configured
        if _PROXY:
            self.session.proxies = {"http": _PROXY, "https": _PROXY}
            # Never log credentials embedded in a proxy URL.
            from urllib.parse import urlsplit
            _parts = urlsplit(_PROXY)
            _proxy_label = f"{_parts.scheme}://{_parts.hostname or 'configured-proxy'}"
            if _parts.port:
                _proxy_label += f":{_parts.port}"
            logger.info("[BINANCE] Using configured proxy: %s", _proxy_label)

        self._last_price: float = 0.0
        self._last_price_ts: float = 0.0
        self._price_ttl: float = 3.0  # cache live price for 3 seconds max
        self._active_base_url: str = BINANCE_BASE_URLS[0]
        self._active_fapi_url: str = BINANCE_FAPI_URLS[0]
        self._last_source: str = "binance_spot"
        self._last_market_semantics: str = "spot"

        # Geo-block cache: if ALL Binance endpoints return 451, skip them for
        # _GEO_BLOCK_TTL seconds and go straight to Bybit / Delta fallback.
        self._binance_blocked: bool = False
        self._binance_blocked_until: float = 0.0
        self._GEO_BLOCK_TTL: float = 300.0   # re-check Binance every 5 minutes

    # ─────────────────────────────────────────────
    # INTERNAL: geo-aware GET with auto-fallback
    # ─────────────────────────────────────────────

    def _get(self, path: str, params: dict = None, timeout: int = 5,
             base_urls: List[str] = None) -> Optional[requests.Response]:
        """
        Try each base URL in order. On HTTP 451 (geo-restricted) or connection
        error, automatically fall back to the next mirror.
        If ALL mirrors are 451-blocked, caches the block state for 5 minutes
        to avoid spamming logs and wasting network calls.
        Returns the first successful Response, or None.
        """
        if base_urls is None:
            base_urls = BINANCE_BASE_URLS

        # Fast-path: if we already know Binance is geo-blocked, skip all mirrors
        if base_urls is BINANCE_BASE_URLS and self._binance_blocked:
            if time.time() < self._binance_blocked_until:
                return None   # still blocked — caller will use Bybit/Delta
            else:
                # Block TTL expired — try Binance again
                self._binance_blocked = False
                logger.info("[BINANCE] Re-checking Binance endpoints after geo-block cooldown...")

        all_451 = True
        if base_urls is BINANCE_BASE_URLS and getattr(self, '_active_base_url', None) in base_urls:
            base_urls = [self._active_base_url] + [u for u in base_urls if u != self._active_base_url]
        elif base_urls is BINANCE_FAPI_URLS and getattr(self, '_active_fapi_url', None) in base_urls:
            base_urls = [self._active_fapi_url] + [u for u in base_urls if u != self._active_fapi_url]
        for base_url in base_urls:
            try:
                url = f"{base_url}{path}"
                resp = self.session.get(url, params=params, timeout=timeout)
                # Retry every mirror for geo blocks, rate limits, and transient
                # server responses. Returning a 429/5xx here used to force an
                # unnecessary venue switch to Bybit.
                if resp.status_code in {429, 500, 502, 503, 504}:
                    logger.debug("[BINANCE] HTTP %s on %s — trying next mirror...", resp.status_code, base_url)
                    all_451 = False
                    continue
                if resp.status_code == 451:
                    logger.debug("[BINANCE] HTTP 451 on %s — trying next mirror...", base_url)
                    continue
                all_451 = False
                # Update active URL to the one that worked
                if base_urls is BINANCE_BASE_URLS:
                    self._active_base_url = base_url
                elif base_urls is BINANCE_FAPI_URLS:
                    self._active_fapi_url = base_url
                return resp
            except Exception as e:
                all_451 = False
                logger.debug(f"[BINANCE] {base_url} failed: {e} — trying next...")
                continue

        if all_451 and base_urls is BINANCE_BASE_URLS:
            if not self._binance_blocked:
                logger.warning("[BINANCE] All endpoints geo-blocked (HTTP 451). "
                               f"Switching to Bybit/Delta fallback for {int(self._GEO_BLOCK_TTL/60)} min.")
            self._binance_blocked = True
            self._binance_blocked_until = time.time() + self._GEO_BLOCK_TTL
        elif not self._binance_blocked:
            if base_urls is BINANCE_BASE_URLS:
                self._binance_blocked = True
                self._binance_blocked_until = time.time() + 60.0
            logger.warning("[BINANCE] All Binance endpoints unreachable/blocked — switching to Bybit fallback.")
        return None

    # ─────────────────────────────────────────────
    # INTERNAL: Bybit fallback helpers
    # ─────────────────────────────────────────────

    def _bybit_get_price(self, symbol: str) -> float:
        """Fetch live price from Bybit (fallback when Binance is geo-blocked)."""
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self.session.get(
                f"{BYBIT_BASE_URL}/v5/market/tickers",
                params={"category": "linear", "symbol": sym},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("result", {}).get("list", [])
                if items:
                    price = float(items[0].get("lastPrice", 0))
                    if price > 0:
                        self._last_source = "bybit_linear_perpetual"
                        self._last_market_semantics = "linear_perpetual"
                        logger.info("[BYBIT FALLBACK] %s = $%,.2f (linear perpetual; not Binance spot)", sym, price)
                        return price
        except Exception as e:
            logger.warning(f"[BYBIT FALLBACK] Price error: {e}")
        return 0.0

    def _bybit_get_candles(self, symbol: str, resolution: str, limit: int) -> List[Dict]:
        """Fetch OHLCV candles from Bybit (fallback when Binance is geo-blocked)."""
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        interval = BYBIT_INTERVAL_MAP.get(resolution)
        if interval is None:
            logger.warning("[BYBIT FALLBACK] Unsupported native interval %s; refusing substitute interval", resolution)
            return []
        try:
            resp = self.session.get(
                f"{BYBIT_BASE_URL}/v5/market/kline",
                params={"category": "linear", "symbol": sym,
                        "interval": interval, "limit": min(limit, 1000)},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("result", {}).get("list", [])
                # Bybit returns newest first — reverse to chronological order
                candles = []
                for k in reversed(raw):
                    # k = [startTime, open, high, low, close, volume, turnover]
                    candles.append({
                        "time":   int(k[0]) // 1000,
                        "open":   float(k[1]),
                        "high":   float(k[2]),
                        "low":    float(k[3]),
                        "close":  float(k[4]),
                        "volume": float(k[5]),
                    })
                self._last_source = "bybit_linear_perpetual"
                self._last_market_semantics = "linear_perpetual"
                for candle in candles:
                    candle["source"] = self._last_source
                    candle["market_semantics"] = self._last_market_semantics
                logger.info("[BYBIT FALLBACK] %s %s: %s candles (linear perpetual; not Binance spot)", sym, interval, len(candles))
                return candles
        except Exception as e:
            logger.warning(f"[BYBIT FALLBACK] Candles error: {e}")
        return []

    # ─────────────────────────────────────────────
    # 1. LIVE PRICE
    # ─────────────────────────────────────────────

    def get_live_price(self, symbol: str = "BTCUSDT") -> float:
        """
        Fetch real-time last traded price from Binance.
        Cached per-symbol for 3s to avoid hammering the API.
        """
        now = time.time()
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()

        # --- FIX: per-symbol cache (old code used single _last_price for ALL symbols) ---
        if not hasattr(self, '_price_cache'):
            self._price_cache = {}  # {sym: (price, ts)}
        cached = self._price_cache.get(sym)
        if cached and cached[0] > 0 and (now - cached[1]) < self._price_ttl:
            return cached[0]  # return cached price for THIS symbol

        try:
            resp = self._get("/api/v3/ticker/price", params={"symbol": sym}, timeout=5)
            if resp is not None and resp.status_code == 200:
                price = float(resp.json().get("price", 0))
                if price > 0:
                    self._price_cache[sym] = (price, now)
                    self._last_price = price  # keep for backward compat
                    self._last_price_ts = now
                    self._last_source = "binance_spot"
                    self._last_market_semantics = "spot"
                    logger.debug("[BINANCE PRICE] %s = $%,.2f (spot)", sym, price)
                    return price
            elif resp is not None:
                logger.warning(f"[BINANCE PRICE] HTTP {resp.status_code}: {resp.text[:80]}")
        except Exception as e:
            logger.warning(f"[BINANCE PRICE] Error: {e}")

        # Binance fully blocked — try Bybit
        bybit_price = self._bybit_get_price(sym)
        if bybit_price > 0:
            self._price_cache[sym] = (bybit_price, now)
            self._last_price = bybit_price
            self._last_price_ts = now
            return bybit_price

        # Return last known price for THIS symbol only (not another symbol's price)
        return cached[0] if cached else 0.0

    def get_last_source(self) -> Dict[str, str]:
        """Describe the venue/market semantics of the latest successful fetch."""
        return {"source": self._last_source, "market_semantics": self._last_market_semantics}

    def _tag_result(self, payload: Dict, source: Optional[str] = None,
                    market_semantics: Optional[str] = None) -> Dict:
        """Attach explicit venue semantics to every dict-shaped result."""
        result = dict(payload or {})
        result['source'] = source or self._last_source
        result['market_semantics'] = market_semantics or self._last_market_semantics
        return result

    # ─────────────────────────────────────────────
    # 2. HISTORICAL CANDLES (OHLCV)
    # ─────────────────────────────────────────────

    def get_historical_candles(
        self,
        symbol: str = "BTCUSDT",
        resolution: str = "1m",
        limit: int = 500,
        *,
        allow_fallback: bool = True,
    ) -> List[Dict]:
        """Fetch validated native Binance Spot OHLCV rows, paging up to ``limit``.

        Binance permits at most 1,000 klines per request. Requests are paged
        backward from the latest bar so large short-timeframe windows retain the
        freshest observations. Callers that require Spot provenance must set
        ``allow_fallback=False``; this prevents Bybit perpetual data being
        mistaken for Binance Spot.
        """
        validate_interval(resolution)
        if isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0:
            raise ValueError("candle limit must be a positive integer")
        requested = int(limit)
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        interval = RESOLUTION_MAP[resolution]
        self.last_candle_source = None
        candles_by_open = {}
        end_time = None
        remaining = requested
        try:
            while remaining > 0:
                page_limit = min(remaining, 1000)
                params = {"symbol": sym, "interval": interval, "limit": page_limit}
                if end_time is not None:
                    params["endTime"] = end_time
                resp = self._get("/api/v3/klines", params=params, timeout=10)
                if resp is None or resp.status_code != 200:
                    if resp is not None:
                        logger.warning("[BINANCE CANDLES] HTTP %s: %s", resp.status_code, resp.text[:100])
                    break
                raw = resp.json()
                if not isinstance(raw, list) or not raw:
                    break
                page_rows = []
                for k in raw:
                    if not isinstance(k, (list, tuple)) or len(k) < 6:
                        raise ValueError("malformed Binance kline row")
                    row = {"time": int(k[0]) // 1000, "open": float(k[1]),
                           "high": float(k[2]), "low": float(k[3]),
                           "close": float(k[4]), "volume": float(k[5])}
                    if not all(__import__("math").isfinite(row[key]) for key in ("open", "high", "low", "close", "volume")):
                        raise ValueError("non-finite Binance kline value")
                    page_rows.append(row)
                page_rows.sort(key=lambda row: row["time"])
                if not page_rows:
                    break
                before = len(candles_by_open)
                for row in page_rows:
                    candles_by_open[row["time"]] = row
                if len(candles_by_open) == before:
                    raise ValueError("Binance kline pagination made no progress")
                remaining = max(0, requested - len(candles_by_open))
                if remaining == 0 or len(page_rows) < page_limit:
                    break
                oldest_ms = min(row["time"] for row in page_rows) * 1000
                end_time = oldest_ms - 1
            candles = [candles_by_open[key] for key in sorted(candles_by_open)][-requested:]
            if len(candles) == requested:
                self.last_candle_source = "binance"
                self._last_source = "binance_spot"
                self._last_market_semantics = "spot"
                for candle in candles:
                    candle["source"] = "binance_spot"
                    candle["market_semantics"] = "spot"
                logger.info("[BINANCE] %s %s: %s/%s candles fetched (spot)", sym, interval, len(candles), requested)
                return candles
        except Exception as e:
            logger.warning("[BINANCE CANDLES] Error: %s", type(e).__name__)
        if not allow_fallback:
            return []

        # Legacy generic callers may use Bybit as an explicit last resort.
        # Strict Binance Spot consumers must always disable this fallback.
        fallback_limit = min(requested, 1000)
        candles = self._bybit_get_candles(symbol, resolution, fallback_limit)
        self.last_candle_source = "bybit" if candles else None
        return candles

    def get_historical_candles_with_metadata(
        self, *, symbol: str, resolution: str, limit: int
    ) -> Dict[str, object]:
        """Return candle rows with response-local, row-verified provenance.

        Strict consumers use this result instead of reading mutable client-wide
        ``last_candle_source`` after a request, which can race with other threads.
        """
        candles = self.get_historical_candles(
            symbol=symbol, resolution=resolution, limit=limit, allow_fallback=False
        )
        verified = bool(candles) and all(
            row.get("source") == "binance_spot" and row.get("market_semantics") == "spot"
            for row in candles if isinstance(row, dict)
        ) and all(isinstance(row, dict) for row in candles)
        return {
            "candles": candles if verified else [],
            "source": "binance" if verified else "unavailable",
            "market_type": "spot" if verified else "unavailable",
            "symbol": symbol,
        }

    # ─────────────────────────────────────────────
    # 3. BID / ASK PRICE (for Part 12)
    # ─────────────────────────────────────────────

    def get_bid_ask(self, symbol: str = "BTCUSDT") -> dict:
        """
        Fetch real-time best Bid and Ask price from Binance.
        Uses /api/v3/ticker/bookTicker — no API key needed.
        Returns: { "bid": float, "ask": float, "spread": float }
        Used by: Part 12 (GPUOrderExecutionEngine)
        """
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self._get("/api/v3/ticker/bookTicker", params={"symbol": sym}, timeout=5)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                bid = float(data.get("bidPrice", 0))
                ask = float(data.get("askPrice", 0))
                spread = round(ask - bid, 2)
                logger.debug(f"[BINANCE BID/ASK] Bid: ${bid:,.2f} | Ask: ${ask:,.2f} | Spread: ${spread}")
                self._last_source = "binance_spot"
                self._last_market_semantics = "spot"
                return self._tag_result({"bid": bid, "ask": ask, "spread": spread})
            elif resp is not None:
                logger.warning(f"[BINANCE BID/ASK] HTTP {resp.status_code} for symbol {sym}")
        except Exception as e:
            logger.warning(f"[BINANCE BID/ASK] Error: {e}")

        # Fallback: mark quote unavailable (never fabricate valid zero spread)
        price = self.get_live_price(symbol)
        self._last_source = "unavailable"
        self._last_market_semantics = "unavailable"
        return {
            "bid": None,
            "ask": None,
            "spread": None,
            "fallback_price": price,
            "source": "unavailable",
            "market_semantics": "unavailable",
        }

    # ─────────────────────────────────────────────
    # 4. MULTI-TIMEFRAME FETCH (MTF)
    # ─────────────────────────────────────────────

    def fetch_mtf_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframes: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """Fetch complete native Binance Spot history for the requested frames.

        By default this uses every native Spot interval and the reviewed
        timeframe-specific closed-history profile. Supplying ``limit`` retains
        an explicit same-count override for compatibility. The low-level method
        returns raw provider rows (including any forming row); live decisions
        must use ``DirectCandleCache`` to separate and validate forming candles.
        A missing/short frame raises instead of silently returning a partial MTF
        bundle. No Bybit/Delta substitution is allowed in this MTF helper.
        """
        selected = list(BINANCE_SPOT_TIMEFRAMES if timeframes is None else timeframes)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("timeframes must be a non-empty unique sequence")

        result = {}
        for tf in selected:
            validate_interval(tf)
            if limit is None:
                closed_history = history_limit(tf)
            else:
                if isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0:
                    raise ValueError("candle limit must be a positive integer")
                closed_history = int(limit)
            # Raw Klines may include the still-forming latest bar. Request one
            # extra row so callers can retain `closed_history` completed bars
            # while keeping the current candle separate.
            requested = closed_history + 1
            candles = self.get_historical_candles(
                symbol, tf, requested, allow_fallback=False
            )
            if len(candles) != requested:
                raise RuntimeError(f"incomplete native Binance Spot history for {tf}: {len(candles)}/{requested} rows (need {closed_history} closed plus current)")
            result[tf] = candles
        return result

    # ─────────────────────────────────────────────
    # 5. MARKET MICROSTRUCTURE (Funding, OI, Depth)
    # ─────────────────────────────────────────────

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Fetch current funding rate and countdown to next funding reset.
        Uses Binance Futures public API (fapi.binance.com).
        """
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self._get("/fapi/v1/premiumIndex", params={"symbol": sym},
                             timeout=5, base_urls=BINANCE_FAPI_URLS)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                rate = float(data.get("lastFundingRate", 0.0))
                next_time_ms = int(data.get("nextFundingTime", 0))
                mark_price = float(data.get("markPrice", 0.0))
                now_ms = int(time.time() * 1000)
                countdown_sec = max(0, (next_time_ms - now_ms) // 1000)
                hours = countdown_sec // 3600
                minutes = (countdown_sec % 3600) // 60

                sentiment = "BULLISH" if rate > 0.0001 else ("BEARISH" if rate < -0.0001 else "NEUTRAL")
                self._last_source = "binance_futures"
                self._last_market_semantics = "perpetual"
                return self._tag_result({
                    "funding_rate": rate,
                    "funding_rate_pct": round(rate * 100, 4),
                    "next_funding_time": next_time_ms,
                    "countdown_str": f"{hours:02d}h {minutes:02d}m",
                    "mark_price": mark_price,
                    "sentiment": sentiment,
                    "description": f"{rate * 100:+.4f}% ({sentiment})"
                })
        except Exception as e:
            logger.warning(f"[BINANCE FUNDING] Error: {e}")

        return self._tag_result({
            "funding_rate": 0.0001,
            "funding_rate_pct": 0.01,
            "countdown_str": "unknown",
            "mark_price": self._last_price,
            "sentiment": "NEUTRAL",
            "description": "+0.0100% (NEUTRAL)"
        }, source="unavailable", market_semantics="unavailable")

    def get_open_interest(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Fetch futures Open Interest from Binance Futures.
        """
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self._get("/fapi/v1/openInterest", params={"symbol": sym},
                             timeout=5, base_urls=BINANCE_FAPI_URLS)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                oi_contracts = float(data.get("openInterest", 0.0))
                self._last_source = "binance_futures"
                self._last_market_semantics = "perpetual"
                return self._tag_result({
                    "open_interest": oi_contracts,
                    "symbol": sym,
                    "timestamp": int(data.get("time", time.time() * 1000))
                })
        except Exception as e:
            logger.warning(f"[BINANCE OI] Error: {e}")

        return self._tag_result({"open_interest": 0.0, "symbol": sym, "timestamp": int(time.time() * 1000)}, source="unavailable", market_semantics="unavailable")

    def get_order_book_depth(self, symbol: str = "BTCUSDT", limit: int = 50) -> Dict:
        """
        Fetch order book depth and compute bid/ask volume and order book imbalance.
        """
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self._get("/api/v3/depth",
                             params={"symbol": sym, "limit": limit}, timeout=5)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                total_bid_vol = sum(float(b[1]) for b in bids)
                total_ask_vol = sum(float(a[1]) for a in asks)
                denom = total_bid_vol + total_ask_vol
                imbalance_pct = round(((total_bid_vol - total_ask_vol) / denom) * 100, 2) if denom > 0 else 0.0

                top_bid = float(bids[0][0]) if bids else 0.0
                top_ask = float(asks[0][0]) if asks else 0.0

                # Find major wall in top levels
                max_bid = max(bids, key=lambda x: float(x[1])) if bids else [0, 0]
                max_ask = max(asks, key=lambda x: float(x[1])) if asks else [0, 0]

                self._last_source = "binance_spot"
                self._last_market_semantics = "spot"
                return self._tag_result({
                    "imbalance_pct": imbalance_pct,
                    "total_bid_vol": round(total_bid_vol, 2),
                    "total_ask_vol": round(total_ask_vol, 2),
                    "top_bid": top_bid,
                    "top_ask": top_ask,
                    "major_bid_wall": {"price": float(max_bid[0]), "qty": float(max_bid[1])},
                    "major_ask_wall": {"price": float(max_ask[0]), "qty": float(max_ask[1])},
                    "bias": "BULLISH" if imbalance_pct > 5 else ("BEARISH" if imbalance_pct < -5 else "NEUTRAL")
                })
        except Exception as e:
            logger.warning(f"[BINANCE DEPTH] Error: {e}")

        return self._tag_result({
            "imbalance_pct": 0.0,
            "total_bid_vol": 0.0,
            "total_ask_vol": 0.0,
            "top_bid": self._last_price,
            "top_ask": self._last_price,
            "major_bid_wall": {"price": 0.0, "qty": 0.0},
            "major_ask_wall": {"price": 0.0, "qty": 0.0},
            "bias": "NEUTRAL"
        }, source="unavailable", market_semantics="unavailable")

    def get_24h_stats(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Fetch 24h ticker price change and volume statistics.
        """
        sym = symbol.upper().replace("USD", "USDT") if "USDT" not in symbol.upper() else symbol.upper()
        try:
            resp = self._get("/api/v3/ticker/24hr", params={"symbol": sym}, timeout=5)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                self._last_source = "binance_spot"
                self._last_market_semantics = "spot"
                return self._tag_result({
                    "price_change_pct": float(data.get("priceChangePercent", 0.0)),
                    "volume_24h_btc": float(data.get("volume", 0.0)),
                    "volume_24h_usdt": float(data.get("quoteVolume", 0.0)),
                    "high_24h": float(data.get("highPrice", 0.0)),
                    "low_24h": float(data.get("lowPrice", 0.0)),
                })
        except Exception as e:
            logger.warning(f"[BINANCE 24H] Error: {e}")

        return self._tag_result({
            "price_change_pct": 0.0,
            "volume_24h_btc": 0.0,
            "volume_24h_usdt": 0.0,
            "high_24h": self._last_price,
            "low_24h": self._last_price,
        }, source="unavailable", market_semantics="unavailable")


# ─────────────────────────────────────────────
# Singleton instance (import and reuse)
# ─────────────────────────────────────────────
_binance_instance: Optional[BinanceData] = None


def get_binance_data() -> BinanceData:
    global _binance_instance
    if _binance_instance is None:
        _binance_instance = BinanceData()
    return _binance_instance


def get_btc_price_binance(symbol: str = "BTCUSDT") -> float:
    return get_binance_data().get_live_price(symbol)


def get_ohlcv_binance(symbol: str = "BTCUSDT", resolution: str = "5m", limit: int = 100):
    candles = get_binance_data().get_historical_candles(symbol, resolution, limit)
    try:
        import pandas as pd
        if candles:
            return pd.DataFrame(candles)
    except Exception:
        pass
    return candles


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    bd = BinanceData()
    price = bd.get_live_price("BTCUSDT")
    print(f"✅ BTC Live Price: ${price:,.2f}")
    candles = bd.get_historical_candles("BTCUSDT", "1m", 5)
    print(f"✅ Last 5 candles: {[c['close'] for c in candles]}")
    funding = bd.get_funding_rate("BTCUSDT")
    print(f"✅ Funding Rate: {funding['description']} (Next in {funding['countdown_str']})")
    depth = bd.get_order_book_depth("BTCUSDT", 20)
    print(f"✅ Order Book Imbalance: {depth['imbalance_pct']:+.2f}% ({depth['bias']})")
    stats = bd.get_24h_stats("BTCUSDT")
    print(f"✅ 24h Vol: ${stats['volume_24h_usdt']/1e9:.2f}B (High: ${stats['high_24h']:,.0f}, Low: ${stats['low_24h']:,.0f})")
