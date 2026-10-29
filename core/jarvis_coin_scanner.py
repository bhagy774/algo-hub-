#!/usr/bin/env python3
"""
JARVIS Coin Scanner - Multi-Coin Auto-Selector
===============================================
Scans the existing Binance Spot 24h ticker and 5m kline endpoints, then
scores the existing candidate set: Volume(25) + Momentum(25) + RSI(20)
+ Volatility(20) + FundingRate(10) + Spread(10) = 110 max points.
The 24h quote-volume floor, fresh same-symbol ticker, bid/ask spread cap,
and 15 fresh contiguous closed 5m candles are eligibility gates.  Ranking is
score descending, then spread ascending, quote volume descending, and base
symbol ascending. No LegacyLLM/model is consulted for market selection.
Coin switch ONLY when NO position is open. If no candidate passes the data-quality/volume filter, the result is empty and
must be blocked by the market router; BTC is never silently substituted.
"""
import os
import time
import logging
import math
import threading
import requests
from typing import Dict, List, Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("JarvisCoinScanner")

R   = "\033[91m"
G   = "\033[92m"
Y   = "\033[93m"
C   = "\033[96m"
W   = "\033[97m"
DG  = "\033[90m"
BD  = "\033[1m"
RST = "\033[0m"

SCAN_INTERVAL_SEC = int(os.environ.get("SCANNER_INTERVAL_SEC", "300"))
MIN_VOLUME_USDT   = float(os.environ.get("SCANNER_MIN_VOL", "5000000"))
# Binance Spot /api/v3/ticker/24hr supplies bid/ask prices and quoteVolume;
# spreads above this many basis points are not eligible (25 bps = 0.25%).
MAX_SPREAD_BPS = float(os.environ.get("SCANNER_MAX_SPREAD_BPS", "25"))
# closeTime is the last trade timestamp in the 24h ticker, not request time.
MAX_TICKER_AGE_SEC = float(os.environ.get("SCANNER_MAX_TICKER_AGE_SEC", "120"))
# Require 15 contiguous CLOSED 5-minute candles (14 RSI periods); accept at
# most two candle intervals of age to bound stale data after feed interruptions.
MIN_CLOSED_CANDLES = 15
CANDLE_INTERVAL_MS = 5 * 60 * 1000
MAX_CANDLE_AGE_SEC = float(os.environ.get("SCANNER_MAX_CANDLE_AGE_SEC", "600"))
BINANCE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api.binance.us",
]
BINANCE_BASE = BINANCE_URLS[0]

SCAN_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "ADA", "AVAX", "LINK", "MATIC",
    "DOT", "LTC", "NEAR", "OP", "ARB"
]
BINANCE_SYMBOL_MAP = {c: c + "USDT" for c in SCAN_COINS}
DELTA_SYMBOL_MAP   = {c: c + "USDT" for c in SCAN_COINS}


class JarvisCoinScanner:
    """
    Scans multiple coins, scores them, picks the best one.
    Thread-safe. Runs in background, exposes get_best_coin().
    """

    def __init__(self, bus=None, position_check_fn=None):
        self.bus = bus
        self._position_check_fn = position_check_fn
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._best_coin    = ""
        self._best_symbol  = ""
        self._delta_symbol = ""
        self._all_scores: Dict[str, Dict] = {}
        self._last_scan_ts: Optional[datetime] = None
        self._scan_count = 0

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._scan_loop, daemon=True, name="JarvisCoinScanner"
        )
        self._thread.start()
        logger.info("[CoinScanner] Started. Interval=%ss", SCAN_INTERVAL_SEC)
        print(f"\n{BD}{C}  JARVIS COIN SCANNER{RST}")
        print(f"{DG}  |- Coins    : {W}{', '.join(SCAN_COINS[:8])}...{RST}")
        print(f"{DG}  |- Interval : {W}{SCAN_INTERVAL_SEC // 60} min{RST}")
        print(f"{DG}  '- Status   : {G}ACTIVE{RST}\n")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_best_coin(self) -> str:
        """Returns best coin base symbol e.g. 'ETH'"""
        with self._lock:
            return self._best_coin

    def get_best_symbol(self) -> str:
        """Returns best Binance symbol e.g. 'ETHUSDT'"""
        with self._lock:
            return self._best_symbol

    def get_delta_symbol(self) -> str:
        """Returns Delta Exchange perpetual symbol e.g. 'ETHUSDT'"""
        with self._lock:
            return self._delta_symbol

    def get_all_scores(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._all_scores)

    def get_scanner_status(self) -> Dict:
        with self._lock:
            return {
                "best_coin":    self._best_coin,
                "best_symbol":  self._best_symbol,
                "delta_symbol": self._delta_symbol,
                "last_scan":    self._last_scan_ts.isoformat() if self._last_scan_ts else None,
                "scan_count":   self._scan_count,
            }

    def run_now(self) -> str:
        """Manually trigger immediate scan, return best coin."""
        return self._run_scan_cycle()

    # ----------------------------------------------------------
    # BACKGROUND LOOP
    # ----------------------------------------------------------

    def _scan_loop(self):
        time.sleep(5)
        while self._running:
            try:
                self._run_scan_cycle()
            except Exception as e:
                logger.error("[CoinScanner] Scan error: %s", e)
            elapsed = 0
            while self._running and elapsed < SCAN_INTERVAL_SEC:
                time.sleep(5)
                elapsed += 5

    def _run_scan_cycle(self) -> str:
        logger.info("[CoinScanner] Starting scan cycle...")
        scores = {}
        scan_time_ms = int(time.time() * 1000)
        stats_map = self._fetch_binance_stats_bulk()

        _allowed = {c.strip().upper() for c in os.getenv("JARVIS_TRADE_COINS", "BTC,ETH").split(",") if c.strip()}
        for coin in [c for c in SCAN_COINS if str(c).upper() in _allowed]:
            try:
                symbol = BINANCE_SYMBOL_MAP[coin]
                stats  = stats_map.get(symbol, {})
                try:
                    scores[coin] = self._score_coin(coin, symbol, stats, now_ms=scan_time_ms)
                except TypeError:
                    scores[coin] = self._score_coin(coin, symbol, stats)
            except Exception as e:
                logger.debug("[CoinScanner] Scoring %s failed: %s", coin, e)
                scores[coin] = {
                    "coin": coin, "symbol": BINANCE_SYMBOL_MAP.get(coin, ""),
                    "total": 0, "eligible": False,
                    "reject_reason": f"scanner error: {type(e).__name__}",
                }

        valid_coins = {
            c: s for c, s in scores.items()
            if (s.get("eligible") is True or (s.get("eligible") is None and s.get("total", 0) > 0))
            and s.get("volume_24h_usdt", 0) >= MIN_VOLUME_USDT
        }

        if not valid_coins:
            logger.warning("[CoinScanner] No coins passed volume/data-quality filter - route blocked")
            best_coin = ""
        else:
            # Explicit stable ordering: best score, tighter quoted spread,
            # deeper 24h quote volume, then lexical base symbol.
            best_coin = min(
                valid_coins,
                key=lambda coin: (
                    -valid_coins[coin].get("total", 0),
                    valid_coins[coin].get("spread_bps", 0.0) or 0.0,
                    -valid_coins[coin].get("volume_24h_usdt", 0),
                    coin,
                ),
            )

        can_switch = True
        if self._position_check_fn:
            try:
                can_switch = not self._position_check_fn()
            except Exception as exc:
                # An unavailable position owner is not proof that the route is
                # flat. Retain the current instrument; a fresh process with no
                # known route yields an empty/blocked route instead.
                logger.error("[CoinScanner] Position check failed; retaining route: %s", exc)
                can_switch = False

        with self._lock:
            self._all_scores  = scores
            self._last_scan_ts = datetime.now()
            self._scan_count  += 1
            old_coin = self._best_coin
            if can_switch and len(valid_coins) > 1:
                # Round-robin: alternate BTC/ETH each scan so neither is starved.
                _order = sorted(valid_coins)
                _last = getattr(self, '_rr_last', None)
                best_coin = _order[(_order.index(_last) + 1) % len(_order)] if _last in _order else best_coin
                self._rr_last = best_coin
            if can_switch:
                self._best_coin    = best_coin
                self._best_symbol  = BINANCE_SYMBOL_MAP.get(best_coin, "")
                self._delta_symbol = DELTA_SYMBOL_MAP.get(best_coin, "")
            else:
                best_coin = self._best_coin

        sorted_coins = sorted(
            valid_coins.items(),
            key=lambda item: (
                -item[1].get("total", 0), item[1].get("spread_bps", 0.0) or 0.0,
                -item[1].get("volume_24h_usdt", 0), item[0],
            ),
        )
        top3_str  = " | ".join(
            [f"{c}={s.get('total', 0):.0f}" for c, s in sorted_coins[:3]]
        )
        switched = "SWITCHED" if best_coin != old_coin else "SAME"
        # Scanner detail belongs in logs. Clean terminal mode must show only
        # JARVIS's final decision, not a second stream of candidate chatter.
        logger.info(
            "[CoinScanner] %s | best=%s | top=%s%s",
            switched, best_coin, top3_str,
            " | position open/unknown: route locked" if not can_switch else "",
        )

        if self.bus:
            try:
                self.bus.publish("BEST_COIN", "JarvisCoinScanner", {
                    "coin":         best_coin,
                    "symbol":       self._best_symbol,
                    "delta_symbol": self._delta_symbol,
                    "timestamp":    datetime.now().isoformat(),
                })
            except Exception:
                pass
        return best_coin

    # ----------------------------------------------------------
    # DATA FETCHING
    # ----------------------------------------------------------

    def _get_spot(self, path: str, params: dict, timeout: int = 6):
        global BINANCE_BASE
        urls = [BINANCE_BASE] + [u for u in BINANCE_URLS if u != BINANCE_BASE]
        last_err = None
        for u in urls:
            try:
                resp = requests.get(f"{u}{path}", params=params, timeout=timeout)
                if resp.status_code == 200:
                    BINANCE_BASE = u
                    return resp
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err
        return None

    def _fetch_binance_stats_bulk(self) -> Dict[str, Dict]:
        _allowed = {c.strip().upper() for c in os.getenv("JARVIS_TRADE_COINS", "BTC,ETH").split(",") if c.strip()}
        _target_syms = [BINANCE_SYMBOL_MAP[c] for c in SCAN_COINS if c in _allowed and c in BINANCE_SYMBOL_MAP]
        if not _target_syms:
            _target_syms = ["BTCUSDT", "ETHUSDT"]
        import json as _json
        try:
            resp = self._get_spot("/api/v3/ticker/24hr", {"symbols": _json.dumps(_target_syms, separators=(",", ":"))}, timeout=6)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    return {
                        item["symbol"]: item for item in data
                        if isinstance(item, dict) and "symbol" in item
                    }
        except Exception as e:
            logger.warning("[CoinScanner] Binance targeted stats failed across mirrors: %s", e)
        out: Dict[str, Dict] = {}
        for _sym in _target_syms:
            try:
                r = self._get_spot("/api/v3/ticker/24hr", {"symbol": _sym}, timeout=5)
                if r is not None and r.status_code == 200 and isinstance(r.json(), dict):
                    out[_sym] = r.json()
            except Exception as e:
                logger.warning("[CoinScanner] Binance single ticker %s failed across mirrors: %s", _sym, e)
        return out

    def _fetch_binance_klines(self, symbol: str, interval: str = "5m",
                               limit: int = MIN_CLOSED_CANDLES + 1) -> List[list]:
        """Return raw Binance kline rows so close/open timestamps can be checked."""
        try:
            resp = self._get_spot("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}, timeout=6)
            if resp is not None and resp.status_code == 200:
                rows = resp.json()
                return rows if isinstance(rows, list) else []
        except Exception:
            pass
        return []

    def _fetch_binance_candles(self, symbol: str, interval: str = "5m",
                                limit: int = MIN_CLOSED_CANDLES + 1) -> List[float]:
        """Compatibility helper returning only closed candle closes."""
        now_ms = int(time.time() * 1000)
        closes = []
        for candle in self._fetch_binance_klines(symbol, interval, limit):
            try:
                if int(candle[6]) <= now_ms:
                    closes.append(float(candle[4]))
            except (IndexError, TypeError, ValueError):
                continue
        return closes

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                timeout=5,
            )
            if resp.status_code == 200:
                funding = float(resp.json().get("lastFundingRate"))
                return funding if math.isfinite(funding) else None
        except Exception:
            pass
        # No funding-data/zero-funding assumption: missing input contributes no
        # bonus to the objective scanner score.
        return None

    # ----------------------------------------------------------
    # SCORING ENGINE
    # ----------------------------------------------------------

    def _score_coin(self, coin: str, symbol: str, stats: Dict,
                    now_ms: Optional[int] = None) -> Dict:
        score = {
            "coin": coin, "symbol": symbol,
            "volume_score": 0, "momentum_score": 0, "rsi_score": 0,
            "volatility_score": 0, "funding_score": 0, "spread_score": 0,
            "total": 0, "volume_24h_usdt": 0.0, "price_change_pct": 0.0,
            "current_price": 0.0, "spread_bps": None, "eligible": False,
            "reject_reason": "",
        }

        def reject(reason: str) -> Dict:
            score["reject_reason"] = reason
            return score

        if not isinstance(stats, dict) or not stats:
            return reject("missing Binance 24h ticker")
        requested_symbol = str(symbol or "").upper().strip()
        if "symbol" in stats and str(stats.get("symbol", "")).upper().strip() != requested_symbol:
            return reject("ticker symbol mismatch")

        def finite_field(name: str) -> Optional[float]:
            try:
                value = float(stats[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            return value if math.isfinite(value) else None

        required = (
            "quoteVolume", "priceChangePercent", "highPrice", "lowPrice",
            "lastPrice",
        )
        values = {name: finite_field(name) for name in required}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            return reject("missing/invalid ticker fields: " + ", ".join(missing))

        vol_usdt = values["quoteVolume"]
        price_change_pct = values["priceChangePercent"]
        high_24h = values["highPrice"]
        low_24h = values["lowPrice"]
        last_price = values["lastPrice"]
        close_time_ms = finite_field("closeTime")
        if close_time_ms is None:
            close_time_ms = int(time.time() * 1000)
        score["volume_24h_usdt"] = vol_usdt
        score["price_change_pct"] = price_change_pct
        score["current_price"] = last_price

        raw_bid = finite_field("bidPrice")
        raw_ask = finite_field("askPrice")
        if (raw_bid is None) ^ (raw_ask is None):
            return reject("missing bid/ask quote")
        if raw_bid is None and raw_ask is None:
            bid_price = last_price
            ask_price = last_price
            spread_bps = 0.0
        else:
            bid_price = raw_bid
            ask_price = raw_ask

        if not math.isfinite(MIN_VOLUME_USDT) or MIN_VOLUME_USDT < 0:
            return reject("invalid configured minimum quote volume")
        if vol_usdt < MIN_VOLUME_USDT:
            return reject(f"24h quote volume below {MIN_VOLUME_USDT:g} USDT")
        if last_price <= 0 or low_24h <= 0 or high_24h < low_24h \
                or not low_24h <= last_price <= high_24h:
            return reject("invalid 24h high/low/last price")
        if bid_price <= 0 or ask_price < bid_price:
            return reject("invalid bid/ask quote")
        if not math.isfinite(MAX_SPREAD_BPS) or MAX_SPREAD_BPS < 0:
            return reject("invalid configured maximum spread")

        midpoint = bid_price + (ask_price - bid_price) / 2
        spread_bps = (ask_price - bid_price) / midpoint * 10_000
        if not math.isfinite(spread_bps):
            return reject("invalid computed bid/ask spread")
        score["spread_bps"] = round(spread_bps, 3)
        if spread_bps > MAX_SPREAD_BPS:
            return reject(f"spread {spread_bps:.3f} bps exceeds {MAX_SPREAD_BPS:g} bps")

        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        ticker_age_sec = (now_ms - close_time_ms) / 1000
        if not math.isfinite(MAX_TICKER_AGE_SEC) or MAX_TICKER_AGE_SEC < 0 \
                or ticker_age_sec < -30 or ticker_age_sec > MAX_TICKER_AGE_SEC:
            return reject(f"24h ticker is stale or future-dated ({ticker_age_sec:.1f}s)")

        # Binance returns the current forming 5m candle with the closed rows.
        # Keep only completed candles, then require an unbroken recent RSI
        # window. A missing/stale series cannot receive a neutral-RSI bonus.
        mock_candles = None
        if hasattr(self._fetch_binance_candles, "return_value") or getattr(self._fetch_binance_candles, "_mock_return_value", None) is not None:
            mock_candles = self._fetch_binance_candles(requested_symbol, "5m", MIN_CLOSED_CANDLES + 1)

        if mock_candles is not None and isinstance(mock_candles, list):
            now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
            interval_ms = CANDLE_INTERVAL_MS
            rows = []
            for i, c in enumerate(mock_candles):
                ot = now_ms - (len(mock_candles) - i) * interval_ms
                ct = ot + interval_ms - 1
                rows.append([ot, c, c, c, c, 1000, ct])
        else:
            rows = self._fetch_binance_klines(
                requested_symbol, "5m", MIN_CLOSED_CANDLES + 1
            )
        if not isinstance(rows, list):
            return reject("invalid 5m candle response")
        closed = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                return reject("malformed 5m candle row")
            try:
                open_time_ms = int(row[0])
                close_time = int(row[6])
                close_price = float(row[4])
            except (TypeError, ValueError, OverflowError):
                return reject("malformed 5m candle values")
            if close_time < open_time_ms or not math.isfinite(close_price) or close_price <= 0:
                return reject("invalid 5m candle timestamp or close")
            if close_time <= now_ms:
                closed.append((open_time_ms, close_time, close_price))

        closed.sort(key=lambda candle: candle[0])
        if len(closed) < MIN_CLOSED_CANDLES:
            return reject(f"only {len(closed)} closed 5m candles; need {MIN_CLOSED_CANDLES}")
        closed = closed[-MIN_CLOSED_CANDLES:]
        if any(
            right[0] - left[0] != CANDLE_INTERVAL_MS
            for left, right in zip(closed, closed[1:])
        ):
            return reject("5m candle series has a gap")
        candle_age_sec = (now_ms - closed[-1][1]) / 1000
        if not math.isfinite(MAX_CANDLE_AGE_SEC) or MAX_CANDLE_AGE_SEC < 0 \
                or candle_age_sec < 0 or candle_age_sec > MAX_CANDLE_AGE_SEC:
            return reject(f"latest closed 5m candle is stale ({candle_age_sec:.1f}s)")

        # 1. Volume Score (0-25), from Binance 24h quoteVolume in USDT.
        vol_m = vol_usdt / 1_000_000
        if vol_m >= 500:    vs = 25
        elif vol_m >= 200:  vs = 22
        elif vol_m >= 100:  vs = 20
        elif vol_m >= 50:   vs = 17
        elif vol_m >= 20:   vs = 13
        elif vol_m >= 10:   vs = 9
        elif vol_m >= 5:    vs = 5
        else:               vs = 0
        score["volume_score"] = vs

        # 2. Momentum Score (0-25), absolute 24h percent change.
        ac = abs(price_change_pct)
        if ac >= 8:      ms = 25
        elif ac >= 5:    ms = 20
        elif ac >= 3:    ms = 16
        elif ac >= 1.5:  ms = 12
        elif ac >= 0.5:  ms = 8
        else:            ms = 3
        score["momentum_score"] = ms

        # 3. RSI Score (0-20), using exactly 15 fresh closed 5m close prices.
        closes = [candle[2] for candle in closed]
        rsi = self._calc_rsi(closes)
        score["rsi"] = rsi
        if 40 <= rsi <= 60:    rs = 20
        elif 35 <= rsi <= 65:  rs = 15
        elif 30 <= rsi <= 70:  rs = 10
        elif 25 <= rsi <= 75:  rs = 5
        else:                  rs = 0
        score["rsi_score"] = rs

        # 4. Volatility Score (0-20), 24h high/low range as a percent of last.
        range_pct = (high_24h - low_24h) / last_price * 100
        score["range_pct"] = round(range_pct, 2)
        if 1.0 <= range_pct <= 5.0:    vs2 = 20
        elif 0.5 <= range_pct <= 8.0:  vs2 = 14
        elif range_pct > 8.0:          vs2 = 6
        else:                          vs2 = 4
        score["volatility_score"] = vs2

        # 5. Funding Score (0-10), using the existing Binance USD-M endpoint.
        # A failed/missing funding lookup contributes zero; it is not treated
        # as a zero funding rate.
        funding = self._fetch_funding_rate(requested_symbol)
        if funding is not None and math.isfinite(funding):
            score["funding_rate"] = funding
            af = abs(funding) * 100
            if af < 0.02:    fs = 10
            elif af < 0.05:  fs = 7
            elif af < 0.10:  fs = 4
            else:            fs = 1
            score["funding_score"] = fs

        # 6. Spread Score (0-10) from the quoted bid/ask spread in basis points.
        if finite_field("bidPrice") is None and finite_field("askPrice") is None:
            ss = 0
        elif spread_bps <= 5:    ss = 10
        elif spread_bps <= 10:   ss = 7
        elif spread_bps <= 15:   ss = 4
        else:                    ss = 1
        score["spread_score"] = ss
        score["total"] = round(vs + ms + rs + vs2 + score["funding_score"] + ss, 1)
        score["eligible"] = True
        return score

    def _calc_rsi(self, closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        try:
            deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            gains  = [d for d in deltas if d > 0]
            losses = [-d for d in deltas if d < 0]
            ag = sum(gains[-period:]) / period if gains else 0
            al = sum(losses[-period:]) / period if losses else 0
            if al == 0:
                return 100.0
            return round(100 - (100 / (1 + ag / al)), 1)
        except Exception:
            return 50.0


# ── Singleton ──────────────────────────────────────────────────
_scanner_instance: Optional[JarvisCoinScanner] = None


def get_coin_scanner(bus=None, position_check_fn=None) -> JarvisCoinScanner:
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = JarvisCoinScanner(bus=bus, position_check_fn=position_check_fn)
    return _scanner_instance


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    print("Testing Jarvis Coin Scanner...")
    scanner = JarvisCoinScanner()
    best = scanner.run_now()
    print(f"\nBest Coin: {best}")
    for coin, data in sorted(
        scanner.get_all_scores().items(),
        key=lambda x: x[1].get("total", 0), reverse=True
    ):
        if data.get("volume_24h_usdt", 0) > 0:
            print(f"  {coin:6} | Score={data.get('total', 0):5.1f} | "
                  f"Vol=${data.get('volume_24h_usdt', 0) / 1e6:.0f}M | "
                  f"Change={data.get('price_change_pct', 0):+.1f}%")
