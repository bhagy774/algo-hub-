import importlib
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jarvis_coin_scanner
from jarvis_coin_scanner import JarvisCoinScanner
from jarvis_market_router import MarketRouter


def _stats(symbol, *, quote_volume=8_000_000, spread_bps=5, change=4.0, close_time=None):
    mid = 100.0
    half_spread = mid * spread_bps / 20_000
    return {
        "symbol": symbol,
        "quoteVolume": str(quote_volume),
        "priceChangePercent": str(change),
        "highPrice": "110",
        "lowPrice": "90",
        "lastPrice": str(mid),
        "bidPrice": str(mid - half_spread),
        "askPrice": str(mid + half_spread),
        "closeTime": int(time.time() * 1000) if close_time is None else close_time,
    }


def _closed_5m_candles(now_ms=None, closes=None):
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    current_open = (now_ms // 300_000) * 300_000
    last_closed_open = current_open - 300_000
    closes = closes or [100.0 + i * 0.01 for i in range(16)]
    rows = []
    first_open = last_closed_open - (len(closes) - 1) * 300_000
    for i, close in enumerate(closes):
        opened = first_open + i * 300_000
        rows.append([opened, "100", "101", "99", str(close), "1000", opened + 299_999])
    return rows


def _scanner(stats_by_symbol, *, position_check_fn=None, candles_by_symbol=None, scanner_class=JarvisCoinScanner):
    scanner = scanner_class(position_check_fn=position_check_fn)
    candles_by_symbol = candles_by_symbol or {}
    scanner._fetch_binance_stats_bulk = lambda: stats_by_symbol
    # The legacy method remains patched so this test is network-free when run
    # against the pre-change scanner (test-first regression proof).
    scanner._fetch_binance_candles = lambda *a, **k: [100.0] * 16
    scanner._fetch_binance_klines = lambda symbol, *a, **k: candles_by_symbol.get(
        symbol, _closed_5m_candles()
    )
    scanner._fetch_funding_rate = lambda *a, **k: 0.0
    return scanner


class CoinScannerTests(unittest.TestCase):
    def test_candidate_requires_fresh_same_symbol_market_and_candle_data(self):
        now_ms = int(time.time() * 1000)
        stats = {
            "BTCUSDT": _stats("BTCUSDT", close_time=now_ms - 10 * 60_000),
            "ETHUSDT": _stats("BTCUSDT"),  # wrong symbol under the ETH key
            "SOLUSDT": _stats("SOLUSDT", spread_bps=100),  # 1% spread, over cap
            "BNBUSDT": _stats("BNBUSDT"),
            "ADAUSDT": _stats("ADAUSDT"),
            "DOGEUSDT": _stats("DOGEUSDT"),
            "XRPUSDT": _stats("XRPUSDT"),
        }
        del stats["XRPUSDT"]["askPrice"]  # required spread input is absent
        stale_candles = _closed_5m_candles(now_ms=now_ms - 90 * 60_000)
        scanner = _scanner(
            stats,
            candles_by_symbol={
                "BNBUSDT": [],
                "ADAUSDT": stale_candles,
                "DOGEUSDT": _closed_5m_candles()[:14],
            },
        )
        # BTC is stale; ETH is not the requested contract; SOL is too wide;
        # BNB has no sufficient candles; ADA's candles are stale; DOGE has too
        # few closed candles. None is allowed to route.
        self.assertEqual(scanner.run_now(), "")
        scores = scanner.get_all_scores()
        self.assertFalse(any(item.get("eligible") for item in scores.values()))
        for coin in ("BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "XRP"):
            self.assertTrue(scores[coin]["reject_reason"], coin)

    def test_uses_narrow_spread_and_stable_coin_name_tie_break(self):
        stats = {
            "ETHUSDT": _stats("ETHUSDT", spread_bps=20),
            "BNBUSDT": _stats("BNBUSDT", spread_bps=5),
        }
        scanner = _scanner(stats)
        self.assertEqual(scanner.run_now(), "BNB")
        first = scanner.get_all_scores()
        self.assertLess(first["BNB"]["spread_bps"], first["ETH"]["spread_bps"])
        self.assertGreater(first["BNB"]["spread_score"], first["ETH"]["spread_score"])

        # Exact score/volume/spread ties resolve lexically, not by SCAN_COINS order.
        tie_stats = {"ETHUSDT": _stats("ETHUSDT"), "BNBUSDT": _stats("BNBUSDT")}
        tie_scanner = _scanner(tie_stats)
        self.assertEqual(tie_scanner.run_now(), "BNB")

    def test_position_check_error_does_not_switch_the_existing_coin(self):
        scanner = _scanner({"ETHUSDT": _stats("ETHUSDT")})
        self.assertEqual(scanner.run_now(), "ETH")

        def position_owner_unavailable():
            raise RuntimeError("owner registry unavailable")

        scanner._position_check_fn = lambda: True  # an owned position is open
        scanner._fetch_binance_stats_bulk = lambda: {
            "SOLUSDT": _stats("SOLUSDT", change=9)
        }
        self.assertEqual(scanner.run_now(), "ETH")
        self.assertEqual(scanner.get_best_symbol(), "ETHUSDT")

        scanner._position_check_fn = position_owner_unavailable
        self.assertEqual(scanner.run_now(), "ETH")
        self.assertEqual(scanner.get_best_symbol(), "ETHUSDT")


class _Delta:
    def get_available_symbols(self):
        return ["ETHUSDT"]

    def get_options_chain(self, _base):
        return {}


class DeterministicRouterTests(unittest.TestCase):
    def test_production_router_selects_without_legacy_llm(self):
        with patch.dict(os.environ, {"JARVIS_MULTI_MARKET": "1"}):
            scanner_module = importlib.reload(jarvis_coin_scanner)
            scanner = _scanner(
                {"ETHUSDT": _stats("ETHUSDT")},
                scanner_class=scanner_module.JarvisCoinScanner,
            )
            route = MarketRouter(scanner=scanner, delta_client=_Delta()).select_crypto()
        self.assertEqual(route.status, "READY")
        self.assertEqual(route.symbol, "ETHUSDT")
        self.assertEqual(scanner.get_best_coin(), "ETH")


    def test_market_router_keeps_actual_route_while_an_owned_position_is_open(self):
        with patch.dict(os.environ, {"JARVIS_MULTI_MARKET": "1"}):
            scanner = _scanner({"ETHUSDT": _stats("ETHUSDT")})
            router = MarketRouter(scanner=scanner, delta_client=_Delta())
            first = router.select_crypto()
            self.assertEqual(first.symbol, "ETHUSDT")
            scanner._fetch_binance_stats_bulk = lambda: {"SOLUSDT": _stats("SOLUSDT", change=9)}
            locked = router.select_crypto(has_open_position=True)
            self.assertEqual(locked.symbol, "ETHUSDT")
            self.assertEqual(scanner.get_scanner_status()["scan_count"], 1)

    def test_no_candidate_blocks_real_market_router(self):
        with patch.dict(os.environ, {"JARVIS_MULTI_MARKET": "1"}):
            scanner = _scanner({})
            route = MarketRouter(scanner=scanner, delta_client=_Delta()).select_crypto()
            self.assertEqual(route.status, "BLOCKED")
            self.assertEqual(route.symbol, "")

    def test_scanner_exception_fails_closed_at_real_market_router(self):
        with patch.dict(os.environ, {"JARVIS_MULTI_MARKET": "1"}):
            scanner = _scanner({})
            scanner._fetch_binance_stats_bulk = Mock(side_effect=RuntimeError("feed down"))
            route = MarketRouter(scanner=scanner, delta_client=_Delta()).select_crypto()
            self.assertEqual(route.status, "BLOCKED")
            self.assertEqual(route.symbol, "")


if __name__ == "__main__":
    unittest.main()
