import importlib
import io
import json
import os
import unittest
from unittest.mock import patch

from jarvis_decision import build_final_decision, confidence_text, normalize_confidence, DecisionContext
from jarvis_risk import calculate_trade_size, contract_quote_value_usdt
from jarvis_dashboard import render_dashboard


class EmptyScanner:
    def __init__(self):
        self._best = ""
    def run_now(self):
        return ""
    def get_delta_symbol(self):
        return ""
    def get_best_coin(self):
        return ""
    def get_all_scores(self):
        return {}


class DeltaSymbols:
    def get_available_symbols(self):
        return ["BTCUSDT"]


class AuditRegressionTests(unittest.TestCase):
    def test_confidence_formats_and_na_are_normalized(self):
        for raw in (85, "85%", "85/100", "85/100 (Autonomy)"):
            self.assertEqual(normalize_confidence(raw), 85)
            self.assertEqual(confidence_text(raw), "85")
        self.assertIsNone(normalize_confidence("N/A"))
        self.assertEqual(confidence_text("N/A"), "N/A")

    def test_conflict_and_non_primary_options_block_execution(self):
        conflict = build_final_decision(
            {"direction": "CALL", "confidence_score": "85%"},
            DecisionContext(symbol="ETHUSDT", opinions=["BUY", "SELL"])
        )
        self.assertFalse(conflict["execution_allowed"])
        self.assertEqual(conflict["direction"], "NO_TRADE")
        macro = build_final_decision(
            {"direction": "CALL", "confidence_score": 85},
            DecisionContext(
                symbol="ETHUSDT", require_options=True,
                options_context={"available": True, "role": "btc_macro_confirmation"}
            )
        )
        self.assertFalse(macro["execution_allowed"])

    def test_mtf_fetch_uses_selected_symbol(self):
        import time
        from direct_candle_cache import DirectCandleCache
        from native_candle_fixture import native_times
        class Delta:
            def __init__(self):
                self.symbols = []
            def get_historical_candles_with_metadata(self, symbol, resolution, limit):
                self.symbols.append(symbol)
                return {"source": "binance", "symbol": symbol, "candles": [
                    {"time": ts, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
                    for ts in native_times(resolution, time.time(), limit)]}
        try:
            import jarvis_FIXED
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional runtime dependency unavailable: {exc.name}")
        engine = object.__new__(jarvis_FIXED.JarvisElite)
        engine.is_backtest_mode = False
        engine.delta_client = Delta()
        engine.direct_candle_cache = DirectCandleCache(engine.delta_client)
        data = engine._fetch_mtf_from_api("ETHUSDT")
        self.assertTrue(data)
        self.assertEqual(set(engine.delta_client.symbols), {"ETHUSDT"})

    def test_empty_scanner_candidate_is_blocked(self):
        from jarvis_market_router import MarketRouter
        route = MarketRouter(scanner=EmptyScanner(), delta_client=DeltaSymbols()).select_crypto()
        self.assertEqual(route.status, "BLOCKED")
        self.assertIn("invalid symbol", route.reason)

    def test_documented_delta_base_contract_value_is_price_converted(self):
        product = {
            "id": 1, "symbol": "BTCUSD", "contract_value": "0.001",
            "contract_unit_currency": "BTC", "contract_type": "perpetual_futures",
            "default_leverage": 100, "max_leverage_notional": "1000000",
        }
        self.assertEqual(contract_quote_value_usdt(product, 65000), 65.0)
        self.assertIsNone(contract_quote_value_usdt({**product, "contract_type": "inverse"}, 65000))
        self.assertIsNone(contract_quote_value_usdt({"symbol": "BTCUSD", "contract_value": 0.001}, 65000))

    def test_non_one_contract_value_sizes_actual_notional(self):
        product = {"contract_value": 2, "contract_value_currency": "USDT"}
        self.assertEqual(contract_quote_value_usdt(product, 100), 2)
        result = calculate_trade_size(
            100, 90, 0.002, max_trade_risk_usdt=10,
            contract_value_usdt=2, require_contract_value=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["notional_usdt"], result["contracts"] * 2)
        self.assertIsNone(contract_quote_value_usdt({"contract_value": 1}, 100))

    def test_dashboard_does_not_append_percent_to_na(self):
        stream = io.StringIO()
        text = render_dashboard(
            {"signal": {"direction": "NO_TRADE", "confidence": "N/A"}},
            stream=stream,
        )
        self.assertIn("NO_TRADE  (N/A)", text)
        self.assertNotIn("N/A%)", text)

    def test_binance_proxy_redaction_and_mirror_retry(self):
        import binance_data

        class Response:
            def __init__(self, status):
                self.status_code = status
                self.text = "temporary"
            def json(self):
                return {"price": "100"}

        class Session:
            def __init__(self):
                self.proxies = {}
                self.headers = {}
                self.calls = []
            def get(self, url, **kwargs):
                self.calls.append(url)
                return Response(503 if len(self.calls) == 1 else 200)

        with patch.object(binance_data, "_PROXY", "http://user:secret@example.test:8080"):
            with patch.object(binance_data.requests, "Session", Session):
                with self.assertLogs(binance_data.logger, level="INFO") as captured:
                    client = binance_data.BinanceData()
                    client.get_live_price("BTCUSDT")
                self.assertEqual(len(client.session.calls), 2)
                joined = " ".join(captured.output)
                self.assertNotIn("secret", joined)

    def test_unavailable_selected_options_block_required_execution(self):
        decision = build_final_decision(
            {"direction": "BUY", "confidence_score": "90%"},
            DecisionContext(
                symbol="SOLUSDT", require_options=True,
                options_context={"available": False, "role": "unavailable"},
            )
        )
        self.assertFalse(decision["execution_allowed"])
        self.assertEqual(decision["direction"], "NO_TRADE")
        self.assertIn("options context", " ".join(decision["reasons"]))

    def test_non_spot_fallback_requires_explicit_compatibility(self):
        from delta_api_wrapper import DeltaExchangeData
        class FakeBinance:
            def get_last_source(self):
                return {"source": "bybit_linear_perpetual", "market_semantics": "linear_perpetual"}
        client = object.__new__(DeltaExchangeData)
        client._binance = FakeBinance()
        with patch.dict(os.environ, {"JARVIS_ALLOW_PERPETUAL_FALLBACK": "0"}):
            self.assertFalse(client._binance_spot_compatible())
        with patch.dict(os.environ, {"JARVIS_ALLOW_PERPETUAL_FALLBACK": "1"}):
            self.assertTrue(client._binance_spot_compatible())

    def test_duplicate_manager_cannot_submit_second_close(self):
        from jarvis_live_trader import JarvisAutoTrader
        from jarvis_position_manager import JarvisPositionManager, PositionRecord
        from jarvis_position_ownership import claim_position, clear_registry

        class PaperVenue:
            def __init__(self):
                self.calls = []
            def place_order(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {"success": True}

        clear_registry()
        venue = PaperVenue()
        auto = JarvisAutoTrader(venue)
        auto.is_enabled = False  # paper mode: close must not reach venue
        manager = JarvisPositionManager(venue)
        auto_pos = {
            "id": "shared-1", "direction": "CALL", "symbol": "BTCUSDT",
            "contracts": 2, "entry_price": 100.0,
        }
        self.assertTrue(claim_position("shared-1", auto._ownership_token))
        pm_pos = PositionRecord("shared-1", "CALL", 100.0, 2, 90, "BTCUSDT")
        manager.open_positions.append(pm_pos)  # simulate stale duplicate wiring
        self.assertFalse(manager._close_position(pm_pos, 101.0, "TP HIT"))
        self.assertEqual(venue.calls, [])
        self.assertTrue(auto._close_position_market(auto_pos)["success"])
        self.assertEqual(venue.calls, [])  # paper close never reaches venue
        clear_registry()


    def test_ambiguous_close_is_not_retried_or_reversed(self):
        from jarvis_live_trader import JarvisAutoTrader
        from jarvis_position_ownership import clear_registry

        class AmbiguousVenue:
            def __init__(self):
                self.calls = 0
            def place_order(self, *args, **kwargs):
                self.calls += 1
                raise TimeoutError("response lost after submission")

        clear_registry()
        venue = AmbiguousVenue()
        auto = JarvisAutoTrader(venue)
        auto.is_enabled = True
        pos = {"id": "ambiguous-1", "direction": "CALL", "symbol": "BTCUSDT", "contracts": 1}
        first = auto._close_position_market(pos)
        second = auto._close_position_market(pos)
        self.assertFalse(first["success"])
        self.assertTrue(first.get("ambiguous"))
        self.assertFalse(second["success"])
        self.assertEqual(venue.calls, 1)
        clear_registry()

    def test_watchdog_bad_threshold_falls_back(self):
        import jarvis_watchdog
        with patch.dict(os.environ, {"DOCTOR_STALE_THRESHOLD": "not-a-number"}):
            module = importlib.reload(jarvis_watchdog)
            self.assertEqual(module.HEARTBEAT_STALE_SECONDS, 600)
        with patch.dict(os.environ, {"DOCTOR_STALE_THRESHOLD": "0"}):
            module = importlib.reload(jarvis_watchdog)
            self.assertEqual(module.HEARTBEAT_STALE_SECONDS, 600)
        importlib.reload(jarvis_watchdog)


if __name__ == "__main__":
    unittest.main()
