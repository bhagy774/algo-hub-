from central_entry_fixture import entry_evidence
import os
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DeltaStub:
    def __init__(self, balance=100.0, order_result=None):
        self.balance = balance
        self.order_result = order_result if order_result is not None else {"success": True, "order_id": "x"}
        self.order_calls = []
        self.leverage_calls = []

    def get_wallet_balance(self):
        return self.balance

    def place_order(self, *args, **kwargs):
        self.order_calls.append((args, kwargs))
        return self.order_result

    def set_leverage(self, *args, **kwargs):
        self.leverage_calls.append((args, kwargs))
        return True


class OracleStub:
    def __init__(self, forecast):
        self.forecast = forecast

    def get_latest_forecast(self):
        return self.forecast


class ExecutionSafetyTests(unittest.TestCase):
    def test_private_request_without_credentials_sends_nothing(self):
        from delta_api_wrapper import DeltaExchangeData
        client = DeltaExchangeData(api_key=None, api_secret=None)
        client._binance = None
        client.session.request = Mock()
        result = client._request("GET", "/v2/wallet/balances", authorized=True)
        self.assertFalse(result["success"])
        client.session.request.assert_not_called()

    def test_order_and_leverage_are_blocked_without_opt_in(self):
        from delta_api_wrapper import DeltaExchangeData
        with patch.dict(os.environ, {"DELTA_ORDER_EXECUTION_ENABLED": "false"}):
            client = DeltaExchangeData(api_key="key", api_secret="secret")
            client.get_product_id = Mock(side_effect=AssertionError("must not query venue"))
            self.assertFalse(client.place_order("BTCUSDT", "buy", 1)["success"])
            self.assertFalse(client.set_leverage("BTCUSDT", 100))

    def test_invalid_order_input_fails_closed(self):
        from delta_api_wrapper import DeltaExchangeData
        with patch.dict(os.environ, {"DELTA_ORDER_EXECUTION_ENABLED": "true"}):
            client = DeltaExchangeData(api_key="key", api_secret="secret")
            client.get_product_id = Mock(side_effect=AssertionError("must not query venue"))
            self.assertFalse(client.place_order("", "buy", 1)["success"])
            self.assertFalse(client.place_order("BTCUSDT", "hold", 1)["success"])
            self.assertFalse(client.place_order("BTCUSDT", "buy", 0)["success"])
            self.assertFalse(client.place_order("BTCUSDT", "buy", 1, "limit", 0)["success"])

    def test_deribit_constructor_does_not_authenticate(self):
        with patch("requests.get") as get:
            from deribit_options_client import DeribitOptionsClient
            client = DeribitOptionsClient(client_id="id", client_secret="secret")
        self.assertIsNone(client.access_token)
        get.assert_not_called()

    def test_sizer_never_invents_balance_or_exceeds_cap(self):
        from jarvis_sizer import JarvisSizer, MAX_RISK_PCT
        sizer = JarvisSizer(DeltaStub(0))
        no_money = sizer.calculate_size(95, force_balance=0)
        self.assertEqual(no_money["contracts"], 0)
        small = sizer.calculate_size(95, force_balance=0.01)
        self.assertLessEqual(small["margin_usdt"], 0.01 * MAX_RISK_PCT + 1e-9)
        self.assertEqual(small["contracts"], 0)

    def test_position_manager_does_not_book_failed_close(self):
        from jarvis_position_manager import JarvisPositionManager, PositionRequest
        delta = DeltaStub(order_result={"success": False, "error": "rejected"})
        # Run in paper mode so register_position doesn't require live contract metadata
        with patch.dict(os.environ, {"DELTA_ORDER_EXECUTION_ENABLED": "false"}):
            manager = JarvisPositionManager(delta)
            req = PositionRequest(position_id="p1", direction="CALL", entry_price=100, contracts=10, confidence=80)
            pos = manager.register_position(req)
            self.assertFalse(manager._close_position(pos, 101, "test"))
            self.assertEqual(pos.status, "CLOSE_UNKNOWN")
            self.assertEqual(manager.daily_pnl, 0.0)
            self.assertIn(pos, manager.open_positions)

    def test_live_paper_mode_never_sets_leverage_or_submits_order(self):
        import jarvis_live_trader
        from jarvis_live_trader import TradeRequest
        trader = jarvis_live_trader.JarvisAutoTrader(DeltaStub(100))
        trader.is_enabled = False
        req = TradeRequest(
            direction="CALL",
            confidence=90,
            price=100.0,
            trade_type=entry_evidence("BTCUSDT", "CALL", 90, 100.0)[1],
            hedge_plan={"do_hedge": False},
            **entry_evidence("BTCUSDT", "CALL", 90, 100.0)[0]
        )
        response = trader._place_trade(req)
        self.assertTrue(response["success"])
        self.assertEqual(trader.delta.leverage_calls, [])
        self.assertEqual(trader.delta.order_calls, [])

    def test_live_monitor_keeps_ambiguous_close_unbooked(self):
        import jarvis_live_trader
        delta = DeltaStub(order_result={"success": False, "error": "timeout"})
        trader = jarvis_live_trader.JarvisAutoTrader(delta)
        trader.is_enabled = True
        position = {
            "id": "p", "direction": "CALL", "entry_price": 100.0,
            "tp_price": 101.0, "sl_price": 99.0, "expiry_time": "2999-01-01T00:00:00",
            "contracts": 10, "symbol": "BTCUSDT", "status": "OPEN",
        }
        trader.open_positions = [position]
        trader._check_positions(101.0)
        self.assertEqual(position["status"], "CLOSE_UNKNOWN")
        self.assertEqual(trader.daily_pnl, 0.0)
        self.assertEqual(trader.open_positions, [position])

    def test_oracle_wait_and_missing_data_fail_closed(self):
        from oracle_trade_gate import OracleTradeGate
        gate = OracleTradeGate(oracle_ref=OracleStub({}), enabled=True, hard_gate=True)
        self.assertFalse(gate.is_trade_aligned("CALL")[0])
        forecast = {
            "model_used": "ready", "trade_suggestion": "WAIT",
            "5min": {"direction": "BULLISH"}, "30min": {"direction": "BULLISH"},
            "entry_zone": {"price_from": 99, "price_to": 101},
        }
        gate = OracleTradeGate(oracle_ref=OracleStub(forecast), enabled=True, hard_gate=True)
        self.assertFalse(gate.is_trade_aligned("CALL")[0])
        self.assertFalse(gate.is_price_in_entry_zone(0)[0])

    def test_oracle_requires_two_timeframe_consensus(self):
        from oracle_trade_gate import OracleTradeGate
        forecast = {
            "model_used": "ready", "trade_suggestion": "CALL",
            "5min": {"direction": "BULLISH"}, "30min": {"direction": "NEUTRAL"},
            "entry_zone": {"price_from": 99, "price_to": 101},
        }
        gate = OracleTradeGate(oracle_ref=OracleStub(forecast), enabled=True, hard_gate=True)
        self.assertFalse(gate.is_trade_aligned("CALL")[0])


if __name__ == "__main__":
    unittest.main()
