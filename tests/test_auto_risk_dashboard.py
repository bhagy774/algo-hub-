from central_entry_fixture import entry_evidence
import os
import unittest
from io import StringIO
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in __import__('sys').path:
    __import__('sys').path.insert(0, ROOT)

from jarvis_dashboard import render_dashboard
from jarvis_risk import calculate_trade_size, derive_auto_leverage


class VenueStub:
    def __init__(self, metadata=None, leverage_ok=True):
        self.metadata = metadata
        self.leverage_ok = leverage_ok
        self.leverage_calls = []
        self.order_calls = []

    def get_wallet_balance(self):
        return 100.0

    def get_product_metadata(self, symbol):
        return self.metadata

    def set_leverage(self, symbol, leverage):
        self.leverage_calls.append((symbol, leverage))
        return self.leverage_ok

    def place_protected_order(self, **kwargs):
        assert self.set_leverage(kwargs["symbol"], kwargs["leverage"]) is False
        return {"status": "REJECTED", "authoritative": True, "reason": "Leverage rejected"}

    def place_order(self, *args, **kwargs):
        self.order_calls.append((args, kwargs))
        return {"success": True, "order_id": "unexpected"}


class AutoRiskDashboardTests(unittest.TestCase):
    def test_leverage_is_derived_from_margin_and_stop_risk(self):
        low_margin = derive_auto_leverage(100, 1, 2, 0.002)
        high_margin = derive_auto_leverage(100, 4, 2, 0.002)
        self.assertTrue(low_margin["ok"] and high_margin["ok"])
        self.assertGreaterEqual(low_margin["leverage"], high_margin["leverage"])
        self.assertLessEqual(low_margin["leverage"], 20)

    def test_missing_balance_and_unfunded_contract_fail_closed(self):
        self.assertFalse(calculate_trade_size(0, 90, 0.002)["ok"])
        self.assertFalse(calculate_trade_size(0.01, 90, 0.002)["ok"])
        self.assertFalse(derive_auto_leverage(10, 1, 1, 0)["ok"])

    def test_live_missing_product_metadata_blocks_before_leverage_or_order(self):
        import jarvis_live_trader
        from jarvis_live_trader import TradeRequest
        venue = VenueStub(metadata=None)
        trader = jarvis_live_trader.JarvisAutoTrader(venue)
        trader.is_enabled = True
        req = TradeRequest(
            direction="CALL",
            confidence=90,
            price=100.0,
            trade_type=entry_evidence("BTCUSDT", "CALL", 90, 100.0)[1],
            hedge_plan={"do_hedge": False},
            symbol="BTCUSDT",
            **entry_evidence("BTCUSDT", "CALL", 90, 100.0)[0]
        )
        result = trader._place_trade(req)
        self.assertFalse(result["success"])
        self.assertEqual(venue.leverage_calls, [])
        self.assertEqual(venue.order_calls, [])

    def test_leverage_rejection_blocks_order(self):
        import jarvis_live_trader
        from jarvis_live_trader import TradeRequest
        venue = VenueStub(metadata={"id": 1, "symbol": "BTCUSD", "contract_value": 1, "contract_value_currency": "USDT"}, leverage_ok=False)
        trader = jarvis_live_trader.JarvisAutoTrader(venue)
        trader.is_enabled = True
        req = TradeRequest(
            direction="CALL",
            confidence=90,
            price=100.0,
            trade_type=entry_evidence("BTCUSDT", "CALL", 90, 100.0)[1],
            hedge_plan={"do_hedge": False},
            symbol="BTCUSDT",
            **entry_evidence("BTCUSDT", "CALL", 90, 100.0)[0]
        )
        result = trader._place_trade(req)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "Leverage rejected")
        self.assertEqual(len(venue.leverage_calls), 1)
        self.assertEqual(venue.order_calls, [])

    def test_dashboard_contains_signal_plan_and_risk_in_one_render(self):
        stream = StringIO()
        text = render_dashboard({
            "symbol": "BTCUSDT", "price": "$100", "signal": {"direction": "CALL", "confidence": 90},
            "reasons": ["trend aligned", "risk accepted"],
            "plan": {"entry": "$100", "tp1": "$100.4", "sl": "$99.8", "action": "CALL"},
            "account": {"delta_available": 100, "margin": 4, "notional": 80, "leverage": 20},
        }, stream=stream)
        self.assertEqual(text, stream.getvalue().rstrip("\n"))
        for field in ("CALL", "trend aligned", "TP1", "DELTA AVAILABLE", "leverage 20x"):
            self.assertIn(field, text)


if __name__ == "__main__":
    unittest.main()
