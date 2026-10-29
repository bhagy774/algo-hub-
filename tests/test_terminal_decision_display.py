"""Offline tests for conservative JARVIS terminal decision formatting."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from terminal_decision_display import format_decision_audit


class TerminalDecisionDisplayTests(unittest.TestCase):
    def test_missing_values_are_unknown_not_plausible_defaults(self):
        output = format_decision_audit({}, None, now=1000)
        self.assertIn("Selected asset: UNKNOWN", output)
        self.assertIn("Candle snapshot: UNKNOWN", output)
        self.assertIn("Final decision: UNKNOWN", output)
        self.assertIn("Venue/account flatness: UNKNOWN", output)
        self.assertNotIn("BTCUSDT", output)
        self.assertNotIn("TP1=", output)
        self.assertNotIn("filled", output.lower())

    def test_candidate_vs_final_and_raw_part_alignment_are_separated(self):
        snapshot = {
            "fetched_at": 995.0,
            "frames": {"1m": {"source": "exchange-test", "current": {"close": 123.0}}},
        }
        output = format_decision_audit(
            {
                "direction": "NO_TRADE", "confidence": None,
                "confidence_display": "N/A", "status": "WAIT",
                "execution_allowed": False,
                "reasons": ["Confidence unavailable", "upstream context note"],
                "price": 123.0, "entry": 123.2, "tp1": 124.0,
            },
            "SOLUSDT",
            candidate={"direction": "CALL", "confidence_score": "81/100"},
            blockers=["1m confirmation pending"],
            parts={
                "part1_breakout": {"signal": 1, "thought": "MTF breakout confirmed"},
                "part2_zone": {"signal": -1, "thought": "supply overhead"},
                "part7_volatility": {"signal": 0, "status": "ready", "reason": "within band"},
            },
            market_context={"trend": "UPTREND", "volatility_status": "HIGH"},
            snapshot=snapshot,
            position_state={"live": [], "paper": []},
            execution_mode="PAPER",
            now=1000.0,
        )
        self.assertIn("Selected asset: SOLUSDT", output)
        self.assertIn("Initial signal: BUY; confidence=81/100", output)
        self.assertIn("Final decision: NO_TRADE", output)
        self.assertIn("Decision change: BUY → NO_TRADE", output)
        self.assertIn("Supports initial BUY signal (1)", output)
        self.assertIn("Part 1 (breakout) [BUY] — MTF breakout confirmed", output)
        self.assertIn("Opposes initial BUY signal (1)", output)
        self.assertIn("Part 2 (zone) [SELL] — supply overhead", output)
        self.assertIn("Neutral parts (1)", output)
        self.assertIn("1m confirmation pending", output)
        self.assertIn("Decision / analysis notes (reported; not automatically treated as blockers)", output)
        self.assertIn("Market context (reported): trend=UPTREND; volatility=HIGH", output)
        self.assertNotIn("regime=", output)  # No regime value was supplied.
        self.assertIn("snapshot age: about 5s", output)
        self.assertIn("forming candle is unconfirmed", output)
        self.assertIn("Venue/account flatness: UNKNOWN", output)

    def test_order_return_does_not_claim_fill_or_close(self):
        output = format_decision_audit(
            {"direction": "BUY", "confidence": 80, "confidence_display": "80", "status": "READY", "execution_allowed": True},
            "ETHUSDT",
            candidate={"direction": "BUY", "confidence_score": 80},
            stage="ORDER_SUBMISSION",
            order_outcome={"success": True, "paper": True, "position": {
                "id": "PAPER-1", "status": "OPEN", "paper": True,
                "contracts": 2, "leverage": 3, "margin_usdt": 4.5,
                "trade_risk_usdt": 0.2, "notional_usdt": 6.0,
            }},
            position_state={"live": [], "paper": [{"id": "PAPER-1", "status": "PENDING_LIMIT"}]},
        )
        self.assertIn("Executor return success field: true", output)
        self.assertIn("not sent to an exchange; this is a paper simulation", output)
        self.assertIn("not applicable to the paper record", output)
        self.assertIn("quantity=2 contract(s)", output)
        self.assertIn("leverage=3x", output)
        self.assertIn("risk=$0.2 USDT", output)
        self.assertIn("Fill confirmation: not supplied; not inferred", output)
        self.assertIn("Close status: not supplied / not observed", output)
        self.assertNotIn("filled successfully", output.lower())

    def test_live_success_is_not_rendered_as_a_confirmed_fill(self):
        output = format_decision_audit(
            {"direction": "SELL", "execution_allowed": True},
            "BTCUSD", stage="ORDER_SUBMISSION",
            order_outcome={"success": True, "position": {"id": "venue-id", "status": "OPEN", "paper": False}},
        )
        self.assertIn("Executor return success field: true", output)
        self.assertIn("Venue acknowledgement: UNKNOWN", output)
        self.assertIn("Fill confirmation: not supplied; not inferred", output)
        self.assertIn("not an independent fill/position confirmation", output)
        self.assertIn("UNKNOWN — no local position snapshot supplied", output)


if __name__ == "__main__":
    unittest.main()
