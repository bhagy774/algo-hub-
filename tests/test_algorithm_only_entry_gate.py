import os
import sys
import unittest
from unittest.mock import Mock

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithm_only_entry_gate import evaluate_algorithm_only_entry


class AlgorithmOnlyGateTests(unittest.TestCase):
    def setUp(self):
        self.good = dict(
            symbol="BTCUSDT", requested_side="BUY",
            signal={"symbol": "BTCUSDT", "side": "BUY"},
            confidence=65, data_fresh=True, data_valid=True, conflict_free=True,
        )

    def test_valid_evidence_still_blocks_until_validator_approved(self):
        result = evaluate_algorithm_only_entry(**self.good)
        self.assertTrue(result.allowed)
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(result.reasons, ())

    def test_buy_sell_and_wait_or_mismatched_identity_fail_closed(self):
        for side in ("BUY", "SELL", "WAIT"):
            args = dict(self.good, requested_side=side,
                        signal={"symbol": "BTCUSDT", "side": side})
            result = evaluate_algorithm_only_entry(**args)
            if side in ("BUY", "SELL"):
                self.assertTrue(result.allowed)
            else:
                self.assertFalse(result.allowed)
        args = dict(self.good, signal={"symbol": "ETHUSDT", "side": "BUY"})
        self.assertIn("symbol_mismatch", evaluate_algorithm_only_entry(**args).reasons)

    def test_bad_confidence_data_and_conflict_fail_closed(self):
        for value in (None, "not-number", -1, 101, float("inf")):
            result = evaluate_algorithm_only_entry(**dict(self.good, confidence=value))
            self.assertFalse(result.allowed)
        self.assertIn("data_not_fresh", evaluate_algorithm_only_entry(**dict(self.good, data_fresh=False)).reasons)
        self.assertIn("data_not_valid", evaluate_algorithm_only_entry(**dict(self.good, data_valid=False)).reasons)
        self.assertIn("signal_conflict", evaluate_algorithm_only_entry(**dict(self.good, conflict_free=False)).reasons)

    def test_no_model_callback_is_accepted_or_called(self):
        model = Mock(side_effect=AssertionError("model must not run"))
        result = evaluate_algorithm_only_entry(**self.good)
        self.assertTrue(result.allowed)
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
