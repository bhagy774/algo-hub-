import unittest
from jarvis_laya_advisor import advise

class LayaAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {"symbol": "BTCUSDT", "last": 1}
        from unittest.mock import patch
        self.enabled = patch.dict('os.environ', {'JARVIS_LAYA_ADVISORY': 'true'})
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
    def test_valid_advice_does_not_change_deterministic_decision(self):
        got = advise(self.snapshot, symbol="BTCUSDT", deterministic_decision="SELL",
                     predictor=lambda _: {"suggestion":"BUY", "confidence":.8})
        self.assertEqual((got.status, got.suggestion, got.deterministic_decision), ("ok", "BUY", "SELL"))
    def test_missing_dependency_is_visible(self):
        got = advise(self.snapshot, symbol="BTCUSDT", deterministic_decision="NO_TRADE",
                     predictor=lambda _: (_ for _ in ()).throw(ModuleNotFoundError()))
        self.assertEqual(got.status, "unavailable")
    def test_errors_and_invalid_values_are_not_fake_confirmation(self):
        for result, expected in [(lambda _: 1, "invalid"), (lambda _: {"suggestion":"WAIT"}, "invalid"),
                                 (lambda _: (_ for _ in ()).throw(RuntimeError()), "error")]:
            self.assertEqual(advise(self.snapshot, symbol="BTCUSDT", deterministic_decision="BUY",
                              predictor=result).status, expected)
    def test_symbol_mismatch_rejected(self):
        self.assertEqual(advise(self.snapshot, symbol="ETHUSDT", deterministic_decision="BUY").status, "invalid")
    def test_disabled_is_explicit(self):
        self.assertEqual(advise(self.snapshot, symbol="BTCUSDT", deterministic_decision="BUY", enabled=False).status, "disabled")

if __name__ == "__main__": unittest.main()
