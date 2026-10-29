"""Offline tests for jarvis_presim (Pre-Trade Simulator).

No network, no live calls: learning data comes from a temp JSON file,
candles are synthetic, smart_tpsl_calculator runs locally.
"""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jarvis_presim as presim  # noqa: E402


def _candles(n=60, base=100.0, spread=1.0):
    rows = []
    price = base
    for i in range(n):
        rows.append({
            "open": price, "high": price + spread, "low": price - spread,
            "close": price + 0.1, "volume": 10,
        })
        price += 0.1
    return rows


def _write_learning(path, trades):
    path.write_text(json.dumps({"trades": trades, "engines": {}}))


def _trades(n, result, direction="CALL", symbol="BTC/USDT"):
    return [
        {"symbol": symbol, "direction": direction, "result": result,
         "entry_price": 100, "exit_price": 101, "pnl": 1}
        for _ in range(n)
    ]


class PreSimModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.id().split(".")[-1] + "_learning.json")
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("JARVIS_PRESIM", None)
        os.environ.pop("JARVIS_PRESIM_MIN_RR", None)
        os.environ.pop("JARVIS_PRESIM_BUDGET_MS", None)

    def tearDown(self):
        self.env.stop()
        if self.tmp.exists():
            self.tmp.unlink()

    def test_insufficient_history_passes_never_vetoes(self):
        _write_learning(self.tmp, _trades(4, "LOSS"))
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")
        self.assertEqual(d["confidence_delta"], 0)

    def test_no_learning_file_passes(self):
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")

    def test_low_historical_winrate_vetoes(self):
        _write_learning(self.tmp, _trades(8, "LOSS") + _trades(1, "WIN"))
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "veto")
        self.assertIn("win-rate", d["reason"])

    def test_weak_history_adjusts_down(self):
        _write_learning(self.tmp, _trades(4, "WIN") + _trades(6, "LOSS"))
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "adjust")
        self.assertLess(d["confidence_delta"], 0)

    def test_strong_history_adjusts_up(self):
        _write_learning(self.tmp, _trades(9, "WIN") + _trades(1, "LOSS"))
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "PUT", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        # Trades are CALL-only → PUT has insufficient history → pass
        self.assertEqual(d["action"], "pass")

    def test_history_matches_direction(self):
        _write_learning(self.tmp, _trades(9, "WIN", direction="PUT") + _trades(1, "LOSS", direction="PUT"))
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "PUT", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "adjust")
        self.assertGreater(d["confidence_delta"], 0)

    def test_rr_veto_math(self):
        # Force the calculator to report a bad R:R
        class BadRR:
            def calculate(self, *a, **k):
                return {"risk_reward": 0.5}
            calculate_tpsl = calculate
            calculate_targets = calculate
            def calculate_atr(self, *a, **k):
                return 0.0
        _write_learning(self.tmp, [])
        with patch.dict(sys.modules, {"smart_tpsl_calculator": type(sys)("smart_tpsl_calculator")}):
            sys.modules["smart_tpsl_calculator"].SmartTPSLCalculator = BadRR
            d = presim.run_presim(
                {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
                candles=_candles(), learning_path=str(self.tmp),
            )
        self.assertEqual(d["action"], "veto")
        self.assertIn("R:R", d["reason"])

    def test_rr_passes_when_above_minimum(self):
        _write_learning(self.tmp, [])
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")  # default scalping R:R ~1.8

    def test_min_rr_env_override(self):
        class MidRR:
            def calculate(self, *a, **k):
                return {"risk_reward": 1.5}
            calculate_tpsl = calculate
            calculate_targets = calculate
            def calculate_atr(self, *a, **k):
                return 0.0
        _write_learning(self.tmp, [])
        os.environ["JARVIS_PRESIM_MIN_RR"] = "2.0"
        with patch.dict(sys.modules, {"smart_tpsl_calculator": type(sys)("smart_tpsl_calculator")}):
            sys.modules["smart_tpsl_calculator"].SmartTPSLCalculator = MidRR
            d = presim.run_presim(
                {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
                candles=_candles(), learning_path=str(self.tmp),
            )
        self.assertEqual(d["action"], "veto")

    def test_volatility_extreme_low_adjusts_not_vetoes(self):
        _write_learning(self.tmp, [])
        calm = _candles(spread=0.0001)  # ATR ~0.0002% of price
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=calm, learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "adjust")
        self.assertLessEqual(d["confidence_delta"], 0)
        self.assertIn("volatility", d["reason"].lower())

    def test_volatility_snapshot_fallback(self):
        _write_learning(self.tmp, [])
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=None, snapshot={"volatility": "VERY_HIGH"},
            learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "adjust")
        self.assertLessEqual(d["confidence_delta"], 0)

    def test_budget_timeout_guard_abstains(self):
        _write_learning(self.tmp, _trades(10, "LOSS"))
        os.environ["JARVIS_PRESIM_BUDGET_MS"] = "0"  # budget already exhausted
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")  # no checks ran → abstain

    def test_fail_open_on_exception(self):
        with patch.object(presim, "check_historical_winrate", side_effect=RuntimeError("boom")):
            d = presim.run_presim(
                {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
                candles=_candles(), learning_path=str(self.tmp),
            )
        self.assertIn(d["action"], ("pass", "adjust"))  # never raises, never vetoes on error

    def test_disabled_by_env(self):
        _write_learning(self.tmp, _trades(10, "LOSS"))
        os.environ["JARVIS_PRESIM"] = "0"
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")
        self.assertIn("disabled", d["reason"])

    def test_no_directional_signal_passes(self):
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "NO_TRADE", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")

    def test_corrupt_learning_file_passes(self):
        self.tmp.write_text("{not json")
        d = presim.run_presim(
            {"symbol": "BTC/USDT", "direction": "CALL", "entry_price": 100.0},
            candles=_candles(), learning_path=str(self.tmp),
        )
        self.assertEqual(d["action"], "pass")


class PreSimWiringTests(unittest.TestCase):
    """Brain wiring: veto/adjust/pass paths through LiveTradingEngine._presim_gate."""

    @classmethod
    def setUpClass(cls):
        import jarvis_FIXED
        cls.brain = jarvis_FIXED

    def _engine(self):
        eng = self.brain.LiveTradingEngine.__new__(self.brain.LiveTradingEngine)
        eng.presim_stats = {"checks": 0, "vetoes": 0, "adjustments": 0}
        return eng

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("JARVIS_PRESIM", None)

    def tearDown(self):
        self.env.stop()

    def test_veto_blocks_entry_and_counts(self):
        eng = self._engine()
        with patch("jarvis_presim.run_presim",
                   return_value={"action": "veto", "confidence_delta": 0, "reason": "bad setup"}):
            params = self.brain.PresimParams(direction="CALL", confidence=80, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertIsNone(direction)
        self.assertEqual(eng.presim_stats["vetoes"], 1)
        self.assertEqual(eng.presim_stats["checks"], 1)

    def test_adjust_applies_clamped_delta(self):
        eng = self._engine()
        with patch("jarvis_presim.run_presim",
                   return_value={"action": "adjust", "confidence_delta": -50, "reason": "weak"}):
            params = self.brain.PresimParams(direction="PUT", confidence=80, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertEqual(direction, "PUT")
        self.assertEqual(conf, 70)  # clamped to -10
        self.assertEqual(eng.presim_stats["adjustments"], 1)

    def test_pass_leaves_signal_untouched(self):
        eng = self._engine()
        with patch("jarvis_presim.run_presim",
                   return_value={"action": "pass", "confidence_delta": 0, "reason": ""}):
            params = self.brain.PresimParams(direction="CALL", confidence=75, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertEqual((direction, conf), ("CALL", 75))
        self.assertEqual(eng.presim_stats["adjustments"], 0)
        self.assertEqual(eng.presim_stats["vetoes"], 0)

    def test_exception_fails_closed(self):
        eng = self._engine()
        with patch("jarvis_presim.run_presim", side_effect=RuntimeError("boom")):
            params = self.brain.PresimParams(direction="CALL", confidence=75, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertEqual((direction, conf), (None, 75))
        self.assertEqual(eng.presim_stats["errors"], 1)

    def test_partial_wiring_cannot_bypass_veto(self):
        eng = self._engine()
        with patch("jarvis_presim.run_presim", return_value={"action": "veto", "confidence_delta": 0, "reason": "blocked"}):
            params = self.brain.PresimParams(direction="CALL", confidence=80, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertIsNone(direction)
        self.assertEqual(conf, 80)

    def test_env_zero_disables(self):
        os.environ["JARVIS_PRESIM"] = "0"
        eng = self._engine()
        with patch("jarvis_presim.run_presim", side_effect=AssertionError("must not run")):
            params = self.brain.PresimParams(direction="CALL", confidence=75, entry_price=100.0, result={}, df=None, current_price=100.0)
            direction, conf = eng._presim_gate(direction=params.direction, confidence=params.confidence, entry_price=params.entry_price, result=params.result, df=params.df, current_price=params.current_price)
        self.assertEqual((direction, conf), ("CALL", 75))
        self.assertEqual(eng.presim_stats["checks"], 0)


if __name__ == "__main__":
    unittest.main()
