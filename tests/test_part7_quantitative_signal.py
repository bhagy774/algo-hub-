"""Offline Part 7 risk-gate regression tests; no exchange clients are imported."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import pandas as pd
    import part7_signal
except ImportError:  # the Part 7 validator is an explicit pandas dependency
    pd = None
    part7_signal = None


def make_bars(*, shock=False, n=100):
    rows = []
    price = 100.0
    for i in range(n):
        ret = (0.0002 * math.sin(i * .8) if i < n - 10 else 0.02 * (-1 if i % 2 else 1)) if shock else 0.0002 * math.sin(i * .8)
        open_price = price
        price = open_price * math.exp(ret)
        rows.append({"open": open_price, "high": max(open_price, price) * 1.0002,
                     "low": min(open_price, price) * .9998, "close": price, "volume": 100.0})
    return rows


class Part7QuantitativeSignalTests(unittest.TestCase):
    @unittest.skipIf(pd is None, "pandas is required by the original Part 7 frame validator")
    def test_live_jarvis_adapter_propagates_quantitative_extreme_volatility_veto(self):
        # Run the actual Jarvis Part7Volatility adapter method without importing the
        # side-effect-heavy controller module (which also depends on GPU libraries).
        source = (ROOT / "jarvis_FIXED.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Part7Volatility")
        namespace = {"_analyze_part7_timeframe": part7_signal.analyze_timeframe}
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(module, str(ROOT / "jarvis_FIXED.py"), "exec"), namespace)
        adapter = namespace["Part7Volatility"]()
        frame = pd.DataFrame(make_bars(shock=True))
        frame.attrs["symbol"] = "BTCUSDT"
        frame.index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(seconds=60),
                                    periods=len(frame), freq="min")
        result = adapter.analyze(frame, context={"selected_symbol": "BTCUSDT", "symbol": "BTCUSDT", "timeframe": "1m"})
        self.assertEqual(result["status"], "veto")
        self.assertEqual(result["signal"], 0)
        self.assertTrue(result["risk_veto"])
        self.assertTrue(result["entry_blocked"])
        self.assertEqual(result["symbol"], "BTCUSDT")

    @unittest.skipIf(pd is None, "pandas is required by the original Part 7 frame validator")
    def test_mtf_part7_aggregate_fails_closed_on_any_extreme_risk_frame(self):
        results = {}
        for tf in part7_signal.DEFAULT_TIMEFRAMES:
            data = pd.DataFrame(make_bars(shock=(tf == "1m")))
            results[tf] = part7_signal.analyze_timeframe(
                data, symbol="BTCUSDT", timeframe=tf,
                context={"is_backtest_mode": True, "selected_symbol": "BTCUSDT"})
        aggregate = part7_signal.aggregate_results(results, symbol="BTCUSDT")
        self.assertEqual(aggregate["status"], "veto")
        self.assertTrue(aggregate["entry_blocked"])
        self.assertTrue(aggregate["risk_veto"])
        self.assertEqual(aggregate["veto_timeframes"], ["1m"])
        self.assertEqual(set(aggregate["timeframe_results"]), set(part7_signal.DEFAULT_TIMEFRAMES))

    @unittest.skipIf(pd is None, "pandas is required by the original Part 7 frame validator")
    def test_part7_gate_keeps_missing_data_fail_closed(self):
        result = part7_signal.analyze_timeframe([], symbol="BTCUSDT", timeframe="1m", context={})
        self.assertTrue(result["entry_blocked"])
        self.assertFalse(result["risk_veto"])
        self.assertEqual(result["signal"], 0)
        self.assertEqual(result["data_status"], "invalid")


if __name__ == "__main__":
    unittest.main()
