"""Execute each current PR Part analyzer entry offline with controlled OHLCV."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENTRIES = {
    1: ("SmartBreakoutAI", "analyze"),
    2: ("AdvancedAnalysisSystem", "analyze_native_zone"),
    3: ("CandlePsychologyMasterGPU", "analyze"),
    4: ("VolumeProfileEngineGPU", "analyze"),
    5: ("MLEngineGPU", "analyze"),
    6: ("TrendEngineGPU", "analyze"),
    7: ("VolatilityEngineGPU", "analyze"),
    8: ("MarketStructureEngineGPU", "analyze"),
    9: ("OrderflowEngineGPU", "analyze"),
    10: ("CandleStatsEngineGPU", "analyze"),
}


def load_entry(part):
    cls_name, method_name = ENTRIES[part]
    path = ROOT / f"part{part}_FIXED.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    src_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in src_cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    original_decorators = copy.deepcopy(method.decorator_list)
    method = copy.deepcopy(method)
    method.decorator_list = []  # Exercise the deterministic body without model imports.
    cls = ast.ClassDef(name="EntryHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    if part == 1:
        from part1_FIXED import SmartBreakoutAI
        return SmartBreakoutAI, original_decorators
    return namespace["EntryHarness"], original_decorators


def bars(n=100, shock=False):
    result, price = [], 100.0
    for i in range(n):
        ret = (0.018 * (-1 if i % 2 else 1) if i >= n - 10 else 0.0002 * math.sin(i * .8)) if shock else 0.0005 + 0.00015 * math.sin(i * .4)
        op = price
        price = op * math.exp(ret)
        result.append({"open": op, "high": max(op, price) * 1.0002,
                       "low": min(op, price) * .9998, "close": price, "volume": 100 + i % 7})
    return result


class QuantitativePartEntryTests(unittest.TestCase):
    def test_all_original_part_analyzers_execute_bounded_quantitative_entry(self):
        sample = bars()
        for part in range(1, 11):
            with self.subTest(part=part):
                cls, decorators = load_entry(part)
                self.assertTrue(any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                                    and d.func.id == "part_advisory_entry" for d in decorators),
                                "native analyzer must retain its Part-owned advisory decorator")
                analyzer = cls() if part == 1 else object.__new__(cls)
                method = getattr(analyzer, ENTRIES[part][1])
                if part == 1:
                    result = method({"price_action": sample}, context={}, advisory_data=sample)
                elif part == 2:
                    result = method(sample, timeframe="1m", context={})
                else:
                    result = method(sample, context={})
                self.assertIsInstance(result, dict)
                self.assertIn(result.get("signal"), (-1, 0, 1))
                self.assertTrue(math.isfinite(float(result.get("confidence", 5.0))))
                json.dumps(result, allow_nan=False)

    def test_part7_entry_fails_closed_for_extreme_volatility(self):
        cls, _ = load_entry(7)
        result = getattr(object.__new__(cls), "analyze")(bars(shock=True), context={})
        self.assertEqual(result.get("signal"), 0)

    def test_invalid_ohlcv_is_neutral_without_fabricated_risk_veto(self):
        cls, _ = load_entry(6)
        malformed = bars()
        malformed[-1]["close"] = float("nan")
        result = getattr(object.__new__(cls), "analyze")(malformed, context={})
        self.assertEqual(result.get("signal"), 0)
        self.assertEqual(result.get("telemetry", {}).get("status"), "insufficient_or_invalid_ohlcv")


if __name__ == "__main__":
    unittest.main()
