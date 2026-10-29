import ast
import copy
import io
import logging
import math
import os
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from terminal_live_diagnostics import (
    INTERVAL_SECONDS,
    PART_KEYS,
    TIMEFRAMES,
    build_timeframe_inputs,
    classify_part_output,
    format_cycle_unavailable,
    format_live_diagnostics,
)


class Frame:
    def __init__(self, *index):
        self.index = index

    def __len__(self):
        return len(self.index)


class TerminalDiagnosticsTests(unittest.TestCase):
    def test_distinguishes_neutral_missing_error_and_confidence_zero(self):
        self.assertEqual(classify_part_output({"signal": 0, "confidence": 0}),
                         {"status": "NEUTRAL", "signal": "NEUTRAL", "confidence": 0, "reason": ""})
        self.assertEqual(classify_part_output({"direction": "NEUTRAL"})["status"], "NEUTRAL")
        self.assertEqual(classify_part_output({"confidence": 0})["status"], "MISSING")
        missing_conf = classify_part_output({"signal": "BUY"})
        self.assertEqual(missing_conf["status"], "OK")
        self.assertIsNone(missing_conf["confidence"])
        self.assertEqual(classify_part_output(error_type="ValueError")["status"], "ERROR")
        self.assertEqual(classify_part_output({"status": "unavailable", "signal": "BUY"})["status"], "UNAVAILABLE")
        self.assertEqual(classify_part_output({"status": "insufficient candles", "signal": 0})["status"], "INSUFFICIENT CANDLES")
        self.assertEqual(classify_part_output({"signal": "BUY", "risk_veto": True})["status"], "BLOCKED")
        self.assertEqual(classify_part_output({"signal": "SELL", "data_status": "timeframe mismatch"})["status"], "TIMEFRAME MISMATCH")

    def test_schema_aliases_and_sequence_signal(self):
        self.assertEqual(classify_part_output({"action": "LONG", "confidence_score": 0})["signal"], "BUY")
        self.assertEqual(classify_part_output({"signal": [0, -1]})["signal"], "SELL")
        self.assertEqual(classify_part_output({"signal": (0, 1)})["signal"], "BUY")
        self.assertEqual(classify_part_output({"signal": "NEUTRAL"})["signal"], "NEUTRAL")

    def test_timeframe_identity_age_and_closed_forming_separation(self):
        from native_candle_fixture import native_times
        from binance_timeframes import next_candle_open
        now = 1_800_000_000.0
        frames = {tf: Frame(native_times(tf, now, 2)[0]) for tf in TIMEFRAMES}
        meta = {tf: SimpleNamespace(symbol="BTCUSDT", timeframe=tf, source="mock-source",
                                    fetched_at=now, current={"close": 101},
                                    current_is_confirmed=False) for tf in TIMEFRAMES}
        snapshot = SimpleNamespace(symbol="BTCUSDT", venue="fixture", frames=meta, fetched_at=now - 5)
        result = build_timeframe_inputs(frames, snapshot, expected_symbol="BTC/USDT", now=now)
        self.assertEqual(tuple(result), TIMEFRAMES)
        for tf in TIMEFRAMES:
            self.assertEqual(result[tf]["status"], "PRESENT")
            self.assertEqual(result[tf]["identity"], "VERIFIED")
            self.assertEqual(result[tf]["current_candle"], "FORMING / UNCONFIRMED")
            self.assertEqual(result[tf]["snapshot_age_seconds"], 0.0)
            self.assertEqual(result[tf]["closed_age_seconds"], now - next_candle_open(tf, frames[tf].index[-1]))
        bad = SimpleNamespace(symbol="ETHUSDT", timeframe="1m", source="x", fetched_at=now,
                              current=None, current_is_confirmed=False)
        mismatch = build_timeframe_inputs({"1m": Frame(now - 120)},
                                         SimpleNamespace(symbol="ETHUSDT", frames={"1m": bad}, fetched_at=now),
                                         timeframes=("1m", "3m"), expected_symbol="BTCUSDT", now=now)
        self.assertEqual(mismatch["1m"]["status"], "SYMBOL MISMATCH")
        self.assertEqual(mismatch["3m"]["status"], "MISSING")
        self.assertEqual(mismatch["3m"]["last_closed"], None)

    def test_matrix_reports_all_native_timeframes_and_statuses(self):
        matrix = {tf: {name: {"status": "NEUTRAL", "signal": "NEUTRAL", "confidence": None}
                       for name in PART_KEYS} for tf in TIMEFRAMES}
        matrix["1m"][PART_KEYS[0]] = {"status": "OK", "signal": "BUY", "confidence": 0,
                                      "reason": "API_KEY=do-not-print momentum"}
        matrix["3m"][PART_KEYS[1]] = {"status": "MISSING", "signal": None, "confidence": None,
                                      "reason": "signal field absent"}
        matrix["5m"][PART_KEYS[2]] = {"status": "ERROR", "signal": None, "confidence": None,
                                      "reason": "ValueError"}
        matrix["15m"][PART_KEYS[3]] = {"status": "BLOCKED", "signal": "NEUTRAL", "confidence": 0,
                                       "reason": "risk veto"}
        matrix["30m"][PART_KEYS[4]] = {"status": "STALE", "signal": "BUY", "confidence": 0,
                                       "reason": "stale analyzer return"}
        before = copy.deepcopy(matrix)
        rendered = format_live_diagnostics(
            symbol="BTCUSDT", timeframe_inputs={}, parts_by_timeframe=matrix,
            fused_parts={"part11_fusion": {"signal": "NEUTRAL", "confidence": 0},
                         "part12_confidence": {"confidence": 0, "status": "heuristic"}},
            decision={"direction": "NO_TRADE", "execution_allowed": False, "status": "blocked",
                      "reasons": ["1m confirmation pending"]},
            candidate={"trade_signal": {"entry_price": 100, "take_profit_1": 110,
                                         "take_profit_2": 120, "stop_loss": 90}},
            setup_mode={"mode": "NO TRADE", "reason": "1m confirmation pending"},
            part7_gate={"status": "blocked", "entry_blocked": True, "risk_veto": True,
                        "reason": "risk veto"},
            entry_trigger={"confirmed": False, "note": "1m confirm wait"},
            runtime={"backend": "cpu", "device": "cpu"},
            neural_status={"status": "not_run", "availability": "not probed", "reason": "no model called"})
        self.assertEqual(matrix, before)
        self.assertIn("1m", rendered)
        self.assertIn("4h", rendered)
        self.assertIn("c=0", rendered)
        self.assertIn("c=?", rendered)
        self.assertIn("MISSING", rendered)
        self.assertIn("ERROR", rendered)
        self.assertIn("BLOCKED/NEUTRAL", rendered)
        self.assertIn("BUY/STALE c=0", rendered)
        self.assertIn("1m entry trigger: waiting / not confirmed", rendered)
        self.assertIn("Part 12 confidence", rendered)
        self.assertIn("confidence(raw)=0", rendered)
        self.assertNotIn("Protective", rendered)
        self.assertNotIn("do-not-print", rendered)
        self.assertLess(len(rendered), 10000)

    def test_candidate_plan_and_unknown_execution_are_not_conflated(self):
        rendered = format_live_diagnostics(
            symbol="BTCUSDT", decision={"direction": "BUY", "execution_allowed": True},
            candidate={"trade_signal": {"entry_price": 10, "take_profit_1": 11,
                                         "take_profit_2": 12, "stop_loss": 9}},
            setup_mode={"mode": "SCALP", "reason": "default expiry route"})
        self.assertIn("Entry=10", rendered)
        self.assertIn("TP1=11", rendered)
        self.assertIn("TP2=12", rendered)
        self.assertIn("SL=9", rendered)
        self.assertIn("not proof of submission/fill", rendered)
        self.assertIn("setup mode", rendered.lower())
        self.assertIn("Final decision: BUY; execution_allowed=ALLOWED", rendered)
        self.assertIn("Risk/Part 7 gate: UNKNOWN", rendered)
        self.assertNotIn("Fill confirmation", rendered)

    def test_synthetic_unavailable_cycle_is_explicit_and_safe(self):
        text = format_cycle_unavailable(
            symbol="BTCUSDT", stage="CANDLE_FETCH",
            reason="account_id=123 password=hunter2 order_id=secret-order" + chr(27) + "[2J request failed",
        )
        self.assertIn("UNAVAILABLE", text)
        self.assertIn("account_id=[redacted]", text)
        self.assertNotIn("hunter2", text)
        self.assertNotIn("secret-order", text)
        self.assertNotIn(chr(27), text)
        self.assertEqual(text.count("=UNAVAILABLE"), len(TIMEFRAMES))

    def test_stale_metadata_is_explicit_but_does_not_rewrite_raw_signal(self):
        now = 500_000.0
        tf = "1m"
        stale_frame = Frame(now - INTERVAL_SECONDS[tf] - 121)
        source = SimpleNamespace(symbol="BTCUSDT", timeframe=tf, source="mock",
                                 fetched_at=now - 46, current=None,
                                 current_is_confirmed=False)
        result = build_timeframe_inputs(
            {tf: stale_frame},
            SimpleNamespace(symbol="BTCUSDT", frames={tf: source}, fetched_at=now - 46),
            timeframes=(tf,), expected_symbol="BTCUSDT", now=now,
            stale_intervals=2.0, fetch_ttl_seconds=45.0,
        )
        self.assertEqual(result[tf]["status"], "STALE")
        self.assertIn("closed age", result[tf]["freshness_reason"])
        self.assertIn("fetch age", result[tf]["freshness_reason"])
        rendered = format_live_diagnostics(
            symbol="BTCUSDT", timeframe_inputs=result,
            parts_by_timeframe={tf: {PART_KEYS[0]: {"status": "OK", "signal": "BUY", "confidence": 0}}},
            timeframes=(tf,), cycle_status={"status": "COMPLETE"},
        )
        self.assertIn("BUY/STALE c=0", rendered)
        self.assertIn("not a strategy gate", rendered)

    def test_cycle_blocked_before_analyzers_marks_parts_not_run(self):
        rendered = format_live_diagnostics(
            symbol="BTCUSDT", timeframe_inputs={}, parts_by_timeframe={},
            cycle_status={"status": "BLOCKED", "reason": "symbol mismatch"},
        )
        self.assertIn("Analysis cycle: BLOCKED; reason=symbol mismatch", rendered)
        self.assertIn("P01 | NOT RUN c=?", rendered)
        self.assertNotIn("P01 | MISSING", rendered)
        self.assertIn("analyzers did not run", rendered)


class LiveUnavailableWiringTests(unittest.TestCase):
    @staticmethod
    def _load_production_method():
        here = Path(__file__).resolve().parent
        candidates = (here / "current_jarvis.py", here / "jarvis_FIXED.py",
                     here.parent / "jarvis_FIXED.py")
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            raise AssertionError(f"Could not locate production Jarvis source in {candidates}")
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        engine = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "LiveTradingEngine")
        method = next(node for node in engine.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_print_unavailable_live_diagnostics")
        harness = ast.ClassDef(name="DiagnosticsHarness", bases=[], keywords=[],
                               body=[method], decorator_list=[])
        helper = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_binance_spot_symbol")
        module = ast.fix_missing_locations(ast.Module(body=[helper, harness], type_ignores=[]))
        namespace = {"math": math, "os": os, "time": time,
                     "logger": logging.getLogger("diagnostics-test")}
        exec(compile(module, str(source_path), "exec"), namespace)
        return namespace["DiagnosticsHarness"]()

    def test_unavailable_and_partial_snapshot_are_throttled_without_orders(self):
        harness = self._load_production_method()
        harness.jarvis = SimpleNamespace(
            direct_candle_cache=SimpleNamespace(stale_intervals=2.0, ttl_seconds=45.0),
            gpu_status={"backend": "cpu", "device": "cpu"},
        )
        with patch.dict(os.environ, {"JARVIS_DIAGNOSTICS_UNAVAILABLE_INTERVAL_SECONDS": "bad"}):
            capture = io.StringIO()
            with redirect_stdout(capture):
                first = harness._print_unavailable_live_diagnostics(
                    "BTCUSDT", "CANDLE_FETCH", "account_id=secret-route failure", snapshot=None)
                throttled = harness._print_unavailable_live_diagnostics(
                    "BTCUSDT", "CANDLE_FETCH", "account_id=secret-route failure", snapshot=None)
            self.assertTrue(first)
            self.assertFalse(throttled)
            unavailable = capture.getvalue()
            self.assertEqual(unavailable.count(" UNAVAILABLE identity"), len(TIMEFRAMES))
            self.assertIn("Analysis cycle: UNAVAILABLE", unavailable)
            self.assertIn("P01 | NOT RUN c=?", unavailable)
            self.assertNotIn("secret-route", unavailable)
            self.assertEqual(unavailable.count("JARVIS | LIVE ANALYSIS DIAGNOSTICS"), 1)

        class MiniFrame:
            def __init__(self, index):
                self.index = (index,)
                self.attrs = {}

            def __len__(self):
                return len(self.index)

        now = time.time()
        partial = {"symbol": "BTCUSDT", "venue": "fixture", "fetched_at": now,
                   "frames": {"1m": {"closed": MiniFrame(now - 62), "source": "fixture",
                                       "symbol": "BTCUSDT", "timeframe": "1m",
                                       "fetched_at": now, "current": {"close": 100},
                                       "current_is_confirmed": False}}}
        capture = io.StringIO()
        with redirect_stdout(capture):
            printed = harness._print_unavailable_live_diagnostics(
                "BTCUSDT", "CANDLE_SNAPSHOT", "7 native frames unavailable", snapshot=partial)
        self.assertTrue(printed)
        partial_text = capture.getvalue()
        self.assertIn("1m PRESENT identity=VERIFIED", partial_text)
        self.assertIn("3m MISSING", partial_text)
        self.assertIn("current=FORMING / UNCONFIRMED", partial_text)


if __name__ == "__main__":
    unittest.main()
