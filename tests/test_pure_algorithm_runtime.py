"""Offline tests of the source's live non-pure compatibility gate.

The isolated guard is production AST, not a full live-loop integration test.
No JarvisElite construction, venue, download, or exchange access occurs.
"""
import ast
import copy
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = (Path(__file__).parent.parent / "jarvis_FIXED.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
mode = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_pure_algorithm_mode")
jarvis = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "JarvisElite")
guard = next(n for n in ast.walk(jarvis) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "not self.is_backtest_mode and (not _pure_algorithm_mode())"
             and n.body and isinstance(n.body[0], ast.Return)
             and "Legacy model-required entry unsupported" in ast.unparse(n.body[0]))
args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], vararg=None,
                     kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
function = ast.FunctionDef(name="evaluate_guard", args=args,
                           body=[copy.deepcopy(guard), ast.Return(value=ast.Constant("allowed"))],
                           decorator_list=[], returns=None, type_comment=None)
ns = {"os": os}
exec(compile(ast.fix_missing_locations(ast.Module(body=[mode, function], type_ignores=[])),
             "jarvis_FIXED.py", "exec"), ns)


class PureModePolicyTests(unittest.TestCase):
    def test_true_spellings_enable_algorithm_mode(self):
        for value in ("1", "true", "yes", "on", " TRUE "):
            with self.subTest(value=value), patch.dict(os.environ, {"JARVIS_PURE_ALGO": value}):
                self.assertTrue(ns["_pure_algorithm_mode"]())

    def test_explicit_false_disables_pure_mode(self):
        for value in ("0", "false", "no", "off", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"JARVIS_PURE_ALGO": value}):
                self.assertFalse(ns["_pure_algorithm_mode"]())

    def test_unsupported_live_nonpure_fails_closed_without_model_call(self):
        no_trade = lambda reason: {"direction": "NO_TRADE", "reason": reason}
        for backtest, pure, blocked in ((False, "false", True),
                                        (False, "true", False), (True, "false", False)):
            with self.subTest(backtest=backtest, pure=pure), patch.dict(os.environ, {"JARVIS_PURE_ALGO": pure}):
                fake = SimpleNamespace(is_backtest_mode=backtest, _get_no_trade_signal=no_trade)
                result = ns["evaluate_guard"](fake)
                if blocked:
                    self.assertEqual(result["direction"], "NO_TRADE")
                    self.assertIn("unsupported", result["reason"])
                else:
                    self.assertEqual(result, "allowed")

    def test_runtime_has_no_model_vote_or_low_score_model_override(self):
        self.assertNotIn("self._get_deepseek_validation(data, score", SOURCE)
        self.assertNotIn("_opinions.append(_verdict)", SOURCE)
        self.assertNotIn("elif ai_signal in ['CALL', 'PUT', 'BUY', 'SELL']:", SOURCE)
        self.assertIn("self._presim_gate(", SOURCE)
        self.assertIn("self._scenario_gate(", SOURCE)
        self.assertIn("self.auto_trader.execute(", SOURCE)


if __name__ == "__main__":
    unittest.main()
