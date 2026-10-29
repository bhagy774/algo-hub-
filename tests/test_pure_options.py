"""Behavior tests against the production Part14 options method, isolated via AST."""
import ast
import os
import unittest
from pathlib import Path

SOURCE = (Path(__file__).parent.parent / "jarvis_FIXED.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Part14OptionsChain")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "analyze_options_deterministically")
deterministic = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "analyze_options_deterministically")
module = ast.Module(body=[deterministic], type_ignores=[])

class Fake:
    last_legacy_llm_whale_tag = "WHALE_BULLISH"
    last_legacy_llm_insight = "stale model result must not leak"
    last_legacy_llm_time = 0
    legacy_llm_cooldown = 0
    def _generate_legacy_llm_options_prompt(self, *_):
        return "test prompt"
    def analyze_options_deterministically(self, telemetry, current_price):
        return ("MODEL_ADVISORY_DISABLED", "Options result is deterministic observed-chain math", telemetry.get("signal", 0))

class OptionsPathTests(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get("JARVIS_PURE_ALGO")
        self.calls = []
        ns = {
            "Dict": dict, "Tuple": tuple,
            "_pure_algorithm_mode": lambda: os.getenv("JARVIS_PURE_ALGO", "true").lower() in ("true", "1", "yes", "on"),
            "time": type("Clock", (), {"time": staticmethod(lambda: 999999)})(),
            "LEGACY_LLM_INTEGRATION_AVAILABLE": True,
            "call_legacy_llm": lambda *a, **kw: self.calls.append(a) or ("[WHALE_BEARISH]", None),
            "logging": __import__("logging"),
        }
        exec(compile(module, "jarvis_FIXED.py", "exec"), ns)
        self.method = ns["analyze_options_deterministically"]
        Fake.analyze_options_deterministically = ns["analyze_options_deterministically"]
    def tearDown(self):
        if self.old is None: os.environ.pop("JARVIS_PURE_ALGO", None)
        else: os.environ["JARVIS_PURE_ALGO"] = self.old
    def test_pure_returns_math_and_unavailable_without_model_or_stale_tag(self):
        os.environ["JARVIS_PURE_ALGO"] = "true"
        result = self.method(Fake(), {"signal": -1}, 100.0)
        self.assertEqual(result, ("DETERMINISTIC_CHAIN_ONLY", "Options result uses validated provider observations; Laya is commentary-only", -1))
        self.assertEqual(self.calls, [])
    def test_non_pure_mode_still_cannot_enable_model_path(self):
        os.environ["JARVIS_PURE_ALGO"] = "false"
        result = self.method(Fake(), {"signal": 0}, 100.0)
        self.assertEqual(result, ("DETERMINISTIC_CHAIN_ONLY", "Options result uses validated provider observations; Laya is commentary-only", 0))
        self.assertEqual(self.calls, [])

if __name__ == "__main__": unittest.main()
