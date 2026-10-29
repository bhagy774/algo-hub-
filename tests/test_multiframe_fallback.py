"""Regression test: missing MTF engine must never switch to Delta-only candles."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace


class _Logger:
    def error(self, *args, **kwargs):
        pass


def test_multitimeframe_fallback_fails_closed_without_delta_or_bad_keyword():
    source_path = Path(__file__).resolve().parents[1] / "jarvis_FIXED.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    target_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "Jarvis4EngineSystem")
    method_node = next(node for node in target_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == "analyze_multi_tf")
    isolated = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"logger": _Logger()}
    exec(compile(isolated, str(source_path), "exec"), namespace)

    calls = []
    jarvis = SimpleNamespace(_get_no_trade_signal=lambda reason: calls.append(reason) or {"blocked": reason})
    instance = SimpleNamespace(multi_tf_engine=None, jarvis=jarvis)
    method = MethodType(namespace["analyze_multi_tf"], instance)

    result = method(symbol="BTCUSDT")

    assert result == {"blocked": "WAIT/NO-DATA: native multi-timeframe engine unavailable"}
    assert calls == ["WAIT/NO-DATA: native multi-timeframe engine unavailable"]
