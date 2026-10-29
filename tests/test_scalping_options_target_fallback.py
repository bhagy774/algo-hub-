import ast
import logging
import math
from pathlib import Path

import pandas as pd


def _load_scalping_engine():
    source = Path(__file__).resolve().parents[1] / "jarvis_FIXED.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "ScalpingEngine")
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {"math": math, "logger": logging.getLogger("test.scalping")}
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["ScalpingEngine"]


def test_null_options_levels_do_not_break_target_calculation():
    engine = _load_scalping_engine()()
    data = pd.DataFrame({"high": [101.0] * 30, "low": [99.0] * 30})

    targets = engine.calculate_targets(
        data,
        "CALL",
        100.0,
        options_data={"support": None, "resistance": None, "max_pain": None},
    )

    assert targets is not None
    assert targets["stop_loss"] < 100.0 < targets["take_profit_1"]
    assert targets["options_magnet"] == 100.0
