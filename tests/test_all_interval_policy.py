"""Focused regressions for all-interval identity, Part 7, and central approval."""
import ast
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace

import pandas as pd

from binance_timeframes import BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_CORRELATION_GROUPS
from part7_signal import _validate_frame, aggregate_results
from jarvis_strategy_approval import evaluate_mtf_central_strategy

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "BTCUSDT"


def _parts(tf: str, signal: int = 1):
    names = (
        "part1_breakout", "part2_zone", "part3_psychology", "part4_volume",
        "part5_ml", "part6_trend", "part7_volatility", "part8_structure",
        "part9_orderflow", "part10_candlestats",
    )
    rows = {}
    for name in names:
        value = signal
        thought = "native bullish evidence"
        row = {"signal": value, "thought": thought, "symbol": SYMBOL, "timeframe": tf}
        if name == "part7_volatility":
            row.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                        "data_status": "valid", "entry_blocked": False, "risk_veto": False})
        if name == "part2_zone":
            row["native_timeframe"] = tf
        rows[name] = row
    return rows


def _gate():
    return {
        "entry_blocked": False, "risk_veto": False, "status": "ok",
        "data_status": "valid", "timeframe": "aggregate", "symbol": SYMBOL,
        "blocked_timeframes": [], "veto_timeframes": [],
        "timeframe_results": {
            tf: {"symbol": SYMBOL, "timeframe": tf, "status": "neutral",
                 "data_status": "valid", "entry_blocked": False, "risk_veto": False}
            for tf in BINANCE_SPOT_TIMEFRAMES
        },
    }


def test_part7_accepts_monthly_frame_and_uses_native_calendar_interval():
    index = pd.date_range("2020-01-01", periods=60, freq="MS", tz="UTC")
    price = [100.0] * len(index)
    frame = pd.DataFrame({"open": price, "high": [101.0] * 60, "low": [99.0] * 60,
                          "close": price, "volume": [1000.0] * 60}, index=index)
    frame.attrs["symbol"] = SYMBOL
    assert _validate_frame(frame, SYMBOL, "1M", {"selected_symbol": SYMBOL,
                            "is_backtest_mode": True}) is None


def test_part7_all_frame_aggregate_rejects_missing_identity_and_blocking_rows():
    rows = {tf: {"symbol": SYMBOL, "timeframe": tf, "signal": 0, "status": "neutral",
                 "data_status": "valid", "entry_blocked": False, "risk_veto": False}
            for tf in BINANCE_SPOT_TIMEFRAMES}
    good = aggregate_results(rows, symbol=SYMBOL)
    assert good["entry_blocked"] is False
    assert set(good["timeframe_results"]) == set(BINANCE_SPOT_TIMEFRAMES)

    malformed = {tf: dict(row) for tf, row in rows.items()}
    malformed["1M"].pop("timeframe")
    assert aggregate_results(malformed, symbol=SYMBOL)["entry_blocked"] is True

    contradictory = {tf: dict(row) for tf, row in rows.items()}
    contradictory["3m"]["entry_blocked"] = True
    assert aggregate_results(contradictory, symbol=SYMBOL)["entry_blocked"] is True


def test_central_approval_consumes_all_fifteen_correlated_frames():
    evidence = {tf: _parts(tf) for tf in BINANCE_SPOT_TIMEFRAMES}
    result = evaluate_mtf_central_strategy(evidence, _gate(), confidence=80,
                                           expected_symbol=SYMBOL)
    assert result["approved"] is True
    assert result["direction"] == "BUY"
    assert result["trade_mode"] == "SWING"
    assert set(result["diagnostic"]["timeframe_decisions"]) == set(BINANCE_SPOT_TIMEFRAMES)
    assert set(result["diagnostic"]["swing"]["horizon_values"]) == {
        "short", "session", "swing", "macro"
    }
    for missing in ("3m", "1M"):
        incomplete = dict(evidence)
        incomplete.pop(missing)
        assert evaluate_mtf_central_strategy(incomplete, _gate(), expected_symbol=SYMBOL)["approved"] is False


def test_part2_legacy_mtf_engine_runs_once_not_once_per_frame():
    source = (ROOT / "jarvis_FIXED.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Part2Zone")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    harness = ast.ClassDef(name="Part2Harness", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    exec(compile(module, str(ROOT / "jarvis_FIXED.py"), "exec"), {})
    # Recreate the method in a namespace with no project imports required.
    namespace = {}
    exec(compile(module, str(ROOT / "jarvis_FIXED.py"), "exec"), namespace)
    calls = []
    engine = SimpleNamespace(process_market_data=lambda *args: (calls.append(args) or [("BUY", 90, "engine")]))
    instance = SimpleNamespace(_engine=engine)
    frame = pd.DataFrame({"close": [100.0] * 120, "high": [101.0] * 120,
                          "low": [99.0] * 120})
    mtf = {tf: frame for tf in ("1m", "5m", "15m")}
    one = namespace["Part2Harness"].analyze(instance, frame, {"timeframe": "1m", "mtf_datasets": mtf})
    five = namespace["Part2Harness"].analyze(instance, frame, {"timeframe": "5m", "mtf_datasets": mtf})
    assert one["thought"] == "Part2: engine"
    assert len(calls) == 1
    assert five["thought"] != "Part2: engine"
    assert len(calls) == 1
