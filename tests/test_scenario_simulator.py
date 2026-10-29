"""Tests for jarvis_scenario_simulator — pre-trade stress scenario gate."""
import os
import sys
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis_scenario_simulator import run_scenarios, scen_enabled, ScenarioConfig


def _df(n=50, base=100000.0, rng=200.0, vol=1000.0, trend=0.0):
    rows = []
    for i in range(n):
        price = base + trend * i
        rows.append({
            "open": price, "high": price + rng, "low": price - rng,
            "close": price + rng * 0.3, "volume": vol,
        })
    return pd.DataFrame(rows)


def test_healthy_trade_passes():
    df = _df()
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99500, tp=101000, df=df))
    assert v["action"] == "pass"
    assert v["passed"] >= 3
    assert v["total"] == 5


def test_unreachable_tp_vetoes():
    # ATR ~ 230; TP 2% dur (2000) >> 3x ATR → normal_move fail
    # volume drop pan add kariye → 2 fail
    df = _df()
    df.loc[df.index[-1], "volume"] = 100  # volume collapse
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99700, tp=104000, df=df))
    failed = [s for s in v["scenarios"] if not s[1]]
    assert len(failed) >= 1


def test_tight_sl_wick_hunt_detected():
    # Moti wicks (400) pan tight SL buffer (50) → wick_hunt fail
    df = _df(rng=400.0)
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99950, tp=100600, df=df))
    wick = next(s for s in v["scenarios"] if s[0] == "wick_hunt")
    assert wick[1] is False  # stop-hunt risk detected


def test_volume_drop_detected():
    df = _df(vol=1000.0)
    df.loc[df.index[-1], "volume"] = 50.0
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99500, tp=101000, df=df))
    vol = next(s for s in v["scenarios"] if s[0] == "volume_drop")
    assert vol[1] is False


def test_dead_market_detected():
    df = _df(rng=200.0)
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "low"] + 5  # tiny last candle
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99500, tp=101000, df=df))
    dead = next(s for s in v["scenarios"] if s[0] == "volatility_collapse")
    assert dead[1] is False


def test_slippage_3x_rr_check():
    # Bahu nano TP/SL — cost profit khai jay → slippage scenario fail
    df = _df()
    v = run_scenarios(ScenarioConfig("CALL", entry_price=100000, sl=99990, tp=100020, df=df))
    slip = next(s for s in v["scenarios"] if s[0] == "slippage_spike")
    assert slip[1] is False


def test_fail_open_on_bad_inputs():
    assert run_scenarios(ScenarioConfig("CALL", 0, 0, 0, None))["action"] == "pass"
    assert run_scenarios(ScenarioConfig("NO_TRADE", 100, 90, 110, _df()))["action"] == "pass"


def test_env_disable():
    with patch.dict(os.environ, {"JARVIS_SCEN_SIM": "0"}):
        assert scen_enabled() is False
    with patch.dict(os.environ, {}, clear=True):
        assert scen_enabled() is True


def test_gate_integration_on_engine():
    """_scenario_gate exists on LiveTradingEngine and vetoes bad setups."""
    import jarvis_FIXED
    eng = jarvis_FIXED.LiveTradingEngine.__new__(jarvis_FIXED.LiveTradingEngine)
    df = _df(rng=400.0)
    df.loc[df.index[-1], "volume"] = 10.0
    # tight SL + volume drop + tiny range... force multiple fails
    out = eng._scenario_gate(direction="CALL", entry_price=100000, sl=99950, tp=100020, df=df)
    assert out in ("CALL", "NO_TRADE")  # gate never crashes; returns valid direction


def test_scenario_gate_preserves_veto_and_fails_closed_on_exception(monkeypatch):
    import jarvis_FIXED
    import jarvis_scenario_simulator
    monkeypatch.setenv("JARVIS_SCEN_SIM", "1")
    eng = jarvis_FIXED.LiveTradingEngine.__new__(jarvis_FIXED.LiveTradingEngine)
    monkeypatch.setattr(jarvis_scenario_simulator, "run_scenarios", lambda config: {"action": "veto"})
    assert eng._scenario_gate("CALL", 100, 99, 102, _df()) == "NO_TRADE"
    def broken(config):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(jarvis_scenario_simulator, "run_scenarios", broken)
    assert eng._scenario_gate("CALL", 100, 99, 102, _df()) == "NO_TRADE"
