from jarvis_scenario_simulator import ScenarioConfig
import os
import pytest
import pandas as pd
from jarvis_scenario_simulator import (
    scen_enabled,
    _min_pass,
    _atr,
    _scenario_normal_move,
    _scenario_slippage_spike,
    _scenario_wick_hunt,
    _scenario_volatility_collapse,
    _scenario_volume_drop,
    run_scenarios
)

def test_scen_enabled(monkeypatch):
    monkeypatch.delenv("JARVIS_SCEN_SIM", raising=False)
    assert scen_enabled() is True
    monkeypatch.setenv("JARVIS_SCEN_SIM", "0")
    assert scen_enabled() is False
    monkeypatch.setenv("JARVIS_SCEN_SIM", "1")
    assert scen_enabled() is True

def test_min_pass(monkeypatch):
    monkeypatch.delenv("JARVIS_SCEN_MIN_PASS", raising=False)
    assert _min_pass() == 3
    monkeypatch.setenv("JARVIS_SCEN_MIN_PASS", "1")
    assert _min_pass() == 1
    monkeypatch.setenv("JARVIS_SCEN_MIN_PASS", "5")
    assert _min_pass() == 5
    monkeypatch.setenv("JARVIS_SCEN_MIN_PASS", "10")
    assert _min_pass() == 5
    monkeypatch.setenv("JARVIS_SCEN_MIN_PASS", "0")
    assert _min_pass() == 1
    monkeypatch.setenv("JARVIS_SCEN_MIN_PASS", "invalid")
    assert _min_pass() == 3

def test_atr():
    # Test with insufficient data
    df_short = pd.DataFrame({"high": [10], "low": [9], "close": [9.5]})
    assert _atr(df_short, n=14) is None

    # Test with exact data
    data = []
    for i in range(20):
        data.append({"high": 10 + i, "low": 9 + i, "close": 9.5 + i})
    df = pd.DataFrame(data)
    atr_val = _atr(df, n=14)
    assert atr_val is not None
    assert atr_val > 0

def test_scenario_normal_move():
    # tp_dist <= 0
    ok, msg = _scenario_normal_move("CALL", 100, 90, 100, 5)
    assert not ok
    assert "distance zero" in msg

    # tp_dist > 3 * atr
    ok, msg = _scenario_normal_move("CALL", 100, 90, 120, 5)
    assert not ok
    assert "unlikely hit" in msg

    # normal
    ok, msg = _scenario_normal_move("CALL", 100, 90, 110, 5)
    assert ok

def test_scenario_slippage_spike():
    # risk <= 0
    ok, msg = _scenario_slippage_spike(100, 100, 110, 0, 0)
    assert not ok
    assert "Risk distance zero" in msg

    # RR < 0.5
    # entry=100, sl=99, tp=100.1, fee=10, slippage=50 => cost = 100*(10+150)/10000 = 1.6
    # reward = 0.1 - 1.6 = -1.5, risk = 1 + 1.6 = 2.6 -> RR < 0
    ok, msg = _scenario_slippage_spike(100, 99, 100.1, 10, 50)
    assert not ok
    assert "cost profit khai jashe" in msg

    # RR >= 0.5
    ok, msg = _scenario_slippage_spike(100, 90, 120, 10, 5)
    assert ok

def test_scenario_wick_hunt():
    # Insufficient candles
    df_short = pd.DataFrame([{"high": 10, "low": 9, "open": 9.5, "close": 9.5}] * 5)
    ok, msg = _scenario_wick_hunt("CALL", 100, 90, df_short, 5)
    assert ok
    assert "insufficient candles" in msg

    # STOP HUNT CALL
    data = []
    for i in range(10):
        # Create lower wick of 5
        data.append({"high": 105, "low": 95, "open": 100, "close": 101})
    df = pd.DataFrame(data)
    # Entry 100, SL 98 => buffer = 2, worst lower wick = min(o,c)-low = 100-95 = 5.
    # buffer (2) < 0.8 * worst (4)
    ok, msg = _scenario_wick_hunt("CALL", 100, 98, df, 5)
    assert not ok
    assert "stop-hunt risk" in msg

    # SAFE CALL
    ok, msg = _scenario_wick_hunt("CALL", 100, 90, df, 5)
    assert ok

    # STOP HUNT PUT
    data_put = []
    for i in range(10):
        # Create upper wick of 5
        data_put.append({"high": 105, "low": 95, "open": 99, "close": 100})
    df_put = pd.DataFrame(data_put)
    # Entry 100, SL 102 => buffer = 2, worst upper wick = high - max(o,c) = 105 - 100 = 5.
    # buffer (2) < 0.8 * worst (4)
    ok, msg = _scenario_wick_hunt("PUT", 100, 102, df_put, 5)
    assert not ok
    assert "stop-hunt risk" in msg

    # SAFE PUT
    ok, msg = _scenario_wick_hunt("PUT", 100, 110, df_put, 5)
    assert ok

def test_scenario_volatility_collapse():
    df_short = pd.DataFrame([{"high": 10, "low": 9}] * 10)
    ok, msg = _scenario_volatility_collapse(df_short, 5)
    assert ok
    assert "insufficient data" in msg

    data = [{"high": 10, "low": 9}] * 20
    # Last candle range = 0.5, ATR = 5 => 0.5 < 1.25 => False
    data[-1] = {"high": 10, "low": 9.5}
    df = pd.DataFrame(data)
    ok, msg = _scenario_volatility_collapse(df, 5)
    assert not ok
    assert "dead market" in msg

    # Alive market
    data[-1] = {"high": 15, "low": 9}
    df2 = pd.DataFrame(data)
    ok, msg = _scenario_volatility_collapse(df2, 5)
    assert ok

def test_scenario_volume_drop():
    df_no_vol = pd.DataFrame([{"high": 10, "low": 9}] * 20)
    ok, msg = _scenario_volume_drop(df_no_vol)
    assert ok
    assert "no volume data" in msg

    data = [{"volume": 100}] * 20
    data[-1] = {"volume": 20} # avg = 100, last = 20 < 0.3 * 100 (30) -> False
    df = pd.DataFrame(data)
    ok, msg = _scenario_volume_drop(df)
    assert not ok
    assert "thin liquidity" in msg

    data[-1] = {"volume": 50}
    df2 = pd.DataFrame(data)
    ok, msg = _scenario_volume_drop(df2)
    assert ok

def test_run_scenarios():
    # Insufficient inputs (fail-open)
    res = run_scenarios(ScenarioConfig(None, 0, 0, 0, None))
    assert res["action"] == "pass"
    assert res["passed"] == 0

    # Create a valid DF that passes all scenarios
    data = []
    for i in range(25):
        data.append({
            "open": 100, "close": 100, "high": 105, "low": 95, "volume": 1000
        })
    df = pd.DataFrame(data)

    res = run_scenarios(ScenarioConfig("CALL", 100, 90, 110, df))
    assert res["action"] == "pass"
    assert res["passed"] == 5

    # Veto case: Let's make ATR very small, volume drop, and tp very far.
    # TP 200 is > 3*ATR if ATR is small.
    # Volume drop: last volume = 100
    # Dead market: last range very small.
    data[-1] = {"open": 100, "close": 100, "high": 100.1, "low": 100.0, "volume": 10}
    df_veto = pd.DataFrame(data)

    res2 = run_scenarios(ScenarioConfig("CALL", 100, 90, 200, df_veto))
    assert res2["action"] == "veto"
    assert res2["passed"] < 3
