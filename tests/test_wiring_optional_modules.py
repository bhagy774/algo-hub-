"""Offline regression coverage for the optional brain wiring."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeDelta:
    def __init__(self):
        self.calls = []

    def get_historical_candles(self, **kwargs):
        self.calls.append(kwargs)
        base = 100.0 if kwargs["resolution"] != "5m" else 110.0
        return [
            {"time": i, "open": base + i, "high": base + i + 2, "low": base + i - 1,
             "close": base + i + 1, "volume": 5}
            for i in range(20)
        ]


def test_safe_config_never_enables_live_execution(monkeypatch):
    monkeypatch.setenv("JARVIS_ANALYSIS_MODE", "live")
    import trading_config
    cfg = importlib.reload(trading_config).get_active_config()
    assert cfg["mode"] == "paper"
    assert cfg["paper_trading"] is True
    assert cfg["live_execution"] is False
    assert cfg["auto_trade"] is False


def test_fetcher_calculator_and_engine_are_offline_and_deterministic():
    from delta_multi_tf_fetcher import DeltaMultiTFDataFetcher
    from multi_tf_engine import MultiTimeframeEngine
    from smart_tpsl_calculator import SmartTPSLCalculator

    delta = FakeDelta()
    fetcher = DeltaMultiTFDataFetcher(delta, default_limit=20)
    candles = fetcher.fetch_multi_timeframe("BTC", ("1m", "5m"))
    assert set(candles) == {"1m", "5m"}
    assert {call["symbol"] for call in delta.calls} == {"BTCUSDT"}

    calculator = SmartTPSLCalculator({"atr_period": 14, "default_atr_pct": 0.003})
    long_levels = calculator.calculate(candles["1m"], "CALL")
    short_levels = calculator.calculate(candles["1m"], "PUT")
    assert long_levels["stop_loss"] < long_levels["entry_price"] < long_levels["take_profit"]
    assert short_levels["take_profit"] < short_levels["entry_price"] < short_levels["stop_loss"]
    assert long_levels["risk_reward"] >= 1.2

    engine = MultiTimeframeEngine(delta_fetcher=fetcher, tpsl_calculator=calculator,
                                  config={"scalping_timeframes": ("1m", "5m"), "min_timeframes": 2})
    result = engine.analyze("BTC", "scalping")
    assert result["direction"] == "CALL"
    assert result["stop_loss"] < result["entry_price"] < result["take_profit_2"]


def test_fetcher_fails_closed_when_client_is_unavailable():
    from delta_multi_tf_fetcher import DeltaMultiTFDataFetcher
    assert DeltaMultiTFDataFetcher().fetch_multi_timeframe("BTC") == {}


def test_brain_imports_cleanly_with_wired_modules_present(tmp_path):
    script = "import jarvis_FIXED as b; assert b.SIZER_AVAILABLE; assert b.POSITION_MANAGER_AVAILABLE"
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               env={**os.environ, "PYTHONPATH": str(ROOT), "JARVIS_ENABLE_EXTERNAL_AI": "0"}, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    assert completed.returncode == 0, completed.stderr[-1000:]


def test_brain_initializes_optional_wiring_without_network(tmp_path):
    # All HTTP paths are blocked before importing the brain, including local LegacyLLM.
    script = '''\
import requests
from requests.sessions import Session
def offline(*args, **kwargs):
    raise AssertionError("network access is prohibited in this test")
requests.get = offline
requests.post = offline
Session.request = offline
import jarvis_FIXED as brain
jarvis = brain.JarvisElite()
assert jarvis.position_sizer is not None
assert jarvis.position_manager is not None
assert jarvis.coin_scanner is not None
assert jarvis.specialist_pool_class is not None
if jarvis.bus:
    jarvis.bus.shutdown()
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               env={**os.environ, "PYTHONPATH": str(ROOT), "JARVIS_ENABLE_EXTERNAL_AI": "0"}, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    assert completed.returncode == 0, completed.stderr[-1000:]


@pytest.mark.parametrize(("module_name", "availability_flag"), [
    ("jarvis_sizer", "SIZER_AVAILABLE"),
    ("jarvis_position_manager", "POSITION_MANAGER_AVAILABLE"),
    ("jarvis_specialist_pool", "SPECIALIST_POOL_AVAILABLE"),
    ("jarvis_coin_scanner", "COIN_SCANNER_AVAILABLE"),
    ("binance_data", "BINANCE_DATA_AVAILABLE"),
    ("upstox_data", "UPSTOX_DATA_AVAILABLE"),
    ("jarvis_backtester", "BACKTESTER_AVAILABLE"),
])
def test_brain_fails_safe_when_each_optional_module_is_missing(tmp_path, module_name, availability_flag):
    # A fresh child process prevents import caches from hiding missing-module behavior.
    script = f'''\
import builtins
import requests
from requests.sessions import Session
def offline(*args, **kwargs):
    raise AssertionError("network access is prohibited in this test")
requests.get = offline
requests.post = offline
Session.request = offline
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == {module_name!r} or name.startswith({module_name!r} + "."):
        raise ModuleNotFoundError("simulated missing optional module")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
import jarvis_FIXED as brain
assert getattr(brain, {availability_flag!r}) is False
jarvis = brain.JarvisElite()
if jarvis.bus:
    jarvis.bus.shutdown()
'''
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               env={**os.environ, "PYTHONPATH": str(ROOT), "JARVIS_ENABLE_EXTERNAL_AI": "0"}, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    assert completed.returncode == 0, completed.stderr[-1000:]
