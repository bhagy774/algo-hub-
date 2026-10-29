"""Offline tests: learning loop. No network."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    monkeypatch.delenv("JARVIS_LEARNING", raising=False)
    monkeypatch.chdir(tmp_path)
    import jarvis_learning as jl
    jl._cache = None
    jl._cache_path = None
    yield


# ── Learning loop ─────────────────────────────────────────────────────

import jarvis_learning as jl


def _trade(result="WIN", direction="CALL"):
    return {"direction": direction, "entry_price": 100.0, "exit_price": 101.0,
            "result": result, "pnl_dollar": 5.0, "close_reason": "test",
            "confidence": 80}


def test_record_trade_and_stats(tmp_path):
    p = str(tmp_path / "learn.json")
    assert jl.record_trade(_trade("WIN"), engine_signals={"part1": 1, "part2": -1}, path=p)
    data = json.load(open(p))
    assert len(data["trades"]) == 1
    t = data["trades"][0]
    assert t["symbol"] == "BTC/USDT" and t["pnl"] == 5.0 and "timestamp" in t
    # only agreeing engine credited
    assert data["engines"]["part1"]["wins"] == 1
    assert "part2" not in data["engines"]


def test_record_trade_without_engine_signals(tmp_path):
    p = str(tmp_path / "learn.json")
    assert jl.record_trade(_trade("LOSS"), path=p)
    data = json.load(open(p))
    assert data["trades"][0]["result"] == "LOSS"
    assert "engine_signals" not in data["trades"][0]


def test_weight_neutral_below_min_trades(tmp_path):
    p = str(tmp_path / "learn.json")
    for _ in range(5):
        jl.record_trade(_trade("WIN"), engine_signals={"e": 1}, path=p)
    assert jl.get_engine_weight("e", path=p) == 1.0


def test_weight_scaling_and_bounds(tmp_path):
    p = str(tmp_path / "learn.json")
    # 10/10 wins -> 1.5
    for _ in range(10):
        jl.record_trade(_trade("WIN"), engine_signals={"hot": 1}, path=p)
    assert jl.get_engine_weight("hot", path=p) == 1.5
    # 0/10 wins -> 0.5
    for _ in range(10):
        jl.record_trade(_trade("LOSS"), engine_signals={"cold": 1}, path=p)
    assert jl.get_engine_weight("cold", path=p) == 0.5
    # 5/10 -> 1.0
    for i in range(10):
        jl.record_trade(_trade("WIN" if i % 2 == 0 else "LOSS"),
                        engine_signals={"mid": 1}, path=p)
    assert jl.get_engine_weight("mid", path=p) == pytest.approx(1.0)


def test_learning_disabled_env(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_LEARNING", "0")
    p = str(tmp_path / "learn.json")
    assert jl.record_trade(_trade("WIN"), engine_signals={"e": 1}, path=p) is False
    assert not os.path.exists(p)
    assert jl.get_engine_weight("e", path=p) == 1.0


def test_corrupt_file_graceful(tmp_path):
    p = tmp_path / "learn.json"
    p.write_text("{not valid json!!")
    jl._cache = None; jl._cache_path = None
    assert jl.get_engine_weight("e", path=str(p)) == 1.0
    # can still record after corruption
    assert jl.record_trade(_trade("WIN"), path=str(p)) is True
    assert len(json.load(open(p))["trades"]) == 1


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = str(tmp_path / "learn.json")
    jl.record_trade(_trade("WIN"), path=p)
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".jarvis_learning_")]
    assert leftovers == []


def test_breakeven_not_win_or_loss(tmp_path):
    p = str(tmp_path / "learn.json")
    for _ in range(10):
        jl.record_trade(_trade("BREAKEVEN"), engine_signals={"e": 1}, path=p)
    stats = jl.get_engine_stats("e", path=p)
    assert stats["trades"] == 10 and stats["wins"] == 0 and stats["losses"] == 0
    assert jl.get_engine_weight("e", path=p) == 1.0  # 0 decided -> win_rate 0 -> but trades>=10
