import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jarvis_part_snapshot as ps


class _F:  # fake frame
    def __init__(self, df): self.closed = df
class _S:
    def __init__(self, df): self.frames = {"15m": _F(df)}


def _df(n=120, start=1_700_000_000):
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, .3, n))
    return pd.DataFrame({"open": c, "high": c + .3, "low": c - .3, "close": c,
                         "volume": rng.uniform(1, 2, n),
                         "close_time": start + np.arange(n) * 900})


def test_flatten_numbers_signals_ignores_text():
    r = ps.flatten_result({"p1": {"signal": "CALL", "conf": 0.7, "thought": "x", "ok": True},
                           "p2": [1, 2], "bad": float("nan"), "s": "hello"})
    assert r == {"p1.signal": 1.0, "p1.conf": 0.7, "p1.ok": 1.0, "p2[0]": 1.0, "p2[1]": 2.0}


def test_record_once_per_bar_and_symbol(tmp_path):
    rec = ps.PartSnapshotRecorder(tmp_path / "s.jsonl")
    snap = _S(_df())
    assert rec.record("btc/usdt", {"a": {"signal": 1}}, snap, "CALL", 70)
    assert not rec.record("BTCUSDT", {"a": {"signal": 1}}, snap, "CALL", 70)  # same bar
    assert rec.record("ETHUSDT", {"a": {"signal": -1}}, snap, "PUT", 60)      # other symbol
    rows = [json.loads(l) for l in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert [r["symbol"] for r in rows] == ["BTCUSDT", "ETHUSDT"]
    assert rows[0]["parts"] == {"a.signal": 1.0}


def test_fail_neutral(tmp_path):
    rec = ps.PartSnapshotRecorder(tmp_path / "s.jsonl")
    assert rec.record("BTCUSDT", None, None) is False
    assert rec.record("BTCUSDT", {"a": 1}, _S(_df(1))) is False
    assert not (tmp_path / "s.jsonl").exists()
