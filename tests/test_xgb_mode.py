import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xgboost as xgb
import jarvis_xgb_mode as m
import jarvis_part_snapshot as ps
import train_xgb_parts as tr

NAMES = [f"f{i}" for i in range(8)]


def _model(tmp, symbol="BTCUSDT", validated=True, favour=0):
    """Test-only booster: class `favour` is predicted whenever f0 > 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8))
    y = np.where(X[:, 0] > 0, favour, rng.integers(0, 5, 600))
    b = xgb.train(dict(objective="multi:softprob", num_class=5, max_depth=3, eta=0.5),
                  xgb.DMatrix(X, label=y, feature_names=NAMES), 30)
    mp, metap = m.model_paths(symbol, tmp)
    b.save_model(str(mp))
    metap.write_text(json.dumps(dict(symbol=symbol, classes=m.CLASSES, feature_names=NAMES,
        data_kind="recorded_live_snapshots", validated=validated, veto_threshold=0.6,
        oos=dict(auc=0.6, n_test=500), version=1)))


def _adv(tmp, mode):
    return m.XGBModeAdvisor(model_dir=tmp, mode=mode, log_path=tmp / "a.jsonl")


F = {f"f{i}": (3.0 if i == 0 else 0.0) for i in range(8)}


def test_unavailable_without_model(tmp_path):
    r = _adv(tmp_path, "veto").advise("BTCUSDT", F, "CALL")
    assert r["status"] == "UNAVAILABLE" and not r["veto"] and r["advice"] == "UNKNOWN"


def test_unvalidated_never_vetoes(tmp_path):
    _model(tmp_path, validated=False)
    r = _adv(tmp_path, "veto").advise("BTCUSDT", F, "CALL")
    assert r["status"] == "UNAVAILABLE" and not r["veto"]


def test_no_cross_symbol_fallback(tmp_path):
    _model(tmp_path, "BTCUSDT")
    assert _adv(tmp_path, "veto").advise("ETHUSDT", F, "CALL")["status"] == "UNAVAILABLE"


def test_shadow_gives_advice_but_never_vetoes(tmp_path):
    _model(tmp_path, favour=0)  # model says NO_TRADE
    r = _adv(tmp_path, "shadow").advise("BTCUSDT", F, "CALL")
    assert r["status"] == "OK" and r["advice"] == "NO_TRADE" and r["veto"] is False


def test_veto_mode_blocks_on_no_trade_or_opposite(tmp_path):
    _model(tmp_path, favour=0)
    assert _adv(tmp_path, "veto").advise("BTCUSDT", F, "CALL")["veto"] is True
    _model(tmp_path, favour=2)  # SCALP_SHORT
    a = _adv(tmp_path, "veto")
    assert a.advise("BTCUSDT", F, "CALL")["veto"] is True   # opposite direction
    assert a.advise("BTCUSDT", F, "PUT")["veto"] is False   # agrees: never forced/created


def test_too_few_features_neutral(tmp_path):
    _model(tmp_path)
    assert _adv(tmp_path, "veto").advise("BTCUSDT", {"f0": 3.0}, "CALL")["status"] == "UNAVAILABLE"


def test_off_mode(tmp_path):
    _model(tmp_path)
    assert _adv(tmp_path, "off").advise("BTCUSDT", F, "CALL")["reason"] == "mode off"


def test_labels_triple_barrier():
    n = 130
    bars = 1.7e9 + np.arange(n) * 900
    up = 100 * np.cumprod(np.full(n, 1.0004))   # +4bp/bar: swing long wins
    y = tr.label_rows(up, bars, (4, .005, .003), (96, .02, .008), 0.0)
    assert y[0] == 3 and np.isnan(y[-1])
    flat = np.full(n, 100.0)
    assert tr.label_rows(flat, bars, (4, .005, .003), (96, .02, .008), 0.0)[0] == 0


def test_all_timeframes_features_and_trainer_refuses_small(tmp_path):
    rng = np.random.default_rng(1)
    def df(n):
        c = 100 + np.cumsum(rng.normal(0, .3, n))
        return pd.DataFrame({"open": c, "high": c + .3, "low": c - .3, "close": c,
                             "volume": rng.uniform(1, 2, n), "close_time": 1.7e9 + np.arange(n) * 900})
    class F_: 
        def __init__(s, d): s.closed = d
    class S_: frames = {"1m": F_(df(120)), "15m": F_(df(120)), "1h": F_(df(120)), "4h": F_(df(10))}
    feats = ps.build_feature_row({"part1": {"signal": "CALL", "c": 0.7}}, S_)
    assert any(k.startswith("tf1m.") for k in feats) and any(k.startswith("tf1h.") for k in feats)
    assert not any(k.startswith("tf4h.") for k in feats)  # too little history: absent, not fabricated
    assert feats["part1.signal"] == 1.0
    (tmp_path / "l.jsonl").write_text("")
    assert tr.main(["--symbol", "BTCUSDT", "--log", str(tmp_path / "l.jsonl")]) == 2
