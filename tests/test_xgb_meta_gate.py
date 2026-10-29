import json

import numpy as np
import pandas as pd
import pytest

import jarvis_xgb_meta as M


def candles(n=200, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.concatenate(([c[0]], c[:-1]))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(10, 20, n)})


def write_model(tmp_path, file_symbol="BTCUSDT", **meta_over):
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, len(M.FEATURE_NAMES)))
    y = (X[:, 13] > 0).astype(int)  # depends on direction feature -> BUY high p, SELL low p
    b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "verbosity": 0},
                  xgb.DMatrix(X, label=y, feature_names=M.FEATURE_NAMES), 30)
    mp, metap = M.model_paths(file_symbol, tmp_path)
    b.save_model(str(mp))
    meta = {"symbol": file_symbol, "timeframe": "15m", "data_kind": "real_historical_ohlcv",
            "feature_names": M.FEATURE_NAMES, "validated": True, "threshold": 0.5,
            "oos": {"auc": 0.6, "n_test": 1000}, "version": "t"}
    meta.update(meta_over)
    metap.write_text(json.dumps(meta))


def gate(tmp_path, mode):
    return M.XGBMetaGate(model_dir=tmp_path, mode=mode, shadow_log=tmp_path / "s.jsonl")


def test_no_model_is_neutral_and_cannot_veto(tmp_path):
    r = gate(tmp_path, "veto").evaluate(candles(), "BUY", "BTCUSDT")
    assert r["status"] == "UNAVAILABLE" and r["probability"] == 0.5
    assert r["can_veto"] is False and r["veto"] is False


def test_unvalidated_or_synthetic_model_never_vetoes(tmp_path):
    for over in ({"validated": False}, {"data_kind": "synthetic"}, {"oos": {"auc": 0.51, "n_test": 1000}},
                 {"feature_names": ["x"]}, {"timeframe": "1m"}):
        write_model(tmp_path, **over)
        r = gate(tmp_path, "veto").evaluate(candles(), "SELL", "BTCUSDT")
        assert r["status"] == "UNAVAILABLE" and r["veto"] is False, over


def test_no_cross_symbol_fallback(tmp_path):
    write_model(tmp_path, "ETHUSDT")
    r = gate(tmp_path, "veto").evaluate(candles(), "BUY", "BTCUSDT")
    assert r["status"] == "UNAVAILABLE"
    write_model(tmp_path, "BTCUSDT", symbol="ETHUSDT")  # meta says ETH but file is BTC's
    assert gate(tmp_path, "veto").evaluate(candles(), "BUY", "BTCUSDT")["status"] == "UNAVAILABLE"


def test_validated_model_veto_only_in_veto_mode(tmp_path):
    write_model(tmp_path)
    df = candles()
    buy = gate(tmp_path, "veto").evaluate(df, "BUY", "BTCUSDT")
    sell = gate(tmp_path, "veto").evaluate(df, "SELL", "BTCUSDT")
    assert buy["status"] == sell["status"] == "OK"
    assert buy["probability"] > 0.5 > sell["probability"]
    assert buy["veto"] is False and sell["veto"] is True
    shadow = gate(tmp_path, "shadow").evaluate(df, "SELL", "BTCUSDT")
    assert shadow["status"] == "OK" and shadow["veto"] is False and shadow["can_veto"] is False
    assert gate(tmp_path, "off").evaluate(df, "SELL", "BTCUSDT")["status"] == "UNAVAILABLE"


def test_bad_input_fails_neutral(tmp_path):
    write_model(tmp_path)
    g = gate(tmp_path, "veto")
    for df, d in ((None, "BUY"), (candles(10), "BUY"), (candles(), "NO_TRADE"), (candles(), None)):
        r = g.evaluate(df, d, "BTCUSDT")
        assert r["probability"] == 0.5 and r["veto"] is False
    bad = candles(); bad.loc[50, "close"] = np.nan
    assert g.evaluate(bad, "SELL", "BTCUSDT")["veto"] is False


def test_features_have_no_lookahead():
    df = candles(300)
    a = M.feature_matrix(df.iloc[:200], 1.0)
    b = M.feature_matrix(df, 1.0)
    np.testing.assert_allclose(a[60:200], b[60:200], equal_nan=True)


def test_synthetic_trainers_are_disabled():
    import jarvis_continuous_learner as L
    with pytest.raises(RuntimeError):
        L.ContinuousLearner()
    import jarvis_xgboost_engine as E
    eng = E.JarvisXGBoostEngine(model_path="/nonexistent/none.json")
    r = eng.evaluate(np.zeros(48), "BUY")
    assert r["approved"] is False and r["status"] == "UNAVAILABLE"
