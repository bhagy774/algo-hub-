#!/usr/bin/env python3
"""Offline trainer + validator for the XGBoost meta-gate (never run by the bot).

Real Binance spot 15m candles -> shared feature_matrix() -> triple-barrier
labels (entry at NEXT bar open, ATR barriers, round-trip cost, same-bar
ambiguity counted as a loss, timeout = 0) -> chronological train/val/test with
a purge gap -> metrics on the untouched test block.

The model is marked ``validated: true`` only if the test block shows real
edge over the unfiltered baseline. Otherwise the sidecar says false and the
runtime gate stays neutral (it cannot veto anything).

Usage:
  python train_xgb_meta.py --symbol BTCUSDT --days 540
  python train_xgb_meta.py --symbol ETHUSDT --csv my_15m.csv   # offline file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jarvis_xgb_meta as M  # noqa: E402

TP_ATR, SL_ATR, HORIZON = 1.5, 1.0, 24          # 24 x 15m = 6h max hold
COST_PCT = float(os.getenv("JARVIS_XGB_COST_PCT", "0.0008"))  # fee+slippage round trip
PURGE = HORIZON + 2


def fetch_binance_15m(symbol: str, days: int) -> pd.DataFrame:
    import requests
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    rows, cur = [], start
    while cur < end:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": "15m", "startTime": cur, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        cur = batch[-1][0] + 15 * 60 * 1000
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "ig"])
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
    # drop the still-forming last candle
    df = df[df["ts"] + 15 * 60 * 1000 <= end].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def triple_barrier(df: pd.DataFrame, atr_pct: np.ndarray, sign: float):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    y = np.full(n, -1, dtype=np.int8)
    r = np.full(n, np.nan)
    for i in range(n - HORIZON - 1):
        a = atr_pct[i] * c[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = o[i + 1]
        cost = entry * COST_PCT
        tp, sl = TP_ATR * a, SL_ATR * a
        outcome = None
        for j in range(i + 1, i + 1 + HORIZON):
            if sign > 0:
                hit_tp, hit_sl = h[j] >= entry + tp + cost, l[j] <= entry - sl
            else:
                hit_tp, hit_sl = l[j] <= entry - tp - cost, h[j] >= entry + sl
            if hit_sl:                      # same-bar ambiguity -> loss (conservative)
                outcome = (0, -(sl + cost) / sl)
                break
            if hit_tp:
                outcome = (1, tp / sl)
                break
        if outcome is None:
            end_px = c[i + HORIZON]
            outcome = (0, (sign * (end_px - entry) - cost) / sl)
        y[i], r[i] = outcome
    return y, r


def build_dataset(df: pd.DataFrame):
    atr = M.compute_atr_pct(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float))
    Xs, ys, rs, idx = [], [], [], []
    for sign in (1.0, -1.0):
        X = M.feature_matrix(df, sign)
        y, r = triple_barrier(df, atr, sign)
        ok = np.all(np.isfinite(X), axis=1) & (y >= 0)
        Xs.append(X[ok]); ys.append(y[ok]); rs.append(r[ok]); idx.append(np.where(ok)[0])
    X, y, r, idx = np.vstack(Xs), np.concatenate(ys), np.concatenate(rs), np.concatenate(idx)
    order = np.argsort(idx, kind="stable")
    return X[order], y[order], r[order], idx[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--csv")
    ap.add_argument("--out", default=str(M.MODEL_DIR))
    a = ap.parse_args()
    symbol = M.clean_symbol(a.symbol)
    import xgboost as xgb

    if a.csv:
        df = pd.read_csv(a.csv)
        df.columns = [c.lower() for c in df.columns]
        if "ts" not in df.columns:
            df["ts"] = np.arange(len(df)) * 900000.0
    else:
        df = fetch_binance_15m(symbol, a.days)
    if len(df) < 4000:
        print(f"ABORT: only {len(df)} candles; need >= 4000 for a meaningful split")
        return 2
    print(f"{symbol}: {len(df)} 15m candles {pd.to_datetime(df.ts.iloc[0], unit='ms')} -> {pd.to_datetime(df.ts.iloc[-1], unit='ms')}")

    X, y, r, idx = build_dataset(df)
    n_bars = len(df)
    cut1, cut2 = int(n_bars * 0.60), int(n_bars * 0.80)
    tr = idx < cut1 - PURGE
    va = (idx >= cut1) & (idx < cut2 - PURGE)
    te = idx >= cut2
    print(f"samples train/val/test: {tr.sum()}/{va.sum()}/{te.sum()}  base win rate train={y[tr].mean():.3f} test={y[te].mean():.3f}")

    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50, "lambda": 5.0, "verbosity": 0,
              "tree_method": "hist", "seed": 7}
    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "cuda"
    except Exception:
        pass
    dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=M.FEATURE_NAMES)
    dva = xgb.DMatrix(X[va], label=y[va], feature_names=M.FEATURE_NAMES)
    dte = xgb.DMatrix(X[te], label=y[te], feature_names=M.FEATURE_NAMES)
    model = xgb.train(params, dtr, num_boost_round=600, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False)
    pva, pte = model.predict(dva), model.predict(dte)

    from sklearn.metrics import log_loss, roc_auc_score
    def auc(yt, p):
        return float(roc_auc_score(yt, p)) if len(set(yt)) > 1 else float("nan")

    # Threshold chosen on VALIDATION only (veto if p < thr): best expectancy of kept trades,
    # keeping at least 40% of validation samples.
    best_thr, best_e = None, -1e9
    for thr in np.quantile(pva, np.linspace(0.05, 0.60, 23)):
        keep = pva >= thr
        if keep.mean() >= 0.40 and keep.sum() >= 100:
            e = float(r[va][keep].mean())
            if e > best_e:
                best_thr, best_e = float(thr), e
    base_exp = float(r[te].mean())
    thr = best_thr if best_thr is not None else float(np.median(pva))
    kept = pte >= thr
    oos = {
        "auc": auc(y[te], pte), "logloss": float(log_loss(y[te], np.clip(pte, 1e-6, 1 - 1e-6))),
        "logloss_base_rate": float(log_loss(y[te], np.full(len(pte), y[tr].mean()))),
        "n_test": int(te.sum()), "base_win_rate": float(y[te].mean()),
        "expectancy_R_unfiltered": base_exp,
        "expectancy_R_filtered": float(r[te][kept].mean()) if kept.any() else None,
        "n_kept": int(kept.sum()), "kept_fraction": float(kept.mean()),
        "win_rate_filtered": float(y[te][kept].mean()) if kept.any() else None,
        "val_auc": auc(y[va], pva),
    }
    reasons = []
    if not (oos["auc"] >= M.MIN_VALID_AUC):
        reasons.append(f"test AUC {oos['auc']:.3f} < {M.MIN_VALID_AUC}")
    if oos["n_test"] < M.MIN_VALID_TEST_SAMPLES:
        reasons.append("too few test samples")
    if oos["expectancy_R_filtered"] is None or oos["expectancy_R_filtered"] <= base_exp:
        reasons.append("filter does not improve expectancy over unfiltered baseline")
    if oos["n_kept"] < 100 or oos["kept_fraction"] < 0.30:
        reasons.append("filter keeps too few trades")
    validated = not reasons
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mp, metap = M.model_paths(symbol, out)
    model.save_model(str(mp))
    meta = {
        "version": time.strftime("%Y%m%d-%H%M%S"), "symbol": symbol, "timeframe": M.TIMEFRAME,
        "data_kind": "real_historical_ohlcv", "feature_names": M.FEATURE_NAMES,
        "label": {"type": "triple_barrier", "tp_atr": TP_ATR, "sl_atr": SL_ATR, "horizon_bars": HORIZON,
                  "cost_pct_round_trip": COST_PCT, "entry": "next_bar_open", "same_bar": "loss"},
        "split": {"train": "first 60%", "val": "next 20% (threshold + early stop)", "test": "last 20% untouched", "purge_bars": PURGE},
        "data_range": [int(df.ts.iloc[0]), int(df.ts.iloc[-1])], "n_candles": int(len(df)),
        "threshold": thr, "oos": oos, "validated": validated, "rejected_reasons": reasons,
        "note": "Probability is P(hit 1.5R before 1R within 24 bars). Veto-only advisory.",
    }
    metap.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("symbol", "threshold", "oos", "validated", "rejected_reasons")}, indent=2))
    print("VALIDATED -> gate may veto in JARVIS_XGB_MODE=veto" if validated else "NOT VALIDATED -> gate stays neutral (cannot veto)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
