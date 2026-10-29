"""Offline trainer: per-symbol XGBoost TRADE-MODE model (NO_TRADE/SCALP/SWING + direction).

Usage: python train_xgb_parts.py --symbol BTCUSDT [--device cuda]

Features: recorded Parts 1-12 / brains / options / MTF outputs + candle features of
EVERY timeframe of that coin (see jarvis_part_snapshot.py).
Labels (assumptions, change with flags): triple barrier on the coin's own later
recorded closes (15m resolution, real outcomes only):
  SCALP : horizon 4 bars (1h),  TP 0.5%, SL 0.3%
  SWING : horizon 96 bars (24h), TP 2.0%, SL 0.8%   (SWING SL/TP = strategy_approval)
  cost (bps) is subtracted from the TP. Class = SWING_x if exactly one side wins a
  swing barrier, else SCALP_x if exactly one side wins a scalp barrier, else NO_TRADE.
Chronological split 60/20/20 with purge; last block untouched test. validated=true
only if macro one-vs-rest OOS AUC >= 0.55 on >= 300 test rows AND log-loss beats the
class-prior baseline. Fewer than 1500 usable snapshots -> no model written.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

CLASSES = ["NO_TRADE", "SCALP_LONG", "SCALP_SHORT", "SWING_LONG", "SWING_SHORT"]
MIN_ROWS, MIN_TEST, MIN_AUC = 1500, 300, 0.55


def load_rows(path, symbol):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("symbol") == symbol:
                rows.append(r)
    rows.sort(key=lambda r: r["bar_ts"])
    seen, out = set(), []
    for r in rows:
        if r["bar_ts"] not in seen:
            seen.add(r["bar_ts"]); out.append(r)
    return out


def _wins(path_ret, tp, sl):
    """(long_wins, short_wins) by first touch along closes; tie/neither -> False."""
    lw = sw = False
    for r in path_ret:
        if r >= tp or r <= -sl:
            lw = r >= tp
            break
    for r in path_ret:
        if -r >= tp or -r <= -sl:
            sw = -r >= tp
            break
    return lw, sw


def label_rows(close, bars, scalp, swing, cost):
    n = len(close)
    y = np.full(n, np.nan)
    hs, hw = scalp[0], swing[0]
    for t in range(n - hw):
        if abs((bars[t + hw] - bars[t]) - hw * 900) > 60 * hw / 4 + 60:
            continue  # gap in recorded bars: no real forward path
        pr = close[t + 1:t + hw + 1] / close[t] - 1.0
        lw, sw = _wins(pr, swing[1] + cost, swing[2])
        if lw != sw:
            y[t] = 3 if lw else 4; continue
        lw, sw = _wins(pr[:hs], scalp[1] + cost, scalp[2])
        y[t] = (1 if lw else 2) if lw != sw else 0
    return y


def build(rows):
    names = sorted({k for r in rows for k in list(r["candle"]) + list(r["parts"])})
    ix = {n: i for i, n in enumerate(names)}
    X = np.full((len(rows), len(names)), np.nan)
    for j, r in enumerate(rows):
        for d in (r["candle"], r["parts"]):
            for k, v in d.items():
                X[j, ix[k]] = v
    return names, X


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--log", default="logs/part_snapshots.jsonl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--cost-bps", type=float, default=8.0)
    ap.add_argument("--scalp", default="4,0.005,0.003")
    ap.add_argument("--swing", default="96,0.02,0.008")
    a = ap.parse_args(argv)
    sym = a.symbol.upper()
    scalp = tuple(float(x) for x in a.scalp.split(",")); scalp = (int(scalp[0]),) + scalp[1:]
    swing = tuple(float(x) for x in a.swing.split(",")); swing = (int(swing[0]),) + swing[1:]
    rows = load_rows(a.log, sym)
    if len(rows) < MIN_ROWS:
        print(f"insufficient data: {len(rows)} snapshots for {sym} (need >= {MIN_ROWS}); no model written")
        return 2
    names, X = build(rows)
    close = np.array([r["close"] for r in rows]); bars = np.array([r["bar_ts"] for r in rows])
    y = label_rows(close, bars, scalp, swing, a.cost_bps / 1e4)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok].astype(int)
    n = len(y)
    purge = swing[0]
    i_tr, i_va = int(n * 0.6), int(n * 0.8)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xva, yva = X[i_tr + purge:i_va], y[i_tr + purge:i_va]
    Xte, yte = X[i_va + purge:], y[i_va + purge:]
    if n < MIN_ROWS or len(yte) < MIN_TEST or len(yva) < 50:
        print(f"insufficient labelled out-of-sample rows (n={n}, test={len(yte)}); no model written"); return 2
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss
    prm = dict(objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss", max_depth=3,
               eta=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=20, tree_method="hist",
               device=a.device)
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=names)
    bst = xgb.train(prm, dtr, 500, evals=[(dva, "val")], early_stopping_rounds=30, verbose_eval=False)
    pte = bst.predict(xgb.DMatrix(Xte, feature_names=names))
    prior = np.bincount(ytr, minlength=len(CLASSES)) / len(ytr)
    base_ll = float(log_loss(yte, np.tile(prior, (len(yte), 1)), labels=range(len(CLASSES))))
    ll = float(log_loss(yte, pte, labels=range(len(CLASSES))))
    aucs = [roc_auc_score(yte == k, pte[:, k]) for k in range(len(CLASSES)) if 0 < (yte == k).sum() < len(yte)]
    auc = float(np.mean(aucs)) if aucs else 0.5
    validated = bool(auc >= MIN_AUC and len(yte) >= MIN_TEST and ll < base_ll)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(out / f"xgb_mode_{sym}.json"))
    meta = dict(symbol=sym, version=int(time.time()), classes=CLASSES, feature_names=names,
                data_kind="recorded_live_snapshots", n_rows=int(n), scalp=scalp, swing=swing,
                cost_bps=a.cost_bps, veto_threshold=0.6, device=a.device,
                class_share_test={c: round(float((yte == i).mean()), 4) for i, c in enumerate(CLASSES)},
                oos=dict(auc=auc, n_test=int(len(yte)), logloss=ll, baseline_logloss=base_ll), validated=validated)
    (out / f"xgb_mode_{sym}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{sym}: rows={n} features={len(names)} test_auc={auc:.3f} logloss={ll:.3f} vs baseline={base_ll:.3f} validated={validated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
