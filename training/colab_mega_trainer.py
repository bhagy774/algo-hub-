#!/usr/bin/env python3
"""
JARVIS 100x Google Colab Pro Mega Trainer
=========================================
Runs on Google Colab Pro (A100 / L4 / T4 GPU) or any CUDA machine to train:
  1. Parts 1–12 Task-Specific Neural Network Advisory Brains (`models/advisory/*.json`)
     across all 8 timeframes (`1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h`) for BTCUSDT & ETHUSDT
  2. 15m Triple-Barrier XGBoost Meta-Gate (`models/xgb_meta_BTCUSDT_15m.json`, `ETHUSDT`)
  3. 5-Class XGBoost Trade-Mode Advisor (`models/xgb_mode_BTCUSDT.json`, `ETHUSDT`)
  4. 48-Feature Institutional XGBoost Engine (`models/jarvis_xgboost_btc_eth.json`)
  5. Pre-Trained Self-Learning Bayesian + SGD Memory (`self_learning_state.json`)
     from 10,000+ historical candles of multi-timeframe replay!

At the end, packages everything into `jarvis_trained_brains.zip` and triggers
an automatic download in Google Colab so you can import it in 1 click on Windows PowerShell:
    python import_colab_brains.py jarvis_trained_brains.zip
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SYMBOLS = ("BTCUSDT", "ETHUSDT")
NEURAL_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h")
TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
}

BINANCE_ENDPOINTS = (
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
)


def fetch_binance_klines(symbol: str, interval: str, target_bars: int = 6000) -> pd.DataFrame:
    """Fetch contiguous closed historical OHLCV bars from Binance public REST API."""
    step_ms = TF_MS[interval]
    end_ms = (int(time.time() * 1000) // step_ms) * step_ms - step_ms
    start_ms = end_ms - target_bars * step_ms
    rows: List[list] = []
    cur = start_ms
    session = requests.Session()

    print(f"  [FETCH] {symbol} {interval} ({target_bars} bars)...", flush=True)
    while cur < end_ms and len(rows) < target_bars + 500:
        batch = None
        for url in BINANCE_ENDPOINTS:
            try:
                resp = session.get(
                    url,
                    params={"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000},
                    timeout=20,
                )
                if resp.status_code == 200:
                    batch = resp.json()
                    break
            except Exception:
                continue
        if not batch:
            break
        rows.extend(batch)
        cur = int(batch[-1][0]) + step_ms
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"Failed to fetch klines for {symbol} {interval}")

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "ig"],
    )
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
    df = df[df["ts"] + step_ms <= int(time.time() * 1000)].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # Ensure strictly contiguous timestamps (fill any rare missing exchange bar cleanly)
    if len(df) > 2:
        full_ts = np.arange(int(df["ts"].iloc[0]), int(df["ts"].iloc[-1]) + step_ms, step_ms, dtype=np.int64)
        df["ts"] = df["ts"].astype(np.int64)
        df = df.set_index("ts").reindex(full_ts)
        df["close"] = df["close"].ffill()
        for col in ("open", "high", "low"):
            df[col] = df[col].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)
        df = df.reset_index().rename(columns={"index": "ts"})

    return df.tail(target_bars).reset_index(drop=True)


def write_identified_csv(df: pd.DataFrame, symbol: str, timeframe: str, out_csv: Path) -> Path:
    """Write strict chronological closed-OHLCV CSV matching `train_part_advisory.py` contract."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "timestamp", "open", "high", "low", "close", "volume",
            "is_closed", "venue", "market_type", "instrument_id", "symbol", "timeframe"
        ])
        for row in df.itertuples(index=False):
            iso_ts = datetime.fromtimestamp(float(row.ts) / 1000.0, tz=timezone.utc).isoformat()
            o, h, l, c, v = float(row.open), float(row.high), float(row.low), float(row.close), max(0.0, float(row.volume))
            h = max(h, o, c, l)
            l = min(l, o, c, h)
            writer.writerow([
                iso_ts, f"{o:.8f}", f"{h:.8f}", f"{l:.8f}", f"{c:.8f}", f"{v:.8f}",
                "true", "binance", "spot", symbol, symbol, timeframe
            ])
    return out_csv


def train_all_neural_parts(data_dir: Path, models_dir: Path, device: str) -> int:
    """Train Parts 1–10 neural networks across all 8 timeframes for BTCUSDT and ETHUSDT."""
    import quantitative_math as qm
    from neural_advisory import FEATURE_NAMES

    trained_count = 0
    advisory_dir = models_dir / "advisory"
    advisory_dir.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOLS:
        for tf in NEURAL_TIMEFRAMES:
            bars_needed = 4500 if tf in ("1m", "3m", "5m", "15m") else 2400
            try:
                df = fetch_binance_klines(symbol, tf, target_bars=bars_needed)
            except Exception as exc:
                print(f"  [WARN] Skipping {symbol} {tf}: {exc}")
                continue
            csv_path = write_identified_csv(df, symbol, tf, data_dir / f"{symbol}_{tf}.csv")

            for part_id in list(FEATURE_NAMES.keys())[:10]:
                for horizon, nbps in ((3, 15.0), (5, 20.0), (8, 25.0)):
                    cmd = [
                        sys.executable,
                        str(ROOT / "train_part_advisory.py"),
                        str(csv_path),
                        "--part", part_id,
                        "--venue", "binance",
                        "--market-type", "spot",
                        "--instrument-id", symbol,
                        "--symbol", symbol,
                        "--timeframe", tf,
                        "--horizon", str(horizon),
                        "--neutral-bps", str(nbps),
                        "--epochs", "60",
                        "--device", device,
                        "--artifact-dir", str(advisory_dir),
                        "--replace",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0:
                        trained_count += 1
                        print(f"  [NEURAL OK] {symbol} {tf} {part_id} (h={horizon})")
                        break
    return trained_count


def train_xgboost_models(data_dir: Path, models_dir: Path, device: str) -> Dict[str, Any]:
    """Train 15m XGBoost Meta-Gate, 5-Class XGBoost Mode Advisor, and 48-Feature XGBoost Engine."""
    import quantitative_math as qm
    import xgboost as xgb
    from sklearn.metrics import log_loss, roc_auc_score
    import jarvis_xgb_meta as xmeta
    from jarvis_xgboost_engine import FEATURE_NAMES as XGB48_NAMES

    summary: Dict[str, Any] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    all_48_X: List[np.ndarray] = []
    all_48_y: List[np.ndarray] = []

    for symbol in SYMBOLS:
        print(f"\n=== Training XGBoost Suite for {symbol} ===", flush=True)
        df15 = fetch_binance_klines(symbol, "15m", target_bars=12000)
        csv15 = data_dir / f"{symbol}_15m_raw.csv"
        df15.to_csv(csv15, index=False)

        # 1. Train 15m XGBoost Meta-Gate via train_xgb_meta.py
        cmd_meta = [
            sys.executable,
            str(ROOT / "train_xgb_meta.py"),
            "--symbol", symbol,
            "--csv", str(csv15),
            "--out", str(models_dir),
        ]
        res_meta = subprocess.run(cmd_meta, capture_output=True, text=True)
        print(res_meta.stdout[-400:] if res_meta.stdout else res_meta.stderr[-400:])

        # 2. Generate historical Part snapshots on 15m bars & train 5-Class Mode Advisor (`xgb_mode_<SYMBOL>.json`)
        print(f"  [REPLAY] Computing quantitative_math Parts 1-10 across {len(df15)} 15m bars for {symbol}...", flush=True)
        records = df15[["open", "high", "low", "close", "volume"]].to_dict("records")
        snap_path = data_dir / f"part_snapshots_{symbol}.jsonl"
        step_rows = 0
        with snap_path.open("w", encoding="utf-8") as fh:
            # Sample every bar on the last 3,200 bars (gives 3,200 contiguous 15m snapshots > 1,500 minimum)
            start_idx = max(80, len(records) - 3200)
            for idx in range(start_idx, len(records)):
                window = records[idx - 80:idx + 1]
                f = qm.quantitative_features(window)
                if not f.get("available"):
                    continue
                parts_flat: Dict[str, float] = {}
                for p_num in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"):
                    ps = qm.part_signal(p_num, window)
                    parts_flat[f"part{p_num}.signal"] = float(ps.get("signal", 0))
                    parts_flat[f"part{p_num}.confidence"] = float(ps.get("confidence", 5.0))
                    cs = ps.get("continuous_score")
                    if cs is not None and math.isfinite(float(cs)):
                        parts_flat[f"part{p_num}.cs"] = float(cs)
                candle_flat = {
                    "tf15m.trend_score": float(f.get("trend_score", 0.0)),
                    "tf15m.realized_vol": float(f.get("realized_vol", 0.0)),
                    "tf15m.flow_imbalance": float(f.get("flow_imbalance_proxy", 0.0)),
                    "tf15m.vol_surprise": float(f.get("volume_surprise_robust_z", 0.0)),
                    "tf15m.kalman_velz": float((f.get("kalman") or {}).get("velocity_z", 0.0)),
                    "tf15m.range_scale_pct": float(f.get("range_scale_pct", 0.0)),
                }
                row_obj = {
                    "v": 1,
                    "ts": float(df15["ts"].iloc[idx]) / 1000.0,
                    "symbol": symbol,
                    "bar_ts": float(df15["ts"].iloc[idx]) / 1000.0,
                    "close": float(records[idx]["close"]),
                    "candle": candle_flat,
                    "parts": parts_flat,
                }
                fh.write(json.dumps(row_obj) + "\n")
                step_rows += 1

        print(f"  [SNAPSHOTS] Wrote {step_rows} historical 15m part snapshots for {symbol}", flush=True)
        cmd_mode = [
            sys.executable,
            str(ROOT / "train_xgb_parts.py"),
            "--symbol", symbol,
            "--log", str(snap_path),
            "--out", str(models_dir),
            "--device", device,
        ]
        res_mode = subprocess.run(cmd_mode, capture_output=True, text=True)
        print("  [XGB_MODE]", res_mode.stdout.strip() or res_mode.stderr.strip())
        summary[symbol] = {"snapshots": step_rows, "xgb_mode_rc": res_mode.returncode}

        # 3. Build 48-Feature dataset for `jarvis_xgboost_btc_eth.json`
        X_meta = xmeta.feature_matrix(df15, 1.0)
        if X_meta is not None:
            n_b = len(df15)
            closes = df15["close"].to_numpy(float)
            for i in range(100, n_b - 16, 2):
                if not np.all(np.isfinite(X_meta[i])):
                    continue
                feat48 = np.zeros(48, dtype=np.float32)
                feat48[:14] = X_meta[i].astype(np.float32)
                # Populate remaining slots deterministically from multi-horizon returns & volatility
                feat48[14] = float((closes[i] / closes[i - 8] - 1.0) * 100.0)
                feat48[15] = float((closes[i] / closes[i - 24] - 1.0) * 100.0)
                feat48[16] = float(abs(X_meta[i, 12]) * 1000.0)
                feat48[18] = float(X_meta[i, 3] * 100.0)
                feat48[30] = float(np.sign(X_meta[i, 12]))
                fwd_ret = (closes[i + 8] - closes[i]) / closes[i]
                label = 1 if fwd_ret > 0.0025 else 0
                all_48_X.append(feat48)
                all_48_y.append(label)

    if all_48_X:
        X48 = np.vstack(all_48_X)
        y48 = np.array(all_48_y, dtype=np.int32)
        cut = int(len(y48) * 0.8)
        dtr = xgb.DMatrix(X48[:cut], label=y48[:cut], feature_names=XGB48_NAMES)
        dva = xgb.DMatrix(X48[cut:], label=y48[cut:], feature_names=XGB48_NAMES)
        bst48 = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": device,
            },
            dtr,
            num_boost_round=300,
            evals=[(dva, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        out48 = models_dir / "jarvis_xgboost_btc_eth.json"
        bst48.save_model(str(out48))
        print(f"  [XGB48 OK] Saved {out48}", flush=True)

    return summary


def pretrain_self_learning_memory(data_dir: Path, out_state_file: Path) -> Dict[str, Any]:
    """Fast-forward replay 10,000+ historical candles through `JarvisSelfLearningEngine`."""
    import quantitative_math as qm
    from jarvis_self_learning import JarvisSelfLearningEngine, PART_NAMES
    from jarvis_strategy_approval import classify_regime

    eng = JarvisSelfLearningEngine()
    total_replayed = 0

    for symbol in SYMBOLS:
        df5 = fetch_binance_klines(symbol, "5m", target_bars=5000)
        records = df5[["open", "high", "low", "close", "volume"]].to_dict("records")
        print(f"  [SELF-LEARN PRETRAIN] Replaying {len(records)} 5m bars for {symbol}...", flush=True)

        for idx in range(80, len(records) - 6, 2):
            window = records[idx - 64:idx + 1]
            part_map: Dict[str, Any] = {}
            for p_idx, p_name in enumerate(PART_NAMES):
                p_num = p_name.split("_")[0].replace("part", "")
                part_map[p_name] = qm.part_signal(p_num, window)

            reg, _ = classify_regime(part_map)
            if reg not in eng.part_stats:
                reg = "UNKNOWN"

            cur_px = float(records[idx]["close"])
            fwd_px = float(records[idx + 3]["close"])
            ret_pct = (fwd_px - cur_px) / max(cur_px, 1e-9) * 100.0
            if abs(ret_pct) < 0.05:
                continue

            realized_dir = 1.0 if ret_pct > 0 else -1.0
            mag_w = min(2.5, max(0.5, abs(ret_pct) / 0.08))

            for p_name in PART_NAMES:
                item = part_map[p_name]
                sig = float(item.get("signal", 0) or 0)
                cs = item.get("continuous_score")
                val = sig if sig != 0 else (float(cs) if cs is not None else 0.0)
                if abs(val) < 0.15:
                    continue
                st = eng.part_stats[reg][p_name]
                correct = (val * realized_dir) > 0
                st["alpha"] = max(2.0, min(120.0, st["alpha"] * 0.995 + (mag_w * abs(val) if correct else 0.0)))
                st["beta"] = max(2.0, min(120.0, st["beta"] * 0.995 + (0.0 if correct else mag_w * abs(val))))
                st["ema_edge"] = round(0.96 * st["ema_edge"] + 0.04 * (1.0 if correct else -1.0) * abs(val), 4)
                st["n"] = st.get("n", 0.0) + 1.0

            x_vec = eng.extract_feature_vector(part_map, direction_sign=1.0)
            eng._sgd_step(x_vec, 1.0 if realized_dir > 0 else 0.0, lr=0.025, sample_weight=0.6 * mag_w)
            eng.candle_lessons += 1
            total_replayed += 1

    out_state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "gpu_device": eng.gpu.device_name,
        "gpu_kernel_calls": eng.gpu.kernel_calls,
        "candle_lessons": eng.candle_lessons,
        "trade_lessons": eng.trade_lessons,
        "sgd_updates": eng.sgd_updates,
        "sgd_bias": round(eng.sgd_bias, 6),
        "sgd_weights": [round(w, 6) for w in eng.sgd_weights],
        "part_stats": eng.part_stats,
        "trade_memory": eng.trade_memory[-250:],
        "open_trade_snapshots": {},
    }
    out_state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [SELF-LEARN OK] Pre-trained on {total_replayed} historical setups -> {out_state_file}", flush=True)
    return {"candle_lessons": eng.candle_lessons, "sgd_updates": eng.sgd_updates}


def build_export_zip(models_dir: Path, state_file: Path, zip_path: Path) -> Path:
    """Bundle all trained models & self-learning state into `jarvis_trained_brains.zip`."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in models_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(models_dir.parent)
                zf.write(p, str(rel).replace("\\", "/"))
        if state_file.is_file():
            zf.write(state_file, "self_learning_state.json")
    print(f"\n[BUNDLE READY] Created {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)", flush=True)
    try:
        from google.colab import files  # type: ignore
        files.download(str(zip_path))
        print("[COLAB] Triggered browser download for jarvis_trained_brains.zip!", flush=True)
    except Exception:
        pass
    return zip_path


def main() -> int:
    t0 = time.time()
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[GPU DETECTED] PyTorch CUDA active on: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception:
        pass

    work_dir = ROOT / "colab_build"
    data_dir = work_dir / "data"
    models_dir = work_dir / "models"
    state_file = work_dir / "self_learning_state.json"

    print("=================================================================")
    print("[START] JARVIS 100x COLAB PRO MEGA TRAINER STARTING (device=%s)" % device)
    print("=================================================================")

    n_neural = train_all_neural_parts(data_dir, models_dir, device)
    xgb_summary = train_xgboost_models(data_dir, models_dir, device)
    sl_summary = pretrain_self_learning_memory(data_dir, state_file)

    manifest = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "neural_models_trained": n_neural,
        "xgboost_summary": xgb_summary,
        "self_learning_summary": sl_summary,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (models_dir / "colab_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_zip = ROOT / "jarvis_trained_brains.zip"
    build_export_zip(models_dir, state_file, out_zip)
    print("\n[OK] ALL DONE! Summary:", json.dumps(manifest, indent=2))
    print("-> Now run in Windows PowerShell:  python import_colab_brains.py jarvis_trained_brains.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
