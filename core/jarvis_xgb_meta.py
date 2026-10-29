"""XGBoost meta-gate (advisory, veto-only, fail-neutral).

What this module is
-------------------
A per-symbol, 15m-candle win-probability estimate for an entry that the
deterministic Jarvis decision path has *already* produced. It can never create
a trade, change a direction, or size a position. At most it can veto one
(``JARVIS_XGB_MODE=veto``) and only when ALL of these hold:

* a model file AND a ``.meta.json`` exist for exactly this symbol and 15m;
* the meta file says ``validated: true`` (out-of-sample evidence, written only
  by ``train_xgb_meta.py``) and the feature list matches this code;
* the candle frame is fresh enough and long enough to build features.

Anything else -> status ``UNAVAILABLE`` and the neutral probability 0.5 with
``can_veto=False``. No random/untrained model is ever consulted, there is no
cross-symbol fallback and nothing is trained at runtime.

Modes (``JARVIS_XGB_MODE``): ``off`` | ``shadow`` (default: log only) | ``veto``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("JarvisXGBMeta")

TIMEFRAME = "15m"
NEUTRAL_PROBABILITY = 0.5
MIN_BARS = 80
MIN_VALID_AUC = 0.55
MIN_VALID_TEST_SAMPLES = 300
MODEL_DIR = Path(os.getenv("JARVIS_XGB_MODEL_DIR", str(Path(__file__).resolve().parent / "models")))
SHADOW_LOG = Path(os.getenv("JARVIS_XGB_SHADOW_LOG", str(Path(__file__).resolve().parent / "logs" / "xgb_shadow.jsonl")))

FEATURE_NAMES = [
    "ret_1", "ret_4", "ret_16", "atr_pct", "range_pct", "body_ratio",
    "upper_wick_ratio", "lower_wick_ratio", "volume_z", "realized_vol_16",
    "dist_high_32", "dist_low_32", "slope_16", "direction",
]


def clean_symbol(symbol: Any) -> str:
    return str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")


def model_paths(symbol: str, model_dir: Optional[Path] = None) -> tuple[Path, Path]:
    base = Path(model_dir) if model_dir else MODEL_DIR
    sym = clean_symbol(symbol)
    return base / f"xgb_meta_{sym}_{TIMEFRAME}.json", base / f"xgb_meta_{sym}_{TIMEFRAME}.meta.json"


def _arrays(df) -> Optional[Dict[str, np.ndarray]]:
    try:
        cols = {c.lower(): c for c in df.columns}
        out = {k: df[cols[k]].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume")}
    except Exception:
        return None
    return out


def compute_atr_pct(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Average true range / close, aligned to bar index (NaN while warming up)."""
    prev_c = np.concatenate(([np.nan], c[:-1]))
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    out = np.full(len(c), np.nan)
    if len(c) > period:
        csum = np.cumsum(np.nan_to_num(tr))
        out[period:] = (csum[period:] - csum[:-period]) / period / c[period:]
    return out


def feature_matrix(df, direction_sign: float) -> Optional[np.ndarray]:
    """Feature rows for every bar of ``df`` (CLOSED candles only). Row i uses
    data up to and including bar i. Shared by trainer and runtime (no skew)."""
    a = _arrays(df)
    if a is None or len(a["close"]) < MIN_BARS:
        return None
    o, h, l, c, v = a["open"], a["high"], a["low"], a["close"], a["volume"]
    if not (np.all(np.isfinite(c)) and np.all(c > 0) and np.all(np.isfinite(h)) and np.all(np.isfinite(l))):
        return None
    n = len(c)
    X = np.full((n, len(FEATURE_NAMES)), np.nan)
    logc = np.log(c)

    def lag_ret(k):
        r = np.full(n, np.nan)
        r[k:] = logc[k:] - logc[:-k]
        return r

    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    X[:, 0], X[:, 1], X[:, 2] = lag_ret(1), lag_ret(4), lag_ret(16)
    X[:, 3] = compute_atr_pct(h, l, c)
    X[:, 4] = (h - l) / c
    X[:, 5] = body / rng
    X[:, 6] = (h - np.maximum(o, c)) / rng
    X[:, 7] = (np.minimum(o, c) - l) / rng
    for i in range(48, n):
        w = v[i - 48:i]
        sd = w.std()
        X[i, 8] = (v[i] - w.mean()) / sd if sd > 0 else 0.0
    r1 = np.concatenate(([np.nan], np.diff(logc)))
    for i in range(16, n):
        X[i, 9] = np.nanstd(r1[i - 15:i + 1])
    for i in range(32, n):
        X[i, 10] = c[i] / np.max(h[i - 31:i + 1]) - 1.0
        X[i, 11] = c[i] / np.min(l[i - 31:i + 1]) - 1.0
    xs = np.arange(16, dtype=float)
    xs -= xs.mean()
    den = float((xs ** 2).sum())
    for i in range(15, n):
        X[i, 12] = float((xs * (logc[i - 15:i + 1] - logc[i - 15:i + 1].mean())).sum() / den)
    X[:, 13] = float(direction_sign)
    return X


def direction_sign(direction: Any) -> Optional[float]:
    d = str(direction or "").upper()
    if d in ("BUY", "CALL", "LONG"):
        return 1.0
    if d in ("SELL", "PUT", "SHORT"):
        return -1.0
    return None


def meta_is_valid(meta: Any, symbol: str) -> tuple[bool, str]:
    """Strict validation of the sidecar written by the offline trainer."""
    if not isinstance(meta, dict):
        return False, "meta missing"
    if meta.get("validated") is not True:
        return False, "model not validated out-of-sample"
    if meta.get("data_kind") != "real_historical_ohlcv":
        return False, "model not trained on real historical data"
    if clean_symbol(meta.get("symbol")) != clean_symbol(symbol):
        return False, "model symbol mismatch"
    if meta.get("timeframe") != TIMEFRAME:
        return False, "model timeframe mismatch"
    if list(meta.get("feature_names") or []) != FEATURE_NAMES:
        return False, "feature schema mismatch"
    oos = meta.get("oos") or {}
    try:
        if float(oos.get("auc")) < MIN_VALID_AUC or int(oos.get("n_test")) < MIN_VALID_TEST_SAMPLES:
            return False, "out-of-sample evidence too weak"
        thr = float(meta.get("threshold"))
    except (TypeError, ValueError):
        return False, "meta metrics malformed"
    if not (0.0 < thr < 1.0):
        return False, "threshold invalid"
    return True, "ok"


class XGBMetaGate:
    def __init__(self, model_dir: Optional[Path] = None, mode: Optional[str] = None,
                 shadow_log: Optional[Path] = None):
        self.model_dir = model_dir
        self._mode = mode
        self.shadow_log = Path(shadow_log) if shadow_log else SHADOW_LOG
        self._cache: Dict[str, tuple] = {}
        self.stats = {"checks": 0, "unavailable": 0, "vetoes": 0}

    @property
    def mode(self) -> str:
        m = (self._mode or os.getenv("JARVIS_XGB_MODE", "shadow")).strip().lower()
        return m if m in ("off", "shadow", "veto") else "shadow"

    def _load(self, symbol: str):
        mp, metap = model_paths(symbol, self.model_dir)
        key = clean_symbol(symbol)
        try:
            stamp = (mp.stat().st_mtime, metap.stat().st_mtime)
        except OSError:
            return None, None, "no model for symbol"
        cached = self._cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1], cached[2], cached[3]
        try:
            meta = json.loads(metap.read_text())
        except Exception:
            self._cache[key] = (stamp, None, None, "meta unreadable")
            return None, None, "meta unreadable"
        ok, why = meta_is_valid(meta, symbol)
        if not ok:
            self._cache[key] = (stamp, None, meta, why)
            return None, meta, why
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(mp))
        except Exception as exc:  # fail neutral
            self._cache[key] = (stamp, None, meta, f"load failed: {type(exc).__name__}")
            return None, meta, f"load failed: {type(exc).__name__}"
        self._cache[key] = (stamp, booster, meta, "ok")
        return booster, meta, "ok"

    def evaluate(self, df_15m, direction: Any, symbol: Any, *, last_closed_ts: Optional[float] = None) -> Dict[str, Any]:
        """Never raises. ``df_15m`` must be CLOSED 15m candles of ``symbol``."""
        sym = clean_symbol(symbol)
        mode = self.mode
        res: Dict[str, Any] = {"status": "UNAVAILABLE", "probability": NEUTRAL_PROBABILITY,
                               "can_veto": False, "veto": False, "mode": mode,
                               "symbol": sym, "reason": ""}
        if mode == "off":
            res["reason"] = "mode off"
            return res
        self.stats["checks"] += 1
        try:
            sign = direction_sign(direction)
            if sign is None or not sym:
                res["reason"] = "no actionable direction/symbol"
            else:
                booster, meta, why = self._load(sym)
                if booster is None:
                    res["reason"] = why
                else:
                    X = feature_matrix(df_15m, sign)
                    if X is None or not np.all(np.isfinite(X[-1])):
                        res["reason"] = "insufficient/invalid candles for features"
                    else:
                        import xgboost as xgb
                        p = float(booster.predict(xgb.DMatrix(X[-1:], feature_names=FEATURE_NAMES))[0])
                        if not math.isfinite(p):
                            res["reason"] = "non-finite prediction"
                        else:
                            thr = float(meta["threshold"])
                            res.update(status="OK", probability=round(p, 4), threshold=thr,
                                       can_veto=(mode == "veto"), reason="validated model",
                                       veto=(mode == "veto" and p < thr),
                                       model_version=meta.get("version"))
        except Exception as exc:
            res.update(status="UNAVAILABLE", probability=NEUTRAL_PROBABILITY, can_veto=False,
                       veto=False, reason=f"error: {type(exc).__name__}")
        if res["status"] == "UNAVAILABLE":
            self.stats["unavailable"] += 1
        if res.get("veto"):
            self.stats["vetoes"] += 1
        self._log_shadow(res, direction)
        return res

    def _log_shadow(self, res: Dict[str, Any], direction: Any) -> None:
        try:
            self.shadow_log.parent.mkdir(parents=True, exist_ok=True)
            row = {"ts": time.time(), "direction": str(direction), **{k: res.get(k) for k in (
                "symbol", "status", "probability", "threshold", "veto", "mode", "reason")}}
            with open(self.shadow_log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:
            pass


_GATE: Optional[XGBMetaGate] = None


def get_gate() -> XGBMetaGate:
    global _GATE
    if _GATE is None:
        _GATE = XGBMetaGate()
    return _GATE
