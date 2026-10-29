"""XGBoost trade-mode advisor: NO_TRADE / SCALP / SWING (+ direction), per symbol.

Input  : everything the system produced for ONE coin this cycle - Parts 1-12 /
         brains / options / MTF outputs (numbers + CALL/PUT/NEUTRAL signals) and
         candle features of EVERY timeframe frame of that coin's own snapshot.
Output : probabilities over 5 classes and an advised mode/direction.
Role   : ADVISORY. It never creates a trade, changes direction, size, SL or TP.
         Modes (JARVIS_XGB_ADVISOR): off | shadow (default, log only) | veto.
         In veto mode only a model whose sidecar says validated=true (offline,
         out-of-sample) may block an entry that jarvis_FIXED already produced:
         when it says NO_TRADE or the OPPOSITE direction with prob >= threshold.
No model / unvalidated / schema or symbol mismatch / any error -> UNAVAILABLE,
neutral, no effect. Nothing is trained at runtime.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("JarvisXGBMode")

CLASSES = ["NO_TRADE", "SCALP_LONG", "SCALP_SHORT", "SWING_LONG", "SWING_SHORT"]
MODEL_DIR = Path(os.getenv("JARVIS_XGB_MODEL_DIR", str(Path(__file__).resolve().parent / "models")))
ADVICE_LOG = Path(os.getenv("JARVIS_XGB_ADVICE_LOG", str(Path(__file__).resolve().parent / "logs" / "xgb_mode_advice.jsonl")))
MIN_AUC = 0.55
MIN_TEST = 300


def clean_symbol(s: Any) -> str:
    return str(s or "").upper().replace("/", "").replace("-", "").replace("_", "")


def model_paths(symbol: str, model_dir: Optional[Path] = None):
    b = Path(model_dir) if model_dir else MODEL_DIR
    s = clean_symbol(symbol)
    return b / f"xgb_mode_{s}.json", b / f"xgb_mode_{s}.meta.json"


def meta_is_valid(meta: Any, symbol: str):
    if not isinstance(meta, dict):
        return False, "meta missing"
    if meta.get("validated") is not True:
        return False, "model not validated out-of-sample"
    if meta.get("data_kind") != "recorded_live_snapshots":
        return False, "model not trained on recorded snapshots"
    if clean_symbol(meta.get("symbol")) != clean_symbol(symbol):
        return False, "model symbol mismatch"
    if list(meta.get("classes") or []) != CLASSES:
        return False, "class schema mismatch"
    if not meta.get("feature_names"):
        return False, "no feature schema"
    oos = meta.get("oos") or {}
    try:
        if float(oos.get("auc")) < MIN_AUC or int(oos.get("n_test")) < MIN_TEST:
            return False, "out-of-sample evidence too weak"
    except (TypeError, ValueError):
        return False, "meta metrics malformed"
    return True, "ok"


class XGBModeAdvisor:
    def __init__(self, model_dir: Optional[Path] = None, mode: Optional[str] = None,
                 log_path: Optional[Path] = None, device: Optional[str] = None):
        self.model_dir, self._mode = model_dir, mode
        self.log_path = Path(log_path) if log_path else ADVICE_LOG
        self.device = device or os.getenv("JARVIS_XGB_DEVICE", "cpu")
        self._cache: Dict[str, tuple] = {}

    @property
    def mode(self) -> str:
        m = (self._mode or os.getenv("JARVIS_XGB_ADVISOR", "shadow")).strip().lower()
        return m if m in ("off", "shadow", "veto") else "shadow"

    def _load(self, symbol: str):
        mp, metap = model_paths(symbol, self.model_dir)
        key = clean_symbol(symbol)
        try:
            stamp = (mp.stat().st_mtime, metap.stat().st_mtime)
        except OSError:
            return None, None, "no model for symbol"
        c = self._cache.get(key)
        if c and c[0] == stamp:
            return c[1], c[2], c[3]
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
            b = xgb.Booster()
            b.load_model(str(mp))
            try:
                b.set_param({"device": self.device})
            except Exception:
                pass
        except Exception as exc:
            self._cache[key] = (stamp, None, meta, f"load failed: {type(exc).__name__}")
            return None, meta, f"load failed: {type(exc).__name__}"
        self._cache[key] = (stamp, b, meta, "ok")
        return b, meta, "ok"

    def advise(self, symbol: Any, features: Dict[str, float], direction: Any = None) -> Dict[str, Any]:
        """Never raises. ``features`` = build_feature_row(result, snapshot) of THIS symbol."""
        sym = clean_symbol(symbol)
        mode = self.mode
        res: Dict[str, Any] = {"status": "UNAVAILABLE", "symbol": sym, "mode": mode,
                               "advice": "UNKNOWN", "probs": {}, "veto": False,
                               "can_veto": False, "reason": ""}
        if mode == "off":
            res["reason"] = "mode off"
            return res
        try:
            if not sym or not features:
                res["reason"] = "no symbol/features"
            else:
                booster, meta, why = self._load(sym)
                if booster is None:
                    res["reason"] = why
                else:
                    names = meta["feature_names"]
                    row = np.array([[features.get(n, np.nan) for n in names]], dtype=float)
                    present = int(np.isfinite(row).sum())
                    if present < max(5, int(0.3 * len(names))):
                        res["reason"] = f"too few known features ({present}/{len(names)})"
                    else:
                        import xgboost as xgb
                        p = booster.predict(xgb.DMatrix(row, feature_names=names))[0]
                        probs = {c: round(float(x), 4) for c, x in zip(CLASSES, p)}
                        top = max(probs, key=probs.get)
                        thr = float(meta.get("veto_threshold", 0.6))
                        veto = False
                        if mode == "veto":
                            d = str(direction or "").upper()
                            want = "LONG" if d in ("CALL", "BUY", "LONG") else "SHORT" if d in ("PUT", "SELL", "SHORT") else None
                            opp = "SHORT" if want == "LONG" else "LONG" if want == "SHORT" else None
                            p_opp = sum(v for k, v in probs.items() if opp and k.endswith(opp))
                            veto = bool(want and (probs["NO_TRADE"] >= thr or p_opp >= thr))
                        res.update(status="OK", probs=probs, advice=top, can_veto=(mode == "veto"),
                                   veto=veto, reason="validated model", known_features=present,
                                   model_version=meta.get("version"))
        except Exception as exc:
            res.update(status="UNAVAILABLE", advice="UNKNOWN", veto=False, can_veto=False,
                       reason=f"error: {type(exc).__name__}")
        self._log(res, direction)
        return res

    def _log(self, res, direction):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({"ts": time.time(), "direction": str(direction),
                                     **{k: res.get(k) for k in ("symbol", "status", "advice", "probs", "veto", "mode", "reason")}}) + "\n")
        except Exception:
            pass


_ADV: Optional[XGBModeAdvisor] = None


def get_advisor() -> XGBModeAdvisor:
    global _ADV
    if _ADV is None:
        _ADV = XGBModeAdvisor()
    return _ADV
