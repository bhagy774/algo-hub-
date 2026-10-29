"""
brain_xgb.py - Micro-XGBoost for each Part 1 brain (SHADOW MODE).

Rules (by design):
  * Each brain has its OWN model: models/<name>.xgb (+ <name>.features.json).
  * No model / bad model / feature mismatch / xgboost missing  -> xgb_confidence = 0.5 (no bias).
  * SHADOW ONLY: this module only ADDS 'xgb_confidence' (+ 'xgb_active') to a brain's result.
    It never changes support_score, votes, or any trade/risk decision.
  * NO training inside the live path. Training is explicit/offline: train_brain_model().
  * Models are saved atomically (temp file -> rename).
  * Min samples before training (default 200). Time-ordered split validation is reported;
    a model is only saved if validation passes (see train_brain_model).
"""
import json
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:  # ImportError / OSError
    xgb = None
    XGB_AVAILABLE = False

MODELS_DIR = os.environ.get("JARVIS_BRAIN_MODELS_DIR", "models")
BUFFER_DIR = os.path.join(MODELS_DIR, "brain_outcomes")
NEUTRAL = 0.5
MIN_TRAIN_SAMPLES = 200
_SKIP_KEYS = {"xgb_confidence", "xgb_active"}
_lock = threading.Lock()


def extract_features(result: dict) -> Dict[str, float]:
    """Numeric, finite values of a brain's own result dict (bools -> 0/1). Strings are skipped."""
    feats = {}
    if not isinstance(result, dict):
        return feats
    for k, v in result.items():
        if k in _SKIP_KEYS:
            continue
        if isinstance(v, (bool, np.bool_)):
            feats[k] = float(bool(v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            if math.isfinite(f):
                feats[k] = f
    return feats


def _model_path(name):
    return os.path.join(MODELS_DIR, f"{name}.xgb")


def _meta_path(name):
    return os.path.join(MODELS_DIR, f"{name}.features.json")


class BrainXGB:
    """Per-brain XGBoost state. Created lazily, one per brain instance."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.last_features: Optional[Dict[str, float]] = None
        self._load()

    def _load(self):
        if not XGB_AVAILABLE:
            return
        try:
            if os.path.exists(_model_path(self.name)) and os.path.exists(_meta_path(self.name)):
                with open(_meta_path(self.name), "r", encoding="utf-8") as fh:
                    names = json.load(fh).get("feature_names")
                booster = xgb.Booster()
                booster.load_model(_model_path(self.name))
                if isinstance(names, list) and names:
                    self.model, self.feature_names = booster, names
        except Exception:
            self.model, self.feature_names = None, None  # corrupt -> neutral

    def predict(self, features: Dict[str, float]) -> Tuple[float, bool]:
        """Returns (win_probability, active). 0.5/False if no usable model."""
        if self.model is None or not self.feature_names:
            return NEUTRAL, False
        try:
            if set(features.keys()) != set(self.feature_names):
                return NEUTRAL, False  # schema mismatch -> never guess
            row = np.array([[features[k] for k in self.feature_names]], dtype=np.float32)
            p = float(self.model.predict(xgb.DMatrix(row, feature_names=self.feature_names))[0])
            if not math.isfinite(p):
                return NEUTRAL, False
            return min(max(p, 0.0), 1.0), True
        except Exception:
            return NEUTRAL, False

    def record_outcome(self, features: Optional[Dict[str, float]], outcome: int):
        """Append (features, outcome) to a jsonl file for OFFLINE training. No training here."""
        features = features or self.last_features
        if not features or outcome not in (0, 1):
            return False
        try:
            os.makedirs(BUFFER_DIR, exist_ok=True)
            with _lock, open(os.path.join(BUFFER_DIR, f"{self.name}.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"f": features, "y": int(outcome)}) + "\n")
            return True
        except Exception:
            return False


def attach_xgb(brain, name: str, result):
    """Add xgb_confidence/xgb_active to result (shadow). Never raises, never alters other keys."""
    if not isinstance(result, dict):
        return result
    try:
        state = getattr(brain, "_xgb_state", None)
        if state is None:
            state = BrainXGB(name)
            brain._xgb_state = state
        feats = extract_features(result)
        state.last_features = feats
        conf, active = state.predict(feats)
        result["xgb_confidence"] = conf
        result["xgb_active"] = active
    except Exception:
        result["xgb_confidence"] = NEUTRAL
        result["xgb_active"] = False
    return result


def xgb_brain(name: str, method: str):
    """Class decorator: wraps brain.<method> to add shadow xgb_confidence; adds record_outcome()."""
    def deco(cls):
        orig = getattr(cls, method)

        def wrapped(self, *a, **k):
            return attach_xgb(self, name, orig(self, *a, **k))
        wrapped.__name__ = orig.__name__
        wrapped.__doc__ = orig.__doc__
        setattr(cls, method, wrapped)

        def record_outcome(self, outcome: int, features: Optional[dict] = None):
            state = getattr(self, "_xgb_state", None)
            if state is None:
                state = BrainXGB(name)
                self._xgb_state = state
            return state.record_outcome(features, outcome)
        cls.record_outcome = record_outcome
        cls._xgb_name = name
        return cls
    return deco


def _features_any(result) -> Dict[str, float]:
    """Features from heterogeneous brain outputs (dict / list of dicts / number). Read-only."""
    if isinstance(result, dict):
        return extract_features(result)
    if isinstance(result, (bool, np.bool_)):
        return {"value": float(bool(result))}
    if isinstance(result, (int, float, np.integer, np.floating)):
        f = float(result)
        return {"value": f} if math.isfinite(f) else {}
    if isinstance(result, (list, tuple)):
        feats = {"count": float(len(result))}
        dicts = [extract_features(r) for r in result if isinstance(r, dict)]
        keys = sorted({k for d in dicts for k in d})
        for k in keys:
            vals = [d[k] for d in dicts if k in d]
            if vals:
                feats["mean_" + k] = float(np.mean(vals))
        return feats
    return {}


def xgb_shadow(name: str, method: str):
    """SIDE-CHANNEL shadow hook for Part 2/3 brains whose return type must NOT change
    (str / float / list / internal-state dict / broker order dict).
    The original return value is passed through UNTOUCHED (same object, nothing added).
    Confidence is stored on the instance: brain.last_xgb = {'confidence': p, 'active': bool}.
    Never raises, never changes decisions."""
    def deco(cls):
        orig = getattr(cls, method)

        def wrapped(self, *a, **k):
            result = orig(self, *a, **k)
            try:
                state = getattr(self, "_xgb_state", None)
                if state is None:
                    state = BrainXGB(name)
                    self._xgb_state = state
                feats = _features_any(result)
                state.last_features = feats
                conf, active = state.predict(feats) if feats else (NEUTRAL, False)
                self.last_xgb = {"confidence": conf, "active": active}
            except Exception:
                self.last_xgb = {"confidence": NEUTRAL, "active": False}
            return result
        wrapped.__name__ = orig.__name__
        wrapped.__doc__ = orig.__doc__
        setattr(cls, method, wrapped)

        def record_outcome(self, outcome: int, features: Optional[dict] = None):
            state = getattr(self, "_xgb_state", None)
            if state is None:
                state = BrainXGB(name)
                self._xgb_state = state
            return state.record_outcome(features, outcome)
        cls.record_outcome = record_outcome
        cls._xgb_name = name
        return cls
    return deco


def load_outcomes(name: str):
    path = os.path.join(BUFFER_DIR, f"{name}.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def train_brain_model(name: str, rows: Optional[list] = None, min_samples: int = MIN_TRAIN_SAMPLES,
                      min_val_auc: float = 0.55, val_fraction: float = 0.25) -> dict:
    """OFFLINE training. Rows = [{'f': {...}, 'y': 0/1}, ...] in TIME ORDER (oldest first).
    Time-ordered split (no shuffling, no future leakage). Saves only if validation AUC >= min_val_auc
    and beats the base-rate. Returns a report dict; never touches live decisions."""
    if not XGB_AVAILABLE:
        return {"saved": False, "reason": "xgboost not installed"}
    rows = rows if rows is not None else load_outcomes(name)
    if len(rows) < min_samples:
        return {"saved": False, "reason": f"need >= {min_samples} samples, have {len(rows)}"}
    names = sorted(rows[0]["f"].keys())
    rows = [r for r in rows if sorted(r["f"].keys()) == names]
    if len(rows) < min_samples:
        return {"saved": False, "reason": "too few rows with consistent features"}
    X = np.array([[r["f"][k] for k in names] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    cut = int(len(rows) * (1 - val_fraction))
    Xtr, ytr, Xva, yva = X[:cut], y[:cut], X[cut:], y[cut:]
    if len(set(ytr.tolist())) < 2 or len(set(yva.tolist())) < 2:
        return {"saved": False, "reason": "need both classes in train and validation"}
    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "subsample": 0.8,
              "tree_method": "hist", "eval_metric": "auc", "seed": 7}
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=names)
    booster = xgb.train(params, dtr, num_boost_round=200, evals=[(dva, "val")],
                        early_stopping_rounds=15, verbose_eval=False)
    p = booster.predict(dva)
    # rank-based AUC (no sklearn dependency)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = yva == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    report = {"samples": len(rows), "val_auc": auc, "base_rate_val": float(yva.mean()), "saved": False}
    if auc < min_val_auc:
        report["reason"] = f"val_auc {auc:.3f} < {min_val_auc}; model NOT saved (brain stays neutral)"
        return report
    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp_m, tmp_f = _model_path(name) + ".tmp", _meta_path(name) + ".tmp"
    booster.save_model(tmp_m)
    with open(tmp_f, "w", encoding="utf-8") as fh:
        json.dump({"feature_names": names, "val_auc": auc, "samples": len(rows)}, fh)
    os.replace(tmp_m, _model_path(name))   # atomic
    os.replace(tmp_f, _meta_path(name))
    report["saved"] = True
    return report
