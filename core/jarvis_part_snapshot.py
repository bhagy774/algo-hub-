"""Part-output snapshot recorder + parts-aware XGBoost features (advisory only).

Every live analysis cycle, once per (symbol, closed 15m bar), this records what
Parts 1-12 / brains / options / MTF context said, together with the symbol's own
15m candle features, to ``logs/part_snapshots.jsonl``. Later, ``train_xgb_parts.py``
joins those rows with REAL forward price outcomes and trains per-symbol XGBoost
models offline. Nothing here can create, change, size or block a trade.

Rules: closed bars only, one symbol per row (no cross-symbol data), missing
values stay NaN (never fabricated), failures are swallowed (fail-neutral).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("JarvisPartSnapshot")

SCHEMA_VERSION = 1
LOG_PATH = Path(os.getenv("JARVIS_PART_SNAPSHOT_LOG",
                          str(Path(__file__).resolve().parent / "logs" / "part_snapshots.jsonl")))
MAX_DEPTH = 5
MAX_FEATURES = 600
SIGNAL_WORDS = {
    "CALL": 1.0, "BUY": 1.0, "LONG": 1.0, "BULL": 1.0, "BULLISH": 1.0, "UP": 1.0,
    "PUT": -1.0, "SELL": -1.0, "SHORT": -1.0, "BEAR": -1.0, "BEARISH": -1.0, "DOWN": -1.0,
    "NEUTRAL": 0.0, "NO_TRADE": 0.0, "NO-TRADE": 0.0, "HOLD": 0.0, "NONE": 0.0, "WAIT": 0.0,
}
SKIP_KEYS = {"thought", "thoughts", "reasoning", "reason", "message", "timestamp", "ts", "time"}


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        w = v.strip().upper()
        if w in SIGNAL_WORDS:
            return SIGNAL_WORDS[w]
    return None


def flatten_result(obj: Any, prefix: str = "", out: Optional[Dict[str, float]] = None, depth: int = 0) -> Dict[str, float]:
    """Flatten nested analysis output to {dotted.key: float}. Numbers, bools and
    signal words (CALL/PUT/NEUTRAL...) only; free text is ignored."""
    out = {} if out is None else out
    if len(out) >= MAX_FEATURES or depth > MAX_DEPTH:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks.lower() in SKIP_KEYS:
                continue
            flatten_result(v, f"{prefix}.{ks}" if prefix else ks, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if len(obj) <= 20:
            for i, v in enumerate(obj):
                flatten_result(v, f"{prefix}[{i}]", out, depth + 1)
    else:
        f = _num(obj)
        if f is not None and prefix:
            out[prefix] = f
    return out


def candle_features(frame_closed, tf: str = "15m") -> Dict[str, float]:
    """Candle features of ONE timeframe (same code as the gate: no train/live skew)."""
    try:
        import numpy as np
        from jarvis_xgb_meta import FEATURE_NAMES, feature_matrix
        X = feature_matrix(frame_closed, 0.0)
        if X is None:
            return {}
        row = X[-1]
        return {f"tf{tf}.{n}": float(v) for n, v in zip(FEATURE_NAMES, row)
                if n != "direction" and np.isfinite(v)}
    except Exception:
        return {}


def all_timeframe_features(snapshot) -> Dict[str, float]:
    """Candle features for EVERY timeframe frame of this coin's snapshot (closed bars only).
    Timeframes with too little history are simply absent (never fabricated)."""
    out: Dict[str, float] = {}
    frames = getattr(snapshot, "frames", None) or {}
    for tf, fr in frames.items():
        closed = getattr(fr, "closed", None)
        if closed is not None:
            out.update(candle_features(closed, str(tf)))
    return out


def build_feature_row(result: Any, snapshot: Any) -> Dict[str, float]:
    """Single feature dict used by BOTH the recorder and the live advisor."""
    feats = all_timeframe_features(snapshot)
    feats.update(flatten_result(
        {k: v for k, v in (result or {}).items() if k != "candle_snapshot"}))
    return feats


class PartSnapshotRecorder:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else LOG_PATH
        self._last: Dict[str, Any] = {}

    def record(self, symbol: Any, result: Any, snapshot: Any, direction: Any = None,
               confidence: Any = None) -> bool:
        """Append one row per (symbol, closed 15m bar). Never raises."""
        try:
            if os.getenv("JARVIS_PART_SNAPSHOT", "1") == "0" or not isinstance(result, dict):
                return False
            sym = str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
            frame = None if snapshot is None else getattr(snapshot, "frames", {}).get("15m")
            closed = None if frame is None else getattr(frame, "closed", None)
            if not sym or closed is None or len(closed) < 2:
                return False
            cols = {c.lower(): c for c in closed.columns}
            ts_col = cols.get("close_time") or cols.get("open_time") or cols.get("timestamp")
            idx = closed.index[-1]
            bar_ts = closed[ts_col].iloc[-1] if ts_col else idx
            bar_ts = float(bar_ts.timestamp()) if hasattr(bar_ts, "timestamp") else float(bar_ts)
            if self._last.get(sym) == bar_ts:
                return False
            self._last[sym] = bar_ts
            parts = flatten_result({k: v for k, v in result.items()
                                    if k not in ("candle_snapshot",)})
            row = {
                "v": SCHEMA_VERSION, "ts": time.time(), "symbol": sym, "bar_ts": bar_ts,
                "close": float(closed[cols["close"]].iloc[-1]),
                "raw_direction": str(direction), "confidence": _num(confidence),
                "candle": all_timeframe_features(snapshot), "parts": parts,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(row, allow_nan=False) + "\n")
            return True
        except Exception as exc:
            logger.debug("[PART-SNAP] skipped (%s)", type(exc).__name__)
            return False


_REC: Optional[PartSnapshotRecorder] = None


def get_recorder() -> PartSnapshotRecorder:
    global _REC
    if _REC is None:
        _REC = PartSnapshotRecorder()
    return _REC
