"""Paper signal outcome tracker: records APPROVED paper signals and checks later 1m candles
for SL/TP1 first-touch, net of taker fee + slippage. Never places orders; never raises."""
from __future__ import annotations
import json, logging, os, threading, time
from typing import Any, Dict, List, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGDIR = os.path.join(_DIR, "logs", "files")
logger = logging.getLogger("jarvis_paper_tracker")
try:
    os.makedirs(_LOGDIR, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and "jarvis_paper_tracker" in getattr(h, "baseFilename", "") for h in logger.handlers):
        _fh = logging.FileHandler(os.path.join(_LOGDIR, "jarvis_paper_tracker.log"), encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_fh)
        logger.setLevel(logging.INFO)
except Exception:
    pass

_OUT = r"C:\jarvis_out"
_BASE = _OUT if os.path.isdir(_OUT) else _DIR
AUDIT = os.path.join(_BASE, "paper_trades_audit.jsonl")
STATE = os.path.join(_BASE, "paper_trades_state.json")
FEE_SIDE_PCT = float(os.getenv("JARVIS_PAPER_FEE_SIDE_PCT", "0.05"))   # taker fee per side
SLIP_SIDE_PCT = float(os.getenv("JARVIS_PAPER_SLIP_SIDE_PCT", "0.02"))  # assumed slippage per side
MAX_AGE_S = {"SCALP": 6 * 3600, "SWING": 7 * 86400}
_lock = threading.Lock()
_trades: List[Dict[str, Any]] = []
_loaded = False


def _base(sym: Any) -> str:
    s = "".join(c for c in str(sym or "").upper() if c.isalnum())
    for suf in ("USDT", "USD"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _load():
    global _loaded, _trades
    if _loaded:
        return
    _loaded = True
    try:
        with open(STATE, "r", encoding="utf-8") as fh:
            _trades = json.load(fh)
    except Exception:
        _trades = []


def _save():
    try:
        tmp = STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_trades, fh)
        os.replace(tmp, STATE)
    except Exception as exc:
        logger.warning("state save failed: %s", exc)


def _audit(ev: Dict[str, Any]):
    try:
        with open(AUDIT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
    except Exception as exc:
        logger.warning("audit write failed: %s", exc)


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except (TypeError, ValueError):
        return None


def record_signal(symbol, mode, direction, entry, sl, tp1, tp2=None, confluence=None, regime=None, xgb_probs=None, extra=None):
    """Register an approved paper signal. Duplicate open signals (same coin/mode/direction) are skipped."""
    try:
        d = str(direction or "").upper()
        e, s, t1, t2 = _f(entry), _f(sl), _f(tp1), _f(tp2)
        if d not in ("CALL", "PUT", "BUY", "SELL") or not (e and s and t1):
            return
        long_ = d in ("CALL", "BUY")
        if (long_ and not (s < e < t1)) or ((not long_) and not (t1 < e < s)):
            logger.warning("skip malformed levels %s %s e=%s sl=%s tp1=%s", symbol, d, e, s, t1)
            return
        with _lock:
            _load()
            b, m = _base(symbol), str(mode or "").upper()
            for t in _trades:
                if t["status"] == "OPEN" and t["base"] == b and t["mode"] == m and t["long"] == long_:
                    return
            tr = {"id": f"{b}-{int(time.time())}", "ts": time.time(), "base": b, "symbol": str(symbol), "mode": m,
                  "long": long_, "entry": e, "sl": s, "tp1": t1, "tp2": t2, "confluence": confluence,
                  "regime": regime, "xgb_probs": xgb_probs, "status": "OPEN", "mae_pct": 0.0, "last_ts": time.time()}
            if isinstance(extra, dict):
                tr.update(extra)
            _trades.append(tr)
            _save()
        _audit({"event": "OPEN", **tr})
        logger.info("OPEN %s %s %s entry=%s sl=%s tp1=%s", b, m, "LONG" if long_ else "SHORT", e, s, t1)
        try:
            from jarvis_self_learning import get_self_learning_engine
            get_self_learning_engine().record_trade_open(tr["id"], str(symbol), m, d, e, regime_info=regime)
        except Exception:
            pass
    except Exception as exc:
        logger.warning("record_signal failed: %s", exc)


def _close(tr, outcome, price, now):
    risk_pct = abs(tr["entry"] - tr["sl"]) / tr["entry"] * 100.0
    cost_pct = 2.0 * (FEE_SIDE_PCT + SLIP_SIDE_PCT)
    if outcome == "TP1":
        gross = abs(tr["tp1"] - tr["entry"]) / tr["entry"] * 100.0
    elif outcome == "SL":
        gross = -risk_pct
    else:
        move = (price - tr["entry"]) / tr["entry"] * 100.0
        gross = move if tr["long"] else -move
    tr.update(status="CLOSED", outcome=outcome, exit=price, closed_ts=now, net_r=round((gross - cost_pct) / risk_pct, 4))
    _audit({"event": "CLOSE", **tr})
    logger.info("CLOSE %s %s %s net_R=%.2f", tr["base"], tr["mode"], outcome, tr["net_r"])
    try:
        from jarvis_self_learning import get_self_learning_engine
        get_self_learning_engine().record_trade_close(tr["id"], tr.get("symbol", tr["base"]), tr["mode"], bool(tr["long"]), outcome, tr["net_r"], regime_info=tr.get("regime"))
    except Exception:
        pass


def update_candles(symbol, rows: List[Dict[str, float]]):
    """rows: iterable of dicts with ts (epoch s), high, low, close, oldest->newest."""
    try:
        b = _base(symbol)
        now = time.time()
        with _lock:
            _load()
            changed = False
            for tr in _trades:
                if tr["status"] != "OPEN" or tr["base"] != b:
                    continue
                for r in rows:
                    if r["ts"] <= tr["last_ts"] - 60:  # bar must open after the signal minute
                        continue
                    hi, lo = r["high"], r["low"]
                    adverse = (tr["entry"] - lo) if tr["long"] else (hi - tr["entry"])
                    tr["mae_pct"] = max(tr["mae_pct"], adverse / tr["entry"] * 100.0)
                    sl_hit = lo <= tr["sl"] if tr["long"] else hi >= tr["sl"]
                    tp_hit = hi >= tr["tp1"] if tr["long"] else lo <= tr["tp1"]
                    if sl_hit:  # SL first when both touch within one bar (conservative)
                        _close(tr, "SL", tr["sl"], now); changed = True; break
                    if tp_hit:
                        _close(tr, "TP1", tr["tp1"], now); changed = True; break
                if tr["status"] == "OPEN":
                    if rows:
                        tr["last_ts"] = max(tr["last_ts"], rows[-1]["ts"])
                    if now - tr["ts"] > MAX_AGE_S.get(tr["mode"], 86400) and rows:
                        _close(tr, "EXPIRED", rows[-1]["close"], now); changed = True
            if changed:
                _save()
    except Exception as exc:
        logger.warning("update failed: %s", exc)


def update_from_snapshot(symbol, snapshot):
    """Pull 1m closed candles from the live candle snapshot and evaluate open signals."""
    try:
        frame = snapshot.frames.get("1m")
        df = frame.closed.tail(30)
        cols = {str(c).lower(): c for c in df.columns}
        tcol = next((cols[k] for k in ("timestamp", "open_time", "time") if k in cols), None)
        rows = []
        for i, (idx, row) in enumerate(df.iterrows()):
            tv = row[tcol] if tcol is not None else idx
            try:
                ts = float(tv.timestamp()) if hasattr(tv, "timestamp") else float(tv)
                if ts > 1e11:
                    ts /= 1000.0
            except Exception:
                continue
            rows.append({"ts": ts, "high": float(row[cols["high"]]), "low": float(row[cols["low"]]), "close": float(row[cols["close"]])})
        update_candles(symbol, rows)
    except Exception as exc:
        logger.debug("snapshot update skipped: %s", exc)


def scorecard_line() -> str:
    with _lock:
        _load()
        closed = [t for t in _trades if t["status"] == "CLOSED"]
        opn = [t for t in _trades if t["status"] == "OPEN"]
    tp = sum(1 for t in closed if t.get("outcome") == "TP1")
    sl = sum(1 for t in closed if t.get("outcome") == "SL")
    dec = tp + sl
    wr = (100.0 * tp / dec) if dec else 0.0
    net = sum(t.get("net_r", 0.0) for t in closed)
    return (f"[PAPER_SCORECARD] Total={len(_trades)} | Open={len(opn)} | TP1_Hits={tp} | SL_Hits={sl} | "
            f"WinRate%={wr:.1f} | Net_R(after fees+slip)={net:+.2f} | closed={len(closed)} (small sample: not proof until 50+)")
