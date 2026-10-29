"""Free public Binance Futures positioning feed (funding, OI delta, top-trader L/S, taker ratio,
L2 imbalance) with 60s TTL cache. Any unreachable endpoint -> that key is None (never fabricated).
Advisory/diagnostic only: does not authorize trades."""
from __future__ import annotations
import json, logging, os, statistics, threading, time, urllib.request
from typing import Any, Dict, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("jarvis_institutional_feed")
try:
    _ld = os.path.join(_DIR, "logs", "files")
    os.makedirs(_ld, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and "jarvis_institutional_feed" in getattr(h, "baseFilename", "") for h in logger.handlers):
        _fh = logging.FileHandler(os.path.join(_ld, "jarvis_institutional_feed.log"), encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_fh)
        logger.setLevel(logging.INFO)
except Exception:
    pass

TTL = 60.0
BASE = "https://fapi.binance.com"
_cache: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _get(path: str, timeout: float = 3.0):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "jarvis-feed/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _try(fn):
    try:
        return fn()
    except Exception as exc:
        logger.info("endpoint failed: %s: %s", type(exc).__name__, str(exc)[:100])
        return None


def _asset(sym: Any) -> str:
    s = "".join(c for c in str(sym or "").upper() if c.isalnum())
    for suf in ("USDT", "USD"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def get_context(symbol: Any) -> Dict[str, Any]:
    a = _asset(symbol)
    if a not in ("BTC", "ETH"):
        return {}
    now = time.time()
    with _lock:
        c = _cache.get(a)
        if c and now - c["_ts"] < TTL:
            return c
    pair = a + "USDT"
    ctx: Dict[str, Any] = {"asset": a, "funding_rate": None, "funding_z": None, "oi_delta_15m_pct": None,
                           "oi_delta_1h_pct": None, "top_long_account": None, "taker_buy_sell": None,
                           "book_imbalance": None}
    fr = _try(lambda: _get(f"/fapi/v1/fundingRate?symbol={pair}&limit=30"))
    if fr:
        rates = [float(x["fundingRate"]) for x in fr]
        ctx["funding_rate"] = rates[-1]
        if len(rates) >= 10 and statistics.pstdev(rates) > 1e-12:
            ctx["funding_z"] = round((rates[-1] - statistics.mean(rates)) / statistics.pstdev(rates), 3)
    oi = _try(lambda: _get(f"/futures/data/openInterestHist?symbol={pair}&period=15m&limit=5"))
    if oi and len(oi) >= 2:
        v = [float(x["sumOpenInterest"]) for x in oi]
        ctx["oi_delta_15m_pct"] = round((v[-1] - v[-2]) / v[-2] * 100.0, 4) if v[-2] else None
        ctx["oi_delta_1h_pct"] = round((v[-1] - v[0]) / v[0] * 100.0, 4) if v[0] and len(v) >= 5 else None
    ls = _try(lambda: _get(f"/futures/data/topLongShortAccountRatio?symbol={pair}&period=15m&limit=1"))
    if ls:
        ctx["top_long_account"] = float(ls[-1]["longAccount"])
    tk = _try(lambda: _get(f"/futures/data/takerlongshortRatio?symbol={pair}&period=15m&limit=1"))
    if tk:
        ctx["taker_buy_sell"] = float(tk[-1]["buySellRatio"])
    bk = _try(lambda: _get(f"/fapi/v1/depth?symbol={pair}&limit=50"))
    if bk and bk.get("bids") and bk.get("asks"):
        mid = (float(bk["bids"][0][0]) + float(bk["asks"][0][0])) / 2.0
        lo, hi = mid * 0.9975, mid * 1.0025
        bq = sum(float(q) for p, q in bk["bids"] if float(p) >= lo)
        aq = sum(float(q) for p, q in bk["asks"] if float(p) <= hi)
        if bq + aq > 0:
            ctx["book_imbalance"] = round((bq - aq) / (bq + aq), 4)
    ctx["_ts"] = now
    with _lock:
        _cache[a] = ctx
    return ctx


def _fmt(v, p=4):
    return "None" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def context_line(symbol: Any, ctx: Dict[str, Any], regime: Optional[Dict[str, Any]] = None, xgb_probs: Any = None,
                 options: Optional[Dict[str, Any]] = None) -> str:
    ctx = ctx or {}
    reg = regime or {}
    opt = options or {}
    return (f"{_asset(symbol)} | Regime={reg.get('counts')} | Hurst~{_fmt(reg.get('hurst_med'), 3)} | "
            f"OI_d15m%={_fmt(ctx.get('oi_delta_15m_pct'))} OI_d1h%={_fmt(ctx.get('oi_delta_1h_pct'))} | "
            f"Funding={_fmt(ctx.get('funding_rate'), 6)} (z={_fmt(ctx.get('funding_z'), 2)}) | "
            f"TopLongAcct={_fmt(ctx.get('top_long_account'))} TakerB/S={_fmt(ctx.get('taker_buy_sell'))} | "
            f"BookImb={_fmt(ctx.get('book_imbalance'))} | Opt_PCR={_fmt(opt.get('pcr'))} MaxPain={_fmt(opt.get('max_pain'))} | XGB={xgb_probs}")


def deep_get(d, key, depth=4):
    if not isinstance(d, dict) or depth < 0:
        return None
    if d.get(key) is not None:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            r = deep_get(v, key, depth - 1)
            if r is not None:
                return r
    return None


def deep_get(d, key, depth=4):
    if not isinstance(d, dict) or depth < 0:
        return None
    if d.get(key) is not None:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            r = deep_get(v, key, depth - 1)
            if r is not None:
                return r
    return None
