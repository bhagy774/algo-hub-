"""
jarvis_scenario_simulator.py — JARVIS Pre-Trade Scenario Simulator
("Timeline Divergence Modeler" — real engineering version)

Trade levata PEHLA future na multiple stress scenarios mathematically
simulate kare chhe. Koi live feed / lookahead nahi — fakt pasina
(historical) closed candles mathi volatility, wick-size, ane volume
estimates. Worst-case ma trade survive kartu nathi to entry VETO.

Design rules:
  - Fail-open: koi pan error -> 'pass' (system kyarey stuck na thay)
  - Fast: pure math, <0.5 sec, koi network call nahi
  - JARVIS_SCEN_SIM=0 -> feature off
  - JARVIS_SCEN_MIN_PASS (default 3) -> ochha ma ochha ketla scenarios
    pass thava joie (5 mathi)
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ScenarioConfig:
    direction: str
    entry_price: float
    sl: float
    tp: float
    df: Any
    fee_bps: float = 10.0
    slippage_bps: float = 5.0


def scen_enabled():
    return os.getenv("JARVIS_SCEN_SIM", "1") != "0"


def _min_pass():
    try:
        return max(1, min(5, int(os.getenv("JARVIS_SCEN_MIN_PASS", "3"))))
    except Exception:
        return 3


def _atr(df, n=14):
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    if len(closes) < n + 1:
        return None
    trs = []
    for i in range(len(closes) - n, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


def _scenario_normal_move(direction, entry, sl, tp, atr):
    """TP recent typical move (ATR) ni andar reachable chhe?"""
    tp_dist = abs(tp - entry)
    if tp_dist <= 0:
        return False, "TP distance zero/invalid"
    if atr and tp_dist > 3.0 * atr:
        return False, f"TP {tp_dist / atr:.1f}x ATR dur — unlikely hit in scalp window"
    return True, "TP within reachable range"


def _scenario_slippage_spike(*args, **kwargs):
    """Slippage 3x thay to pan net R:R positive rahe?"""
    if len(args) == 6:
        _, entry, sl, tp, fee_bps, slippage_bps = args
    elif len(args) == 5:
        if isinstance(args[0], str):
            _, entry, sl, tp, fee_bps = args
            slippage_bps = kwargs.get("slippage_bps", 5.0)
        else:
            entry, sl, tp, fee_bps, slippage_bps = args
    else:
        entry = kwargs.get("entry", kwargs.get("entry_price", 0))
        sl = kwargs.get("sl", 0)
        tp = kwargs.get("tp", 0)
        fee_bps = kwargs.get("fee_bps", 10.0)
        slippage_bps = kwargs.get("slippage_bps", 5.0)

    try:
        entry_f = float(entry or 0)
        sl_f = float(sl or 0)
        tp_f = float(tp or 0)
        fee_bps_f = float(fee_bps or 0)
        slippage_bps_f = float(slippage_bps or 0)
    except (ValueError, TypeError):
        return False, "Invalid numeric parameters"

    cost = entry_f * (fee_bps_f + 3.0 * slippage_bps_f) / 10000.0
    reward = abs(tp_f - entry_f) - cost
    risk = abs(entry_f - sl_f) + cost
    if risk <= 0:
        return False, "Risk distance zero/invalid"
    rr = reward / risk
    if rr < 0.5:
        return False, f"3x slippage ma R:R {rr:.2f} — cost profit khai jashe"
    return True, f"3x slippage ma pan R:R {rr:.2f}"


def _scenario_wick_hunt(direction, entry, sl, df, atr):
    """Recent candles na wicks SL sudhi pohchi shake? (stop-hunt check)"""
    if len(df) < 10:
        return True, "insufficient candles (skip)"
    recent = df.tail(10)
    highs = recent["high"].astype(float).values
    lows = recent["low"].astype(float).values
    opens = recent["open"].astype(float).values
    closes = recent["close"].astype(float).values
    upper_wicks = highs - [max(o, c) for o, c in zip(opens, closes)]
    lower_wicks = [min(o, c) for o, c in zip(opens, closes)] - lows
    if direction == "CALL":
        worst = float(max(lower_wicks))
        buffer = entry - sl
        if worst > 0 and buffer < 0.8 * worst:
            return False, f"SL buffer {buffer:.2f} < recent wick {worst:.2f} — stop-hunt risk"
    else:
        worst = float(max(upper_wicks))
        buffer = sl - entry
        if worst > 0 and buffer < 0.8 * worst:
            return False, f"SL buffer {buffer:.2f} < recent wick {worst:.2f} — stop-hunt risk"
    return True, "SL beyond recent wick zone"


def _scenario_volatility_collapse(df, atr):
    """Volatility collapse (dead market) ma TP kadi na poche."""
    if len(df) < 20 or not atr:
        return True, "insufficient data (skip)"
    last_range = float(df["high"].iloc[-1]) - float(df["low"].iloc[-1])
    if last_range < 0.25 * atr:
        return False, f"Last candle range {last_range:.2f} << ATR {atr:.2f} — dead market"
    return True, "Volatility alive"


def _scenario_volume_drop(df):
    """Volume collapse = move unreliable / thin liquidity."""
    if len(df) < 20 or "volume" not in df.columns:
        return True, "no volume data (skip)"
    vols = df["volume"].astype(float).values[-20:]
    avg = sum(vols[:-1]) / max(1, len(vols) - 1)
    last = vols[-1]
    if avg > 0 and last < 0.3 * avg:
        return False, f"Volume {last:.0f} << avg {avg:.0f} — thin liquidity"
    return True, "Volume healthy"


def run_scenarios(*args, **kwargs):
    """Badha scenarios run kari verdict aape.

    Supports:
      - run_scenarios(config: ScenarioConfig)
      - run_scenarios(direction, entry_price, sl, tp, df, fee_bps=..., slippage_bps=...)
      - run_scenarios(direction=..., entry_price=..., sl=..., tp=..., df=...)

    Returns dict:
      action: 'pass' | 'veto'
      passed: int, total: int
      scenarios: list of (name, ok, reason)
      reason: summary string
    """
    if len(args) == 1 and isinstance(args[0], ScenarioConfig):
        cfg = args[0]
        direction = cfg.direction
        entry = cfg.entry_price
        sl = cfg.sl
        tp = cfg.tp
        df = cfg.df
        fee_bps = cfg.fee_bps
        slippage_bps = cfg.slippage_bps
    elif len(args) >= 4:
        direction = args[0]
        entry = args[1]
        sl = args[2]
        tp = args[3]
        df = args[4] if len(args) > 4 else kwargs.get("df")
        fee_bps = args[5] if len(args) > 5 else kwargs.get("fee_bps", 10.0)
        slippage_bps = args[6] if len(args) > 6 else kwargs.get("slippage_bps", 5.0)
    else:
        direction = kwargs.get("direction")
        entry = kwargs.get("entry_price", kwargs.get("entry", 0))
        sl = kwargs.get("sl", 0)
        tp = kwargs.get("tp", 0)
        df = kwargs.get("df")
        fee_bps = kwargs.get("fee_bps", 10.0)
        slippage_bps = kwargs.get("slippage_bps", 5.0)

    direction = (str(direction) if direction is not None else "").upper()
    try:
        entry = float(entry or 0)
        sl = float(sl or 0)
        tp = float(tp or 0)
    except (ValueError, TypeError):
        entry, sl, tp = 0.0, 0.0, 0.0

    if direction not in ("CALL", "PUT") or entry <= 0 or sl <= 0 or tp <= 0 or df is None or len(df) == 0:
        return {
            "action": "pass",
            "passed": 0,
            "total": 0,
            "scenarios": [],
            "reason": "insufficient inputs (fail-open pass)",
        }

    atr = _atr(df)
    scenarios = [
        ("normal_move", *_scenario_normal_move(direction, entry, sl, tp, atr)),
        ("slippage_spike", *_scenario_slippage_spike(entry, sl, tp, fee_bps, slippage_bps)),
        ("wick_hunt", *_scenario_wick_hunt(direction, entry, sl, df, atr)),
        ("volatility_collapse", *_scenario_volatility_collapse(df, atr)),
        ("volume_drop", *_scenario_volume_drop(df)),
    ]
    passed = sum(1 for _, ok, _ in scenarios if ok)
    failed_reasons = [f"{n}: {r}" for n, ok, r in scenarios if not ok]
    action = "pass" if passed >= _min_pass() else "veto"
    return {
        "action": action,
        "passed": passed,
        "total": len(scenarios),
        "scenarios": scenarios,
        "reason": "; ".join(failed_reasons) if failed_reasons else "all scenarios pass",
    }
