"""Compact, human-readable "what is Jarvis thinking right now" terminal panel.

Display only: it reads values already produced by the pipeline and never
changes a decision, confidence, gate, or order.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from terminal_live_diagnostics import (
    PART_KEYS, TIMEFRAMES, classify_part_output, _text, _present,
)

PART_NAMES = {
    "part1_breakout": "Breakout", "part2_zone": "S/R Zone", "part3_psychology": "Psychology",
    "part4_volume": "Volume", "part5_ml": "ML Model", "part6_trend": "Trend",
    "part7_volatility": "Volatility", "part8_structure": "Structure",
    "part9_orderflow": "Orderflow", "part10_candlestats": "Candle Stats",
}
GROUPS = (
    ("Scalp   (1s-5m)", ("1s", "1m", "3m", "5m")),
    ("Intraday(15m-2h)", ("15m", "30m", "1h", "2h")),
    ("Swing   (4h-1d)", ("4h", "6h", "8h", "12h", "1d")),
    ("Macro   (3d-1M)", ("3d", "1w", "1M")),
)

G, R, Y, C, W, DG, BD, RST = ("\033[92m", "\033[91m", "\033[93m", "\033[96m",
                              "\033[97m", "\033[90m", "\033[1m", "\033[0m")


def _side_word(buy: int, sell: int) -> str:
    if buy == 0 and sell == 0:
        return f"{DG}NO VIEW{RST}"
    if buy > sell * 1.5:
        return f"{G}LONG{RST}"
    if sell > buy * 1.5:
        return f"{R}SHORT{RST}"
    return f"{Y}MIXED{RST}"


def _bar(buy: int, sell: int, neutral: int, width: int = 16) -> str:
    total = buy + sell + neutral
    if total <= 0:
        return DG + "·" * width + RST
    b = round(width * buy / total)
    s = round(width * sell / total)
    n = max(0, width - b - s)
    return f"{G}{'█' * b}{DG}{'░' * n}{R}{'█' * s}{RST}"


def _count(per_tf: Mapping[str, Any], parts: tuple[str, ...], tfs: tuple[str, ...]):
    buy = sell = neutral = bad = 0
    for tf in tfs:
        row = per_tf.get(tf) if isinstance(per_tf.get(tf), Mapping) else {}
        for part in parts:
            item = row.get(part)
            if not isinstance(item, Mapping):
                bad += 1
                continue
            classified = classify_part_output(item)
            status = str(classified.get("status") or "").upper()
            signal = classified.get("signal")
            if signal == "BUY" and status in ("OK", "BLOCKED", ""):
                buy += 1
            elif signal == "SELL" and status in ("OK", "BLOCKED", ""):
                sell += 1
            elif status == "NEUTRAL" or signal == "NEUTRAL":
                neutral += 1
            else:
                bad += 1
    return buy, sell, neutral, bad


def format_live_thinking(*, symbol: Any, price: Any = None, parts_by_timeframe: Any = None,
                         fused_parts: Any = None, decision: Any = None, candidate: Any = None,
                         part7_gate: Any = None, entry_trigger: Any = None,
                         cycle_status: Any = None, timeframes: tuple[str, ...] = TIMEFRAMES) -> str:
    per_tf = parts_by_timeframe if isinstance(parts_by_timeframe, Mapping) else {}
    fused = fused_parts if isinstance(fused_parts, Mapping) else {}
    decision = decision if isinstance(decision, Mapping) else {}
    candidate = candidate if isinstance(candidate, Mapping) else {}
    cycle = cycle_status if isinstance(cycle_status, Mapping) else {}
    tfs = tuple(tf for tf in timeframes)
    now = datetime.now().strftime("%H:%M:%S")
    try:
        price_text = f"${float(price):,.2f}"
    except (TypeError, ValueError):
        price_text = "—"

    line = f"{C}{'━' * 84}{RST}"
    out = [line, f"{BD}{W}🧠 JARVIS LIVE THINKING{RST}  {DG}[{now}]{RST}  {BD}{_text(symbol, 20) or 'UNKNOWN'}{RST} @ {W}{price_text}{RST}", line]

    cycle_label = str(cycle.get("status") or "").upper()
    if not per_tf:
        reason = _text(cycle.get("reason"), 140) or "analysis did not run this cycle"
        out.append(f"  {Y}⏸  Analysis not run:{RST} {reason}")
    else:
        all_parts = tuple(PART_KEYS)
        tb, ts, tn, tbad = _count(per_tf, all_parts, tfs)
        out.append(f"  {BD}Overall vote (10 parts × {len(tfs)} TF):{RST} "
                   f"{G}BUY {tb}{RST} | {R}SELL {ts}{RST} | {DG}NEUTRAL {tn}{RST}"
                   + (f" | {Y}NO-DATA {tbad}{RST}" if tbad else "")
                   + f"  → leaning {_side_word(tb, ts)}  {_bar(tb, ts, tn, 24)}")
        out.append(f"  {BD}By horizon:{RST}")
        for label, group in GROUPS:
            group_tfs = tuple(tf for tf in group if tf in tfs)
            if not group_tfs:
                continue
            b, s, n, _ = _count(per_tf, all_parts, group_tfs)
            out.append(f"    {label}  {_bar(b, s, n)}  {G}{b:>3}↑{RST} {R}{s:>3}↓{RST} {DG}{n:>3}·{RST}  {_side_word(b, s)}")
        out.append(f"  {BD}By part (all timeframes):{RST}")
        row_items = []
        for part in all_parts:
            b, s, n, _ = _count(per_tf, (part,), tfs)
            row_items.append(f"{PART_NAMES.get(part, part):<12} {_side_word(b, s):<14} {DG}({b}↑/{s}↓/{n}·){RST}")
        for i in range(0, len(row_items), 2):
            out.append("    " + "   ".join(row_items[i:i + 2]))

    for name, label in (("part11_fusion", "Fusion (P11)"), ("part12_confidence", "Confidence (P12)")):
        item = fused.get(name)
        if isinstance(item, Mapping):
            o = classify_part_output(item)
            conf = item.get("confidence", item.get("confidence_score"))
            sig = o.get("signal") or "—"
            out.append(f"  {BD}{label}:{RST} signal={sig} conf={_text(conf, 12) or '?'}"
                       + (f" {DG}— {_text(o.get('reason'), 90)}{RST}" if o.get("reason") else ""))

    direction = str(decision.get("execution_direction") or decision.get("direction") or "NO_TRADE").upper()
    allowed = decision.get("execution_allowed") is True
    conf = decision.get("confidence")
    side = {"CALL": f"{G}{BD}LONG / BUY{RST}", "BUY": f"{G}{BD}LONG / BUY{RST}",
            "PUT": f"{R}{BD}SHORT / SELL{RST}", "SELL": f"{R}{BD}SHORT / SELL{RST}"}.get(direction, f"{Y}{BD}WAIT (no trade){RST}")
    out.append(line)
    out.append(f"  {BD}🎯 DECISION:{RST} {side}   conf={_text(conf, 10) or '?'}   "
               + (f"{G}ENTRY ALLOWED{RST}" if allowed else f"{R}ENTRY BLOCKED{RST}"))

    sig = candidate.get("trade_signal") if isinstance(candidate.get("trade_signal"), Mapping) else {}
    plan = []
    for label, keys in (("Entry", ("entry", "entry_price")), ("SL", ("sl", "stop_loss")),
                        ("TP1", ("tp1", "take_profit_1")), ("TP2", ("tp2", "take_profit_2"))):
        val = next((decision.get(k) for k in keys if _present(decision.get(k))), None)
        if val is None:
            val = next((sig.get(k) for k in keys if _present(sig.get(k))), None)
        if val is not None:
            try:
                plan.append(f"{label} ${float(val):,.2f}")
            except (TypeError, ValueError):
                plan.append(f"{label} {_text(val, 20)}")
    if plan and direction in ("CALL", "PUT", "BUY", "SELL"):
        out.append(f"  {BD}Plan:{RST} " + " | ".join(plan))

    reasons = decision.get("reasons")
    if isinstance(reasons, (list, tuple)):
        reasons = [_text(r, 110) for r in reasons if _present(r)]
    elif _present(reasons):
        reasons = [_text(reasons, 110)]
    else:
        reasons = []
    if not reasons and _present(candidate.get("no_trade_reason")):
        reasons = [_text(candidate.get("no_trade_reason"), 110)]
    if reasons:
        out.append(f"  {BD}Why:{RST}")
        for r in reasons[:4]:
            out.append(f"    • {r}")

    if isinstance(part7_gate, Mapping) and part7_gate.get("entry_blocked"):
        out.append(f"  {Y}⚠ Risk gate (P7):{RST} {_text(part7_gate.get('reason'), 110) or 'blocked'}")
    if isinstance(entry_trigger, Mapping):
        ok = entry_trigger.get("confirmed")
        out.append(f"  {BD}1m trigger:{RST} " + (f"{G}confirmed{RST}" if ok else f"{Y}waiting{RST}")
                   + f" {DG}{_text(entry_trigger.get('note'), 90)}{RST}")
    if cycle_label and cycle_label not in ("OK", "COMPLETE", "COMPLETED") and per_tf:
        out.append(f"  {DG}cycle: {cycle_label} — {_text(cycle.get('reason'), 100)}{RST}")
    out.append(line)
    return "\n".join(out)
