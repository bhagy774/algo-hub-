"""Single terminal dashboard for the live engine.

The renderer accepts a plain snapshot so it is offline-testable and keeps signal,
reasoning, plan, execution status and account risk in one updating view.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable


def _text(value: Any, default: str = "—") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _line(label: str, value: Any) -> str:
    return f"{label:<18} {value}"


def render_dashboard(snapshot: Dict[str, Any], stream=None, clear: bool = False) -> str:
    """Render exactly one consolidated dashboard and return its text."""
    s = snapshot or {}
    signal = s.get("signal", {}) or {}
    plan = s.get("plan", {}) or {}
    account = s.get("account", {}) or {}
    status = s.get("status", {}) or {}
    reasons = list(s.get("reasons", []) or [])
    events = list(s.get("events", []) or [])
    ts = _text(s.get("timestamp"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    direction = _text(signal.get("direction"), "NO_TRADE")
    confidence = _text(signal.get("confidence"), "N/A")
    confidence_label = confidence if str(confidence).upper() == "N/A" else f"{confidence}%"
    mode = _text(status.get("mode"), "PAPER")
    confidence_suffix = "%" if confidence not in ("N/A", "—", "unknown") else ""
    out = [
        "=" * 96,
        f"JARVIS UNIFIED DASHBOARD  |  {ts}  |  {_text(s.get('symbol'), '—')}  |  {_text(s.get('price'), '—')}",
        "-" * 96,
        _line("SIGNAL", f"{direction}  ({confidence}{confidence_suffix})"),
        _line("DECISION REASONS", "; ".join(_text(x) for x in reasons[:3]) if reasons else "Awaiting analysis"),
        _line("PLAN", f"Entry {_text(plan.get('entry'))} | TP1 {_text(plan.get('tp1'))} | SL {_text(plan.get('sl'))} | Expiry {_text(plan.get('expiry'))}"),
        _line("ACTION / GATES", f"{_text(plan.get('action'), 'WAIT')} | {_text(plan.get('gates'), 'not evaluated')}"),
        _line("STATUS", f"{mode} | {_text((status.get('readiness') or {}).get('status'), 'UNKNOWN')} | open {_text(status.get('open_trades'), '0')} | pending {_text(status.get('pending'), '0')} | {_text(status.get('uptime'), '—')}"),
        _line("AI SUGGESTION", _text((status.get('ai_suggestion') or {}).get('decision'), 'unavailable') + " (advisory; local gates authoritative)"),
        _line("AI RATIONALE", _text((status.get('ai_suggestion') or {}).get('rationale'), 'unavailable')),
        _line("GPU BACKEND", _text((status.get('gpu') or {}).get('backend'), 'unknown') + " | " + _text((status.get('gpu') or {}).get('device_name'), 'unknown')),
        _line("PART 7 SIGNAL", _text((status.get('part7') or {}).get('signal_name'), 'NEUTRAL') + " | " + _text((status.get('part7') or {}).get('signal_identity'), 'not run')),
        _line("PART 7 REASON", _text((status.get('part7') or {}).get('reason'), 'not evaluated')),
        _line("PART 7 RISK", _text((status.get('part7') or {}).get('volatility_status'), 'unknown') + " | " + ("ENTRY BLOCKED" if (status.get('part7') or {}).get('entry_blocked') else "entry allowed")),
        _line("PART 7 DATA", _text((status.get('part7') or {}).get('data_status'), 'missing') + " | backend " + _text((status.get('part7') or {}).get('computation_backend'), 'unknown')),
        "-" * 96,
        _line("DELTA AVAILABLE", f"{_money(account.get('delta_available'))} ({_text(account.get('delta_status'), 'unavailable')})"),
        _line("PAPER BALANCE", _money(account.get("paper_balance"))),
        _line("RISK / MARGIN", f"risk {_money(account.get('trade_risk'))} | margin {_money(account.get('margin'))}"),
        _line("EXPOSURE", f"{_text(account.get('contracts'), '0')} contracts @ {_money(account.get('contract_value_usdt'))} | notional {_money(account.get('notional'))} | leverage {_text(account.get('leverage'), 'AUTO')}x"),
        _line("CAPS", f"max leverage {_text(account.get('max_leverage_cap'))}x | max risk {_money(account.get('max_risk'))}"),
    ]
    if events:
        out.append(_line("RECENT EVENTS", " | ".join(_text(x) for x in events[-2:])))
    out.extend(["-" * 96, _line("LIMITATION", "No live validation; venue calls remain explicitly gated."), "=" * 96])
    text = "\n".join(out)
    if stream is None:
        stream = sys.stdout
    if clear:
        stream.write("\033[2J\033[H")
    stream.write(text + "\n")
    stream.flush()
    return text


class UnifiedDashboard:
    def __init__(self, clear: bool = True):
        self.clear = clear
        self.snapshot: Dict[str, Any] = {}

    def update(self, **sections: Any) -> Dict[str, Any]:
        self.snapshot.update(sections)
        return self.snapshot

    def render(self, stream=None) -> str:
        return render_dashboard(self.snapshot, stream=stream, clear=self.clear)
