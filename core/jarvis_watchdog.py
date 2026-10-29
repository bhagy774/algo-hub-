"""JARVIS Crash Recovery + Watchdog.

Fail-safe by design:
- Startup reconciliation compares persisted local state (jarvis_state.json) with
  exchange truth (delta_api_wrapper). Exchange truth always wins. Any API error
  is logged and startup continues; NO orders are ever placed here.
- A lightweight daemon thread checks heartbeats every 60 s (brain loop alive,
  data feed freshness) and logs/alerts (telegram_notifier if configured,
  guarded). Detection + alerting only — no auto-restart.
- get_health() returns a plain dict for HUD/tests.

Disable with JARVIS_WATCHDOG=0. Paper-mode safety guards are untouched.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("JarvisWatchdog")

STATE_FILE = os.environ.get("JARVIS_STATE_FILE", "jarvis_state.json")
CHECK_INTERVAL_SECONDS = 60
# How long before a heartbeat is considered stale.
# Configurable via DOCTOR_STALE_THRESHOLD env var (default: 600s = 10 min).
# Old default was 180s which caused false alerts when no trades were open.
try:
    HEARTBEAT_STALE_SECONDS = int(os.environ.get("DOCTOR_STALE_THRESHOLD", "600"))
    if HEARTBEAT_STALE_SECONDS <= 0:
        raise ValueError("threshold must be positive")
except (TypeError, ValueError):
    HEARTBEAT_STALE_SECONDS = 600

_state_lock = threading.Lock()
_heartbeats: Dict[str, float] = {}
_health: Dict[str, Any] = {
    "started_at": None,
    "last_check": None,
    "services": {},
    "alerts": [],
    "reconcile": None,
}


# ─────────────────────────── State persistence ───────────────────────────

def save_state(open_trades: List[Dict], path: str = STATE_FILE, extra: Optional[Dict] = None) -> bool:
    """Persist open-trade state to JSON. Never raises."""
    try:
        payload = {
            "saved_at": datetime.now().isoformat(),
            "open_trades": open_trades or [],
        }
        if extra:
            payload["extra"] = extra

        tmp = f"{path}.{threading.get_ident()}.tmp"

        # Serialize and perform I/O outside the lock to increase concurrency
        payload_str = json.dumps(payload, indent=2, default=str)
        with open(tmp, "w") as fh:
            fh.write(payload_str)

        with _state_lock:
            os.replace(tmp, path)
        return True
    except Exception as exc:  # fail-safe
        if 'tmp' in locals() and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        logger.warning("[WATCHDOG] state save failed: %s", exc)
        return False


def load_state(path: str = STATE_FILE) -> Dict[str, Any]:
    """Load persisted state. Returns {'open_trades': []} on any failure."""
    try:
        with _state_lock:
            with open(path) as fh:
                content = fh.read()

        # Parse JSON outside the lock
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("state file is not a JSON object")
        data.setdefault("open_trades", [])
        return data
    except FileNotFoundError:
        return {"open_trades": [], "saved_at": None}
    except Exception as exc:
        logger.warning("[WATCHDOG] state load failed (%s); starting fresh", exc)
        return {"open_trades": [], "saved_at": None, "error": str(exc)}


# ─────────────────────────── Startup reconciliation ──────────────────────

def reconcile_state(exchange_client=None, state_path: str = STATE_FILE) -> Dict[str, Any]:
    """Compare persisted local open trades with exchange open positions.

    Exchange truth is the source of truth. Fail-safe: any API error -> log and
    continue. NEVER places or cancels an order.
    """
    report: Dict[str, Any] = {
        "checked_at": datetime.now().isoformat(),
        "exchange_ok": False,
        "exchange_positions": 0,
        "local_trades": 0,
        "mismatches": [],
        "error": None,
    }
    try:
        local = load_state(state_path)
        local_trades = local.get("open_trades", []) or []
        report["local_trades"] = len(local_trades)

        if exchange_client is None:
            report["error"] = "no exchange client provided; skipping exchange check"
            logger.info("[WATCHDOG] reconcile: no exchange client; local trades=%d", len(local_trades))
            _health["reconcile"] = report
            return report

        try:
            positions = exchange_client.get_open_positions() or []
            if getattr(exchange_client, 'last_positions_query_ok', True) is False:
                report['error'] = 'exchange positions unavailable (not verified flat); skipping mismatch check'
                logger.info('[WATCHDOG] reconcile: exchange positions unavailable; not treating as 0 positions')
                _health['reconcile'] = report
                return report
            report["exchange_ok"] = True
        except Exception as api_err:
            report["error"] = f"exchange API error: {api_err}"
            logger.warning("[WATCHDOG] reconcile: exchange unreachable (%s) — continuing startup", api_err)
            _health["reconcile"] = report
            return report

        report["exchange_positions"] = len(positions)
        # Exchange truth wins: log anything local that the exchange doesn't know
        # about, and anything on the exchange we have no local record of.
        if len(local_trades) != len(positions):
            msg = (
                f"state mismatch: local open trades={len(local_trades)} "
                f"vs exchange open positions={len(positions)} — adopting exchange truth"
            )
            report["mismatches"].append(msg)
            logger.warning("[WATCHDOG] %s", msg)
        for pos in positions:
            sym = pos.get("symbol") or pos.get("product_symbol") or "?"
            if not any(sym in str(t) for t in local_trades):
                msg = f"exchange position with no local record: {sym}"
                report["mismatches"].append(msg)
                logger.warning("[WATCHDOG] %s", msg)
        if not report["mismatches"]:
            logger.info("[WATCHDOG] reconcile OK: %d open position(s) match local state", len(positions))
    except Exception as exc:  # absolute fail-safe
        report["error"] = f"reconcile failed: {exc}"
        logger.error("[WATCHDOG] reconcile unexpected error: %s", exc)
    _health["reconcile"] = report
    return report


# ─────────────────────────── Watchdog thread ─────────────────────────────

def beat(service: str) -> None:
    """Record a heartbeat for a core service (brain loop, data feed, ...)."""
    _heartbeats[service] = time.time()


def _alert(text: str) -> None:
    logger.warning("[WATCHDOG] ALERT: %s", text)
    _health["alerts"].append({"at": datetime.now().isoformat(), "message": text})
    _health["alerts"] = _health["alerts"][-50:]
    try:
        if os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN"):
            from telegram_notifier import send_message
            send_message(f"🐕 JARVIS watchdog: {text}")
    except Exception:
        pass  # telegram is best-effort only


def get_health() -> Dict[str, Any]:
    """Return current watchdog health as a plain dict."""
    now = time.time()
    services = {}
    for name, ts in _heartbeats.items():
        services[name] = {
            "last_beat": datetime.fromtimestamp(ts).isoformat(),
            "age_seconds": round(now - ts, 1),
            "alive": (now - ts) <= HEARTBEAT_STALE_SECONDS,
        }
    return {
        "watchdog_running": _health.get("started_at") is not None and _thread_alive(),
        "started_at": _health.get("started_at"),
        "last_check": _health.get("last_check"),
        "services": services,
        "alerts": list(_health.get("alerts", [])),
        "reconcile": _health.get("reconcile"),
    }


_watchdog_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _thread_alive() -> bool:
    return bool(_watchdog_thread and _watchdog_thread.is_alive())


def _loop() -> None:
    while not _stop_event.wait(CHECK_INTERVAL_SECONDS):
        try:
            now = time.time()
            _health["last_check"] = datetime.now().isoformat()
            if not _heartbeats:
                logger.debug("[WATCHDOG] no heartbeats registered yet")
                continue
            for name, ts in list(_heartbeats.items()):
                age = now - ts
                if age > HEARTBEAT_STALE_SECONDS:
                    _alert(f"service '{name}' heartbeat stale ({age:.0f}s > {HEARTBEAT_STALE_SECONDS}s)")
                else:
                    logger.debug("[WATCHDOG] service '%s' alive (%.0fs)", name, age)
        except Exception as exc:  # watchdog must never die
            logger.error("[WATCHDOG] loop error: %s", exc)


def start_watchdog(exchange_client=None, state_path: str = STATE_FILE) -> bool:
    """Run startup reconciliation and launch the watchdog thread.

    Guarded: returns False instead of raising. Disabled if JARVIS_WATCHDOG=0.
    """
    global _watchdog_thread
    if os.environ.get("JARVIS_WATCHDOG", "1") == "0":
        logger.info("[WATCHDOG] disabled via JARVIS_WATCHDOG=0")
        return False
    try:
        reconcile_state(exchange_client=exchange_client, state_path=state_path)
        if _thread_alive():
            return True
        _stop_event.clear()
        _health["started_at"] = datetime.now().isoformat()
        _watchdog_thread = threading.Thread(target=_loop, name="jarvis-watchdog", daemon=True)
        _watchdog_thread.start()
        logger.info("[WATCHDOG] started (interval=%ds)", CHECK_INTERVAL_SECONDS)
        return True
    except Exception as exc:
        logger.error("[WATCHDOG] start failed (continuing without watchdog): %s", exc)
        return False


def stop_watchdog() -> None:
    _stop_event.set()
    if _watchdog_thread and _watchdog_thread.is_alive():
        _watchdog_thread.join(timeout=2)
