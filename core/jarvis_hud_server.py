#!/usr/bin/env python3
"""Local, read-only JARVIS HUD server.

The browser receives state over ``/ws``.  The coordinator may submit bounded
telemetry to ``/api/telemetry``; that endpoint deliberately has no order,
strategy, or execution action.
"""

import asyncio
import hmac
import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, Set
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [HUD] %(message)s")
logger = logging.getLogger("JarvisHUD")

app = FastAPI(title="JARVIS HUD Server")
connected_clients: Set[WebSocket] = set()
state: Dict[str, Any] = {
    "parts": {},
    "trades": [],
    "stats": {"wins": 0, "losses": 0, "total_pnl": 0.0, "equity_curve": [], "win_rate": 0.0},
    "health": {},
    "pipeline": {"watcher_active": False, "last_packet_ts": None, "specialist_opinions": {}},
    "oracle": {},
    "chat_history": [],
    "system_online": True,
    "coordinator_telemetry": {},
    "dashboard": {},
}

MAX_TEXT = 500
MAX_THOUGHTS = 50
MAX_CHAT_HISTORY = 100
MAX_TELEMETRY_BYTES = 64 * 1024


def _bounded_text(value: Any, limit: int = MAX_TEXT) -> str:
    return str(value).replace("\x00", "")[:limit]


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Produce a small JSON-only display value from untrusted telemetry."""
    if depth >= 5:
        return _bounded_text(value, 120)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, dict):
        return {
            _bounded_text(key, 80): _json_safe(item, depth + 1)
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value[:MAX_THOUGHTS]]
    return _bounded_text(value)


def _extract_direction(text: Any) -> str:
    upper = _bounded_text(text).upper()
    if any(word in upper for word in ("BULLISH", "CALL", "BUY", "LONG")):
        return "BULLISH"
    if any(word in upper for word in ("BEARISH", "PUT", "SELL", "SHORT")):
        return "BEARISH"
    return "NEUTRAL"


def _extract_confidence(text: Any) -> float:
    match = re.search(r"(?:confidence|score|avg score)[^\d]*([0-9]+\.?[0-9]*)", _bounded_text(text), re.I)
    if not match:
        return 0.0
    value = float(match.group(1))
    value = value * 100.0 if value <= 1.5 else value
    return max(0.0, min(100.0, value))


def apply_coordinator_telemetry(payload: Any) -> Dict[str, Any]:
    """Apply display-only coordinator output; never invokes a trading module."""
    if not isinstance(payload, dict):
        raise ValueError("telemetry must be an object")
    try:
        encoded = json.dumps(payload, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("telemetry is not serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_TELEMETRY_BYTES:
        raise ValueError("telemetry is too large")

    allowed = ("market_data", "thoughts", "matrix", "regime", "signal", "ai_consensus", "dashboard")
    telemetry = {key: _json_safe(payload[key]) for key in allowed if key in payload}
    telemetry["received_at"] = datetime.now().isoformat()
    state["coordinator_telemetry"] = telemetry
    if isinstance(telemetry.get("dashboard"), dict):
        state["dashboard"] = telemetry["dashboard"]
    state["pipeline"]["coordinator_regime"] = _bounded_text(telemetry.get("regime", "NEUTRAL"), 80)
    state["pipeline"]["last_packet_ts"] = telemetry["received_at"]

    signal = telemetry.get("signal")
    if isinstance(signal, dict):
        direction = _extract_direction(signal.get("direction", ""))
        confidence = _extract_confidence(signal.get("confidence", signal.get("score", "")))
        raw = _bounded_text(
            f"Coordinator: {direction}; confidence {confidence:.1f}%; "
            f"regime {state['pipeline']['coordinator_regime']}",
            200,
        )
        state["parts"]["Coordinator"] = {
            "direction": direction,
            "confidence": confidence,
            "raw": raw,
            "ts": telemetry["received_at"],
        }
    return telemetry


def _allowed_origin(origin: str | None, host: str | None) -> bool:
    """Allow same-host browsers or explicit deployment origins."""
    if not origin:
        return True  # non-browser local clients have no Origin header
    configured = {item.strip() for item in os.environ.get("JARVIS_ALLOWED_ORIGINS", "").split(",") if item.strip()}
    if origin in configured:
        return True
    return bool(host and urlparse(origin).netloc == host)


def _loopback_request(request: Request) -> bool:
    return bool(request.client and request.client.host in {"127.0.0.1", "::1", "localhost", "testclient"})


async def broadcast(payload: Dict[str, Any]) -> None:
    dead = set()
    message = json.dumps(payload, default=str, separators=(",", ":"))
    for ws in tuple(connected_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


@app.get("/api/status")
async def status() -> Dict[str, str]:
    """Small readiness endpoint; it exposes no telemetry or controls."""
    return {"status": "online"}


@app.post("/api/telemetry")
async def receive_telemetry(request: Request) -> Dict[str, Any]:
    """Accept bounded local display telemetry, with optional launcher token."""
    if not _loopback_request(request):
        raise HTTPException(status_code=403, detail="telemetry is local only")
    expected_token = os.environ.get("JARVIS_HUD_INGEST_TOKEN")
    supplied_token = request.headers.get("X-Jarvis-Hud-Token", "")
    if expected_token and not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(status_code=403, detail="invalid telemetry token")
    try:
        payload = await request.json()
        accepted = apply_coordinator_telemetry(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await broadcast({"type": "STATE_UPDATE", "data": state})
    return {"accepted": True, "received_at": accepted["received_at"]}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    if not _allowed_origin(ws.headers.get("origin"), ws.headers.get("host")):
        await ws.close(code=1008)
        return
    await ws.accept()
    connected_clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "STATE_UPDATE", "data": state}, default=str))
        while True:
            try:
                message = json.loads(await ws.receive_text())
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(message, dict) and message.get("type") == "CHAT":
                await handle_chat(message.get("text", ""))
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)


async def handle_chat(user_text: Any) -> str:
    text = _bounded_text(user_text).strip()
    if not text:
        return "Please enter a message."
    state["chat_history"].append({"role": "user", "text": text, "ts": datetime.now().isoformat()})
    reply = _bounded_text(await asyncio.get_running_loop().run_in_executor(None, _call_jarvis_chat, text))
    state["chat_history"].append({"role": "jarvis", "text": reply, "ts": datetime.now().isoformat()})
    state["chat_history"] = state["chat_history"][-MAX_CHAT_HISTORY:]
    await broadcast({"type": "CHAT_REPLY", "role": "jarvis", "text": reply})
    return reply


@app.post("/chat")
async def http_chat(request: Request) -> Dict[str, str]:
    if not _allowed_origin(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(status_code=403, detail="origin not allowed")
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="chat payload must be an object")
    return {"reply": await handle_chat(body.get("message", body.get("text", "")))}


@app.get("/chat")
async def chat_status() -> Dict[str, str]:
    """Compatibility/readiness route; no history or execution data is exposed."""
    return {"status": "available"}


def _call_jarvis_chat(user_text: str) -> str:
    return "Model chat retired; Laya is limited to isolated market audit."


@app.get("/")
async def serve_dashboard() -> HTMLResponse:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_hud", "index.html")
    with open(path, encoding="utf-8") as file:
        return HTMLResponse(file.read())


@app.get("/api/oracle")
async def get_oracle_map() -> Dict[str, Any]:
    return _json_safe(state.get("oracle", {}))


async def state_push_loop() -> None:
    while True:
        await asyncio.sleep(1)
        if connected_clients:
            await broadcast({"type": "STATE_UPDATE", "data": state})


def start_bus_listener() -> None:
    """Attach optional local cognitive-bus telemetry; failure does not affect HUD."""
    try:
        from jarvis_cognitive_bus import CognitiveBus
        from jarvis_watcher_ai import JarvisWatcherAI
        bus = CognitiveBus()
        watcher = JarvisWatcherAI(bus=bus)
        watcher.start()
        state["pipeline"]["watcher_active"] = True

        def on_thoughts(message: Dict[str, Any]) -> None:
            sender = _bounded_text(message.get("sender", "Unknown"), 80)
            payload = message.get("payload", "")
            state["parts"][sender] = {
                "direction": _extract_direction(payload), "confidence": _extract_confidence(payload),
                "raw": _bounded_text(payload, 100), "ts": datetime.now().isoformat(),
            }

        def on_health(message: Dict[str, Any]) -> None:
            sender = _bounded_text(message.get("sender", "Unknown"), 80)
            payload = message.get("payload", {})
            severity = payload.get("severity", "WARNING") if isinstance(payload, dict) else "WARNING"
            state["health"][sender] = {"severity": _bounded_text(severity, 40), "ts": datetime.now().isoformat()}

        def on_oracle(message: Dict[str, Any]) -> None:
            if isinstance(message.get("payload"), dict):
                state["oracle"] = _json_safe(message["payload"])

        bus.subscribe("THOUGHTS", on_thoughts)
        bus.subscribe("HEALTH", on_health)
        bus.subscribe("ORACLE_FORECAST", on_oracle)
    except Exception as exc:
        logger.info("CognitiveBus unavailable; starting demo display data: %s", exc)
        _start_demo_data()


def _start_demo_data() -> None:
    """Retain the existing display-only fallback when local bus modules are absent."""
    state["pipeline"]["watcher_active"] = True
    now = datetime.now().isoformat()
    for number in range(1, 13):
        name = f"Part{number}_Demo"
        state["parts"][name] = {
            "direction": "NEUTRAL", "confidence": 0.0,
            "raw": "Demo telemetry — no local CognitiveBus connected", "ts": now,
        }


def start_terminal_tailer() -> None:
    """Tail the optional local terminal log without affecting coordinator state."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "jarvis_terminal.log")
    state["terminal_logs"] = []

    async def tail() -> None:
        try:
            import aiofiles
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # aiofiles doesn't strictly have a+, but a+b is supported, or we can just touch the file and use r
            # Let's ensure the file exists first:
            with open(path, "a+", encoding="utf-8"):
                pass

            async with aiofiles.open(path, mode="r", encoding="utf-8") as file:
                await file.seek(0, 2)
                while True:
                    line = await file.readline()
                    if line:
                        state["terminal_logs"] = (state["terminal_logs"] + [_bounded_text(line, 500)])[-200:]
                    else:
                        await asyncio.sleep(0.1)
        except Exception as exc:
            logger.debug("terminal tailer stopped: %s", exc)

    task = asyncio.create_task(tail())
    # Keep a strong reference to the task so it doesn't get garbage collected
    state["_tail_task"] = task


@app.on_event("startup")
async def on_startup() -> None:
    threading.Thread(target=start_bus_listener, daemon=True).start()
    start_terminal_tailer()
    asyncio.create_task(state_push_loop())
    logger.info("HUD Server online on local address")


if __name__ == "__main__":
    host = os.environ.get("JARVIS_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("JARVIS_PORT", "7788"))
    except ValueError:
        port = 7788
    uvicorn.run(app, host=host, port=port, log_level="warning")
