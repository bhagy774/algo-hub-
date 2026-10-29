"""Process-local close ownership and ambiguous-close reconciliation helpers."""
from __future__ import annotations

import threading
import time
import math
from typing import Any, Dict, Optional

_lock = threading.Lock()
_claims: Dict[str, Dict[str, Any]] = {}


def _key(symbol: str, position_id: Any = None) -> str:
    return f"{str(symbol or '').upper()}:{position_id or 'position'}"


def claim_close(symbol: str, position_id: Any = None, owner: str = "unknown") -> bool:
    """Claim one close operation; a second monitor cannot submit a duplicate."""
    key = _key(symbol, position_id)
    with _lock:
        if key in _claims:
            return False
        _claims[key] = {"owner": owner, "claimed_at": time.time()}
        return True


def release_close(symbol: str, position_id: Any = None, *, force: bool = False) -> None:
    """Release only after a confirmed close or explicit operator reconciliation."""
    key = _key(symbol, position_id)
    with _lock:
        if force or key in _claims:
            _claims.pop(key, None)


def close_claim_status(symbol: str, position_id: Any = None) -> Optional[Dict[str, Any]]:
    with _lock:
        claim = _claims.get(_key(symbol, position_id))
        return dict(claim) if claim else None


def reconcile_close(exchange: Any, symbol: str, side: str, requested_size: int) -> Dict[str, Any]:
    """Read venue position state after timeout/error; never retries an order.

    A close is confirmed only when the signed venue position is absent or has
    been reduced in the expected direction.  API failure and malformed state
    stay ambiguous so callers can fail closed.
    """
    try:
        positions = exchange.get_open_positions(symbol)
    except Exception as exc:
        return {"confirmed": False, "ambiguous": True, "reason": f"reconcile failed: {type(exc).__name__}"}
    if getattr(exchange, 'last_positions_query_ok', True) is False:
        return {'confirmed': False, 'ambiguous': True, 'reason': 'positions query unverified (API unavailable)'}
    if not isinstance(positions, list):
        return {"confirmed": False, "ambiguous": True, "reason": "malformed positions response"}
    requested = max(0, int(requested_size))
    remaining = 0.0
    for position in positions:
        if not isinstance(position, dict):
            return {"confirmed": False, "ambiguous": True, "reason": "malformed position record"}
        # If a stub/client returns a symbol, reject a mixed-symbol response;
        # never treat another contract's flat position as this close.
        reported = position.get("symbol") or (position.get("product") or {}).get("symbol")
        if reported and str(reported).upper() != str(symbol).upper():
            return {"confirmed": False, "ambiguous": True, "reason": "symbol mismatch"}
        raw = position.get("size", position.get("position_size", 0))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return {"confirmed": False, "ambiguous": True, "reason": "malformed position size"}
        if not math.isfinite(value):
            return {"confirmed": False, "ambiguous": True, "reason": "non-finite position size"}
        remaining += abs(value)
    # These callers submit a full-position close. A partial fill remains an
    # unresolved exposure and must not release ownership or trigger a retry.
    confirmed = remaining == 0.0
    return {"confirmed": confirmed, "ambiguous": not confirmed,
            "remaining_size": remaining, "requested_size": requested,
            "reason": "position reconciled" if confirmed else "position remains"}
