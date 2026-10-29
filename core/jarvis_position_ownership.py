"""Process-local position ownership and close idempotency registry.

The active auto-trader is the canonical lifecycle owner.  The legacy position
manager may observe positions, but it must not submit a second reduce-only
close for a position already claimed by another manager.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

_lock = threading.RLock()
_claims: Dict[str, str] = {}
_close_claims: Dict[str, str] = {}


def _key(position_id: object) -> str:
    return str(position_id or "").strip()


def claim_position(position_id: object, owner: str) -> bool:
    """Atomically claim a position for one lifecycle owner."""
    key = _key(position_id)
    owner = str(owner or "").strip()
    if not key or not owner:
        return False
    with _lock:
        existing = _claims.get(key)
        if existing is None:
            _claims[key] = owner
            return True
        return existing == owner


def position_owner(position_id: object) -> Optional[str]:
    with _lock:
        return _claims.get(_key(position_id))


def claim_close(position_id: object, owner: str) -> Tuple[bool, Optional[str]]:
    """Claim the one allowed close attempt; return (allowed, current owner).

    A second monitor seeing the same position must not retry or reverse it.  A
    caller that does not own the lifecycle position cannot claim its close.
    """
    key = _key(position_id)
    owner = str(owner or "").strip()
    if not key or not owner:
        return False, None
    with _lock:
        current = _claims.get(key)
        if current is None:
            _claims[key] = owner
            current = owner
        if current != owner:
            return False, current
        close_owner = _close_claims.get(key)
        if close_owner is None:
            _close_claims[key] = owner
            return True, owner
        # A close claim is one-shot, including for the same owner.  The
        # original request may have reached the venue before its response was
        # lost, so retrying could duplicate the close or reverse exposure.
        return False, close_owner


def release_position(position_id: object, owner: str) -> None:
    """Release an unfilled ownership claim (used only before an order)."""
    key = _key(position_id)
    with _lock:
        if _claims.get(key) == owner and key not in _close_claims:
            _claims.pop(key, None)


def clear_registry() -> None:
    """Test-only reset; never called by runtime code."""
    with _lock:
        _claims.clear()
        _close_claims.clear()
