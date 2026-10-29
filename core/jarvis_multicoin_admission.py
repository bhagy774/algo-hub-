"""Fail-closed portfolio admission/reservation primitives for future review.

This module is intentionally not wired to exchange submission.  It provides a
small transactional interface for serializing concurrent admission checks and
reconciling positions by full instrument identity; callers must still apply all
existing deterministic risk, ownership, oracle, and execution gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True, order=True)
class PositionIdentity:
    venue: str
    market_type: str
    instrument_id: str
    symbol: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PositionIdentity":
        fields = [str(value.get(k) or "").strip() for k in ("venue", "market_type", "instrument_id", "symbol")]
        if any(not x for x in fields):
            raise ValueError("full position identity required")
        return cls(*(x.lower() if i < 2 else x.upper() for i, x in enumerate(fields)))


@dataclass(frozen=True)
class RiskReservation:
    token: str
    identity: PositionIdentity
    notional: float
    created_at: float


class PortfolioAdmission:
    """Atomic open-position/risk budget reservations; disabled by default."""
    def __init__(self, *, enabled: bool = False, max_positions: int = 2,
                 max_total_notional: float = 0.0, clock=time.time):
        self.enabled = bool(enabled)
        self.max_positions = max(1, int(max_positions))
        self.max_total_notional = max(0.0, float(max_total_notional))
        self.clock = clock
        self._lock = threading.RLock()
        self._open: Dict[PositionIdentity, float] = {}
        self._reservations: Dict[str, RiskReservation] = {}
        self._generation = 0

    def reconcile(self, positions: Iterable[Mapping[str, object]]) -> bool:
        """Replace known opens only from a complete caller-provided snapshot.

        Identity/notional errors reject the whole update, preserving the prior
        state. This method has no exchange client and performs no order action.
        """
        candidate: Dict[PositionIdentity, float] = {}
        try:
            for position in positions:
                identity = PositionIdentity.from_mapping(position)
                notional = float(position.get("notional", 0.0))
                if notional < 0 or not math.isfinite(notional):
                    return False
                if identity in candidate:
                    return False
                candidate[identity] = notional
        except (TypeError, ValueError, AttributeError):
            return False
        with self._lock:
            self._open = candidate
            self._reservations.clear()
            self._generation += 1
        return True

    def reserve(self, identity: Mapping[str, object], notional: float) -> Optional[RiskReservation]:
        """Reserve capacity under one lock; no execution eligibility is implied."""
        if not self.enabled:
            return None
        try:
            key = PositionIdentity.from_mapping(identity)
            amount = float(notional)
            if amount <= 0 or not math.isfinite(amount):
                return None
        except (TypeError, ValueError):
            return None
        with self._lock:
            if key in self._open or any(r.identity == key for r in self._reservations.values()):
                return None
            occupied = len(self._open) + len(self._reservations)
            total = sum(self._open.values()) + sum(r.notional for r in self._reservations.values())
            if occupied >= self.max_positions:
                return None
            if self.max_total_notional and total + amount > self.max_total_notional:
                return None
            self._generation += 1
            token = f"reservation-{self._generation}"
            item = RiskReservation(token, key, amount, float(self.clock()))
            self._reservations[token] = item
            return item

    def release(self, token: str) -> bool:
        with self._lock:
            return self._reservations.pop(str(token), None) is not None

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "open": {"|".join(k.__dict__.values()): v for k, v in self._open.items()},
                "reservations": {token: item.notional for token, item in self._reservations.items()},
                "position_count": len(self._open) + len(self._reservations),
                "total_notional": sum(self._open.values()) + sum(r.notional for r in self._reservations.values()),
                "max_positions": self.max_positions,
                "max_total_notional": self.max_total_notional,
            }
