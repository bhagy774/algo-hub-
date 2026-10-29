"""Symbol-safe market routing for JARVIS.

A route is selected before data collection.  Its symbol is then carried through
candles, validation, analysis, paper ledger and (only after separate live
opt-in) execution.  This prevents a BTC signal from being submitted as an
altcoin order.

The router deliberately supports crypto execution routing only.  India/NSE is
kept data-only until an independently tested broker execution adapter is
provided; market data must never be mistaken for broker permission.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class MarketRoute:
    venue: str
    symbol: str
    base_asset: str
    execution_venue: str
    options_venue: str
    selected_at: str
    score: float = 0.0
    options_available: bool = False
    status: str = "READY"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class MarketRouter:
    """Select a tradeable market and retain it while a position is open."""

    def __init__(self, scanner: Any = None, delta_client: Any = None):
        self.scanner = scanner
        self.delta = delta_client
        self._route: Optional[MarketRoute] = None
        self._last_selection_epoch = 0.0
        self._rejected: Dict[str, float] = {}

    def reject_symbol(self, symbol: str, reason: str = "", ttl_seconds: Optional[float] = None) -> None:
        """Temporarily skip a symbol whose analysis data is unusable (never while a position is open)."""
        ttl = float(ttl_seconds if ttl_seconds is not None else os.getenv("JARVIS_REJECT_TTL_SEC", "600"))
        base = self._base_asset(self._normalise_symbol(symbol))
        self._rejected[base] = time.time() + ttl
        self._route = None
        import logging
        logging.getLogger(__name__).warning("[MARKET-ROUTER] Skipping %s for %.0fs: %s", base, ttl, reason)

    def _is_rejected(self, symbol: str) -> bool:
        base = self._base_asset(self._normalise_symbol(symbol))
        until = self._rejected.get(base)
        if until is None:
            return False
        if time.time() >= until:
            self._rejected.pop(base, None)
            return False
        return True

    @staticmethod
    def _base_asset(symbol: str) -> str:
        symbol = str(symbol or "").upper().strip()
        for quote in ("USDT", "USD", "INR"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return symbol[:-len(quote)]
        return symbol

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        return str(symbol or "").upper().replace("-", "").replace("_", "").strip()

    def current(self) -> Optional[MarketRoute]:
        return self._route

    def select_crypto(self, has_open_position: bool = False) -> MarketRoute:
        """Run scanner once, then verify the exact selected Delta contract.

        If a position is open, the prior route remains immutable.  If a scan,
        products lookup, or candidate validation fails, this method returns a
        BLOCKED route rather than silently submitting BTC or another asset.
        """
        if has_open_position and self._route is not None:
            return self._route
        interval = max(30, int(os.getenv("JARVIS_SELECTION_INTERVAL_SEC", "300")))
        if self._route is not None and self._route.status == "READY" and time.time() - self._last_selection_epoch < interval:
            return self._route
        if not _truthy(os.getenv("JARVIS_MULTI_MARKET", "1")):
            candidate = self._normalise_symbol(os.getenv("JARVIS_DEFAULT_SYMBOL", "BTCUSDT"))
            score = 0.0
        else:
            if self.scanner is None:
                return self._blocked("Coin scanner is unavailable")
            try:
                self.scanner.run_now()
                candidate = self._normalise_symbol(self.scanner.get_delta_symbol())
                score = float(self.scanner.get_all_scores().get(self.scanner.get_best_coin(), {}).get("total", 0.0))
            except Exception as exc:
                return self._blocked(f"Coin scan failed: {type(exc).__name__}")

        if not candidate or candidate == "USDT":
            return self._blocked("Scanner returned an invalid symbol")
        if self._is_rejected(candidate):
            fallback = self._normalise_symbol(os.getenv("JARVIS_DEFAULT_SYMBOL", "BTCUSDT"))
            if self._is_rejected(fallback):
                return self._blocked(f"{candidate} and fallback {fallback} have unusable candle data")
            candidate = fallback
        if self.delta is None:
            return self._blocked("Delta client is unavailable")
        try:
            available = {self._normalise_symbol(s) for s in self.delta.get_available_symbols()}
        except Exception as exc:
            return self._blocked(f"Delta product lookup failed: {type(exc).__name__}")
        if candidate not in available:
            # Quote-currency fallback: Delta India lists perps as XXXUSD while
            # scanners typically emit XXXUSDT. Try the alternate quote before
            # declaring the symbol untradeable.
            alt = None
            if candidate.endswith("USDT"):
                alt = candidate[:-4] + "USD"
            elif candidate.endswith("USD"):
                alt = candidate[:-3] + "USDT"
            if alt and alt in available:
                import logging
                logging.getLogger(__name__).info("[MARKET-ROUTER] Mapped %s -> %s (Delta quote format)", candidate, alt)
                candidate = alt
            else:
                return self._blocked(f"Selected symbol {candidate} is not tradeable on Delta")

        base = self._base_asset(candidate)
        options_available = False
        try:
            options_available = bool(self.delta.get_options_chain(base))
        except Exception:
            # Options data is advisory. Its absence must be explicit but cannot
            # change the selected futures symbol.
            options_available = False
        self._route = MarketRoute(
            venue="crypto",
            symbol=candidate,
            base_asset=base,
            execution_venue="delta",
            options_venue="delta" if options_available else "none",
            selected_at=datetime.now(timezone.utc).isoformat(),
            score=score,
            options_available=options_available,
            reason="Scanner candidate verified against Delta products",
        )
        self._last_selection_epoch = time.time()
        return self._route

    def india_status(self, upstox_data: Any = None) -> MarketRoute:
        """Expose India data readiness without pretending it can place orders."""
        if not _truthy(os.getenv("JARVIS_INDIA_DATA", "1")):
            return MarketRoute("india", "", "", "none", "none", datetime.now(timezone.utc).isoformat(), status="DISABLED", reason="JARVIS_INDIA_DATA=0")
        if upstox_data is None or not getattr(upstox_data, "_enabled", False):
            return MarketRoute("india", "", "", "none", "upstox", datetime.now(timezone.utc).isoformat(), status="BLOCKED", reason="Upstox data is not authenticated")
        if not getattr(upstox_data, "is_active", lambda: False)():
            return MarketRoute("india", "", "", "none", "upstox", datetime.now(timezone.utc).isoformat(), status="BLOCKED", reason="NSE is closed or Upstox data is unavailable")
        return MarketRoute("india", "NSE_INDEX|Nifty 50", "NIFTY", "none", "upstox", datetime.now(timezone.utc).isoformat(), status="DATA_ONLY", reason="Data and options-chain ready; no verified India execution adapter")

    def _blocked(self, reason: str) -> MarketRoute:
        return MarketRoute("crypto", "", "", "none", "none", datetime.now(timezone.utc).isoformat(), status="BLOCKED", reason=reason)
