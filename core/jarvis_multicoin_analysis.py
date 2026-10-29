"""Opt-in, bounded multi-coin Part 7 shadow analysis.

This is deliberately not a Part 1-12 signal generator. It runs the stateless
shared Part 7 analyzer against independent closed-candle snapshots and exposes
diagnostics only. Candidate output is never consumed by selected-symbol entry,
risk, ownership, reconciliation, or order paths.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from direct_candle_cache import CandleDataError, CandleSnapshot, DirectCandleCache, LIVE_TIMEFRAMES

logger = logging.getLogger(__name__)


def multicoin_analysis_enabled(value: Optional[str] = None) -> bool:
    """Shadow processing is explicitly disabled unless the flag is set to 1."""
    selected = os.getenv("JARVIS_MULTICOIN_ANALYSIS", "0") if value is None else value
    return str(selected).strip().lower() in {"1", "true", "yes", "on"}


def _canonical_symbol(value: Any) -> str:
    # No quote conversion or ticker aliasing; punctuation normalization only
    # follows the existing candle-cache symbol format.
    return str(value or "").upper().replace("/", "").replace("-", "").replace("_", "").strip()


class MultiCoinPart7Shadow:
    """Bounded background Part 7 diagnostics over exact candidate symbols.

    ``candidate_supplier`` must return symbols already verified by the caller
    against the provider's exact product list. This class never maps symbols or
    treats a shadow result as an executable recommendation.
    """

    def __init__(
        self,
        candle_cache: DirectCandleCache,
        candidate_supplier: Callable[[], Iterable[str]],
        *,
        enabled: bool = False,
        max_workers: int = 2,
        max_candidates: int = 15,
        refresh_interval_seconds: float = 300.0,
        result_max_age_seconds: float = 120.0,
        clock: Callable[[], float] = time.time,
        analyzer: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.candle_cache = candle_cache
        self.candidate_supplier = candidate_supplier
        self.max_workers = max(1, min(4, int(max_workers)))
        self.max_candidates = max(1, min(30, int(max_candidates)))
        self.refresh_interval_seconds = max(30.0, float(refresh_interval_seconds))
        self.result_max_age_seconds = max(1.0, float(result_max_age_seconds))
        self.clock = clock
        self._analyzer = analyzer
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="JarvisCoinShadow")
        self._lock = threading.RLock()
        self._inflight: Dict[str, Future] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._last_attempt: Dict[str, float] = {}
        self._candidate_symbols: Sequence[str] = ()
        self._next_candidate_refresh = 0.0
        self._closed = False

    def _refresh_candidates(self, now: float) -> None:
        if now < self._next_candidate_refresh:
            return
        self._next_candidate_refresh = now + self.refresh_interval_seconds
        try:
            candidates = []
            seen = set()
            for raw in self.candidate_supplier() or ():
                symbol = _canonical_symbol(raw)
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    candidates.append(symbol)
                if len(candidates) >= self.max_candidates:
                    break
            self._candidate_symbols = tuple(candidates)
        except Exception as exc:
            logger.warning("[MULTICOIN-SHADOW] Candidate discovery failed: %s", type(exc).__name__)
            self._candidate_symbols = ()

    def _collect_completed(self) -> None:
        with self._lock:
            done = [(symbol, future) for symbol, future in self._inflight.items() if future.done()]
            for symbol, future in done:
                self._inflight.pop(symbol, None)
                try:
                    self._results[symbol] = future.result()
                except Exception as exc:
                    self._results[symbol] = self._blocked(symbol, f"worker error: {type(exc).__name__}")

    def poll(
        self,
        *,
        selected_symbol: Optional[str] = None,
        selected_snapshot: Optional[CandleSnapshot] = None,
    ) -> Dict[str, Any]:
        """Schedule at most ``max_workers`` jobs and return a bounded status view."""
        if not self.enabled or self._closed:
            return {"enabled": False, "status": "DISABLED", "results": {}}
        self._collect_completed()
        now = float(self.clock())
        self._refresh_candidates(now)
        selected = _canonical_symbol(selected_symbol)
        with self._lock:
            for symbol in self._candidate_symbols:
                if len(self._inflight) >= self.max_workers:
                    break
                if symbol in self._inflight:
                    continue
                previous = self._results.get(symbol)
                if previous is not None:
                    fetched_at = previous.get("snapshot_fetched_at")
                    if fetched_at is not None and 0 <= now - float(fetched_at) <= self.result_max_age_seconds:
                        continue
                    # Keep an expired diagnostic visible as STALE while a refresh
                    # is throttled or running; absence must never look like a
                    # fresh/cleared signal. Retry cadence is measured from the
                    # last attempt for both successful and failed refreshes.
                    if now - self._last_attempt.get(symbol, 0.0) < self.refresh_interval_seconds:
                        continue
                elif now - self._last_attempt.get(symbol, 0.0) < self.refresh_interval_seconds:
                    continue
                reuse = selected_snapshot if symbol == selected and selected_snapshot is not None else None
                future = self._executor.submit(self._analyze_symbol, symbol, reuse)
                self._inflight[symbol] = future
                self._last_attempt[symbol] = now
            results = {}
            for symbol, result in self._results.items():
                current = dict(result)
                fetched_at = current.get("snapshot_fetched_at")
                if fetched_at is None:
                    current["freshness_status"] = "UNAVAILABLE"
                else:
                    age = now - float(fetched_at)
                    current["freshness_status"] = "FRESH" if 0 <= age <= self.result_max_age_seconds else "STALE"
                    current["age_seconds"] = max(0.0, age)
                    if current["freshness_status"] == "STALE":
                        current["status"] = "STALE"
                        current["execution_eligible"] = False
                results[symbol] = current
            return {
                "enabled": True,
                "status": "RUNNING" if self._inflight else "IDLE",
                "candidate_count": len(self._candidate_symbols),
                "inflight_count": len(self._inflight),
                "max_workers": self.max_workers,
                "results": results,
            }

    def _analyze_symbol(self, symbol: str, selected_snapshot: Optional[CandleSnapshot]) -> Dict[str, Any]:
        try:
            # Candidate symbols have already been matched exactly to the
            # Delta product list. The product type and venue contract ID are
            # not exposed by that adapter, so we keep the market explicit but
            # unverified and use the exact listed symbol (never a quote alias).
            snapshot = selected_snapshot or self.candle_cache.refresh(
                symbol,
                venue="delta",
                market_type="unverified",
                instrument_id=symbol,
            )
            expected_identity = ("delta", "unverified", symbol, symbol)
            if tuple(snapshot.identity) != expected_identity:
                raise CandleDataError("snapshot request identity does not match exact Delta candidate")
            if _canonical_symbol(snapshot.symbol) != symbol:
                raise CandleDataError("snapshot symbol does not match candidate")
            if not snapshot.frames or any(tf not in snapshot.frames for tf in LIVE_TIMEFRAMES):
                raise CandleDataError("snapshot does not include every required native timeframe")
            age = float(self.clock()) - float(snapshot.fetched_at)
            if age < -2 or age > self.result_max_age_seconds:
                raise CandleDataError("snapshot is stale or from the future")
            from part7_signal import analyze_timeframe
            analyzer = self._analyzer or analyze_timeframe
            results: Dict[str, Dict[str, Any]] = {}
            for timeframe in LIVE_TIMEFRAMES:
                candle_frame = snapshot.frames[timeframe]
                frame = candle_frame.closed.copy(deep=True)
                frame.attrs["symbol"] = symbol
                frame.attrs["timeframe"] = timeframe
                results[timeframe] = analyzer(
                    frame,
                    symbol=symbol,
                    timeframe=timeframe,
                    context={"selected_symbol": symbol, "shadow_analysis": True},
                )
            sources = {tf: snapshot.frames[tf].source for tf in LIVE_TIMEFRAMES}
            versions = {
                tf: {
                    "source": snapshot.frames[tf].source,
                    "last_closed": str(snapshot.frames[tf].closed.index[-1]),
                    "current_time": snapshot.frames[tf].current.get("time") if snapshot.frames[tf].current else None,
                }
                for tf in LIVE_TIMEFRAMES
            }
            version = hashlib.sha256(json.dumps(
                {"identity": snapshot.identity, "timeframes": versions},
                sort_keys=True, default=str,
            ).encode("utf-8")).hexdigest()[:20]
            complete = all(value.get("data_status") == "valid" for value in results.values())
            return {
                "symbol": symbol,
                "status": "COMPLETE" if complete else "PARTIAL",
                "scope": "part7-shadow-only",
                "coverage": ["Part7"],
                "part7_by_timeframe": results,
                "request_identity": list(snapshot.identity),
                "provider_by_timeframe": sources,
                "snapshot_version": version,
                "snapshot_fetched_at": float(snapshot.fetched_at),
                "analysis_completed_at": float(self.clock()),
                "analysis_only": True,
                "decision_authority": "none",
                "execution_eligible": False,
            }
        except Exception as exc:
            return self._blocked(symbol, f"{type(exc).__name__}: candidate snapshot/analyzer rejected")

    @staticmethod
    def _blocked(symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "status": "BLOCKED",
            "scope": "part7-shadow-only",
            "coverage": [],
            "reason": reason[:180],
            "analysis_only": True,
            "decision_authority": "none",
            "execution_eligible": False,
        }

    def close(self) -> None:
        """Stop accepting diagnostics; running fetches are not cancelled mid-call."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
