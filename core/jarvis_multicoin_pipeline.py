"""Opt-in full Part1--Part12 analysis with fail-closed execution handoffs.

The selected-symbol Delta route and background analysis namespace are separate.
A caller must supply complete product identities and an isolated analysis owner.
Analysis outputs are never live-order eligible by themselves: the paper or
protected-Delta consumer must independently validate a complete plan, current
quotes, per-asset policy, risk, lot limits, and account state.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from direct_candle_cache import CandleDataError, CandleSnapshot, DirectCandleCache, LIVE_TIMEFRAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class InstrumentKey:
    venue: str
    market_type: str
    instrument_id: str
    symbol: str
    execution_identity: Optional[Tuple[str, str, str, str]] = None
    mapping_policy_id: Optional[str] = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "InstrumentKey":
        # Product metadata must be authoritative; no ticker-only venue aliases.
        values = [str(record.get(k) or "").strip() for k in ("venue", "market_type", "instrument_id", "symbol")]
        if any(not v for v in values):
            raise CandleDataError("candidate is missing venue/market/product/symbol identity")
        if values[0].lower() not in {"delta", "binance"}:
            raise CandleDataError("unsupported analysis venue")
        execution_raw = record.get("execution_identity")
        execution_identity = None
        if execution_raw is not None:
            if not isinstance(execution_raw, Mapping):
                raise CandleDataError("execution identity must be a complete mapping")
            exec_values = [str(execution_raw.get(k) or "").strip() for k in ("venue", "market_type", "instrument_id", "symbol")]
            if any(not v for v in exec_values) or exec_values[0].lower() != "delta":
                raise CandleDataError("mapped execution identity must be a full Delta product key")
            execution_identity = (exec_values[0].lower(), exec_values[1].lower(), exec_values[2], exec_values[3].upper())
        mapping_policy_id = str(record.get("mapping_policy_id") or "").strip() or None
        if execution_identity and not mapping_policy_id:
            raise CandleDataError("cross-venue product mapping policy ID is required")
        return cls(values[0].lower(), values[1].lower(), values[2], values[3].upper(),
                   execution_identity, mapping_policy_id)

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return self.venue, self.market_type, self.instrument_id, self.symbol

    def as_dict(self) -> Dict[str, str]:
        return {"venue": self.venue, "market_type": self.market_type,
                "instrument_id": self.instrument_id, "symbol": self.symbol}

    def mapped_execution_dict(self) -> Optional[Dict[str, str]]:
        if self.execution_identity is None:
            return None
        return dict(zip(("venue", "market_type", "instrument_id", "symbol"), self.execution_identity))


class DeltaNativeCandleClient:
    """Expose Delta-only candles to DirectCandleCache; never use Binance fallback."""
    def __init__(self, delta_client: Any, *, min_request_interval_seconds: float = 0.25,
                 clock: Callable[[], float] = time.monotonic, sleeper: Callable[[float], None] = time.sleep):
        self._client = delta_client
        self._interval = max(0.0, float(min_request_interval_seconds))
        self._clock = clock
        self._sleep = sleeper
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

    def get_historical_candles_with_metadata(self, *, symbol: str, resolution: str, limit: int) -> Mapping[str, Any]:
        method = getattr(self._client, "get_delta_native_candles_with_metadata", None)
        if not callable(method):
            raise CandleDataError("Delta-only native candle adapter unavailable")
        # Conservative process-local request spacing; actual provider quota is
        # deployment-specific and remains unverified until an authorized live check.
        with self._rate_lock:
            wait = self._next_request_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_request_at = self._clock() + self._interval
        result = method(symbol=symbol, resolution=resolution, limit=limit)
        if not isinstance(result, Mapping) or str(result.get("source", "")).lower() != "delta":
            raise CandleDataError("Delta-only candle response has unverified source")
        return result


class BinanceSpotCandleClient:
    """Strict Binance-spot adapter for an explicitly separate data pipeline.

    BinanceData has a Bybit fallback. This wrapper inspects that provider
    marker and rejects fallback data rather than attaching it to a Binance key.
    It never creates or modifies a Delta execution route.
    """
    def __init__(self, binance_client: Any, *, min_request_interval_seconds: float = 0.25,
                 clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep):
        self._client = binance_client
        self._interval = max(0.0, float(min_request_interval_seconds))
        self._clock = clock
        self._sleep = sleeper
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

    def get_historical_candles_with_metadata(self, *, symbol: str, resolution: str, limit: int) -> Mapping[str, Any]:
        method = getattr(self._client, "get_historical_candles_with_metadata", None)
        legacy_method = getattr(self._client, "get_historical_candles", None)
        if not callable(method) and not callable(legacy_method):
            raise CandleDataError("Binance-spot candle adapter unavailable")
        with self._rate_lock:
            wait = self._next_request_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_request_at = self._clock() + self._interval
        # Do not infer a cross-quote mapping from ticker spelling (e.g. USD to
        # USDT). The Delta-to-Spot mapping must be explicit and exact before
        # candles may be attached to a selected execution product.
        normalized_symbol = str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
        if not normalized_symbol.endswith("USDT") or normalized_symbol != str(symbol or "").upper():
            raise CandleDataError("explicit exact Binance Spot USDT symbol required")
        if callable(method):
            result = method(symbol=normalized_symbol, resolution=resolution, limit=limit)
            if not isinstance(result, Mapping):
                raise CandleDataError("Binance candle metadata response is malformed")
            candles = result.get("candles")
            source = str(result.get("source", "")).lower()
            market_type = str(result.get("market_type", "spot")).lower()
            result_symbol = str(result.get("symbol", "")).upper()
        else:
            try:
                candles = legacy_method(symbol=normalized_symbol, resolution=resolution,
                                        limit=limit, allow_fallback=False)
            except TypeError as exc:
                raise CandleDataError("Binance client does not support strict no-fallback candles") from exc
            source = "binance" if isinstance(candles, (list, tuple)) and candles else "unavailable"
            market_type = "spot"
            result_symbol = normalized_symbol
        if (not isinstance(candles, (list, tuple)) or not candles
                or source != "binance" or market_type != "spot"
                or result_symbol != normalized_symbol):
            raise CandleDataError("Binance provider source/market/symbol verification failed")
        if any(not isinstance(row, Mapping) or row.get("source") != "binance_spot"
               or row.get("market_semantics") != "spot" for row in candles):
            raise CandleDataError("Binance candle rows lack verified Spot provenance")
        return {"candles": candles, "source": "binance", "market_type": "spot",
                "symbol": normalized_symbol}


class MultiCoinPipeline:
    """Fair, bounded full-pipeline scheduler for every discovered eligible product.

    ``instrument_supplier`` is a real adapter callback that returns exact product
    records after venue and liquidity checks.  The cap is a hard resource budget;
    the admitted set is traversed round-robin so low-ranked eligible products
    are not starved.  Jobs are single-flight per full identity and results are
    stale/rejected fail-closed.  ``analyzer`` must be analysis-only.
    """
    def __init__(self, candle_cache: DirectCandleCache,
                 instrument_supplier: Callable[[], Iterable[Mapping[str, Any]]],
                 analyzer: Callable[[InstrumentKey, CandleSnapshot], Mapping[str, Any]], *,
                 enabled: bool = False, max_workers: int = 2, max_candidates: int = 200,
                 discovery_interval_seconds: float = 300.0,
                 retry_interval_seconds: float = 60.0,
                 result_max_age_seconds: float = 180.0,
                 clock: Callable[[], float] = time.time):
        self.enabled = bool(enabled)
        self.candle_cache = candle_cache
        self.instrument_supplier = instrument_supplier
        self.analyzer = analyzer
        self.max_workers = max(1, min(4, int(max_workers)))
        # Hard bound chosen for memory/resource safety, not an exchange coverage claim.
        self.max_candidates = max(1, min(500, int(max_candidates)))
        self.discovery_interval_seconds = max(30.0, float(discovery_interval_seconds))
        self.retry_interval_seconds = max(10.0, float(retry_interval_seconds))
        self.result_max_age_seconds = max(10.0, float(result_max_age_seconds))
        self.clock = clock
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="JarvisMultiCoin")
        self._lock = threading.RLock()
        self._candidates: Tuple[InstrumentKey, ...] = ()
        self._all_candidates: Tuple[InstrumentKey, ...] = ()
        self._discovery_offset = 0
        self._overflow_count = 0
        self._ambiguous_symbol_count = 0
        self._cursor = 0
        self._next_discovery = 0.0
        self._last_attempt: Dict[InstrumentKey, float] = {}
        self._last_complete_at: Dict[InstrumentKey, float] = {}
        self._results: Dict[InstrumentKey, Dict[str, Any]] = {}
        self._inflight: Dict[InstrumentKey, Future] = {}
        self._closed = False
        self._scheduler_stop = threading.Event()
        self._scheduler_thread = None
        if self.enabled:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop, name="JarvisMultiCoinScheduler", daemon=True
            )
            self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        while not self._scheduler_stop.is_set():
            try:
                self._dispatch()
            except RuntimeError as exc:
                if "interpreter shutdown" in str(exc):
                    break
                logger.warning("[MULTICOIN] Scheduler cycle rejected: %r", exc)
            except Exception as exc:
                logger.warning("[MULTICOIN] Scheduler cycle rejected: %r", exc)
            self._scheduler_stop.wait(0.25)

    def _refresh_candidates(self, now: float) -> None:
        if now < self._next_discovery:
            return
        self._next_discovery = now + self.discovery_interval_seconds
        try:
            by_key: Dict[InstrumentKey, InstrumentKey] = {}
            for item in self.instrument_supplier() or ():
                try:
                    key = InstrumentKey.from_record(item)
                except (CandleDataError, AttributeError, TypeError):
                    continue
                by_key[key] = key
            discovered = list(by_key)  # preserve supplier liquidity rank
            # Candle adapters are addressed by symbol rather than product ID.
            # If a venue lists the same symbol under multiple contract/product
            # identities, that endpoint cannot disambiguate safely: exclude all
            # such records instead of assigning one ticker's data to a contract.
            symbols: Dict[Tuple[str, str], set] = {}
            for item in discovered:
                symbols.setdefault((item.venue, item.symbol), set()).add(
                    (item.market_type, item.instrument_id, item.execution_identity)
                )
            ambiguous = {pair for pair, identities in symbols.items() if len(identities) > 1}
            self._ambiguous_symbol_count = len(ambiguous)
            discovered = [item for item in discovered if (item.venue, item.symbol) not in ambiguous]
            # Keep all discovered products in a bounded 5k work universe; the
            # scheduler round-robins every member. ``max_candidates`` bounds
            # result retention, not a top-N-only analysis shortlist.
            universe = discovered[:5000]
            self._all_candidates = tuple(universe)
            self._candidates = self._all_candidates
            self._overflow_count = max(0, len(discovered) - len(universe))
            self._last_attempt = {k: v for k, v in self._last_attempt.items() if k in set(universe)}
            self._last_complete_at = {k: v for k, v in self._last_complete_at.items() if k in set(universe)}
            self._cursor = self._cursor % len(universe) if universe else 0
        except Exception as exc:
            logger.warning("[MULTICOIN] Product discovery failed: %s", type(exc).__name__)
            self._candidates = ()

    def _collect(self) -> None:
        done = [(key, future) for key, future in self._inflight.items() if future.done()]
        for key, future in done:
            self._inflight.pop(key, None)
            try:
                self._results[key] = future.result()
            except Exception as exc:
                self._results[key] = self._blocked(key, f"worker error: {type(exc).__name__}")
            if self._results[key].get('status') == 'COMPLETE':
                self._last_complete_at[key] = float(self._results[key].get('analysis_completed_at', self.clock()))
        if len(self._results) > self.max_candidates:
            keep = sorted(
                self._results,
                key=lambda key: max(
                    float(self._last_complete_at.get(key, 0.0)),
                    float(self._last_attempt.get(key, 0.0)),
                ),
                reverse=True,
            )[:self.max_candidates]
            self._results = {key: self._results[key] for key in keep}

    def _dispatch(self) -> None:
        if not self.enabled or self._closed:
            return
        now = float(self.clock())
        with self._lock:
            self._collect()
            self._refresh_candidates(now)
            candidates = self._candidates
            if not candidates:
                return
            count = len(candidates)
            for offset in range(count):
                if len(self._inflight) >= self.max_workers:
                    break
                idx = (self._cursor + offset) % count
                key = candidates[idx]
                if key in self._inflight:
                    continue
                complete_at = self._last_complete_at.get(key, 0.0)
                if 0 <= now - complete_at <= self.result_max_age_seconds and complete_at > 0:
                    continue
                if now - self._last_attempt.get(key, 0.0) < self.retry_interval_seconds:
                    continue
                self._last_attempt[key] = now
                self._inflight[key] = self._executor.submit(self._run, key)
                self._cursor = (idx + 1) % count

    def poll(self) -> Dict[str, Any]:
        """Return a bounded status view; scheduling is independent of live route cycles."""
        if not self.enabled or self._closed:
            return {"enabled": False, "status": "DISABLED", "results": {}}
        now = float(self.clock())
        with self._lock:
            candidates = self._candidates
            result_view: Dict[str, Dict[str, Any]] = {}
            for key, raw in self._results.items():
                view = dict(raw)
                fetched = view.get("snapshot_fetched_at")
                age = float("inf") if fetched is None else now - float(fetched)
                fresh = 0 <= age <= self.result_max_age_seconds
                view["freshness_status"] = "FRESH" if fresh else "STALE"
                view["age_seconds"] = None if fetched is None else max(0.0, age)
                if not fresh:
                    view["status"] = "STALE" if view.get("status") == "COMPLETE" else view.get("status", "BLOCKED")
                    view["execution_eligible"] = False
                result_view["|".join(key.as_tuple())] = view
            return {
                "enabled": True, "status": "RUNNING" if self._inflight else "IDLE",
                "eligible_count": len(candidates), "discovered_universe_count": len(self._all_candidates),
                "over_budget_count": self._overflow_count,
                "ambiguous_symbol_count": self._ambiguous_symbol_count,
                "inflight_count": len(self._inflight), "max_workers": self.max_workers,
                "candidate_cursor": self._cursor, "results": result_view,
            }

    def _run(self, key: InstrumentKey) -> Dict[str, Any]:
        try:
            snapshot = self.candle_cache.refresh(
                key.symbol, venue=key.venue, market_type=key.market_type,
                instrument_id=key.instrument_id,
            )
            if snapshot.identity != key.as_tuple() or snapshot.symbol != key.symbol:
                raise CandleDataError("snapshot full identity mismatch")
            required_timeframes = tuple(getattr(self.candle_cache, "timeframes", LIVE_TIMEFRAMES))
            if set(snapshot.frames) != set(required_timeframes):
                raise CandleDataError("incomplete native timeframe snapshot")
            # For this Delta adapter no Binance/other venue fallback is allowed.
            for tf, frame in snapshot.frames.items():
                if frame.venue != key.venue or frame.instrument_id != key.instrument_id:
                    raise CandleDataError(f"{tf}: frame identity mismatch")
                if frame.source.lower() != key.venue:
                    raise CandleDataError(f"{tf}: candle provider does not match venue identity")
                required = int(getattr(self.candle_cache, "history_limits", {}).get(tf, 500))
                if len(frame.closed) != required or frame.current is None:
                    raise CandleDataError(f"{tf}: expected {required} closed plus one forming candle")
            now = float(self.clock())
            age = now - float(snapshot.fetched_at)
            if age < -2 or age > self.result_max_age_seconds:
                raise CandleDataError("snapshot stale or from future")
            raw = self.analyzer(key, snapshot)
            if not isinstance(raw, Mapping):
                raise CandleDataError("pipeline analyzer returned no mapping")
            completed = float(self.clock())
            if completed - float(snapshot.fetched_at) > self.result_max_age_seconds:
                raise CandleDataError("analysis exceeded maximum result age")
            parts_by_tf = raw.get("parts_by_timeframe")

            from jarvis_strategy_approval import PART_WEIGHTS
            canonical_parts = set(PART_WEIGHTS)
            expected_adapter_parts = {f"part{i}" for i in range(1, 11)}
            complete = (isinstance(parts_by_tf, Mapping)
                     and set(parts_by_tf) == set(required_timeframes)
                     and all(isinstance(parts_by_tf[tf], Mapping)
                             and (canonical_parts.issubset(parts_by_tf[tf])
                                  or expected_adapter_parts.issubset(parts_by_tf[tf]))
                             for tf in required_timeframes)
                     and set(raw.get("once_per_symbol_parts", ())) == {"part11", "part12"})

            versions = {tf: {
                "source": snapshot.frames[tf].source,
                "last_closed": str(snapshot.frames[tf].closed.index[-1]),
                "forming_time": snapshot.frames[tf].current.get("time"),
            } for tf in required_timeframes}
            version = hashlib.sha256(json.dumps({"identity": key.as_tuple(), "frames": versions}, sort_keys=True).encode()).hexdigest()[:24]
            return {
                "symbol": key.symbol, "request_identity": key.as_dict(),
                "execution_identity": key.mapped_execution_dict(),
                "mapping_policy_id": key.mapping_policy_id,
                "status": "COMPLETE" if complete else "PARTIAL",
                "scope": "parts1-12-analysis-only", "coverage": [f"Part{i}" for i in range(1, 13)],
                "parts_by_timeframe": parts_by_tf if isinstance(parts_by_tf, Mapping) else {},
                "once_per_symbol_parts": raw.get("once_per_symbol_parts", {}),
                "part7_gate": raw.get("part7_gate"),
                "central_strategy_decision": raw.get("central_strategy_decision"),
                "central_strategy_evidence": raw.get("central_strategy_evidence"),
                "central_strategy_approval": raw.get("central_strategy_approval"),
                "central_execution_plan": dict(raw["central_execution_plan"])
                    if isinstance(raw.get("central_execution_plan"), Mapping) else None,
                "deterministic_decision": raw.get("deterministic_decision"),
                "analysis_reference": raw.get("analysis_reference"),
                "part14": "not_in_scope",
                "snapshot_version": version, "snapshot_fetched_at": float(snapshot.fetched_at),
                "analysis_completed_at": completed,
                # The only plan passed here is the Jarvis price-level plan.
                # The Delta bridge replaces it with a broker plan only after
                # fresh exact product/quote/balance and risk checks.
                "execution_plan": dict(raw["execution_plan"])
                    if isinstance(raw.get("execution_plan"), Mapping) else None,
                "execution_candidate_status": "PLAN_PRESENT_REQUIRES_DELTA_VALIDATION" if isinstance(raw.get("execution_plan"), Mapping) else "BLOCKED_MISSING_EXPLICIT_PLAN",
                "analysis_only": True,
                # Pipeline snapshots are always analysis-only.  The nested
                # Jarvis decision/approval remain auditable evidence and are
                # independently recomputed by each consumer; they do not turn
                # this raw Parts result into execution-authorized output.
                "decision_authority": "none",
                "execution_eligible": False,
                "reason": None if complete else "one or more Part adapters did not complete for all native frames",
            }
        except Exception as exc:
            return self._blocked(key, f"{type(exc).__name__}: {str(exc)[:120]}")

    @staticmethod
    def _blocked(key: InstrumentKey, reason: str) -> Dict[str, Any]:
        return {
            "symbol": key.symbol, "request_identity": key.as_dict(),
            "execution_identity": key.mapped_execution_dict(), "mapping_policy_id": key.mapping_policy_id,
            "status": "BLOCKED", "scope": "parts1-12-analysis-only", "coverage": [], "reason": reason[:180],
            "analysis_only": True, "decision_authority": "none", "execution_eligible": False,
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._scheduler_stop.set()
        if self._scheduler_thread is not None and self._scheduler_thread is not threading.current_thread():
            self._scheduler_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)
