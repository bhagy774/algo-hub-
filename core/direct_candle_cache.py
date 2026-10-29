"""Direct native-interval candle snapshots for live Jarvis analysis.

This module deliberately does not resample, pad, or fabricate candles. A live
snapshot contains a configured, timeframe-specific number of completed native
candles and one separate forming candle. After warm-up, refresh requests a short
native delta, merges only contiguous updates, rejects revisions to previously
closed bars, and falls back to a full window when continuity is uncertain.
Consumers must use ``closed`` for indicators; ``current`` is unconfirmed metadata.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, RLock
import math
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

from binance_timeframes import (
    BINANCE_SPOT_TIMEFRAMES, DEFAULT_STRATEGY_TIMEFRAMES,
    TIMEFRAME_HISTORY_CANDLES, candle_is_closed, history_limit, is_aligned_open,
    candle_open_time, next_candle_open, validate_interval,
)

# All provider-native Binance Spot intervals are part of the live MTF snapshot.
# Backtests may explicitly retain the smaller legacy strategy set when only
# 1m source history is available; the live path must not silently do that.
LIVE_TIMEFRAMES: Tuple[str, ...] = BINANCE_SPOT_TIMEFRAMES
INTERVAL_SECONDS: Mapping[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
    "3d": 259200, "1w": 604800,
}
REQUIRED_CLOSED_CANDLES = 500  # compatibility default; configured per frame below
FETCH_CANDLES = REQUIRED_CLOSED_CANDLES + 1


class CandleDataError(ValueError):
    """A live candle snapshot cannot be trusted for decision-making."""


@dataclass(frozen=True)
class CandleFrame:
    symbol: str
    source: str
    timeframe: str
    closed: pd.DataFrame
    current: Optional[Dict[str, Any]]
    fetched_at: float
    venue: str = "unknown"
    market_type: str = "unknown"
    instrument_id: str = ""

    @property
    def current_is_confirmed(self) -> bool:
        return False

    def sma(self, period: int) -> float:
        """Calculate a latest-window SMA from completed candles only.

        ``JARVIS_RUST_MATH=1`` opts this pure calculation into the Rust
        extension; the adapter provides an explicit equivalent Python fallback.
        The forming candle is intentionally never included.
        """
        from jarvis_rust_integration import calculate_sma

        return calculate_sma(self.closed["close"].tolist(), period)


@dataclass(frozen=True)
class CandleSnapshot:
    symbol: str
    frames: Mapping[str, CandleFrame]
    fetched_at: float
    venue: str = "unknown"
    market_type: str = "unknown"
    instrument_id: str = ""

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        """Request identity; provider source remains explicit per timeframe."""
        return (self.venue, self.market_type, self.instrument_id, self.symbol)

    @property
    def current_candles(self) -> Dict[str, Optional[Dict[str, Any]]]:
        return {tf: frame.current for tf, frame in self.frames.items()}

    def analysis_frames(self) -> Dict[str, pd.DataFrame]:
        """Only completed candles, suitable for Parts 1-12 and native engines."""
        return {tf: frame.closed.copy() for tf, frame in self.frames.items()}

    def context(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "venue": self.venue,
            "market_type": self.market_type,
            "instrument_id": self.instrument_id,
            "identity": self.identity,
            "source_by_timeframe": {tf: f.source for tf, f in self.frames.items()},
            "current_candles": self.current_candles,
            "current_is_confirmed": False,
            "fetched_at": self.fetched_at,
        }


class DirectCandleCache:
    """Bounded, full-request-identity/native-timeframe candle cache.

    A snapshot request is isolated by (venue, market_type, instrument_id,
    canonical_symbol). ``source`` is still recorded for every timeframe because
    a multi-provider adapter can legitimately fall back between providers.
    Per-identity locks let analysis jobs validate different markets without
    sharing mutable snapshot state. Completed idle identities are evicted LRU
    when the cache budget is reached; active identities are never evicted. A
    separate bounded semaphore protects a shared client from excessive
    simultaneous network calls.
    """

    def __init__(
        self,
        client: Any,
        *,
        clock: Callable[[], float] = time.time,
        ttl_seconds: float = 45.0,
        stale_intervals: float = 2.0,
        max_concurrent_fetches: int = 1,
        max_cached_identities: int = 64,
        timeframes: Tuple[str, ...] = LIVE_TIMEFRAMES,
        history_limits: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.client = client
        self.timeframes = tuple(timeframes)
        if not self.timeframes or len(set(self.timeframes)) != len(self.timeframes):
            raise CandleDataError("timeframes must be a non-empty unique sequence")
        try:
            for _tf in self.timeframes:
                validate_interval(_tf)
            # Use the reviewed per-timeframe profile unless a caller explicitly
            # overrides individual frames. Unspecified entries keep their profile.
            self.history_limits = {tf: history_limit(tf, history_limits) for tf in self.timeframes}
        except ValueError as exc:
            raise CandleDataError(str(exc)) from exc
        self.clock = clock
        self.ttl_seconds = float(ttl_seconds)
        self.stale_intervals = float(stale_intervals)
        self.max_cached_identities = max(1, min(256, int(max_cached_identities)))
        self._cache: Dict[Tuple[Tuple[str, str, str, str], str], CandleFrame] = {}
        self._snapshots: Dict[Tuple[str, str, str, str], CandleSnapshot] = {}
        self._identity_locks: Dict[Tuple[str, str, str, str], RLock] = {}
        self._identity_users: Dict[Tuple[str, str, str, str], int] = {}
        self._lock = RLock()
        self._fetch_semaphore = BoundedSemaphore(max(1, min(4, int(max_concurrent_fetches))))

    def _identity_key(
        self, symbol: str, venue: str, market_type: str, instrument_id: Optional[str]
    ) -> Tuple[str, str, str, str]:
        canonical = self._canonical_symbol(symbol)
        venue_key = str(venue or "unknown").strip().lower()
        market_key = str(market_type or "unknown").strip().lower()
        instrument_key = str(instrument_id or canonical).strip()
        if not venue_key or not market_key or not instrument_key:
            raise CandleDataError("venue, market type, and instrument identity are required")
        return venue_key, market_key, instrument_key, canonical

    def _identity_lock(self, key: Tuple[str, str, str, str]) -> RLock:
        """Reserve a per-key lock, evicting only inactive LRU identities."""
        with self._lock:
            lock = self._identity_locks.get(key)
            if lock is None:
                if len(self._identity_locks) >= self.max_cached_identities:
                    idle = [candidate for candidate in self._identity_locks
                            if self._identity_users.get(candidate, 0) == 0]
                    if not idle:
                        raise CandleDataError("bounded candle identity capacity reached")
                    victim = min(idle, key=lambda candidate:
                                 self._snapshots[candidate].fetched_at if candidate in self._snapshots else float("-inf"))
                    self._identity_locks.pop(victim, None)
                    self._identity_users.pop(victim, None)
                    self._snapshots.pop(victim, None)
                    for frame_key in [item for item in self._cache if item[0] == victim]:
                        self._cache.pop(frame_key, None)
                lock = RLock()
                self._identity_locks[key] = lock
            self._identity_users[key] = self._identity_users.get(key, 0) + 1
            return lock

    @contextmanager
    def _identity_access(self, key: Tuple[str, str, str, str]):
        lock = self._identity_lock(key)
        try:
            with lock:
                yield
        finally:
            with self._lock:
                users = self._identity_users.get(key, 0)
                if users <= 1:
                    self._identity_users.pop(key, None)
                else:
                    self._identity_users[key] = users - 1

    @staticmethod
    def _canonical_symbol(symbol: str) -> str:
        value = str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
        if not value:
            raise CandleDataError("empty symbol")
        return value

    @staticmethod
    def _row_value(row: Mapping[str, Any], name: str) -> Any:
        if name in row:
            return row[name]
        # Some native clients expose milliseconds under timestamp.
        if name == "time" and "timestamp" in row:
            return row["timestamp"]
        raise CandleDataError(f"missing candle field: {name}")

    def _fetch(self, symbol: str, timeframe: str, limit: int = FETCH_CANDLES) -> Tuple[Iterable[Mapping[str, Any]], str, str]:
        # The client may hold mutable source metadata on itself; serialize the
        # complete request and metadata read unless explicitly configured for a
        # small bounded amount of concurrency.
        with self._fetch_semaphore:
            return self._fetch_unlocked(symbol, timeframe, limit)

    def _fetch_unlocked(self, symbol: str, timeframe: str, limit: int = FETCH_CANDLES) -> Tuple[Iterable[Mapping[str, Any]], str, str]:
        method = getattr(self.client, "get_historical_candles_with_metadata", None)
        if callable(method):
            result = method(symbol=symbol, resolution=timeframe, limit=limit)
            if not isinstance(result, Mapping):
                raise CandleDataError("candle metadata response is not a mapping")
            candles = result.get("candles")
            source = str(result.get("source") or "unknown").strip().lower()
            instrument = str(result.get("symbol") or result.get("instrument") or symbol)
        else:
            method = getattr(self.client, "get_historical_candles", None)
            if not callable(method):
                raise CandleDataError("client has no historical candle method")
            candles = method(symbol=symbol, resolution=timeframe, limit=limit)
            source = str(getattr(self.client, "last_candle_source", None) or "provided").strip().lower()
            instrument = symbol
        if not isinstance(candles, Iterable) or isinstance(candles, (str, bytes)):
            raise CandleDataError(f"{timeframe}: invalid candle response")
        return candles, source, instrument

    def _normalize(
        self,
        symbol: str,
        timeframe: str,
        candles: Iterable[Mapping[str, Any]],
        source: str,
        instrument: str,
        now: float,
        venue: str = "unknown",
        market_type: str = "unknown",
        instrument_id: str = "",
        required_closed_candles: Optional[int] = None,
    ) -> CandleFrame:
        try:
            validate_interval(timeframe)
        except ValueError as exc:
            raise CandleDataError(str(exc)) from exc
        required_closed = int(self.history_limits.get(timeframe, REQUIRED_CLOSED_CANDLES) if required_closed_candles is None else required_closed_candles)
        if required_closed < 20:
            raise CandleDataError(f"{timeframe}: closed-candle history must be at least 20")
        if self._canonical_symbol(instrument) != self._canonical_symbol(symbol):
            raise CandleDataError(f"{timeframe}: instrument mismatch ({instrument!r} != {symbol!r})")
        source = source or "unknown"
        rows = list(candles)
        if len(rows) < required_closed + 1:
            raise CandleDataError(f"{timeframe}: need at least {required_closed + 1} rows, got {len(rows)}")
        normalized = []
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise CandleDataError(f"{timeframe}: malformed candle row")
            raw_time = self._row_value(row, "time")
            try:
                ts = float(raw_time)
                # Accept millisecond provider timestamps only when they denote
                # an exact whole-second candle open. Never truncate malformed
                # fractional-second opens into a valid aligned timestamp.
                if ts > 10_000_000_000:
                    ts /= 1000.0
                if not math.isfinite(ts) or not ts.is_integer():
                    raise ValueError("candle open timestamp must be an exact whole second")
                ts = int(ts)
                values = {field: float(self._row_value(row, field)) for field in ("open", "high", "low", "close", "volume")}
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandleDataError(f"{timeframe}: invalid numeric candle ({exc})") from exc
            if ts <= 0 or any(not math.isfinite(v) for v in values.values()):
                raise CandleDataError(f"{timeframe}: invalid numeric candle")
            if not is_aligned_open(timeframe, ts):
                raise CandleDataError(f"{timeframe}: candle open is not aligned to native interval")
            if any(values[name] <= 0 for name in ("open", "high", "low", "close")) or values["volume"] < 0:
                raise CandleDataError(f"{timeframe}: non-positive price or negative volume")
            if ts in seen:
                raise CandleDataError(f"{timeframe}: duplicate timestamp {ts}")
            if ts > now + 2:
                raise CandleDataError(f"{timeframe}: future candle {ts}")
            if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]):
                raise CandleDataError(f"{timeframe}: OHLC bounds invalid")
            seen.add(ts)
            normalized.append({"time": ts, **values})
        normalized.sort(key=lambda row: row["time"])
        closed_rows = [row for row in normalized if candle_is_closed(timeframe, row["time"], now)]
        forming_rows = [row for row in normalized if row["time"] <= now < next_candle_open(timeframe, row["time"])]
        if len(forming_rows) != 1:
            if timeframe == "1s" and not forming_rows and normalized and (now - normalized[-1]["time"] <= 15.0):
                forming_rows = [normalized[-1]]
            elif (timeframe != "1s" and not forming_rows and normalized and closed_rows
                  and normalized[-1]["time"] == closed_rows[-1]["time"]
                  and 0.0 <= now - next_candle_open(timeframe, normalized[-1]["time"]) <= 2.5):
                # Boundary rollover: bar closed <=2.5s ago, new bar has no tick yet. Fresh data only.
                _lc = closed_rows[-1]["close"]
                forming_rows = [{"time": next_candle_open(timeframe, normalized[-1]["time"]), "open": _lc, "high": _lc, "low": _lc, "close": _lc, "volume": 0.0}]
                normalized.append(forming_rows[0])
                import logging as _lg
                _lg.getLogger(__name__).info("[CANDLE_BOUNDARY_ROLLOVER] %s just-closed bar, synthetic empty forming bar (age %.2fs)", timeframe, now - forming_rows[0]["time"])
            else:
                raise CandleDataError(f"{timeframe}: expected one current forming candle, got {len(forming_rows)}")
        current = forming_rows[0]
        prior_closed = [row for row in closed_rows if row["time"] < current["time"]]
        if len(prior_closed) < required_closed:
            raise CandleDataError(f"{timeframe}: need {required_closed} closed candles, got {len(prior_closed)}")
        prior_closed = prior_closed[-required_closed:]
        expected = prior_closed[0]["time"]
        for row in prior_closed:
            if row["time"] != expected:
                raise CandleDataError(f"{timeframe}: missing/non-contiguous closed candle at {expected}")
            expected = next_candle_open(timeframe, expected)
        if current["time"] != expected:
            raise CandleDataError(f"{timeframe}: current candle is not after the closed window")
        # A forming row must be fresh, not a stale exchange response.
        max_stale = 15.0 if timeframe == "1s" else 0.0
        if next_candle_open(timeframe, current["time"]) + max_stale <= now:
            raise CandleDataError(f"{timeframe}: current candle is stale")
        frame = pd.DataFrame(prior_closed, columns=["time", "open", "high", "low", "close", "volume"])
        frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
        frame.index.name = "timestamp"
        frame.attrs.update({
            "symbol": self._canonical_symbol(symbol),
            "timeframe": timeframe,
            "source": source,
            "venue": venue,
            "market_type": market_type,
            "instrument_id": instrument_id or symbol,
        })
        return CandleFrame(
            symbol=symbol, source=source, timeframe=timeframe, closed=frame,
            current=current, fetched_at=now, venue=venue,
            market_type=market_type, instrument_id=instrument_id or symbol,
        )

    def _merge_incremental_rows(
        self,
        previous: CandleFrame,
        new_rows: Iterable[Mapping[str, Any]],
        *,
        symbol: str,
        timeframe: str,
        source: str,
        instrument: str,
        now: float,
        venue: str,
        market_type: str,
        instrument_id: str,
        required_closed_candles: int = REQUIRED_CLOSED_CANDLES,
    ) -> CandleFrame:
        """Merge a short native update into the bounded rolling frame.

        Previously closed candles are immutable: if the provider revises one,
        reject the update instead of silently rewriting analyzed history. The
        prior forming candle may be replaced when it closes. Gaps/short updates
        are rejected here so ``refresh`` can request a full window as recovery.
        """
        if source != previous.source:
            raise CandleDataError("provider source changed during incremental update")
        merged: Dict[int, Dict[str, Any]] = {}
        old_closed: Dict[int, Dict[str, float]] = {}
        for stamp, row in previous.closed.iterrows():
            ts = int(pd.Timestamp(stamp).timestamp())
            values = {name: float(row[name]) for name in ("open", "high", "low", "close", "volume")}
            old_closed[ts] = values
            merged[ts] = {"time": ts, **values}
        if previous.current is not None:
            current = dict(previous.current)
            merged[int(current["time"])] = current
        incoming_seen = set()
        for row in new_rows:
            if not isinstance(row, Mapping):
                raise CandleDataError(f"{timeframe}: malformed incremental candle row")
            raw_time = self._row_value(row, "time")
            try:
                ts = float(raw_time)
                if ts > 10_000_000_000:
                    ts /= 1000.0
                if not math.isfinite(ts) or not ts.is_integer():
                    raise ValueError("candle open timestamp must be an exact whole second")
                ts = int(ts)
                values = {field: float(self._row_value(row, field)) for field in ("open", "high", "low", "close", "volume")}
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandleDataError(f"{timeframe}: non-numeric incremental candle") from exc
            if ts in incoming_seen:
                raise CandleDataError(f"{timeframe}: duplicate incremental timestamp {ts}")
            incoming_seen.add(ts)
            if ts in old_closed:
                if any(not math.isclose(values[k], old_closed[ts][k], rel_tol=0.0, abs_tol=1e-12) for k in values):
                    raise CandleDataError(f"{timeframe}: previously closed candle revision")
                # Identical overlap is expected and not a duplicate update.
                continue
            merged[ts] = {"time": ts, **values}
        combined = [merged[ts] for ts in sorted(merged)]
        combined = combined[-(required_closed_candles + 1):]
        return self._normalize(
            symbol, timeframe, combined, source, instrument, now,
            venue=venue, market_type=market_type, instrument_id=instrument_id,
            required_closed_candles=required_closed_candles,
        )

    def _incremental_request_limit(self, timeframe: str, previous: CandleFrame, now: float) -> int:
        """Fetch enough recent native bars to bridge elapsed time without gaps.

        The previous forming bar may have closed since the last refresh. A fixed
        three-row delta silently loses short bars whenever polling is slower than
        a few seconds; size the delta from native bar boundaries instead.
        """
        max_rows = self.history_limits[timeframe] + 1
        try:
            previous_open = int((previous.current or {})["time"])
            current_open = candle_open_time(timeframe, now)
            cursor = previous_open
            steps = 0
            while cursor < current_open and steps < max_rows:
                cursor = next_candle_open(timeframe, cursor)
                steps += 1
            if cursor != current_open:
                return max_rows
            # Two extra bars cover the previous forming bar and the new current
            # bar; a minimum of three also handles same-bar refreshes safely.
            return min(max_rows, max(3, steps + 2))
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            return max_rows

    def refresh(
        self,
        symbol: str,
        *,
        force: bool = False,
        venue: str = "unknown",
        market_type: str = "unknown",
        instrument_id: Optional[str] = None,
    ) -> CandleSnapshot:
        identity = self._identity_key(symbol, venue, market_type, instrument_id)
        venue_key, market_key, instrument_key, symbol_key = identity
        with self._identity_access(identity):
            now = float(self.clock())
            with self._lock:
                cached = self._snapshots.get(identity)
            if not force and cached and now - cached.fetched_at < self.ttl_seconds:
                return cached
            frames: Dict[str, CandleFrame] = {}
            for timeframe in self.timeframes:
                request_limit = self.history_limits[timeframe] + 1
                with self._lock:
                    previous = self._cache.get((identity, timeframe))
                if previous is not None:
                    request_limit = self._incremental_request_limit(timeframe, previous, now)
                candles, source, instrument = self._fetch(symbol_key, timeframe, request_limit)
                # Validate freshness at the time each response arrives rather
                # than anchoring all eight sequential requests to cycle start.
                frame_now = float(self.clock())
                if previous is None:
                    frame = self._normalize(
                        symbol_key, timeframe, candles, source, instrument, frame_now,
                        venue=venue_key, market_type=market_key,
                        instrument_id=instrument_key, required_closed_candles=self.history_limits[timeframe],
                    )
                else:
                    rows = list(candles)
                    try:
                        frame = self._merge_incremental_rows(
                            previous, rows, symbol=symbol_key, timeframe=timeframe,
                            source=source, instrument=instrument, now=frame_now,
                            venue=venue_key, market_type=market_key,
                            instrument_id=instrument_key, required_closed_candles=self.history_limits[timeframe],
                        )
                    except CandleDataError as incremental_error:
                        # A long pause, source switch, malformed/short delta, detected gap, or
                        # closed-bar settlement revision gets one bounded full-window recovery
                        # normalized cleanly from the authoritative source.
                        import logging as _lg
                        _lg.getLogger(__name__).info(
                            "[CANDLE_CACHE_RECONCILE] %s %s incremental merge rejected (%s); refreshing full authoritative window",
                            symbol_key, timeframe, incremental_error,
                        )
                        full_rows, full_source, full_instrument = self._fetch(
                            symbol_key, timeframe, self.history_limits[timeframe] + 1
                        )
                        full_now = float(self.clock())
                        if full_source == previous.source and "previously closed candle revision" not in str(incremental_error):
                            frame = self._merge_incremental_rows(
                                previous, full_rows, symbol=symbol_key, timeframe=timeframe,
                                source=full_source, instrument=full_instrument, now=full_now,
                                venue=venue_key, market_type=market_key,
                                instrument_id=instrument_key, required_closed_candles=self.history_limits[timeframe],
                            )
                        else:
                            frame = self._normalize(
                                symbol_key, timeframe, full_rows, full_source,
                                full_instrument, full_now, venue=venue_key,
                                market_type=market_key, instrument_id=instrument_key,
                                required_closed_candles=self.history_limits[timeframe],
                            )
                with self._lock:
                    # One current provider frame per request identity/timeframe;
                    # provider provenance remains attached to that frame.
                    self._cache[(identity, timeframe)] = frame
                frames[timeframe] = frame
            completed_at = float(self.clock())
            snapshot = CandleSnapshot(
                symbol=symbol_key, frames=frames, fetched_at=completed_at,
                venue=venue_key, market_type=market_key,
                instrument_id=instrument_key,
            )
            with self._lock:
                self._snapshots[identity] = snapshot
            return snapshot

    def get_snapshot(
        self,
        symbol: str,
        *,
        venue: str = "unknown",
        market_type: str = "unknown",
        instrument_id: Optional[str] = None,
    ) -> Optional[CandleSnapshot]:
        identity = self._identity_key(symbol, venue, market_type, instrument_id)
        with self._lock:
            return self._snapshots.get(identity)

    def clear_symbol(self, symbol: str) -> None:
        symbol = self._canonical_symbol(symbol)
        with self._lock:
            identities = [key for key in self._identity_locks
                          if key[3] == symbol and self._identity_users.get(key, 0) == 0]
            for identity in identities:
                self._snapshots.pop(identity, None)
                self._identity_locks.pop(identity, None)
                self._identity_users.pop(identity, None)
                for key in [frame_key for frame_key in self._cache if frame_key[0] == identity]:
                    self._cache.pop(key, None)
