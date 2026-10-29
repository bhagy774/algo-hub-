"""Canonical Binance Spot candle intervals and bounded history policy.

These intervals describe data availability, not automatic strategy votes.
Consumers must opt into a reviewed timeframe set. Counts are initial history
windows sized by timeframe; validate them against indicator warm-up and native
backtests before changing strategy participation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Mapping

BINANCE_SPOT_TIMEFRAMES = (
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "8h", "12h", "1d", "3d", "1w", "1M",
)

# Existing strategy frames are kept distinct from the full provider catalogue.
DEFAULT_STRATEGY_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h")

# Closed bars only. The extra live/forming bar is fetched separately by callers.
# Short frames receive deeper intraday warm-up; very long frames avoid asking
# Binance for decades of monthly/weekly history.
TIMEFRAME_HISTORY_CANDLES: Mapping[str, int] = {
    "1m": 2000, "3m": 1500, "5m": 1000, "15m": 750,
    "30m": 500, "1h": 500, "2h": 500, "4h": 500, "6h": 365,
    "8h": 365, "12h": 365, "1d": 365, "3d": 180, "1w": 104, "1M": 60,
}

_FIXED_SECONDS: Mapping[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
    "3d": 259200, "1w": 604800,
}

def validate_interval(timeframe: str) -> str:
    if timeframe not in BINANCE_SPOT_TIMEFRAMES:
        raise ValueError(f"unsupported Binance Spot interval: {timeframe!r}")
    return timeframe

def candle_open_time(timeframe: str, timestamp: float) -> int:
    """Return the native UTC candle open containing ``timestamp``."""
    validate_interval(timeframe)
    ts = int(timestamp)
    if timeframe == "1M":
        dt = datetime.fromtimestamp(ts, timezone.utc)
        return int(datetime(dt.year, dt.month, 1, tzinfo=timezone.utc).timestamp())
    if timeframe == "1w":
        dt = datetime.fromtimestamp(ts, timezone.utc)
        midnight = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        monday = midnight.timestamp() - dt.weekday() * 86400
        return int(monday)
    width = _FIXED_SECONDS[timeframe]
    # Binance 3d bars are anchored one day after the epoch (verified live: open % 259200 == 86400).
    anchor = 86400 if timeframe == '3d' else 0
    return ts - ((ts - anchor) % width)

def is_aligned_open(timeframe: str, open_time: float) -> bool:
    """Whether a timestamp is exactly on the provider's native bar boundary."""
    try:
        ts = int(open_time)
        return float(open_time) == ts and candle_open_time(timeframe, ts) == ts
    except (TypeError, ValueError, OverflowError, OSError):
        return False

def next_candle_open(timeframe: str, open_time: float) -> int:
    """Return the next native candle's UTC open timestamp."""
    validate_interval(timeframe)
    if timeframe != "1M":
        return int(open_time) + _FIXED_SECONDS[timeframe]
    dt = datetime.fromtimestamp(open_time, timezone.utc)
    year, month = dt.year, dt.month + 1
    if month == 13:
        year, month = year + 1, 1
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())

def candle_is_closed(timeframe: str, open_time: float, now: float) -> bool:
    return next_candle_open(timeframe, open_time) <= now

def stale_deadline(timeframe: str, open_time: float, intervals: float) -> float:
    """Return a calendar-correct freshness deadline from a candle's open."""
    validate_interval(timeframe)
    if intervals <= 0:
        return float(open_time)
    whole = int(intervals)
    cursor = int(open_time)
    for _ in range(whole):
        cursor = next_candle_open(timeframe, cursor)
    fraction = intervals - whole
    if fraction:
        following = next_candle_open(timeframe, cursor)
        cursor += int((following - cursor) * fraction)
    return float(cursor)

def history_limit(timeframe: str, overrides: Mapping[str, int] | None = None) -> int:
    validate_interval(timeframe)
    value = (overrides or {}).get(timeframe, TIMEFRAME_HISTORY_CANDLES[timeframe])
    if isinstance(value, bool) or int(value) != value or int(value) < 20:
        raise ValueError(f"invalid closed-candle history size for {timeframe}: {value!r}")
    return int(value)


def validate_closed_candle_timestamps(timeframe: str, open_times: list[float], decision_time: float) -> None:
    """Reject malformed/future native bars not fully closed at a decision instant."""
    validate_interval(timeframe)
    if not open_times:
        raise ValueError("native timeframe has no candle timestamps")
    previous = None
    for raw in open_times:
        stamp = float(raw)
        if not is_aligned_open(timeframe, stamp):
            raise ValueError("native candle timestamp is not aligned")
        if previous is not None:
            if stamp <= previous:
                raise ValueError("native candle timestamps are not strictly increasing")
            if stamp != next_candle_open(timeframe, previous):
                raise ValueError("native candle timestamps contain a gap")
        if next_candle_open(timeframe, stamp) > decision_time:
            raise ValueError("native candle is not closed at decision time")
        previous = stamp

# Nearby resolutions are correlated; summarize each bucket once before using
# them as multi-timeframe confluence evidence.
TIMEFRAME_CORRELATION_GROUPS = {
    "fast": ("1m", "3m"),
    "short": ("5m", "15m"),
    "session": ("30m", "1h", "2h"),
    "swing": ("4h", "6h", "8h", "12h"),
    "macro": ("1d", "3d", "1w", "1M"),
}
TIMEFRAME_GROUP_WEIGHTS = {"fast": 1.0, "short": 2.0, "session": 3.0, "swing": 4.0, "macro": 5.0}

def synchronize_native_frames_to_1m_close(
    frames: Mapping[str, object], *, as_of: float | None = None,
) -> dict[str, object]:
    """Truncate native frames to the latest fully closed 1m decision instant.

    This prevents faster frames (sub-minute frames are not used) from contributing bars that close
    after the reference 1m candle. Inputs are copied and must have timezone-aware
    DatetimeIndex values; insufficient history, a future/forming 1m reference,
    or malformed frames fail closed. ``as_of`` exists for deterministic tests.
    """
    import pandas as pd

    if not isinstance(frames, Mapping) or "1m" not in frames:
        raise ValueError("native frames require a 1m reference")
    reference = frames["1m"]
    if not isinstance(reference, pd.DataFrame) or reference.empty or not isinstance(reference.index, pd.DatetimeIndex):
        raise ValueError("1m reference frame is empty or malformed")
    if reference.index.tz is None:
        raise ValueError("1m reference index must be timezone-aware")
    reference_open = reference.index[-1].tz_convert("UTC").timestamp()
    decision_time = float(next_candle_open("1m", reference_open))
    observed_as_of = float(time.time() if as_of is None else as_of)
    if not math.isfinite(observed_as_of) or decision_time > observed_as_of:
        raise ValueError("latest 1m reference candle is not closed as of the decision")
    synchronized: dict[str, object] = {}
    for timeframe, frame in frames.items():
        validate_interval(timeframe)
        if not isinstance(frame, pd.DataFrame) or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{timeframe} frame is empty or malformed")
        if frame.index.tz is None:
            raise ValueError(f"{timeframe} index must be timezone-aware")
        utc_index = frame.index.tz_convert("UTC")
        stamps = [float(value.timestamp()) for value in utc_index]
        previous = None
        for stamp in stamps:
            if not is_aligned_open(timeframe, stamp):
                raise ValueError(f"{timeframe} contains a misaligned candle timestamp")
            if previous is not None and (stamp <= previous or stamp != next_candle_open(timeframe, previous)):
                raise ValueError(f"{timeframe} contains duplicate, unordered, or gapped candles")
            previous = stamp
        keep = [next_candle_open(timeframe, stamp) <= decision_time for stamp in stamps]
        closed_stamps = [stamp for stamp, is_closed in zip(stamps, keep) if is_closed]
        if not closed_stamps:
            raise ValueError(f"{timeframe} has no closed candles at the decision cutoff")
        validate_closed_candle_timestamps(timeframe, closed_stamps, decision_time)
        aligned = frame.loc[keep].copy(deep=True)
        if len(aligned) < 30:
            raise ValueError(f"{timeframe} has insufficient synchronized closed history")
        synchronized[timeframe] = aligned
    return synchronized


def grouped_timeframe_values(values: Mapping[str, float], *, exclude: tuple[str, ...] = ()) -> dict[str, float]:
    """Average same-horizon frames, preventing correlated frames from multiplying votes."""
    excluded = set(exclude)
    grouped = {}
    for name, frames in TIMEFRAME_CORRELATION_GROUPS.items():
        present = []
        for tf in frames:
            if tf in excluded or tf not in values:
                continue
            try:
                value = float(values[tf])
            except (TypeError, ValueError, OverflowError):
                continue
            if value == value and abs(value) != float("inf"):
                present.append(value)
        if present:
            grouped[name] = sum(present) / len(present)
    return grouped

def grouped_timeframe_consensus(values: Mapping[str, float], *, exclude: tuple[str, ...] = ()) -> tuple[float, dict[str, float]]:
    groups = grouped_timeframe_values(values, exclude=exclude)
    denominator = sum(TIMEFRAME_GROUP_WEIGHTS[name] for name in groups)
    if denominator <= 0:
        return 0.0, groups
    score = sum(groups[name] * TIMEFRAME_GROUP_WEIGHTS[name] for name in groups) / denominator
    return score, groups

# Backward-compatible alias (internal code that still references old name)
synchronize_native_frames_to_base_close = synchronize_native_frames_to_1m_close
