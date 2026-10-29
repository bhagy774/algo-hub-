"""Offline contracts for native Binance Spot candles and live refreshes."""
from datetime import datetime, timezone
import pytest
pd = pytest.importorskip("pandas")
from binance_timeframes import BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_HISTORY_CANDLES, candle_open_time, next_candle_open
from direct_candle_cache import DirectCandleCache, CandleDataError, LIVE_TIMEFRAMES

WIDTH = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800}
class FakeClock:
    def __init__(self, value): self.value = float(value)
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds

def previous_open(tf, stamp):
    if tf == "1M":
        dt = datetime.fromtimestamp(stamp, timezone.utc); year, month = dt.year, dt.month - 1
        if month == 0: year, month = year - 1, 12
        return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    return int(stamp - WIDTH[tf])

def make_rows(now, tf, limit, mutate=None):
    stamps = [candle_open_time(tf, now)]
    for _ in range(limit - 1): stamps.append(previous_open(tf, stamps[-1]))
    rows = []
    for stamp in reversed(stamps):
        price = 1000 + (stamp // max(1, WIDTH.get(tf, 86400))) % 1000
        rows.append({"time": stamp, "open": price, "high": price + 2, "low": price - 2, "close": price + 1, "volume": 1})
    return mutate(rows, tf) if mutate else rows

class FakeClient:
    def __init__(self, clock, mutator=None): self.clock, self.mutator, self.calls = clock, mutator, []
    def get_historical_candles_with_metadata(self, symbol, resolution, limit):
        self.calls.append((symbol, resolution, limit))
        return {"candles": make_rows(self.clock(), resolution, limit, self.mutator), "source": "binance", "market_type": "spot", "symbol": symbol}

def test_all_16_native_intervals_use_variable_history_and_separate_forming_bar():
    clock = FakeClock(1_727_503_200); client = FakeClient(clock)
    snap = DirectCandleCache(client, clock=clock).refresh("ETHUSDT", venue="binance", market_type="spot")
    assert tuple(snap.frames) == BINANCE_SPOT_TIMEFRAMES == LIVE_TIMEFRAMES
    assert len(client.calls) == 15
    for tf, frame in snap.frames.items():
        assert len(frame.closed) == TIMEFRAME_HISTORY_CANDLES[tf]
        assert client.calls[list(LIVE_TIMEFRAMES).index(tf)][2] == TIMEFRAME_HISTORY_CANDLES[tf] + 1
        assert frame.current and frame.current["time"] not in set(frame.closed.index.astype("int64") // 10**9)
        assert frame.current_is_confirmed is False and frame.source == "binance"

def test_calendar_month_week_and_minute_month_token_are_distinct():
    stamp = datetime(2026, 10, 1, 14, tzinfo=timezone.utc).timestamp()
    assert datetime.fromtimestamp(candle_open_time("1M", stamp), timezone.utc).day == 1
    assert datetime.fromtimestamp(candle_open_time("1w", stamp), timezone.utc).weekday() == 0
    assert next_candle_open("1M", candle_open_time("1M", stamp)) == datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()


def test_one_second_frames_are_never_requested():
    clock = FakeClock(1_727_503_200); client = FakeClient(clock)
    cache = DirectCandleCache(client, clock=clock, ttl_seconds=5)
    snap = cache.refresh("ETHUSDT", venue="binance", market_type="spot")
    assert "1s" not in snap.frames
    assert all(row[1] != "1s" for row in client.calls)


def test_refresh_reuses_fresh_snapshot_then_refreshes_after_ttl():
    clock = FakeClock(1_727_503_200); client = FakeClient(clock)
    cache = DirectCandleCache(client, clock=clock, ttl_seconds=45,
                              timeframes=("1m",), history_limits={"1m": 20})
    first = cache.refresh("ETHUSDT", venue="binance", market_type="spot")
    assert cache.refresh("ETHUSDT", venue="binance", market_type="spot") is first
    assert len(client.calls) == 1
    clock.advance(46)
    second = cache.refresh("ETHUSDT", venue="binance", market_type="spot")
    assert second is not first
    assert len(client.calls) == 2


def test_symbol_cache_isolation_and_identity_mismatch():
    clock = FakeClock(1_727_503_200); client = FakeClient(clock)
    cache = DirectCandleCache(client, clock=clock, timeframes=("1m",))
    eth = cache.refresh("ETHUSDT", venue="binance", market_type="spot")
    btc = cache.refresh("BTCUSDT", venue="binance", market_type="spot")
    assert eth.symbol == "ETHUSDT" and btc.symbol == "BTCUSDT"
    assert cache.get_snapshot("ETHUSDT", venue="binance", market_type="spot").symbol == "ETHUSDT"
    class WrongInstrument(FakeClient):
        def get_historical_candles_with_metadata(self, symbol, resolution, limit):
            result = super().get_historical_candles_with_metadata(symbol, resolution, limit); result["symbol"] = "BTCUSDT"; return result
    with pytest.raises(CandleDataError): DirectCandleCache(WrongInstrument(clock), clock=clock, timeframes=("1m",)).refresh("ETHUSDT")


def test_bad_missing_duplicate_or_gap_data_fails_closed():
    mutators = [lambda r, tf: r[:-1], lambda r, tf: r[:-1] + [dict(r[-1], time=r[-2]["time"])],
                lambda r, tf: r[:-1] + [dict(r[-1], time=next_candle_open(tf, r[-1]["time"]))]]
    for mutate in mutators:
        clock = FakeClock(1_727_503_200)
        with pytest.raises(CandleDataError): DirectCandleCache(FakeClient(clock, mutate), clock=clock, timeframes=("1m",)).refresh("ETHUSDT")


def test_fractional_second_open_is_not_truncated_into_valid_candle():
    clock = FakeClock(1_727_503_200)
    def fractional(rows, tf):
        rows[0]["time"] = float(rows[0]["time"]) + 0.25
        return rows
    with pytest.raises(CandleDataError, match="exact whole second"):
        DirectCandleCache(FakeClient(clock, fractional), clock=clock, timeframes=("1m",)).refresh("ETHUSDT")


def test_unsupported_interval_rejected():
    with pytest.raises(CandleDataError): DirectCandleCache(FakeClient(FakeClock(1_727_503_200)), timeframes=("7m",))
