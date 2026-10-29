"""Calendar-correct offline native Binance candle timestamps for shared fixtures."""
from datetime import datetime, timezone
from binance_timeframes import candle_open_time, next_candle_open, TIMEFRAME_HISTORY_CANDLES

def native_times(timeframe, now, limit):
    forming = candle_open_time(timeframe, now)
    times = [forming]
    for _ in range(limit - 1):
        if timeframe == "1M":
            dt = datetime.fromtimestamp(times[-1], timezone.utc)
            year, month = (dt.year - 1, 12) if dt.month == 1 else (dt.year, dt.month - 1)
            previous = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
        else:
            # All other native intervals have fixed width, including Monday weeks.
            previous = times[-1] - (next_candle_open(timeframe, times[-1]) - times[-1])
        times.append(previous)
    return list(reversed(times))
