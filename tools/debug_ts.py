from datetime import datetime, timezone
import math
import time
import pandas as pd

def candle_open_time(ts):
    return ts - (ts % 300)

ts = datetime(2026, 10, 1, 10, 59, 0, tzinfo=timezone.utc).timestamp()
ts_open = candle_open_time(ts)
decision = ts_open + 300
as_of = datetime(2026, 10, 1, 11, 0, 10, tzinfo=timezone.utc).timestamp()

print(f"ts: {ts}")
print(f"ts_open: {ts_open}")
print(f"decision: {decision}")
print(f"as_of: {as_of}")
print(f"decision > as_of: {decision > as_of}")

index = pd.date_range("2026-10-01T10:30:00Z", periods=30, freq="min")
print("Last index:", index[-1].timestamp())
