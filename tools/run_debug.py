import pandas as pd
from datetime import datetime, timezone
import sys
sys.path.append('.')
from binance_timeframes import synchronize_native_frames_to_base_close

five_minute = pd.DataFrame(
    {'close': range(30)},
    index=pd.date_range('2026-10-01T10:30:00Z', periods=30, freq='5min'),
)
seconds = pd.DataFrame(
    {'close': range(151)},
    index=pd.date_range('2026-10-01T12:58:00Z', periods=151, freq='s'),
)

try:
    result = synchronize_native_frames_to_base_close(
        {'5m': five_minute, '1s': seconds}, as_of=datetime(2026, 10, 1, 13, 0, 10, tzinfo=timezone.utc).timestamp()
    )
    print('SUCCESS')
except Exception as e:
    print('ERROR:', e)
