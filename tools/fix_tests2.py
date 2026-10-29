import sys

with open('tests/test_binance_timeframes.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace all 1m mentions to 5m in the synchronization tests
code = code.replace('synchronize_native_frames_to_1m_close', 'synchronize_native_frames_to_5m_close')
code = code.replace('latest_completed_1m_close', 'latest_completed_5m_close')
code = code.replace('forming_1m_reference', 'forming_5m_reference')
code = code.replace('\"1m\"', '\"5m\"')
code = code.replace('one_minute', 'five_minute')

# Fix frequencies and periods to create valid 5m boundaries
code = code.replace('freq=\"min\"', 'freq=\"5min\"')
# If the period is 30 at 5m frequency starting at 10:30, it ends at 10:30 + 145 min = 12:55.
# Let's adjust the cutoff timestamps so the logic holds.
# For first test:
# start: 10:30
# end: 12:55
# second start: 12:58
# cutoff: 13:00
code = code.replace('2026-10-01T10:58:00Z', '2026-10-01T12:58:00Z')
code = code.replace('datetime(2026, 10, 1, 11, 0, 10', 'datetime(2026, 10, 1, 13, 0, 10')
code = code.replace('datetime(2026, 10, 1, 11, 0, tzinfo=timezone.utc)', 'datetime(2026, 10, 1, 13, 0, tzinfo=timezone.utc)')

# For second test:
code = code.replace('pd.Timedelta(minutes=1)', 'pd.Timedelta(minutes=5)')
code = code.replace('datetime(2026, 10, 1, 11, 0, 30', 'datetime(2026, 10, 1, 13, 0, 30')

with open('tests/test_binance_timeframes.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Updated test_binance_timeframes.py for 5m reference.')
