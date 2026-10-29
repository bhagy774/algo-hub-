import sys

with open('tests/test_binance_timeframes.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace all 1m mentions to 5m in the synchronization tests
code = code.replace('synchronize_native_frames_to_1m_close', 'synchronize_native_frames_to_5m_close')
code = code.replace('latest_completed_1m_close', 'latest_completed_5m_close')
code = code.replace('forming_1m_reference', 'forming_5m_reference')
code = code.replace('\"1m\"', '\"5m\"')
code = code.replace('one_minute', 'five_minute')

with open('tests/test_binance_timeframes.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Updated test_binance_timeframes.py for 5m reference.')
