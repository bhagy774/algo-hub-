import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from jarvis_FIXED import LiveTradingEngine
import datetime

def test_killswitch_logic():
    mock_jarvis = MagicMock()
    engine = LiveTradingEngine(jarvis_system=mock_jarvis)
    engine.is_running = True
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_file = os.path.join(tmpdir, "STOP_JARVIS")
        with open(stop_file, 'w') as f:
            f.write("stop")
            
        with patch('os.path.exists') as mock_exists:
            # We want to trigger when C:\jarvis\STOP_JARVIS is checked
            mock_exists.side_effect = lambda path: True if path == "C:\\jarvis\\STOP_JARVIS" else False
            with patch.dict(os.environ, {"JARVIS_KILL_SWITCH": "1"}):
                # We can't easily run the whole _live_loop because it has an infinite while True.
                # However, the killswitch breaks the loop. So if we run it and it exits immediately, we are good.
                
                # We mock time.sleep to avoid hanging in case it doesn't break
                def mock_sleep(seconds):
                    if seconds > 1:
                        raise InterruptedError("Should not reach sleep if kill-switch works")
                    import time as real_time
                    # actually don't sleep to be fast, just return
                with patch('time.sleep', side_effect=mock_sleep):
                    try:
                        # start_live_trading starts a thread, but we can just test the inner loop function directly
                        # by triggering it directly if possible. But _live_loop is nested.
                        # We can just start_live_trading and wait for thread to finish.
                        engine.start_live_trading()
                        
                        import time
                        start_time = time.time()
                        while engine.is_running and time.time() - start_time < 2:
                            pass
                            
                        assert engine.is_running == False, "Killswitch did not stop the engine"
                    except InterruptedError:
                        pytest.fail("Killswitch didn't break the loop, reached sleep instead")

def test_daily_report_logic():
    mock_jarvis = MagicMock()
    engine = LiveTradingEngine(jarvis_system=mock_jarvis)
    engine.is_running = True
    
    with patch('jarvis_FIXED.datetime') as mock_datetime:
        # Mock time to exactly report time
        mock_now = datetime.datetime(2026, 1, 1, 20, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.fromisoformat = datetime.datetime.fromisoformat
        
        with patch.dict(os.environ, {"JARVIS_DAILY_REPORT": "1", "JARVIS_REPORT_TIME": "20:00", "JARVIS_KILL_SWITCH": "0"}):
            with patch('telegram_notifier.send_daily_report') as mock_send_report:
                # We want to break the loop after one cycle
                with patch('time.sleep', side_effect=Exception("Break loop")):
                    engine.start_live_trading()
                    
                    import time
                    start_time = time.time()
                    # Wait for thread to hit our exception
                    while engine.is_running and time.time() - start_time < 2:
                        pass
                        
                    mock_send_report.assert_called_once()
