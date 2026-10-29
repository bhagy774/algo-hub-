import pytest
import os
import tempfile
import json
import time
from jarvis_cognitive_bus import CognitiveLogger, PartHealthMonitor, CognitiveBus

def test_cognitive_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = CognitiveLogger(log_dir=tmpdir, filename="test.log")
        assert os.path.exists(tmpdir)

        logger.log("TOPIC1", "SENDER1", "payload1")
        logger.log("TOPIC2", "SENDER2", {"key": "value"})

        # Give the thread some time to write
        time.sleep(1.5)
        logger.stop()

        filepath = os.path.join(tmpdir, "test.log")
        assert os.path.exists(filepath)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert "[TOPIC1]" in lines[0]
        assert "[SENDER1]" in lines[0]
        assert "payload1" in lines[0]

        assert "[TOPIC2]" in lines[1]
        assert "[SENDER2]" in lines[1]
        assert '{"key": "value"}' in lines[1]

def test_part_health_monitor():
    monitor = PartHealthMonitor()
    assert monitor.get_status() == {}

    # First error -> WARNING
    status1 = monitor.record_error("part1", Exception("err1"))
    assert status1 == "WARNING"
    assert monitor.get_status()["part1"].startswith("WARNING (1 errors)")

    # Second error -> WARNING
    status2 = monitor.record_error("part1", Exception("err2"))
    assert status2 == "WARNING"

    # Third error -> CRITICAL
    status3 = monitor.record_error("part1", Exception("err3"))
    assert status3 == "CRITICAL"
    assert monitor.get_status()["part1"].startswith("CRITICAL (3 errors)")

    # Record OK resets errors
    monitor.record_ok("part1")
    assert monitor.get_status()["part1"] == "OK"

    # Error for another part
    monitor.record_error("part2", Exception("err4"))
    assert monitor.get_status()["part2"].startswith("WARNING (1 errors)")

def test_cognitive_bus():
    bus = CognitiveBus()

    messages_received = []
    def callback(message):
        messages_received.append(message)

    bus.subscribe("TEST_TOPIC", callback)
    bus.publish("TEST_TOPIC", "TEST_SENDER", "test payload")

    assert len(messages_received) == 1
    assert messages_received[0]['topic'] == "TEST_TOPIC"
    assert messages_received[0]['sender'] == "TEST_SENDER"
    assert messages_received[0]['payload'] == "test payload"

    # Test special topics
    bus.publish("THOUGHTS", "SENDER_A", "thought A")
    bus.publish("SIGNALS", "SENDER_B", "signal B")

    recent = bus.get_recent_thoughts()
    assert len(recent) == 2
    assert recent[0]['sender'] == "SENDER_A"
    assert recent[1]['sender'] == "SENDER_B"

    latest_by_sender = bus.get_latest_by_sender()
    assert "SENDER_A" in latest_by_sender
    assert "SENDER_B" in latest_by_sender

    bus.publish("ORACLE_FORECAST", "ORACLE", {"forecast": "sunny"})
    assert bus.get_latest_oracle_forecast() == {"forecast": "sunny"}

    bus.publish("HEALTH_REPORT", "DOCTOR", "report content")
    reports = bus.get_doctor_reports()
    assert len(reports) == 1
    assert reports[0]['payload'] == "report content"

    # Test report_error
    bus.report_error("TestPart", Exception("Test error"), context="test context")
    health = bus.get_system_health()
    assert health["TestPart"].startswith("WARNING (1 errors)")

    # Test report_ok
    bus.report_ok("TestPart", context="recovered")
    health = bus.get_system_health()
    assert health["TestPart"] == "OK"

    # Shutdown bus
    bus.shutdown()
