import json
import os
import time
from unittest.mock import patch, MagicMock

import pytest

import jarvis_watchdog
from jarvis_watchdog import (
    save_state,
    load_state,
    reconcile_state,
    beat,
    get_health,
    start_watchdog,
    stop_watchdog,
)

@pytest.fixture
def temp_state_file(tmp_path):
    return str(tmp_path / "test_state.json")

def test_save_and_load_state(temp_state_file):
    # Test saving state
    open_trades = [{"symbol": "BTCUSD", "qty": 1}]
    extra = {"info": "test"}
    assert save_state(open_trades, path=temp_state_file, extra=extra) is True

    # Test loading state
    loaded_state = load_state(path=temp_state_file)
    assert loaded_state["open_trades"] == open_trades
    assert loaded_state.get("extra") == extra
    assert "saved_at" in loaded_state

def test_load_state_file_not_found():
    state = load_state(path="nonexistent_file.json")
    assert state == {"open_trades": [], "saved_at": None}

def test_load_state_invalid_json(temp_state_file):
    with open(temp_state_file, "w") as f:
        f.write("invalid json")
    state = load_state(path=temp_state_file)
    assert state["open_trades"] == []
    assert state["saved_at"] is None
    assert "error" in state

def test_reconcile_state_no_exchange(temp_state_file):
    # Save some local state
    open_trades = [{"symbol": "BTCUSD", "qty": 1}]
    save_state(open_trades, path=temp_state_file)

    report = reconcile_state(exchange_client=None, state_path=temp_state_file)
    assert report["local_trades"] == 1
    assert report["exchange_ok"] is False
    assert report["error"] == "no exchange client provided; skipping exchange check"

def test_reconcile_state_exchange_success(temp_state_file):
    open_trades = [{"symbol": "BTCUSD", "qty": 1}]
    save_state(open_trades, path=temp_state_file)

    mock_client = MagicMock()
    mock_client.get_open_positions.return_value = [{"symbol": "BTCUSD", "size": 1}]

    report = reconcile_state(exchange_client=mock_client, state_path=temp_state_file)
    assert report["local_trades"] == 1
    assert report["exchange_ok"] is True
    assert report["exchange_positions"] == 1
    assert len(report["mismatches"]) == 0

def test_reconcile_state_exchange_mismatch(temp_state_file):
    open_trades = [{"symbol": "BTCUSD", "qty": 1}]
    save_state(open_trades, path=temp_state_file)

    mock_client = MagicMock()
    # Missing position locally but present on exchange, and different count
    mock_client.get_open_positions.return_value = [
        {"symbol": "ETHUSD", "size": 2},
        {"symbol": "SOLUSD", "size": 10},
    ]

    report = reconcile_state(exchange_client=mock_client, state_path=temp_state_file)
    assert report["local_trades"] == 1
    assert report["exchange_ok"] is True
    assert report["exchange_positions"] == 2
    assert len(report["mismatches"]) > 0

def test_reconcile_state_exchange_error(temp_state_file):
    mock_client = MagicMock()
    mock_client.get_open_positions.side_effect = Exception("API Error")

    report = reconcile_state(exchange_client=mock_client, state_path=temp_state_file)
    assert report["exchange_ok"] is False
    assert "API error" in report["error"]

def test_beat_and_get_health():
    # Clear previous heartbeats
    jarvis_watchdog._heartbeats.clear()

    beat("test_service")
    health = get_health()
    assert "test_service" in health["services"]
    assert health["services"]["test_service"]["alive"] is True
    assert health["services"]["test_service"]["age_seconds"] >= 0

@patch.dict(os.environ, {"JARVIS_WATCHDOG": "0"})
def test_start_watchdog_disabled():
    assert start_watchdog() is False

@patch.dict(os.environ, {"JARVIS_WATCHDOG": "1"})
def test_start_and_stop_watchdog(temp_state_file):
    # Ensure it's not already running
    stop_watchdog()

    # Start the watchdog
    started = start_watchdog(state_path=temp_state_file)
    assert started is True

    # Health should indicate it's running
    health = get_health()
    assert health["watchdog_running"] is True
    assert health["started_at"] is not None

    # Stop the watchdog
    stop_watchdog()
    health = get_health()
    assert health["watchdog_running"] is False
