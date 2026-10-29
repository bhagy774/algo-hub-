import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from jarvis_live_trader import JarvisAutoTrader, MAX_DAILY_LOSS_USDT, CONSEC_LOSS_LIMIT, MIN_CONFIDENCE

@pytest.fixture
def mock_delta_client():
    client = Mock()
    client.get_wallet_balance.return_value = 1000.0  # Safe balance
    client.cancel_order = Mock()
    client.place_order = Mock(return_value={"success": True, "order_id": "test_order"})
    client.get_open_orders = Mock(return_value=[{"id": "test_open_order"}])
    return client

@pytest.fixture
def mock_hedge_advisor():
    advisor = Mock()
    return advisor

@pytest.fixture
def trader(mock_delta_client, mock_hedge_advisor):
    trader = JarvisAutoTrader(mock_delta_client, mock_hedge_advisor)
    # Stop the monitor thread immediately for clean tests
    trader.stop_monitor()
    return trader

def test_initialization(trader, mock_delta_client, mock_hedge_advisor):
    assert trader.delta == mock_delta_client
    assert trader.hedge_advisor == mock_hedge_advisor
    assert trader.emergency_stop is False

def test_check_risk_gates_pass(trader):
    # Default state should pass risk gates
    # We pass a confidence level well above MIN_CONFIDENCE
    result = trader._check_risk_gates(max(MIN_CONFIDENCE + 10, 80))
    assert result["ok"] is True

def test_check_risk_gates_low_confidence(trader):
    result = trader._check_risk_gates(MIN_CONFIDENCE - 1)
    assert result["ok"] is False
    assert "Confidence" in result["reason"]

def test_check_risk_gates_emergency_stop(trader):
    trader.emergency_stop = True
    result = trader._check_risk_gates(80)
    assert result["ok"] is False
    assert "Emergency stop" in result["reason"]

def test_check_risk_gates_daily_loss(trader):
    trader.daily_pnl = - (MAX_DAILY_LOSS_USDT + 1)
    result = trader._check_risk_gates(80)
    assert result["ok"] is False
    assert "Daily loss limit" in result["reason"]

def test_check_risk_gates_consec_losses(trader):
    trader.consec_losses = CONSEC_LOSS_LIMIT + 1
    result = trader._check_risk_gates(80)
    assert result["ok"] is False
    assert "consecutive losses" in result["reason"]

def test_check_risk_gates_cooldown(trader):
    trader.last_trade_time = datetime.now()
    result = trader._check_risk_gates(80)
    assert result["ok"] is False
    assert "Cooldown" in result["reason"]

def test_execute_emergency_stop(trader):
    trader.emergency_stop = True
    result = trader.execute("CALL", 80, 50000.0)
    assert result["success"] is False
    assert result.get("skipped") is True
    assert "EMERGENCY STOP" in result["reason"]

def test_trigger_emergency_stop(trader, mock_delta_client):
    trader.open_positions = [{"id": "pos1", "direction": "CALL", "contracts": 1, "symbol": "BTCUSDT"}]

    # Mocking close position
    with patch.object(trader, '_close_position_market', return_value={"success": True}):
        trader.trigger_emergency_stop()

        assert trader.emergency_stop is True
        assert len(trader.open_positions) == 0  # Assuming it closes successfully
        mock_delta_client.cancel_order.assert_called_with("test_open_order")

def test_resume(trader):
    trader.emergency_stop = True
    trader.resume()
    assert trader.emergency_stop is False
