from jarvis_backtester import BacktestConfig
import pytest
from datetime import datetime
from pathlib import Path

from jarvis_backtester import (
    extract_signal,
    HistoricalCandleLoader,
    BacktestPosition,
    BacktestRiskGate,
    JarvisFullBacktester,
)


def test_extract_signal():
    """Test extracting direction and confidence from decision result."""

    # Test BUY signal
    result_buy = {
        "trade_signal": {
            "direction": "BUY",
            "confidence_score": "85/100"
        }
    }
    direction, confidence = extract_signal(result_buy)
    assert direction == "CALL"
    assert confidence == 85

    # Test SELL signal
    result_sell = {
        "trade_signal": {
            "direction": "SELL",
            "confidence_score": "92/100"
        }
    }
    direction, confidence = extract_signal(result_sell)
    assert direction == "PUT"
    assert confidence == 92

    # Test NO_TRADE signal
    result_none = {
        "trade_signal": {
            "direction": "NO_TRADE",
            "confidence_score": "0/100"
        }
    }
    direction, confidence = extract_signal(result_none)
    assert direction == "NO_TRADE"
    assert confidence == 0

    # Test empty dict
    direction, confidence = extract_signal({})
    assert direction == "NO_TRADE"
    assert confidence == 0

    # Test malformed confidence
    result_malformed = {
        "trade_signal": {
            "direction": "BUY",
            "confidence_score": "not_a_number"
        }
    }
    direction, confidence = extract_signal(result_malformed)
    assert direction == "CALL"
    assert confidence == 0

def test_historical_candle_loader_init():
    """Test HistoricalCandleLoader initialization."""
    path = Path("dummy.csv")
    loader = HistoricalCandleLoader(path)
    assert loader.source == path

def test_backtest_position_init():
    """Test BacktestPosition initialization."""
    entry_time = datetime(2024, 1, 1, 12, 0)
    position = BacktestPosition(
        direction="CALL",
        entry_price=50000.0,
        entry_time=entry_time,
        contracts=1.5,
        trade_type="SCALP",
        confidence=85,
        slippage_bps=5.0,
        fee_bps=10.0
    )
    assert position.direction == "CALL"
    assert position.entry_price == 50000.0
    assert position.entry_time == entry_time
    assert position.contracts == 1.5
    assert position.trade_type == "SCALP"
    assert position.confidence == 85
    assert position.slippage_bps == 5.0
    assert position.fee_bps == 10.0
    assert position.is_call is True
    assert position.status == "OPEN"

def test_backtest_risk_gate_init():
    """Test BacktestRiskGate initialization."""
    gate = BacktestRiskGate()
    assert gate.daily_pnl == 0.0
    assert gate.consec_losses == 0
    assert gate.last_trade_dt is None
    assert gate.open_positions == []
    assert gate.today_date is None

def test_jarvis_full_backtester_init():
    """Test JarvisFullBacktester initialization."""
    backtester = JarvisFullBacktester(BacktestConfig(
        symbol="BTCUSDT",
        timeframe="5m",
        years=3,
        starting_capital=1000.0,
        warmup=100,
        slippage_bps=5.0,
        fee_bps=10.0,
        live_audit=False
    ))
    assert backtester.symbol == "BTCUSDT"
    assert backtester.timeframe == "5m"
    assert backtester.years == 3
    assert backtester.starting_capital == 1000.0
    assert backtester.warmup == 100
    assert backtester.slippage_bps == 5.0
    assert backtester.fee_bps == 10.0
    assert backtester.live_audit is False
    assert backtester.balance == 1000.0
    assert backtester.peak == 1000.0
    assert isinstance(backtester.gate, BacktestRiskGate)
    assert backtester.all_trades == []
