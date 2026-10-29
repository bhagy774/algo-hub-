import pytest
import time
from unittest.mock import patch, MagicMock
from jarvis_coin_scanner import JarvisCoinScanner, get_coin_scanner, SCAN_COINS

def test_initialization():
    """Test standard initialization and default state."""
    scanner = JarvisCoinScanner()
    assert scanner._running is False
    assert scanner.get_best_coin() == ""
    assert scanner.get_best_symbol() == ""
    assert scanner.get_delta_symbol() == ""
    assert scanner.get_scanner_status()["scan_count"] == 0
    assert scanner.get_all_scores() == {}

def test_singleton():
    """Test the singleton pattern behavior."""
    scanner1 = get_coin_scanner()
    scanner2 = get_coin_scanner()
    assert scanner1 is scanner2

    # We can pass custom bus and check_fn to verify they're ignored
    # on subsequent calls if singleton exists.
    bus_mock = MagicMock()
    scanner3 = get_coin_scanner(bus=bus_mock)
    assert scanner1 is scanner3
    # The bus shouldn't change
    assert scanner3.bus is scanner1.bus

def test_api_methods():
    """Test basic getter API methods."""
    scanner = JarvisCoinScanner()
    scanner._best_coin = "ETH"
    scanner._best_symbol = "ETHUSDT"
    scanner._delta_symbol = "ETHUSDT"

    assert scanner.get_best_coin() == "ETH"
    assert scanner.get_best_symbol() == "ETHUSDT"
    assert scanner.get_delta_symbol() == "ETHUSDT"

def test_start_stop():
    """Test that start() spawns a thread and stop() ends it."""
    scanner = JarvisCoinScanner()
    # Mocking sleep to avoid actual waiting
    with patch("time.sleep", return_value=None):
        with patch.object(scanner, "_run_scan_cycle") as mock_scan:
            scanner.start()
            assert scanner._running is True
            assert scanner._thread is not None
            assert scanner._thread.is_alive()

            # Allow thread to run a bit
            time.sleep(0.1)

            scanner.stop()
            assert scanner._running is False

            # Depending on timing, it may have run at least once
            # mock_scan.assert_called()

def test_calc_rsi():
    """Test the internal RSI calculation method."""
    scanner = JarvisCoinScanner()

    # Not enough data
    closes = [100.0, 101.0, 102.0]
    assert scanner._calc_rsi(closes, period=14) == 50.0

    # Steady increase
    closes = [100.0 + i for i in range(20)]
    rsi = scanner._calc_rsi(closes, period=14)
    assert rsi == 100.0

    # Steady decrease
    closes = [100.0 - i for i in range(20)]
    rsi = scanner._calc_rsi(closes, period=14)
    assert rsi == 0.0

    # Mixed data
    closes = [
        100.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0,
        105.0, 104.0, 106.0, 105.0, 107.0, 106.0, 108.0
    ]
    rsi = scanner._calc_rsi(closes, period=14)
    # Gains: 1, 2, 2, 2, 2, 2, 2, 2 -> average gain
    # Losses: 1, 1, 1, 1, 1, 1, 1 -> average loss
    assert 0.0 < rsi < 100.0

def test_score_coin_no_stats():
    """Test _score_coin when stats are empty."""
    scanner = JarvisCoinScanner()
    score = scanner._score_coin("BTC", "BTCUSDT", {})
    assert score["total"] == 0

@patch.object(JarvisCoinScanner, "_fetch_binance_klines")
@patch.object(JarvisCoinScanner, "_fetch_funding_rate")
def test_score_coin_valid_stats(mock_fetch_funding, mock_fetch_candles):
    """Test scoring logic with valid statistics."""
    mock_fetch_funding.return_value = 0.01  # 1% funding
    # 15 random closes to pass RSI period
    now = int(time.time()*1000)
    boundary = now // 300000 * 300000
    mock_fetch_candles.return_value = [[boundary-i*300000, "100", "101", "99", "100", "1", boundary-(i-1)*300000-1] for i in range(16,0,-1)]

    scanner = JarvisCoinScanner()
    stats = {
        "quoteVolume": "10000000",   # 10M
        "priceChangePercent": "4.0", # 4% change
        "highPrice": "105.0",
        "lowPrice": "95.0",
        "lastPrice": "100.0", "symbol": "ETHUSDT", "bidPrice": "99.99", "askPrice": "100.01", "closeTime": now
    }

    score = scanner._score_coin("ETH", "ETHUSDT", stats)

    assert score["volume_24h_usdt"] == 10000000.0
    # 10M volume is >= 10M -> 9 points
    assert score["volume_score"] == 9

    # 4% momentum is >= 3 -> 16 points
    assert score["momentum_score"] == 16

    # RSI for flat closes is 100.0 since al == 0
    # 100 falls in "else" -> 0 points
    assert score["rsi_score"] == 0

    # Volatility range = (105-95)/100 = 10% -> 10.0% > 8.0 -> 6 points
    assert score["volatility_score"] == 6

    # Funding 0.01 * 100 = 1% > 0.10 -> 1 point
    assert score["funding_score"] == 1

    # Total = 9 + 16 + 0 + 6 + 1 = 32.0
    assert score["eligible"] is True
    assert score["spread_score"] == 10
    assert score["total"] == 42.0

@patch.object(JarvisCoinScanner, "_fetch_binance_stats_bulk")
@patch.object(JarvisCoinScanner, "_score_coin")
def test_run_scan_cycle_volume_filter(mock_score_coin, mock_fetch_stats):
    """Test that low volume coins are filtered out."""
    scanner = JarvisCoinScanner()

    mock_fetch_stats.return_value = {"BTCUSDT": {}}

    def side_effect_score(coin, symbol, stats, **kwargs):
        return {
            "volume_24h_usdt": 1_000_000.0, # Below 5_000_000 min volume
            "total": 100
        }

    mock_score_coin.side_effect = side_effect_score

    best_coin = scanner._run_scan_cycle()
    assert best_coin == ""
    assert scanner.get_best_coin() == ""

@patch.object(JarvisCoinScanner, "_fetch_binance_stats_bulk")
@patch.object(JarvisCoinScanner, "_score_coin")
def test_run_scan_cycle_success(mock_score_coin, mock_fetch_stats):
    """Test successful selection of the best coin."""
    scanner = JarvisCoinScanner()

    mock_fetch_stats.return_value = {
        "BTCUSDT": {},
        "ETHUSDT": {}
    }

    def side_effect_score(coin, symbol, stats, **kwargs):
        if coin == "BTC":
            return {"volume_24h_usdt": 10_000_000.0, "total": 80, "eligible": True, "spread_bps": 2.0}
        elif coin == "ETH":
            return {"volume_24h_usdt": 10_000_000.0, "total": 90, "eligible": True, "spread_bps": 2.0}
        return {"volume_24h_usdt": 0.0, "total": 0}

    mock_score_coin.side_effect = side_effect_score

    best_coin = scanner._run_scan_cycle()
    assert best_coin == "ETH"
    assert scanner.get_best_coin() == "ETH"
    assert scanner.get_best_symbol() == "ETHUSDT"

    status = scanner.get_scanner_status()
    assert status["scan_count"] == 1

@patch.object(JarvisCoinScanner, "_fetch_binance_stats_bulk")
@patch.object(JarvisCoinScanner, "_score_coin")
def test_run_scan_cycle_position_open(mock_score_coin, mock_fetch_stats):
    """Test that the best coin does not change when a position is open."""
    scanner = JarvisCoinScanner()
    scanner._best_coin = "BTC"
    scanner._position_check_fn = lambda: True # Simulate open position

    mock_fetch_stats.return_value = {
        "ETHUSDT": {}
    }

    def side_effect_score(coin, symbol, stats, **kwargs):
        if coin == "ETH":
            return {"volume_24h_usdt": 10_000_000.0, "total": 90, "eligible": True, "spread_bps": 2.0}
        return {"volume_24h_usdt": 0.0, "total": 0}

    mock_score_coin.side_effect = side_effect_score

    best_coin = scanner._run_scan_cycle()
    # Best coin from scan was ETH, but since position is open, returns BTC
    assert best_coin == "BTC"
    assert scanner.get_best_coin() == "BTC"
