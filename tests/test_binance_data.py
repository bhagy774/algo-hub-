import pytest
from unittest.mock import patch, MagicMock
import binance_data
from binance_data import BinanceData

def test_get_live_price_success():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"price": "60000.0"}
    with patch.object(bd, '_get', return_value=mock_resp):
        price = bd.get_live_price("BTCUSDT")
    assert price == 60000.0
    assert bd.get_last_source()["source"] == "binance_spot"
    assert bd.get_last_source()["market_semantics"] == "spot"

def test_get_live_price_bybit_fallback():
    bd = BinanceData()
    mock_bybit_resp = MagicMock()
    mock_bybit_resp.status_code = 200
    mock_bybit_resp.json.return_value = {"result": {"list": [{"lastPrice": "59000.0"}]}}
    with patch.object(bd, '_get', return_value=None):
        with patch.object(bd.session, 'get', return_value=mock_bybit_resp):
            price = bd.get_live_price("BTCUSDT")
    assert price == 59000.0
    assert bd.get_last_source()["source"] == "bybit_linear_perpetual"

def test_get_historical_candles_success():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [[1700000000000, "10.0", "15.0", "5.0", "12.0", "100.0"]]
    with patch.object(bd, '_get', return_value=mock_resp):
        candles = bd.get_historical_candles("BTCUSDT", "1m", 1)
    assert len(candles) == 1
    c = candles[0]
    assert c["time"] == 1700000000
    assert c["open"] == 10.0
    assert c["high"] == 15.0
    assert c["low"] == 5.0
    assert c["close"] == 12.0
    assert c["volume"] == 100.0
    assert bd.get_last_source()["source"] == "binance_spot"

def test_all_native_interval_tokens_are_supported_distinctly():
    assert set(binance_data.RESOLUTION_MAP) == {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
    assert binance_data.RESOLUTION_MAP["1m"] != binance_data.RESOLUTION_MAP["1M"]

def test_historical_fetch_pages_over_binance_per_request_limit():
    bd = BinanceData()
    first = MagicMock(status_code=200)
    second = MagicMock(status_code=200)
    first.json.return_value = [[i * 60_000, "10", "11", "9", "10.5", "3"] for i in range(1, 1001)]
    second.json.return_value = [[0, "10", "11", "9", "10.5", "3"]]
    with patch.object(bd, '_get', side_effect=[first, second]) as request:
        candles = bd.get_historical_candles("ETHUSDT", "1m", 1001, allow_fallback=False)
    assert len(candles) == 1001
    assert candles[0]["time"] == 0
    assert candles[-1]["time"] == 1000 * 60
    assert request.call_count == 2
    assert request.call_args_list[0].kwargs["params"]["limit"] == 1000
    assert request.call_args_list[1].kwargs["params"]["endTime"] == 59_999

def test_mtf_helper_defaults_to_all_native_frames_and_timeframe_history_profile():
    from binance_timeframes import BINANCE_SPOT_TIMEFRAMES, TIMEFRAME_HISTORY_CANDLES

    bd = BinanceData()
    calls = []

    def fake_fetch(symbol, resolution, limit, *, allow_fallback=True):
        calls.append((resolution, limit, allow_fallback))
        return [{"time": i} for i in range(limit)]

    with patch.object(bd, "get_historical_candles", side_effect=fake_fetch):
        result = bd.fetch_mtf_candles()

    assert tuple(result) == BINANCE_SPOT_TIMEFRAMES
    assert [(tf, limit) for tf, limit, _ in calls] == [
        (tf, TIMEFRAME_HISTORY_CANDLES[tf] + 1) for tf in BINANCE_SPOT_TIMEFRAMES
    ]
    assert all(allow_fallback is False for _, _, allow_fallback in calls)


def test_mtf_helper_rejects_partial_native_history():
    bd = BinanceData()
    with patch.object(bd, "get_historical_candles", return_value=[]):
        with pytest.raises(RuntimeError, match="incomplete native Binance Spot history"):
            bd.fetch_mtf_candles(timeframes=["1m"], limit=20)


def test_get_bid_ask():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"bidPrice": "100.0", "askPrice": "102.0"}
    with patch.object(bd, '_get', return_value=mock_resp):
        result = bd.get_bid_ask("BTCUSDT")
    assert result["bid"] == 100.0
    assert result["ask"] == 102.0
    assert result["spread"] == 2.0
    assert result["source"] == "binance_spot"

def test_get_funding_rate():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"lastFundingRate": "0.001", "nextFundingTime": 1700000000000, "markPrice": "60000.0"}
    with patch.object(bd, '_get', return_value=mock_resp):
        with patch('time.time', return_value=1699996400):
            result = bd.get_funding_rate("BTCUSDT")
    assert result["funding_rate"] == 0.001
    assert result["funding_rate_pct"] == 0.1
    assert result["next_funding_time"] == 1700000000000
    assert result["countdown_str"] == "01h 00m"
    assert result["markPrice" if "markPrice" in result else "mark_price"] == 60000.0
    assert result["sentiment"] == "BULLISH"
    assert result["source"] == "binance_futures"

def test_get_open_interest():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"openInterest": "500.5", "time": 1700000000000}
    with patch.object(bd, '_get', return_value=mock_resp):
        result = bd.get_open_interest("BTCUSDT")
    assert result["open_interest"] == 500.5
    assert result["timestamp"] == 1700000000000
    assert result["source"] == "binance_futures"

def test_get_order_book_depth():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"bids": [["100.0", "1.0"]], "asks": [["102.0", "2.0"]]}
    with patch.object(bd, '_get', return_value=mock_resp):
        result = bd.get_order_book_depth("BTCUSDT")
    assert result["top_bid"] == 100.0
    assert result["top_ask"] == 102.0
    assert result["total_bid_vol"] == 1.0
    assert result["total_ask_vol"] == 2.0
    assert result["imbalance_pct"] == -33.33
    assert result["source"] == "binance_spot"

def test_get_24h_stats():
    bd = BinanceData()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"priceChangePercent": "5.5", "volume": "100.0", "quoteVolume": "6000000.0", "highPrice": "65000.0", "lowPrice": "55000.0"}
    with patch.object(bd, '_get', return_value=mock_resp):
        result = bd.get_24h_stats("BTCUSDT")
    assert result["price_change_pct"] == 5.5
    assert result["volume_24h_btc"] == 100.0
    assert result["volume_24h_usdt"] == 6000000.0
    assert result["high_24h"] == 65000.0
    assert result["low_24h"] == 55000.0
    assert result["source"] == "binance_spot"


def test_get_bid_ask_fallback_marks_unavailable():
    bd = BinanceData()
    with patch.object(bd, '_get', return_value=None):
        with patch.object(bd, 'get_live_price', return_value=60500.0):
            result = bd.get_bid_ask("BTCUSDT")
    assert result["source"] == "unavailable"
    assert result["bid"] is None
    assert result["ask"] is None
    assert result["spread"] is None
    assert result["fallback_price"] == 60500.0
