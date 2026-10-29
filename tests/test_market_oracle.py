import os
import sys
import pytest
from unittest.mock import MagicMock

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jarvis_market_oracle import JarvisMarketOracle

def test_oracle_initial_state():
    """Test that the Oracle initializes safely without network calls and has a valid fallback forecast."""
    oracle = JarvisMarketOracle(bus=MagicMock(), live_trader=MagicMock())
    forecast = oracle.get_latest_forecast()

    assert forecast is not None
    assert isinstance(forecast, dict)
    assert forecast.get("model_used") == "startup_default"
    assert forecast.get("trade_suggestion") == "WAIT"
    assert "options_intel" in forecast
    assert forecast["options_intel"]["pcr"] == 0.0

def test_generate_local_forecast_bullish():
    """Test that the local fallback synthesizer can generate a deterministic forecast based on inputs (Bullish case)."""
    oracle = JarvisMarketOracle()

    # Fake market data that is strongly bullish
    market_data = {
        "btc_spot": 95000.0,
        "microstructure": {
            "imbalance_pct": 5.0, # Positive imbalance
            "funding_rate": "+0.0100%"
        },
        "deribit": {
            "pcr": 0.65, # Bullish PCR < 0.8
            "max_pain": 96000.0,
            "gamma_wall_call": 97000.0,
            "gamma_wall_put": 94000.0
        }
    }

    legacy_llm_board = {
        "analyst": "Bullish",
        "validator": "Bullish",
        "risk_officer": "Safe",
        "chairman": "CONSENSUS_EXECUTE (CALL, 80%)"
    }

    forecast = oracle._generate_local_forecast(market_data, legacy_llm_board)

    assert forecast.get("model_used") == "local_synthesizer"
    assert forecast["5min"]["direction"] == "BULLISH"
    assert forecast["trade_suggestion"] == "CALL"

def test_generate_local_forecast_bearish():
    """Test that the local fallback synthesizer can generate a deterministic forecast based on inputs (Bearish case)."""
    oracle = JarvisMarketOracle()

    # Fake market data that is strongly bearish
    market_data = {
        "btc_spot": 94000.0,
        "microstructure": {
            "imbalance_pct": -20.0, # Highly negative imbalance
        },
        "deribit": {
            "pcr": 1.2, # Bearish PCR > 1.1
            "max_pain": 93000.0,
            "gamma_wall_call": 95000.0,
            "gamma_wall_put": 92000.0
        }
    }

    legacy_llm_board = {} # Don't really care about the board for local_synthesizer heuristics, but good to pass it

    forecast = oracle._generate_local_forecast(market_data, legacy_llm_board)

    assert forecast.get("model_used") == "local_synthesizer"
    assert forecast["5min"]["direction"] == "BEARISH"
    assert forecast["trade_suggestion"] == "PUT"

def test_generate_local_forecast_neutral():
    """Test that the local fallback synthesizer can generate a deterministic forecast based on inputs (Neutral case)."""
    oracle = JarvisMarketOracle()

    # Fake market data that is neutral
    market_data = {
        "btc_spot": 94500.0,
        "microstructure": {
            "imbalance_pct": 0.0,
        },
        "deribit": {
            "pcr": 0.9,
            "max_pain": 94500.0,
            "gamma_wall_call": 95000.0,
            "gamma_wall_put": 94000.0
        }
    }

    forecast = oracle._generate_local_forecast(market_data, {})

    assert forecast.get("model_used") == "local_synthesizer"
    assert forecast["5min"]["direction"] == "NEUTRAL"
    assert forecast["trade_suggestion"] == "WAIT"
