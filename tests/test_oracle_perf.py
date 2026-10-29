import time
import json
import logging
from unittest.mock import patch, MagicMock
from jarvis_market_oracle import JarvisMarketOracle

def test_oracle_api_retry_performance():
    # Setup mock oracle
    oracle = JarvisMarketOracle()
    oracle._gemini_client = MagicMock()
    oracle._gemini_types = MagicMock()
    # Mock running so the wait actually happens
    oracle._running = True

    # Create market_data and legacy_llm_board dicts
    market_data = {"btc_spot": 90000.0}
    legacy_llm_board = {"analyst": "Bullish", "validator": "Bullish", "risk_officer": "Bullish", "chairman": "CALL"}

    # We want to measure the time it takes when the API fails multiple times with 503
    # Configure the mock to raise Exception with "503" string

    def mock_generate_content(*args, **kwargs):
        raise Exception("503 Service Unavailable")

    oracle._gemini_client.models.generate_content.side_effect = mock_generate_content

    start_time = time.time()
    result = oracle._call_gemini_oracle(market_data, legacy_llm_board)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\\nDuration of _call_gemini_oracle with 503 errors: {duration:.2f} seconds")

if __name__ == "__main__":
    test_oracle_api_retry_performance()
