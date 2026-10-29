"""
Tests for Jarvis Auxiliary Institutional Telemetry (Order Book, Funding, OI, 24h Stats).
Verifies:
- Live fetching and TTL throttling.
- Point-in-time offline backtest isolation.
- Telemetry persistence and replay without lookahead bias.
- Integration with XGBoost 48-feature store.
"""

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from jarvis_auxiliary_telemetry import AuxiliaryTelemetryCollector
from jarvis_xgboost_engine import JarvisXGBoostEngine


class TestAuxiliaryTelemetry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.snapshot_path = os.path.join(self.temp_dir, "telemetry_test.jsonl")
        self.collector = AuxiliaryTelemetryCollector(ttl_seconds=1.0, snapshot_path=self.snapshot_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_live_telemetry_collection_and_caching(self):
        mock_binance = MagicMock()
        mock_binance.get_order_book_depth.return_value = {
            "imbalance_pct": 25.5,
            "total_bid_vol": 100.0,
            "total_ask_vol": 59.7,
            "bias": "BULLISH",
            "source": "binance_spot",
        }
        mock_binance.get_funding_rate.return_value = {
            "funding_rate_pct": 0.04,
            "sentiment": "BULLISH",
            "source": "binance_futures",
        }
        mock_binance.get_open_interest.return_value = {
            "open_interest": 98500.0,
            "source": "binance_futures",
        }
        mock_binance.get_bid_ask.return_value = {
            "bid": 85000.0,
            "ask": 85001.0,
            "source": "binance_spot",
        }
        mock_binance.get_24h_stats.return_value = {
            "price_change_pct": 3.45,
            "volume_24h_btc": 25000.0,
            "source": "binance_spot",
        }

        # First call: hits client
        t1 = self.collector.get_telemetry("BTCUSDT", mock_binance, is_backtest_mode=False, current_price=85000.0)
        self.assertEqual(t1["order_book"]["bias"], "BULLISH")
        self.assertEqual(t1["order_book"]["imbalance_pct"], 25.5)
        self.assertAlmostEqual(t1["funding_rate"]["funding_rate_zscore"], 1.0, places=2)
        self.assertEqual(t1["open_interest"]["open_interest"], 98500.0)
        self.assertEqual(mock_binance.get_order_book_depth.call_count, 1)

        # Second call within TTL: should hit cache
        t2 = self.collector.get_telemetry("BTCUSDT", mock_binance, is_backtest_mode=False, current_price=85000.0)
        self.assertEqual(mock_binance.get_order_book_depth.call_count, 1)  # Cached!

    def test_backtest_isolation_guarantee(self):
        mock_binance = MagicMock()
        t = self.collector.get_telemetry(
            "BTCUSDT", mock_binance, is_backtest_mode=True, current_price=84000.0
        )
        # In backtest mode, client must NEVER be called
        mock_binance.get_order_book_depth.assert_not_called()
        mock_binance.get_funding_rate.assert_not_called()
        self.assertTrue(t["is_backtest"])
        self.assertEqual(t["order_book"]["source"], "backtest_baseline")
        self.assertEqual(t["funding_rate"]["source"], "backtest_baseline")

    def test_persistence_and_lookahead_free_replay(self):
        mock_binance = MagicMock()
        mock_binance.get_funding_rate.return_value = {"funding_rate_pct": 0.05, "source": "binance_futures"}
        mock_binance.get_order_book_depth.return_value = {"imbalance_pct": -40.0, "source": "binance_spot"}

        # Generate live telemetry to persist
        self.collector.get_telemetry("BTCUSDT", mock_binance, is_backtest_mode=False, current_price=84000.0)
        self.assertTrue(os.path.exists(self.snapshot_path))

        # Replay at a later timestamp
        decision_time = datetime.now(timezone.utc)
        replayed = self.collector.get_telemetry(
            "BTCUSDT", None, is_backtest_mode=True, decision_time=decision_time, current_price=84000.0
        )
        self.assertEqual(replayed["order_book"]["imbalance_pct"], -40.0)

    def test_xgboost_incorporates_auxiliary_telemetry(self):
        engine = JarvisXGBoostEngine()
        snapshot = {
            "auxiliary_telemetry": {
                "funding_rate": {"funding_rate_zscore": 2.5},
                "order_book": {"imbalance_pct": 35.0},
            }
        }
        parts_results = {
            "part1_breakout": {"signal": 1, "breakout_intensity": 1.2},
            "part2_zone": {"demand_distance_pct": 0.5},
            "part3_psychology": {},  # Missing funding zscore, should fallback to aux
            "part4_volume": {},      # Missing bid_ask_imbalance, should fallback to aux
        }
        options_data = {"telemetry": {"pcr": 0.65}}

        features = engine.extract_48_features(snapshot, parts_results, options_data)
        self.assertEqual(len(features), 48)
        # Feature 6 is p3_funding_rate_zscore -> should be 2.5
        self.assertAlmostEqual(features[6], 2.5, places=2)
        # Feature 11 is p4_bid_ask_imbalance -> should be 35.0 / 100.0 = 0.35
        self.assertAlmostEqual(features[11], 0.35, places=2)


if __name__ == "__main__":
    unittest.main()
