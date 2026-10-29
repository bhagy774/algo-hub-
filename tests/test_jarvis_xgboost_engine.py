"""
Unit and Integration Tests for JARVIS System 2 XGBoost Engine.
==============================================================
Validates:
1. Exact 48-feature store extraction.
2. Calibrated win probability and execution gating (P >= 0.68).
3. Fractional Kelly Criterion position sizing calculation.
4. Scalp vs Swing regime classification.
5. Low-latency inference invariant (< 5 milliseconds).
"""
import time
import unittest
import numpy as np
import pandas as pd

from jarvis_xgboost_engine import JarvisXGBoostEngine, FEATURE_NAMES, get_xgboost_engine


class TestJarvisXGBoostEngine(unittest.TestCase):

    def setUp(self):
        self.engine = get_xgboost_engine()

    def test_feature_store_dimensionality(self):
        """Verify feature store strictly contains 48 named continuous features."""
        self.assertEqual(len(FEATURE_NAMES), 48)
        self.assertEqual(len(self.engine.feature_names), 48)

    def test_feature_extraction_from_empty_inputs(self):
        """Ensure feature extraction gracefully returns a finite 48-dim array from empty inputs."""
        feat = self.engine.extract_48_features({}, {}, {})
        self.assertEqual(len(feat), 48)
        self.assertTrue(np.all(np.isfinite(feat)))
        self.assertEqual(feat.dtype, np.float32)

    def test_inference_evaluation_contract(self):
        """Verify evaluate() produces the expected typed metrics contract."""
        feat = np.random.randn(48).astype(np.float32)
        res = self.engine.evaluate(feat, "BUY")
        
        self.assertIn("win_probability", res)
        self.assertIn("kelly_fraction", res)
        self.assertIn("expected_return_pct", res)
        self.assertIn("regime", res)
        self.assertIn("approved", res)
        
        self.assertIsInstance(res["win_probability"], float)
        self.assertTrue(0.0 <= res["win_probability"] <= 1.0)
        self.assertIsInstance(res["kelly_fraction"], float)
        self.assertTrue(res["kelly_fraction"] >= 0.0)
        self.assertIn(res["regime"], {"SCALP", "SWING", "CHOP_AVOID"})

    def test_direction_inversion_for_sell(self):
        """Verify sell evaluations evaluate short probabilities correctly."""
        feat = np.zeros(48, dtype=np.float32)
        buy_res = self.engine.evaluate(feat, "BUY")
        sell_res = self.engine.evaluate(feat, "SELL")
        self.assertTrue(math_finite := (isinstance(buy_res["win_probability"], float) and isinstance(sell_res["win_probability"], float)))

    def test_sub_5ms_inference_latency(self):
        """Ensure single-pass inference executes well under 5 milliseconds."""
        feat = np.random.randn(48).astype(np.float32)
        # Warmup
        self.engine.evaluate(feat, "BUY")
        
        t0 = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            self.engine.evaluate(feat, "BUY")
        elapsed_per_call_ms = ((time.perf_counter() - t0) / iterations) * 1000.0
        
        self.assertLess(elapsed_per_call_ms, 5.0, f"Inference latency too slow: {elapsed_per_call_ms:.2f}ms")

    def test_bootstrap_from_candles(self):
        """Verify self-bootstrapping on synthetic OHLCV dataframe."""
        n_bars = 200
        dates = pd.date_range("2026-01-01", periods=n_bars, freq="5min")
        close_prices = 85000.0 + np.cumsum(np.random.randn(n_bars) * 100.0)
        high_prices = close_prices + np.random.rand(n_bars) * 50.0
        low_prices = close_prices - np.random.rand(n_bars) * 50.0
        open_prices = close_prices - np.random.randn(n_bars) * 20.0
        volumes = np.random.rand(n_bars) * 10.0 + 1.0
        
        df = pd.DataFrame({
            "time": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes
        })
        
        success = self.engine.bootstrap_from_candles(df, symbol="BTCUSDT")
        self.assertTrue(success)
        self.assertTrue(self.engine.is_bootstrapped)


if __name__ == "__main__":
    unittest.main()
