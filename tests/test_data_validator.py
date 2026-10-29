"""Offline tests for jarvis_data_validator.py — Pre-Brain Data Quality Gate.

All tests are fully offline (no network). Covers:
  - Completeness: missing/NaN/Inf OHLC fields rejected
  - Price sanity: zero/negative prices rejected, 20%+ spikes rejected
  - Staleness: data older than MAX_STALE_SECONDS rejected, future rejected
  - Cross-source: > 1% divergence warns, > 3% divergence blocks
  - Multi-source combined validation
  - Spike filter does NOT update last price (so recovery works)
  - Validator disabled via env var → all data passes
  - Internal errors → fail-open (never blocks brain)
  - Stats tracking accuracy
  - DataFrame validation path
"""

import math
import os
import time
from unittest.mock import patch

import pytest

# Ensure validator starts enabled for tests
os.environ.pop("JARVIS_DATA_VALIDATOR", None)

import jarvis_data_validator as jdv
from jarvis_data_validator import (
    JarvisDataValidator,
    ValidationResult,
    _is_valid_number,
)


@pytest.fixture
def validator():
    """Fresh validator instance for each test."""
    v = JarvisDataValidator()
    return v


def _candle(open=100.0, high=101.0, low=99.0, close=100.5,
            volume=1000.0, ts=None):
    """Build a valid candle dict."""
    c = {"open": open, "high": high, "low": low, "close": close,
         "volume": volume}
    if ts is not None:
        c["time"] = ts
    return c


# ══════════════════════════════════════════════════════════════════════════════
#  COMPLETENESS CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestCompleteness:
    def test_valid_candle_passes(self, validator):
        r = validator.validate_candle(_candle(), source="test")
        assert r.ok is True
        assert r.status == "VALID"

    def test_missing_open_fails(self, validator):
        c = _candle()
        del c["open"]
        r = validator.validate_candle(c, source="test")
        assert r.ok is False
        assert any("Missing open" in f for f in r.failures)

    def test_missing_close_fails(self, validator):
        c = _candle()
        del c["close"]
        r = validator.validate_candle(c, source="test")
        assert r.ok is False
        assert any("Missing close" in f for f in r.failures)

    def test_missing_high_low_fails(self, validator):
        c = _candle()
        del c["high"]
        del c["low"]
        r = validator.validate_candle(c, source="test")
        assert r.ok is False
        assert len(r.failures) == 2

    def test_nan_close_fails(self, validator):
        r = validator.validate_candle(_candle(close=float('nan')), source="test")
        assert r.ok is False
        assert any("Invalid close" in f for f in r.failures)

    def test_inf_open_fails(self, validator):
        r = validator.validate_candle(_candle(open=float('inf')), source="test")
        assert r.ok is False
        assert any("Invalid open" in f for f in r.failures)

    def test_none_high_fails(self, validator):
        r = validator.validate_candle(_candle(high=None), source="test")
        assert r.ok is False

    def test_string_value_fails(self, validator):
        r = validator.validate_candle(_candle(low="not_a_number"), source="test")
        assert r.ok is False


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestPriceSanity:
    def test_zero_price_fails(self, validator):
        r = validator.validate_candle(_candle(close=0.0), source="test")
        assert r.ok is False
        assert any("<= 0" in f for f in r.failures)

    def test_negative_price_fails(self, validator):
        r = validator.validate_candle(_candle(close=-50.0), source="test")
        assert r.ok is False

    def test_first_candle_always_passes_price_sanity(self, validator):
        """No last price → can't check spike → passes."""
        r = validator.validate_candle(_candle(close=50000.0), source="btc")
        assert r.ok is True

    def test_normal_price_change_passes(self, validator):
        """< 20% change should pass."""
        validator.validate_candle(_candle(close=100.0), source="test")
        r = validator.validate_candle(_candle(close=115.0), source="test")
        assert r.ok is True  # 15% < 20%

    def test_spike_over_20pct_fails(self, validator):
        """Jump > 20% from last price → rejected."""
        validator.validate_candle(_candle(close=100.0), source="test")
        r = validator.validate_candle(_candle(close=125.0), source="test")
        assert r.ok is False
        assert any("spike" in f.lower() for f in r.failures)

    def test_spike_drop_fails(self, validator):
        """Drop > 20% from last price → rejected."""
        validator.validate_candle(_candle(close=100.0), source="test")
        r = validator.validate_candle(_candle(close=70.0), source="test")
        assert r.ok is False
        assert any("spike" in f.lower() for f in r.failures)

    def test_spike_does_not_update_last_price(self, validator):
        """After spike rejection, last price stays at the good value."""
        validator.validate_candle(_candle(close=100.0), source="test")
        # Spike — rejected, last_price should stay 100
        validator.validate_candle(_candle(close=200.0), source="test")
        # Normal candle relative to 100 → should pass
        r = validator.validate_candle(_candle(close=110.0), source="test")
        assert r.ok is True  # 10% from 100, not 45% from 200

    def test_separate_sources_track_separately(self, validator):
        """Each source has its own last price tracker."""
        validator.validate_candle(_candle(close=100.0), source="delta")
        validator.validate_candle(_candle(close=50000.0), source="binance")
        # Delta: 115 vs 100 = 15% → OK
        r1 = validator.validate_candle(_candle(close=115.0), source="delta")
        assert r1.ok is True
        # Binance: 55000 vs 50000 = 10% → OK
        r2 = validator.validate_candle(_candle(close=55000.0), source="binance")
        assert r2.ok is True


# ══════════════════════════════════════════════════════════════════════════════
#  STALENESS CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestStaleness:
    def test_fresh_data_passes(self, validator):
        now = time.time()
        r = validator.validate_candle(_candle(ts=now - 5), source="test", now=now)
        assert r.ok is True

    def test_stale_data_beyond_limit_fails(self, validator):
        import jarvis_data_validator as dv
        now = time.time()
        r = validator.validate_candle(
            _candle(ts=now - dv.MAX_STALE_SECONDS - 5), source="test", now=now)
        assert r.ok is False
        assert any("stale" in f.lower() for f in r.failures)

    def test_within_limit_passes(self, validator):
        import jarvis_data_validator as dv
        now = time.time()
        r = validator.validate_candle(
            _candle(ts=now - (dv.MAX_STALE_SECONDS - 5)), source="test", now=now)
        assert r.ok is True

    def test_millisecond_timestamp_auto_converted(self, validator):
        """Timestamps > 1e12 are treated as milliseconds."""
        now = time.time()
        ts_ms = (now - 5) * 1000  # 5 seconds ago in ms
        r = validator.validate_candle(_candle(ts=ts_ms), source="test", now=now)
        assert r.ok is True

    def test_future_timestamp_fails(self, validator):
        now = time.time()
        r = validator.validate_candle(_candle(ts=now + 400), source="test", now=now)
        assert r.ok is False
        assert any("future" in f.lower() for f in r.failures)

    def test_no_timestamp_passes(self, validator):
        """If candle has no timestamp, staleness check is skipped."""
        r = validator.validate_candle(_candle(), source="test")
        assert r.ok is True


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-SOURCE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossSource:
    def test_same_price_passes(self, validator):
        r = validator.cross_source_check(100.0, 100.0)
        assert r.ok is True

    def test_small_divergence_passes(self, validator):
        r = validator.cross_source_check(100.0, 100.5)
        assert r.ok is True  # 0.5% < 1%
        assert len(r.warnings) == 0

    def test_1pct_divergence_warns(self, validator):
        r = validator.cross_source_check(100.0, 101.5)
        assert r.ok is True  # warning, not block
        assert len(r.warnings) > 0
        assert any("WARNING" in w for w in r.warnings)

    def test_3pct_divergence_blocks(self, validator):
        r = validator.cross_source_check(100.0, 104.0)
        assert r.ok is False
        assert r.status == "NO-DATA"
        assert any("BLOCK" in f for f in r.failures)

    def test_zero_price_a_passes_through(self, validator):
        """Can't compare if one price is invalid → pass through."""
        r = validator.cross_source_check(0.0, 100.0)
        assert r.ok is True

    def test_negative_price_passes_through(self, validator):
        r = validator.cross_source_check(-50.0, 100.0)
        assert r.ok is True


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-SOURCE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiSource:
    def test_all_good_passes(self, validator):
        now = time.time()
        sources = {
            "delta": _candle(close=100.0, ts=now - 2),
            "binance": _candle(close=100.3, ts=now - 3),
        }
        r = validator.validate_multi_source(sources, now=now)
        assert r.ok is True

    def test_one_source_stale_fails(self, validator):
        now = time.time()
        sources = {
            "delta": _candle(close=100.0, ts=now - 2),
            "binance": _candle(close=100.3, ts=now - 400),  # stale
        }
        r = validator.validate_multi_source(sources, now=now)
        assert r.ok is False

    def test_cross_source_block_in_multi(self, validator):
        now = time.time()
        sources = {
            "delta": _candle(close=100.0, ts=now - 2),
            "binance": _candle(close=105.0, ts=now - 2),  # 5% divergence
        }
        r = validator.validate_multi_source(sources, now=now)
        assert r.ok is False

    def test_none_source_skipped(self, validator):
        now = time.time()
        sources = {
            "delta": _candle(close=100.0, ts=now - 2),
            "binance": None,  # unavailable
        }
        r = validator.validate_multi_source(sources, now=now)
        assert r.ok is True  # delta alone is fine, no cross-check possible


# ══════════════════════════════════════════════════════════════════════════════
#  DATAFRAME VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestDataFrame:
    def test_valid_df_passes(self, validator):
        import pandas as pd
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0]
        })
        r = validator.validate_dataframe(df, source="test")
        assert r.ok is True

    def test_empty_df_fails(self, validator):
        import pandas as pd
        df = pd.DataFrame()
        r = validator.validate_dataframe(df, source="test")
        assert r.ok is False
        assert r.status == "NO-DATA"

    def test_none_df_fails(self, validator):
        r = validator.validate_dataframe(None, source="test")
        assert r.ok is False
        assert r.status == "NO-DATA"

    def test_df_with_nan_fails(self, validator):
        import pandas as pd
        df = pd.DataFrame({
            "open": [100.0], "high": [float('nan')], "low": [99.0],
            "close": [100.5], "volume": [1000.0]
        })
        r = validator.validate_dataframe(df, source="test")
        assert r.ok is False

    def test_closed_5m_dataframe_within_two_intervals_passes(self, validator):
        import pandas as pd
        now = time.time()
        # Closed 5m bars: latest closed bar opened 580s ago (formed 580s..280s ago, current forming bar 280s old)
        idx = pd.to_datetime([now - 880, now - 580], unit="s", utc=True)
        df = pd.DataFrame({
            "open": [100.0, 100.2], "high": [101.0, 101.2], "low": [99.0, 99.2],
            "close": [100.5, 100.8], "volume": [1000.0, 1100.0]
        }, index=idx)
        r = validator.validate_dataframe(df, source="binance_spot", now=now, symbol="BTCUSDT")
        assert r.ok is True

    def test_df_with_datetime_index_staleness(self, validator):
        import pandas as pd
        now = time.time()
        ts = pd.Timestamp.now() - pd.Timedelta(seconds=400)
        df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0],
             "close": [100.5], "volume": [1000.0]},
            index=pd.DatetimeIndex([ts])
        )
        r = validator.validate_dataframe(df, source="test", now=now)
        assert r.ok is False  # older than staleness limit


# ══════════════════════════════════════════════════════════════════════════════
#  STATS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_increment(self, validator):
        validator.validate_candle(_candle(), source="test")
        validator.validate_candle(_candle(close=0.0), source="test")
        stats = validator.get_stats()
        assert stats["total_checks"] == 2
        assert stats["passed"] == 1
        assert stats["rejected"] == 1

    def test_stats_reset(self, validator):
        validator.validate_candle(_candle(), source="test")
        validator.reset_stats()
        stats = validator.get_stats()
        assert stats["total_checks"] == 0
        assert stats["passed"] == 0

    def test_status_line_format(self, validator):
        validator.validate_candle(_candle(), source="test")
        line = validator.status_line()
        assert "DATA-VALIDATOR" in line
        assert "ON" in line

    def test_reject_reason_tracked(self, validator):
        now = time.time()
        validator.validate_candle(_candle(ts=now - 400), source="test", now=now)
        stats = validator.get_stats()
        assert stats["rejects_by_reason"]["staleness"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
#  DISABLED MODE
# ══════════════════════════════════════════════════════════════════════════════

class TestDisabled:
    def test_disabled_passes_everything(self):
        v = JarvisDataValidator()
        v._enabled = False
        # Even obviously bad data passes when disabled
        r = v.validate_candle({"close": float('nan')}, source="test")
        assert r.ok is True

    def test_disabled_cross_source_passes(self):
        v = JarvisDataValidator()
        v._enabled = False
        r = v.cross_source_check(100.0, 200.0)
        assert r.ok is True

    def test_disabled_status_line(self):
        v = JarvisDataValidator()
        v._enabled = False
        line = v.status_line()
        assert "DISABLED" in line


# ══════════════════════════════════════════════════════════════════════════════
#  FAIL-OPEN ON INTERNAL ERROR
# ══════════════════════════════════════════════════════════════════════════════

class TestFailOpen:
    def test_completeness_crash_is_fail_open(self, validator):
        """If _check_completeness crashes, validator still returns VALID."""
        with patch.object(validator, "_check_completeness", side_effect=RuntimeError("boom")):
            r = validator.validate_candle(_candle(), source="test")
        assert r.ok is True  # fail-open

    def test_cross_source_crash_is_fail_open(self, validator):
        with patch("jarvis_data_validator._is_valid_number", side_effect=RuntimeError("boom")):
            r = validator.cross_source_check(100.0, 200.0)
        assert r.ok is True  # fail-open

    def test_dataframe_crash_is_fail_open(self, validator):
        """If DataFrame validation crashes internally, returns VALID."""
        r = validator.validate_dataframe("not_a_dataframe", source="test")
        assert r.ok is True  # fail-open on TypeError


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_is_valid_number(self):
        assert _is_valid_number(100.0) is True
        assert _is_valid_number(0) is True
        assert _is_valid_number(-5) is True
        assert _is_valid_number("100") is True  # string that can be floated
        assert _is_valid_number(None) is False
        assert _is_valid_number(float('nan')) is False
        assert _is_valid_number(float('inf')) is False
        assert _is_valid_number("abc") is False

    def test_validation_result_bool(self):
        assert bool(ValidationResult(ok=True, status="VALID")) is True
        assert bool(ValidationResult(ok=False, status="WAIT")) is False


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleton:
    def test_get_validator_returns_instance(self):
        # Reset singleton for clean test
        jdv._instance = None
        v = jdv.get_validator()
        assert isinstance(v, JarvisDataValidator)
        # Same instance on second call
        v2 = jdv.get_validator()
        assert v is v2
        # Cleanup
        jdv._instance = None
