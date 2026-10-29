"""Regression tests: cross-source check must compare the SAME asset on both
exchanges (SOL vs SOL, not SOL vs BTC), and staleness must tolerate a
forming 1m candle."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cross_source_uses_active_base_asset():
    """Brain must request Binance price for the selected coin (e.g. SOLUSDT),
    not the hardcoded BTCUSDT default."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'jarvis_FIXED.py'), encoding='utf-8').read()
    assert 'active_base_asset' in src, "brain must track active_base_asset"
    # Raw cross-source fallback was retired in favor of exact Spot cache identity.
    assert "snapshot.identity != ('binance', 'spot', _canonical, _canonical)" in src
    assert "_canonical = _binance_spot_symbol(symbol)" in src
    assert "_fetch_mtf_from_api(getattr(self, 'active_symbol', None))" in src
    assert 'self.binance_data.get_live_price()' not in src



def test_loop_sets_active_symbol_on_brain():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'jarvis_FIXED.py'), encoding='utf-8').read()
    assert 'self.jarvis.active_symbol = symbol' in src
    assert 'self.jarvis.active_base_asset = base_asset' in src


def test_staleness_threshold_candle_aware():
    import jarvis_data_validator as dv
    assert dv.MAX_STALE_SECONDS >= 60, (
        "1m candles need >=60s staleness tolerance")
    assert dv.MAX_STALE_SECONDS <= 300, "must still catch dead feeds"


def test_stale_check_passes_forming_candle():
    import time
    import jarvis_data_validator as dv
    v = dv.get_validator()
    now = time.time()
    # 55-second-old candle timestamp: previously blocked, now must pass
    failures = v._check_staleness({"time": now - 55, "close": 100.0}, now)
    assert failures == [], f"forming candle wrongly blocked: {failures}"
    # genuinely dead feed (10 min) must still block
    failures = v._check_staleness({"time": now - 600, "close": 100.0}, now)
    assert failures, "dead feed must still be blocked"
