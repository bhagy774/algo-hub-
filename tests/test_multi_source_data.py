"""Offline tests for the cross-exchange multi-source perspective.

All network is mocked. Covers:
  - Binance failure -> perspective skipped, never blocks (no veto, no adjustment)
  - Disagreement -> confidence reduced (strong disagreement also vetoes)
  - Agreement -> small confidence boost
  - Upstox without credentials -> skipped gracefully
"""
import time
from unittest.mock import MagicMock, patch

import pytest

import multi_source_data as msd


@pytest.fixture(autouse=True)
def _clear_cache():
    msd._cache.clear()
    yield
    msd._cache.clear()


def _klines(closes):
    """Build a fake Binance klines payload from close prices."""
    return [[0, 0, 0, 0, c, 0] for c in closes]


def _mock_binance_response(closes, price):
    def fake_get(url, params=None, timeout=None):
        assert timeout <= 5.0, "timeout must be <= 5s"
        resp = MagicMock()
        resp.status_code = 200
        if "klines" in url:
            resp.json.return_value = _klines(closes)
        else:
            resp.json.return_value = {"price": str(price)}
        return resp
    return fake_get


# ── Binance fetch behaviour ─────────────────────────────────────────────────

def test_binance_success_normalizes():
    with patch("multi_source_data.requests.get",
               side_effect=_mock_binance_response([100, 101, 102], 102.0)):
        ctx = msd.fetch_binance_context()
    assert ctx["ok"] is True
    assert ctx["source"] == "binance"
    assert ctx["trend"] == "BULLISH"
    assert ctx["price"] == 102.0
    assert ctx["change_pct"] > 0


def test_binance_failure_is_fail_closed_and_not_cached():
    with patch("multi_source_data.requests.get", side_effect=TimeoutError("slow")):
        ctx = msd.fetch_binance_context()
    assert ctx["ok"] is False
    assert ctx["price"] == 0.0
    assert "binance" not in msd._cache  # failures never cached as good


def test_binance_failure_uses_fresh_cache():
    good = {"source": "binance", "symbol": "BTCUSDT", "price": 100.0,
            "trend": "BULLISH", "change_pct": 0.6, "timestamp": time.time(), "ok": True}
    msd._cache["binance"] = good
    with patch("multi_source_data.requests.get", side_effect=ConnectionError("down")):
        ctx = msd.fetch_binance_context()
    assert ctx["ok"] is True and ctx["price"] == 100.0  # served from cache


def test_binance_stale_cache_not_used():
    msd._cache["binance"] = {"source": "binance", "symbol": "BTCUSDT", "price": 100.0,
                             "trend": "BULLISH", "change_pct": 0.6,
                             "timestamp": time.time() - 120, "ok": True}
    with patch("multi_source_data.requests.get", side_effect=ConnectionError("down")):
        ctx = msd.fetch_binance_context()
    assert ctx["ok"] is False  # stale cache (>60s) must not be served


# ── Evaluation logic ────────────────────────────────────────────────────────

def _ctx(trend, change_pct, ok=True, source="binance"):
    return {"source": source, "symbol": "BTCUSDT", "price": 100.0,
            "trend": trend, "change_pct": change_pct,
            "timestamp": time.time(), "ok": ok}


def test_binance_failure_skipped_not_blocking():
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BEARISH", -2.0, ok=False),
                                      upstox_ctx=None)
    assert res["active"] is False
    assert res["adjustment"] == 0
    assert res["veto"] is False  # never blocks when Binance unreachable


def test_agreement_boosts_confidence():
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BULLISH", 0.8))
    assert res["adjustment"] == msd.AGREE_BOOST
    assert res["veto"] is False
    res = msd.evaluate_cross_exchange(-1, binance_ctx=_ctx("BEARISH", -0.8))
    assert res["adjustment"] == msd.AGREE_BOOST


def test_mild_disagreement_reduces_confidence_no_veto():
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BEARISH", -0.2))
    assert res["adjustment"] == -msd.DISAGREE_PENALTY
    assert res["veto"] is False


def test_strong_disagreement_vetoes_entry():
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BEARISH", -1.5))
    assert res["adjustment"] == -msd.STRONG_DISAGREE_PENALTY
    assert res["veto"] is True


def test_no_signal_is_inactive():
    res = msd.evaluate_cross_exchange(0, binance_ctx=_ctx("BULLISH", 2.0))
    assert res["active"] is False and res["adjustment"] == 0 and res["veto"] is False


def test_upstox_informational_only_never_vetoes():
    up = _ctx("BEARISH", -3.0, source="upstox")
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BULLISH", 1.0), upstox_ctx=up)
    assert res["veto"] is False  # Upstox can never veto
    assert res["adjustment"] == msd.AGREE_BOOST - msd.UPSTOX_BIAS


# ── Upstox graceful degradation ─────────────────────────────────────────────

def test_upstox_without_credentials_skipped_gracefully():
    fake_ud = MagicMock()
    fake_ud._enabled = False  # no token configured
    with patch("upstox_data.get_upstox_data", return_value=fake_ud):
        ctx = msd.fetch_upstox_context()
    assert ctx["ok"] is False
    assert ctx["source"] == "upstox"
    # and it contributes nothing to evaluation
    res = msd.evaluate_cross_exchange(1, binance_ctx=_ctx("BULLISH", 1.0), upstox_ctx=ctx)
    assert res["adjustment"] == msd.AGREE_BOOST  # only the Binance boost


def test_upstox_api_error_skipped_gracefully():
    fake_ud = MagicMock()
    fake_ud._enabled = True
    fake_ud.get_full_quote.return_value = {}  # API failed
    with patch("upstox_data.get_upstox_data", return_value=fake_ud):
        ctx = msd.fetch_upstox_context()
    assert ctx["ok"] is False


# ── End-to-end perspective helper ───────────────────────────────────────────

def test_perspective_helper_never_raises_and_skips_on_outage():
    with patch("multi_source_data.requests.get", side_effect=ConnectionError("no internet")):
        fake_ud = MagicMock(); fake_ud._enabled = False
        with patch("upstox_data.get_upstox_data", return_value=fake_ud):
            res = msd.get_cross_exchange_perspective(1)
    assert res["active"] is False
    assert res["veto"] is False
    assert res["adjustment"] == 0
