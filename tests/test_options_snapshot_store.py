from datetime import datetime, timezone, timedelta

from options_chain import build_provider_chain, combine_provider_chains
from options_snapshot_store import append_snapshot, replay_snapshot, replay_institutional_bias

AS_OF = datetime(2026, 9, 28, 10, tzinfo=timezone.utc)


def chain():
    row = {"symbol": "ETH-261030-4000-C", "underlying_asset_symbol": "ETH",
           "contract_type": "call_options", "strike_price": 4000,
           "expiry": "2026-10-30", "timestamp": AS_OF.isoformat(),
           "oi": 2, "volume": 1, "implied_volatility": 0.5,
           "greeks": {"delta": 0.3}}
    provider = build_provider_chain("ETH", "Delta", [[row]],
        {"complete": True, "truncated": False, "marker_seen": True}, retrieved_at=AS_OF)
    return combine_provider_chains("ETH", {"Delta": provider})


def test_recorded_snapshot_replays_only_at_or_after_its_point_in_time(tmp_path):
    path = tmp_path / "options.jsonl"
    assert append_snapshot(str(path), chain(), captured_at=AS_OF)
    decision = AS_OF + timedelta(minutes=5)
    out = replay_snapshot(str(path), "ETHUSDT", decision, venue="Delta")
    assert out["available"]
    assert out["source"] == "recorded_options_snapshots"
    assert out["not_ohlc_derived"]
    assert out["snapshot"]["providers"]["Delta"]["contracts"][0]["delta"] == 0.3
    future = replay_snapshot(str(path), "ETH", AS_OF - timedelta(seconds=1), venue="Delta")
    assert not future["available"]
    assert "future_or_invalid_capture_time" in future["rejected_reasons"] or "no_admissible_historical_options_snapshot" == future["reason"]


def test_replay_rejects_wrong_asset_stale_chain_and_absent_history(tmp_path):
    path = tmp_path / "options.jsonl"
    assert append_snapshot(str(path), chain(), captured_at=AS_OF)
    assert replay_snapshot(str(path), "BTC", AS_OF + timedelta(minutes=1), venue="Delta")["available"] is False
    assert replay_snapshot(str(path), "ETH", AS_OF + timedelta(minutes=11), venue="Delta")["available"] is False
    missing = replay_snapshot(str(tmp_path / "missing.jsonl"), "ETH", AS_OF)
    assert not missing["available"] and missing["source"] == "none"
    assert missing["not_ohlc_derived"]


def test_future_capture_cannot_be_used_for_earlier_decision(tmp_path):
    path = tmp_path / "options.jsonl"
    assert append_snapshot(str(path), chain(), captured_at=AS_OF + timedelta(seconds=1))
    out = replay_snapshot(str(path), "ETH", AS_OF, venue="Delta")
    assert not out["available"]
    assert "future_or_invalid_capture_time" in out["rejected_reasons"]


def _provider(venue, retrieved_at, *, call_oi=100, put_oi=160):
    rows = []
    for side, suffix, oi in (("call_options", "C", call_oi), ("put_options", "P", put_oi)):
        rows.append({"symbol": f"ETH-261030-4000-{suffix}", "underlying_asset_symbol": "ETH",
                     "contract_type": side, "strike_price": 4000,
                     "expiry": "2026-10-30", "timestamp": retrieved_at.isoformat(),
                     "oi": oi, "volume": 1, "implied_volatility": 0.5})
    return build_provider_chain("ETH", venue, [rows],
        {"complete": True, "truncated": False, "marker_seen": True}, retrieved_at=retrieved_at)


def test_bias_replay_uses_only_provider_that_passed_point_in_time_validation(tmp_path):
    path = tmp_path / "options.jsonl"
    stale_delta = _provider("Delta", AS_OF - timedelta(minutes=30))
    fresh_binance = _provider("Binance Options", AS_OF + timedelta(seconds=90))
    snapshot = combine_provider_chains("ETH", {"Delta": stale_delta, "Binance Options": fresh_binance})
    assert append_snapshot(str(path), snapshot, captured_at=AS_OF + timedelta(seconds=90))
    result = replay_institutional_bias(str(path), "ETH", AS_OF + timedelta(seconds=100))
    assert result["available"]
    assert result["source"] == "Binance Options"


def test_bias_replay_rejects_incomplete_open_interest_instead_of_scoring_subset(tmp_path):
    path = tmp_path / "options.jsonl"
    incomplete = _provider("Delta", AS_OF, call_oi=100, put_oi=None)
    snapshot = combine_provider_chains("ETH", {"Delta": incomplete})
    assert snapshot["validation"]["usable"]
    assert not snapshot["providers"]["Delta"]["metrics"]["open_interest_complete"]
    assert append_snapshot(str(path), snapshot, captured_at=AS_OF)
    result = replay_institutional_bias(str(path), "ETH", AS_OF + timedelta(minutes=1))
    assert not result["available"]
    assert result["reason"] == "complete_historical_open_interest_and_pcr_required"
