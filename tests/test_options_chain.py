from datetime import datetime, timezone, timedelta

from options_chain import build_provider_chain, payout_max_pain

AS_OF = datetime.now(timezone.utc)


def row(symbol, kind, strike, expiry="2026-10-30", **extra):
    item = {
        "symbol": symbol,
        "underlying_asset_symbol": "BTC",
        "contract_type": "call_options" if kind == "call" else "put_options",
        "strike_price": strike,
        "expiry": expiry,
        "timestamp": AS_OF.isoformat(),
        "best_bid": 10,
        "best_ask": 12,
        "oi": 10,
        "volume": 2,
        "implied_volatility": 0.6,
        "greeks": {"delta": 0.25 if kind == "call" else -0.25,
                    "gamma": 0.001, "theta": -0.1, "vega": 0.3, "rho": 0.02},
    }
    item.update(extra)
    return item


def pagination(**extra):
    return {"complete": True, "truncated": False, "marker_seen": True,
            "termination_evidence": "provider_terminal_pagination_marker", **extra}


def test_all_pages_strikes_expiries_and_observed_greeks_are_retained():
    page1 = [row("BTC-261030-90000-C", "call", 90000),
             row("BTC-261030-90000-P", "put", 90000)]
    sparse = row("BTC-261106-100000-C", "call", 100000, expiry="2026-11-06")
    sparse["greeks"] = {"delta": 0.2}
    sparse["best_bid"] = None
    sparse["implied_volatility"] = None
    page2 = [sparse, row("BTC-261106-110000-P", "put", 110000, expiry="2026-11-06")]
    chain = build_provider_chain("BTC", "Delta", [page1, page2], pagination(pages_received=2), retrieved_at=AS_OF)
    assert chain["validation"]["usable"] is True
    assert chain["validation"]["pagination"]["pages_received"] == 2
    assert len(chain["contracts"]) == 4
    assert {c["expiry"] for c in chain["contracts"]} == {"2026-10-30", "2026-11-06"}
    assert set(chain["coverage"]["by_expiry"]["2026-10-30"]["strikes"]) == {90000}
    assert chain["coverage"]["greek_observed_count"] == {"delta": 4, "gamma": 3, "theta": 3, "vega": 3, "rho": 3}
    sparse_out = next(c for c in chain["contracts"] if c["symbol"] == "BTC-261106-100000-C")
    assert sparse_out["delta"] == 0.2 and sparse_out["gamma"] is None
    assert sparse_out["bid"] is None and sparse_out["implied_volatility"] is None
    assert "gamma" in sparse_out["missing_fields"] and "bid" in sparse_out["missing_fields"]
    assert sparse_out["greeks_model_derived"] is False


def test_pagination_truncation_or_missing_evidence_fails_closed():
    good = [row("BTC-261030-90000-C", "call", 90000)]
    truncated = build_provider_chain("BTC", "Delta", [good],
        {"complete": False, "truncated": True, "marker_seen": True}, retrieved_at=AS_OF)
    no_marker = build_provider_chain("BTC", "Delta", [good],
        {"complete": True, "truncated": False, "marker_seen": False}, retrieved_at=AS_OF)
    assert not truncated["validation"]["usable"]
    assert "pagination_incomplete_or_truncated" in truncated["validation"]["reasons"]
    assert not no_marker["validation"]["usable"]


def test_stale_identity_mismatch_and_provider_error_are_unusable():
    stale = row("BTC-261030-90000-C", "call", 90000,
                timestamp=(AS_OF - timedelta(minutes=11)).isoformat())
    assert build_provider_chain("BTC", "Delta", [[stale]], pagination(), retrieved_at=AS_OF)["validation"]["freshness_status"] == "stale"
    other = row("ETH-261030-90000-C", "call", 90000)
    other["underlying_asset_symbol"] = "ETH"
    mismatch = build_provider_chain("BTC", "Delta", [[other]], pagination(), retrieved_at=AS_OF)
    assert mismatch["validation"]["identity_valid"] is False
    assert not mismatch["validation"]["usable"]
    error = build_provider_chain("BTC", "Delta", [], pagination(), retrieved_at=AS_OF, provider_error="offline")
    assert not error["validation"]["usable"]
    assert "provider_error" in error["validation"]["reasons"]


def test_expired_and_malformed_rows_are_not_silently_used():
    expired = row("BTC-260901-90000-C", "call", 90000, expiry="2026-09-01")
    chain = build_provider_chain("BTC", "Delta", [[expired]], pagination(), retrieved_at=AS_OF)
    assert chain["contracts"] == []
    assert chain["validation"]["same_day_expiry_rows_excluded"] == 1
    bad = row("BTC-261030-0-C", "call", 0)
    invalid = build_provider_chain("BTC", "Delta", [[bad]], pagination(), retrieved_at=AS_OF)
    assert invalid["validation"]["schema_valid"] is False


def test_max_pain_is_payout_minimum_not_oi_peak():
    contracts = [
        {"expiry": "2026-10-30", "strike": 90000, "type": "call", "open_interest": 10},
        {"expiry": "2026-10-30", "strike": 90000, "type": "put", "open_interest": 100},
        {"expiry": "2026-10-30", "strike": 110000, "type": "call", "open_interest": 1000},
        {"expiry": "2026-10-30", "strike": 110000, "type": "put", "open_interest": 1},
    ]
    pain = payout_max_pain(contracts)
    oi_peak = max({s: sum(c["open_interest"] for c in contracts if c["strike"] == s)
                   for s in {c["strike"] for c in contracts}}, key=lambda k: sum(c["open_interest"] for c in contracts if c["strike"] == k))
    assert oi_peak == 110000
    assert pain["available"] and pain["strike"] == 90000
    assert "argmin" in pain["method"]
    assert "unsigned open interest gives no dealer positioning direction" in pain["assumptions"]
    mixed = payout_max_pain([{**contracts[0], "contract_multiplier": 0.1}, contracts[1]])
    assert mixed["available"] is False
    assert mixed["by_expiry"]["2026-10-30"]["reason"] == "mixed_known_and_missing_contract_multipliers"


def test_max_pain_is_never_aggregated_across_different_expiries():
    rows = [
        {"expiry":"2026-10-30", "strike":90, "type":"call", "open_interest":10},
        {"expiry":"2026-10-30", "strike":90, "type":"put", "open_interest":100},
        {"expiry":"2026-10-30", "strike":110, "type":"call", "open_interest":1000},
        {"expiry":"2026-10-30", "strike":110, "type":"put", "open_interest":1},
        {"expiry":"2026-11-06", "strike":90, "type":"call", "open_interest":1},
        {"expiry":"2026-11-06", "strike":90, "type":"put", "open_interest":1},
        {"expiry":"2026-11-06", "strike":110, "type":"call", "open_interest":1},
        {"expiry":"2026-11-06", "strike":110, "type":"put", "open_interest":100},
    ]
    pain = payout_max_pain(rows)
    assert pain["available"] and pain["selected_expiry"] == "2026-10-30"
    assert pain["by_expiry"]["2026-10-30"]["strike"] == 90
    assert pain["by_expiry"]["2026-11-06"]["strike"] == 110


def test_iv_metrics_require_explicit_data_and_state_method():
    contracts = [
        {"expiry": "2026-10-30", "strike": 100, "type": "call", "delta": 0.25, "implied_volatility": 0.60, "open_interest": 1},
        {"expiry": "2026-10-30", "strike": 100, "type": "put", "delta": -0.25, "implied_volatility": 0.65, "open_interest": 1},
    ]
    from options_chain import analyze_provider_chain
    metrics = analyze_provider_chain(contracts)
    assert metrics["iv"]["iv_skew"]["available"]
    assert abs(metrics["iv"]["iv_skew"]["by_expiry"]["2026-10-30"]["risk_reversal_call_minus_put"] + 0.05) < 1e-9
    assert not metrics["iv"]["iv_term_structure"]["available"]
    assert metrics["dealer_gamma_direction"]["available"] is False
    aligned = analyze_provider_chain(contracts, spot=100, spot_as_of=AS_OF, chain_as_of=AS_OF)
    future = analyze_provider_chain(contracts, spot=100, spot_as_of=AS_OF + timedelta(seconds=1), chain_as_of=AS_OF)
    assert aligned["iv"]["iv_term_structure"]["available"]
    assert not future["iv"]["iv_term_structure"]["available"]
    assert future["iv"]["iv_term_structure"]["reason"] == "underlying_price_after_or_unaligned_with_chain_as_of"
