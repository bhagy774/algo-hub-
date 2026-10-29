from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from jarvis_multicoin_execution import (
    CandidateRejected, DeterministicPaperAdapter, PortfolioCoordinator,
    candidate_from_analysis,
)
from jarvis_strategy_approval import REQUIRED_TIMEFRAMES, evaluate_mtf_central_strategy, make_entry_approval


NOW = 1_800_000_000.0
POLICIES = {"BTC": "btc-paper-v1", "ETH": "eth-paper-v1", "SOL": "sol-paper-v1"}


def bullish_evidence():
    return {
        "part1_breakout": {"signal": 1, "thought": "bullish breakout"},
        "part2_zone": {"signal": 1, "thought": "bullish demand zone"},
        "part3_psychology": {"signal": 1, "thought": "bullish candle"},
        "part4_volume": {"signal": 1, "thought": "volume confirmation"},
        "part5_ml": {"signal": 1, "thought": "model confirms"},
        "part6_trend": {"signal": 1, "thought": "trend bullish"},
        "part7_volatility": {"signal": 1, "thought": "volatility expansion"},
        "part8_structure": {"signal": 1, "thought": "structure bullish"},
        "part9_orderflow": {"signal": 1, "thought": "orderflow bullish"},
        "part10_candlestats": {"signal": 1, "thought": "candlestats bullish"},
    }


def part7_gate(symbol):
    return {
        "symbol": symbol, "entry_blocked": False, "risk_veto": False,
        "status": "ok", "data_status": "valid", "timeframe": "aggregate",
        "blocked_timeframes": [], "veto_timeframes": [],
        "timeframe_results": {
            tf: {"symbol": symbol, "timeframe": tf, "status": "neutral",
                 "data_status": "valid", "entry_blocked": False, "risk_veto": False}
            for tf in REQUIRED_TIMEFRAMES
        },
    }


def timeframe_evidence(symbol):
    result = {}
    for timeframe in REQUIRED_TIMEFRAMES:
        row = {}
        for name, item in bullish_evidence().items():
            scoped = {**item, "symbol": symbol, "timeframe": timeframe}
            if name == "part7_volatility":
                scoped.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                               "data_status": "valid", "entry_blocked": False, "risk_veto": False})
            if name == "part2_zone":
                scoped["native_timeframe"] = timeframe
            row[name] = scoped
        result[timeframe] = row
    return result


def analysis(symbol="BTCUSDT", product_id="101", *, plan=None, status="COMPLETE", fetched=NOW-10, completed=NOW-2):
    asset = symbol
    for quote in ("USDT", "USD"):
        if asset.endswith(quote):
            asset = asset[:-len(quote)]
            break
    identity = {"venue": "delta", "market_type": "perpetual", "instrument_id": product_id, "symbol": symbol}
    evidence = bullish_evidence()
    native_evidence = timeframe_evidence(symbol)
    gate = part7_gate(symbol)
    decision = evaluate_mtf_central_strategy(native_evidence, gate, confidence=80, expected_symbol=symbol)
    approval = make_entry_approval(
        decision, direction="BUY", symbol=symbol, exchange="delta", contract=symbol,
        instrument_id=product_id, market_type="perpetual", analysis_symbol=symbol,
        analysis_exchange="delta", snapshot_version="snap-abc",
        analysis_timestamp=completed-1, confidence=80,
    )
    return {
        "status": status, "scope": "parts1-12-analysis-only", "analysis_only": True,
        "decision_authority": "none", "execution_eligible": False,
        "request_identity": identity, "execution_identity": identity,
        "parts_by_timeframe": native_evidence,
        "snapshot_version": "snap-abc", "snapshot_fetched_at": fetched,
        "analysis_completed_at": completed, "freshness_status": "FRESH",
        "central_strategy_decision": decision, "central_strategy_evidence": evidence,
        "central_strategy_approval": approval, "part7_gate": gate,
        "execution_plan": plan if plan is not None else {
            "timeframe": "5m", "decision_timestamp": NOW-3, "reference_price_timestamp": NOW-3,
            "reference_price": 100.0, "max_slippage_pct": 0.1, "max_chase_pct": 0.25,
            "direction": "BUY", "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 105.0,
            "quantity": 1.0, "size_unit": "base_asset_quantity",
            "sizing_provenance": "fixture-risk-engine-v1", "risk_notional": 2.0,
            "policy_id": POLICIES.get(asset),
        },
    }


def candidate(symbol="BTCUSDT", product_id="101", **kwargs):
    return candidate_from_analysis(analysis(symbol, product_id, **kwargs), policy_registry=POLICIES, now=NOW)


class Adapter(DeterministicPaperAdapter):
    def __init__(self, status="FILLED", *, fail=False, authoritative=True, quantity=1.0):
        self.status, self.fail, self.authoritative, self.quantity = status, fail, authoritative, quantity
        self.submissions, self.closes = [], []

    def submit(self, cand, idempotency_key):
        self.submissions.append((cand.candidate_id, idempotency_key))
        if self.fail:
            raise TimeoutError("simulated timeout after submit attempt")
        return {"status": self.status, "order_id": "order-" + idempotency_key[:8],
                "authoritative": self.authoritative, "filled_quantity": self.quantity}

    def close(self, candidate_id, identity, quantity):
        self.closes.append((candidate_id, identity, quantity))
        return {"status": "CLOSE_PENDING", "authoritative": False}


def coordinator(tmp_path, **kwargs):
    return PortfolioCoordinator(journal_path=str(tmp_path / "ledger.json"), enabled=True,
                                max_positions=kwargs.pop("max_positions", 3),
                                max_total_notional=kwargs.pop("max_total_notional", 1000),
                                max_total_risk=kwargs.pop("max_total_risk", 100), **kwargs)


def test_candidate_contract_requires_all_explicit_fields_and_asset_policy():
    good = candidate()
    assert good.direction == "BUY" and good.identity.instrument_id == "101"
    assert good.notional == 100.0 and good.risk_notional == 2.0
    contract_plan = {**analysis()["execution_plan"], "size_unit": "contracts", "contract_multiplier": 10.0, "risk_notional": 20.0}
    contract_candidate = candidate_from_analysis(analysis(plan=contract_plan), policy_registry=POLICIES, now=NOW)
    assert contract_candidate.notional == 1000.0 and contract_candidate.risk_notional == 20.0
    zone_veto = analysis()
    # Frames are vote-averaged by horizon group, so a real zone veto must hold across frames.
    for _tf in REQUIRED_TIMEFRAMES:
        zone_veto["parts_by_timeframe"][_tf]["part2_zone"].update({
            "signal": -1, "thought": "resistance zone", "symbol": "BTCUSDT",
            "timeframe": _tf, "native_timeframe": _tf,
        })
    wrong_frame = analysis()
    wrong_frame["parts_by_timeframe"]["4h"]["part6_trend"]["symbol"] = "ETHUSDT"
    for bad_result, policies in [
        (analysis(status="PARTIAL"), POLICIES),
        (zone_veto, POLICIES),
        ({**analysis(), "parts_by_timeframe": {}}, POLICIES),
        (wrong_frame, POLICIES),
        ({**analysis(), "snapshot_fetched_at": NOW-500}, POLICIES),
        ({**analysis(), "execution_plan": None}, POLICIES),
        ({**analysis(), "request_identity": {"venue": "delta", "symbol": "BTCUSDT"}}, POLICIES),
        (analysis(plan={**analysis()["execution_plan"], "direction": "NO TRADE"}), POLICIES),
        (analysis(plan={**analysis()["execution_plan"], "quantity": None}), POLICIES),
        (analysis(plan={**analysis()["execution_plan"], "stop_loss": 101}), POLICIES),
        (analysis(plan={k: v for k, v in analysis()["execution_plan"].items() if k != "reference_price"}), POLICIES),
        (analysis(plan={**analysis()["execution_plan"], "entry_price": 101.0}), POLICIES),
        (analysis(plan={**analysis()["execution_plan"], "size_unit": "contracts"}), POLICIES),
        (analysis(), {"BTC": "different-policy"}),
    ]:
        with pytest.raises(CandidateRejected):
            candidate_from_analysis(bad_result, policy_registry=policies, now=NOW)


def test_concurrent_competing_signals_reserve_atomically_and_are_idempotent(tmp_path):
    c = coordinator(tmp_path, max_positions=1)
    a, b = candidate("BTCUSDT", "101"), candidate("ETHUSDT", "202")
    adapter = Adapter()
    barrier = threading.Barrier(2)
    def submit(item):
        barrier.wait()
        return c.submit(item, adapter)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (a, b)))
    assert sorted(r["status"] for r in results) == ["BLOCKED", "FILLED"]
    winner = a if results[0]["status"] == "FILLED" else b
    assert c.submit(winner, adapter)["duplicate"] is True
    assert len(adapter.submissions) == 1


def test_correlation_cap_and_full_identity_isolation(tmp_path):
    c = coordinator(tmp_path, max_positions=4, correlation_groups={"BTC": "majors", "ETH": "majors"}, max_per_correlation_group=1)
    one = candidate("BTCUSDT", "101")
    # Same ticker on another product is not a distinct concurrent position.
    same_symbol_different_id = candidate("BTCUSDT", "102")
    other = candidate("SOLUSDT", "303")
    adapter = Adapter()
    assert c.submit(one, adapter)["status"] == "FILLED"
    assert c.submit(same_symbol_different_id, adapter)["status"] == "BLOCKED"
    assert c.submit(candidate("ETHUSDT", "202"), adapter)["status"] == "BLOCKED"
    assert c.submit(other, adapter)["status"] == "FILLED"


def test_risk_budget_is_reserved_atomically_in_addition_to_notional(tmp_path):
    c = coordinator(tmp_path, max_positions=5, max_total_risk=2.0)
    adapter = Adapter()
    assert c.submit(candidate("BTCUSDT", "101"), adapter)["status"] == "FILLED"
    assert c.submit(candidate("ETHUSDT", "202"), adapter)["status"] == "BLOCKED"


def test_timeout_unknown_and_partial_fill_keep_reservation_across_restart(tmp_path):
    path = str(tmp_path / "ledger.json")
    a, b = candidate("BTCUSDT", "101"), candidate("ETHUSDT", "202")
    c = PortfolioCoordinator(journal_path=path, enabled=True, max_positions=1, max_total_notional=1000, max_total_risk=100)
    assert c.submit(a, Adapter(fail=True))["status"] == "SUBMISSION_UNKNOWN"
    reopened = PortfolioCoordinator(journal_path=path, enabled=True, max_positions=1, max_total_notional=1000, max_total_risk=100)
    assert reopened.submit(a, Adapter())["duplicate"] is True
    assert reopened.submit(b, Adapter())["status"] == "BLOCKED"
    # A complete snapshot that does not resolve the submission is not a rejection.
    assert reopened.reconcile({"complete": True, "positions": [], "orders": {}})
    assert reopened.snapshot()["orders"][a.candidate_id]["status"] == "SUBMISSION_UNKNOWN"
    assert reopened.submit(b, Adapter())["status"] == "BLOCKED"

    p = candidate("SOLUSDT", "303")
    c2 = coordinator(tmp_path / "partial", max_positions=1)
    assert c2.submit(p, Adapter(status="PARTIAL", quantity=0.4))["status"] == "PARTIAL"
    assert c2.reconcile({"complete": True,
                          "positions": [{**p.identity.to_dict(), "notional": p.entry_price*0.4, "risk_notional": 0.8}],
                          "orders": {p.candidate_id: {"status": "PARTIAL", "authoritative": True, "filled_quantity": 0.4}}})
    assert c2.snapshot()["orders"][p.candidate_id]["status"] == "PARTIAL"


def test_account_snapshot_mismatch_preserves_owned_position_state(tmp_path):
    c = coordinator(tmp_path)
    p = candidate()
    c.submit(p, Adapter())
    before = c.snapshot()
    assert not c.reconcile({"complete": True, "positions": [],
                            "orders": {p.candidate_id: {"status": "FILLED", "authoritative": True}}})
    assert c.snapshot()["orders"][p.candidate_id]["status"] == before["orders"][p.candidate_id]["status"] == "FILLED"
    assert not c.reconcile({"complete": False, "positions": [], "orders": {}})
    assert not c.reconcile({"complete": True, "positions": [{"symbol": "BTCUSDT", "notional": 10}], "orders": {}})


def test_rejection_release_only_after_authoritative_reconciliation(tmp_path):
    c = coordinator(tmp_path, max_positions=1)
    p, other = candidate(), candidate("ETHUSDT", "202")
    assert c.submit(p, Adapter(status="REJECTED", authoritative=True))["status"] == "SUBMISSION_UNKNOWN"
    assert c.submit(other, Adapter())["status"] == "BLOCKED"
    assert c.reconcile({"complete": True, "positions": [],
                        "orders": {p.candidate_id: {"status": "REJECTED", "authoritative": True}}})
    assert c.snapshot()["orders"][p.candidate_id]["status"] == "REJECTED"
    assert c.submit(other, Adapter())["status"] == "FILLED"


def test_owned_position_close_is_per_identity_and_reconciled_not_assumed(tmp_path):
    c = coordinator(tmp_path, max_positions=2)
    first, second = candidate("BTCUSDT", "101"), candidate("SOLUSDT", "303")
    adapter = Adapter()
    c.submit(first, adapter); c.submit(second, adapter)
    assert c.close_position(first.candidate_id, second.identity.to_dict(), adapter)["status"] == "BLOCKED"
    assert c.close_position(first.candidate_id, first.identity.to_dict(), adapter)["status"] == "CLOSE_PENDING"
    assert c.snapshot()["orders"][second.candidate_id]["status"] == "FILLED"
    assert not c.reconcile({"complete": True, "positions": [{**first.identity.to_dict(), "notional": 100, "risk_notional": 2}],
                            "orders": {first.candidate_id: {"status": "CLOSED", "authoritative": True}}})
    assert c.snapshot()["orders"][first.candidate_id]["status"] == "CLOSE_PENDING"
    assert c.reconcile({"complete": True, "positions": [{**second.identity.to_dict(), "notional": 100, "risk_notional": 2}],
                        "orders": {first.candidate_id: {"status": "CLOSED", "authoritative": True},
                                   second.candidate_id: {"status": "FILLED", "authoritative": True, "filled_quantity": 1}}})
    assert c.snapshot()["orders"][first.candidate_id]["status"] == "CLOSED"
    assert c.snapshot()["orders"][second.candidate_id]["status"] == "FILLED"


def test_live_engine_paper_handoff_rejects_native_analysis_but_accepts_explicit_plan(tmp_path):
    from unittest.mock import patch
    import jarvis_FIXED as jarvis

    engine = object.__new__(jarvis.LiveTradingEngine)
    engine.paper_open_trades = []
    engine._dashboard_events = []
    assert engine._open_paper_trade(
        'CALL', 100.0, 80, 'SCALP', 101.0, 102.0, 99.0, symbol='BTCUSDT'
    ) is None
    assert engine.paper_open_trades == []
    engine.multicoin_paper_coordinator = coordinator(tmp_path)
    engine.multicoin_paper_adapter = DeterministicPaperAdapter()
    engine._multicoin_paper_policies = POLICIES
    engine.multicoin_paper_status = {"enabled": True, "submitted": 0, "blocked": 0}
    native_result = analysis(plan=None)
    native_result["execution_plan"] = None
    status = {"results": {"delta|perpetual|101|BTCUSDT": native_result}}
    with patch("jarvis_FIXED.time.time", return_value=NOW):
        engine._consume_multicoin_paper_results(status)
    assert engine.multicoin_paper_status["submitted"] == 0
    assert engine.multicoin_paper_status["last_status"] == "BLOCKED_INVALID_CANDIDATE"

    explicit = analysis()
    status["results"]["delta|perpetual|101|BTCUSDT"] = explicit
    with patch("jarvis_FIXED.time.time", return_value=NOW):
        engine._consume_multicoin_paper_results(status)
    assert engine.multicoin_paper_status["submitted"] == 1
    assert engine.multicoin_paper_status["last_status"] == "FILLED"
    assert len(engine.multicoin_paper_coordinator.snapshot()["orders"]) == 1


def test_bad_journal_fails_closed_and_disabled_coordinator_never_submits(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text('{broken')
    c = PortfolioCoordinator(journal_path=str(path), enabled=True, max_total_notional=1000, max_total_risk=100)
    assert c.healthy is False and c.submit(candidate(), Adapter())["status"] == "BLOCKED"
    disabled = PortfolioCoordinator(journal_path=str(tmp_path / "disabled.json"), enabled=False, max_total_notional=1000, max_total_risk=100)
    assert disabled.submit(candidate(), Adapter())["status"] == "BLOCKED"
