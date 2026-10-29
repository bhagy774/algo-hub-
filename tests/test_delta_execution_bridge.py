import json

import pytest

from jarvis_delta_execution import DeltaExecutionAdapter, build_delta_candidate
from jarvis_multicoin_execution import CandidateRejected, PortfolioCoordinator
from jarvis_strategy_approval import (
    REQUIRED_TIMEFRAMES, build_execution_plan, evaluate_mtf_central_strategy,
    make_entry_approval,
)


NOW = 1_800_000_000.0
ASSET_POLICY = {
    "ETH": {
        "policy_id": "eth-delta-risk-v1",
        "mapping_policy_id": "binance_spot_usdt_to_delta_linear_perpetual_usdt",
        "binance_symbol": "ETHUSDT",
        "delta_symbol": "ETHUSDT",
        "delta_instrument_id": "22",
        "delta_market_type": "perpetual_futures",
        "delta_quote_asset": "USDT",
        "delta_settling_asset": "USDT",
        "risk_currency": "USDT",
        "min_confidence": 70,
        "max_quote_age_seconds": 5,
        "max_spread_pct": 0.2,
        "max_chase_pct": 1.0,
        "max_slippage_pct": 0.2,
        "max_trade_risk_currency": 5.0,
    }
}


def central_evidence():
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


def valid_part7_gate(symbol):
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


def native_parts_by_timeframe(symbol="ETHUSDT"):
    rows = {}
    for timeframe in REQUIRED_TIMEFRAMES:
        frame = {}
        for index, (name, item) in enumerate(central_evidence().items(), 1):
            scoped = {**item, "symbol": symbol, "timeframe": timeframe}
            if name == "part7_volatility":
                scoped.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                               "data_status": "valid", "entry_blocked": False, "risk_veto": False})
            if name == "part2_zone":
                scoped["native_timeframe"] = timeframe
            frame[f"part{index}"] = scoped
        rows[timeframe] = frame
    return rows


def analysis(**changes):
    snapshot_version = changes.get("snapshot_version", "snap-eth-1m-abc")
    evidence = central_evidence()
    by_timeframe = native_parts_by_timeframe("ETHUSDT")
    gate = valid_part7_gate("ETHUSDT")
    normalized = {
        timeframe: {canonical: dict(row[f"part{index}"])
                    for index, canonical in enumerate(evidence, 1)}
        for timeframe, row in by_timeframe.items()
    }
    decision = evaluate_mtf_central_strategy(normalized, gate, confidence=90, expected_symbol="ETHUSDT")
    approval = make_entry_approval(
        decision, direction="BUY", symbol="ETHUSDT", exchange="delta", contract="ETHUSDT",
        instrument_id="22", market_type="perpetual_futures", analysis_symbol="ETHUSDT",
        analysis_exchange="binance", snapshot_version=snapshot_version,
        analysis_timestamp=NOW - 3, confidence=90,
    )
    result = {
        "status": "COMPLETE",
        "scope": "parts1-12-analysis-only",
        "analysis_only": True,
        "decision_authority": "none",
        "execution_eligible": False,
        "central_strategy_decision": decision,
        "central_strategy_evidence": evidence,
        "central_strategy_approval": approval,
        "freshness_status": "FRESH",
        "coverage": [f"Part{i}" for i in range(1, 13)],
        "snapshot_version": "snap-eth-1m-abc",
        "parts_by_timeframe": by_timeframe,
        "request_identity": {"venue": "binance", "market_type": "spot", "instrument_id": "ETHUSDT", "symbol": "ETHUSDT"},
        "execution_identity": {"venue": "delta", "market_type": "perpetual_futures", "instrument_id": "22", "symbol": "ETHUSDT"},
        "mapping_policy_id": ASSET_POLICY["ETH"]["mapping_policy_id"],
        "snapshot_fetched_at": NOW - 5,
        "analysis_completed_at": NOW - 2,
        "analysis_reference": {"source": "binance", "symbol": "ETHUSDT", "timeframe": "1m", "timestamp": NOW - 60, "price": 100.0},
        "once_per_symbol_parts": {"part11": {"signal": 1}, "part12": {"confidence": 90}},
        "part7_gate": gate,
        "deterministic_decision": {
            "origin": "jarvis_FIXED_central_strategy", "direction": "BUY", "confidence": 90,
            "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 105.0,
        },
    }
    result.update(changes)
    return result


class FakeDelta:
    def __init__(self, clock=lambda: NOW, *, protected_response=None):
        self.clock = clock
        self.protected_response = protected_response
        self.position = None
        self.close_order_id = None
        self.entry_calls = []
        self.close_calls = []
        self.product = {
            "id": 22, "symbol": "ETHUSDT", "state": "active",
            "product_type": "perpetual_futures", "base_asset": "ETH", "quote_asset": "USDT",
            "settling_asset": "USDT", "contract_value_usdt": 1.0, "max_leverage": 10, "contract_size": 1,
        }

    def get_product_metadata(self, symbol):
        return dict(self.product) if symbol == "ETHUSDT" else None

    def get_delta_executable_quote(self, *, symbol, product_id):
        if symbol != "ETHUSDT" or product_id != "22":
            return None
        return {"source": "delta", "symbol": symbol, "product_id": product_id,
                "bid": 100.0, "ask": 100.1, "observed_at": self.clock()}

    def get_available_balance(self, currency):
        return 500.0 if currency == "USDT" else 0.0

    def place_protected_order(self, **kwargs):
        self.entry_calls.append(kwargs)
        if self.protected_response is not None:
            response = dict(self.protected_response)
            if response.get("status") == "FILLED":
                response.setdefault("order_id", "delta-entry-1")
                response.setdefault("filled_quantity", kwargs["size"])
                response.setdefault("average_fill_price", 100.1)
                response.setdefault("protection_state", "ACTIVE")
                response.setdefault("protective_exits", {"stop_loss_order_id": "sl-1", "take_profit_order_id": "tp-1"})
            if response.get("status") == "FILLED":
                self.position = {"quantity": response["filled_quantity"], "entry": response["average_fill_price"]}
            return response
        self.position = {"quantity": kwargs["size"], "entry": 100.1}
        return {"status": "FILLED", "authoritative": True, "order_id": "delta-entry-1",
                "filled_quantity": kwargs["size"], "average_fill_price": 100.1,
                "protection_state": "ACTIVE",
                "protective_exits": {"stop_loss_order_id": "sl-1", "take_profit_order_id": "tp-1"}}

    def get_complete_account_snapshot(self, *, owned_orders):
        positions, orders = [], {}
        if self.position:
            for candidate_id, record in owned_orders.items():
                identity = record.get("identity")
                if identity:
                    positions.append({**identity, "notional": self.position["quantity"] * self.position["entry"],
                                      "risk_notional": 0.1, "risk_currency": "USDT"})
                    if self.close_order_id:
                        orders[candidate_id] = {"status": "CLOSED", "authoritative": True}
                    else:
                        orders[candidate_id] = {"status": "FILLED", "authoritative": True,
                            "filled_quantity": self.position["quantity"], "protection_state": "ACTIVE",
                            "protective_exits": {"stop_loss_order_id": "sl-1", "take_profit_order_id": "tp-1"}}
        else:
            for candidate_id in owned_orders:
                orders[candidate_id] = {"status": "CLOSED", "authoritative": True}
        return {"complete": True, "as_of": self.clock(), "positions": positions, "orders": orders,
                "external_orders": [], "risk_currency": "USDT"}

    def place_reduce_only_order(self, **kwargs):
        self.close_calls.append(kwargs)
        if kwargs.get("side") != "sell" or int(kwargs.get("size", 0)) <= 0:
            return {"success": False}
        self.close_order_id = kwargs.get("client_order_id")
        self.position = None
        return {"success": True, "order_id": "delta-close-1", "client_order_id": self.close_order_id}


def build(fake=None, result=None):
    fake = fake or FakeDelta()
    return build_delta_candidate(result or analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW), fake


def test_valid_eth_bridge_sizes_from_delta_metadata_quote_and_risk_engine():
    candidate, fake = build()
    assert candidate.identity.venue == "delta"
    assert candidate.identity.symbol == "ETHUSDT"
    assert candidate.analysis_identity.venue == "binance"
    assert candidate.direction == "BUY"
    assert candidate.entry_price == pytest.approx(100.1)  # Delta ask, never Binance candle price
    assert candidate.stop_loss == 98.0 and candidate.take_profit == 105.0
    assert candidate.quantity > 0 and candidate.size_unit == "contracts"
    assert candidate.sizing_provenance == "jarvis_risk.calculate_trade_size+jarvis_lot_limits.enforce_entry_lots"
    assert fake.entry_calls == []


def test_mapping_quote_product_freshness_and_missing_data_all_fail_closed():
    fake = FakeDelta()
    zone_veto = analysis()
    # Frames are vote-averaged by horizon group, so a real zone veto must hold across frames.
    for _tf in REQUIRED_TIMEFRAMES:
        zone_veto["parts_by_timeframe"][_tf]["part2"]["signal"] = -1
        zone_veto["parts_by_timeframe"][_tf]["part2"]["thought"] = "resistance zone"
    bad_results = [
        zone_veto,
        analysis(parts_by_timeframe=None),
        analysis(request_identity={"venue": "binance", "market_type": "spot", "instrument_id": "SOLUSDT", "symbol": "SOLUSDT"}),
        analysis(request_identity={"venue": "binance", "market_type": "spot", "instrument_id": "unrelated-id", "symbol": "ETHUSDT"}),
        analysis(execution_identity={"venue": "delta", "market_type": "perpetual_futures", "instrument_id": "33", "symbol": "SOLUSDT"}),
        analysis(analysis_reference=None),
        analysis(analysis_reference={"source": "binance", "symbol": "ETHUSDT", "timeframe": "1m", "timestamp": NOW - 900, "price": 100}),
        analysis(deterministic_decision={"origin": "jarvis_deterministic_parts11_12", "direction": "NO_TRADE", "confidence": 90}),
        analysis(part7_gate={"entry_blocked": True, "risk_veto": True}),
        analysis(part7_gate={"entry_blocked": False}),
        analysis(part7_gate={"symbol": "BTCUSDT", "entry_blocked": False, "risk_veto": False}),
        analysis(freshness_status=None),
        analysis(coverage=["Part1"]),
        analysis(parts_by_timeframe={"1m": {"part1": {}}}),
    ]
    for result in bad_results:
        with pytest.raises(CandidateRejected):
            build_delta_candidate(result, delta=fake, policy_registry=ASSET_POLICY, now=NOW)
    fake.product["base_asset"] = "SOL"
    with pytest.raises(CandidateRejected):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)
    fake.product["base_asset"] = "ETH"
    fake.product["product_type"] = "perpetual"
    with pytest.raises(CandidateRejected, match="metadata"):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)
    fake.product["product_type"] = "perpetual_futures"
    fake.get_delta_executable_quote = lambda **kwargs: {"source": "binance", "symbol": "ETHUSDT", "product_id": "22", "bid": 100, "ask": 100.1, "observed_at": NOW}
    with pytest.raises(CandidateRejected):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)


def test_non_atomic_broker_and_non_complete_account_capability_are_rejected():
    fake = FakeDelta()
    fake.place_protected_order = None
    with pytest.raises(CandidateRejected, match="staged entry/bracket"):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)

    fake = FakeDelta()
    fake.get_complete_account_snapshot = None
    with pytest.raises(CandidateRejected, match="reconciliation"):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)

    fake = FakeDelta()
    fake.get_available_balance = None
    with pytest.raises(CandidateRejected, match="exact-currency"):
        build_delta_candidate(analysis(), delta=fake, policy_registry=ASSET_POLICY, now=NOW)


def test_analysis_plan_risk_execution_fill_monitor_close_and_restart(tmp_path):
    fake = FakeDelta()
    adapter = DeltaExecutionAdapter(fake, policy_registry=ASSET_POLICY, authorization_check=lambda: True, clock=lambda: NOW)
    ledger_path = tmp_path / "delta-ledger.json"
    coordinator = PortfolioCoordinator(
        journal_path=str(ledger_path), enabled=True, require_reconciliation=True,
        require_protective_reconciliation=True, max_positions=2,
        max_total_notional=10_000, max_total_risk=100,
    )
    assert coordinator.snapshot()["ready"] is False
    assert adapter.reconcile(coordinator) is True
    candidate = adapter.prepare_candidate(analysis(), now=NOW)
    first = coordinator.submit(candidate, adapter)
    assert first["status"] == "FILLED"
    assert len(fake.entry_calls) == 1
    assert fake.entry_calls[0]["client_order_id"].startswith("jme")
    assert len(fake.entry_calls[0]["client_order_id"]) == 32
    assert fake.entry_calls[0]["leverage"] == candidate.leverage
    assert fake.entry_calls[0]["stop_loss"] == 98 and fake.entry_calls[0]["take_profit"] == 105
    assert coordinator.submit(candidate, adapter)["duplicate"] is True
    assert len(fake.entry_calls) == 1
    assert coordinator.snapshot()["orders"][candidate.candidate_id]["protection_state"] == "ACTIVE"

    # Target crossed at executable Delta bid: issue only an identity-owned,
    # reduce-only close; state remains pending until authoritative account proof.
    fake.get_delta_executable_quote = lambda **kwargs: {"source": "delta", "symbol": "ETHUSDT", "product_id": "22",
                                                        "bid": 105.1, "ask": 105.2, "observed_at": NOW}
    monitored = adapter.monitor_once(coordinator)
    assert monitored["status"] == "MONITORED" and monitored["close_requests"] == 1
    assert fake.close_calls and fake.close_calls[0]["side"] == "sell"
    assert fake.close_calls[0]["product_id"] == 22
    assert len(fake.close_calls[0]["client_order_id"]) == 32
    assert coordinator.snapshot()["orders"][candidate.candidate_id]["status"] == "CLOSE_PENDING"

    # Restart recovery cannot declare close on the close acknowledgement; the
    # full authoritative snapshot transitions the owned row to CLOSED.
    reopened = PortfolioCoordinator(
        journal_path=str(ledger_path), enabled=True, require_reconciliation=True,
        require_protective_reconciliation=True, max_positions=2,
        max_total_notional=10_000, max_total_risk=100,
    )
    assert reopened.snapshot()["ready"] is False
    assert adapter.reconcile(reopened) is True
    assert reopened.snapshot()["orders"][candidate.candidate_id]["status"] == "CLOSED"
    assert reopened.snapshot()["ready"] is True


def test_unknown_fill_or_missing_protection_keeps_reservation_and_blocks_restart(tmp_path):
    fake = FakeDelta(protected_response={"status": "FILLED", "authoritative": False,
                                         "order_id": "lost-authority", "filled_quantity": 1,
                                         "average_fill_price": 100.1})
    adapter = DeltaExecutionAdapter(fake, policy_registry=ASSET_POLICY, authorization_check=lambda: True, clock=lambda: NOW)
    candidate = adapter.prepare_candidate(analysis(), now=NOW)
    path = tmp_path / "unknown.json"
    coordinator = PortfolioCoordinator(journal_path=str(path), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=1, max_total_notional=10_000, max_total_risk=100)
    assert adapter.reconcile(coordinator)
    assert coordinator.submit(candidate, adapter)["status"] == "SUBMISSION_UNKNOWN"
    reopened = PortfolioCoordinator(journal_path=str(path), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=1, max_total_notional=10_000, max_total_risk=100)
    assert reopened.submit(candidate, adapter)["duplicate"] is True
    assert reopened.submit(adapter.prepare_candidate(analysis(snapshot_version="another-snapshot"), now=NOW), adapter)["status"] == "BLOCKED"
    assert reopened.snapshot()["orders"][candidate.candidate_id]["status"] == "SUBMISSION_UNKNOWN"


def test_simultaneous_distinct_snapshots_reserve_one_delta_identity_only(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    fake = FakeDelta()
    adapter = DeltaExecutionAdapter(fake, policy_registry=ASSET_POLICY, authorization_check=lambda: True, clock=lambda: NOW)
    coordinator = PortfolioCoordinator(journal_path=str(tmp_path / "concurrent.json"), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=2, max_total_notional=10_000, max_total_risk=100)
    assert adapter.reconcile(coordinator)
    first = adapter.prepare_candidate(analysis(snapshot_version="snap-a"), now=NOW)
    second = adapter.prepare_candidate(analysis(snapshot_version="snap-b"), now=NOW)
    assert first.candidate_id != second.candidate_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda candidate: coordinator.submit(candidate, adapter), (first, second)))

    assert sorted(item["status"] for item in outcomes) == ["BLOCKED", "FILLED"]
    assert len(fake.entry_calls) == 1
    open_rows = [row for row in coordinator.snapshot()["orders"].values() if row["status"] in {"FILLED", "PARTIAL"}]
    assert len(open_rows) == 1
    assert coordinator.snapshot()["ready"] is True


def test_live_authorization_accepts_configured_numeric_true_flags(monkeypatch):
    for name in ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                 "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("JARVIS_KILL_SWITCH", "0")
    assert DeltaExecutionAdapter._live_flags_authorized() is True


def test_protected_broker_authorization_binds_plan_and_broker_levels(monkeypatch):
    import importlib
    import sys
    import time
    import types

    shim = types.ModuleType("options_chain")
    shim.build_provider_chain = lambda *args, **kwargs: {}
    shim.combine_provider_chains = lambda *args, **kwargs: {}
    shim.payout_max_pain = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "options_chain", shim)
    wrapper = importlib.import_module("delta_api_wrapper")

    evidence = central_evidence()
    by_timeframe = native_parts_by_timeframe("ETHUSDT")
    gate = valid_part7_gate("ETHUSDT")
    normalized = {
        timeframe: {canonical: dict(row[f"part{index}"])
                    for index, canonical in enumerate(evidence, 1)}
        for timeframe, row in by_timeframe.items()
    }
    decision = evaluate_mtf_central_strategy(normalized, gate, confidence=90, expected_symbol="ETHUSDT")
    plan = build_execution_plan(
        direction="BUY", recommended_expiry="SWING", entry_price=100,
        stop_loss=99.2, take_profit=102, symbol="ETHUSDT",
        snapshot_version="snap-protected", confidence=90,
    )
    approval = make_entry_approval(
        decision, direction="BUY", symbol="ETHUSDT", exchange="delta",
        contract="ETHUSDT", instrument_id="22", market_type="perpetual_futures",
        analysis_symbol="ETHUSDT", analysis_exchange="delta",
        snapshot_version="snap-protected", analysis_timestamp=time.time(),
        confidence=90, execution_plan=plan,
    )
    auth = {
        "central_approval": approval, "part_results": evidence,
        "parts_by_timeframe": normalized, "part7_gate": gate,
        "strategy_plan": plan, "snapshot_version": "snap-protected",
        "analysis_symbol": "ETHUSDT", "confidence": 90,
        "broker_plan": {"symbol": "ETHUSDT", "instrument_id": "22", "direction": "BUY",
            "quantity": 1, "leverage": 2, "entry_price": 100,
            "stop_loss": 99.2, "take_profit": 102, "risk_budget_usdt": 5,
            "trade_mode": "SWING"},
    }
    args = {"symbol": "ETHUSDT", "side": "buy", "product_id": 22,
            "size": 1, "stop_loss": 99.2, "take_profit": 102, "leverage": 2}
    monkeypatch.setenv("JARVIS_MAX_RISK_USDT", "10.0")
    assert wrapper._validate_jarvis_broker_entry_authorization(auth, **args)[0]
    mutated = {**auth, "broker_plan": {**auth["broker_plan"], "stop_loss": 98}}
    rejected, reason = wrapper._validate_jarvis_broker_entry_authorization(
        mutated, **(args | {"stop_loss": 98}),
    )
    assert not rejected and "plan" in reason.lower()
    moved_entry = {**auth, "broker_plan": {**auth["broker_plan"], "entry_price": 100.01}}
    rejected_entry, entry_reason = wrapper._validate_jarvis_broker_entry_authorization(
        moved_entry, **args,
    )
    assert not rejected_entry and "plan" in entry_reason.lower()
    assert wrapper._same_decimal_price("100.00", 100.0)
    assert not wrapper._same_decimal_price("100.01", 100.0)

    for name in ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                 "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED"):
        monkeypatch.setenv(name, "true")
    monkeypatch.setenv("JARVIS_KILL_SWITCH", "0")
    delta = object.__new__(wrapper.DeltaExchangeData)
    delta._USE_MAINNET = True
    delta.get_available_products_snapshot = lambda: {"complete": True, "products": [{
        "id": 22, "symbol": "ETHUSDT", "state": "active", "product_type": "perpetual_futures",
        "base_asset": "ETH", "quote_asset": "USDT", "settling_asset": "USDT",
        "product_specs": {}, "tick_size": 0.01,
    }]}
    delta.get_delta_executable_quote = lambda **kwargs: {
        "source": "delta", "symbol": "ETHUSDT", "product_id": "22", "bid": 100.0,
        "ask": 100.01, "observed_at": time.time(),
    }
    delta._request = lambda *args, **kwargs: pytest.fail("quote mismatch must block before venue mutation")
    mismatch = delta.place_protected_order(
        product_id=22, symbol="ETHUSDT", side="buy", size=1, order_type="market",
        stop_loss=99.2, take_profit=102, leverage=2, client_order_id="quote-mismatch-test",
        entry_authorization=auth,
    )
    assert mismatch["status"] == "REJECTED"
    assert "exact Jarvis plan entry" in mismatch["reason"]


def test_protective_child_reconciliation_binds_prices_quantity_and_reduce_only(monkeypatch):
    import importlib
    import sys
    import types

    shim = types.ModuleType("options_chain")
    shim.build_provider_chain = lambda *args, **kwargs: {}
    shim.combine_provider_chains = lambda *args, **kwargs: {}
    shim.payout_max_pain = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "options_chain", shim)
    wrapper = importlib.import_module("delta_api_wrapper")
    row = {"id": "stop-1", "product_id": 22, "state": "open", "reduce_only": True,
           "side": "sell", "stop_order_type": "stop_loss_order", "size": 2,
           "stop_price": "99", "limit_price": "98.9"}
    expected = {"product_id": 22, "side": "sell", "kind": "stop_loss_order",
                "quantity": 2, "trigger": 99, "limit": 98.9}
    assert wrapper._valid_protective_child(row, **expected)
    mutations = [
        ("stop_price", "98.9"), ("limit_price", "99"), ("size", 1),
        ("reduce_only", False), ("side", "buy"), ("state", "filled"),
        ("product_id", 33),
    ]
    for field, value in mutations:
        assert not wrapper._valid_protective_child(row | {field: value}, **expected)


def test_raw_delta_entries_are_rejected_but_reduce_only_close_route_remains(monkeypatch):
    import importlib
    import sys
    import types

    shim = types.ModuleType("options_chain")
    shim.build_provider_chain = lambda *args, **kwargs: {}
    shim.combine_provider_chains = lambda *args, **kwargs: {}
    shim.payout_max_pain = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "options_chain", shim)
    wrapper = importlib.import_module("delta_api_wrapper")
    delta = object.__new__(wrapper.DeltaExchangeData)
    calls = []
    delta.get_product_id = lambda symbol: 22
    delta._request = lambda *args, **kwargs: (calls.append((args, kwargs)) or {
        "success": True, "data": {"result": {"id": "close-1"}},
    })
    monkeypatch.setenv("DELTA_ORDER_EXECUTION_ENABLED", "true")
    denied = delta.place_order("ETHUSDT", "buy", 1)
    assert not denied["success"] and not calls
    closed = delta.place_order("ETHUSDT", "sell", 1, reduce_only=True)
    assert closed["success"] and len(calls) == 1
    assert calls[0][0][2]["reduce_only"] is True
