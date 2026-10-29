"""Offline request/response-layer tests for concrete DeltaExchangeData capabilities.

These tests exercise the real wrapper parsers and lifecycle orchestration while
replacing only the authenticated/public transport boundary. They never open a
socket or use exchange credentials.
"""
import copy
import os
import time

import pytest

from delta_api_wrapper import DeltaExchangeData
from jarvis_delta_execution import DeltaExecutionAdapter
from jarvis_multicoin_execution import PortfolioCoordinator
from jarvis_strategy_approval import (
    REQUIRED_TIMEFRAMES, build_execution_plan, evaluate_mtf_central_strategy,
    make_entry_approval,
)


NOW = time.time()
PRODUCT = {
    "id": 22, "symbol": "ETHUSDT", "state": "active",
    "product_type": "perpetual_futures", "underlying_asset": {"symbol": "ETH"},
    "quoting_asset": {"symbol": "USDT"}, "settling_asset": {"symbol": "USDT"},
    "contract_value": "1", "contract_unit_currency": "USDT", "notional_type": "vanilla",
    "tick_size": "0.1", "max_leverage": 10, "trading_status": "operational",
}


def response(result, **fields):
    return {"success": True, "data": {"result": result, **fields}}


def jarvis_broker_authorization(*, entry=100.1, stop=98.0, target=105.0,
                               symbol="ETHUSDT", product_id="22", snapshot="transport-snapshot",
                               confidence=90, now=None):
    """Issue a genuine scoped test approval through Jarvis' deterministic policy helpers."""
    issued = time.time() if now is None else float(now)
    evidence = {
        name: {"signal": 1, "thought": "valid bullish test evidence"}
        for name in (
            "part1_breakout", "part2_zone", "part3_psychology", "part4_volume",
            "part5_ml", "part6_trend", "part7_volatility", "part8_structure",
            "part9_orderflow", "part10_candlestats",
        )
    }
    gate = {
        "symbol": symbol, "entry_blocked": False, "risk_veto": False,
        "status": "ok", "data_status": "valid", "timeframe": "aggregate",
        "blocked_timeframes": [], "veto_timeframes": [],
        "timeframe_results": {
            tf: {"symbol": symbol, "timeframe": tf, "status": "neutral",
                 "data_status": "valid", "entry_blocked": False, "risk_veto": False}
            for tf in REQUIRED_TIMEFRAMES
        },
    }
    timeframe_parts = {}
    for timeframe in REQUIRED_TIMEFRAMES:
        bullish = timeframe in {"1m", "3m", "5m", "15m"}
        row = {}
        for name, item in evidence.items():
            scoped = {**item, "symbol": symbol, "timeframe": timeframe}
            if name == "part7_volatility":
                scoped.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                              "data_status": "valid", "entry_blocked": False, "risk_veto": False})
            elif name == "part2_zone":
                scoped["native_timeframe"] = timeframe
                if not bullish:
                    scoped.update({"signal": 0, "thought": "no native zone"})
            elif not bullish:
                scoped.update({"signal": 0, "thought": "neutral native-frame evidence"})
            row[name] = scoped
        timeframe_parts[timeframe] = row
    decision = evaluate_mtf_central_strategy(
        timeframe_parts, gate, confidence=confidence, expected_symbol=symbol,
    )
    plan = build_execution_plan(
        direction="BUY", recommended_expiry="SCALP", entry_price=entry,
        stop_loss=stop, take_profit=target, symbol=symbol,
        snapshot_version=snapshot, confidence=confidence,
    )
    approval = make_entry_approval(
        decision, direction="BUY", symbol=symbol, exchange="delta", contract=symbol,
        instrument_id=product_id, market_type="perpetual_futures",
        analysis_symbol=symbol, analysis_exchange="binance",
        snapshot_version=snapshot, analysis_timestamp=issued,
        confidence=confidence, execution_plan=plan,
    )
    approval["trade_mode"] = "SCALP"
    return {
        "central_approval": approval, "part_results": evidence,
        "timeframe_parts": timeframe_parts, "parts_by_timeframe": timeframe_parts,
        "part7_gate": gate,
        "strategy_plan": plan, "snapshot_version": snapshot,
        "analysis_symbol": symbol, "confidence": confidence,
        "broker_plan": {
            "symbol": symbol, "instrument_id": str(product_id), "direction": "BUY",
            "quantity": 2, "leverage": 2, "entry_price": entry,
            "stop_loss": stop, "take_profit": target, "risk_budget_usdt": 5.0,
            "trade_mode": "SCALP",
        },
    }


class TransportDelta(DeltaExchangeData):
    def __init__(self):
        # Bypass the inherited provider setup entirely; the public/private
        # _request method below is the only boundary mocked in these tests.
        self._USE_MAINNET = True
        self.calls = []
        self.products = [copy.deepcopy(PRODUCT)]
        self.wallet_rows = [{"asset_symbol": "USDT", "available_balance": "500"},
                            {"asset_symbol": "USD", "available_balance": "999"}]
        self.position = {"product_id": 22, "size": 0, "mark_price": "100.1"}
        self.other_positions = {}
        self.entry_order = {"id": 501, "product_id": 22, "size": 2, "unfilled_size": 0,
                            "average_fill_price": "100.1", "state": "closed",
                            "client_order_id": "jme123"}
        self.close_order = None
        self.children = []
        self.quote_timestamp = int(time.time() * 1_000_000)
        self.quote_bid, self.quote_ask = 100.0, 100.1
        self.bracket_ok = True
        self.suppress_bracket_children = False
        self.entry_response = None
        self.custom_entry_order = None
        self.leverage = 2
        self.leverage_confirm = True
        self.external_orders = []

    def _request(self, method, endpoint, payload=None, authorized=False):
        self.calls.append((method.upper(), endpoint, copy.deepcopy(payload), authorized))
        if endpoint == "/v2/products":
            size = (payload or {}).get("page_size", 100)
            rows = self.products[:size]
            return {"success": True, "data": {"result": rows, "meta": {"after": None}}}
        if endpoint == "/v2/wallet/balances":
            return response(self.wallet_rows)
        if endpoint.startswith("/v2/l2orderbook/"):
            symbol = endpoint.rsplit("/", 1)[-1]
            return response({"symbol": symbol, "buy": [{"price": str(self.quote_bid)}],
                             "sell": [{"price": str(self.quote_ask)}],
                             "last_updated_at": self.quote_timestamp})
        if endpoint.endswith("/orders/leverage"):
            if method.upper() == "POST":
                self.leverage = int((payload or {}).get("leverage", 1))
                return response({"product_id": 22, "leverage": str(self.leverage)})
            return response({"product_id": 22, "leverage": str(self.leverage if self.leverage_confirm else 1)})
        if endpoint.startswith("/v2/positions?product_id="):
            pid = endpoint.split("=", 1)[-1]
            position = self.position if pid == "22" else self.other_positions.get(pid, {"product_id": int(pid), "size": 0})
            return response(copy.deepcopy(position))
        if endpoint == "/v2/orders/bracket" and method.upper() == "POST":
            if self.bracket_ok and not self.suppress_bracket_children:
                closing_side = "sell" if self.position.get("size", 0) > 0 else "buy"
                stop = payload["stop_loss_order"]
                target = payload["take_profit_order"]
                quantity = abs(int(self.position.get("size", 0)))
                self.children = [
                    {"id": 701, "product_id": 22, "state": "open", "stop_order_type": "stop_loss_order",
                     "reduce_only": True, "side": closing_side, "size": quantity,
                     "stop_price": stop["stop_price"], "limit_price": stop["limit_price"]},
                    {"id": 702, "product_id": 22, "state": "open", "stop_order_type": "take_profit_order",
                     "reduce_only": True, "side": closing_side, "size": quantity,
                     "stop_price": target["stop_price"], "limit_price": target["limit_price"]},
                ]
            return {"success": self.bracket_ok, "data": {"result": {}} if self.bracket_ok else None}
        if endpoint == "/v2/orders" and method.upper() == "POST":
            if self.entry_response is not None and not (payload or {}).get("reduce_only"):
                return self.entry_response
            if (payload or {}).get("reduce_only"):
                self.close_order = {"id": 601, "product_id": 22, "size": int(payload["size"]),
                                    "unfilled_size": 0, "state": "closed", "reduce_only": True,
                                    "client_order_id": payload["client_order_id"]}
                self.position = {"product_id": 22, "size": 0, "mark_price": str(self.quote_bid)}
                self.children = []
                return response(copy.deepcopy(self.close_order))
            if self.custom_entry_order is not None:
                self.entry_order = copy.deepcopy(self.custom_entry_order)
                return response(copy.deepcopy(self.entry_order))
            qty = int(payload["size"])
            signed = qty if payload["side"] == "buy" else -qty
            self.position = {"product_id": 22, "size": signed, "mark_price": str(self.quote_ask)}
            self.entry_order = {"id": 501, "product_id": 22, "size": qty, "unfilled_size": 0,
                                "average_fill_price": str(self.quote_ask), "state": "closed",
                                "client_order_id": payload.get("client_order_id")}
            return response(copy.deepcopy(self.entry_order))
        if endpoint == "/v2/orders" and method.upper() == "GET":
            rows = self.children + self.external_orders
            return {"success": True, "data": {"result": rows, "meta": {"after": None}}}
        if endpoint == "/v2/orders/501" and method.upper() == "DELETE":
            self.entry_order["state"] = "cancelled"
            return response(copy.deepcopy(self.entry_order))
        if endpoint == "/v2/orders/501" and method.upper() == "GET":
            return response(copy.deepcopy(self.entry_order))
        if endpoint.startswith("/v2/orders/client_order_id/") and method.upper() == "GET":
            client_id = endpoint.rsplit("/", 1)[-1]
            order = self.close_order if self.close_order and self.close_order.get("client_order_id") == client_id else self.entry_order
            return response(copy.deepcopy(order)) if order else {"success": False}
        if endpoint.startswith("/v2/orders/") and method.upper() == "GET":
            raw_id = endpoint.rsplit("/", 1)[-1]
            if raw_id == "501":
                return response(copy.deepcopy(self.entry_order))
            if raw_id == "601" and self.close_order:
                return response(copy.deepcopy(self.close_order))
            for child in self.children:
                if str(child.get("id")) == raw_id:
                    return response(copy.deepcopy(child))
            return {"success": False}
        if endpoint == "/v2/positions/margined":
            rows = ([{"product_id": 22, "size": self.position.get("size", 0)}]
                    if self.position.get("size", 0) else [])
            rows.extend({"product_id": int(pid), "size": position.get("size", 0)}
                        for pid, position in self.other_positions.items() if position.get("size", 0))
            return response(rows)
        return {"success": False, "error": "NOT_FOUND"}


def set_live_flags(monkeypatch):
    for name in ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                 "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("JARVIS_KILL_SWITCH", "0")
    monkeypatch.setenv("JARVIS_MAX_RISK_USDT", "100.0")


def test_real_pagination_parser_requires_terminal_cursor_and_rejects_repeats():
    client = TransportDelta()
    rows, complete = client._get_all_pages("/v2/products", authorized=False, page_size=1)
    assert complete and rows == [PRODUCT]

    replies = iter([
        {"success": True, "data": {"result": [{"id": 1}], "meta": {"after": "cursor-a"}}},
        {"success": True, "data": {"result": [{"id": 2}], "meta": {"after": "cursor-a"}}},
    ])
    client._request = lambda *args, **kwargs: next(replies)
    rows, complete = client._get_all_pages("/v2/products", authorized=False, page_size=1)
    assert rows == [] and complete is False

    client._request = lambda *args, **kwargs: {"success": True, "data": {"result": [{"id": 1}]}}
    rows, complete = client._get_all_pages("/v2/products", authorized=False, page_size=1)
    assert rows == [] and complete is False


def test_execution_metadata_requires_complete_unique_exact_symbol():
    client = TransportDelta()
    assert client.get_product_metadata("ETHUSDT")["id"] == 22
    client.products.append({**copy.deepcopy(PRODUCT), "id": 23})
    assert client.get_product_metadata("ETHUSDT") is None
    client.products = [copy.deepcopy(PRODUCT)]
    client._request = lambda method, endpoint, payload=None, authorized=False: (
        {"success": True, "data": {"result": [{"id": i, "symbol": f"X{i}"} for i in range(100)]}}
        if endpoint == "/v2/products" else {"success": False}
    )
    assert client.get_product_metadata("X1") is None


def test_wallet_balance_is_exact_denomination_and_ambiguous_rows_fail_closed():
    client = TransportDelta()
    assert client.get_available_balance("USDT") == 500
    assert client.get_available_balance("USD") == 999
    assert client.get_available_balance("INR") == 0
    client.wallet_rows.append({"asset_symbol": "USDT", "available_balance": "1"})
    assert client.get_available_balance("USDT") == 0
    client.wallet_rows = [{"asset_symbol": "USDT", "available_balance": "NaN"}]
    assert client.get_available_balance("USDT") == 0


def test_l2_quote_parses_microseconds_and_rejects_stale_future_crossed_or_wrong_identity():
    client = TransportDelta()
    good = client.get_delta_executable_quote(symbol="ETHUSDT", product_id="22")
    assert good and good["source"] == "delta" and good["bid"] == 100 and good["ask"] == 100.1
    client.quote_timestamp = int((time.time() - 30) * 1_000_000)
    assert client.get_delta_executable_quote(symbol="ETHUSDT", product_id="22") is None
    client.quote_timestamp = int((time.time() + 30) * 1_000_000)
    assert client.get_delta_executable_quote(symbol="ETHUSDT", product_id="22") is None
    client.quote_timestamp = int(time.time() * 1_000_000)
    original_request = client._request
    def crossed(method, endpoint, payload=None, authorized=False):
        if endpoint.startswith("/v2/l2orderbook/"):
            return response({"symbol": "ETHUSDT", "buy": [{"price": "101"}],
                             "sell": [{"price": "100"}], "last_updated_at": int(time.time() * 1_000_000)})
        return original_request(method, endpoint, payload, authorized)
    client._request = crossed
    assert client.get_delta_executable_quote(symbol="ETHUSDT", product_id="22") is None
    client._request = original_request
    assert client.get_delta_executable_quote(symbol="SOLUSDT", product_id="22") is None


def test_official_object_position_schema_and_malformed_shapes():
    client = TransportDelta()
    client.position = {"product_id": 22, "size": 2, "mark_price": "100.1"}
    assert client._get_positions_for_product_id("22") == [{"product_id": 22, "size": 2, "mark_price": "100.1"}]
    client._request = lambda *args, **kwargs: response({"wrong": "shape"})
    assert client._get_positions_for_product_id("22") is None
    assert client._get_positions_for_product_id("bad") is None


def test_legacy_set_leverage_uses_exact_product_endpoint_and_readback(monkeypatch):
    monkeypatch.setenv("DELTA_ORDER_EXECUTION_ENABLED", "true")
    client = TransportDelta()
    assert client.set_leverage("ETHUSDT", 3) is True
    calls = [call for call in client.calls if call[1].endswith("/orders/leverage")]
    assert [(call[0], call[1], call[2]) for call in calls] == [
        ("POST", "/v2/products/22/orders/leverage", {"leverage": "3"}),
        ("GET", "/v2/products/22/orders/leverage", None),
    ]
    assert not any(call[1] == "/v2/orders/leverage" for call in client.calls)


def test_protected_entry_without_jarvis_authorization_is_denied_before_venue_calls(monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    result = client.place_protected_order(
        product_id=22, symbol="ETHUSDT", side="buy", size=2,
        order_type="market", stop_loss=98, take_profit=105,
        leverage=2, client_order_id="missing-approval",
    )
    assert result["status"] == "REJECTED"
    assert "authorization is required" in result["reason"].lower()
    assert client.calls == []


def test_staged_real_transport_entry_confirms_leverage_fill_and_child_protection(monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "FILLED" and result["protection_atomic"] is False
    assert result["protection_state"] == "ACTIVE"
    assert result["protective_exits"] == {"stop_loss_order_id": "701", "take_profit_order_id": "702"}
    assert any(c[0] == "POST" and c[1].endswith("/orders/leverage") for c in client.calls)
    assert any(c[0] == "GET" and c[1].endswith("/orders/leverage") for c in client.calls)
    entry = next(c[2] for c in client.calls if c[0] == "POST" and c[1] == "/v2/orders")
    assert entry["product_id"] == 22 and entry["product_symbol"] == "ETHUSDT"
    assert entry["order_type"] == "market_order" and entry["client_order_id"] == "jme123"
    bracket = next(c[2] for c in client.calls if c[0] == "POST" and c[1] == "/v2/orders/bracket")
    assert set(bracket) >= {"stop_loss_order", "take_profit_order", "bracket_stop_trigger_method"}


def test_live_entry_requires_currency_contract_and_confirmed_leverage(monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    client.products[0]["settling_asset"] = {"symbol": "USD"}
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "REJECTED"
    assert not any(c[0] == "POST" and c[1] == "/v2/orders" for c in client.calls)

    client = TransportDelta()
    client.products[0].pop("contract_unit_currency")
    client.products[0]["contract_value"] = 1
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "REJECTED"
    assert not any(c[0] == "POST" and c[1] == "/v2/orders" for c in client.calls)

    client = TransportDelta()
    client.leverage_confirm = False
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "REJECTED"
    assert not any(c[0] == "POST" and c[1] == "/v2/orders" for c in client.calls)


def test_entry_unknown_partial_cancel_and_protection_failure_attempt_reduce_only_close(monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    client.entry_response = {"success": False, "error": "Delta request failed"}
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "SUBMISSION_UNKNOWN" and result["authoritative"] is False

    client = TransportDelta()
    client.custom_entry_order = {"id": 501, "product_id": 22, "size": 2, "unfilled_size": 1,
                                 "average_fill_price": "100.1", "state": "open", "client_order_id": "jme123"}
    client.position = {"product_id": 22, "size": 1, "mark_price": "100.1"}
    # DELETE and subsequent GET re-read are both accepted by this fixture.
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "PARTIAL" and result["filled_quantity"] == 1

    client = TransportDelta()
    client.bracket_ok = False
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "SUBMISSION_UNKNOWN"
    assert result["protection_state"] == "FAILED_CLOSE_ATTEMPTED"
    emergency = [c for c in client.calls if c[0] == "POST" and c[1] == "/v2/orders" and c[2].get("reduce_only")]
    assert len(emergency) == 1 and emergency[0][2]["side"] == "sell"


def test_missing_bracket_children_attempts_emergency_close(monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    client.suppress_bracket_children = True
    result = client.place_protected_order(product_id=22, symbol="ETHUSDT", side="buy", size=2,
                                          order_type="market", stop_loss=98, take_profit=105,
                                          leverage=2, client_order_id="jme123",
                                          entry_authorization=jarvis_broker_authorization())
    assert result["status"] == "SUBMISSION_UNKNOWN"
    assert result["protection_state"] == "UNVERIFIED_CLOSE_ATTEMPTED"
    assert any(c[0] == "POST" and c[1] == "/v2/orders" and c[2].get("reduce_only") for c in client.calls)


def test_reduce_only_close_has_independent_full_live_authorization(monkeypatch):
    client = TransportDelta()
    for name in ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                 "DELTA_USE_MAINNET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DELTA_ORDER_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("JARVIS_KILL_SWITCH", "0")
    monkeypatch.setattr(client, "_USE_MAINNET", True)
    result = client.place_reduce_only_order(product_id=22, symbol="ETHUSDT", side="sell", size=1,
                                            client_order_id="jmc123")
    assert result.get("success") is not True
    assert not any(c[0] == "POST" and c[1] == "/v2/orders" for c in client.calls)


def test_complete_snapshot_uses_exact_product_positions_and_reports_external_orders(tmp_path):
    client = TransportDelta()
    client.position = {"product_id": 22, "size": 2, "mark_price": "100.1"}
    client.children = [
        {"id": 701, "product_id": 22, "state": "open", "stop_order_type": "stop_loss_order",
         "reduce_only": True, "side": "sell", "size": 2,
         "stop_price": "98", "limit_price": "97.9"},
        {"id": 702, "product_id": 22, "state": "open", "stop_order_type": "take_profit_order",
         "reduce_only": True, "side": "sell", "size": 2,
         "stop_price": "105", "limit_price": "105"},
    ]
    client.external_orders = [{"id": 888, "product_id": 22, "client_order_id": "manual", "state": "open"}]
    snapshot = client.get_complete_account_snapshot(owned_orders={
        "candidate": {"identity": {"instrument_id": "22"}, "order_id": "501",
                      "client_order_id": "jme123", "direction": "BUY", "quantity": 2,
                      "stop_loss": 98, "take_profit": 105,
                      "protective_exits": {"stop_loss_order_id": "701", "take_profit_order_id": "702"}}
    })
    assert snapshot["complete"] is True
    assert snapshot["risk_currency"] == "USDT"
    assert snapshot["positions"][0]["notional"] == pytest.approx(2 * 1)
    assert snapshot["orders"]["candidate"]["protection_state"] == "ACTIVE"
    assert snapshot["external_orders"] == client.external_orders
    policy = {"ETH": {"risk_currency": "USDT"}}
    adapter = DeltaExecutionAdapter(client, policy_registry=policy, authorization_check=lambda: True)
    coordinator = PortfolioCoordinator(journal_path=str(tmp_path / "external-order.json"), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=2, max_total_notional=10_000, max_total_risk=100)
    assert adapter.reconcile(coordinator) is False  # surfaced manual order blocks readiness

    client = TransportDelta()
    client.products[0]["settling_asset"] = {"symbol": "USD"}
    client.position = {"product_id": 22, "size": 0, "mark_price": "100.1"}
    snapshot = client.get_complete_account_snapshot(owned_orders={})
    # No open position means unused denomination metadata need not be projected.
    assert snapshot["complete"] is True
    client.position = {"product_id": 22, "size": 1, "mark_price": "100.1"}
    snapshot = client.get_complete_account_snapshot(owned_orders={})
    assert snapshot["complete"] is False

    client = TransportDelta()
    sol = copy.deepcopy(PRODUCT)
    sol.update({"id": 23, "symbol": "SOLUSD", "underlying_asset": {"symbol": "SOL"},
                "quoting_asset": {"symbol": "USD"}, "settling_asset": {"symbol": "USD"},
                "contract_value": "2", "contract_unit_currency": "USD"})
    client.products.append(sol)
    client.position = {"product_id": 22, "size": 1, "mark_price": "100.1"}
    client.other_positions["23"] = {"product_id": 23, "size": 1, "mark_price": "10"}
    snapshot = client.get_complete_account_snapshot(owned_orders={})
    assert snapshot["complete"] is False and "mixed settlement" in snapshot["reason"]


def test_explicit_jarvis_authorized_non_btc_delta_fixture_runs_adapter_lifecycle(tmp_path, monkeypatch):
    set_live_flags(monkeypatch)
    client = TransportDelta()
    asset_policy = {
        "ETH": {
            "policy_id": "eth-delta-risk-v1",
            "mapping_policy_id": "binance_spot_usdt_to_delta_linear_perpetual_usdt",
            "binance_symbol": "ETHUSDT", "delta_symbol": "ETHUSDT",
            "delta_instrument_id": "22", "delta_market_type": "perpetual_futures",
            "delta_quote_asset": "USDT", "delta_settling_asset": "USDT", "risk_currency": "USDT",
            "min_confidence": 70, "max_quote_age_seconds": 5, "max_spread_pct": 0.2,
            "max_chase_pct": 1.0, "max_slippage_pct": 0.2, "max_trade_risk_currency": 5.0,
        }
    }
    adapter = DeltaExecutionAdapter(client, policy_registry=asset_policy,
                                    authorization_check=lambda: True, clock=time.time)
    journal = tmp_path / "real-transport-ledger.json"
    coordinator = PortfolioCoordinator(journal_path=str(journal), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=2, max_total_notional=10_000, max_total_risk=100)
    assert adapter.reconcile(coordinator)

    now = time.time()
    # This is a complete, test-only Jarvis-produced authorization envelope for
    # exercising the Delta consumer. The production multicoin analysis producer
    # still does not emit a plan and remains non-submitting.
    test_authorization = jarvis_broker_authorization(
        entry=100.1, stop=98.0, target=105.0, snapshot="eth-snapshot-01", now=now - 0.5,
    )
    central_decision = evaluate_mtf_central_strategy(
        test_authorization["timeframe_parts"], test_authorization["part7_gate"],
        confidence=90, expected_symbol="ETHUSDT",
    )
    result = {
        "status": "COMPLETE", "scope": "parts1-12-analysis-only", "analysis_only": True,
        "decision_authority": "none", "execution_eligible": False, "freshness_status": "FRESH",
        "central_strategy_decision": central_decision,
        "central_strategy_evidence": test_authorization["part_results"],
        "central_strategy_approval": test_authorization["central_approval"],
        "central_execution_plan": test_authorization["strategy_plan"],
        "coverage": [f"Part{i}" for i in range(1, 13)], "snapshot_version": "eth-snapshot-01",
        "parts_by_timeframe": {
            tf: {f"part{i}": dict(test_authorization["timeframe_parts"][tf][name])
                 for i, name in enumerate(test_authorization["part_results"], 1)}
            for tf in REQUIRED_TIMEFRAMES
        },
        "request_identity": {"venue": "binance", "market_type": "spot", "instrument_id": "ETHUSDT", "symbol": "ETHUSDT"},
        "execution_identity": {"venue": "delta", "market_type": "perpetual_futures", "instrument_id": "22", "symbol": "ETHUSDT"},
        "mapping_policy_id": asset_policy["ETH"]["mapping_policy_id"],
        "snapshot_fetched_at": now - 2, "analysis_completed_at": now - 0.5,
        "analysis_reference": {"source": "binance", "symbol": "ETHUSDT", "timeframe": "1m",
                               "timestamp": now - 30, "price": 100.0},
        "once_per_symbol_parts": {"part11": {"signal": 1}, "part12": {"confidence": 90}},
        "part7_gate": test_authorization["part7_gate"],
        "deterministic_decision": {"origin": "jarvis_FIXED_central_strategy", "direction": "BUY",
            "confidence": 90, "entry_price": 100.0, "stop_loss": 98.0, "take_profit": 105.0},
    }
    candidate = adapter.prepare_candidate(result, now=now)
    assert candidate.identity.symbol == "ETHUSDT" and candidate.identity.instrument_id == "22"
    first = coordinator.submit(candidate, adapter)
    assert first["status"] == "FILLED"
    assert len([c for c in client.calls if c[0] == "POST" and c[1] == "/v2/orders"]) == 1
    assert coordinator.submit(candidate, adapter)["duplicate"] is True
    assert len([c for c in client.calls if c[0] == "POST" and c[1] == "/v2/orders"]) == 1
    assert coordinator.snapshot()["orders"][candidate.candidate_id]["protection_state"] == "ACTIVE"

    client.quote_bid, client.quote_ask = 105.1, 105.2
    client.quote_timestamp = int(time.time() * 1_000_000)
    monitored = adapter.monitor_once(coordinator)
    assert monitored["status"] == "MONITORED" and monitored["close_requests"] == 1
    assert client.close_order is not None and client.close_order["reduce_only"] is True
    assert client.position["size"] == 0

    reopened = PortfolioCoordinator(journal_path=str(journal), enabled=True,
        require_reconciliation=True, require_protective_reconciliation=True,
        max_positions=2, max_total_notional=10_000, max_total_risk=100)
    assert adapter.reconcile(reopened)
    assert reopened.snapshot()["orders"][candidate.candidate_id]["status"] == "CLOSED"
    assert reopened.snapshot()["ready"] is True
