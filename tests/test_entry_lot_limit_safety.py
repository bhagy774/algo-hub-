"""Offline regressions for final entry-lot enforcement and risk accounting."""
from __future__ import annotations
from central_entry_fixture import entry_evidence

from pathlib import Path

import pytest

from jarvis_lot_limits import enforce_entry_lots, entry_lot_limits


@pytest.fixture(autouse=True)
def clear_lot_policy_env(monkeypatch):
    # Keep the process-local ownership registry isolated from adjacent safety
    # suites, which can otherwise reuse timestamp-based paper order IDs.
    from jarvis_position_ownership import clear_registry

    clear_registry()
    monkeypatch.delenv("JARVIS_MIN_ENTRY_LOTS", raising=False)
    monkeypatch.delenv("JARVIS_MAX_ENTRY_LOTS", raising=False)
    yield
    clear_registry()


def test_unconfigured_bounds_and_downward_floor_only():
    assert entry_lot_limits() == (None, None)
    assert enforce_entry_lots(3.9) == 3


def test_configured_bounds_and_steps_only_reduce(monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "4")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "10")
    assert enforce_entry_lots(99) == 10
    assert enforce_entry_lots(9, metadata={"size_increment": 2}) == 8
    with pytest.raises(ValueError, match="below minimum"):
        enforce_entry_lots(3.9)
    with pytest.raises(ValueError, match="below minimum"):
        enforce_entry_lots(5, metadata={"size_increment": 3})


def test_contract_size_is_not_an_order_quantity_step():
    # Delta contract_size represents underlying amount/contract value, not a
    # discrete quantity increment. Explicit order-size metadata remains valid.
    assert enforce_entry_lots(7, metadata={"contract_size": "0.001"}) == 7
    assert enforce_entry_lots(7, metadata={"contract_size": 0}) == 7
    assert enforce_entry_lots(7, metadata={"contract_step": 2}) == 6


def test_invalid_config_quantity_balance_and_explicit_step_fail_closed(monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "4")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "8")
    for value in (None, True, 0, -1, float("nan"), float("inf"), "bad"):
        with pytest.raises(ValueError):
            enforce_entry_lots(value)
    for value in (0, -1, float("nan"), float("inf"), "bad"):
        with pytest.raises(ValueError):
            enforce_entry_lots(4, available_balance=value)
    for step in (0, -1, float("nan"), float("inf"), 1.5, "bad"):
        with pytest.raises(ValueError):
            enforce_entry_lots(4, metadata={"size_increment": step})
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "9")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "8")
    with pytest.raises(ValueError, match="must not exceed"):
        enforce_entry_lots(9)


def test_reduce_only_quantity_is_not_limited_by_entry_minimum_or_step(monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "4")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "8")
    # Entry balance and order-step metadata, even when malformed, must not
    # block a protective exit.
    assert enforce_entry_lots(
        1, metadata={"size_increment": 0}, available_balance=0, reduce_only=True
    ) == 1


class DeltaStub:
    """In-process venue stub; the tests never create network or venue traffic."""

    def __init__(self, product=None, balance=100.0):
        self.product = product
        self.balance = balance
        self.calls = []

    def get_wallet_balance(self):
        return self.balance

    def get_product_metadata(self, symbol):
        return self.product

    def set_leverage(self, symbol, leverage):
        self.calls.append(("leverage", symbol, leverage))
        return True

    def place_protected_order(self, **kwargs):
        assert kwargs["stop_loss"] < kwargs["entry_authorization"]["broker_plan"]["entry_price"] < kwargs["take_profit"]
        assert self.set_leverage(kwargs["symbol"], kwargs["leverage"])
        self.calls.append(("order", kwargs))
        return {"status": "FILLED", "authoritative": True, "protection_state": "ACTIVE",
                "filled_quantity": kwargs["size"], "average_fill_price": kwargs["entry_authorization"]["broker_plan"]["entry_price"],
                "order_id": "offline-order"}

    def place_order(self, **kwargs):
        self.calls.append(("order", kwargs))
        return {"success": True, "order_id": "offline-order"}


def _make_trader(monkeypatch, delta, *, enabled):
    import jarvis_live_trader

    monkeypatch.setattr(jarvis_live_trader, "_get_sizer", lambda _delta: None)
    monkeypatch.setattr(jarvis_live_trader, "_get_oracle_gate", lambda: None)
    trader = jarvis_live_trader.JarvisAutoTrader(delta)
    trader.is_enabled = enabled
    return trader


def test_capped_entry_recomputes_notional_margin_and_trade_risk(monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "2")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "4")
    monkeypatch.setenv("DELTA_ORDER_EXECUTION_ENABLED", "true")
    delta = DeltaStub({
        "id": 1,
        "symbol": "BTCUSD",
        "contract_value": "2",
        "contract_value_currency": "USDT",
        "contract_size": "0.001",
        "max_leverage": 10,
    })
    trader = _make_trader(monkeypatch, delta, enabled=True)

    result = trader._place_trade("CALL", 95, 100.0, entry_evidence("BTCUSDT", "CALL", 95, 100.0)[1], {"do_hedge": False}, "BTCUSDT", **entry_evidence("BTCUSDT", "CALL", 95, 100.0)[0])

    assert result["success"] is True
    position = result["position"]
    assert position["contracts"] == 4
    assert position["contract_value_usdt"] == pytest.approx(2.0)
    assert position["notional_usdt"] == pytest.approx(8.0)
    assert position["margin_usdt"] == pytest.approx(8.0 / position["leverage"])
    plan = entry_evidence("BTCUSDT", "CALL", 95, 100.0)[0]["execution_plan"]
    assert position["trade_risk_usdt"] == pytest.approx(8.0 * plan["risk_fraction"])
    from jarvis_live_trader import MAX_RISK_USDT
    assert position["trade_risk_usdt"] <= MAX_RISK_USDT
    assert delta.calls[0][0] == "leverage"
    assert delta.calls[1][0] == "order"
    assert delta.calls[1][1]["size"] == 4


def test_below_minimum_blocks_before_venue_mutations(monkeypatch):
    monkeypatch.setenv("JARVIS_MIN_ENTRY_LOTS", "1000")
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "1000")
    delta = DeltaStub({
        "id": 1,
        "symbol": "BTCUSD",
        "contract_value": "2",
        "contract_value_currency": "USDT",
        "contract_size": "0.001",
        "max_leverage": 10,
    })
    trader = _make_trader(monkeypatch, delta, enabled=True)
    result = trader._place_trade("CALL", 95, 100.0, entry_evidence("BTCUSDT", "CALL", 95, 100.0)[1], {"do_hedge": False}, "BTCUSDT", **entry_evidence("BTCUSDT", "CALL", 95, 100.0)[0])
    assert result["success"] is False
    assert result["reason"] == "Entry lot policy blocked"
    assert delta.calls == []


def test_paper_entry_keeps_lot_policy_but_never_mutates_venue(monkeypatch):
    monkeypatch.setenv("JARVIS_MAX_ENTRY_LOTS", "2")
    delta = DeltaStub(balance=100.0)
    trader = _make_trader(monkeypatch, delta, enabled=False)
    result = trader._place_trade("CALL", 90, 100.0, entry_evidence("BTCUSDT", "CALL", 90, 100.0)[1], {"do_hedge": False}, "BTCUSDT", **entry_evidence("BTCUSDT", "CALL", 90, 100.0)[0])
    assert result["success"] is True
    assert result["position"]["contracts"] == 2
    assert delta.calls == []


def test_entry_gate_remains_before_live_mutations_and_exits_remain_reduce_only():
    root = Path(__file__).resolve().parent
    if not (root / "jarvis_live_trader.py").exists():
        root = root.parent
    text = (root / "jarvis_live_trader.py").read_text(encoding="utf-8")
    gate = text.index("contracts = enforce_entry_lots")
    assert gate < text.index('submitter = getattr(self.delta, "place_protected_order"', gate)
    assert gate < text.index("protected = submitter(", gate)
    assert "raw entries are disabled" in text
    assert "reduce_only=True" in text
