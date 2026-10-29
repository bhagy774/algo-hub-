import ast
import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping
import importlib
import sys
import time
import types

import pytest
import numpy as np
import pandas as pd

from part7_signal import analyze_timeframe
from binance_timeframes import BINANCE_SPOT_TIMEFRAMES
from jarvis_strategy_approval import (
    build_execution_plan, evaluate_central_strategy, make_entry_approval,
    validate_entry_approval, validate_execution_plan,
)


NOW = 1_800_000_000.0


@pytest.fixture(autouse=True)
def isolate_process_local_position_ownership():
    """Always release test-only lifecycle claims around each independent test."""
    from jarvis_position_ownership import clear_registry
    clear_registry()
    yield
    clear_registry()


def bullish_parts(symbol="BTCUSDT"):
    return {
        "part1_breakout": {"signal": 1, "thought": "bullish breakout"},
        "part2_zone": {"signal": 1, "thought": "bullish demand zone"},
        "part3_psychology": {"signal": 1, "thought": "bullish candle"},
        "part4_volume": {"signal": 1, "thought": "volume confirmation"},
        "part5_ml": {"signal": 1, "thought": "model confirms"},
        "part6_trend": {"signal": 1, "thought": "trend bullish"},
        "part7_volatility": {"signal": 1, "thought": "volatility expansion"},
        "part8_structure": {"signal": 1, "thought": "structure bullish"},
        "part9_orderflow": {"signal": 1, "thought": "order flow bullish"},
        "part10_candlestats": {"signal": 1, "thought": "bullish candle stats"},
    }


def timeframe_parts(symbol="BTCUSDT", *, low_only=True):
    """Complete exact-frame evidence for an approved, deterministic SCALP."""
    rows = {}
    for timeframe in BINANCE_SPOT_TIMEFRAMES:
        bullish = not low_only or timeframe in {"1m", "3m", "5m", "15m"}
        row = {}
        for name, item in bullish_parts(symbol).items():
            value = dict(item)
            value.update({"symbol": symbol, "timeframe": timeframe})
            if name == "part7_volatility":
                value.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                              "data_status": "valid", "entry_blocked": False, "risk_veto": False})
            elif not bullish:
                value.update({"signal": 0, "thought": "neutral native-frame evidence"})
            if name == "part2_zone":
                value["native_timeframe"] = timeframe
                if not bullish:
                    value.update({"signal": 0, "thought": "no native zone"})
            row[name] = value
        rows[timeframe] = row
    return rows


def clear_gate(symbol=None):
    return {
        "entry_blocked": False, "risk_veto": False, "status": "ok",
        "data_status": "valid", "timeframe": "aggregate", "symbol": symbol,
        "blocked_timeframes": [], "veto_timeframes": [],
        "timeframe_results": {
            tf: {"symbol": symbol, "timeframe": tf, "status": "neutral",
                 "data_status": "valid", "entry_blocked": False, "risk_veto": False}
            for tf in BINANCE_SPOT_TIMEFRAMES
        },
    }


def approval_for(symbol="BTCUSDT", snapshot="snap-1", confidence=80, now=NOW):
    decision = {"approved": True, "direction": "BUY", "confidence": confidence, "symbol": symbol}
    return make_entry_approval(
        decision, direction="BUY", symbol=symbol, exchange="delta", contract=symbol,
        instrument_id=symbol, market_type="unverified", analysis_symbol=symbol,
        analysis_exchange="delta", snapshot_version=snapshot,
        analysis_timestamp=now, confidence=confidence,
    )


def test_central_strategy_positive_neutral_veto_and_fail_closed_semantics():
    positive = evaluate_central_strategy(bullish_parts(), clear_gate(), confidence=80)
    assert positive["approved"] and positive["direction"] == "BUY"
    assert not evaluate_central_strategy(
        bullish_parts(), clear_gate(), expected_symbol="BTCUSDT"
    )["approved"]
    assert evaluate_central_strategy(
        bullish_parts(), clear_gate("BTCUSDT"), expected_symbol="BTCUSDT"
    )["approved"]
    assert not evaluate_central_strategy(
        bullish_parts(), clear_gate("ETHUSDT"), expected_symbol="BTCUSDT"
    )["approved"]
    partial_gate = clear_gate("BTCUSDT")
    partial_gate["timeframe_results"].pop("4h")
    assert not evaluate_central_strategy(
        bullish_parts(), partial_gate, expected_symbol="BTCUSDT"
    )["approved"]
    stale_gate = clear_gate("BTCUSDT")
    stale_gate["timeframe_results"]["1m"]["data_status"] = "stale"
    assert not evaluate_central_strategy(
        bullish_parts(), stale_gate, expected_symbol="BTCUSDT"
    )["approved"]

    neutral_parts = bullish_parts()
    neutral_parts["part3_psychology"] = {"signal": 0, "thought": "indecisive"}
    assert evaluate_central_strategy(neutral_parts, clear_gate())["direction"] == "BUY"

    blocked = {**clear_gate(), "entry_blocked": True, "risk_veto": True, "reason": "extreme volatility"}
    assert not evaluate_central_strategy(bullish_parts(), blocked)["approved"]
    assert not evaluate_central_strategy(bullish_parts(), None)["approved"]
    assert not evaluate_central_strategy({k: v for k, v in bullish_parts().items() if k != "part4_volume"}, clear_gate())["approved"]

    zone = bullish_parts()
    zone["part2_zone"] = {"signal": -1, "thought": "resistance zone"}
    assert not evaluate_central_strategy(zone, clear_gate())["approved"]
    dissent = bullish_parts()
    dissent["part6_trend"] = {"signal": -1, "thought": "bearish trend"}
    dissent["part8_structure"] = {"signal": -1, "thought": "bearish structure"}
    assert not evaluate_central_strategy(dissent, clear_gate())["approved"]

    malformed = bullish_parts()
    malformed["part1_breakout"] = {"signal": float("nan"), "thought": "bad"}
    assert not evaluate_central_strategy(malformed, clear_gate())["approved"]
    unavailable = bullish_parts()
    unavailable["part5_ml"] = {"signal": 1, "thought": "offline model fallback"}
    assert evaluate_central_strategy(unavailable, clear_gate())["direction"] == "BUY"


def test_part7_stale_wrong_symbol_and_partial_data_blocks_are_fail_closed():
    now = pd.Timestamp.now(tz="UTC")
    values = np.full(60, 100.0)
    def frame(end, columns=True, symbol="ETHUSDT"):
        data = {"open": values, "high": values + .2, "low": values - .2,
                "close": values, "volume": np.full(60, 1000.0)}
        if not columns:
            data.pop("volume")
        result = pd.DataFrame(data, index=pd.date_range(end=end, periods=60, freq="min"))
        result.attrs["symbol"] = symbol
        return result
    context = {"selected_symbol": "ETHUSDT", "timeframe": "1m",
               "is_backtest_mode": False, "shared_candle_source": "shared_exchange"}
    stale = analyze_timeframe(frame(now - pd.Timedelta(minutes=15)), symbol="ETHUSDT",
                              timeframe="1m", context=context)
    wrong = analyze_timeframe(frame(now - pd.Timedelta(minutes=2), symbol="BTCUSDT"),
                              symbol="ETHUSDT", timeframe="1m", context=context)
    partial = analyze_timeframe(frame(now - pd.Timedelta(minutes=2), columns=False),
                                symbol="ETHUSDT", timeframe="1m", context=context)
    assert stale["entry_blocked"] is True
    assert wrong["entry_blocked"] is True
    assert partial["entry_blocked"] is True


def test_approval_scope_freshness_identity_and_snapshot_are_mandatory():
    approval = approval_for()
    args = dict(direction="BUY", symbol="BTCUSDT", exchange="delta", contract="BTCUSDT",
                instrument_id="BTCUSDT", market_type="unverified", snapshot_version="snap-1",
                confidence=80, now=NOW + 1)
    assert validate_entry_approval(approval, **args)[0]
    for override in (
        {"symbol": "ETHUSDT"}, {"instrument_id": "other"},
        {"snapshot_version": "old"}, {"direction": "SELL"},
    ):
        assert not validate_entry_approval(approval, **(args | override))[0]
    assert not validate_entry_approval(approval, **(args | {"now": NOW + 181}))[0]


def _live_trader_module(monkeypatch):
    risk = types.ModuleType("jarvis_risk")
    risk.calculate_trade_size = lambda *a, **k: {}
    risk.MAX_LEVERAGE_CAP = 20
    risk.contract_quote_value_usdt = lambda *a, **k: None
    lots = types.ModuleType("jarvis_lot_limits")
    lots.enforce_entry_lots = lambda *a, **k: 0
    monkeypatch.setitem(sys.modules, "jarvis_risk", risk)
    monkeypatch.setitem(sys.modules, "jarvis_lot_limits", lots)
    monkeypatch.delitem(sys.modules, "jarvis_live_trader", raising=False)
    module = importlib.import_module("jarvis_live_trader")
    # Restore the prior cached module after this test instead of leaking a
    # trader whose sizing dependencies were intentionally stubbed above.
    monkeypatch.setitem(sys.modules, "jarvis_live_trader", module)
    return module


def test_live_auto_trader_direct_entry_requires_approval_but_exit_is_ungated(monkeypatch):
    module = _live_trader_module(monkeypatch)
    trader = object.__new__(module.JarvisAutoTrader)
    trader.emergency_stop = False
    result = trader.execute("CALL", 80, 100.0, bullish_parts(), symbol="BTCUSDT")
    assert not result["success"] and "approval" in result["reason"].lower()
    direct = trader._place_trade("CALL", 80, 100.0, "SCALP", {}, symbol="BTCUSDT")
    assert not direct["success"] and "approval" in direct["reason"].lower()

    orders = []
    trader.is_enabled = True
    trader._ownership_token = "test-owner"
    trader.delta = types.SimpleNamespace(place_order=lambda **kwargs: (orders.append(kwargs) or {"success": True}))
    close = trader._close_position_market({"id": "p1", "symbol": "BTCUSDT", "direction": "CALL", "contracts": 1})
    assert close["success"] and orders[0]["reduce_only"] is True
    assert orders[0]["client_order_id"].startswith("jvc")
    assert "idempotency_key" not in orders[0]


def test_paper_entry_helper_direct_bypass_attempt_is_rejected():
    source = Path(__file__).resolve().parents[1] / "jarvis_FIXED.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "LiveTradingEngine")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_open_paper_trade")
    harness = ast.ClassDef(name="PaperEntryHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_binance_spot_symbol")
    module_ast = ast.fix_missing_locations(ast.Module(body=[helper, harness], type_ignores=[]))
    namespace = {"datetime": datetime, "timedelta": timedelta}
    exec(compile(module_ast, str(source), "exec"), namespace)
    engine = namespace["PaperEntryHarness"]()
    engine.paper_open_trades = []
    engine.PAPER_CONFIG = {"max_open_trades": 5, "expiry_map": {}}
    engine._dashboard_events = []
    engine._paper_size = lambda *args, **kwargs: {"ok": False, "reason": "blocked by bypass test"}
    opened = engine._open_paper_trade("CALL", 100.0, 80, "SCALP", 101.0, 102.0, 99.0)
    assert opened is None
    assert not engine._dashboard_events or "blocked" in engine._dashboard_events[-1]


def _isolated_async_method(repo_file, class_name, method_name):
    source = Path(__file__).resolve().parents[1] / repo_file
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name)
    harness = ast.ClassDef(name="EntryHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    module_ast = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {"Any": Any, "Dict": Dict, "Mapping": Mapping}
    exec(compile(module_ast, str(source), "exec"), namespace)
    return namespace["EntryHarness"]()


def test_legacy_part10_and_part12_direct_entry_apis_require_central_approval():
    legacy = _isolated_async_method("part10_FIXED.py", "SwingScalpTradeExecutor", "execute_trade")
    legacy_result = asyncio.run(legacy.execute_trade("CALL"))
    assert legacy_result["status"] == "rejected"
    assert "central" in legacy_result["reason"].lower()

    engine = _isolated_async_method("part12_FIXED.py", "GPUOrderExecutionEngine", "execute_trade")
    result = asyncio.run(engine.execute_trade("CALL", 9.0, 100.0, 100.1))
    assert result["status"] == "rejected"
    assert "central" in result["reason"].lower()


def test_execution_plan_is_deterministic_and_scope_bound_for_scalp_and_swing():
    scalp = build_execution_plan(
        direction="BUY", recommended_expiry="SCALP", entry_price=100,
        stop_loss=99, take_profit=102, symbol="BTCUSDT",
        snapshot_version="snap-plan-1", confidence=80,
    )
    swing = build_execution_plan(
        direction="SELL", recommended_expiry="DAY_TRADE", entry_price=100,
        stop_loss=101, take_profit=98, symbol="BTCUSDT",
        snapshot_version="snap-plan-1", confidence=80,
    )
    assert scalp["trade_mode"] == "SCALP" and scalp["stop_loss"] == 99 and scalp["take_profit"] == 102
    assert swing["trade_mode"] == "SWING" and swing["stop_loss"] == 100.8 and swing["take_profit"] == 98.0
    assert validate_execution_plan(scalp, direction="BUY", symbol="BTCUSDT",
                                   snapshot_version="snap-plan-1", confidence=80,
                                   trade_mode="SCALP")[0]
    for kwargs in (
        {"symbol": "ETHUSDT"}, {"snapshot_version": "stale"},
        {"direction": "SELL"}, {"confidence": 81}, {"trade_mode": "SWING"},
    ):
        params = dict(direction="BUY", symbol="BTCUSDT", snapshot_version="snap-plan-1",
                      confidence=80, trade_mode="SCALP")
        params.update(kwargs)
        assert not validate_execution_plan(scalp, **params)[0]
    assert not validate_execution_plan(None, direction="BUY", symbol="BTCUSDT",
                                       snapshot_version="snap-plan-1")[0]


def test_live_central_validation_requires_plan_bound_approval(monkeypatch):
    module = _live_trader_module(monkeypatch)
    symbol = "BTCUSDT"
    evidence = bullish_parts(symbol)
    timeframe_evidence = timeframe_parts(symbol)
    gate = clear_gate(symbol)
    from jarvis_strategy_approval import evaluate_mtf_central_strategy
    decision = evaluate_mtf_central_strategy(timeframe_evidence, gate, confidence=80, expected_symbol=symbol)
    now = time.time()
    plan = build_execution_plan(
        direction="BUY", recommended_expiry="SCALP", entry_price=100,
        stop_loss=99, take_profit=102, symbol=symbol,
        snapshot_version="snap-live", confidence=80,
    )
    approval = make_entry_approval(
        {**decision, "symbol": symbol}, direction="BUY", symbol=symbol,
        exchange="delta", contract=symbol, instrument_id=symbol,
        market_type="unverified", snapshot_version="snap-live",
        analysis_timestamp=now, confidence=80, execution_plan=plan,
    )
    approval["trade_mode"] = "SCALP"
    assert module._validate_central_entry(approval, evidence, gate, "CALL", 80,
           symbol, "snap-live", execution_plan=plan, trade_mode="SCALP",
           timeframe_parts=timeframe_evidence)[0]
    assert not module._validate_central_entry(approval, evidence, gate, "CALL", 80,
           symbol, "snap-live", execution_plan=None, trade_mode="SCALP",
           timeframe_parts=timeframe_evidence)[0]
    assert not module._validate_central_entry(approval, evidence, gate, "CALL", 80,
           symbol, "snap-live", execution_plan=plan, trade_mode="SWING",
           timeframe_parts=timeframe_evidence)[0]


def test_paper_trade_uses_bound_plan_levels_and_never_an_injected_tp2():
    source = Path(__file__).resolve().parents[1] / "jarvis_FIXED.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "LiveTradingEngine")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_open_paper_trade")
    harness = ast.ClassDef(name="PaperEntryHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_binance_spot_symbol")
    module_ast = ast.fix_missing_locations(ast.Module(body=[helper, harness], type_ignores=[]))
    namespace = {"datetime": datetime, "timedelta": timedelta}
    exec(compile(module_ast, str(source), "exec"), namespace)
    engine = namespace["PaperEntryHarness"]()
    engine._dashboard_events = []
    engine.paper_open_trades = []
    engine.PAPER_CONFIG = {"max_open_trades": 2, "expiry_map": {"SCALP": 5}}
    engine._paper_size = lambda confidence, entry, stop, direction: {
        "ok": True, "contracts": 1, "margin_usdt": 2, "notional_usdt": 100,
        "leverage": 2, "trade_risk_usdt": 1,
    }
    symbol = "BTCUSDT"
    evidence = bullish_parts(symbol)
    timeframe_evidence = timeframe_parts(symbol)
    gate = clear_gate(symbol)
    from jarvis_strategy_approval import evaluate_mtf_central_strategy
    decision = evaluate_mtf_central_strategy(timeframe_evidence, gate, confidence=80, expected_symbol=symbol)
    plan = build_execution_plan(
        direction="BUY", recommended_expiry="SCALP", entry_price=100,
        stop_loss=99, take_profit=102, symbol=symbol,
        snapshot_version="snap-paper", confidence=80,
    )
    approval = make_entry_approval(
        {**decision, "symbol": symbol}, direction="BUY", symbol=symbol,
        exchange="delta", contract=symbol, instrument_id=symbol,
        market_type="unverified", snapshot_version="snap-paper",
        analysis_timestamp=time.time(), confidence=80, execution_plan=plan,
    )
    approval["trade_mode"] = "SCALP"
    trade = engine._open_paper_trade(
        "CALL", 100.0, 80, "SCALP", 102.0, 102.0, 99.0,
        current_price=100, symbol=symbol, central_approval=approval,
        snapshot_version="snap-paper", part_results=evidence, timeframe_parts=timeframe_evidence,
        part7_gate=gate, execution_plan=plan,
    )
    assert trade is not None
    assert trade["entry_price"] == 100
    assert trade["sl"] == 99 and trade["tp1"] == 102 and trade["tp2"] == 102


@pytest.mark.parametrize("fill_price", [100.0, 100.01])
def test_live_auto_trader_uses_plan_levels_and_protected_delta_adapter(monkeypatch, fill_price):
    monkeypatch.setenv("JARVIS_MAX_RISK_USDT", "100.0")
    module = _live_trader_module(monkeypatch)
    symbol = "BTCUSDT"
    evidence = bullish_parts(symbol)
    timeframe_evidence = timeframe_parts(symbol)
    gate = clear_gate(symbol)
    from jarvis_strategy_approval import evaluate_mtf_central_strategy
    decision = evaluate_mtf_central_strategy(timeframe_evidence, gate, confidence=80, expected_symbol=symbol)
    plan = build_execution_plan(
        direction="BUY", recommended_expiry="SCALP", entry_price=100,
        stop_loss=99, take_profit=102, symbol=symbol,
        snapshot_version="snap-live-order", confidence=80,
    )
    approval = make_entry_approval(
        {**decision, "symbol": symbol}, direction="BUY", symbol=symbol,
        exchange="delta", contract=symbol, instrument_id=symbol,
        market_type="unverified", snapshot_version="snap-live-order",
        analysis_timestamp=time.time(), confidence=80, execution_plan=plan,
    )
    approval["trade_mode"] = "SCALP"
    submitted = []
    delta = types.SimpleNamespace(
        get_product_metadata=lambda requested: {"id": 22, "symbol": requested, "max_leverage": 10},
        get_wallet_balance=lambda: 100.0,
        place_protected_order=lambda **kwargs: (submitted.append(kwargs) or {
            "status": "FILLED", "authoritative": True, "order_id": "entry-1",
            "filled_quantity": kwargs["size"], "average_fill_price": fill_price,
            "protection_state": "ACTIVE",
        }),
        place_order=lambda **kwargs: pytest.fail("new entry used raw place_order"),
    )
    trader = object.__new__(module.JarvisAutoTrader)
    trader.emergency_stop = False
    trader._entry_reconciliation_required = False
    trader.is_enabled = True
    trader.delta = delta
    trader._ownership_token = "test-owner"
    trader.open_positions = []
    trader.daily_trades = 0
    trader.last_trade_time = None
    trader._oracle_gate = None
    monkeypatch.setattr(module, "contract_quote_value_usdt", lambda product, price: 100.0)
    monkeypatch.setattr(module, "calculate_trade_size", lambda *args, **kwargs: {
        "ok": True, "contracts": 1, "leverage": 2, "margin_usdt": 5.0,
        "notional_usdt": 100.0, "contract_value_usdt": 100.0,
        "risk_amount": 1.0,
    })
    monkeypatch.setattr(module, "enforce_entry_lots", lambda quantity, **kwargs: quantity)
    monkeypatch.setattr(module, "_get_sizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "claim_position", lambda *args, **kwargs: True)
    result = trader._place_trade(
        "CALL", 80, 100.0, "SCALP", {}, symbol=symbol,
        central_approval=approval, part_results=evidence, timeframe_parts=timeframe_evidence,
        part7_gate=gate, snapshot_version="snap-live-order", execution_plan=plan,
    )
    assert submitted[0]["stop_loss"] == 99 and submitted[0]["take_profit"] == 102
    assert submitted[0]["entry_authorization"]["strategy_plan"] == plan
    assert submitted[0]["entry_authorization"]["broker_plan"]["entry_price"] == 100
    if fill_price == 100.0:
        assert result["success"]
        assert trader.open_positions[0]["entry_price"] == 100.0
    else:
        assert not result["success"]
        assert result["reconciliation_required"] is True
        assert trader._pending_entry_reconciliation["execution_plan"] == plan
        assert trader.open_positions == []


def _pending_test_entry(trader, module, *, direction="BUY"):
    symbol = "BTCUSDT"
    plan = build_execution_plan(
        direction=direction, recommended_expiry="SCALP", entry_price=100,
        stop_loss=99 if direction == "BUY" else 101,
        take_profit=102 if direction == "BUY" else 98,
        symbol=symbol, snapshot_version="snap-reconcile", confidence=80,
    )
    trader._remember_entry_reconciliation(
        product={"id": 22, "symbol": symbol, "contract_value_usdt": 100.0},
        symbol=symbol, direction=direction, confidence=80, quantity=1,
        leverage=2, balance=100.0, execution_plan=plan,
        client_order_id="jvtestreconcile1",
        adoption_allowed=True,
        response={"order_id": "entry-reconcile-1", "protective_exits": {
            "stop_loss_order_id": "stop-1", "take_profit_order_id": "target-1"}},
    )
    trader.open_positions = []
    trader._ownership_token = "reconcile-test"
    trader.daily_trades = 0
    trader.last_trade_time = None
    return plan


def test_uncertain_entry_latch_clears_only_on_complete_terminal_broker_truth(monkeypatch):
    module = _live_trader_module(monkeypatch)
    trader = object.__new__(module.JarvisAutoTrader)
    trader.delta = types.SimpleNamespace(get_complete_account_snapshot=lambda **kwargs: {
        "complete": False, "reason": "incomplete account snapshot"})
    _pending_test_entry(trader, module)
    assert not trader.reconcile_pending_entry()
    assert trader._entry_reconciliation_required
    trader.delta.get_complete_account_snapshot = lambda **kwargs: {
        "complete": True, "as_of": time.time(), "positions": [], "external_orders": [],
        "orders": {next(iter(kwargs["owned_orders"])): {"status": "REJECTED", "authoritative": True}},
    }
    assert trader.reconcile_pending_entry()
    assert not trader._entry_reconciliation_required
    assert trader._pending_entry_reconciliation is None
    trader.emergency_stop = True
    assert trader.resume()


def test_uncertain_protected_position_is_adopted_only_after_exact_reconciliation(monkeypatch):
    monkeypatch.setenv("JARVIS_MAX_RISK_USDT", "100.0")
    module = _live_trader_module(monkeypatch)
    monkeypatch.setattr(module, "contract_quote_value_usdt", lambda product, price: 100.0)
    monkeypatch.setattr(module, "claim_position", lambda *args: True)
    trader = object.__new__(module.JarvisAutoTrader)
    plan = _pending_test_entry(trader, module)
    def complete_snapshot(**kwargs):
        candidate_id = next(iter(kwargs["owned_orders"]))
        return {
            "complete": True, "as_of": time.time(), "external_orders": [],
            "orders": {candidate_id: {"status": "FILLED", "authoritative": True,
                "filled_quantity": 1, "average_fill_price": 100.0,
                "protection_state": "ACTIVE",
                "protective_exits": {"stop_loss_order_id": "stop-1", "take_profit_order_id": "target-1"}}},
            "positions": [{"venue": "delta", "market_type": "perpetual_futures",
                "instrument_id": "22", "symbol": "BTCUSDT", "size": 1.0}],
        }
    trader.delta = types.SimpleNamespace(get_complete_account_snapshot=complete_snapshot)
    trader.emergency_stop = True
    assert trader.reconcile_pending_entry()
    assert not trader._entry_reconciliation_required
    assert trader._pending_entry_reconciliation is None
    assert len(trader.open_positions) == 1
    position = trader.open_positions[0]
    assert position["entry_price"] == 100.0
    assert position["sl_price"] == plan["stop_loss"] and position["tp_price"] == plan["take_profit"]
    assert position["protective_exits"] == {"stop_loss_order_id": "stop-1", "take_profit_order_id": "target-1"}
    assert trader.emergency_stop  # reconciliation never auto-resumes execution
