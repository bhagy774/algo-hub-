"""Fresh deterministic approvals for offline execution-stub tests."""
import time
from test_multitimeframe_strategy_policy import evidence_by_timeframe, aggregate_gate
from jarvis_strategy_approval import evaluate_mtf_central_strategy, build_execution_plan, make_entry_approval

def entry_evidence(symbol, direction, confidence, price):
    side = "BUY" if direction in ("BUY", "CALL") else "SELL"
    sign = 1 if side == "BUY" else -1
    parts = evidence_by_timeframe(symbol=symbol, default_direction=sign)
    gate = aggregate_gate(symbol)
    decision = evaluate_mtf_central_strategy(parts, gate, confidence=confidence, expected_symbol=symbol)
    assert decision["approved"] is True, decision
    version = "offline-approved-native-16"
    plan = build_execution_plan(direction=side, recommended_expiry=decision["trade_mode"],
        entry_price=price, stop_loss=price*(1-.002*sign), take_profit=price*(1+.004*sign),
        symbol=symbol, snapshot_version=version, confidence=confidence)
    approval = make_entry_approval(decision, direction=side, symbol=symbol,
        exchange="delta", contract=symbol, instrument_id=symbol, market_type="unverified",
        snapshot_version=version, analysis_timestamp=time.time(), confidence=confidence, execution_plan=plan)
    return {"central_approval":approval, "part_results":parts["1m"], "part7_gate":gate,
            "snapshot_version":version, "execution_plan":plan, "timeframe_parts":parts}, decision["trade_mode"]
