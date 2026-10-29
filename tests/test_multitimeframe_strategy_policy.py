import time

from jarvis_strategy_approval import (
    REQUIRED_TIMEFRAMES,
    TIMEFRAME_CORRELATION_GROUPS,
    TIMEFRAME_GROUP_WEIGHTS,
    build_execution_plan,
    evaluate_mtf_central_strategy,
    make_entry_approval,
    validate_entry_approval,
    validate_execution_plan,
)


SYMBOL = "BTCUSDT"
PARTS = (
    "part1_breakout", "part2_zone", "part3_psychology", "part4_volume",
    "part5_ml", "part6_trend", "part7_volatility", "part8_structure",
    "part9_orderflow", "part10_candlestats",
)


def frame_parts(direction=1, symbol=SYMBOL, timeframe="1m"):
    """Deterministic, symbol/timeframe-scoped offline evidence fixture."""
    values = {}
    for name in PARTS:
        signal = direction
        thought = "bullish evidence" if direction > 0 else "bearish evidence" if direction < 0 else "neutral evidence"
        if name == "part2_zone":
            thought = "support zone" if direction > 0 else "resistance zone" if direction < 0 else "no native zone"
        if name == "part7_volatility":
            signal = 0
            thought = "stable volatility"
            values[name] = {
                "signal": signal, "thought": thought, "symbol": symbol,
                "timeframe": timeframe, "status": "neutral", "data_status": "valid",
                "entry_blocked": False, "risk_veto": False,
            }
        else:
            values[name] = {
                "signal": signal, "thought": thought, "symbol": symbol,
                "timeframe": timeframe,
            }
    # Part 2 uses this native identity field in Jarvis' adapter.
    values["part2_zone"]["native_timeframe"] = timeframe
    return values


def evidence_by_timeframe(*, default_direction=1, overrides=None, symbol=SYMBOL):
    overrides = overrides or {}
    return {
        timeframe: frame_parts(overrides.get(timeframe, default_direction), symbol, timeframe)
        for timeframe in REQUIRED_TIMEFRAMES
    }


def aggregate_gate(symbol=SYMBOL):
    rows = {
        timeframe: {
            "symbol": symbol, "timeframe": timeframe, "status": "neutral",
            "data_status": "valid", "entry_blocked": False, "risk_veto": False,
        }
        for timeframe in REQUIRED_TIMEFRAMES
    }
    return {
        "symbol": symbol, "timeframe": "aggregate", "status": "ok",
        "data_status": "valid", "entry_blocked": False, "risk_veto": False,
        "blocked_timeframes": [], "veto_timeframes": [], "timeframe_results": rows,
    }


def test_swing_pending_until_1m_trigger_then_approves_and_opposite_blocks():
    gate = aggregate_gate()
    pending = evaluate_mtf_central_strategy(
        evidence_by_timeframe(overrides={"1m": 0}), gate,
        confidence=80, expected_symbol=SYMBOL,
    )
    assert pending["status"] == "PENDING"
    assert pending["approved"] is False
    assert pending["trade_mode"] == "SWING"
    assert pending["setup_direction"] == "BUY"
    assert pending["entry_trigger"] == "NEUTRAL"

    aligned = evaluate_mtf_central_strategy(
        evidence_by_timeframe(), gate, confidence=80, expected_symbol=SYMBOL,
    )
    assert aligned["approved"] is True
    assert aligned["trade_mode"] == "SWING"
    assert aligned["direction"] == "BUY"
    assert aligned["entry_trigger"] == "ALIGNED"

    opposite = evaluate_mtf_central_strategy(
        evidence_by_timeframe(overrides={"1m": -1}), gate,
        confidence=80, expected_symbol=SYMBOL,
    )
    assert opposite["approved"] is False
    assert opposite["status"] == "BLOCKED"
    assert opposite["entry_trigger"] == "OPPOSITE"


def test_scalp_mode_is_distinct_and_countertrend_higher_frames_veto_it():
    # A low-frame setup is SCALP only when the higher swing group is not aligned.
    low_only = evidence_by_timeframe(overrides={tf: 0 for tf in ("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M")})
    scalp = evaluate_mtf_central_strategy(low_only, aggregate_gate(), confidence=80, expected_symbol=SYMBOL)
    assert scalp["approved"] is True
    assert scalp["trade_mode"] == "SCALP"

    # The upper guard is countertrend, but is not large enough to independently
    # establish the broader 3m–4h SWING direction; it therefore vetoes SCALP.
    countertrend = evidence_by_timeframe(overrides={**{tf: -1 for tf in ("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")}, "3d": 0, "1w": 0, "1M": 0})
    blocked = evaluate_mtf_central_strategy(countertrend, aggregate_gate(), confidence=80, expected_symbol=SYMBOL)
    assert blocked["approved"] is False
    assert blocked["trade_mode"] == "SCALP"
    assert blocked["entry_trigger"] == "HTF_VETO"


def test_all_eight_frames_and_complete_parts_are_required():
    evidence = evidence_by_timeframe()
    assert set(evidence) == set(REQUIRED_TIMEFRAMES)
    for timeframe in REQUIRED_TIMEFRAMES:
        assert set(PARTS).issubset(evidence[timeframe])
    group_of = {tf: g for g, tfs in TIMEFRAME_CORRELATION_GROUPS.items() for tf in tfs}
    assert TIMEFRAME_GROUP_WEIGHTS[group_of["1m"]] < TIMEFRAME_GROUP_WEIGHTS[group_of["4h"]]

    missing_frame = dict(evidence)
    missing_frame.pop("4h")
    assert not evaluate_mtf_central_strategy(missing_frame, aggregate_gate(), expected_symbol=SYMBOL)["approved"]

    missing_part = evidence_by_timeframe()
    del missing_part["2h"]["part9_orderflow"]
    assert not evaluate_mtf_central_strategy(missing_part, aggregate_gate(), expected_symbol=SYMBOL)["approved"]

    wrong_frame = evidence_by_timeframe()
    wrong_frame["3m"]["part2_zone"]["native_timeframe"] = "1m"
    assert not evaluate_mtf_central_strategy(wrong_frame, aggregate_gate(), expected_symbol=SYMBOL)["approved"]

    wrong_symbol = evidence_by_timeframe()
    wrong_symbol["4h"]["part6_trend"]["symbol"] = "ETHUSDT"
    assert not evaluate_mtf_central_strategy(wrong_symbol, aggregate_gate(), expected_symbol=SYMBOL)["approved"]


def test_part7_stale_incomplete_wrong_symbol_and_blocked_gates_fail_closed():
    evidence = evidence_by_timeframe()
    assert evaluate_mtf_central_strategy(evidence, aggregate_gate(), expected_symbol=SYMBOL)["approved"]

    stale = aggregate_gate()
    stale["timeframe_results"]["1h"]["data_status"] = "stale"
    assert not evaluate_mtf_central_strategy(evidence, stale, expected_symbol=SYMBOL)["approved"]

    wrong_symbol = aggregate_gate("ETHUSDT")
    assert not evaluate_mtf_central_strategy(evidence, wrong_symbol, expected_symbol=SYMBOL)["approved"]

    blocked = aggregate_gate()
    blocked["timeframe_results"]["15m"]["entry_blocked"] = True
    assert not evaluate_mtf_central_strategy(evidence, blocked, expected_symbol=SYMBOL)["approved"]

    incomplete = aggregate_gate()
    incomplete["timeframe_results"].pop("4h")
    assert not evaluate_mtf_central_strategy(evidence, incomplete, expected_symbol=SYMBOL)["approved"]

    bad_part7 = evidence_by_timeframe()
    bad_part7["5m"]["part7_volatility"]["data_status"] = "stale"
    assert not evaluate_mtf_central_strategy(bad_part7, aggregate_gate(), expected_symbol=SYMBOL)["approved"]

    assert not evaluate_mtf_central_strategy(evidence, aggregate_gate(), expected_symbol="ETHUSDT")["approved"]
    assert not evaluate_mtf_central_strategy(evidence, aggregate_gate(), expected_symbol=None)["approved"]


def test_part11_part12_consume_complete_native_evidence_and_slash8_support():
    from part11_FIXED import SignalFusionEngineGPU
    from part12_FIXED import ConfidenceEngineGPU

    evidence = evidence_by_timeframe()
    fusion = SignalFusionEngineGPU().analyze_multi_timeframe(evidence)
    confidence = ConfidenceEngineGPU().analyze_multi_timeframe(evidence)
    assert fusion["data_status"] == "valid"
    assert set(fusion["timeframe_coverage"]) == set(REQUIRED_TIMEFRAMES)
    assert set(fusion["timeframe_results"]) == set(REQUIRED_TIMEFRAMES)
    assert confidence["data_status"] == "valid"
    assert set(confidence["timeframe_coverage"]) == set(REQUIRED_TIMEFRAMES)
    assert set(confidence["timeframe_results"]) == set(REQUIRED_TIMEFRAMES)

    low_support = evidence_by_timeframe()
    for row in low_support.values():
        for name, item in row.items():
            if name != "part7_volatility":
                item["thought"] += "; MTF: 3/8 TFs agree"
    low_fusion = SignalFusionEngineGPU().analyze_multi_timeframe(low_support)
    low_confidence = ConfidenceEngineGPU().analyze_multi_timeframe(low_support)
    assert low_fusion["signal"] == 0
    assert low_confidence["confidence"] < confidence["confidence"]

    incomplete = dict(evidence)
    incomplete.pop("4h")
    assert SignalFusionEngineGPU().analyze_multi_timeframe(incomplete)["data_status"] == "error"
    assert ConfidenceEngineGPU().analyze_multi_timeframe(incomplete)["data_status"] == "error"


def test_plan_mode_and_stale_approval_remain_bound_to_mtf_setup():
    evidence = evidence_by_timeframe()
    decision = evaluate_mtf_central_strategy(evidence, aggregate_gate(), confidence=80, expected_symbol=SYMBOL)
    assert decision["approved"]
    plan = build_execution_plan(
        direction=decision["direction"], recommended_expiry=decision["trade_mode"],
        entry_price=100, stop_loss=99, take_profit=102, symbol=SYMBOL,
        snapshot_version="snapshot-mtf", confidence=80,
    )
    assert plan["trade_mode"] == decision["trade_mode"]
    approval = make_entry_approval(
        {**decision, "symbol": SYMBOL}, direction=decision["direction"], symbol=SYMBOL,
        exchange="delta", contract=SYMBOL, instrument_id=SYMBOL,
        market_type="unverified", snapshot_version="snapshot-mtf",
        analysis_timestamp=time.time() - 181, confidence=80, execution_plan=plan,
    )
    assert not validate_entry_approval(
        approval, direction="BUY", symbol=SYMBOL, exchange="delta",
        contract=SYMBOL, instrument_id=SYMBOL, market_type="unverified",
        snapshot_version="snapshot-mtf", confidence=80, now=time.time(),
        execution_plan=plan,
    )[0]
    assert not validate_execution_plan(
        plan, direction="BUY", symbol=SYMBOL, snapshot_version="snapshot-mtf",
        confidence=80, trade_mode="SCALP",
    )[0]
