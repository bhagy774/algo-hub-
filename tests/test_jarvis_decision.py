from jarvis_decision import DecisionContext
import pytest
from jarvis_decision import (
    normalize_confidence,
    confidence_text,
    canonical_direction,
    execution_direction,
    _opinion_direction,
    build_final_decision
)

def test_normalize_confidence():
    # None and default handling
    assert normalize_confidence(None) is None
    assert normalize_confidence(None, default=50) == 50
    assert normalize_confidence(True) is None
    assert normalize_confidence(False) is None

    # Real numbers
    assert normalize_confidence(0.85) == 85
    assert normalize_confidence(1.0) == 100
    assert normalize_confidence(0.0) == 0
    assert normalize_confidence(85) == 85
    assert normalize_confidence(100) == 100
    assert normalize_confidence(0) == 0

    # Out of bounds / small values
    assert normalize_confidence(1.5) == 2  # Not multiplied by 100, returns 2
    assert normalize_confidence(150) is None
    assert normalize_confidence(-5) is None

    # Strings: Fractions
    assert normalize_confidence("85/100") == 85
    assert normalize_confidence("8.5/10") == 85
    assert normalize_confidence("85/100 (Autonomy)") == 85
    assert normalize_confidence("1/0") is None  # Denominator <= 0
    assert normalize_confidence("1/-10") is None

    # Strings: Percentages
    assert normalize_confidence("85%") == 85
    assert normalize_confidence(" 85 % ") == 85

    # Strings: Numbers
    assert normalize_confidence("85") == 85
    assert normalize_confidence("0.85") == 85

    # NA / Unavailable strings
    assert normalize_confidence("N/A") is None
    assert normalize_confidence("MISSING", default=10) == 10

    # Invalid strings
    assert normalize_confidence("random string") is None

def test_confidence_text():
    assert confidence_text(85) == "85"
    assert confidence_text(0.85) == "85"
    assert confidence_text("N/A") == "N/A"
    assert confidence_text(None) == "N/A"
    assert confidence_text(None, default=50) == "50"

def test_canonical_direction():
    assert canonical_direction("BUY") == "BUY"
    assert canonical_direction("CALL") == "BUY"
    assert canonical_direction("LONG") == "BUY"
    assert canonical_direction("BULLISH") == "BUY"

    assert canonical_direction("SELL") == "SELL"
    assert canonical_direction("PUT") == "SELL"
    assert canonical_direction("SHORT") == "SELL"
    assert canonical_direction("BEARISH") == "SELL"

    assert canonical_direction("WAIT") == "WAIT"
    assert canonical_direction("PENDING") == "WAIT"
    assert canonical_direction("STANDBY") == "WAIT"

    assert canonical_direction("XYZ") == "NO_TRADE"
    assert canonical_direction("") == "NO_TRADE"
    assert canonical_direction(None) == "NO_TRADE"
    assert canonical_direction("long") == "BUY"
    assert canonical_direction("  buy  ") == "BUY"

def test_execution_direction():
    assert execution_direction("BUY") == "CALL"
    assert execution_direction("SELL") == "PUT"
    assert execution_direction("WAIT") is None
    assert execution_direction("NO_TRADE") is None
    assert execution_direction("XYZ") is None

def test_opinion_direction():
    assert _opinion_direction("BUY") == "BUY"
    assert _opinion_direction("SELL") == "SELL"
    assert _opinion_direction("WAIT") is None
    assert _opinion_direction("XYZ") is None

    assert _opinion_direction({"direction": "BUY"}) == "BUY"
    assert _opinion_direction({"signal": "SELL"}) == "SELL"
    assert _opinion_direction({"bias": "BUY"}) == "BUY"
    assert _opinion_direction({"verdict": "SELL"}) == "SELL"
    assert _opinion_direction({"other": "BUY"}) is None

def test_build_final_decision_basic_buy():
    signal = {"direction": "BUY", "confidence": 85, "entry_price": 100}
    decision = build_final_decision(signal, DecisionContext(symbol="BTC"))

    assert decision["direction"] == "BUY"
    assert decision["execution_direction"] == "CALL"
    assert decision["confidence"] == 85
    assert decision["confidence_display"] == "85"
    assert decision["symbol"] == "BTC"
    assert decision["entry"] == 100
    assert decision["status"] == "READY"
    assert decision["execution_allowed"] is True
    assert decision["reasons"] == []

def test_build_final_decision_no_confidence():
    signal = {"direction": "BUY"}
    decision = build_final_decision(signal, DecisionContext(symbol="BTC"))

    assert decision["direction"] == "NO_TRADE"
    assert decision["confidence"] is None
    assert decision["status"] == "WAIT"
    assert decision["execution_allowed"] is False
    assert "Confidence unavailable" in decision["reasons"]

def test_build_final_decision_conflict():
    signal = {"direction": "BUY", "confidence": 85}
    opinions = [{"direction": "BUY"}, {"direction": "SELL"}]
    decision = build_final_decision(signal, DecisionContext(symbol="BTC", opinions=opinions))

    assert decision["direction"] == "NO_TRADE"
    assert decision["status"] == "CONFLICT"
    assert decision["execution_allowed"] is False
    assert "Unresolved BUY/SELL opinions; consensus required" in decision["reasons"]

def test_build_final_decision_options_context():
    signal = {"direction": "BUY", "confidence": 85}

    # Require options but none provided
    decision1 = build_final_decision(signal, DecisionContext(symbol="BTC", require_options=True))
    assert decision1["direction"] == "NO_TRADE"
    assert decision1["execution_allowed"] is False
    assert "Required selected-asset options context unavailable or non-primary" in decision1["reasons"]

    # Require options and provided
    decision2 = build_final_decision(signal, DecisionContext(symbol="BTC", require_options=True, options_context={"available": True, "role": "asset_primary"}))
    assert decision2["direction"] == "BUY"
    assert decision2["execution_allowed"] is True
    assert decision2["options_context"] == {"available": True, "role": "asset_primary"}

def test_build_final_decision_gate_and_blocking():
    signal = {"direction": "BUY", "confidence": 85}

    decision = build_final_decision(signal, DecisionContext(symbol="BTC", gate_reason="Market Closed", blocking_reasons=["High volatility"]))

    assert decision["direction"] == "NO_TRADE"
    assert decision["execution_allowed"] is False
    assert "High volatility" in decision["reasons"]
    assert "Market Closed" in decision["reasons"]

def test_build_final_decision_plan_overrides():
    signal = {"direction": "BUY", "confidence": 85, "entry_price": 100, "take_profit_1": 110}
    plan = {"entry": 105, "tp1": 115}
    decision = build_final_decision(signal, DecisionContext(symbol="BTC", plan=plan))

    assert decision["entry"] == 105
    assert decision["tp1"] == 115
