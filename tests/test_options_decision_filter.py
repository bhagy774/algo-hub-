from jarvis_decision import DecisionContext
from jarvis_decision import build_final_decision


def macro(asset, bias):
    return {"source_asset": asset, "role": "macro_context", "available": True,
            "bias": bias,
            "validation": {"underlying": asset, "usable": True, "complete": True,
                          "fresh": True, "freshness_status": "fresh", "identity_valid": True}}


def test_unanimous_fresh_btc_eth_conflict_is_deterministic_altcoin_veto():
    out = build_final_decision({"direction": "BUY", "confidence_score": 80}, DecisionContext(symbol="SOLUSDT", options_context={"available": False, "role": "unavailable",
            "macro_contexts": [macro("BTC", "BEARISH"), macro("ETH", "BEARISH")]}))
    assert out["direction"] == "NO_TRADE"
    assert out["execution_allowed"] is False
    assert any("unanimously conflicts" in r for r in out["reasons"])


def test_missing_disagreeing_or_not_fresh_macro_data_is_not_weighted_or_scored():
    for macros in ([macro("BTC", "BEARISH")],
                   [macro("BTC", "BEARISH"), macro("ETH", "BULLISH")],
                   [macro("BTC", "BEARISH"), {**macro("ETH", "BEARISH"), "validation": {"usable": False}}],
                   [macro("BTC", "BEARISH"), {**macro("ETH", "BEARISH"), "validation": {**macro("ETH", "BEARISH")["validation"], "freshness_status": "stale"}}]):
        out = build_final_decision({"direction": "BUY", "confidence_score": 80}, DecisionContext(symbol="SOLUSDT", options_context={"macro_contexts": macros}))
        assert out["direction"] == "BUY"
        assert out["confidence"] == 80


def test_macro_consensus_does_not_replace_live_selected_asset_chain_requirement():
    out = build_final_decision({"direction": "BUY", "confidence_score": 80}, DecisionContext(symbol="SOLUSDT", require_options=True, options_context={"available": False, "role": "unavailable",
                         "macro_contexts": [macro("BTC", "BULLISH"), macro("ETH", "BULLISH")]}))
    assert out["direction"] == "NO_TRADE"
    assert any("Required selected-asset options context unavailable" in r for r in out["reasons"])
