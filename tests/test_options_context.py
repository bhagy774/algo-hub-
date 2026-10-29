from jarvis_options_context import apply_options_confirmation, resolve_options_context


def valid_chain(asset, bias="NEUTRAL", score=0, pcr=0.8):
    return {
        "bias": bias, "score": score, "pcr": pcr, "reasons": ["validated chain"],
        "raw_data": {
            "source_provider": "Delta", "options_validation": {
                "underlying": asset, "usable": True, "complete": True,
                "identity_valid": True, "fresh": True, "freshness_status": "fresh",
            },
            "provider_metrics": {"open_interest_complete": True},
        },
    }


class Delta:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
    def get_institutional_bias(self, asset):
        self.calls.append(asset)
        return self.responses.get(asset, {"bias": "NEUTRAL", "score": 0})


def intel(bias, score=3):
    return valid_chain("SOL", bias, score, 0.8)


def test_selected_asset_chain_is_primary_and_macro_is_separate():
    d = Delta({"ETH": valid_chain("ETH", "BULLISH", 3), "BTC": valid_chain("BTC", "BEARISH", 3)})
    result = resolve_options_context(d, "ETHUSDT")
    assert result.role == "asset_primary"
    assert result.source_asset == "ETH"
    assert result.bias == "BULLISH"
    assert [m["source_asset"] for m in result.macro_contexts] == ["BTC"]
    assert d.calls == ["ETH", "BTC"]


def test_missing_selected_chain_does_not_promote_btc_or_eth_to_primary():
    d = Delta({"SOL": {"bias": "NEUTRAL", "score": 0}, "BTC": valid_chain("BTC", "BULLISH", 3), "ETH": valid_chain("ETH", "BULLISH", 2)})
    result = resolve_options_context(d, "SOLUSDT")
    assert result.role == "unavailable" and result.available is False
    assert result.selected_asset == "SOL" and result.source_asset == ""
    assert {m["source_asset"] for m in result.macro_contexts} == {"BTC", "ETH"}
    assert d.calls == ["SOL", "BTC", "ETH"]
    assert apply_options_confirmation("CALL", 70, result)[0] == 70


def test_real_delta_no_data_shape_is_not_available():
    no_data = {"bias": "NEUTRAL", "score": 0, "reasons": ["No Data"],
               "raw_data": {"options_validation": {"usable": False, "complete": False}}}
    result = resolve_options_context(Delta({"BTC": no_data, "ETH": no_data}), "BTCUSDT")
    assert result.available is False
    assert result.role == "unavailable"
    assert result.macro_contexts == []


def test_only_validated_selected_chain_can_adjust_legacy_confidence():
    own = resolve_options_context(Delta({"SOL": valid_chain("SOL", "BULLISH", 3)}), "SOLUSDT")
    assert apply_options_confirmation("CALL", 70, own)[0] == 74
    assert apply_options_confirmation("PUT", 70, own)[0] == 64


def test_macro_is_never_selected_asset_contract_or_directional_score():
    d = Delta({"ETH": {"bias": "NEUTRAL", "score": 0}, "BTC": valid_chain("BTC", "BEARISH", 3), "SOL": {"bias": "NEUTRAL", "score": 0}})
    result = resolve_options_context(d, "ETHUSDT")
    assert result.role == "unavailable"
    assert result.source_asset == ""
    assert [m["source_asset"] for m in result.macro_contexts] == ["BTC"]
    adjusted, _ = apply_options_confirmation("CALL", 70, result)
    assert adjusted == 70
