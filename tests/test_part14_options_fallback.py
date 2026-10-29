import pytest
pd = pytest.importorskip("pandas")

from jarvis_FIXED import Part14OptionsChain


class FakeOptionsProvider:
    def __init__(self, response=None, raises=False):
        self.response = response or {}
        self.raises = raises
        self.calls = []

    def get_institutional_bias(self, query):
        self.calls.append(query)
        if self.raises:
            raise RuntimeError("provider unavailable")
        return dict(self.response)


def _usable(bias="BULLISH"):
    return {
        "available": True,
        "bias": bias,
        "score": 1,
        "pcr": 1.1,
        "reasons": ["fixture"],
        "raw_data": {"options_validation": {"usable": True}},
    }


def test_btc_deribit_is_tried_when_delta_options_are_unavailable():
    delta = FakeOptionsProvider({"available": False, "bias": "NEUTRAL"})
    deribit = FakeOptionsProvider(_usable())
    part = Part14OptionsChain(delta_client=delta, asset="BTC")
    part.deribit = deribit

    result = part.analyze(pd.DataFrame({"close": [100.0]}), context={"symbol": "BTCUSDT"})

    assert result["telemetry"]["available"] is True
    assert result["telemetry"]["exchange"] == "Deribit"
    assert result["signal"] == 1
    assert delta.calls == ["BTC"]
    assert deribit.calls == [100.0]


def test_deribit_never_substitutes_btc_options_for_an_altcoin():
    delta = FakeOptionsProvider({"available": False, "bias": "NEUTRAL"})
    part = Part14OptionsChain(delta_client=delta, asset="ETH")
    part.deribit = FakeOptionsProvider(_usable())

    result = part.analyze(pd.DataFrame({"close": [100.0]}), context={"symbol": "ETHUSDT"})

    assert result["telemetry"]["available"] is False
    assert result["telemetry"]["exchange"] == "None"
    assert result["signal"] == 0
    assert delta.calls == ["ETH"]
    assert part.deribit.calls == []
