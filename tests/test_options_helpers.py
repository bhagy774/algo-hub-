from delta_api_wrapper import DeltaExchangeData


def test_nearest_expiry_and_contract_selection_require_usable_single_provider():
    client = DeltaExchangeData.__new__(DeltaExchangeData)
    chain = {
        "providers": {
            "Delta": {
                "validation": {"usable": True},
                "contracts": [
                    {"expiry": "2026-10-16", "strike": 100.0, "type": "call", "symbol": "BTC-16OCT26-100-C"},
                    {"expiry": "2026-10-09", "strike": 110.0, "type": "call", "symbol": "BTC-09OCT26-110-C"},
                    {"expiry": "2026-10-09", "strike": 95.0, "type": "put", "symbol": "BTC-09OCT26-95-P"},
                ],
            },
            # This contract must not be mixed into Delta's validated chain.
            "Binance Options": {
                "validation": {"usable": False},
                "contracts": [{"expiry": "2026-10-02", "strike": 100.0, "type": "call", "symbol": "BTC-02OCT26-100-C"}],
            },
        }
    }
    client.get_options_chain = lambda _underlying: chain

    assert client.get_nearest_expiries("BTC") == ["2026-10-09", "2026-10-16"]
    nearest = client.get_option_by_criteria("BTC", "CALL", 101.0)
    assert nearest["expiry"] == "2026-10-09"
    assert nearest["symbol"] == "BTC-09OCT26-110-C"
    assert client.get_option_by_criteria("BTC", "CALL", 101.0, "2026-10-16")["expiry"] == "2026-10-16"
    assert client.get_option_by_criteria("BTC", "INVALID", 101.0) is None


def test_unusable_option_provider_cannot_supply_expiry_or_contract():
    client = DeltaExchangeData.__new__(DeltaExchangeData)
    client.get_options_chain = lambda _underlying: {
        "providers": {"Delta": {"validation": {"usable": False}, "contracts": [
            {"expiry": "2026-10-09", "strike": 100.0, "type": "call"},
        ]}}
    }

    assert client.get_nearest_expiries("BTC") == []
    assert client.get_option_by_criteria("BTC", "CALL", 100.0) is None
