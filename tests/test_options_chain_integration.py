from datetime import datetime, timezone

from delta_api_wrapper import DeltaExchangeData
from binance_options_client import BinanceOptionsClient

AS_OF = datetime.now(timezone.utc)


def delta_row(symbol, kind, strike):
    return {"symbol": symbol, "underlying_asset_symbol": "BTC",
            "contract_type": "call_options" if kind == "call" else "put_options",
            "strike_price": strike, "expiry": "2026-10-30",
            "timestamp": AS_OF.isoformat(), "oi": 150, "volume": 5,
            "best_bid": 4, "best_ask": 5, "implied_volatility": 0.5,
            "greeks": {"delta": 0.3, "gamma": 0.01, "theta": -0.1, "vega": 0.2, "rho": 0.01}}


def client_with_responses(responses):
    client = DeltaExchangeData.__new__(DeltaExchangeData)
    client._options_chain_cache = {}
    client._options_snapshot_last_capture = {}
    client._binance_options = None
    calls = []
    def fake_request(method, endpoint, params):
        calls.append(dict(params))
        return responses[len(calls) - 1]
    client._request = fake_request
    return client, calls


def test_delta_cursor_pagination_reaches_terminal_marker_and_keeps_all_rows():
    page1 = {"success": True, "data": {"result": [delta_row("BTC-261030-90000-C", "call", 90000)], "meta": {"after": "cursor-2"}}}
    page2 = {"success": True, "data": {"result": [delta_row("BTC-261030-90000-P", "put", 90000)], "meta": {"after": None}}}
    client, calls = client_with_responses([page1, page2])
    chain = client.get_options_chain("BTCUSDT")
    assert len(calls) == 2
    assert calls[1]["after"] == "cursor-2"
    delta = chain["providers"]["Delta"]
    assert delta["validation"]["usable"]
    assert delta["validation"]["pagination"]["pages_received"] == 2
    assert {c["type"] for c in delta["contracts"]} == {"call", "put"}


def test_delta_missing_page_marker_provider_error_and_cursor_loop_fail_closed():
    rows = [delta_row("BTC-261030-90000-C", "call", 90000)]
    client, _ = client_with_responses([{"success": True, "data": {"result": rows}}])
    assert not client.get_options_chain("BTC")["providers"]["Delta"]["validation"]["usable"]
    client, _ = client_with_responses([{"success": False, "error": "offline"}])
    chain = client.get_options_chain("BTC")
    assert chain["providers"]["Delta"]["status"] == "provider_error"
    client, _ = client_with_responses([
        {"success": True, "data": {"result": rows, "meta": {"after": "same"}}},
        {"success": True, "data": {"result": rows, "meta": {"after": "same"}}},
    ])
    chain = client.get_options_chain("BTC")
    assert not chain["providers"]["Delta"]["validation"]["usable"]


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): return None
    def json(self): return self.payload


class FakeBinanceSession:
    def __init__(self, now_ms):
        self.now_ms = now_ms
        self.symbols = [
            {"symbol": "BTC-261030-90000-C", "underlying": "BTC", "expiryDate": 1793318400000, "strikePrice": "90000", "side": "CALL"},
            {"symbol": "BTC-261030-90000-P", "underlying": "BTC", "expiryDate": 1793318400000, "strikePrice": "90000", "side": "PUT"},
        ]
    def get(self, url, timeout):
        if url.endswith("/eapi/v1/exchangeInfo"):
            return FakeResponse({"serverTime": self.now_ms, "optionSymbols": self.symbols})
        if url.endswith("/eapi/v1/mark"):
            return FakeResponse([{"symbol": x["symbol"], "time": self.now_ms, "markIV": 0.5,
                                 "delta": 0.25 if x["side"] == "CALL" else -0.25,
                                 "gamma": 0.01, "theta": -0.1, "vega": 0.2, "rho": 0.01}
                                for x in self.symbols])
        if url.endswith("/eapi/v1/ticker"):
            return FakeResponse([{"symbol": x["symbol"], "closeTime": self.now_ms,
                                 "bidPrice": 4, "askPrice": 5, "openInterest": 150, "volume": 5}
                                for x in self.symbols])
        raise AssertionError("unexpected provider endpoint")


def test_binance_manifest_read_only_provider_maps_all_symbols_and_greeks():
    now_ms = int(AS_OF.timestamp() * 1000)
    session = FakeBinanceSession(now_ms)
    # Use a near future date from the fixed AS_OF for exact point-in-time fixture.
    for item in session.symbols:
        item["expiryDate"] = int(datetime(2026, 10, 30, tzinfo=timezone.utc).timestamp() * 1000)
    chain = BinanceOptionsClient(session=session).get_options_chain("BTC")
    assert chain["validation"]["usable"]
    assert chain["validation"]["pagination"]["listed_symbol_count"] == 2
    assert len(chain["contracts"]) == 2
    assert all(c["greeks_source"] == "exchange_observed" for c in chain["contracts"])
    assert chain["coverage"]["greek_observed_count"]["rho"] == 2


def test_binance_api_error_is_not_an_empty_complete_chain():
    class Broken:
        def get(self, *args, **kwargs): raise TimeoutError("fixture")
    chain = BinanceOptionsClient(session=Broken()).get_options_chain("ETH")
    assert chain["status"] == "provider_error"
    assert chain["validation"]["usable"] is False
