"""Router must map scanner USDT symbols to Delta India USD-quoted perps."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_market_router import MarketRouter as JarvisMarketRouter

class FakeScanner:
    def __init__(self, sym): self._sym = sym
    def run_now(self): pass
    def get_delta_symbol(self): return self._sym
    def get_best_coin(self): return "SOL"
    def get_all_scores(self): return {"SOL": {"total": 9.0}}

class FakeDelta:
    def __init__(self, symbols): self._s = symbols
    def get_available_symbols(self): return self._s
    def get_options_chain(self, base): return []

def _router(sym, available):
    return JarvisMarketRouter(scanner=FakeScanner(sym), delta_client=FakeDelta(available))

def test_usdt_maps_to_usd_perp():
    r = _router("SOLUSDT", ["BTCUSD", "ETHUSD", "SOLUSD"])
    route = r.select_crypto()
    assert route.status == "READY", route
    assert route.symbol == "SOLUSD"
    assert route.base_asset == "SOL"

def test_exact_match_still_works():
    r = _router("BTCUSDT", ["BTCUSDT"])
    assert r.select_crypto().symbol == "BTCUSDT"

def test_truly_untradeable_still_blocked():
    r = _router("FAKECOINUSDT", ["BTCUSD"])
    route = r.select_crypto()
    assert route.status == "BLOCKED"

if __name__ == "__main__":
    test_usdt_maps_to_usd_perp(); test_exact_match_still_works(); test_truly_untradeable_still_blocked()
    print("3/3 router quote-fallback tests PASS")
