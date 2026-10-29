from jarvis_market_router import MarketRouter


class Scanner:
    def __init__(self, symbol="ETHUSDT", fail=False):
        self.symbol, self.fail, self.runs = symbol, fail, 0
    def run_now(self):
        self.runs += 1
        if self.fail:
            raise RuntimeError("network")
    def get_delta_symbol(self): return self.symbol
    def get_best_coin(self): return self.symbol.replace("USDT", "")
    def get_all_scores(self): return {self.get_best_coin(): {"total": 73.5}}


class Delta:
    def __init__(self, products=("BTCUSDT", "ETHUSDT"), options=True):
        self.products, self.options = list(products), options
    def get_available_symbols(self): return self.products
    def get_options_chain(self, base): return {"calls": [{}]} if self.options else {}


def test_selected_symbol_is_verified_and_keeps_its_own_context(monkeypatch):
    monkeypatch.setenv("JARVIS_MULTI_MARKET", "1")
    router = MarketRouter(Scanner("ETHUSDT"), Delta())
    route = router.select_crypto()
    assert route.status == "READY"
    assert route.symbol == "ETHUSDT"
    assert route.base_asset == "ETH"
    assert route.options_available is True


def test_unavailable_scanner_symbol_blocks_instead_of_falling_back_to_btc(monkeypatch):
    monkeypatch.setenv("JARVIS_MULTI_MARKET", "1")
    route = MarketRouter(Scanner("SOLUSDT"), Delta()).select_crypto()
    assert route.status == "BLOCKED"
    assert "not tradeable" in route.reason
    assert route.symbol == ""


def test_open_position_locks_existing_market(monkeypatch):
    monkeypatch.setenv("JARVIS_MULTI_MARKET", "1")
    scanner = Scanner("ETHUSDT")
    router = MarketRouter(scanner, Delta(("ETHUSDT", "SOLUSDT")))
    first = router.select_crypto()
    scanner.symbol = "SOLUSDT"
    locked = router.select_crypto(has_open_position=True)
    assert locked.symbol == "ETHUSDT"
    assert scanner.runs == 1


def test_scanner_failure_blocks_execution(monkeypatch):
    monkeypatch.setenv("JARVIS_MULTI_MARKET", "1")
    route = MarketRouter(Scanner(fail=True), Delta()).select_crypto()
    assert route.status == "BLOCKED"
    assert "scan failed" in route.reason.lower()


def test_india_is_explicit_data_only_even_when_authenticated(monkeypatch):
    class Upstox:
        _enabled = True
        def is_active(self): return True
    route = MarketRouter().india_status(Upstox())
    assert route.status == "DATA_ONLY"
    assert route.execution_venue == "none"
