"""A coin with unusable candle data is skipped and the default symbol is used."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_router_quote_fallback import _router


def test_rejected_symbol_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("JARVIS_DEFAULT_SYMBOL", "BTCUSDT")
    r = _router("OPUSDT", ["BTCUSD", "OPUSD"])
    assert r.select_crypto().symbol == "OPUSD"
    r.reject_symbol("OPUSD", "too young")
    route = r.select_crypto()
    assert route.status == "READY" and route.symbol == "BTCUSD"


def test_rejection_expires():
    r = _router("OPUSDT", ["BTCUSD", "OPUSD"])
    r.reject_symbol("OPUSD", "x", ttl_seconds=-1)
    assert r.select_crypto().symbol == "OPUSD"


def test_both_rejected_blocks(monkeypatch):
    monkeypatch.setenv("JARVIS_DEFAULT_SYMBOL", "BTCUSDT")
    r = _router("OPUSDT", ["BTCUSD", "OPUSD"])
    r.reject_symbol("OPUSD", "x"); r.reject_symbol("BTCUSD", "x")
    assert r.select_crypto().status == "BLOCKED"
