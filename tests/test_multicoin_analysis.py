from __future__ import annotations

import ast
import concurrent.futures
import time
from pathlib import Path

import pytest

from native_candle_fixture import native_times, TIMEFRAME_HISTORY_CANDLES
from direct_candle_cache import (
    CandleDataError,
    DirectCandleCache,
    FETCH_CANDLES,
    LIVE_TIMEFRAMES,
)
from jarvis_multicoin_analysis import MultiCoinPart7Shadow, multicoin_analysis_enabled

STEP = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
BASE = {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}


class FakeClock:
    def __init__(self, value=None):
        self.value = float(value if value is not None else time.time())

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class FakeClient:
    def __init__(self, clock, *, delay=0.0, wrong_symbol=None):
        self.clock = clock
        self.delay = delay
        self.wrong_symbol = wrong_symbol
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = __import__("threading").Lock()

    def get_historical_candles_with_metadata(self, symbol, resolution, limit):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append((symbol, resolution, limit))
            now = self.clock()
            base = BASE.get(symbol, 100.0)
            rows = []
            for i, ts in enumerate(native_times(resolution, now, limit)):
                close = base + i * base * 0.00002
                rows.append({"time": ts, "open": close - 0.01,
                             "high": close + 0.10, "low": close - 0.10,
                             "close": close, "volume": 10.0 + (i % 7)})
            return {"candles": rows, "source": "synthetic-binance-spot",
                    "symbol": self.wrong_symbol or symbol}
        finally:
            with self.lock:
                self.active -= 1


def make_cache(*, workers=1, delay=0.0, wrong_symbol=None, clock=None):
    clock = clock or FakeClock()
    client = FakeClient(clock, delay=delay, wrong_symbol=wrong_symbol)
    cache = DirectCandleCache(client, clock=clock, max_concurrent_fetches=workers)
    return cache, client, clock


def drain(shadow, selected="BTCUSDT", snapshot=None, attempts=3000):
    state = shadow.poll(selected_symbol=selected, selected_snapshot=snapshot)
    for _ in range(attempts):
        if state.get("inflight_count", 0) == 0 and len(state.get("results", {})) >= 3:
            return state
        time.sleep(0.01)
        state = shadow.poll(selected_symbol=selected, selected_snapshot=snapshot)
    raise AssertionError(f"shadow tasks did not drain: {state}")


def test_same_symbol_market_contract_snapshots_do_not_overwrite_each_other():
    cache, _, _ = make_cache()
    spot = cache.refresh("ETHUSDT", venue="binance", market_type="spot", instrument_id="ETHUSDT")
    perpetual = cache.refresh("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="71234")
    assert cache.get_snapshot("ETHUSDT", venue="binance", market_type="spot", instrument_id="ETHUSDT") is spot
    assert cache.get_snapshot("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="71234") is perpetual
    assert spot is not perpetual
    assert spot.identity == ("binance", "spot", "ETHUSDT", "ETHUSDT")
    assert perpetual.identity == ("delta", "perpetual", "71234", "ETHUSDT")
    assert len(spot.frames["1m"].closed) == TIMEFRAME_HISTORY_CANDLES["1m"]
    assert spot.frames["1m"].venue == "binance"
    assert perpetual.frames["1m"].venue == "delta"
    assert spot.frames["1m"].instrument_id != perpetual.frames["1m"].instrument_id


def test_concurrent_refreshes_are_per_identity_but_network_calls_stay_bounded():
    cache, client, _ = make_cache(workers=2, delay=0.002)
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        snapshots = list(pool.map(cache.refresh, symbols))
    assert {snapshot.symbol for snapshot in snapshots} == set(symbols)
    assert client.max_active == 2
    assert all(len(snapshot.frames) == len(LIVE_TIMEFRAMES) for snapshot in snapshots)


def test_bad_instrument_is_rejected_without_caching_candidate():
    cache, _, _ = make_cache(wrong_symbol="BTCUSDT")
    with pytest.raises(CandleDataError, match="instrument mismatch"):
        cache.refresh("ETHUSDT", venue="delta", market_type="unverified", instrument_id="ETHUSDT")
    assert cache.get_snapshot("ETHUSDT", venue="delta", market_type="unverified", instrument_id="ETHUSDT") is None


def test_actual_part7_shadow_runs_three_symbols_all_native_timeframes_and_is_read_only():
    cache, _, _ = make_cache(workers=1)
    selected = cache.refresh(
        "BTCUSDT", venue="delta", market_type="unverified", instrument_id="BTCUSDT"
    )
    candidates = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    shadow = MultiCoinPart7Shadow(cache, lambda: candidates, enabled=True, max_workers=2)
    try:
        state = drain(shadow, selected="BTCUSDT", snapshot=selected)
        assert state["candidate_count"] == 3
        assert state["inflight_count"] == 0
        assert set(state["results"]) == set(candidates)
        for symbol, result in state["results"].items():
            assert result["symbol"] == symbol
            assert result["coverage"] == ["Part7"]
            assert set(result["part7_by_timeframe"]) == set(LIVE_TIMEFRAMES)
            assert set(result["provider_by_timeframe"]) == set(LIVE_TIMEFRAMES)
            assert result["analysis_only"] is True
            assert result["decision_authority"] == "none"
            assert result["execution_eligible"] is False
            assert len(result["snapshot_version"]) == 20
            assert all(out["symbol"] == symbol for out in result["part7_by_timeframe"].values())
    finally:
        shadow.close()


def test_serial_and_parallel_shadow_outputs_are_deterministically_equivalent():
    cache, _, _ = make_cache(workers=1)
    selected = cache.refresh(
        "BTCUSDT", venue="delta", market_type="unverified", instrument_id="BTCUSDT"
    )
    candidates = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    outputs = []
    for workers in (1, 2):
        shadow = MultiCoinPart7Shadow(cache, lambda: candidates, enabled=True, max_workers=workers)
        try:
            outputs.append(drain(shadow, selected="BTCUSDT", snapshot=selected)["results"])
        finally:
            shadow.close()
    for symbol in candidates:
        left, right = outputs[0][symbol], outputs[1][symbol]
        assert left["snapshot_version"] == right["snapshot_version"]
        assert left["status"] == right["status"]
        for timeframe in LIVE_TIMEFRAMES:
            a = left["part7_by_timeframe"][timeframe]
            b = right["part7_by_timeframe"][timeframe]
            assert (a["signal"], a["status"], a["data_status"], a["telemetry"]) == (
                b["signal"], b["status"], b["data_status"], b["telemetry"])


def test_jobs_and_symbol_universe_are_bounded_without_queued_backlog():
    cache, _, _ = make_cache(workers=1, delay=0.02)
    candidates = tuple(f"COIN{i}USDT" for i in range(12))
    shadow = MultiCoinPart7Shadow(cache, lambda: candidates, enabled=True,
                                  max_workers=2, max_candidates=8)
    try:
        state = shadow.poll()
        assert state["candidate_count"] == 8
        assert state["inflight_count"] == 2
        assert len(shadow._inflight) <= 2
        assert shadow._executor._work_queue.qsize() == 0
    finally:
        shadow.close()


def test_stale_results_are_marked_non_executable():
    cache, _, clock = make_cache()
    selected = cache.refresh(
        "BTCUSDT", venue="delta", market_type="unverified", instrument_id="BTCUSDT"
    )
    shadow = MultiCoinPart7Shadow(cache, lambda: ("BTCUSDT",), enabled=True,
                                  max_workers=1, result_max_age_seconds=2.0, clock=clock)
    try:
        state = shadow.poll(selected_symbol="BTCUSDT", selected_snapshot=selected)
        for _ in range(1500):
            time.sleep(0.01)
            state = shadow.poll(selected_symbol="BTCUSDT", selected_snapshot=selected)
            if state["inflight_count"] == 0 and "BTCUSDT" in state["results"]:
                break
        assert state["results"]["BTCUSDT"]["freshness_status"] == "FRESH"
        clock.advance(3)
        stale = shadow.poll(selected_symbol="BTCUSDT", selected_snapshot=selected)["results"]["BTCUSDT"]
        assert stale["status"] == "STALE"
        assert stale["freshness_status"] == "STALE"
        assert stale["execution_eligible"] is False
    finally:
        shadow.close()


def test_feature_flag_defaults_off_and_runtime_wiring_is_shadow_only():
    assert multicoin_analysis_enabled("0") is False
    assert multicoin_analysis_enabled("false") is False
    assert multicoin_analysis_enabled("1") is True
    source = Path(__file__).parents[1].joinpath("jarvis_FIXED.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    live = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiveTradingEngine")
    methods = {node.name: node for node in live.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    init_text = ast.unparse(methods["__init__"])
    loop_text = ast.unparse(methods["start_live_trading"])
    assert "multicoin_analysis_enabled()" in init_text
    assert "MultiCoinPipeline" in init_text
    assert "self.multicoin_pipeline.poll" in loop_text
    assert "self.jarvis.analyze_trade_setup" in loop_text
    # Full background pipeline results remain diagnostic-only; they are not
    # passed into selected-route analysis or the decision/order builder.
    assert "multicoin_pipeline_status" in loop_text
    pipeline_handoffs = []
    for node in ast.walk(methods["start_live_trading"]):
        if isinstance(node, ast.Call) and any("multicoin_pipeline_status" in ast.unparse(arg) for arg in node.args):
            if isinstance(node.func, ast.Attribute):
                pipeline_handoffs.append(node.func.attr)
    assert sorted(pipeline_handoffs) == ["_consume_multicoin_delta_results", "_consume_multicoin_paper_results"]
    assert "self._consume_multicoin_paper_results(self.multicoin_pipeline_status)" in loop_text
    assert "self._consume_multicoin_delta_results(self.multicoin_pipeline_status)" in loop_text


def test_existing_router_position_lock_remains_immutable():
    from jarvis_market_router import MarketRouter

    class Scanner:
        def __init__(self): self.calls = 0
        def run_now(self): self.calls += 1
        def get_delta_symbol(self): return "ETHUSDT"
        def get_best_coin(self): return "ETH"
        def get_all_scores(self): return {"ETH": {"total": 90}}

    class Delta:
        def __init__(self): self.lookups = 0
        def get_available_symbols(self): self.lookups += 1; return ["ETHUSDT"]
        def get_options_chain(self, base): return {}

    scanner, delta = Scanner(), Delta()
    router = MarketRouter(scanner=scanner, delta_client=delta)
    first = router.select_crypto()
    assert first.symbol == "ETHUSDT"
    locked = router.select_crypto(has_open_position=True)
    assert locked is first
    assert scanner.calls == 1
    assert delta.lookups == 1
