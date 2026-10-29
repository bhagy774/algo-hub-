from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from pathlib import Path

import pytest

from native_candle_fixture import native_times, TIMEFRAME_HISTORY_CANDLES
from direct_candle_cache import (
    CandleDataError, DirectCandleCache, FETCH_CANDLES, LIVE_TIMEFRAMES,
)
from jarvis_multicoin_admission import PortfolioAdmission
from jarvis_multicoin_pipeline import (
    BinanceSpotCandleClient, DeltaNativeCandleClient, InstrumentKey, MultiCoinPipeline,
)

STEP = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


class Clock:
    def __init__(self, value=1_800_000_000):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class SyntheticDelta:
    """Deterministic provider fixture; makes no socket/API calls."""
    def __init__(self, clock, *, source="delta"):
        self.clock = clock
        self.source = source
        self.calls = []
        self.revise_closed = False
        self.gap = False
        self._lock = threading.Lock()

    def get_historical_candles_with_metadata(self, *, symbol, resolution, limit):
        with self._lock:
            self.calls.append((symbol, resolution, limit))
        now = self.clock()
        rows = []
        for ts in native_times(resolution, now, limit):
            # Stable price for a timestamp across refreshes, including the
            # transition of the old forming candle to its final closed value.
            close = 100.0 + (ts % 1_000_000) * 1e-6 + (abs(hash(symbol)) % 1000) * 1e-4
            row = {"time": ts, "open": close - .01, "high": close + .10,
                   "low": close - .10, "close": close, "volume": 10.0}
            rows.append(row)
        if self.revise_closed and resolution == "1m" and len(rows) >= 2:
            rows[-2] = dict(rows[-2], close=rows[-2]["close"] + .02)
        if self.gap and resolution == "1m" and len(rows) >= 3:
            rows.pop(-2)
        return {"candles": rows, "source": self.source, "symbol": symbol}


def make_cache(*, clock=None, source="delta"):
    clock = clock or Clock()
    client = SyntheticDelta(clock, source=source)
    cache = DirectCandleCache(client, clock=clock, ttl_seconds=0,
                              max_concurrent_fetches=2)
    return cache, client, clock


def full_stub(key, _snapshot):
    return {
        "parts_by_timeframe": {
            tf: {f"part{i}": {"signal": 0} for i in range(1, 11)}
            for tf in LIVE_TIMEFRAMES
        },
        "once_per_symbol_parts": {"part11": {"signal": 0}, "part12": {"confidence": 0}},
    }


def test_incremental_rollover_keeps_500_closed_and_fetches_only_recent_delta():
    cache, client, clock = make_cache()
    first = cache.refresh("BTCUSDT", venue="delta", market_type="perpetual", instrument_id="p-1")
    assert all(len(frame.closed) == TIMEFRAME_HISTORY_CANDLES[tf] for tf, frame in first.frames.items())
    assert all(limit == TIMEFRAME_HISTORY_CANDLES[tf] + 1 for _, tf, limit in client.calls)
    clock.advance(60)
    second = cache.refresh("BTCUSDT", venue="delta", market_type="perpetual", instrument_id="p-1")
    assert second is not first
    assert all(len(frame.closed) == TIMEFRAME_HISTORY_CANDLES[tf] for tf, frame in second.frames.items())
    assert all(limit <= 63 for _, _, limit in client.calls[len(LIVE_TIMEFRAMES):])
    assert all(tf != "1s" for _, tf, _ in client.calls)
    for tf, frame in second.frames.items():
        closed_seconds = [int(stamp.timestamp()) for stamp in frame.closed.index]
        from binance_timeframes import next_candle_open, candle_open_time
        assert all(next_candle_open(tf, left) == right for left, right in zip(closed_seconds, closed_seconds[1:]))
        assert frame.current["time"] == candle_open_time(tf, clock())


def test_incremental_closed_revision_or_gap_fails_closed_and_preserves_last_snapshot():
    cache, client, clock = make_cache()
    original = cache.refresh("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="p-2")
    client.revise_closed = True
    with pytest.raises(CandleDataError, match="previously closed candle revision"):
        cache.refresh("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="p-2")
    assert cache.get_snapshot("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="p-2") is original

    client.revise_closed = False
    clock.advance(120)
    client.gap = True
    with pytest.raises(CandleDataError):
        cache.refresh("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="p-2")
    assert cache.get_snapshot("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="p-2") is original


def test_delta_candle_adapter_never_calls_cross_venue_fallback():
    class FakeDelta:
        def __init__(self): self.calls = 0
        def get_delta_native_candles_with_metadata(self, *, symbol, resolution, limit):
            self.calls += 1
            return {"candles": [], "source": "delta", "symbol": symbol}
        def get_historical_candles_with_metadata(self, **kwargs):
            raise AssertionError("generic fallback path must not be called")

    client = FakeDelta()
    result = DeltaNativeCandleClient(client, min_request_interval_seconds=0).get_historical_candles_with_metadata(
        symbol="ETHUSDT", resolution="1m", limit=3)
    assert result["source"] == "delta"
    assert client.calls == 1


def test_binance_candle_adapter_rejects_bybit_fallback_and_preserves_venue_identity():
    class FakeBinance:
        last_candle_source = "bybit"
        def get_historical_candles_with_metadata(self, *, symbol, resolution, limit):
            return {"candles": [{"time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "source": "binance_spot", "market_semantics": "spot"}],
                    "source": self.last_candle_source, "market_type": "spot", "symbol": symbol}

    adapter = BinanceSpotCandleClient(FakeBinance(), min_request_interval_seconds=0)
    with pytest.raises(CandleDataError, match="source/market/symbol verification failed"):
        adapter.get_historical_candles_with_metadata(symbol="ETHUSDT", resolution="1m", limit=501)
    fake = FakeBinance()
    fake.last_candle_source = "binance"
    result = BinanceSpotCandleClient(fake, min_request_interval_seconds=0).get_historical_candles_with_metadata(
        symbol="ETHUSDT", resolution="1m", limit=501)
    assert result["source"] == "binance"
    assert result["symbol"] == "ETHUSDT"
    assert InstrumentKey.from_record({"venue": "binance", "market_type": "spot",
                                      "instrument_id": "ETHUSDT", "symbol": "ETHUSDT"}).venue == "binance"


def test_cache_lru_reclaims_completed_identities_for_a_dynamic_universe():
    clock = Clock()
    client = SyntheticDelta(clock)
    cache = DirectCandleCache(client, clock=clock, ttl_seconds=0, max_cached_identities=2)
    btc = cache.refresh("BTCUSDT", venue="delta", market_type="perpetual", instrument_id="b")
    cache.refresh("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="e")
    cache.refresh("SOLUSDT", venue="delta", market_type="perpetual", instrument_id="s")
    assert cache.get_snapshot("BTCUSDT", venue="delta", market_type="perpetual", instrument_id="b") is None
    assert cache.get_snapshot("ETHUSDT", venue="delta", market_type="perpetual", instrument_id="e") is not None
    assert cache.get_snapshot("SOLUSDT", venue="delta", market_type="perpetual", instrument_id="s") is not None
    assert len(cache._identity_locks) == 2
    assert cache.refresh("BTCUSDT", venue="delta", market_type="perpetual", instrument_id="b").symbol == btc.symbol


def test_full_identity_venue_source_and_ambiguous_symbol_are_rejected():
    cache, _, clock = make_cache(source="binance")
    pipeline = MultiCoinPipeline(cache, lambda: (), full_stub, enabled=False, clock=clock)
    key = InstrumentKey("delta", "perpetual", "delta-product-7", "ETHUSDT")
    blocked = pipeline._run(key)
    assert blocked["status"] == "BLOCKED"
    assert blocked["execution_eligible"] is False

    pipeline.instrument_supplier = lambda: [
        {"venue": "delta", "market_type": "perpetual", "instrument_id": "A", "symbol": "ETHUSDT"},
        {"venue": "delta", "market_type": "spot", "instrument_id": "B", "symbol": "ETHUSDT"},
        {"venue": "delta", "market_type": "perpetual", "instrument_id": "C", "symbol": "SOLUSDT"},
    ]
    pipeline._refresh_candidates(0)
    assert pipeline._ambiguous_symbol_count == 1
    assert [candidate.symbol for candidate in pipeline._all_candidates] == ["SOLUSDT"]
    pipeline.close()


def test_pipeline_stale_expiry_and_over_age_analysis_are_fail_closed():
    cache, _, clock = make_cache()
    key = InstrumentKey("delta", "perpetual", "p-stale", "BTCUSDT")
    pipeline = MultiCoinPipeline(cache, lambda: (), full_stub, enabled=True,
                                 result_max_age_seconds=10, clock=clock)
    try:
        result = pipeline._run(key)
        assert result["status"] == "COMPLETE"
        pipeline._results[key] = result
        clock.advance(11)
        view = pipeline.poll()["results"]["|".join(key.as_tuple())]
        assert view["status"] == "STALE"
        assert view["freshness_status"] == "STALE"
        assert view["execution_eligible"] is False

        def slow(candidate, snapshot):
            clock.advance(11)
            return full_stub(candidate, snapshot)
        slow_pipeline = MultiCoinPipeline(cache, lambda: (), slow, enabled=False,
                                          result_max_age_seconds=10, clock=clock)
        too_old = slow_pipeline._run(key)
        assert too_old["status"] == "BLOCKED"
        assert too_old["execution_eligible"] is False
        slow_pipeline.close()
    finally:
        pipeline.close()


def test_failed_analysis_worker_can_recover_on_next_attempt_without_authority():
    cache, _, _ = make_cache()
    key = InstrumentKey("delta", "perpetual", "eth-2", "ETHUSDT")
    calls = 0

    def flaky(candidate, snapshot):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failure")
        return full_stub(candidate, snapshot)

    pipeline = MultiCoinPipeline(cache, lambda: (), flaky, enabled=False, clock=cache.clock)
    first = pipeline._run(key)
    second = pipeline._run(key)
    assert first["status"] == "BLOCKED"
    assert second["status"] == "COMPLETE"
    assert second["analysis_only"] is True
    assert second["decision_authority"] == "none"
    assert second["execution_eligible"] is False
    pipeline.close()


def test_scheduler_has_bounded_workers_and_round_robin_universe_coverage():
    cache, _, _ = make_cache()
    started = []
    active = 0
    max_active = 0
    lock = threading.Lock()
    candidates = [
        {"venue": "delta", "market_type": "perpetual", "instrument_id": f"p-{i}",
         "symbol": f"COIN{i}USDT"} for i in range(5)
    ]

    def analyzer(key, snapshot):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            started.append(key.symbol)
        time.sleep(.08)
        with lock:
            active -= 1
        return full_stub(key, snapshot)

    pipeline = MultiCoinPipeline(cache, lambda: candidates, analyzer,
                                 enabled=True, max_workers=1, max_candidates=2,
                                 discovery_interval_seconds=30, retry_interval_seconds=10,
                                 clock=cache.clock)
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if len(set(started)) == len(candidates):
                break
            time.sleep(.05)
        assert set(started) == {item["symbol"] for item in candidates}
        assert max_active == 1
        state = pipeline.poll()
        assert state["max_workers"] == 1
        assert state["discovered_universe_count"] == 5
        assert len(state["results"]) <= 2
    finally:
        pipeline.close()


def test_portfolio_admission_is_atomic_bounded_disabled_by_default_and_reconciles():
    disabled = PortfolioAdmission()
    identity = {"venue": "delta", "market_type": "perpetual", "instrument_id": "p-1", "symbol": "BTCUSDT"}
    assert disabled.reserve(identity, 100) is None

    admission = PortfolioAdmission(enabled=True, max_positions=2, max_total_notional=900)
    def attempt(i):
        return admission.reserve({"venue": "delta", "market_type": "perpetual",
                                 "instrument_id": f"p-{i}", "symbol": f"COIN{i}USDT"}, 400)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        reservations = list(pool.map(attempt, range(12)))
    accepted = [item for item in reservations if item is not None]
    assert len(accepted) == 2
    assert admission.snapshot()["position_count"] == 2
    assert admission.snapshot()["total_notional"] == 800
    assert admission.reserve({"venue": "delta", "market_type": "perpetual", "instrument_id": "nan", "symbol": "BADUSDT"}, math.nan) is None
    before = admission.snapshot()
    malformed = [identity, identity]
    assert admission.reconcile(malformed) is False
    assert admission.snapshot() == before
    assert admission.release(accepted[0].token) is True
    assert admission.snapshot()["position_count"] == 1
    assert admission.reconcile([{"venue": "delta", "market_type": "perpetual", "instrument_id": "real-1",
                                 "symbol": "BTCUSDT", "notional": 500}]) is True
    assert admission.snapshot()["position_count"] == 1
    assert admission.reserve({"venue": "delta", "market_type": "perpetual", "instrument_id": "real-2", "symbol": "ETHUSDT"}, 400) is True or admission.snapshot()["position_count"] == 2


def test_full_jarvis_parts_1_to_12_are_isolated_per_symbol_and_serial_parallel_equivalent(monkeypatch):
    import jarvis_FIXED as jarvis

    cache, _, _ = make_cache()
    keys = [InstrumentKey("delta", "perpetual", f"product-{symbol}", symbol) for symbol in SYMBOLS]
    snapshots = [cache.refresh(key.symbol, venue=key.venue, market_type=key.market_type,
                               instrument_id=key.instrument_id) for key in keys]
    engine = object.__new__(jarvis.LiveTradingEngine)
    original_factory = jarvis.JarvisElite
    owners = []
    owner_lock = threading.Lock()

    def tracked_owner(*args, **kwargs):
        owner = original_factory(*args, **kwargs)
        with owner_lock:
            owners.append(owner)
        return owner

    monkeypatch.setattr(jarvis, "JarvisElite", tracked_owner)
    def analyze(pair):
        key, snapshot = pair
        return engine._analyze_multicoin_instrument(key, snapshot)

    inputs = list(zip(keys, snapshots))
    serial = [analyze(item) for item in inputs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parallel = list(pool.map(analyze, inputs))

    integrated_pipeline = MultiCoinPipeline(
        cache, lambda: (),
        lambda key, snapshot: engine._analyze_multicoin_instrument(key, snapshot),
        enabled=False, clock=cache.clock,
    )
    integrated_result = integrated_pipeline._run(keys[0])
    integrated_pipeline.close()
    parts_by_tf = integrated_result.get('parts_by_timeframe', {})
    from jarvis_strategy_approval import PART_WEIGHTS
    expected_adapter_parts = set(PART_WEIGHTS)
    for tf in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h']:
        if tf not in parts_by_tf:
            print(f'MISSING TF: {tf}')
        else:
            missing = expected_adapter_parts - set(parts_by_tf[tf].keys())
            if missing:
                print(f'TF {tf} MISSING PARTS: {missing}')
    print('ONCE PER SYMBOL:', list(integrated_result.get('once_per_symbol_parts', {}).keys()))
    assert integrated_result["status"] == "COMPLETE", {"diagnostics": [getattr(o,"latest_diagnostics_cycle_status",None) for o in owners], "timeframes": list(parts_by_tf), "once": list(integrated_result.get("once_per_symbol_parts", {})), "missing": {tf: sorted(expected_adapter_parts-set(parts)) for tf,parts in parts_by_tf.items()}}
    assert integrated_result["coverage"] == [f"Part{i}" for i in range(1, 13)]
    assert integrated_result["execution_eligible"] is False
    assert integrated_result["execution_plan"] is None
    assert integrated_result["execution_candidate_status"] == "BLOCKED_MISSING_EXPLICIT_PLAN"
    assert len(owners) == 7
    assert len({id(owner) for owner in owners}) == 7
    # assert all(len(owner.parts["part1_breakout"]._engines) == len(LIVE_TIMEFRAMES) for owner in owners)
    assert len({id(owner.parts["part2_zone"]) for owner in owners}) == 7
    required = set(PART_WEIGHTS)
    for key, serial_result, parallel_result in zip(keys, serial, parallel):
        for result in (serial_result, parallel_result):
            assert set(result["parts_by_timeframe"]) == set(LIVE_TIMEFRAMES)
            assert all(required.issubset(set(parts)) for parts in result["parts_by_timeframe"].values())
            assert set(result["once_per_symbol_parts"]) == {"part11", "part12"}
            assert set(result["adapter_engine_status"]) >= {"part1_native", "part2_native"}
        assert serial_result["parts_by_timeframe"] == parallel_result["parts_by_timeframe"]
        assert serial_result["once_per_symbol_parts"] == parallel_result["once_per_symbol_parts"]
        assert serial_result["adapter_engine_status"] == parallel_result["adapter_engine_status"]


def test_actual_live_product_discovery_joins_scanner_liquidity_to_exact_delta_products():
    from types import SimpleNamespace
    import jarvis_FIXED as jarvis

    class Scanner:
        def get_all_scores(self):
            return {"BTC": {"volume_24h_usdt": 20_000_000},
                    "ETH": {"volume_24h_usdt": 12_000_000},
                    "SOL": {"volume_24h_usdt": 1_000_000}}

    class Products:
        def get_available_products(self):
            return [
                {"id": 42, "symbol": "BTCUSDT", "contract_type": "perpetual_futures", "state": "live", "underlying_asset": {"symbol": "BTC"}, "quoting_asset": {"symbol": "USDT"}, "settling_asset": {"symbol": "USDT"}, "contract_value": "0.001", "contract_unit_currency": "BTC", "tick_size": "0.1", "max_leverage": 20},
                {"id": 84, "symbol": "ETHUSDT", "contract_type": "perpetual_futures", "state": "active", "underlying_asset": {"symbol": "ETH"}, "quoting_asset": {"symbol": "USDT"}, "settling_asset": {"symbol": "USDT"}, "contract_value": "0.01", "contract_unit_currency": "ETH", "tick_size": "0.01", "max_leverage": 20},
                {"id": 86, "symbol": "SOLUSDT", "contract_type": "perpetual_futures", "state": "active", "underlying_asset": {"symbol": "SOL"}, "quoting_asset": {"symbol": "USDT"}, "settling_asset": {"symbol": "USDT"}, "contract_value": "0.1", "contract_unit_currency": "SOL", "tick_size": "0.01", "max_leverage": 20},
                {"symbol": "ETHUSD", "contract_type": "perpetual", "state": "active"},
            ]

    engine = object.__new__(jarvis.LiveTradingEngine)
    engine.market_router = SimpleNamespace(scanner=Scanner(), delta=Products())
    records = engine._multicoin_candidate_instruments()
    assert [item["symbol"] for item in records] == ["BTCUSDT", "ETHUSDT"]
    assert records[0]["instrument_id"] == "BTCUSDT"
    assert records[0]["market_type"] == "spot"
    assert all(item["venue"] == "binance" for item in records)
    assert records[0]["execution_identity"] == {
        "venue": "delta", "market_type": "perpetual_futures", "instrument_id": "42", "symbol": "BTCUSDT"
    }
    assert records[0]["mapping_policy_id"]
    assert "liquidity_24h_usdt" in records[0]


def test_actual_entrypoint_wiring_is_default_off_and_results_never_enter_orders():
    import ast
    source = Path(__file__).parents[1].joinpath("jarvis_FIXED.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    live = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiveTradingEngine")
    methods = {node.name: node for node in live.body if isinstance(node, ast.FunctionDef)}
    init = ast.unparse(methods["__init__"])
    loop = ast.unparse(methods["start_live_trading"])
    assert "multicoin_analysis_enabled()" in init
    assert "MultiCoinPipeline" in init
    assert "self.multicoin_pipeline.poll" in loop
    assert "self.jarvis.analyze_trade_setup" in loop
    assert "self.multicoin_pipeline_status" in loop
    pipeline_handoffs = []
    for node in ast.walk(methods["start_live_trading"]):
        if isinstance(node, ast.Call) and any("multicoin_pipeline_status" in ast.unparse(arg) for arg in node.args):
            if isinstance(node.func, ast.Attribute):
                pipeline_handoffs.append(node.func.attr)
    assert sorted(pipeline_handoffs) == ["_consume_multicoin_delta_results", "_consume_multicoin_paper_results"]
    assert "self._consume_multicoin_paper_results(self.multicoin_pipeline_status)" in loop
    assert "self._consume_multicoin_delta_results(self.multicoin_pipeline_status)" in loop
