"""Offline end-to-end coverage for the full multicoin analysis-to-Delta handoff."""
from __future__ import annotations

import hashlib
import json

import pytest

from native_candle_fixture import native_times, TIMEFRAME_HISTORY_CANDLES
from direct_candle_cache import DirectCandleCache, FETCH_CANDLES, LIVE_TIMEFRAMES
from jarvis_delta_execution import DeltaExecutionAdapter, build_delta_candidate
from jarvis_multicoin_execution import CandidateRejected
from jarvis_multicoin_pipeline import InstrumentKey, MultiCoinPipeline
from jarvis_strategy_approval import (
    PART_WEIGHTS, REQUIRED_TIMEFRAMES, build_execution_plan,
    evaluate_mtf_central_strategy, make_entry_approval,
)

NOW = 1_800_000_000.0
STEPS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900,
         "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
POLICY = {
    "ETH": {
        "policy_id": "eth-delta-risk-v1",
        "mapping_policy_id": "binance_spot_usdt_to_delta_linear_perpetual_usdt",
        "binance_symbol": "ETHUSDT", "delta_symbol": "ETHUSDT",
        "delta_instrument_id": "22", "delta_market_type": "perpetual_futures",
        "delta_quote_asset": "USDT", "delta_settling_asset": "USDT", "risk_currency": "USDT",
        "min_confidence": 70, "max_quote_age_seconds": 5, "max_spread_pct": 0.2,
        "max_chase_pct": 1.0, "max_slippage_pct": 0.2,
        "max_trade_risk_currency": 5.0,
    }
}


class OfflineBinanceCandles:
    """Deterministic candle transport; no network/API calls."""
    def __init__(self):
        self.last_candle_source = "binance"

    def get_historical_candles_with_metadata(self, *, symbol, resolution, limit):
        rows = []
        for timestamp in native_times(resolution, NOW, limit):
            close = 100.0 + (timestamp % 1_000_000) * 1e-6
            rows.append({"time": timestamp, "open": close - 0.01, "high": close + 0.10,
                         "low": close - 0.10, "close": close, "volume": 10.0})
        return {"candles": rows, "source": "binance", "symbol": symbol}


def _valid_gate(symbol):
    return {
        "symbol": symbol, "entry_blocked": False, "risk_veto": False,
        "status": "ok", "data_status": "valid", "timeframe": "aggregate",
        "blocked_timeframes": [], "veto_timeframes": [],
        "timeframe_results": {
            tf: {"symbol": symbol, "timeframe": tf, "status": "neutral", "data_status": "valid",
                 "entry_blocked": False, "risk_veto": False}
            for tf in LIVE_TIMEFRAMES
        },
    }


def _central_evidence():
    return {
        "part1_breakout": {"signal": 1, "thought": "breakout"},
        "part2_zone": {"signal": 1, "thought": "demand"},
        "part3_psychology": {"signal": 1, "thought": "bullish"},
        "part4_volume": {"signal": 1, "thought": "volume"},
        "part5_ml": {"signal": 1, "thought": "model"},
        "part6_trend": {"signal": 1, "thought": "trend"},
        "part7_volatility": {"signal": 1, "thought": "volatility"},
        "part8_structure": {"signal": 1, "thought": "structure"},
        "part9_orderflow": {"signal": 1, "thought": "orderflow"},
        "part10_candlestats": {"signal": 1, "thought": "candles"},
    }


def _expected_pipeline_version(key, snapshot):
    frames = {tf: {"source": snapshot.frames[tf].source,
                   "last_closed": str(snapshot.frames[tf].closed.index[-1]),
                   "forming_time": snapshot.frames[tf].current.get("time")}
              for tf in LIVE_TIMEFRAMES}
    return hashlib.sha256(json.dumps({"identity": key.as_tuple(), "frames": frames}, sort_keys=True).encode()).hexdigest()[:24]


class FakeDeltaBroker:
    """Read/entry fixture for adapter orchestration; never reaches a venue."""
    def __init__(self):
        self.quote_calls = 0
        self.entry_calls = []
        self.product = {"id": 22, "symbol": "ETHUSDT", "state": "active",
                        "product_type": "perpetual_futures", "base_asset": "ETH",
                        "quote_asset": "USDT", "settling_asset": "USDT",
                        "contract_value_usdt": 1.0, "max_leverage": 10, "contract_size": 1}
        self.reference = 100.0

    def get_product_metadata(self, symbol):
        return dict(self.product) if symbol == "ETHUSDT" else None

    def get_delta_executable_quote(self, *, symbol, product_id):
        self.quote_calls += 1
        if symbol != "ETHUSDT" or str(product_id) != "22":
            return None
        return {"source": "delta", "symbol": symbol, "product_id": "22",
                "bid": self.reference, "ask": self.reference * 1.001, "observed_at": NOW}

    def get_available_balance(self, currency):
        return 500.0 if currency == "USDT" else 0.0

    def get_complete_account_snapshot(self, *, owned_orders):
        return {"complete": True, "as_of": NOW, "positions": [], "orders": {},
                "external_orders": [], "risk_currency": "USDT"}

    def place_protected_order(self, **kwargs):
        self.entry_calls.append(kwargs)
        return {"status": "FILLED", "authoritative": True, "order_id": "offline-order-1",
                "filled_quantity": kwargs["size"], "average_fill_price": self.reference * 1.001,
                "protection_state": "ACTIVE",
                "protective_exits": {"stop_loss_order_id": "stop-1", "take_profit_order_id": "target-1"}}


def _raw_analysis(key, snapshot):
    version = _expected_pipeline_version(key, snapshot)
    price = float(snapshot.frames["1m"].closed["close"].iloc[-1])
    closed_at = float(snapshot.frames["1m"].closed.index[-1].timestamp()) + 60.0
    gate = _valid_gate(key.symbol)
    evidence = _central_evidence()
    by_timeframe = {}
    normalized = {}
    for timeframe in LIVE_TIMEFRAMES:
        row = {}
        canonical_row = {}
        bullish = timeframe in {"1m", "3m", "5m", "15m"}
        for index, (name, item) in enumerate(evidence.items(), 1):
            scoped = {**item, "symbol": key.symbol, "timeframe": timeframe}
            if name == "part7_volatility":
                scoped.update({"signal": 0, "thought": "neutral volatility", "status": "neutral",
                               "data_status": "valid", "entry_blocked": False, "risk_veto": False})
            elif not bullish:
                scoped.update({"signal": 0, "thought": "neutral native-frame evidence"})
            if name == "part2_zone":
                scoped["native_timeframe"] = timeframe
            row[f"part{index}"] = scoped
            canonical_row[name] = scoped
        by_timeframe[timeframe] = row
        normalized[timeframe] = canonical_row
    decision = evaluate_mtf_central_strategy(normalized, gate, confidence=90, expected_symbol=key.symbol)
    strategy_plan = build_execution_plan(
        direction=decision["direction"], recommended_expiry=decision["trade_mode"], entry_price=price,
        stop_loss=price * 0.98, take_profit=price * 1.05,
        symbol=key.mapped_execution_dict()["symbol"], snapshot_version=version, confidence=90,
    )
    approval = make_entry_approval(
        {**decision, "symbol": "ETHUSDT"}, direction=decision["direction"], symbol="ETHUSDT",
        exchange="delta", contract="ETHUSDT", instrument_id="22", market_type="perpetual_futures",
        analysis_symbol=key.symbol, analysis_exchange="binance", snapshot_version=version,
        analysis_timestamp=float(snapshot.fetched_at), confidence=90, execution_plan=strategy_plan,
    )
    return {
        "parts_by_timeframe": by_timeframe,
        "once_per_symbol_parts": {"part11": {"signal": 1}, "part12": {"confidence": 90}},
        "part7_gate": gate, "central_strategy_decision": decision,
        "central_strategy_evidence": evidence, "central_strategy_approval": approval,
        "central_execution_plan": strategy_plan, "execution_plan": strategy_plan,
        "decision_authority": "jarvis_FIXED",
        "deterministic_decision": {"origin": "jarvis_FIXED_central_strategy", "direction": decision["direction"],
                                   "confidence": 90, "entry_price": price,
                                   "stop_loss": strategy_plan["stop_loss"],
                                   "take_profit": strategy_plan["take_profit"]},
        "analysis_reference": {"source": "binance", "symbol": key.symbol, "timeframe": "1m",
                               "timestamp": closed_at, "price": price},
    }


def _pipeline_case(analyzer):
    key = InstrumentKey.from_record({
        "venue": "binance", "market_type": "spot", "instrument_id": "ETHUSDT", "symbol": "ETHUSDT",
        "execution_identity": {"venue": "delta", "market_type": "perpetual_futures",
                               "instrument_id": "22", "symbol": "ETHUSDT"},
        "mapping_policy_id": POLICY["ETH"]["mapping_policy_id"],
    })
    cache = DirectCandleCache(OfflineBinanceCandles(), clock=lambda: NOW,
                              max_concurrent_fetches=1, ttl_seconds=0)
    pipeline = MultiCoinPipeline(cache, lambda: (), analyzer, enabled=False, clock=lambda: NOW)
    try:
        result = pipeline._run(key)
        return key, result
    finally:
        pipeline.close()


def test_full_binance_analysis_pipeline_produces_scoped_plan_then_fake_delta_entry():
    broker = FakeDeltaBroker()
    key, result = _pipeline_case(lambda key, snapshot: _raw_analysis(key, snapshot))
    assert result["status"] == "COMPLETE"
    assert result["scope"] == "parts1-12-analysis-only"
    assert result["decision_authority"] == "none"
    assert result["execution_eligible"] is False
    assert result["central_execution_plan"]["symbol"] == "ETHUSDT"
    assert result["central_strategy_approval"]["execution_plan_id"] == result["central_execution_plan"]["plan_id"]
    assert result["execution_plan"]["schema_version"] == "jarvis-execution-plan-v1"
    result["freshness_status"] = "FRESH"  # equivalent to the pipeline poll's fresh-result view
    broker.reference = float(result["analysis_reference"]["price"])
    candidate = build_delta_candidate(result, delta=broker, policy_registry=POLICY, now=NOW)
    assert candidate.identity.venue == "delta" and candidate.analysis_identity.venue == "binance"
    assert candidate.direction == "BUY" and candidate.size_unit == "contracts"
    assert candidate.entry_authorization["central_approval"]["execution_plan_id"] == result["central_execution_plan"]["plan_id"]
    adapter = DeltaExecutionAdapter(broker, policy_registry=POLICY,
                                    authorization_check=lambda: True, clock=lambda: NOW)
    response = adapter.submit(candidate, "synthetic-e2e-entry")
    assert response["status"] == "FILLED"
    assert len(broker.entry_calls) == 1
    # The broker request carries only the validated contract-level plan; the
    # separate Jarvis approval and strategy plan were recomputed and bound on
    # the candidate before sizing, not leaked as venue API parameters.
    assert broker.entry_calls[0]["stop_loss"] == candidate.stop_loss
    assert broker.entry_calls[0]["take_profit"] == candidate.take_profit


def test_pipeline_analysis_stays_non_executable_and_part7_veto_fails_before_broker_quote():
    broker = FakeDeltaBroker()
    key, result = _pipeline_case(lambda key, snapshot: _raw_analysis(key, snapshot))
    result["freshness_status"] = "FRESH"
    assert result["execution_eligible"] is False
    assert result["execution_plan"]["authority"] == "jarvis_FIXED"
    result["part7_gate"] = {"symbol": key.symbol, "entry_blocked": True, "risk_veto": True}
    with pytest.raises(CandidateRejected):
        build_delta_candidate(result, delta=broker, policy_registry=POLICY, now=NOW)
    assert broker.quote_calls == 0
    assert broker.entry_calls == []
