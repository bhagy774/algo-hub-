"""Strict bridge from Binance-backed Parts 1-12 analysis to Delta execution.

This module deliberately does not invent a signal, entry, stop, target, or size.
It consumes the deterministic Jarvis decision and its stop/target levels, uses a
fresh Delta-native executable quote and exact active product metadata, then
sizes through the repository's existing risk and lot-limit functions. A live
entry requires a broker-provided atomic protected-order API and authoritative
account snapshot API. The current DeltaExchangeData adapter does not provide
those capabilities, so the production path remains fail-closed.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional

from direct_candle_cache import LIVE_TIMEFRAMES
from jarvis_multicoin_execution import (
    CandidateRejected, FullIdentity, PortfolioCoordinator, ValidatedCandidate,
    _base_asset, _nonnegative_number, _positive_number, candidate_from_analysis,
)


_ALLOWED_DELTA_TYPES = {"perpetual", "perpetual_futures", "perpetual_swap", "perpetual_swaps"}
_ALLOWED_QUOTES = {"USDT"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _asset_symbol(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("symbol") or value.get("asset_symbol") or value.get("name")
    return _text(value).upper()


def _product_fields(product: Mapping[str, Any]) -> tuple[str, str, str, str]:
    base = _asset_symbol(product.get("underlying_asset_symbol") or product.get("underlying_asset")
                         or product.get("base_asset"))
    quote = _asset_symbol(product.get("quoting_asset") or product.get("quote_asset")
                          or product.get("quote_currency"))
    settle = _asset_symbol(product.get("settling_asset") or product.get("settlement_asset")
                           or product.get("settlement_currency"))
    marker = _text(product.get("product_type") or product.get("market_type")
                   or product.get("contract_type")).lower()
    return base, quote, settle, marker


def _active_product(product: Mapping[str, Any], expected: FullIdentity,
                    policy: Optional[Mapping[str, Any]] = None) -> bool:
    try:
        if str(product.get("id", product.get("product_id"))) != expected.instrument_id:
            return False
        if _text(product.get("symbol")).upper() != expected.symbol:
            return False
        if _text(product.get("state") or product.get("status")).lower() not in {"active", "live", "trading", "listed"}:
            return False
        base, quote, settle, marker = _product_fields(product)
        if marker not in _ALLOWED_DELTA_TYPES or marker != expected.market_type:
            return False
        if base != _base_asset(expected.symbol) or not quote or quote != settle:
            return False
        # Risk and notional are held in the venue's actual settlement currency.
        # A quote/settlement conversion is intentionally unsupported until a
        # separately sourced, fresh, authoritative FX adapter exists.
        if policy is not None and (
                quote != _asset_symbol(policy.get("delta_quote_asset"))
                or settle != _asset_symbol(policy.get("delta_settling_asset"))
                or settle != _asset_symbol(policy.get("risk_currency"))):
            return False
        return True
    except (CandidateRejected, AttributeError):
        return False


def _require_analysis(result: Mapping[str, Any], policy: Mapping[str, Any], now: float, max_age: float) -> tuple[FullIdentity, Dict[str, Any], float, float, str, int]:
    if not isinstance(result, Mapping) or result.get("status") != "COMPLETE":
        raise CandidateRejected("only a fresh COMPLETE Parts 1-12 analysis may be planned")
    if (result.get("scope") != "parts1-12-analysis-only" or result.get("analysis_only") is not True
            or result.get("decision_authority") != "none" or result.get("execution_eligible") is not False):
        raise CandidateRejected("analysis result has an unexpected authority/scope marker")
    if result.get("freshness_status") != "FRESH":
        raise CandidateRejected("analysis result freshness is missing or stale")
    expected_coverage = [f"Part{i}" for i in range(1, 13)]
    if result.get("coverage") != expected_coverage:
        raise CandidateRejected("analysis does not explicitly cover Parts 1-12")
    if not _text(result.get("snapshot_version")):
        raise CandidateRejected("analysis snapshot version is missing")
    parts_by_timeframe = result.get("parts_by_timeframe")
    required_parts = {f"part{i}" for i in range(1, 11)}
    if (not isinstance(parts_by_timeframe, Mapping)
            or set(parts_by_timeframe) != set(LIVE_TIMEFRAMES)
            or any(not isinstance(parts_by_timeframe.get(tf), Mapping)
                   or not required_parts.issubset(set(parts_by_timeframe[tf])) for tf in LIVE_TIMEFRAMES)):
        raise CandidateRejected("complete Parts 1-10 timeframe coverage is missing")
    source_identity = FullIdentity.parse(result.get("request_identity"))
    if (source_identity.venue != "binance" or source_identity.market_type != "spot"
            or source_identity.instrument_id != source_identity.symbol):
        raise CandidateRejected("entry analysis must come from the matching Binance spot instrument")
    execution_identity = FullIdentity.parse(result.get("execution_identity"))
    if execution_identity.venue != "delta" or execution_identity.market_type not in _ALLOWED_DELTA_TYPES:
        raise CandidateRejected("execution target must be an exact Delta perpetual identity")
    base = _base_asset(execution_identity.symbol)
    if _base_asset(source_identity.symbol) != base:
        raise CandidateRejected("Binance and Delta underlying assets do not match")
    if _text(policy.get("binance_symbol")).upper() != source_identity.symbol:
        raise CandidateRejected("Binance symbol is not allow-listed by the execution policy")
    if _text(policy.get("delta_symbol")).upper() != execution_identity.symbol:
        raise CandidateRejected("Delta symbol is not allow-listed by the execution policy")
    if _text(policy.get("delta_instrument_id")) != execution_identity.instrument_id:
        raise CandidateRejected("Delta product id is not allow-listed by the execution policy")
    if _text(policy.get("delta_market_type")).lower() != execution_identity.market_type:
        raise CandidateRejected("Delta market type is not allow-listed by the execution policy")
    mapping_policy_id = _text(policy.get("mapping_policy_id"))
    policy_id = _text(policy.get("policy_id"))
    if not mapping_policy_id or not policy_id:
        raise CandidateRejected("explicit asset-specific market-mapping and execution policy are required")
    if _text(result.get("mapping_policy_id")) != mapping_policy_id:
        raise CandidateRejected("analysis/execution market mapping is not the discovered mapping")

    try:
        fetched = float(result["snapshot_fetched_at"])
        completed = float(result["analysis_completed_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise CandidateRejected("analysis snapshot timestamps are missing")
    if not all(math.isfinite(x) for x in (fetched, completed, now)) or fetched > completed or completed > now + 2:
        raise CandidateRejected("analysis snapshot timestamps are invalid")
    if now - fetched < -2 or now - fetched > max(1.0, max_age):
        raise CandidateRejected("analysis snapshot is stale or from the future")

    reference = result.get("analysis_reference")
    if not isinstance(reference, Mapping):
        raise CandidateRejected("Binance closed-candle reference is missing")
    if (_text(reference.get("source")).lower() != "binance"
            or _text(reference.get("symbol")).upper() != source_identity.symbol
            or _text(reference.get("timeframe")) != "1m"):
        raise CandidateRejected("reference data is not the matching Binance 1m candle")
    try:
        reference_at = float(reference["timestamp"])
        reference_price = _positive_number(reference["price"], "Binance reference price")
    except (KeyError, TypeError, ValueError, OverflowError):
        raise CandidateRejected("Binance reference candle is malformed")
    if not math.isfinite(reference_at) or reference_at > fetched + 2 or reference_at > now + 2 or now - reference_at > max_age:
        raise CandidateRejected("Binance reference candle is stale or from the future")

    decision = result.get("deterministic_decision")
    if not isinstance(decision, Mapping) or decision.get("origin") != "jarvis_FIXED_central_strategy":
        raise CandidateRejected("deterministic Parts 1-12 decision is missing")
    direction = _text(decision.get("direction")).upper()
    if direction in {"NO_TRADE", "NONE", "HOLD", "NEUTRAL", ""}:
        raise CandidateRejected("NO TRADE decision")
    direction = {"CALL": "BUY", "PUT": "SELL"}.get(direction, direction)
    if direction not in {"BUY", "SELL"}:
        raise CandidateRejected("unsupported deterministic direction")
    try:
        confidence = int(decision.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("deterministic confidence is missing")
    min_confidence = int(policy.get("min_confidence", 0))
    if min_confidence <= 0 or confidence < min_confidence or confidence > 100:
        raise CandidateRejected("confidence does not meet the asset policy")

    parts = result.get("once_per_symbol_parts")
    p11 = parts.get("part11") if isinstance(parts, Mapping) else None
    p12 = parts.get("part12") if isinstance(parts, Mapping) else None
    if not isinstance(p11, Mapping) or not isinstance(p12, Mapping):
        raise CandidateRejected("Part11/Part12 deterministic consensus is incomplete")
    try:
        p11_signal = int(p11.get("signal"))
        p12_confidence = float(p12.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("Part11/Part12 consensus values are malformed")
    expected_signal = 1 if direction == "BUY" else -1
    if p11_signal != expected_signal or not math.isfinite(p12_confidence) or p12_confidence < min_confidence:
        raise CandidateRejected("Part11/Part12 do not confirm the deterministic trade decision")
    gate = result.get("part7_gate")
    if not isinstance(gate, Mapping) or gate.get("entry_blocked") is not False or gate.get("risk_veto") is not False:
        raise CandidateRejected("Part7 new-entry gate is missing or blocks entry")
    try:
        entry_reference = _positive_number(decision.get("entry_price"), "decision reference entry")
        stop = _positive_number(decision.get("stop_loss"), "deterministic stop loss")
        target = _positive_number(decision.get("take_profit"), "deterministic take profit")
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("deterministic entry/stop/target are incomplete")
    if not math.isclose(entry_reference, reference_price, rel_tol=1e-8, abs_tol=1e-8):
        raise CandidateRejected("decision reference price does not match Binance candle")
    if direction == "BUY" and not stop < entry_reference < target:
        raise CandidateRejected("BUY stop/target geometry is invalid")
    if direction == "SELL" and not target < entry_reference < stop:
        raise CandidateRejected("SELL stop/target geometry is invalid")
    return execution_identity, dict(decision), reference_at, reference_price, direction, confidence


def _quote_for(delta: Any, identity: FullIdentity, policy: Mapping[str, Any], now: float) -> tuple[float, float, float]:
    quote_reader = getattr(delta, "get_delta_executable_quote", None)
    if not callable(quote_reader):
        raise CandidateRejected("Delta-native executable quote adapter is unavailable")
    quote = quote_reader(symbol=identity.symbol, product_id=identity.instrument_id)
    if not isinstance(quote, Mapping) or _text(quote.get("source")).lower() != "delta":
        raise CandidateRejected("executable quote source is not Delta")
    if _text(quote.get("symbol")).upper() != identity.symbol or _text(quote.get("product_id")) != identity.instrument_id:
        raise CandidateRejected("Delta quote instrument identity mismatch")
    try:
        observed = float(quote.get("observed_at"))
        bid = _positive_number(quote.get("bid"), "Delta bid")
        ask = _positive_number(quote.get("ask"), "Delta ask")
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("Delta executable quote is malformed")
    max_quote_age = _positive_number(policy.get("max_quote_age_seconds"), "max_quote_age_seconds")
    if not math.isfinite(observed) or observed > now + 2 or now - observed < -2 or now - observed > max_quote_age:
        raise CandidateRejected("Delta executable quote is stale or from the future")
    if ask < bid:
        raise CandidateRejected("Delta order book is crossed")
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0
    if spread_pct > _nonnegative_number(policy.get("max_spread_pct"), "max_spread_pct"):
        raise CandidateRejected("Delta spread exceeds the asset policy")
    return bid, ask, observed


def build_delta_candidate(
    result: Mapping[str, Any], *, delta: Any, policy_registry: Mapping[str, Mapping[str, Any]],
    now: Optional[float] = None, max_age_seconds: float = 180.0,
) -> ValidatedCandidate:
    """Build a candidate only from deterministic analysis + authoritative Delta sizing data."""
    current = time.time() if now is None else float(now)
    try:
        execution_identity = FullIdentity.parse(result.get("execution_identity"))
        asset = _base_asset(execution_identity.symbol)
    except (AttributeError, CandidateRejected):
        raise CandidateRejected("full mapped Delta execution identity is missing")
    policy = policy_registry.get(asset)
    if not isinstance(policy, Mapping):
        raise CandidateRejected("no explicit per-asset Delta execution policy")
    identity, decision, reference_at, reference_price, direction, confidence = _require_analysis(
        result, policy, current, float(max_age_seconds)
    )
    if identity != execution_identity:
        raise CandidateRejected("execution identity changed during candidate validation")

    metadata_reader = getattr(delta, "get_product_metadata", None)
    if not callable(metadata_reader):
        raise CandidateRejected("Delta product metadata adapter is unavailable")
    metadata = metadata_reader(identity.symbol)
    if not isinstance(metadata, Mapping) or not _active_product(metadata, identity, policy):
        raise CandidateRejected("Delta product is inactive, mismatched, or lacks exact quote/settlement metadata")
    _base, quote_currency, settlement_currency, _kind = _product_fields(metadata)
    if not settlement_currency or quote_currency != settlement_currency:
        raise CandidateRejected("Delta quote and settlement currencies differ; validated FX conversion is unavailable")
    risk_currency = _asset_symbol(policy.get("risk_currency"))
    if risk_currency != settlement_currency:
        raise CandidateRejected("asset policy risk currency does not match Delta settlement currency")
    bid, ask, quote_time = _quote_for(delta, identity, policy, current)
    entry = ask if direction == "BUY" else bid
    max_chase = _nonnegative_number(policy.get("max_chase_pct"), "max_chase_pct")
    max_slippage = _nonnegative_number(policy.get("max_slippage_pct"), "max_slippage_pct")
    if quote_currency == "USDT" and abs(entry - reference_price) / reference_price * 100.0 > max_chase:
        raise CandidateRejected("Delta executable quote is too far from Binance USDT reference")
    source_stop = _positive_number(decision.get("stop_loss"), "deterministic stop loss")
    source_target = _positive_number(decision.get("take_profit"), "deterministic take profit")
    # Map price levels by the exact same-asset Delta/Binance quote ratio. This
    # preserves deterministic percent geometry when the actual Delta quote
    # denomination differs (e.g. USD vs Binance USDT); it is not an FX rate.
    scale = 1.0 if quote_currency == "USDT" else entry / reference_price
    stop, target = source_stop * scale, source_target * scale
    if direction == "BUY" and not stop < entry < target:
        raise CandidateRejected("Delta ask invalidates scaled deterministic BUY stop/target geometry")
    if direction == "SELL" and not target < entry < stop:
        raise CandidateRejected("Delta bid invalidates scaled deterministic SELL stop/target geometry")

    try:
        from jarvis_risk import calculate_trade_size, contract_quote_value_in_currency
        from jarvis_lot_limits import enforce_entry_lots
    except Exception as exc:
        raise CandidateRejected(f"deterministic risk/lot sizing unavailable: {type(exc).__name__}")
    balance_reader = getattr(delta, "get_available_balance", None)
    if not callable(balance_reader):
        raise CandidateRejected("Delta exact-currency available-balance adapter is unavailable")
    balance = _positive_number(balance_reader(settlement_currency), f"Delta available {settlement_currency} balance")
    contract_value = contract_quote_value_in_currency(dict(metadata), entry, settlement_currency)
    if contract_value is None or not math.isfinite(float(contract_value)) or float(contract_value) <= 0:
        raise CandidateRejected("Delta contract value is not explicitly convertible to its settlement currency")
    stop_distance_fraction = abs(entry - stop) / entry
    risk_budget = _positive_number(policy.get("max_trade_risk_currency"), "max_trade_risk_currency")
    leverage_value = metadata.get("max_leverage", metadata.get("max_leverage_allowed", metadata.get("leverage_max")))
    # The current public Product schema exposes default_leverage and notional
    # caps but does not expose a reliable tier-wide maximum. Never treat the
    # default as the maximum: if no explicit max is in metadata, size at 1x.
    if leverage_value is None:
        leverage_value = 1
    size = calculate_trade_size(
        balance, confidence, stop_distance_fraction,
        max_trade_risk_currency=risk_budget, risk_currency=settlement_currency,
        product_max_leverage=leverage_value,
        contract_value_usdt=float(contract_value), require_contract_value=True,
    )
    if not isinstance(size, Mapping) or size.get("ok") is not True:
        raise CandidateRejected("Delta risk sizing blocked: " + _text(size.get("reason") if isinstance(size, Mapping) else "invalid result"))
    quantity = enforce_entry_lots(size.get("contracts"), metadata=dict(metadata), available_balance=balance)
    if not math.isfinite(float(quantity)) or quantity <= 0 or not float(quantity).is_integer():
        raise CandidateRejected("Delta lot limits did not produce a positive whole-contract entry size")
    contract_multiplier = float(contract_value) / entry
    stop_risk = stop_distance_fraction * quantity * float(contract_value)
    if stop_risk <= 0 or stop_risk > risk_budget + 1e-8:
        raise CandidateRejected("rounded Delta quantity exceeds the per-trade risk budget")

    policy_id = _text(policy.get("policy_id"))
    mapping_policy_id = _text(policy.get("mapping_policy_id"))
    plan = {
        "timeframe": "1m", "decision_timestamp": float(result["analysis_completed_at"]),
        "reference_price_timestamp": reference_at, "reference_price": reference_price,
        "max_slippage_pct": max_slippage, "max_chase_pct": max_chase,
        "direction": direction, "entry_price": entry, "stop_loss": stop, "take_profit": target,
        "quantity": quantity, "size_unit": "contracts", "contract_multiplier": contract_multiplier,
        "sizing_provenance": "jarvis_risk.calculate_trade_size+jarvis_lot_limits.enforce_entry_lots",
        "risk_notional": stop_risk, "risk_currency": settlement_currency,
        "policy_id": policy_id, "mapping_policy_id": mapping_policy_id,
        "execution_identity": identity.to_dict(), "delta_quote_observed_at": quote_time,
        "delta_bid": bid, "delta_ask": ask,
        "delta_product_value": float(contract_value), "delta_value_currency": settlement_currency,
        "delta_risk_budget": risk_budget, "delta_available_balance": balance,
        "delta_balance_currency": settlement_currency,
        "leverage": int(size.get("leverage", 1)),
        "analysis_to_delta_price_ratio": scale,
    }
    enriched = dict(result)
    enriched["execution_plan"] = plan
    policy_ids = {asset: policy_id}
    mapping_policy_ids = {asset: mapping_policy_id}
    candidate = candidate_from_analysis(
        enriched, policy_registry=policy_ids, mapping_policy_registry=mapping_policy_ids,
        now=current, max_age_seconds=max_age_seconds,
    )
    # Delta's documented bracket endpoint is a separate post-fill position
    # attachment, not an atomic entry+protection order. The concrete wrapper
    # implements this staged lifecycle and reports FILLED only after it sees
    # both active protective exits; UNKNOWN keeps the reservation locked.
    if not callable(getattr(delta, "place_protected_order", None)):
        raise CandidateRejected("Delta staged entry/bracket lifecycle adapter is unavailable")
    if not callable(getattr(delta, "get_complete_account_snapshot", None)):
        raise CandidateRejected("Delta broker does not provide complete authoritative account reconciliation")
    return candidate


class DeltaExecutionAdapter:
    """Live-only Delta adapter; read-only preparation is separable from submission."""
    def __init__(self, delta: Any, *, policy_registry: Mapping[str, Mapping[str, Any]],
                 authorization_check: Optional[Callable[[], bool]] = None, clock: Callable[[], float] = time.time):
        self.delta = delta
        self.policy_registry = {str(k).upper(): dict(v) for k, v in policy_registry.items() if isinstance(v, Mapping)}
        self.clock = clock
        self._authorization_check = authorization_check or self._live_flags_authorized
        self._direction_by_candidate: Dict[str, str] = {}

    def authorization_ready(self) -> bool:
        """Return the current independent live-order authorization state."""
        try:
            return bool(self._authorization_check())
        except Exception:
            return False

    @staticmethod
    def _live_flags_authorized() -> bool:
        required = ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION", "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED")
        truthy = {"true", "1", "yes", "on"}
        if any(os.environ.get(name, "false").strip().lower() not in truthy for name in required):
            return False
        if os.environ.get("JARVIS_KILL_SWITCH", "1") != "0" and os.path.exists("C:\\jarvis\\STOP_JARVIS"):
            return False
        return True

    def prepare_candidate(self, result: Mapping[str, Any], *, now: Optional[float] = None,
                          max_age_seconds: float = 180.0) -> ValidatedCandidate:
        return build_delta_candidate(result, delta=self.delta, policy_registry=self.policy_registry,
                                     now=self.clock() if now is None else now, max_age_seconds=max_age_seconds)

    def reconcile(self, coordinator: PortfolioCoordinator) -> bool:
        getter = getattr(self.delta, "get_complete_account_snapshot", None)
        if not callable(getter):
            return False
        ledger = coordinator.snapshot().get("orders", {})
        owned = {}
        for cid, row in ledger.items():
            candidate = row.get("candidate", {})
            client_id = "jme" + hashlib.sha256(str(cid).encode()).hexdigest()[:29]
            owned[cid] = {"order_id": row.get("order_id"),
                          "identity": candidate.get("identity"),
                          "direction": candidate.get("direction"),
                          "quantity": candidate.get("quantity"),
                          "stop_loss": candidate.get("stop_loss"),
                          "take_profit": candidate.get("take_profit"),
                          "protective_exits": row.get("protective_exits"),
                          "client_order_id": client_id,
                          "close_order_id": row.get("close_order_id"),
                          "close_client_order_id": row.get("close_client_order_id")}
        try:
            snapshot = getter(owned_orders=owned)
        except Exception:
            return False
        if not isinstance(snapshot, Mapping) or snapshot.get("complete") is not True:
            return False
        try:
            as_of = float(snapshot.get("as_of"))
            if not math.isfinite(as_of) or as_of > self.clock() + 2 or self.clock() - as_of > 15:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        if snapshot.get("external_orders"):
            return False
        policy_currencies = {_asset_symbol(p.get("risk_currency")) for p in self.policy_registry.values()}
        policy_currencies.discard("")
        if len(policy_currencies) != 1:
            return False
        if _asset_symbol(snapshot.get("risk_currency")) not in {"", next(iter(policy_currencies))}:
            return False
        for position in snapshot.get("positions", ()):
            if not isinstance(position, Mapping) or _asset_symbol(position.get("risk_currency")) != next(iter(policy_currencies)):
                return False
        return coordinator.reconcile(snapshot)

    def submit(self, candidate: ValidatedCandidate, idempotency_key: str) -> Mapping[str, Any]:
        if not self._authorization_check():
            return {"status": "REJECTED", "authoritative": True, "reason": "live Delta execution flags are not all enabled"}
        if candidate.identity.venue != "delta" or candidate.size_unit != "contracts":
            return {"status": "REJECTED", "authoritative": True, "reason": "candidate is not a Delta contract order"}
        identity = candidate.identity
        meta_reader = getattr(self.delta, "get_product_metadata", None)
        metadata = meta_reader(identity.symbol) if callable(meta_reader) else None
        policy = self.policy_registry.get(_base_asset(identity.symbol), {})
        if not isinstance(metadata, Mapping) or not _active_product(metadata, identity, policy):
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta product identity/status/denomination changed"}
        try:
            bid, ask, _observed = _quote_for(self.delta, identity, policy, self.clock())
            current_entry = ask if candidate.direction == "BUY" else bid
            if abs(current_entry - candidate.entry_price) / candidate.entry_price * 100.0 > candidate.max_slippage_pct:
                return {"status": "REJECTED", "authoritative": True, "reason": "Delta executable quote moved beyond allowed slippage"}
            _, quote_currency, _, _ = _product_fields(metadata)
            if (quote_currency == "USDT"
                    and abs(current_entry - candidate.reference_price) / candidate.reference_price * 100.0 > candidate.max_chase_pct):
                return {"status": "REJECTED", "authoritative": True, "reason": "Delta quote exceeds Binance USDT reference chase limit"}
            if candidate.direction == "BUY" and not candidate.stop_loss < current_entry < candidate.take_profit:
                return {"status": "REJECTED", "authoritative": True, "reason": "current quote invalidates BUY protective geometry"}
            if candidate.direction == "SELL" and not candidate.take_profit < current_entry < candidate.stop_loss:
                return {"status": "REJECTED", "authoritative": True, "reason": "current quote invalidates SELL protective geometry"}
        except CandidateRejected as exc:
            return {"status": "REJECTED", "authoritative": True, "reason": str(exc)}
        submitter = getattr(self.delta, "place_protected_order", None)
        if not callable(submitter):
            return {"status": "REJECTED", "authoritative": True, "reason": "atomic Delta protective-order API unavailable"}
        client_id = "jme" + hashlib.sha256(str(idempotency_key).encode()).hexdigest()[:29]
        self._direction_by_candidate[str(idempotency_key)] = candidate.direction
        try:
            response = submitter(
                product_id=int(identity.instrument_id), symbol=identity.symbol,
                side="buy" if candidate.direction == "BUY" else "sell", size=int(candidate.quantity),
                order_type="market", stop_loss=candidate.stop_loss, take_profit=candidate.take_profit,
                leverage=int(candidate.leverage), client_order_id=client_id,
                entry_authorization=candidate.entry_authorization,
            )
        except Exception:
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "reason": "Delta submit response lost"}
        if not isinstance(response, Mapping):
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "reason": "malformed Delta submit response"}
        state = _text(response.get("status")).upper()
        if state not in {"REJECTED", "SUBMISSION_UNKNOWN", "PARTIAL", "FILLED"}:
            state = "SUBMISSION_UNKNOWN"
        if response.get("authoritative") is not True or (state in {"FILLED", "PARTIAL"} and not _text(response.get("order_id"))):
            state = "SUBMISSION_UNKNOWN"
        try:
            filled = float(response.get("filled_quantity", 0.0) or 0.0)
            avg_price = float(response.get("average_fill_price", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            filled, avg_price = float("nan"), float("nan")
        if not math.isfinite(filled) or filled < 0 or filled > candidate.quantity:
            state, filled = "SUBMISSION_UNKNOWN", 0.0
        if state in {"FILLED", "PARTIAL"}:
            protected = response.get("protective_exits")
            if (not math.isfinite(avg_price) or avg_price <= 0
                    or not isinstance(protected, Mapping)
                    or _text(response.get("protection_state")).upper() != "ACTIVE"
                    or not _text(protected.get("stop_loss_order_id"))
                    or not _text(protected.get("take_profit_order_id"))):
                state = "SUBMISSION_UNKNOWN"
            elif (candidate.direction == "BUY" and not candidate.stop_loss < avg_price < candidate.take_profit) or (
                    candidate.direction == "SELL" and not candidate.take_profit < avg_price < candidate.stop_loss):
                state = "SUBMISSION_UNKNOWN"
            elif filled <= 0:
                state = "SUBMISSION_UNKNOWN"
            elif state == "FILLED" and not math.isclose(filled, candidate.quantity, rel_tol=1e-8, abs_tol=1e-8):
                state = "PARTIAL"
        return {"status": state, "authoritative": response.get("authoritative") is True,
                "order_id": _text(response.get("order_id")) or None, "filled_quantity": filled,
                "average_fill_price": avg_price, "protection_state": response.get("protection_state"),
                "protective_exits": response.get("protective_exits"),
                "close_order_id": response.get("close_order_id"),
                "close_client_order_id": response.get("close_client_order_id")}

    def close(self, candidate_id: str, identity: FullIdentity, quantity: float) -> Mapping[str, Any]:
        # Close requests are identity-specific and reduce-only. If authorization
        # or side ownership is unknown, preserve the position and require
        # operator/broker reconciliation rather than guessing a side.
        if not self.authorization_ready():
            return {"status": "CLOSE_PENDING", "authoritative": False}
        direction = self._direction_by_candidate.get(str(candidate_id))
        if direction not in {"BUY", "SELL"}:
            return {"status": "CLOSE_PENDING", "authoritative": False, "reason": "position side is unknown"}
        try:
            qty = int(quantity)
        except (TypeError, ValueError, OverflowError):
            return {"status": "CLOSE_PENDING", "authoritative": False}
        if qty <= 0 or identity.venue != "delta":
            return {"status": "CLOSE_PENDING", "authoritative": False}
        closer = getattr(self.delta, "place_reduce_only_order", None)
        if not callable(closer):
            return {"status": "CLOSE_PENDING", "authoritative": False}
        cid = "jmc" + hashlib.sha256(str(candidate_id).encode()).hexdigest()[:29]
        try:
            response = closer(product_id=int(identity.instrument_id), symbol=identity.symbol,
                              side="sell" if direction == "BUY" else "buy",
                              size=qty, client_order_id=cid)
        except Exception:
            return {"status": "CLOSE_PENDING", "authoritative": False, "client_order_id": cid}
        if not isinstance(response, Mapping) or response.get("success") is not True:
            return {"status": "CLOSE_PENDING", "authoritative": False, "client_order_id": cid,
                    "order_id": _text(response.get("order_id")) or None if isinstance(response, Mapping) else None}
        return {"status": "CLOSE_PENDING", "authoritative": False,
                "order_id": _text(response.get("order_id")) or None, "client_order_id": cid}

    def monitor_once(self, coordinator: PortfolioCoordinator) -> Dict[str, Any]:
        """Reconcile first, then request idempotent reduce-only closes at Delta TP/SL."""
        if not self.reconcile(coordinator):
            return {"status": "RECONCILIATION_BLOCKED", "close_requests": 0}
        close_requests = 0
        rows = coordinator.snapshot().get("orders", {})
        for candidate_id, row in rows.items():
            direction = str(row.get("candidate", {}).get("direction") or "").upper()
            if direction in {"BUY", "SELL"}:
                self._direction_by_candidate[str(candidate_id)] = direction
        for candidate_id, row in rows.items():
            if row.get("status") not in {"FILLED", "PARTIAL"}:
                continue
            raw = row.get("candidate", {})
            identity_raw = raw.get("identity")
            try:
                identity = FullIdentity.parse(identity_raw)
                direction = str(raw.get("direction") or "").upper()
                stop = float(raw["stop_loss"]); target = float(raw["take_profit"])
                quantity = float(row.get("filled_quantity") or 0.0)
                if identity.venue != "delta" or quantity <= 0:
                    continue
                bid, ask, _ = _quote_for(self.delta, identity, self.policy_registry.get(_base_asset(identity.symbol), {}), self.clock())
            except (CandidateRejected, KeyError, TypeError, ValueError, OverflowError):
                continue
            triggered = ((direction == "BUY" and (bid >= target or bid <= stop))
                         or (direction == "SELL" and (ask <= target or ask >= stop)))
            if triggered:
                outcome = coordinator.close_position(candidate_id, identity.to_dict(), self)
                if outcome.get("status") == "CLOSE_PENDING":
                    close_requests += 1
        return {"status": "MONITORED", "close_requests": close_requests}
