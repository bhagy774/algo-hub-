"""Fail-closed multicoin candidate validation and portfolio lifecycle.

No exchange implementation lives here. Candidate conversion requires an
explicit plan, an external allow-listed asset policy, and a fresh COMPLETE
Parts 1-12 result. The coordinator is thread-atomic and persists a conservative
journal before adapter submission; UNKNOWN/PARTIAL/FILLED exposure retains its
reservation until a complete authoritative reconciliation proves a terminal
state. Paper/live authorization is external and defaults disabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from jarvis_strategy_approval import (
    PART_WEIGHTS, REQUIRED_TIMEFRAMES, evaluate_mtf_central_strategy,
    validate_entry_approval,
)

_CANONICAL_PARTS = tuple(PART_WEIGHTS)
_PART_ALIASES = {f"part{index}": name for index, name in enumerate(_CANONICAL_PARTS, 1)}


class CandidateRejected(ValueError):
    pass


@dataclass(frozen=True, order=True)
class FullIdentity:
    venue: str
    market_type: str
    instrument_id: str
    symbol: str

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "FullIdentity":
        if not isinstance(raw, Mapping):
            raise CandidateRejected("full instrument identity required")
        vals = [str(raw.get(k) or "").strip() for k in ("venue", "market_type", "instrument_id", "symbol")]
        if any(not x for x in vals):
            raise CandidateRejected("full instrument identity required")
        return cls(vals[0].lower(), vals[1].lower(), vals[2], vals[3].upper())

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected(f"{name} must be explicit numeric data")
    if not math.isfinite(number) or number <= 0:
        raise CandidateRejected(f"{name} must be finite and positive")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected(f"{name} must be explicit numeric data")
    if not math.isfinite(number) or number < 0:
        raise CandidateRejected(f"{name} must be finite and non-negative")
    return number


def _base_asset(symbol: str) -> str:
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[:-len(quote)]
    raise CandidateRejected("symbol has no recognized quote suffix")


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate_id: str
    identity: FullIdentity
    timeframe: str
    snapshot_version: str
    snapshot_fetched_at: float
    analysis_completed_at: float
    decision_timestamp: float
    reference_price_timestamp: float
    reference_price: float
    max_slippage_pct: float
    max_chase_pct: float
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    size_unit: str
    contract_multiplier: float
    sizing_provenance: str
    policy_id: str
    risk_notional: float
    analysis_identity: Optional[FullIdentity] = None
    risk_currency: str = "USDT"
    leverage: int = 1
    entry_authorization: Optional[Mapping[str, Any]] = field(default=None, repr=False, compare=False)

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity * self.contract_multiplier

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row.pop("entry_authorization", None)
        row["identity"] = self.identity.to_dict()
        row["analysis_identity"] = self.analysis_identity.to_dict() if self.analysis_identity else self.identity.to_dict()
        return row


def candidate_from_analysis(
    result: Mapping[str, Any], *, policy_registry: Mapping[str, str],
    mapping_policy_registry: Optional[Mapping[str, str]] = None,
    now: Optional[float] = None, max_age_seconds: float = 180.0,
) -> ValidatedCandidate:
    """Validate an explicit, fully sized plan for a completed Parts 1-12 result.

    ``request_identity`` identifies the analysis provider. ``execution_identity``
    is separate and mandatory when the analysis provider differs from the broker;
    price-source tickers are never silently reinterpreted as Delta contracts.
    ``policy_registry`` is trusted runtime configuration (asset -> policy id).
    """
    if not isinstance(result, Mapping) or result.get("status") != "COMPLETE":
        raise CandidateRejected("only COMPLETE analysis can produce a candidate")
    if (result.get("scope") != "parts1-12-analysis-only" or result.get("analysis_only") is not True
            or result.get("decision_authority") != "none" or result.get("execution_eligible") is not False):
        raise CandidateRejected("analysis-only Parts result has an unexpected authority/scope marker")
    if result.get("freshness_status") not in (None, "FRESH"):
        raise CandidateRejected("analysis result is stale")
    analysis_identity = FullIdentity.parse(result.get("request_identity"))
    plan = result.get("execution_plan")
    identity = FullIdentity.parse(
        (plan.get("execution_identity") if isinstance(plan, Mapping) else None)
        or result.get("execution_identity") or result.get("request_identity")
    )
    if identity.venue not in {"delta", "binance"} or analysis_identity.venue not in {"delta", "binance"}:
        raise CandidateRejected("unsupported analysis or execution venue")
    if analysis_identity != identity:
        if _base_asset(analysis_identity.symbol) != _base_asset(identity.symbol):
            raise CandidateRejected("analysis/execution underlying mismatch")
        if analysis_identity.venue != "binance" or identity.venue != "delta":
            raise CandidateRejected("unsupported cross-venue mapping")
        if not str(plan.get("mapping_policy_id") or "").strip():
            raise CandidateRejected("explicit analysis-to-Delta mapping policy required")
    snapshot_version = str(result.get("snapshot_version") or "").strip()
    if not snapshot_version:
        raise CandidateRejected("snapshot version missing")
    try:
        fetched = float(result["snapshot_fetched_at"])
        completed = float(result["analysis_completed_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise CandidateRejected("analysis timestamps missing")
    current = time.time() if now is None else float(now)
    if not all(math.isfinite(t) for t in (fetched, completed, current)) or fetched > completed or completed > current + 2:
        raise CandidateRejected("analysis timestamps invalid")
    if current - fetched > max(1.0, float(max_age_seconds)):
        raise CandidateRejected("analysis result expired")
    if fetched > completed or completed > current + 2:
        raise CandidateRejected("analysis timestamps are invalid")
    if current - fetched < -2:
        raise CandidateRejected("analysis snapshot is from the future")
    snapshot_version = str(result.get("snapshot_version") or "").strip()
    central_decision = result.get("central_strategy_decision")
    central_evidence = result.get("central_strategy_evidence")
    if not isinstance(central_decision, Mapping) or central_decision.get("approved") is not True:
        raise CandidateRejected("Jarvis central strategy did not approve the entry")
    gate = result.get("part7_gate")
    if not isinstance(gate, Mapping) or gate.get("entry_blocked") is not False or gate.get("risk_veto") is not False:
        raise CandidateRejected("Part7 new-entry gate is missing or blocks entry")
    gate_symbol = str(gate.get("symbol") or "").upper().replace("/", "").replace("-", "").replace("_", "")
    analysis_symbol = analysis_identity.symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    if not gate_symbol or gate_symbol != analysis_symbol:
        raise CandidateRejected("Part7 gate symbol identity is missing or mismatched")
    central_direction = str(central_decision.get("direction") or "").upper()
    try:
        central_confidence = int(round(float(central_decision.get("confidence"))))
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("Jarvis central confidence is missing")
    raw_parts_by_timeframe = result.get("parts_by_timeframe")
    if (not isinstance(raw_parts_by_timeframe, Mapping)
            or set(raw_parts_by_timeframe) != set(REQUIRED_TIMEFRAMES)):
        raise CandidateRejected("complete native 8-timeframe Part evidence is missing")
    expected_symbol = analysis_identity.symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    normalized_parts: Dict[str, Dict[str, Any]] = {}
    for timeframe in REQUIRED_TIMEFRAMES:
        raw_frame = raw_parts_by_timeframe.get(timeframe)
        if not isinstance(raw_frame, Mapping):
            raise CandidateRejected(f"{timeframe} Part evidence is malformed")
        frame: Dict[str, Any] = {}
        for alias, canonical in _PART_ALIASES.items():
            item = raw_frame.get(canonical, raw_frame.get(alias))
            if not isinstance(item, Mapping):
                raise CandidateRejected(f"{timeframe} {canonical} evidence is missing")
            item_symbol = str(item.get("symbol") or item.get("selected_symbol") or "").upper().replace("/", "").replace("-", "").replace("_", "")
            item_timeframe = str(item.get("timeframe") or item.get("native_timeframe") or "")
            if item_symbol != expected_symbol or item_timeframe != timeframe:
                raise CandidateRejected(f"{timeframe} {canonical} symbol/timeframe identity mismatch")
            frame[canonical] = dict(item)
        normalized_parts[timeframe] = frame
    recomputed = evaluate_mtf_central_strategy(
        normalized_parts, gate, confidence=central_confidence,
        expected_symbol=analysis_identity.symbol,
    )
    if (not recomputed.get("approved") or recomputed.get("direction") != central_direction
            or recomputed.get("trade_mode") != central_decision.get("trade_mode")):
        raise CandidateRejected("central native-timeframe evidence or Part7 gate does not approve this mode/direction")
    approval = result.get("central_strategy_approval")
    if not isinstance(approval, Mapping):
        raise CandidateRejected("scope-bound Jarvis central strategy approval is missing")
    if (str(approval.get("analysis_symbol") or "").upper() != analysis_identity.symbol
            or str(approval.get("analysis_exchange") or "").lower() != analysis_identity.venue):
        raise CandidateRejected("Jarvis approval analysis identity does not match the source instrument")
    source_plan = (result.get("central_execution_plan") or result.get("jarvis_execution_plan")
                   or (result.get("execution_plan") if isinstance(result.get("execution_plan"), Mapping)
                       and result.get("execution_plan", {}).get("schema_version") == "jarvis-execution-plan-v1" else None))
    bound_plan = bool(str(approval.get("execution_plan_id") or ""))
    if bound_plan and not isinstance(source_plan, Mapping):
        raise CandidateRejected("Jarvis-bound central execution plan is missing")
    approval_ok, approval_reason = validate_entry_approval(
        approval, direction=central_direction, symbol=identity.symbol, exchange=identity.venue,
        contract=identity.symbol, instrument_id=identity.instrument_id,
        market_type=identity.market_type, snapshot_version=snapshot_version,
        confidence=central_confidence, execution_plan=source_plan if bound_plan else None,
        now=current, max_age_seconds=max_age_seconds,
    )
    if not approval_ok:
        raise CandidateRejected(approval_reason)
    try:
        approved_at = float(approval.get("analysis_timestamp"))
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("Jarvis approval timestamp is malformed")
    if approved_at < fetched or approved_at > completed + 2:
        raise CandidateRejected("Jarvis approval timestamp is outside the analysis interval")
    if not isinstance(plan, Mapping):
        raise CandidateRejected("explicit execution plan missing; no inferred entry/size")
    tf = str(plan.get("timeframe") or "").strip()
    if tf not in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"}:
        raise CandidateRejected("unsupported/missing decision timeframe")
    decision_at = _positive_number(plan.get("decision_timestamp"), "decision_timestamp")
    if decision_at < fetched or decision_at > completed + 2 or current - decision_at > max(1.0, float(max_age_seconds)):
        raise CandidateRejected("decision timestamp invalid or stale")
    # A closed Binance candle normally timestamps its open, so its valid close
    # can precede snapshot retrieval. Freshness and the completed-time upper
    # bound are authoritative; requiring timestamp >= fetch would reject it.
    reference_at = _positive_number(plan.get("reference_price_timestamp"), "reference_price_timestamp")
    if reference_at > completed + 2 or current - reference_at > max(1.0, float(max_age_seconds)):
        raise CandidateRejected("reference price timestamp invalid or stale")
    reference_price = _positive_number(plan.get("reference_price"), "reference_price")
    max_slippage_pct = _nonnegative_number(plan.get("max_slippage_pct"), "max_slippage_pct")
    max_chase_pct = _nonnegative_number(plan.get("max_chase_pct"), "max_chase_pct")
    direction = str(plan.get("direction") or "").strip().upper().replace("-", "_").replace(" ", "_")
    if direction in {"NO_TRADE", "NOTRADE", "HOLD", "NONE"}:
        raise CandidateRejected("NO TRADE is not an executable candidate")
    if direction not in {"BUY", "SELL"}:
        raise CandidateRejected("direction must be BUY or SELL")
    if direction != central_direction:
        raise CandidateRejected("execution plan direction differs from Jarvis central approval")
    entry = _positive_number(plan.get("entry_price"), "entry_price")
    stop = _positive_number(plan.get("stop_loss"), "stop_loss")
    target = _positive_number(plan.get("take_profit"), "take_profit")
    qty = _positive_number(plan.get("quantity"), "quantity")
    if direction == "BUY" and not stop < entry < target:
        raise CandidateRejected("BUY stop/entry/target geometry invalid")
    if direction == "SELL" and not target < entry < stop:
        raise CandidateRejected("SELL stop/entry/target geometry invalid")
    chase_pct = abs(entry - reference_price) / reference_price * 100.0
    if chase_pct > max_chase_pct:
        raise CandidateRejected("entry exceeds explicit max chase from Binance reference")
    size_unit = str(plan.get("size_unit") or "").strip().lower()
    provenance = str(plan.get("sizing_provenance") or "").strip()
    if size_unit == "base_asset_quantity":
        contract_multiplier = 1.0
    elif size_unit == "contracts":
        contract_multiplier = _positive_number(plan.get("contract_multiplier"), "contract_multiplier")
    else:
        raise CandidateRejected("explicit supported size unit required")
    if not provenance:
        raise CandidateRejected("sizing provenance required")
    risk_notional = _positive_number(plan.get("risk_notional"), "risk_notional")
    risk_currency = str(plan.get("risk_currency") or "USDT").strip().upper()
    if not risk_currency or len(risk_currency) > 12:
        raise CandidateRejected("explicit risk currency is malformed")
    raw_leverage = plan.get("leverage", 1)
    try:
        leverage = int(raw_leverage)
    except (TypeError, ValueError, OverflowError):
        raise CandidateRejected("integer leverage plan is required")
    if leverage < 1 or leverage > 1000 or float(raw_leverage) != leverage:
        raise CandidateRejected("leverage must be an integer within venue bounds")
    stop_risk = abs(entry - stop) * qty * contract_multiplier
    if risk_notional > stop_risk + max(1e-8, stop_risk * 1e-6):
        raise CandidateRejected("declared risk exceeds stop-distance exposure")
    asset = _base_asset(identity.symbol)
    policy_id = str(policy_registry.get(asset) or "").strip()
    if not policy_id or str(plan.get("policy_id") or "").strip() != policy_id:
        raise CandidateRejected("no matching approved per-asset policy")
    if identity.venue == "delta" and identity != analysis_identity:
        allowed_mapping_id = str((mapping_policy_registry or {}).get(asset) or "").strip()
        if not allowed_mapping_id or str(plan.get("mapping_policy_id") or "").strip() != allowed_mapping_id:
            raise CandidateRejected("analysis-to-Delta mapping policy is not allow-listed")
    # Stable across quote refreshes: one decision/snapshot can submit at most once.
    raw_id = "|".join((analysis_identity.venue, analysis_identity.market_type, analysis_identity.instrument_id,
                       analysis_identity.symbol, identity.venue, identity.market_type, identity.instrument_id,
                       identity.symbol, tf, snapshot_version, direction, str(decision_at), policy_id))
    candidate_id = hashlib.sha256(raw_id.encode()).hexdigest()
    entry_authorization = {
        "central_approval": dict(approval),
        "part_results": dict(central_evidence) if isinstance(central_evidence, Mapping) else None,
        "parts_by_timeframe": normalized_parts,
        "part7_gate": dict(gate),
        "snapshot_version": snapshot_version,
        "analysis_symbol": analysis_identity.symbol,
        "confidence": central_confidence,
        "strategy_plan": dict(source_plan) if isinstance(source_plan, Mapping) else None,
        "broker_plan": {
            "direction": direction, "symbol": identity.symbol,
            "instrument_id": identity.instrument_id, "quantity": qty,
            "leverage": leverage, "entry_price": entry,
            "stop_loss": stop, "take_profit": target,
            "risk_budget_usdt": risk_notional if risk_currency == "USDT" else 0,
            "trade_mode": str((source_plan or {}).get("trade_mode") or ""),
        },
    }
    return ValidatedCandidate(candidate_id, identity, tf, snapshot_version, fetched, completed,
                              decision_at, reference_at, reference_price, max_slippage_pct, max_chase_pct,
                              direction, entry, stop, target, qty, size_unit, contract_multiplier,
                              provenance, policy_id, risk_notional, analysis_identity,
                              risk_currency, leverage, entry_authorization)


class PaperExecutionAdapter(Protocol):
    def submit(self, candidate: ValidatedCandidate, idempotency_key: str) -> Mapping[str, Any]: ...
    def close(self, candidate_id: str, identity: FullIdentity, quantity: float) -> Mapping[str, Any]: ...


class DeterministicPaperAdapter:
    """No-network paper adapter; assumes exact requested entry (no slippage model)."""
    def submit(self, candidate: ValidatedCandidate, idempotency_key: str) -> Mapping[str, Any]:
        return {"status": "FILLED", "authoritative": True,
                "order_id": "paper-" + idempotency_key[:20],
                "filled_quantity": candidate.quantity, "model": "requested-price-zero-slippage"}

    def close(self, candidate_id: str, identity: FullIdentity, quantity: float) -> Mapping[str, Any]:
        return {"status": "CLOSED", "authoritative": True,
                "order_id": "paper-close-" + str(candidate_id)[:20], "filled_quantity": quantity}


_ACTIVE = {"RESERVED", "SUBMISSION_UNKNOWN", "PARTIAL", "FILLED", "CLOSE_PENDING"}
_TERMINAL = {"REJECTED", "CLOSED"}
_VALID_STATES = _ACTIVE | _TERMINAL


class PortfolioCoordinator:
    """Bounded, lock-serialized portfolio with restart-safe journal.

    Submission is deliberately separate from authorization: this coordinator
    accepts only ValidatedCandidate objects and is disabled by default. Orders
    are retained on timeout/unknown, and per-position close never sweeps peers.
    Its adapter may be a no-network paper model or a separately authorized
    broker adapter that meets the strict reconciliation/protection contract.
    """
    def __init__(self, *, journal_path: str, enabled: bool = False,
                 max_positions: int = 2, max_total_notional: float, max_total_risk: float,
                 per_asset_caps: Optional[Mapping[str, int]] = None,
                 correlation_groups: Optional[Mapping[str, str]] = None,
                 max_per_correlation_group: int = 1, clock: Callable[[], float] = time.time,
                 require_reconciliation: bool = False,
                 require_protective_reconciliation: bool = False):
        self.enabled = bool(enabled)
        self.journal_path = Path(journal_path)
        self.max_positions = max(1, int(max_positions))
        self.max_total_notional = _positive_number(max_total_notional, "max_total_notional")
        self.max_total_risk = _positive_number(max_total_risk, "max_total_risk")
        self.per_asset_caps = {str(k).upper(): max(1, int(v)) for k, v in (per_asset_caps or {}).items()}
        self.correlation_groups = {str(k).upper(): str(v) for k, v in (correlation_groups or {}).items()}
        self.max_per_correlation_group = max(1, int(max_per_correlation_group))
        self.clock = clock
        self.require_reconciliation = bool(require_reconciliation)
        self.require_protective_reconciliation = bool(require_protective_reconciliation)
        self._reconciled = not self.require_reconciliation
        self._lock = threading.RLock()
        self.healthy = True
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._external_positions: Dict[str, Dict[str, float]] = {}
        self._load()

    @staticmethod
    def _identity_key(identity: FullIdentity) -> str:
        return "|".join((identity.venue, identity.market_type, identity.instrument_id, identity.symbol))

    def _load(self) -> None:
        try:
            if self.journal_path.exists():
                raw = json.loads(self.journal_path.read_text(encoding="utf-8"))
                if raw.get("schema_version") != 1 or not isinstance(raw.get("orders"), dict) or not isinstance(raw.get("external_positions", {}), dict):
                    raise ValueError("bad journal schema")
                orders = raw["orders"]
                for key, item in orders.items():
                    if not isinstance(item, dict) or item.get("status") not in _VALID_STATES:
                        raise ValueError("bad journal order")
                    if key != item.get("candidate_id") or not isinstance(item.get("candidate"), dict):
                        raise ValueError("bad journal identity")
                    FullIdentity.parse(item["candidate"].get("identity"))
                    _positive_number(item.get("reserved_notional"), "journal reserved notional")
                    _positive_number(item.get("reserved_risk"), "journal reserved risk")
                external = {}
                for key, value in raw.get("external_positions", {}).items():
                    if not isinstance(value, dict):
                        raise ValueError("external position risk/notional required")
                    currency = str(value.get("risk_currency") or "USDT").strip().upper()
                    if not currency or len(currency) > 12:
                        raise ValueError("external position currency required")
                    external[str(key)] = {"notional": _positive_number(value.get("notional"), "external notional"),
                                          "risk_notional": _positive_number(value.get("risk_notional"), "external risk"),
                                          "risk_currency": currency}
                self._orders, self._external_positions = orders, external
        except Exception:
            # Corrupt state means no orders, no opens, and no new admission.
            self.healthy = False
            self._orders, self._external_positions = {}, {}

    def _save(self) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"schema_version": 1, "orders": self._orders,
                              "external_positions": self._external_positions}, sort_keys=True, separators=(",", ":"))
        fd, tmp = tempfile.mkstemp(prefix=".multicoin-", suffix=".tmp", dir=str(self.journal_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.journal_path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def _capacity(self, candidate: ValidatedCandidate) -> bool:
        active = [x for x in self._orders.values() if x["status"] in _ACTIVE]
        external = self._external_positions
        currencies = {str(x.get("candidate", {}).get("risk_currency") or "USDT").upper() for x in active}
        currencies.update(str(x.get("risk_currency") or "USDT").upper() for x in external.values())
        if currencies and currencies != {str(candidate.risk_currency).upper()}:
            return False
        identity_key = self._identity_key(candidate.identity)
        if any(FullIdentity.parse(x["candidate"]["identity"]) == candidate.identity for x in active):
            return False
        if identity_key in external:
            return False
        count = len(active) + len(external)
        total = sum(float(x["reserved_notional"]) for x in active) + sum(float(v["notional"]) for v in external.values())
        risk = sum(float(x["reserved_risk"]) for x in active) + sum(float(v["risk_notional"]) for v in external.values())
        if (count >= self.max_positions or total + candidate.notional > self.max_total_notional
                or risk + candidate.risk_notional > self.max_total_risk):
            return False
        asset = _base_asset(candidate.identity.symbol)
        asset_count = sum(1 for x in active if _base_asset(FullIdentity.parse(x["candidate"]["identity"]).symbol) == asset)
        asset_count += sum(1 for k in external if _base_asset(k.split("|", 3)[-1]) == asset)
        if asset in self.per_asset_caps and asset_count >= self.per_asset_caps[asset]:
            return False
        group = self.correlation_groups.get(asset)
        if group:
            group_count = sum(1 for x in active if self.correlation_groups.get(_base_asset(FullIdentity.parse(x["candidate"]["identity"]).symbol)) == group)
            group_count += sum(1 for k in external if self.correlation_groups.get(_base_asset(k.split("|", 3)[-1])) == group)
            if group_count >= self.max_per_correlation_group:
                return False
        return True

    def submit(self, candidate: ValidatedCandidate, adapter: PaperExecutionAdapter) -> Dict[str, Any]:
        with self._lock:
            if not self.enabled or not self.healthy:
                return {"status": "BLOCKED", "reason": "coordinator disabled or journal unhealthy"}
            # A journaled candidate is an idempotent no-op even while restart
            # reconciliation is pending. This lookup never submits or releases
            # its reservation; new identities remain blocked until reconciliation.
            old = self._orders.get(candidate.candidate_id)
            if old:
                return {"status": old["status"], "candidate_id": candidate.candidate_id, "duplicate": True}
            if self.require_reconciliation and not self._reconciled:
                return {"status": "BLOCKED", "reason": "authoritative account reconciliation required before submission"}
            if not self._capacity(candidate):
                return {"status": "BLOCKED", "reason": "portfolio/correlation capacity exceeded"}
            row = {"candidate_id": candidate.candidate_id, "candidate": candidate.to_dict(),
                   "status": "RESERVED", "reserved_notional": candidate.notional,
                   "reserved_risk": candidate.risk_notional, "created_at": float(self.clock()), "order_id": None, "close_requested": False}
            self._orders[candidate.candidate_id] = row
            try:
                self._save()  # durable reservation before external side effect
            except Exception:
                self._orders.pop(candidate.candidate_id, None)
                self.healthy = False
                return {"status": "BLOCKED", "reason": "reservation journal write failed"}
            try:
                response = adapter.submit(candidate, candidate.candidate_id)
                if not isinstance(response, Mapping):
                    raise ValueError("malformed adapter response")
            except Exception:
                response = {"status": "SUBMISSION_UNKNOWN", "authoritative": False}
            state = str(response.get("status") or "SUBMISSION_UNKNOWN").upper()
            if state not in {"SUBMISSION_UNKNOWN", "PARTIAL", "FILLED", "REJECTED"}:
                state = "SUBMISSION_UNKNOWN"
            row["status"] = state
            row["order_id"] = str(response.get("order_id") or "") or None
            row["last_response_authoritative"] = response.get("authoritative") is True
            try:
                filled_quantity = float(response.get("filled_quantity", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                filled_quantity = float("nan")
            if not math.isfinite(filled_quantity) or filled_quantity < 0 or filled_quantity > candidate.quantity:
                state, filled_quantity = "SUBMISSION_UNKNOWN", 0.0
            if state in {"FILLED", "PARTIAL"} and not row["last_response_authoritative"]:
                state = "SUBMISSION_UNKNOWN"
            if state == "FILLED" and filled_quantity < candidate.quantity:
                state = "PARTIAL" if filled_quantity > 0 else "SUBMISSION_UNKNOWN"
            row["filled_quantity"] = filled_quantity
            try:
                avg_fill = float(response.get("average_fill_price", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                avg_fill = 0.0
            if math.isfinite(avg_fill) and avg_fill > 0:
                row["average_fill_price"] = avg_fill
            if response.get("protection_state") is not None:
                row["protection_state"] = str(response.get("protection_state"))
            if isinstance(response.get("protective_exits"), Mapping):
                row["protective_exits"] = dict(response["protective_exits"])
            if str(response.get("close_order_id") or "").strip():
                row["close_order_id"] = str(response.get("close_order_id")).strip()
            if str(response.get("close_client_order_id") or "").strip():
                row["close_client_order_id"] = str(response.get("close_client_order_id")).strip()
            if state == "REJECTED":
                # Even a submit acknowledgement is not the authoritative account/order
                # reconciliation required to release capacity.
                row["reported_rejection"] = bool(row["last_response_authoritative"])
                row["status"] = "SUBMISSION_UNKNOWN"
            try:
                self._save()
            except Exception:
                self.healthy = False
                row["status"] = "SUBMISSION_UNKNOWN"
            return {"status": row["status"], "candidate_id": candidate.candidate_id, "order_id": row["order_id"]}

    def reconcile(self, snapshot: Mapping[str, Any]) -> bool:
        """Apply only complete full-identity account snapshots and explicit order states."""
        if not isinstance(snapshot, Mapping) or snapshot.get("complete") is not True:
            return False
        positions_raw = snapshot.get("positions")
        orders_raw = snapshot.get("orders")
        if not isinstance(positions_raw, Sequence) or isinstance(positions_raw, (str, bytes)) or not isinstance(orders_raw, Mapping):
            return False
        try:
            as_of = float(snapshot.get("as_of", self.clock()))
            if not math.isfinite(as_of):
                return False
            positions: Dict[str, Dict[str, float]] = {}
            for item in positions_raw:
                identity = FullIdentity.parse(item)
                _base_asset(identity.symbol)
                notional = _positive_number(item.get("notional"), "position notional")
                risk_notional = _positive_number(item.get("risk_notional"), "position risk notional")
                key = self._identity_key(identity)
                if key in positions:
                    return False
                currency = str(item.get("risk_currency") or "USDT").strip().upper()
                if not currency or len(currency) > 12:
                    return False
                positions[key] = {"notional": notional, "risk_notional": risk_notional,
                                  "risk_currency": currency}
            order_states = {}
            for cid, detail in orders_raw.items():
                if not isinstance(detail, Mapping) or detail.get("authoritative") is not True:
                    return False
                state = str(detail.get("status") or "").upper()
                if state not in {"SUBMISSION_UNKNOWN", "PARTIAL", "FILLED", "REJECTED", "CLOSED"}:
                    return False
                order_states[str(cid)] = (state, detail)
        except (CandidateRejected, TypeError, ValueError, AttributeError):
            return False
        with self._lock:
            new_orders = json.loads(json.dumps(self._orders))
            currencies = {str(value.get("risk_currency") or "USDT").upper() for value in positions.values()}
            currencies.update(str(row.get("candidate", {}).get("risk_currency") or "USDT").upper()
                              for row in new_orders.values() if row.get("status") in _ACTIVE)
            if len(currencies) > 1:
                return False
            for cid, row in new_orders.items():
                identity = FullIdentity.parse(row["candidate"]["identity"])
                ikey = self._identity_key(identity)
                update = order_states.get(cid)
                present = ikey in positions
                if self.require_protective_reconciliation and present and row.get("status") in {"FILLED", "PARTIAL", "CLOSE_PENDING"}:
                    if update is None:
                        return False
                    detail = update[1]
                    exits = detail.get("protective_exits")
                    if (str(detail.get("protection_state") or "").upper() != "ACTIVE"
                            or not isinstance(exits, Mapping)
                            or not str(exits.get("stop_loss_order_id") or "").strip()
                            or not str(exits.get("take_profit_order_id") or "").strip()):
                        return False
                if update:
                    state, detail = update
                    if state in {"FILLED", "PARTIAL"} and not present:
                        return False  # account/open-position mismatch: preserve previous ledger
                    if state == "CLOSED" and present:
                        return False
                    if state == "REJECTED" and present:
                        return False
                    row["status"] = state
                    if state == "CLOSED":
                        row["closed_at"] = as_of
                    if "filled_quantity" in detail:
                        quantity = float(detail["filled_quantity"])
                        planned = float(row["candidate"]["quantity"])
                        if not math.isfinite(quantity) or quantity < 0 or quantity > planned:
                            return False
                        if state == "FILLED" and quantity + 1e-9 < planned:
                            return False
                        if state == "PARTIAL" and not 0 < quantity < planned:
                            return False
                        row["filled_quantity"] = quantity
                elif row["status"] in {"FILLED", "PARTIAL", "CLOSE_PENDING"} and not present:
                    # A full snapshot proves closure but does not account for unknown orders.
                    row["status"] = "CLOSED"
                    row["closed_at"] = as_of
                elif row["status"] in {"FILLED", "PARTIAL"} and present:
                    pass
                elif row["status"] == "SUBMISSION_UNKNOWN":
                    pass  # absence does not prove rejection; reservation remains
            local_keys = {self._identity_key(FullIdentity.parse(r["candidate"]["identity"])) for r in new_orders.values() if r["status"] in _ACTIVE}
            external = {key: value for key, value in positions.items() if key not in local_keys}
            old_orders, old_external = self._orders, self._external_positions
            self._orders, self._external_positions = new_orders, external
            try:
                self._save()
            except Exception:
                self._orders, self._external_positions = old_orders, old_external
                self.healthy = False
                return False
            self._reconciled = True
            return True

    def close_position(self, candidate_id: str, identity: Mapping[str, Any], adapter: PaperExecutionAdapter) -> Dict[str, Any]:
        with self._lock:
            if not self.enabled or not self.healthy:
                return {"status": "BLOCKED"}
            row = self._orders.get(str(candidate_id))
            if not row or row["status"] not in {"FILLED", "PARTIAL"}:
                return {"status": "BLOCKED", "reason": "position not open/owned"}
            expected = FullIdentity.parse(row["candidate"]["identity"])
            if FullIdentity.parse(identity) != expected:
                return {"status": "BLOCKED", "reason": "position identity mismatch"}
            if row.get("close_requested"):
                return {"status": "CLOSE_PENDING", "duplicate": True}
            previous_status = row["status"]
            row["status"] = "CLOSE_PENDING"
            row["close_requested"] = True
            try:
                self._save()
            except Exception:
                row["status"] = previous_status
                row["close_requested"] = False
                self.healthy = False
                return {"status": "BLOCKED", "reason": "close journal write failed"}
            try:
                response = adapter.close(str(candidate_id), expected, float(row.get("filled_quantity") or 0.0))
                if not isinstance(response, Mapping):
                    raise ValueError("malformed close response")
            except Exception:
                response = {"status": "CLOSE_PENDING"}
            if str(response.get("order_id") or "").strip():
                row["close_order_id"] = str(response.get("order_id")).strip()
            if str(response.get("client_order_id") or "").strip():
                row["close_client_order_id"] = str(response.get("client_order_id")).strip()
            if str(response.get("status", "")).upper() == "REJECTED" and response.get("authoritative") is True:
                row["status"] = "FILLED" if row.get("filled_quantity") else "PARTIAL"
                row["close_requested"] = False
            try:
                self._save()
            except Exception:
                self.healthy = False
                return {"status": "CLOSE_PENDING"}
            return {"status": row["status"], "candidate_id": candidate_id}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "healthy": self.healthy,
                    "ready": bool(self.enabled and self.healthy and self._reconciled),
                    "orders": json.loads(json.dumps(self._orders)),
                    "external_positions": dict(self._external_positions)}
