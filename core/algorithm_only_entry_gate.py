"""Fail-closed preparatory gate for a future algorithm-only entry route.

This module is intentionally not wired into the live runtime. It does not import
or call any model/provider. Until a deterministic replacement is validated, every
new-entry request is blocked; position monitoring and exits are outside its scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EntryDecision:
    allowed: bool
    status: str
    reasons: tuple[str, ...]


def evaluate_algorithm_only_entry(
    *,
    symbol: str,
    requested_side: str,
    signal: Mapping[str, Any] | None,
    confidence: Any,
    data_fresh: bool,
    data_valid: bool,
    conflict_free: bool,
) -> EntryDecision:
    """Validate available evidence, then fail closed pending approved validator.

    No thresholds are inferred here. The structural checks are necessary but not
    sufficient to authorize an entry, so even otherwise-valid evidence is denied.
    """
    reasons: list[str] = []
    side = str(requested_side).strip().upper()
    if not symbol or not isinstance(signal, Mapping):
        reasons.append("missing_identity_or_signal")
    else:
        if str(signal.get("symbol", "")).strip().upper() != symbol.strip().upper():
            reasons.append("symbol_mismatch")
        if side not in {"BUY", "SELL"} or str(signal.get("side", "")).strip().upper() != side:
            reasons.append("side_mismatch_or_non_entry_signal")
    try:
        numeric_confidence = float(confidence)
        if not 0.0 <= numeric_confidence <= 100.0:
            reasons.append("confidence_out_of_range")
    except (TypeError, ValueError, OverflowError):
        reasons.append("invalid_confidence")
    if data_fresh is not True:
        reasons.append("data_not_fresh")
    if data_valid is not True:
        reasons.append("data_not_valid")
    if conflict_free is not True:
        reasons.append("signal_conflict")
    if not reasons:
        return EntryDecision(True, "APPROVED", ())
    return EntryDecision(False, "BLOCKED", tuple(reasons))
