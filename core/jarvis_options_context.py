"""Selected-asset options evidence and BTC/ETH macro risk-context policy.

Only complete, identity-matched, fresh provider snapshots can be marked
available. Macro chains are separate risk context; they never replace the
selected asset's primary chain or become its contract/strike source.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class OptionsContext:
    selected_asset: str
    source_asset: str
    role: str                 # asset_primary | unavailable
    available: bool
    bias: str = "NEUTRAL"
    strength: int = 0
    pcr: float | None = None
    reason: str = ""
    expiry_coverage: Dict[str, Any] | None = None
    validation: Dict[str, Any] | None = None
    macro_contexts: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _base(symbol_or_asset: str) -> str:
    value = str(symbol_or_asset or "").upper().replace("-", "").replace("_", "")
    for quote in ("USDT", "USD"):
        if value.endswith(quote) and len(value) > len(quote):
            return value[:-len(quote)]
    return value


def _valid_bias(raw: Any, asset: str) -> OptionsContext | None:
    """Accept only fresh complete chain evidence matching the requested base."""
    if not isinstance(raw, dict):
        return None
    raw_data = raw.get("raw_data") if isinstance(raw.get("raw_data"), dict) else {}
    validation = raw_data.get("options_validation")
    if not isinstance(validation, dict):
        return None
    if not (validation.get("usable") is True and validation.get("complete") is True
            and validation.get("identity_valid") is True and validation.get("fresh") is True
            and validation.get("freshness_status") == "fresh"):
        return None
    if str(validation.get("underlying", "")).upper() != asset:
        return None
    if not isinstance(raw_data.get("source_provider"), str) or not raw_data.get("source_provider"):
        return None
    if raw.get("bias") not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        return None
    score_raw = raw.get("score")
    if isinstance(score_raw, bool):
        return None
    try:
        score_value = float(score_raw)
        pcr_value = float(raw.get("pcr"))
        if not math.isfinite(score_value) or not math.isfinite(pcr_value) or pcr_value < 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    provider_metrics = raw_data.get("provider_metrics")
    if raw.get("pcr") is None or not isinstance(provider_metrics, dict) or provider_metrics.get("open_interest_complete") is not True:
        return None
    reasons = raw.get("reasons")
    if isinstance(reasons, (str, bytes)) or not isinstance(reasons, (list, tuple)):
        return None
    if any(str(reason).strip().lower() == "no data" for reason in reasons):
        return None
    coverage = (raw_data.get("expiry_coverage") or raw_data.get("coverage"))
    return OptionsContext(
        selected_asset=asset,
        source_asset=asset,
        role="asset_primary",
        available=True,
        bias=raw["bias"],
        strength=abs(int(score_value)),
        pcr=pcr_value,
        reason="; ".join(str(x) for x in reasons[:2]),
        expiry_coverage=coverage,
        validation=validation,
    )

def _macro_entry(raw: Any, asset: str) -> Dict[str, Any] | None:
    context = _valid_bias(raw, asset)
    if context is None:
        return None
    return {
        "selected_asset": "macro",
        "source_asset": asset,
        "role": "macro_context",
        "available": True,
        "bias": context.bias,
        "validation": context.validation,
        "provider": (raw.get("raw_data") or {}).get("source_provider"),
        "method": "observed_exchange_options_pcr_summary_only; no unsigned-OI dealer-direction inference",
    }


def resolve_options_context(delta_client: Any, selected_symbol: str) -> OptionsContext:
    """Fetch own-asset primary plus independent BTC/ETH macro chain evidence.

    Missing own-asset data remains unavailable (never replaced by a macro
    contract or score). Macro context is retained separately for a conservative
    deterministic two-asset veto; it is not converted into predictive weights.
    """
    asset = _base(selected_symbol)
    if not asset or delta_client is None:
        return OptionsContext(asset, "", "unavailable", False, reason="No selected asset or options client")

    def fetch(base: str) -> Any:
        try:
            return delta_client.get_institutional_bias(base)
        except Exception:
            return None

    primary = _valid_bias(fetch(asset), asset)
    macros = []
    macro_assets = ("ETH",) if os.getenv("JARVIS_BTC_OPTIONS_CONFIRMATION", "1") == "0" else ("BTC", "ETH")
    for macro_asset in macro_assets:
        if macro_asset == asset:
            continue
        entry = _macro_entry(fetch(macro_asset), macro_asset)
        if entry is not None:
            macros.append(entry)

    if primary is None:
        return OptionsContext(
            selected_asset=asset, source_asset="", role="unavailable", available=False,
            reason="Selected-asset options chain unavailable or failed identity/freshness/completeness validation",
            macro_contexts=macros,
        )
    return OptionsContext(
        selected_asset=asset, source_asset=asset, role="asset_primary", available=True,
        bias=primary.bias, strength=primary.strength, pcr=primary.pcr,
        reason=primary.reason, expiry_coverage=primary.expiry_coverage,
        validation=primary.validation, macro_contexts=macros,
    )


def apply_options_confirmation(direction: str, confidence: int,
                               context: OptionsContext) -> Tuple[int, str]:
    """Apply legacy modest adjustment to validated selected-asset data only."""
    try:
        confidence = max(0, min(100, int(confidence)))
    except (TypeError, ValueError):
        return 0, "Invalid confidence"
    direction = str(direction or "").upper()
    if direction not in {"CALL", "PUT", "BUY", "SELL"} or not context.available or context.role != "asset_primary":
        return confidence, context.reason or "No selected-asset options confirmation"
    wanted = "BULLISH" if direction in {"CALL", "BUY"} else "BEARISH"
    if context.bias == "NEUTRAL":
        return confidence, "asset_primary: neutral options bias"
    delta = 4 if context.bias == wanted else -6
    return max(0, min(100, confidence + delta)), f"asset_primary {context.source_asset} {context.bias}: {delta:+d} confidence"
