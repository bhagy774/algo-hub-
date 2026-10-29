"""Strict, offline-testable normalization and analysis for listed options.

Provider values are preserved as provider observations; this module never
manufactures quotes or Greeks. Cross-venue totals are intentionally avoided.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import math
import re
from statistics import median
from typing import Any, Iterable, Mapping

GREEK_FIELDS = ("delta", "gamma", "theta", "vega", "rho")
CHAIN_FIELDS = ("bid", "ask", "open_interest", "volume", "implied_volatility", *GREEK_FIELDS)
DEFAULT_MAX_AGE_SECONDS = 600


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_instant(value: Any) -> datetime | None:
    """Parse ISO-8601 or Unix timestamps (seconds or milliseconds), UTC-aware only."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                return None
            if number > 10_000_000_000:
                number /= 1000.0
            return datetime.fromtimestamp(number, tz=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"\d{10,13}", text):
            return parse_instant(int(text))
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _first(raw: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
    return None


def _expiry_date(raw: Mapping[str, Any], symbol_parts: list[str], venue: str) -> date | None:
    candidate = _first(raw, "expiry", "expiry_date", "expiryDate", "delivery_date", "deliveryDate")
    from_symbol = candidate is None
    if from_symbol and len(symbol_parts) >= 2:
        candidate = symbol_parts[1]
    if isinstance(candidate, (int, float)):
        parsed = parse_instant(candidate)
        return parsed.date() if parsed else None
    text = str(candidate or "").strip().upper()
    formats = ["%Y-%m-%d", "%Y%m%d", "%d%b%y", "%d%b%Y"]
    # Binance option symbols use YYMMDD. A six-digit Delta date is ambiguous
    # without explicit expiry metadata, so reject it rather than invent a year.
    if re.fullmatch(r"\d{6}", text):
        if "binance" in str(venue).lower() or not from_symbol:
            formats.append("%y%m%d")
        else:
            return None
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    parsed = parse_instant(candidate)
    return parsed.date() if parsed else None


def normalize_contract(raw: Mapping[str, Any], underlying: str, venue: str,
                       as_of: datetime | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize one provider row, returning (contract, rejection reason)."""
    if not isinstance(raw, Mapping):
        return None, "row_not_object"
    as_of = (as_of or utc_now()).astimezone(timezone.utc)
    target = str(underlying or "").upper().replace("USDT", "").replace("USD", "")
    symbol = str(_first(raw, "symbol", "instrument_name", "instrumentName") or "").strip()
    parts = symbol.upper().split("-") if symbol else []
    row_underlying = str(_first(raw, "underlying_asset_symbol", "underlying", "underlyingAsset", "baseAsset") or "").upper()
    if not row_underlying and parts:
        row_underlying = parts[0]
    row_underlying = row_underlying.replace("USDT", "").replace("USD", "")
    if not symbol or not target or row_underlying != target:
        return None, "underlying_identity_mismatch" if symbol else "missing_contract_identity"

    side_raw = str(_first(raw, "contract_type", "side", "option_type", "optionType") or "").lower()
    if "call" in side_raw or side_raw in {"c", "call_options"} or (parts and parts[-1] in {"C", "CALL"}):
        option_type = "call"
    elif "put" in side_raw or side_raw in {"p", "put_options"} or (parts and parts[-1] in {"P", "PUT"}):
        option_type = "put"
    else:
        return None, "missing_or_invalid_option_type"

    strike = _finite(_first(raw, "strike_price", "strikePrice", "strike"))
    if strike is None and len(parts) >= 3:
        strike = _finite(parts[-2])
    expiry = _expiry_date(raw, parts, venue)
    if strike is None or strike <= 0:
        return None, "missing_or_invalid_strike"
    if expiry is None:
        return None, "missing_or_invalid_expiry"
    if expiry < as_of.date():
        return None, "expired_contract"
    if expiry == as_of.date():
        # Contract timestamps expose dates but not a reliable settlement instant.
        # Exclude same-day contracts in both live chain scoring and historical replay.
        return None, "expired_or_same_day_expiry_unverified"

    greeks_raw = raw.get("greeks") if isinstance(raw.get("greeks"), Mapping) else {}
    values: dict[str, float | None] = {}
    aliases = {
        "bid": ("best_bid", "best_bid_price", "bid_price", "bidPrice", "bid"),
        "ask": ("best_ask", "best_ask_price", "ask_price", "askPrice", "ask"),
        "open_interest": ("oi", "open_interest", "openInterest"),
        "volume": ("volume", "volume_24h", "volume24h"),
        "implied_volatility": ("implied_volatility", "mark_iv", "markIV", "mark_iv_percent"),
    }
    for field, names in aliases.items():
        values[field] = _finite(_first(raw, *names))
    for greek in GREEK_FIELDS:
        values[greek] = _finite(_first(greeks_raw, greek) if greeks_raw else raw.get(greek))
    for field in ("open_interest", "volume", "bid", "ask", "implied_volatility"):
        if values[field] is not None and values[field] < 0:
            values[field] = None

    mark_price = _finite(_first(raw, "mark_price", "markPrice", "price"))
    contract_multiplier = _finite(_first(raw, "contract_multiplier", "contractMultiplier", "contract_value", "contractValue"))
    if contract_multiplier is not None and contract_multiplier <= 0:
        contract_multiplier = None
    observed_at = parse_instant(_first(raw, "timestamp", "updated_at", "updatedAt", "last_update_time", "time", "closeTime"))
    greeks_source = "exchange_observed" if any(values[g] is not None for g in GREEK_FIELDS) else "unavailable"
    missing = [field for field in CHAIN_FIELDS if values[field] is None]
    contract = {
        "symbol": symbol,
        "underlying": target,
        "expiry": expiry.isoformat(),
        "strike": strike,
        "type": option_type,
        "venue": str(venue),
        "mark_price": mark_price,
        "price": mark_price,
        "contract_multiplier": contract_multiplier,
        **values,
        # Compatibility aliases retain None rather than replacing missing data with zero.
        "oi": values["open_interest"],
        "iv": values["implied_volatility"],
        "observed_at": observed_at.isoformat() if observed_at else None,
        "greeks_source": greeks_source,
        "greeks_model_derived": False,
        "model_assumptions": None,
        "missing_fields": missing,
        "units": {
            "price": "provider_native_unverified",
            "bid": "provider_native_unverified",
            "ask": "provider_native_unverified",
            "open_interest": "provider_native_unverified",
            "volume": "provider_native_unverified",
            "implied_volatility": "provider_native_unverified",
            "greeks": "provider_native_unverified",
        },
    }
    return contract, None


def _contract_coverage(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    greek_coverage = {g: sum(c.get(g) is not None for c in contracts) for g in GREEK_FIELDS}
    field_coverage = {field: sum(c.get(field) is not None for c in contracts) for field in CHAIN_FIELDS}
    by_expiry: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "puts": 0, "strikes": set()})
    missing = Counter()
    for contract in contracts:
        expiry = contract["expiry"]
        item = by_expiry[expiry]
        item["calls" if contract["type"] == "call" else "puts"] += 1
        item["strikes"].add(contract["strike"])
        missing.update(contract.get("missing_fields", []))
    return {
        "contract_count": len(contracts),
        "by_expiry": {e: {"calls": v["calls"], "puts": v["puts"], "strike_count": len(v["strikes"]), "strikes": sorted(v["strikes"])} for e, v in sorted(by_expiry.items())},
        "field_observed_count": field_coverage,
        "greek_observed_count": greek_coverage,
        "missing_field_count": dict(sorted(missing.items())),
        "greeks_model_derived": False,
    }


def payout_max_pain(contracts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute intrinsic payout minimum independently for each expiration.

    Expiries are never combined because their settlement payoffs occur at
    different times. Top-level ``strike`` is only the nearest listed expiry's
    result; the full per-expiry availability/results are retained. OI is treated
    as contract count. A unit multiplier is used only when *all* contracts in
    that expiry omit the multiplier; loss is normalized underlying-price units,
    never currency. Unsigned OI is not dealer direction.
    """
    rows = [c for c in contracts if isinstance(c, Mapping)]
    if not rows or any(not c.get("expiry") for c in rows):
        return {"available": False, "reason": "valid_expiry_required", "method": "payout_minimizer_v1"}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["expiry"])].append(row)

    def one_expiry(expiry_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        if any(_finite(c.get("open_interest")) is None or _finite(c.get("open_interest")) < 0
               or _finite(c.get("strike")) is None or _finite(c.get("strike")) <= 0
               or c.get("type") not in {"call", "put"} for c in expiry_rows):
            return {"available": False, "reason": "complete_valid_open_interest_strike_and_option_type_required"}
        multipliers = [_finite(c.get("contract_multiplier")) for c in expiry_rows]
        if any(value is not None and value <= 0 for value in multipliers):
            return {"available": False, "reason": "invalid_contract_multiplier"}
        if any(value is not None for value in multipliers) and any(value is None for value in multipliers):
            return {"available": False, "reason": "mixed_known_and_missing_contract_multipliers"}
        strikes = sorted({_finite(c.get("strike")) for c in expiry_rows})
        payouts = []
        for settlement in strikes:
            total = 0.0
            for c in expiry_rows:
                oi = _finite(c.get("open_interest"))
                strike = _finite(c.get("strike"))
                multiplier = _finite(c.get("contract_multiplier")) or 1.0
                if c.get("type") == "call":
                    total += oi * multiplier * max(settlement - strike, 0.0)
                else:
                    total += oi * multiplier * max(strike - settlement, 0.0)
            payouts.append((total, settlement))
        min_loss = min(value for value, _ in payouts)
        best = min(strike for value, strike in payouts if math.isclose(value, min_loss, rel_tol=1e-12, abs_tol=1e-12))
        return {"available": True, "strike": best, "normalized_payout": min_loss,
                "contract_count": len(expiry_rows), "strike_count": len(strikes)}

    by_expiry = {expiry: one_expiry(grouped[expiry]) for expiry in sorted(grouped)}
    selected_expiry = sorted(grouped)[0]
    selected = by_expiry[selected_expiry]
    return {
        "available": bool(selected.get("available")),
        "strike": selected.get("strike"),
        "normalized_payout": selected.get("normalized_payout"),
        "selected_expiry": selected_expiry,
        "by_expiry": by_expiry,
        "payout_unit": "OI_contracts x underlying_price_units (unit multiplier=1 only when all contracts in expiry omit it)",
        "method": "per_expiry argmin_s_over_listed_strikes_sum(OI*max(S-K,0) calls + OI*max(K-S,0) puts); nearest-expiry strike exposed at top level",
        "assumptions": ["listed strikes are candidate settlement prices per expiry", "unit multiplier=1 for an expiry only if all its provider multiplier values are missing", "unsigned open interest gives no dealer positioning direction"],
        "reason": selected.get("reason"),
    }


def _iv_surface(contracts: list[dict[str, Any]], spot: float | None, spot_as_of: Any = None,
                chain_as_of: datetime | None = None) -> dict[str, Any]:
    skew: dict[str, Any] = {}
    expiries = sorted({c["expiry"] for c in contracts})
    for expiry in expiries:
        calls = [c for c in contracts if c["expiry"] == expiry and c["type"] == "call" and c.get("delta") is not None and c.get("implied_volatility") is not None and abs(abs(c["delta"]) - 0.25) <= 0.10]
        puts = [c for c in contracts if c["expiry"] == expiry and c["type"] == "put" and c.get("delta") is not None and c.get("implied_volatility") is not None and abs(abs(c["delta"]) - 0.25) <= 0.10]
        if calls and puts:
            call = min(calls, key=lambda c: abs(abs(c["delta"]) - 0.25))
            put = min(puts, key=lambda c: abs(abs(c["delta"]) - 0.25))
            skew[expiry] = {"call_25d_iv": call["implied_volatility"], "put_25d_iv": put["implied_volatility"], "risk_reversal_call_minus_put": call["implied_volatility"] - put["implied_volatility"], "method": "nearest observed absolute delta to 0.25 within 0.10; call IV minus put IV"}
    term: dict[str, Any]
    spot_value = _finite(spot)
    spot_time = parse_instant(spot_as_of)
    if spot_value is None or spot_value <= 0 or spot_time is None:
        term = {"available": False, "reason": "timestamped_underlying_price_required"}
    elif chain_as_of is None or spot_time > chain_as_of.astimezone(timezone.utc):
        term = {"available": False, "reason": "underlying_price_after_or_unaligned_with_chain_as_of"}
    elif (chain_as_of.astimezone(timezone.utc) - spot_time).total_seconds() > DEFAULT_MAX_AGE_SECONDS:
        term = {"available": False, "reason": "underlying_price_stale"}
    else:
        by_expiry = {}
        for expiry in expiries:
            eligible = [c for c in contracts if c["expiry"] == expiry and c.get("implied_volatility") is not None]
            if not eligible:
                continue
            distance = min(abs(c["strike"] - spot_value) for c in eligible)
            atm = [c["implied_volatility"] for c in eligible if math.isclose(abs(c["strike"] - spot_value), distance, rel_tol=1e-10, abs_tol=1e-10)]
            by_expiry[expiry] = {"nearest_strike_distance": distance, "median_at_nearest_strike_iv": median(atm), "contract_count": len(atm)}
        term = {"available": bool(by_expiry), "underlying_price": spot_value, "underlying_as_of": spot_time.isoformat(), "method": "median provider IV among contracts at the nearest listed strike to the supplied timestamped spot, separately by expiry", "by_expiry": by_expiry}
    return {
        "iv_skew": {"available": bool(skew), "method": "observed 25-delta risk reversal; call IV minus put IV per expiry", "iv_units": "provider_native_unverified", "by_expiry": skew, "reason": None if skew else "paired_observed_25_delta_call_and_put_iv_required"},
        "iv_term_structure": term,
    }


def analyze_provider_chain(contracts: list[dict[str, Any]], spot: float | None = None,
                           spot_as_of: Any = None, chain_as_of: datetime | None = None) -> dict[str, Any]:
    """Provider-local metrics; no cross-exchange OI/IV aggregation."""
    rows = list(contracts or [])
    oi_complete = bool(rows) and all(_finite(c.get("open_interest")) is not None for c in rows)
    calls = [c for c in rows if c.get("type") == "call"]
    puts = [c for c in rows if c.get("type") == "put"]
    pcr = None
    if oi_complete:
        call_oi = sum(c["open_interest"] for c in calls)
        put_oi = sum(c["open_interest"] for c in puts)
        if call_oi > 0:
            pcr = put_oi / call_oi
    max_pain = payout_max_pain(rows)
    return {
        "put_call_ratio": pcr,
        "pcr_available": pcr is not None,
        "open_interest_complete": oi_complete,
        "max_pain": max_pain,
        "iv": _iv_surface(rows, spot, spot_as_of, chain_as_of),
        "dealer_gamma_direction": {"available": False, "reason": "unsigned_open_interest_does_not_identify_dealer_long_or_short_gamma"},
        "coverage": _contract_coverage(rows),
        "units_and_provenance": {"greeks": "exchange_observed_only; provider_native_unverified", "implied_volatility": "exchange_observed_only; provider_native_unverified", "model_derived_greeks": False, "contract_multiplier_assumption": "unit multiplier=1 only for normalized max-pain payout where provider multiplier is unavailable"},
    }


def build_provider_chain(underlying: str, venue: str, pages: list[Any],
                         pagination: Mapping[str, Any], *, retrieved_at: datetime | None = None,
                         max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
                         spot: float | None = None, spot_as_of: Any = None,
                         provider_error: str | None = None) -> dict[str, Any]:
    """Build a validated chain from page arrays plus explicit completeness evidence."""
    retrieved = (retrieved_at or utc_now()).astimezone(timezone.utc)
    target = str(underlying or "").upper().replace("USDT", "").replace("USD", "")
    page_arrays = pages if isinstance(pages, list) else []
    schema_valid = bool(page_arrays) and all(isinstance(page, list) for page in page_arrays)
    raw_rows = [row for page in page_arrays if isinstance(page, list) for row in page]
    contracts: list[dict[str, Any]] = []
    rejected = Counter()
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        contract, error = normalize_contract(row, target, venue, retrieved)
        if error:
            rejected[error] += 1
            continue
        previous = by_symbol.get(contract["symbol"])
        if previous is None or (contract.get("observed_at") or "") >= (previous.get("observed_at") or ""):
            by_symbol[contract["symbol"]] = contract
    contracts = sorted(by_symbol.values(), key=lambda c: (c["expiry"], c["strike"], c["type"], c["symbol"]))
    # Expired and same-day rows are expected on listing endpoints; they are
    # counted and excluded consistently because only an expiry date is known.
    expired_rows = rejected.pop("expired_contract", 0)
    same_day_rows = rejected.pop("expired_or_same_day_expiry_unverified", 0)
    identity_valid = not any(k in rejected for k in ("underlying_identity_mismatch", "missing_contract_identity"))
    schema_valid = schema_valid and not bool(rejected)

    ages = []
    timestamp_missing = 0
    timestamp_invalid = 0
    for c in contracts:
        observed = parse_instant(c.get("observed_at"))
        if observed is None:
            timestamp_missing += 1
        else:
            age = (retrieved - observed).total_seconds()
            if age < 0:
                timestamp_invalid += 1
            else:
                ages.append(age)
    if timestamp_missing or timestamp_invalid or (contracts and len(ages) != len(contracts)):
        freshness = "unknown"
    elif not contracts:
        freshness = "unknown"
    elif any(age > max_age_seconds for age in ages):
        freshness = "stale"
    else:
        freshness = "fresh"
    marker_evidence = bool(pagination.get("marker_seen") or pagination.get("manifest_verified"))
    complete = bool(pagination.get("complete")) and marker_evidence and not bool(pagination.get("truncated")) and not bool(provider_error)
    usable = bool(schema_valid and identity_valid and complete and freshness == "fresh" and contracts and not provider_error)
    validation = {
        "underlying": target,
        "schema_valid": schema_valid,
        "identity_valid": identity_valid,
        "complete": complete,
        "fresh": freshness == "fresh",
        "freshness_status": freshness,
        "usable": usable,
        "reasons": [reason for reason, condition in (
            ("provider_error", bool(provider_error)),
            ("invalid_page_schema", not schema_valid),
            ("underlying_identity_mismatch", not identity_valid),
            ("pagination_incomplete_or_truncated", not complete),
            ("freshness_unknown_or_stale", freshness != "fresh"),
            ("no_valid_future_contracts", not bool(contracts)),
        ) if condition],
        "retrieved_at": retrieved.isoformat(),
        "provider_observed_at_min": min((c["observed_at"] for c in contracts if c.get("observed_at")), default=None),
        "provider_observed_at_max": max((c["observed_at"] for c in contracts if c.get("observed_at")), default=None),
        "max_age_seconds": int(max_age_seconds),
        "max_observed_age_seconds": max(ages) if ages else None,
        "pagination": dict(pagination),
        "raw_row_count": len(raw_rows),
        "normalized_contract_count": len(contracts),
        "rejected_rows": dict(rejected),
        "expired_rows_excluded": expired_rows,
        "same_day_expiry_rows_excluded": same_day_rows,
        "provider_error": provider_error,
    }
    return {
        "venue": venue,
        "underlying": target,
        "retrieved_at": retrieved.isoformat(),
        "contracts": contracts,
        "calls": [c for c in contracts if c["type"] == "call"],
        "puts": [c for c in contracts if c["type"] == "put"],
        "coverage": _contract_coverage(contracts),
        "validation": validation,
        "metrics": analyze_provider_chain(contracts, spot, spot_as_of, retrieved),
    }


def combine_provider_chains(underlying: str, providers: Mapping[str, Any]) -> dict[str, Any]:
    """Expose every venue contract, while keeping metrics separated by venue."""
    providers = {str(k): v for k, v in (providers or {}).items() if isinstance(v, dict)}
    all_contracts = [contract for provider in providers.values() for contract in provider.get("contracts", []) if isinstance(contract, dict)]
    all_contracts.sort(key=lambda c: (c.get("venue", ""), c.get("expiry", ""), c.get("strike", 0), c.get("type", ""), c.get("symbol", "")))
    preferred = next((providers.get(name) for name in ("Delta", "Binance Options")
                      if isinstance(providers.get(name), dict)
                      and providers[name].get("validation", {}).get("usable")), None)
    preferred = preferred or providers.get("Delta") or providers.get("Binance Options")
    preferred_oi = None
    if preferred and preferred.get("validation", {}).get("usable") and preferred.get("metrics", {}).get("open_interest_complete"):
        preferred_oi = sum(c["open_interest"] for c in preferred.get("contracts", []) if c.get("open_interest") is not None)
    return {
        "underlying": str(underlying).upper(),
        "calls": [c for c in all_contracts if c.get("type") == "call"],
        "puts": [c for c in all_contracts if c.get("type") == "put"],
        "contracts": all_contracts,
        "total_oi": preferred_oi,
        "pcr": preferred.get("metrics", {}).get("put_call_ratio") if preferred else None,
        "max_pain": preferred.get("metrics", {}).get("max_pain", {}).get("strike") if preferred else None,
        "providers": providers,
        "provider_metrics_separate": True,
        "selected_provider": preferred.get("venue") if preferred else None,
        "validation": preferred.get("validation", {"usable": False, "reasons": ["no_provider_chain"]}),
        "coverage": _contract_coverage(all_contracts),
        "units_and_provenance": {"cross_venue_oi_aggregated": False, "provider_native_values_preserved": True, "model_derived_greeks": False},
    }
