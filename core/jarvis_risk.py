"""Conservative, balance/risk/margin-aware sizing shared by paper and live paths.

Leverage is an output of risk math, never a user-selected operating value.  The
only leverage setting is a conservative hard cap; venue/product limits further
reduce it.  Missing or non-positive inputs return a blocked result.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

BASE_RISK_PCT = float(os.environ.get("SIZER_BASE_RISK_PCT", "0.02"))
MAX_RISK_PCT = float(os.environ.get("SIZER_MAX_RISK_PCT", "0.05"))
MIN_MARGIN_USDT = float(os.environ.get("SIZER_MIN_MARGIN", "0.05"))
COMPOUND_RATIO = float(os.environ.get("SIZER_COMPOUND_RATIO", "0.5"))
# Caps are policy guardrails, not a requested leverage value.
MIN_LEVERAGE_CAP = max(1, int(os.environ.get("JARVIS_MIN_LEVERAGE", "1")))
MAX_LEVERAGE_CAP = max(MIN_LEVERAGE_CAP, int(os.environ.get("JARVIS_MAX_LEVERAGE", "20")))

CONF_MULTIPLIERS = [(95, 2.5), (90, 2.0), (80, 1.5), (70, 1.0), (60, 0.5), (0, 0.25)]


def confidence_multiplier(confidence: int) -> float:
    try:
        value = int(confidence)
    except (TypeError, ValueError):
        return 0.0
    for threshold, multiplier in CONF_MULTIPLIERS:
        if value >= threshold:
            return multiplier
    return 0.0


def derive_auto_leverage(
    available_balance: float,
    margin_usdt: float,
    trade_risk_usdt: float,
    stop_distance_pct: float,
    product_max_leverage: Optional[float] = None,
) -> Dict[str, Any]:
    """Derive integer leverage from collateral, margin and stop-loss risk.

    The risk-limited notional is ``trade_risk / stop_distance``.  Leverage is
    the largest integer that does not exceed that notional for the selected
    margin, then bounded by explicit policy/product caps.  Actual notional is
    still capped by the risk-limited notional after contract rounding.
    """
    try:
        balance = float(available_balance)
        margin = float(margin_usdt)
        risk = float(trade_risk_usdt)
        stop = float(stop_distance_pct)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid risk inputs"}
    if balance <= 0 or margin <= 0 or risk <= 0 or stop <= 0:
        return {"ok": False, "reason": "balance, margin, risk and stop distance must be positive"}
    if margin > balance:
        return {"ok": False, "reason": "margin exceeds available balance"}
    venue_cap = MAX_LEVERAGE_CAP
    if product_max_leverage is not None:
        try:
            venue_cap = min(venue_cap, int(float(product_max_leverage)))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "invalid product leverage metadata"}
    if venue_cap < MIN_LEVERAGE_CAP:
        return {"ok": False, "reason": "product leverage limit below policy minimum"}
    risk_notional = risk / stop
    unconstrained = risk_notional / margin
    leverage = max(MIN_LEVERAGE_CAP, min(venue_cap, int(math.floor(unconstrained))))
    # At least one contract of notional may be smaller than the margin; keep
    # the selected margin as a reservation ceiling, not a forced exposure.
    max_notional = min(margin * leverage, risk_notional, balance * leverage)
    if max_notional <= 0:
        return {"ok": False, "reason": "risk budget cannot fund exposure"}
    return {
        "ok": True,
        "leverage": int(leverage),
        "risk_notional_usdt": round(risk_notional, 8),
        "max_notional_usdt": round(max_notional, 8),
        "margin_usdt": round(margin, 8),
        "trade_risk_usdt": round(risk, 8),
        "stop_distance_pct": stop,
        "policy_cap": MAX_LEVERAGE_CAP,
        "product_cap": product_max_leverage,
    }


def contract_quote_value_usdt(product: Dict[str, Any], price: float) -> Optional[float]:
    """Resolve one venue contract to quote-currency notional.

    Delta product schemas have used several names over time.  We accept only
    explicit, recognized quote/asset-unit metadata and return None otherwise;
    live sizing must never assume one contract equals one USDT.
    """
    if not isinstance(product, dict):
        return None
    # Inverse/coin-margined contracts do not have a stable quote notional
    # that this USDT risk model can safely represent.  Never reinterpret them
    # as linear contracts merely because a contract_value field is present.
    for key in ("contract_type", "notional_type", "margin_currency_type", "settlement_type"):
        marker = str(product.get(key, "")).strip().lower().replace("-", "_")
        if "inverse" in marker or "coin_margined" in marker or marker in {"coin", "base"}:
            return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    raw = None
    value_key = None
    for key in ("contract_value_usdt", "contract_value_quote", "contract_value", "contract_size", "notional_per_contract", "multiplier"):
        if product.get(key) is not None:
            raw = product.get(key)
            value_key = key
            break
    if raw is None and isinstance(product.get("contract_unit"), dict):
        unit_info = product["contract_unit"]
        raw = unit_info.get("value", unit_info.get("size"))
        unit = unit_info.get("currency", unit_info.get("unit"))
    else:
        # ``contract_unit_currency`` is the documented Delta field for base
        # denominated contracts (for example BTCUSD contract_value=0.001,
        # contract_unit_currency=BTC).  It must take precedence over a
        # settlement asset: settlement currency alone does not identify the
        # unit of contract_value.
        unit = product.get(
            "contract_value_currency",
            product.get("contract_unit_currency", product.get(
                "quote_currency", product.get("settlement_asset", product.get("contract_unit"))
            )),
        )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if not unit and value_key in {"contract_value_usdt", "contract_value_quote"}:
        unit = "USDT"
    unit_text = str(unit or "").upper().strip()
    if unit_text in {"USD", "USDT", "USDC", "QUOTE", "QUOTE_CURRENCY"}:
        return value
    if unit_text in {"BASE", "ASSET", "COIN"}:
        # Generic labels are safe only when the product explicitly identifies
        # itself as linear/base-unit; otherwise they are ambiguous.
        base = str(product.get("base_asset", product.get("underlying_asset", ""))).upper().strip()
        if not base:
            symbol = str(product.get("symbol", "")).upper().replace("-", "").replace("_", "")
            for quote in ("USDT", "USDC", "USD"):
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    break
        if not base:
            return None
        return value * price
    # For a documented base-unit currency, require that it matches the
    # product's actual underlying asset.  This prevents a malformed fixture
    # from converting an arbitrary unit at the selected price.
    base = str(product.get("base_asset", product.get("underlying_asset", ""))).upper().strip()
    if not base:
        symbol = str(product.get("symbol", "")).upper().replace("-", "").replace("_", "")
        for quote in ("USDT", "USDC", "USD"):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                break
    if base and unit_text == base:
        return value * price
    # A bare or unrelated unit is ambiguous: fail closed instead of guessing.
    return None


def contract_quote_value_in_currency(product: Dict[str, Any], price: float, currency: str) -> Optional[float]:
    """Return one contract's notional only when Delta quote and collateral units match.

    Delta's product schema explicitly distinguishes quoting_asset,
    settling_asset, contract_value, and contract_unit_currency. The adapter
    accepts no implicit USD/USDT/INR conversion: if the quote currency differs
    from the actual settlement currency a separately validated FX adapter is
    required and this helper returns None.
    """
    if not isinstance(product, dict):
        return None
    target = str(currency or "").strip().upper()
    if not target:
        return None
    def symbol(value):
        if isinstance(value, dict):
            value = value.get("symbol") or value.get("asset_symbol") or value.get("name")
        return str(value or "").strip().upper()
    quote = symbol(product.get("quoting_asset") or product.get("quote_asset") or product.get("quote_currency"))
    settle = symbol(product.get("settling_asset") or product.get("settlement_asset") or product.get("settlement_currency"))
    if quote != target or settle != target:
        return None
    try:
        mark = float(price)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(mark) or mark <= 0:
        return None
    for key in ("contract_type", "notional_type", "margin_currency_type", "settlement_type"):
        marker = str(product.get(key, "")).strip().lower().replace("-", "_")
        if "inverse" in marker or "coin_margined" in marker or marker in {"coin", "base"}:
            return None
    raw = None
    value_key = None
    for key in ("contract_value", "contract_size_value", "contract_value_quote", "contract_value_usdt",
                "notional_per_contract", "contract_size", "multiplier"):
        if product.get(key) is not None:
            raw, value_key = product.get(key), key
            break
    if raw is None and isinstance(product.get("contract_unit"), dict):
        unit_info = product["contract_unit"]
        raw = unit_info.get("value", unit_info.get("size"))
        unit = symbol(unit_info.get("currency") or unit_info.get("unit"))
    else:
        unit = symbol(product.get("contract_value_currency") or product.get("contract_unit_currency")
                      or product.get("contract_unit"))
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    if value_key in {"contract_value_usdt", "contract_value_quote", "notional_per_contract"} and not unit:
        unit = target
    base = symbol(product.get("underlying_asset_symbol") or product.get("underlying_asset") or product.get("base_asset"))
    if unit == target:
        return value
    if base and unit == base:
        return value * mark
    return None


def calculate_trade_size(
    available_balance: float,
    confidence: int,
    stop_distance_pct: float,
    max_trade_risk_usdt: Optional[float] = None,
    compound_pool: float = 0.0,
    product_max_leverage: Optional[float] = None,
    base_risk_pct: float = BASE_RISK_PCT,
    max_margin_pct: float = MAX_RISK_PCT,
    min_margin_usdt: float = MIN_MARGIN_USDT,
    contract_value_usdt: Optional[float] = None,
    require_contract_value: bool = False,
    max_trade_risk_currency: Optional[float] = None,
    risk_currency: str = "USDT",
) -> Dict[str, Any]:
    """Return one consistent size/leverage decision for paper and live paths."""
    try:
        balance = float(available_balance)
        stop = float(stop_distance_pct)
        conf = int(confidence)
    except (TypeError, ValueError):
        balance, stop, conf = 0.0, 0.0, 0
    mult = confidence_multiplier(conf)
    if balance <= 0 or stop <= 0 or mult <= 0:
        return {"ok": False, "reason": "missing available balance or stop distance", "contracts": 0, "leverage": 0}
    base_margin = balance * max(0.0, float(base_risk_pct)) * mult
    compound = min(max(0.0, float(compound_pool)) * COMPOUND_RATIO, balance * max(0.0, float(base_risk_pct)))
    max_margin = balance * max(0.0, float(max_margin_pct))
    margin = min(max_margin, max(min_margin_usdt, base_margin + compound))
    if margin <= 0:
        return {"ok": False, "reason": "margin policy leaves no collateral", "contracts": 0, "leverage": 0}
    currency = str(risk_currency or "").strip().upper()
    if not currency:
        return {"ok": False, "reason": "risk currency is required", "contracts": 0, "leverage": 0}
    configured_trade_cap = max_trade_risk_currency if max_trade_risk_currency is not None else max_trade_risk_usdt
    risk_budget = min(balance * max(0.0, float(MAX_RISK_PCT)) * mult, configured_trade_cap if configured_trade_cap is not None else float("inf"))
    if risk_budget <= 0:
        return {"ok": False, "reason": "trade risk budget is zero", "contracts": 0, "leverage": 0}
    try:
        contract_value = float(contract_value_usdt) if contract_value_usdt is not None else 0.0
    except (TypeError, ValueError):
        contract_value = 0.0
    if require_contract_value and contract_value <= 0:
        return {"ok": False, "reason": "recognized contract value metadata is required", "contracts": 0, "leverage": 0}
    if contract_value <= 0:
        # Paper/backtest callers may explicitly use the historical one-unit
        # convention, but live callers must set require_contract_value=True.
        contract_value = 1.0
    lev = derive_auto_leverage(balance, margin, risk_budget, stop, product_max_leverage)
    if not lev.get("ok"):
        return {**lev, "contracts": 0, "margin_usdt": round(margin, 8), "confidence": conf, "multiplier": mult}
    contracts = int(math.floor(lev["max_notional_usdt"] / contract_value))
    if contracts <= 0:
        return {**lev, "ok": False, "reason": "risk budget cannot fund one whole contract", "contracts": 0, "confidence": conf, "multiplier": mult, "contract_value_usdt": contract_value}
    actual_notional = float(contracts) * contract_value
    actual_margin = min(margin, actual_notional / lev["leverage"])
    result = {
        **lev, "ok": True, "contracts": contracts,
        "contract_value_usdt": round(contract_value, 8),
        "notional_usdt": round(actual_notional, 8),
        "margin_usdt": round(actual_margin, 8),
        "balance": round(balance, 8), "confidence": conf,
        "multiplier": mult, "compound_used": round(compound, 8),
        "sizing_note": f"balance={currency} {balance:.4f}, margin={currency} {actual_margin:.4f}, risk={currency} {risk_budget:.4f}, exposure={currency} {actual_notional:.4f}, contracts={contracts}, leverage={lev['leverage']}x",
        "risk_currency": currency,
        "available_balance_currency": currency,
        "available_balance_amount": round(balance, 8),
        "contract_value_currency": currency,
        "contract_value_amount": round(contract_value, 8),
        "notional_currency": currency,
        "notional_amount": round(actual_notional, 8),
        "margin_currency": currency,
        "margin_amount": round(actual_margin, 8),
        "risk_amount": round(min(actual_notional * stop, risk_budget), 8),
    }
    if currency != "USDT":
        # Legacy *_usdt keys are intentionally omitted for USD/INR/etc.; no
        # conversion or denomination substitution is represented by these values.
        for key in ("risk_notional_usdt", "max_notional_usdt", "margin_usdt",
                    "risk_notional_usdt", "contract_value_usdt", "notional_usdt"):
            result.pop(key, None)
        result["max_notional_currency"] = round(float(lev["max_notional_usdt"]), 8)
        result["trade_risk_currency_amount"] = round(risk_budget, 8)
    return result
