#!/usr/bin/env python3
"""
Delta Exchange Unified Data Wrapper for Jarvis
Replaces:
1. Binance (Charts/Prices)
2. Deribit (Options/Greeks)

Features:
- Live Price Fetching
- Historical Data (Pagination for Deep Backtesting)
- Options Chain & Greeks
- Order Book Level 2
- Authentication Support
"""

import os
import time
import hmac
import hashlib
import math
import requests
import logging
import json
import re
import copy
from urllib.parse import urlencode
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

from options_chain import build_provider_chain, combine_provider_chains, payout_max_pain

# FIX #9: Load API keys from environment variables (or .env file)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

# API keys loaded from environment. Set these in your .env file:
#   DELTA_API_KEY=your_key_here
#   DELTA_API_SECRET=your_secret_here
# Credentials must be supplied by the deployment environment.  Do not put
# fallback credentials in source code: an unauthenticated private request is
# safer than silently signing it with a leaked or unintended account.
DEMO_API_KEY = os.environ.get("DELTA_API_KEY")
DEMO_API_SECRET = os.environ.get("DELTA_API_SECRET")

logger = logging.getLogger(__name__)


def _same_decimal_price(left: Any, right: Any) -> bool:
    """Require exact price equality; no unconfigured slippage is authorized."""
    try:
        a, b = Decimal(str(left)), Decimal(str(right))
        return a.is_finite() and b.is_finite() and a == b
    except (InvalidOperation, TypeError, ValueError):
        return False


def _valid_protective_child(row: Any, *, product_id: Any, side: str, kind: str,
                            quantity: float, trigger: float, limit: float) -> bool:
    """Verify an exact active reduce-only bracket child, not just its label."""
    try:
        return (isinstance(row, dict)
                and int(row.get("product_id", -1)) == int(product_id)
                and str(row.get("state") or "").lower() in {"open", "pending"}
                and row.get("reduce_only") is True
                and str(row.get("side") or "").lower() == str(side).lower()
                and str(row.get("stop_order_type") or "").lower() == str(kind).lower()
                and math.isclose(abs(float(row.get("size"))), float(quantity), rel_tol=1e-10, abs_tol=1e-10)
                and _same_decimal_price(row.get("stop_price"), trigger)
                and _same_decimal_price(row.get("limit_price"), limit)
                and row.get("id") is not None)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return False


def _validate_jarvis_broker_entry_authorization(
    authorization: Any, *, symbol: str, side: str, product_id: Any,
    size: Any, stop_loss: Any, take_profit: Any, leverage: Any,
) -> tuple[bool, str]:
    """Recompute Jarvis policy and bind a protected broker entry to its plan."""
    if not isinstance(authorization, dict):
        return False, "Jarvis central entry authorization is required"
    approval = authorization.get("central_approval")
    timeframe_evidence = authorization.get("parts_by_timeframe")
    gate = authorization.get("part7_gate")
    strategy_plan = authorization.get("strategy_plan")
    broker_plan = authorization.get("broker_plan")
    snapshot_version = authorization.get("snapshot_version")
    analysis_symbol = authorization.get("analysis_symbol")
    try:
        from jarvis_strategy_approval import (
            PART_WEIGHTS, REQUIRED_TIMEFRAMES, evaluate_mtf_central_strategy,
            validate_entry_approval, validate_execution_plan,
        )
        direction = "BUY" if str(side).strip().lower() == "buy" else "SELL" if str(side).strip().lower() == "sell" else "NO_TRADE"
        try:
            confidence = int(authorization.get("confidence"))
        except (TypeError, ValueError, OverflowError):
            return False, "Jarvis central entry confidence is malformed"
        if direction == "NO_TRADE" or not analysis_symbol:
            return False, "Jarvis central entry scope is incomplete"
        if (not isinstance(timeframe_evidence, dict)
                or set(timeframe_evidence) != set(REQUIRED_TIMEFRAMES)):
            return False, "Complete native 15-timeframe Jarvis evidence is required"
        from jarvis_strategy_approval import _clean_symbol
        expected_clean = _clean_symbol(analysis_symbol)
        for timeframe in REQUIRED_TIMEFRAMES:
            frame = timeframe_evidence.get(timeframe)
            if not isinstance(frame, dict) or set(frame) != set(PART_WEIGHTS):
                return False, f"{timeframe} Jarvis part evidence is incomplete"
            for part_name in PART_WEIGHTS:
                item = frame.get(part_name)
                item_symbol = item.get("symbol", item.get("selected_symbol")) if isinstance(item, dict) else None
                item_timeframe = (item.get("timeframe", item.get("native_timeframe"))
                                  if isinstance(item, dict) else None)
                if (_clean_symbol(item_symbol) != expected_clean
                        or str(item_timeframe or "") != timeframe):
                    return False, f"{timeframe} {part_name} symbol/timeframe identity mismatch"
        central = evaluate_mtf_central_strategy(
            timeframe_evidence, gate, confidence=confidence, expected_symbol=analysis_symbol,
        )
        if (not central.get("approved") or central.get("direction") != direction
                or str(central.get("trade_mode") or "").upper()
                != str((strategy_plan or {}).get("trade_mode") or "").upper()):
            return False, "Jarvis central MTF strategy does not approve this broker entry/mode"
        if not isinstance(approval, dict):
            return False, "Jarvis central approval is missing"
        # The approved execution symbol must be exact; instrument_id can be the
        # selected canonical symbol for the ordinary route or Delta's exact
        # product id for an explicitly mapped multicoin route. The protected
        # broker call separately verifies that product id maps to this symbol.
        from jarvis_strategy_approval import _clean_symbol
        if (_clean_symbol(approval.get("symbol")) != _clean_symbol(symbol)
                or _clean_symbol(approval.get("contract")) != _clean_symbol(symbol)
                or str(approval.get("exchange") or "").strip().lower() != "delta"
                or str(approval.get("instrument_id") or "").strip() not in {str(product_id).strip(), _clean_symbol(symbol)}):
            return False, "Jarvis approval does not match the exact Delta symbol/product"
        market_type = str(approval.get("market_type") or "").strip().lower()
        if market_type not in {"unverified", "perpetual", "perpetual_futures", "perpetual_swap", "perpetual_swaps"}:
            return False, "Jarvis approval market type is unsupported"
        approval_ok, reason = validate_entry_approval(
            approval, direction=direction, symbol=symbol, exchange="delta",
            contract=symbol, instrument_id=approval.get("instrument_id"),
            market_type=market_type, snapshot_version=snapshot_version,
            confidence=confidence, execution_plan=strategy_plan,
        )
        if not approval_ok:
            return False, reason
        strategy_ok, plan_reason = validate_execution_plan(
            strategy_plan, direction=direction, symbol=approval.get("symbol"),
            snapshot_version=snapshot_version, confidence=confidence,
            trade_mode=central.get("trade_mode"),
        )
        if not strategy_ok:
            return False, plan_reason
        if not isinstance(broker_plan, dict):
            return False, "Jarvis protected broker plan is missing"
        try:
            requested_size, requested_lev = int(size), int(leverage)
            stop, target = float(stop_loss), float(take_profit)
            planned_size, planned_lev = int(broker_plan.get("quantity")), int(broker_plan.get("leverage"))
            planned_stop, planned_target = float(broker_plan.get("stop_loss")), float(broker_plan.get("take_profit"))
            planned_entry = float(broker_plan.get("entry_price"))
            risk_budget = float(broker_plan.get("risk_budget_usdt"))
            strategy_stop = float(strategy_plan.get("stop_loss"))
            strategy_target = float(strategy_plan.get("take_profit"))
            strategy_entry = float(strategy_plan.get("entry_price"))
        except (TypeError, ValueError, OverflowError):
            return False, "Jarvis protected broker plan values are malformed"
        if (_clean_symbol(broker_plan.get("symbol")) != _clean_symbol(symbol)
                or str(broker_plan.get("instrument_id") or "").strip() != str(product_id).strip()
                or str(broker_plan.get("direction") or "").upper() != direction
                or str(broker_plan.get("trade_mode") or "").upper() != str(central.get("trade_mode") or "").upper()
                or requested_size != planned_size or requested_lev != planned_lev
                or not math.isclose(stop, planned_stop, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(target, planned_target, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(planned_stop, strategy_stop, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(planned_target, strategy_target, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(planned_entry, strategy_entry, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isfinite(risk_budget) or risk_budget <= 0
                or risk_budget > float(os.environ.get("JARVIS_MAX_RISK_USDT", "10")) + 1e-8):
            return False, "Broker request differs from Jarvis protected plan"
        return True, "Jarvis protected entry authorization validated"
    except Exception as exc:
        return False, f"Jarvis central authorization validation failed: {type(exc).__name__}"


class DeltaExchangeData:
    """
    Unified Data Wrapper for Delta Exchange.
    Handles all data ingestion for Jarvis.
    """
    
    # ── MAINNET vs TESTNET controlled via .env ──────────────────────────────
    # DELTA_USE_MAINNET=true  → real orders on api.delta.exchange
    # DELTA_USE_MAINNET=false → paper orders on testnet (default, safe)
    _USE_MAINNET = os.environ.get("DELTA_USE_MAINNET", "false").lower() == "true"
    PRIVATE_URL  = ("https://api.india.delta.exchange"      # MAINNET India (real money!)
                    if _USE_MAINNET else
                    "https://cdn-ind.testnet.deltaex.org")  # TESTNET (safe test)
    PUBLIC_URL   = "https://api.india.delta.exchange"  # India endpoint for real prices
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        # Explicit None = no credentials provided by caller (safety tests rely on this).
        # Empty string also = no credentials. Only fall back to env-loaded DEMO keys
        # when caller uses the default (passes nothing at all).
        self._credentials_explicit = (api_key is not None or api_secret is not None)
        self.api_key    = api_key    if api_key    is not None else (DEMO_API_KEY    or "")
        self.api_secret = api_secret if api_secret is not None else (DEMO_API_SECRET or "")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "JarvisTradingSystem/2.0"
        })
        self._cache = {}
        self._options_chain_cache = {}
        self._options_snapshot_last_capture = {}

        # ── HYBRID: Use Binance for price+candles (more accurate, no API key needed) ──
        try:
            from binance_data import BinanceData
            self._binance = BinanceData()
            logger.info("[HYBRID] ✅ Binance data source loaded for price + candles")
        except ImportError:
            self._binance = None
            logger.warning("[HYBRID] ⚠️  binance_data.py not found, using Delta for everything")

        # Public, read-only options adapter. It is constructed lazily without
        # making a network request; unsupported eAPI schemas fail closed.
        try:
            from binance_options_client import BinanceOptionsClient
            self._binance_options = BinanceOptionsClient()
        except ImportError:
            self._binance_options = None

    @staticmethod
    def _perpetual_fallback_allowed() -> bool:
        """Require an explicit opt-in before treating Bybit perps as spot data."""
        return os.environ.get("JARVIS_ALLOW_PERPETUAL_FALLBACK", "0").lower() in {"1", "true", "yes"}

    def _binance_spot_compatible(self) -> bool:
        if not self._binance:
            return False
        try:
            source = self._binance.get_last_source()
            semantics = str(source.get("market_semantics", "")).lower()
            return semantics == "spot" or self._perpetual_fallback_allowed()
        except Exception:
            return False

    def _generate_signature(self, method: str, path: str, payload: str = "") -> Dict[str, str]:
        """Generate HMAC SHA256 Signature for authenticated endpoints."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Delta API credentials are not configured")
        timestamp = str(int(time.time()))
        msg = method + timestamp + path + payload
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"api-key": self.api_key, "timestamp": timestamp, "signature": signature}

    @staticmethod
    def _diagnostic_error(response) -> str:
        """Return only a bounded, structured, non-secret API diagnostic.

        Response bodies are never logged or echoed: even an error body can
        reflect credentials. Only a strict, short machine error code is safe.
        """
        safe_code = None
        try:
            body = response.json()
            candidate = body.get("error") if isinstance(body, dict) else None
            if isinstance(candidate, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", candidate):
                safe_code = candidate
        except (ValueError, TypeError, AttributeError):
            pass
        if safe_code:
            return f"HTTP {response.status_code}: {safe_code}"
        return f"HTTP {response.status_code}"

    def _request(self, method: str, endpoint: str, payload: Dict = None, authorized: bool = False) -> Dict:
        """
        Unified Request Handler.
        authorized=True -> Uses Testnet/Private URL
        authorized=False -> Uses Production/Public URL (Real Prices)
        """
        try:
            # Fail before constructing or sending a private request when the
            # process was not explicitly provisioned with credentials.
            # Also block if caller explicitly passed None (no credentials intended).
            _no_creds = (not self.api_key or not self.api_secret)
            _explicit_none = getattr(self, "_credentials_explicit", True) is False
            if authorized and (_no_creds or _explicit_none):
                return {"success": False, "error": "Delta API credentials are not configured"}
            base_url = self.PRIVATE_URL if authorized else self.PUBLIC_URL
            headers = {}
            payload_str = ""
            req_args = {}

            if authorized:
                if method in ["POST", "PUT", "DELETE"] and payload:
                    # JSON body — sign the JSON string
                    payload_str = json.dumps(payload, separators=(',', ':'))
                    auth_headers = self._generate_signature(method, endpoint, payload_str)
                    headers.update(auth_headers)
                    req_args["data"] = payload_str  # send exact signed string

                elif method == "GET" and payload:
                    # FIX BUG1: Build query string, append to path for signature,
                    # then use full URL directly — don't also pass params= (double send bug)
                    query_str = urlencode(payload)
                    sign_path = f"{endpoint}?{query_str}"
                    auth_headers = self._generate_signature(method, sign_path, "")
                    headers.update(auth_headers)
                    url = f"{base_url}{sign_path}"  # full URL with query baked in
                    start_t = time.time()
                    response = self.session.request(method, url, headers=headers, timeout=10)
                    elapsed = time.time() - start_t
                    logger.debug(f"[DELTA API] {method} {endpoint} took {elapsed:.2f}s")
                    if 200 <= response.status_code < 300:
                        return {"success": True, "data": response.json()}
                    logger.error("[DELTA API] %s %s status=%s", method, endpoint.split("?", 1)[0], response.status_code)
                    return {"success": False, "error": self._diagnostic_error(response)}

                else:
                    # authorized GET with no payload
                    auth_headers = self._generate_signature(method, endpoint, "")
                    headers.update(auth_headers)

            # Non-authorized OR authorized POST/DELETE path continues here
            url = f"{base_url}{endpoint}"
            if method.upper() == "GET" and not authorized:
                req_args["params"] = payload
            elif not req_args.get("data"):
                req_args["json"] = payload

            start_t = time.time()
            response = self.session.request(method, url, headers=headers, timeout=10, **req_args)
            elapsed = time.time() - start_t
            logger.debug(f"[DELTA API] {method} {endpoint} took {elapsed:.2f}s")

            if 200 <= response.status_code < 300:
                return {"success": True, "data": response.json()}
            logger.error("[DELTA API] %s %s status=%s", method, endpoint.split("?", 1)[0], response.status_code)
            return {"success": False, "error": self._diagnostic_error(response)}

        except Exception as e:
            logger.error("[DELTA API] Connection failed: %s", type(e).__name__)
            return {"success": False, "error": "Delta request failed"}

    # ==========================================
    # 1. LIVE MARKET DATA (Replaces Binance)
    # ==========================================

    def get_balance(self) -> Dict:
        """Fetch the authenticated wallet response without selecting a currency."""
        return self._request("GET", "/v2/wallet/balances", authorized=True)

    @staticmethod
    def _delta_page(data: Dict[str, Any], page_size: int) -> tuple[list[dict], Optional[str], bool]:
        """Validate one cursor-paginated Delta response (rows, cursor, terminal)."""
        if not isinstance(data, dict) or not isinstance(data.get("result"), list):
            raise ValueError("Delta response has no result list")
        rows = data["result"]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Delta result contains a malformed row")
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        pagination = data.get("pagination") if isinstance(data.get("pagination"), dict) else meta
        cursor_key = next((k for k in ("after", "next_cursor", "nextCursor", "next") if k in pagination), None)
        more_key = next((k for k in ("has_more", "hasMore", "has_next", "hasNextPage") if k in pagination), None)
        if more_key and pagination.get(more_key) is False:
            return rows, None, True
        cursor = pagination.get(cursor_key) if cursor_key else None
        if cursor_key and cursor in (None, ""):
            # Delta documents meta.after=null when no following page exists;
            # that explicit terminal cursor remains authoritative even on a
            # full-sized final page.
            return rows, None, True
        if cursor not in (None, ""):
            return rows, str(cursor), False
        # Without an explicit cursor or a short page, completeness is unknown.
        if len(rows) < page_size:
            return rows, None, True
        return rows, None, False

    def _get_all_pages(self, endpoint: str, *, authorized: bool, params: Optional[Dict[str, Any]] = None,
                       page_size: int = 100, max_pages: int = 100) -> tuple[list[dict], bool]:
        rows: list[dict] = []
        cursor = None
        seen_cursors = set()
        for _ in range(max_pages):
            query = dict(params or {})
            query["page_size"] = page_size
            if cursor:
                query["after"] = cursor
            response = self._request("GET", endpoint, query, authorized=authorized)
            if not isinstance(response, dict) or response.get("success") is not True:
                return [], False
            try:
                page, next_cursor, terminal = self._delta_page(response.get("data"), page_size)
            except ValueError:
                return [], False
            rows.extend(page)
            if terminal:
                return rows, True
            if not next_cursor or next_cursor in seen_cursors:
                return [], False
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return [], False

    @staticmethod
    def _product_asset_fields(product: Dict[str, Any]) -> tuple[str, str, str, str]:
        """Read Delta's explicit product metadata; never infer assets from ticker suffixes."""
        def asset_symbol(value):
            if isinstance(value, dict):
                value = value.get("symbol") or value.get("asset_symbol") or value.get("name")
            return str(value or "").strip().upper()
        base = asset_symbol(product.get("underlying_asset_symbol") or product.get("underlying_asset")
                            or product.get("base_asset"))
        quote = asset_symbol(product.get("quoting_asset") or product.get("quote_asset")
                             or product.get("quote_currency"))
        settle = asset_symbol(product.get("settling_asset") or product.get("settlement_asset")
                              or product.get("settlement_currency"))
        product_type = str(product.get("product_type") or product.get("market_type")
                           or product.get("contract_type") or "").strip().lower()
        return base, quote, settle, product_type

    def get_available_products_snapshot(self) -> Dict[str, Any]:
        """Fetch the complete, cursor-paginated Delta product catalogue."""
        products, complete = self._get_all_pages("/v2/products", authorized=False, page_size=100, max_pages=100)
        return {"complete": bool(complete), "as_of": time.time(), "products": products if complete else []}

    def get_available_balance(self, currency: str) -> Optional[float]:
        """Return only the exact currency's available wallet balance; never convert or substitute."""
        symbol = str(currency or "").strip().upper()
        if not symbol or not re.fullmatch(r"[A-Z0-9]{2,12}", symbol):
            return None
        response = self.get_balance()
        body = response.get("data") if isinstance(response, dict) else None
        rows = body.get("result") if isinstance(body, dict) else None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return None
        matches = [row for row in rows if str(row.get("asset_symbol") or "").strip().upper() == symbol]
        if len(matches) != 1:
            return None
        try:
            value = float(matches[0].get("available_balance"))
        except (TypeError, ValueError, OverflowError):
            return None
        return value if value > 0 and value < float("inf") else 0.0

    def get_available_balance_usdt(self) -> float:
        """Compatibility accessor for a real USDT row only (no USD/INR fallback)."""
        return self.get_available_balance("USDT")

    def get_delta_executable_quote(self, *, symbol: str, product_id: str) -> Optional[Dict[str, Any]]:
        """Read the exact Delta L2 top of book and its exchange update timestamp."""
        requested = str(symbol or "").strip().upper()
        pid = str(product_id or "").strip()
        if not requested or not pid:
            return None
        catalogue = self.get_available_products_snapshot()
        if catalogue.get("complete") is not True:
            return None
        matches = [p for p in catalogue["products"]
                   if str(p.get("symbol") or "").strip().upper() == requested
                   and str(p.get("id", p.get("product_id"))) == pid]
        if len(matches) != 1:
            return None
        book_response = self._request("GET", f"/v2/l2orderbook/{requested}", authorized=False)
        body = book_response.get("data") if isinstance(book_response, dict) and book_response.get("success") else None
        book = body.get("result") if isinstance(body, dict) else None
        if not isinstance(book, dict) or str(book.get("symbol") or "").strip().upper() != requested:
            return None
        buys, sells = book.get("buy"), book.get("sell")
        if not isinstance(buys, list) or not buys or not isinstance(sells, list) or not sells:
            return None
        try:
            bid, ask = float(buys[0]["price"]), float(sells[0]["price"])
            raw_ts = float(book.get("last_updated_at"))
            # Delta L2 timestamps are documented as Unix microseconds.
            observed = raw_ts / 1_000_000.0 if raw_ts > 10_000_000_000 else raw_ts
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        max_age = max(0.1, min(60.0, float(os.environ.get("JARVIS_DELTA_QUOTE_MAX_AGE_SEC", "5"))))
        now = time.time()
        if (not all(map(lambda n: n > 0 and n < float("inf"), (bid, ask, observed)))
                or ask < bid or observed > now + 2 or now - observed > max_age):
            return None
        return {"source": "delta", "symbol": requested, "product_id": pid,
                "bid": bid, "ask": ask, "observed_at": observed}

    def _get_positions_for_product_id(self, product_id: str) -> Optional[list[dict]]:
        try:
            pid = int(product_id)
            if pid <= 0:
                return None
        except (TypeError, ValueError):
            return None
        response = self._request("GET", f"/v2/positions?product_id={pid}", authorized=True)
        body = response.get("data") if isinstance(response, dict) and response.get("success") else None
        result = body.get("result") if isinstance(body, dict) else None
        # Delta documents GET /v2/positions?product_id=... as one position
        # object (not a list). Accept the official object schema; list support
        # is retained for compatible wrapper/test transports but ambiguous or
        # malformed payloads remain unavailable.
        if isinstance(result, dict):
            # Treat only an actual position-shaped object as the documented
            # object response; an arbitrary JSON object must not be converted
            # into a synthetic zero/unknown position for reconciliation.
            if "size" not in result:
                return None
            try:
                size = float(result.get("size"))
                raw_pid = result.get("product_id") or (result.get("product") or {}).get("id")
                if not math.isfinite(size) or (raw_pid is not None and int(raw_pid) != int(product_id)):
                    return None
            except (TypeError, ValueError, OverflowError, AttributeError):
                return None
            row = dict(result)
            row.setdefault("product_id", int(product_id))
            return [row]
        if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
            return None
        return result

    @staticmethod
    def _order_result(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body = response.get("data") if isinstance(response, dict) else None
        result = body.get("result") if isinstance(body, dict) else None
        return result if isinstance(result, dict) else None

    def _get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        client_id = str(client_order_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", client_id):
            return None
        response = self._request("GET", f"/v2/orders/client_order_id/{client_id}", authorized=True)
        return self._order_result(response) if isinstance(response, dict) and response.get("success") else None

    def _set_and_confirm_product_leverage(self, product_id: int, leverage: int) -> bool:
        """Set then read back Delta's per-product order leverage before entry."""
        try:
            pid, lev = int(product_id), int(leverage)
        except (TypeError, ValueError, OverflowError):
            return False
        if pid <= 0 or lev < 1 or lev > 1000:
            return False
        endpoint = f"/v2/products/{pid}/orders/leverage"
        changed = self._request("POST", endpoint, {"leverage": str(lev)}, authorized=True)
        result = self._order_result(changed) if isinstance(changed, dict) and changed.get("success") else None
        if not result:
            return False
        try:
            if int(result.get("product_id")) != pid or int(float(result.get("leverage"))) != lev:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        confirmed = self._request("GET", endpoint, authorized=True)
        result = self._order_result(confirmed) if isinstance(confirmed, dict) and confirmed.get("success") else None
        try:
            return bool(result and int(result.get("product_id")) == pid
                        and int(float(result.get("leverage"))) == lev)
        except (TypeError, ValueError, OverflowError):
            return False

    def place_protected_order(self, *, product_id: int, symbol: str, side: str, size: int,
                              order_type: str, stop_loss: float, take_profit: float,
                              leverage: int, client_order_id: str,
                              entry_authorization: Any = None) -> Dict[str, Any]:
        """Stage a Delta entry, then attach documented TP/SL bracket orders after confirmed fill.

        Delta's documented bracket endpoint attaches exits to an existing position;
        this is NOT atomic entry+protection. Any unclear fill/protection outcome
        returns SUBMISSION_UNKNOWN and preserves the coordinator reservation.
        """
        auth_ok, auth_reason = _validate_jarvis_broker_entry_authorization(
            entry_authorization, symbol=symbol, side=side, product_id=product_id,
            size=size, stop_loss=stop_loss, take_profit=take_profit, leverage=leverage,
        )
        if not auth_ok:
            return {"status": "REJECTED", "authoritative": True, "reason": auth_reason}
        truthy = {"1", "true", "yes", "on"}
        required = ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                    "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED")
        if any(os.environ.get(name, "false").strip().lower() not in truthy for name in required):
            return {"status": "REJECTED", "authoritative": True, "reason": "live execution flags are not all enabled"}
        if not bool(getattr(self, "_USE_MAINNET", False)):
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta mainnet client is not selected"}
        if os.environ.get("JARVIS_KILL_SWITCH", "1") != "0" and os.path.exists("C:\\jarvis\\STOP_JARVIS"):
            return {"status": "REJECTED", "authoritative": True, "reason": "kill switch is active"}
        try:
            pid, qty, lev = int(product_id), int(size), int(leverage)
            entry, stop, target = str(side).lower(), float(stop_loss), float(take_profit)
            client_id = str(client_order_id).strip()
            if (pid <= 0 or qty <= 0 or lev < 1 or lev > 1000 or entry not in {"buy", "sell"}
                    or order_type != "market"
                    or not all(map(lambda x: x > 0 and x < float("inf"), (stop, target)))
                    or not client_id or len(client_id) > 32):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return {"status": "REJECTED", "authoritative": True, "reason": "invalid protected-entry request"}
        products = self.get_available_products_snapshot()
        if products.get("complete") is not True:
            return {"status": "REJECTED", "authoritative": True, "reason": "complete Delta product catalogue unavailable"}
        matched = [p for p in products["products"] if str(p.get("id", p.get("product_id"))) == str(pid)
                   and str(p.get("symbol") or "").strip().upper() == str(symbol).strip().upper()]
        if len(matched) != 1:
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta product identity is ambiguous"}
        product = matched[0]
        base, quote, settle, product_type = self._product_asset_fields(product)
        state = str(product.get("state") or product.get("status") or "").strip().lower()
        specs = product.get("product_specs") if isinstance(product.get("product_specs"), dict) else {}
        trading_status = str(product.get("trading_status") or "").strip().lower()
        if (state not in {"active", "live", "trading", "listed"}
                or product_type not in {"perpetual", "perpetual_futures", "perpetual_swap", "perpetual_swaps"}
                or not base or quote != "USDT" or settle != "USDT" or quote != settle
                or str(product.get("notional_type") or "").strip().lower() == "inverse"
                or product.get("is_quanto") is True or specs.get("only_reduce_only_orders_allowed") is True
                or trading_status not in {"", "operational"}):
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta product metadata is not eligible"}
        # Delta documents product leverage as a separate per-product setting.
        # Set and GET it back before sending any exposure-creating order.
        executable = self.get_delta_executable_quote(symbol=str(symbol), product_id=str(pid))
        if not isinstance(executable, dict):
            return {"status": "REJECTED", "authoritative": True, "reason": "fresh Delta executable quote unavailable"}
        current_entry = float(executable["ask"] if entry == "buy" else executable["bid"])
        try:
            planned_entry = float(entry_authorization["broker_plan"]["entry_price"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"status": "REJECTED", "authoritative": True, "reason": "Jarvis approved entry price is unavailable"}
        if not _same_decimal_price(current_entry, planned_entry):
            return {"status": "REJECTED", "authoritative": True,
                    "reason": "fresh Delta executable quote differs from the exact Jarvis plan entry; no slippage policy is configured"}
        if (not math.isfinite(current_entry) or current_entry <= 0
                or (entry == "buy" and not stop < current_entry < target)
                or (entry == "sell" and not target < current_entry < stop)):
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta quote invalidates protective price geometry"}
        try:
            from jarvis_risk import contract_quote_value_in_currency
            if contract_quote_value_in_currency(product, current_entry, settle) is None:
                return {"status": "REJECTED", "authoritative": True, "reason": "Delta contract value/settlement unit is unsupported"}
        except Exception:
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta contract sizing metadata is unavailable"}
        if not self._set_and_confirm_product_leverage(pid, lev):
            return {"status": "REJECTED", "authoritative": True, "reason": "Delta product leverage could not be confirmed"}
        # The venue's documented bracket API creates limit exits on the current
        # position. Market entry is accepted only if it resolves to an exact fill.
        payload = {"product_id": pid, "product_symbol": str(symbol).strip().upper(),
                   "size": qty, "side": entry, "order_type": "market_order",
                   "client_order_id": client_id}
        response = self._request("POST", "/v2/orders", payload, authorized=True)
        order = self._order_result(response) if isinstance(response, dict) and response.get("success") else None
        if not order:
            # A transport failure may happen after the exchange accepted the POST.
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False,
                    "client_order_id": client_id, "reason": "Delta entry response is unknown"}
        order_id = order.get("id")
        if order_id is None:
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False,
                    "client_order_id": client_id, "reason": "Delta entry order id missing"}
        try:
            total = int(order.get("size", qty))
            unfilled = int(order.get("unfilled_size", max(0, total - int(order.get("filled_size", 0) or 0))))
            filled = max(0, min(total, total - unfilled))
            average = float(order.get("average_fill_price", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "reason": "Delta entry fill schema is invalid"}
        state = str(order.get("state") or "").strip().lower()
        if state in {"open", "pending", "partially_filled"} and unfilled > 0:
            cancel = self._request("DELETE", f"/v2/orders/{order_id}", authorized=True)
            if not isinstance(cancel, dict) or cancel.get("success") is not True:
                return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                        "client_order_id": client_id, "reason": "unfilled entry remainder could not be canceled"}
            reread = self._request("GET", f"/v2/orders/{order_id}", authorized=True)
            confirmed = self._order_result(reread) if isinstance(reread, dict) and reread.get("success") else None
            if not confirmed:
                return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                        "client_order_id": client_id, "reason": "entry cancellation state is unknown"}
            try:
                total = int(confirmed.get("size", qty))
                unfilled = int(confirmed.get("unfilled_size", 0))
                filled = max(0, min(total, total - unfilled))
                average = float(confirmed.get("average_fill_price", average) or average)
            except (TypeError, ValueError, OverflowError):
                return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                        "client_order_id": client_id, "reason": "canceled entry fill schema is invalid"}
            if str(confirmed.get("state") or "").lower() not in {"cancelled", "canceled", "filled", "closed"}:
                return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                        "client_order_id": client_id, "reason": "entry remainder cancellation is not terminal"}
        if filled <= 0:
            if state in {"rejected", "cancelled", "canceled", "closed"}:
                return {"status": "REJECTED", "authoritative": True, "order_id": str(order_id),
                        "client_order_id": client_id, "filled_quantity": 0}
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "reason": "Delta did not confirm a positive fill"}
        if not math.isfinite(average) or average <= 0:
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "reason": "filled entry has no authoritative average fill price"}
        # Confirm the current position size before attaching a whole-position bracket.
        positions = self._get_positions_for_product_id(str(pid))
        if positions is None:
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "reason": "Delta position state unavailable before protection"}
        signed_size = 0.0
        for pos in positions:
            try:
                raw_pid = pos.get("product_id") or (pos.get("product") or {}).get("id")
                if raw_pid is not None and int(raw_pid) != pid:
                    continue
                signed_size = float(pos.get("size", 0) or 0)
            except (TypeError, ValueError, AttributeError):
                return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                        "client_order_id": client_id, "filled_quantity": filled,
                        "average_fill_price": average, "reason": "Delta position identity/size malformed"}
            if signed_size:
                break
        if abs(signed_size) != float(filled) or (entry == "buy" and signed_size <= 0) or (entry == "sell" and signed_size >= 0):
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "reason": "Delta position does not match confirmed entry fill"}
        # Delta documents POST /orders/bracket as separate TP/SL orders that
        # close the entire position. Stops use limit orders; a gap can leave a
        # triggered stop-limit unfilled, so monitoring/reconciliation remains required.
        tick = float(product.get("tick_size", 0) or 0)
        if tick <= 0 or tick == float("inf"):
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "reason": "Delta tick size unavailable for protective exits"}
        closing_side = "sell" if signed_size > 0 else "buy"
        emergency_client_id = "jmf" + hashlib.sha256(client_id.encode()).hexdigest()[:29]
        def emergency_reduce_only_close():
            return self._request("POST", "/v2/orders", {
                "product_id": pid, "product_symbol": str(symbol).strip().upper(),
                "size": int(abs(signed_size)), "side": closing_side,
                "order_type": "market_order", "reduce_only": True,
                "client_order_id": emergency_client_id,
            }, authorized=True)
        fill_entry_match = _same_decimal_price(average, planned_entry)
        fill_geometry_ok = ((entry == "buy" and stop < average < target)
                            or (entry == "sell" and target < average < stop))
        try:
            risk_budget = float(entry_authorization["broker_plan"]["risk_budget_usdt"])
            from jarvis_risk import contract_quote_value_in_currency
            fill_contract_value = contract_quote_value_in_currency(product, average, settle)
            fill_risk = (abs(signed_size) * float(fill_contract_value)
                         * abs(average - stop) / average) if fill_contract_value else float("inf")
            risk_within_budget = (math.isfinite(risk_budget) and risk_budget > 0
                                  and risk_budget <= float(os.environ.get("JARVIS_MAX_RISK_USDT", "10")) + 1e-8
                                  and math.isfinite(fill_risk) and fill_risk <= risk_budget + 1e-8)
        except (KeyError, TypeError, ValueError, OverflowError):
            risk_within_budget = False
            fill_risk = float("inf")
        if not fill_entry_match or not fill_geometry_ok or not risk_within_budget:
            close = emergency_reduce_only_close()
            close_order = self._order_result(close) if isinstance(close, dict) and close.get("success") else None
            reason = ("Delta fill price differs from the exact Jarvis plan entry" if not fill_entry_match
                      else "Delta fill is outside Jarvis stop/target geometry" if not fill_geometry_ok
                      else "Delta filled exposure exceeds Jarvis risk budget")
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "protection_state": "RISK_CLOSE_ATTEMPTED",
                    "reason": reason,
                    "close_order_id": str(close_order.get("id")) if close_order and close_order.get("id") is not None else None,
                    "close_client_order_id": emergency_client_id,
                    "close_order_response_received": close_order is not None}
        pad = max(tick, stop * 0.001)
        stop_limit = stop - pad if closing_side == "sell" else stop + pad
        bracket_payload = {
            "product_id": pid, "product_symbol": str(symbol).strip().upper(),
            "stop_loss_order": {"order_type": "limit_order", "stop_price": str(stop), "limit_price": str(stop_limit)},
            "take_profit_order": {"order_type": "limit_order", "stop_price": str(target), "limit_price": str(target)},
            "bracket_stop_trigger_method": "last_traded_price",
        }
        bracket = self._request("POST", "/v2/orders/bracket", bracket_payload, authorized=True)
        if not isinstance(bracket, dict) or bracket.get("success") is not True:
            # No atomic guarantee exists. Try a strictly reduce-only emergency
            # close; regardless of its acknowledgement, hold as UNKNOWN until
            # a complete account snapshot reconciles the position and bracket.
            close = emergency_reduce_only_close()
            close_order = self._order_result(close) if isinstance(close, dict) and close.get("success") else None
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "protection_state": "FAILED_CLOSE_ATTEMPTED",
                    "reason": "Delta protective bracket placement failed; reduce-only close outcome requires reconciliation",
                    "close_order_id": str(close_order.get("id")) if close_order and close_order.get("id") is not None else None,
                    "close_client_order_id": emergency_client_id,
                    "close_order_response_received": close_order is not None}
        # Verify both child exits are actually active before reporting protection.
        active_orders, complete = self._get_all_pages("/v2/orders", authorized=True,
                                                       params={"product_ids": str(pid), "states": "open,pending"},
                                                       page_size=100, max_pages=20)
        if not complete:
            close = emergency_reduce_only_close()
            close_order = self._order_result(close) if isinstance(close, dict) and close.get("success") else None
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "protection_state": "UNVERIFIED_CLOSE_ATTEMPTED",
                    "close_order_id": str(close_order.get("id")) if close_order and close_order.get("id") is not None else None,
                    "close_client_order_id": emergency_client_id,
                    "reason": "protective child-order verification incomplete; reduce-only close outcome requires reconciliation"}
        def active_child(row, kind):
            try:
                expected_trigger = stop if kind == "stop_loss_order" else target
                expected_limit = stop_limit if kind == "stop_loss_order" else target
                return (int(row.get("product_id")) == pid
                        and str(row.get("state") or "").lower() in {"open", "pending"}
                        and str(row.get("stop_order_type") or "").lower() == kind
                        and row.get("reduce_only") is True
                        and str(row.get("side") or "").lower() == closing_side
                        and int(row.get("size")) == filled
                        and _same_decimal_price(row.get("stop_price"), expected_trigger)
                        and _same_decimal_price(row.get("limit_price"), expected_limit)
                        and row.get("id") is not None)
            except (TypeError, ValueError, AttributeError, OverflowError):
                return False
        stop_orders = [o for o in active_orders if active_child(o, "stop_loss_order")]
        target_orders = [o for o in active_orders if active_child(o, "take_profit_order")]
        stop_ids = [str(o.get("id")) for o in stop_orders]
        target_ids = [str(o.get("id")) for o in target_orders]
        if len(stop_ids) != 1 or len(target_ids) != 1:
            close = emergency_reduce_only_close()
            close_order = self._order_result(close) if isinstance(close, dict) and close.get("success") else None
            return {"status": "SUBMISSION_UNKNOWN", "authoritative": False, "order_id": str(order_id),
                    "client_order_id": client_id, "filled_quantity": filled,
                    "average_fill_price": average, "protection_state": "UNVERIFIED_CLOSE_ATTEMPTED",
                    "close_order_id": str(close_order.get("id")) if close_order and close_order.get("id") is not None else None,
                    "close_client_order_id": emergency_client_id,
                    "reason": "Delta bracket acknowledged but protective child orders were not independently visible; reduce-only close requires reconciliation"}
        return {"status": "FILLED" if filled == qty else "PARTIAL", "authoritative": True,
                "order_id": str(order_id), "client_order_id": client_id,
                "filled_quantity": filled, "average_fill_price": average,
                "protection_state": "ACTIVE",
                "protective_exits": {"stop_loss_order_id": stop_ids[0], "take_profit_order_id": target_ids[0]},
                "protection_atomic": False, "settlement_currency": settle}

    def get_complete_account_snapshot(self, *, owned_orders: Dict[str, Any]) -> Dict[str, Any]:
        """Build a bounded Delta account snapshot using documented products, positions, orders and wallet endpoints.

        Position risk is gross contract notional in each product's explicit
        settlement currency. Since the current coordinator aggregates a single
        numeric portfolio currency, mixed settlement currencies are incomplete
        and are rejected rather than converted implicitly.
        """
        from jarvis_risk import contract_quote_value_in_currency
        catalogue = self.get_available_products_snapshot()
        if catalogue.get("complete") is not True:
            return {"complete": False, "reason": "product catalogue incomplete"}
        products = catalogue["products"]
        perpetuals = []
        product_by_id = {}
        for product in products:
            base, quote, settle, kind = self._product_asset_fields(product)
            state = str(product.get("state") or product.get("status") or "").strip().lower()
            if state in {"active", "live", "trading", "listed"} and kind in {
                    "perpetual", "perpetual_futures", "perpetual_swap", "perpetual_swaps"}:
                raw_id = product.get("id", product.get("product_id"))
                if raw_id is None:
                    return {"complete": False, "reason": "perpetual product id missing"}
                product_by_id[str(raw_id)] = product
                perpetuals.append(product)
        max_products = max(1, min(250, int(os.environ.get("JARVIS_DELTA_MAX_RECONCILE_PERPETUALS", "100"))))
        if len(perpetuals) > max_products:
            return {"complete": False, "reason": "active perpetual universe exceeds bounded reconciliation limit"}
        balances_response = self.get_balance()
        balance_body = balances_response.get("data") if isinstance(balances_response, dict) and balances_response.get("success") else None
        balance_rows = balance_body.get("result") if isinstance(balance_body, dict) else None
        if not isinstance(balance_rows, list) or any(not isinstance(row, dict) for row in balance_rows):
            return {"complete": False, "reason": "wallet balance schema incomplete"}
        positions_margined = self._request("GET", "/v2/positions/margined", authorized=True)
        margined_body = positions_margined.get("data") if isinstance(positions_margined, dict) and positions_margined.get("success") else None
        margined_rows = margined_body.get("result") if isinstance(margined_body, dict) else None
        if not isinstance(margined_rows, list) or any(not isinstance(row, dict) for row in margined_rows):
            return {"complete": False, "reason": "margined position snapshot unavailable"}
        all_orders, orders_complete = self._get_all_pages("/v2/orders", authorized=True, page_size=100, max_pages=20)
        if not orders_complete:
            return {"complete": False, "reason": "active order pagination incomplete"}

        # Query every live perpetual product by its documented exact product id
        # endpoint; /positions/margined may lag a change by up to 10 seconds.
        positions = []
        exact_by_pid: Dict[str, Dict[str, Any]] = {}
        currencies = set()
        for product in perpetuals:
            pid = str(product.get("id", product.get("product_id")))
            rows = self._get_positions_for_product_id(pid)
            if rows is None:
                return {"complete": False, "reason": "exact product position query failed"}
            raw_position = None
            for pos in rows:
                raw_pid = pos.get("product_id") or (pos.get("product") or {}).get("id")
                if raw_pid is not None and str(raw_pid) != pid:
                    return {"complete": False, "reason": "position product identity mismatch"}
                try:
                    sz = float(pos.get("size", 0) or 0)
                except (TypeError, ValueError, OverflowError):
                    return {"complete": False, "reason": "position size malformed"}
                if not math.isfinite(sz):
                    return {"complete": False, "reason": "position size nonfinite"}
                if sz:
                    if raw_position is not None:
                        return {"complete": False, "reason": "multiple nonzero position rows for one product"}
                    raw_position = pos
            exact_by_pid[pid] = raw_position or {"product_id": int(pid), "size": 0}
            if raw_position is None:
                continue
            base, quote, settle, kind = self._product_asset_fields(product)
            if not base or not quote or not settle or quote != settle:
                return {"complete": False, "reason": "open position has unsupported quote/settlement currency"}
            try:
                size_abs = abs(float(raw_position.get("size")))
                price = float(raw_position.get("mark_price") or raw_position.get("entry_price")
                              or raw_position.get("average_entry_price") or 0)
                if price <= 0:
                    quote_data = self.get_delta_executable_quote(symbol=str(product.get("symbol")), product_id=pid)
                    price = (float(quote_data["bid"]) + float(quote_data["ask"])) / 2 if quote_data else 0
                contract_value = contract_quote_value_in_currency(product, price, settle)
            except (TypeError, ValueError, OverflowError):
                return {"complete": False, "reason": "position notional cannot be valued"}
            if not contract_value or not math.isfinite(float(contract_value)) or size_abs <= 0:
                return {"complete": False, "reason": "position notional unavailable"}
            currencies.add(settle)
            positions.append({"venue": "delta", "market_type": kind, "instrument_id": pid,
                              "symbol": str(product.get("symbol") or "").strip().upper(),
                              "notional": size_abs * float(contract_value),
                              "risk_notional": size_abs * float(contract_value),
                              "risk_currency": settle, "size": float(raw_position.get("size"))})
        # Do not silently ignore an option/spot position from the broad margin
        # endpoint; it is outside this coordinator's valuation model.
        for pos in margined_rows:
            try:
                size = float(pos.get("size", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                return {"complete": False, "reason": "margined position size malformed"}
            if size and str(pos.get("product_id") or (pos.get("product") or {}).get("id") or "") not in product_by_id:
                return {"complete": False, "reason": "non-perpetual account position needs separate risk reconciliation"}

        # Match owned order IDs/client IDs and explicitly confirm protection child IDs.
        output_orders: Dict[str, Any] = {}
        owned_exchange_ids = set()
        owned_client_ids = set()
        for candidate_id, owned in (owned_orders or {}).items():
            if not isinstance(owned, dict):
                return {"complete": False, "reason": "owned-order metadata malformed"}
            identity = owned.get("identity") or {}
            pid = str(identity.get("instrument_id") or "")
            if pid not in product_by_id or pid not in exact_by_pid:
                return {"complete": False, "reason": "owned Delta product is not in account catalogue"}
            order_id = str(owned.get("order_id") or "")
            client_id = str(owned.get("client_order_id") or "")
            order = None
            if order_id:
                order_response = self._request("GET", f"/v2/orders/{order_id}", authorized=True)
                order = self._order_result(order_response) if isinstance(order_response, dict) and order_response.get("success") else None
            if order is None and client_id:
                order = self._get_order_by_client_id(client_id)
                if order:
                    order_id = str(order.get("id") or "")
            if not order or not order_id:
                return {"complete": False, "reason": "owned entry order cannot be authoritatively resolved"}
            try:
                if int(order.get("product_id")) != int(pid):
                    return {"complete": False, "reason": "owned order product mismatch"}
                total = int(order.get("size", owned.get("quantity", 0)))
                unfilled = int(order.get("unfilled_size", 0))
                filled = max(0, total - unfilled)
                average = float(order.get("average_fill_price", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                return {"complete": False, "reason": "owned order fill fields malformed"}
            owned_exchange_ids.add(order_id)
            owned_client_ids.update({client_id, str(order.get("client_order_id") or "")})
            close_order_id = str(owned.get("close_order_id") or "")
            close_client_id = str(owned.get("close_client_order_id") or "")
            close_order = None
            if close_order_id:
                close_response = self._request("GET", f"/v2/orders/{close_order_id}", authorized=True)
                close_order = self._order_result(close_response) if isinstance(close_response, dict) and close_response.get("success") else None
            if close_order is None and close_client_id:
                close_order = self._get_order_by_client_id(close_client_id)
                if close_order:
                    close_order_id = str(close_order.get("id") or "")
            if (close_order_id or close_client_id) and not close_order:
                return {"complete": False, "reason": "owned reduce-only close cannot be authoritatively resolved"}
            if close_order:
                try:
                    if (int(close_order.get("product_id")) != int(pid)
                            or close_order.get("reduce_only") is not True
                            or str(close_order.get("client_order_id") or "") != close_client_id):
                        return {"complete": False, "reason": "owned close order identity/reduce-only mismatch"}
                except (TypeError, ValueError):
                    return {"complete": False, "reason": "owned close order schema malformed"}
                owned_exchange_ids.add(close_order_id)
                owned_client_ids.add(close_client_id)
            protective = owned.get("protective_exits") if isinstance(owned.get("protective_exits"), dict) else {}
            stop_id = str(protective.get("stop_loss_order_id") or "")
            target_id = str(protective.get("take_profit_order_id") or "")
            if not stop_id or not target_id:
                matching_protection = [row for row in all_orders if str(row.get("product_id")) == pid
                                       and str(row.get("state") or "").lower() in {"open", "pending"}
                                       and row.get("reduce_only") is True]
                stop_rows = [row for row in matching_protection if str(row.get("stop_order_type") or "").lower() == "stop_loss_order"]
                target_rows = [row for row in matching_protection if str(row.get("stop_order_type") or "").lower() == "take_profit_order"]
                if len(stop_rows) == 1 and len(target_rows) == 1:
                    stop_id, target_id = str(stop_rows[0].get("id") or ""), str(target_rows[0].get("id") or "")
            position = exact_by_pid[pid]
            try:
                signed_size = float(position.get("size", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                return {"complete": False, "reason": "owned exact position malformed"}
            if signed_size:
                expected_sign = 1 if str(owned.get("direction") or "").upper() in {"BUY", "CALL"} else -1
                if (expected_sign > 0 and signed_size < 0) or (expected_sign < 0 and signed_size > 0):
                    return {"complete": False, "reason": "owned position direction mismatches candidate"}
                try:
                    expected_stop = float(owned.get("stop_loss"))
                    expected_target = float(owned.get("take_profit"))
                    tick = float(product_by_id[pid].get("tick_size", 0) or 0)
                    if (not math.isfinite(expected_stop) or not math.isfinite(expected_target)
                            or expected_stop <= 0 or expected_target <= 0
                            or not math.isfinite(tick) or tick <= 0):
                        raise ValueError
                    closing_side = "sell" if signed_size > 0 else "buy"
                    stop_pad = max(tick, expected_stop * 0.001)
                    expected_stop_limit = (expected_stop - stop_pad if closing_side == "sell"
                                           else expected_stop + stop_pad)
                    expected_qty = abs(signed_size)
                except (TypeError, ValueError, OverflowError, KeyError):
                    return {"complete": False, "reason": "owned protective levels or product tick are missing"}

                def exact_protective(row, kind):
                    trigger = expected_stop if kind == "stop_loss_order" else expected_target
                    limit = expected_stop_limit if kind == "stop_loss_order" else expected_target
                    return _valid_protective_child(
                        row, product_id=pid, side=closing_side, kind=kind,
                        quantity=expected_qty, trigger=trigger, limit=limit,
                    )

                active_by_id = {str(row.get("id")): row for row in all_orders}
                stop_row, target_row = active_by_id.get(stop_id), active_by_id.get(target_id)
                if not stop_id or not exact_protective(stop_row or {}, "stop_loss_order"):
                    matches = [row for row in all_orders if exact_protective(row, "stop_loss_order")]
                    if len(matches) != 1:
                        return {"complete": False, "reason": "exact active stop child cannot be reconciled"}
                    stop_row, stop_id = matches[0], str(matches[0].get("id"))
                if not target_id or not exact_protective(target_row or {}, "take_profit_order"):
                    matches = [row for row in all_orders if exact_protective(row, "take_profit_order")]
                    if len(matches) != 1:
                        return {"complete": False, "reason": "exact active target child cannot be reconciled"}
                    target_row, target_id = matches[0], str(matches[0].get("id"))
                detail = {"status": "FILLED" if filled >= total else "PARTIAL", "authoritative": True,
                          "filled_quantity": filled, "average_fill_price": average,
                          "protection_state": "ACTIVE",
                          "protective_exits": {"stop_loss_order_id": stop_id, "take_profit_order_id": target_id}}
                owned_exchange_ids.update({order_id, stop_id, target_id})
            else:
                state = str(order.get("state") or "").lower()
                child_filled = False
                for child_id in (stop_id, target_id):
                    if child_id:
                        child_response = self._request("GET", f"/v2/orders/{child_id}", authorized=True)
                        child = self._order_result(child_response) if isinstance(child_response, dict) and child_response.get("success") else None
                        if child and str(child.get("state") or "").lower() in {"closed", "filled"}:
                            child_filled = True
                close_filled = False
                if close_order:
                    close_state = str(close_order.get("state") or "").lower()
                    try:
                        close_total = int(close_order.get("size", 0))
                        close_unfilled = int(close_order.get("unfilled_size", 0))
                        close_filled = close_state in {"closed", "filled"} and close_total > close_unfilled
                    except (TypeError, ValueError, OverflowError):
                        return {"complete": False, "reason": "owned reduce-only close fill fields malformed"}
                if (close_filled and filled > 0) or (child_filled and state in {"closed", "filled"}):
                    detail = {"status": "CLOSED", "authoritative": True, "filled_quantity": filled}
                    owned_exchange_ids.update({order_id, stop_id, target_id, close_order_id})
                elif state in {"rejected", "cancelled", "canceled"} and filled == 0:
                    detail = {"status": "REJECTED", "authoritative": True, "filled_quantity": 0}
                    owned_exchange_ids.add(order_id)
                else:
                    return {"complete": False, "reason": "flat position does not prove order termination/close"}
            output_orders[str(candidate_id)] = detail
        external_orders = [order for order in all_orders if str(order.get("id") or "") not in owned_exchange_ids
                           and str(order.get("client_order_id") or "") not in owned_client_ids]
        if currencies and len(currencies) > 1:
            return {"complete": False, "reason": "mixed settlement currencies cannot be aggregated safely"}
        return {"complete": True, "as_of": time.time(), "positions": positions,
                "orders": output_orders, "external_orders": external_orders,
                "wallet_balances": balance_rows, "risk_currency": next(iter(currencies), "")}

    def get_live_price(self, symbol: str = "BTCUSDT") -> float:
        """Get real-time BTC price — uses Binance (most accurate), falls back to Delta."""
        # ── PRIMARY: Binance real-time price (no API key, most accurate) ──
        if self._binance:
            try:
                price = self._binance.get_live_price(symbol)
                if price > 0 and self._binance_spot_compatible():
                    return price
                if price > 0:
                    logger.warning("[HYBRID] Rejecting non-spot fallback price for %s; set JARVIS_ALLOW_PERPETUAL_FALLBACK=1 only when explicitly compatible", symbol)
            except Exception as e:
                logger.warning(f"[BINANCE PRICE] Failed: {e}")

        # ── FALLBACK: Delta Exchange mark_price ──
        symbols_to_try = ["BTCUSDT", "BTCUSD"] if "BTC" in symbol.upper() else [symbol]
        for sym in symbols_to_try:
            try:
                res = self._request("GET", "/v2/tickers", {"symbol": sym})
                if res["success"]:
                    for t in res["data"].get("result", []):
                        if t.get("symbol") == sym:
                            price = float(t.get("mark_price", 0) or t.get("spot_price", 0) or t.get("close", 0))
                            if price > 0:
                                return price
            except Exception as e:
                logger.warning(f"[DELTA PRICE] Failed for {sym}: {e}")

        logger.error("[DELTA PRICE] Could not fetch live price")
        return 0.0

    def get_order_book(self, symbol: str = "BTCUSD") -> Dict:
        """Get Bid/Ask — uses Binance bookTicker (real-time), falls back to Delta L2."""
        # ── PRIMARY: Binance bookTicker (real-time, no API key) ──
        if self._binance:
            try:
                ba = self._binance.get_bid_ask(symbol)
                if ba["bid"] > 0 and (
                    str(ba.get("market_semantics", "spot")).lower() == "spot"
                    or self._perpetual_fallback_allowed()
                ):
                    return {
                        "buy": [{"price": str(ba["bid"]), "size": "1"}],
                        "sell": [{"price": str(ba["ask"]), "size": "1"}],
                        "bid": ba["bid"],
                        "ask": ba["ask"],
                        "spread": ba["spread"]
                    }
            except Exception as e:
                logger.warning(f"[BINANCE BID/ASK] Failed: {e}")

        # ── FALLBACK: Delta L2 order book ──
        res = self._request("GET", f"/v2/l2orderbook/{symbol}")
        if res["success"]:
            return res["data"].get("result", {})
        return {}

    def get_bid_ask(self, symbol: str = "BTCUSDT") -> Dict:
        """Convenience: Returns {bid, ask, spread} from Binance bookTicker."""
        if self._binance:
            result = self._binance.get_bid_ask(symbol)
            if result.get("bid", 0) > 0 and (
                str(result.get("market_semantics", "spot")).lower() == "spot"
                or self._perpetual_fallback_allowed()
            ):
                return result
        price = self.get_live_price(symbol)
        return {"bid": None, "ask": None, "spread": None, "fallback_price": price, "source": "unavailable", "market_semantics": "unavailable"}

    # ==========================================
    # 2. HISTORICAL DATA (Replaces Binance)
    # ==========================================

    def get_historical_candles_with_metadata(
        self, symbol: str = "BTCUSD", resolution: str = "5m", limit: int = 100
    ) -> Dict[str, Any]:
        """Fetch candles with explicit same-instrument source provenance.

        The legacy method below remains list-shaped for existing callers.  Live
        analysis uses this metadata form so Binance/Bybit/Delta fallback is
        visible and cannot overwrite another source's cache entry silently.
        """
        requested_symbol = str(symbol)
        # ── PRIMARY: Binance (which may itself use explicit Bybit fallback) ──
        if self._binance:
            try:
                candles = self._binance.get_historical_candles(symbol, resolution, limit)
                if candles and len(candles) > 0:
                    source = getattr(self._binance, "last_candle_source", None) or "binance"
                    return {"candles": candles, "source": source, "symbol": requested_symbol}
            except Exception as e:
                logger.warning(f"[BINANCE CANDLES] Failed: {e}")

        # ── FALLBACK: Delta Exchange candles, same requested instrument ──
        end_time = int(time.time())
        multipliers = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
        }
        if resolution not in multipliers:
            logger.warning("[DELTA CANDLES] Unsupported native resolution: %s", resolution)
            return {"candles": [], "source": "delta", "symbol": requested_symbol}
        start_time = end_time - (limit * multipliers[resolution])
        query_sym = requested_symbol
        if query_sym.upper() in ("BTCUSD", "BTCUSDT", "BTC_USDT"):
            query_sym = "BTCUSDT"
        params = {"symbol": query_sym, "resolution": resolution, "start": start_time, "end": end_time}
        res = self._request("GET", "/v2/history/candles", params)
        if res.get("success"):
            results = res.get("data", {}).get("result", [])
            for row in results:
                if row.get("volume") is None:
                    row["volume"] = 0.0
            return {"candles": results, "source": "delta", "symbol": requested_symbol}
        return {"candles": [], "source": "delta", "symbol": requested_symbol}

    def get_delta_native_candles_with_metadata(
        self, *, symbol: str, resolution: str, limit: int
    ) -> Dict[str, Any]:
        """Fetch Delta candles only, with no Binance or Bybit substitution.

        The multicoin analysis namespace uses this method so a Delta execution
        product is never silently paired with another venue's similarly-named
        ticker. This is a public read-only request and does not authorize orders.
        """
        multipliers = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
        }
        requested_symbol = str(symbol or "").strip().upper()
        if not requested_symbol or resolution not in multipliers:
            return {"candles": [], "source": "delta", "symbol": requested_symbol}
        end_time = int(time.time())
        start_time = end_time - (int(limit) * multipliers[resolution])
        params = {"symbol": requested_symbol, "resolution": resolution,
                  "start": start_time, "end": end_time}
        response = self._request("GET", "/v2/history/candles", params)
        if not response.get("success"):
            return {"candles": [], "source": "delta", "symbol": requested_symbol}
        rows = response.get("data", {}).get("result", [])
        if not isinstance(rows, list):
            return {"candles": [], "source": "delta", "symbol": requested_symbol}
        for row in rows:
            if isinstance(row, dict) and row.get("volume") is None:
                row["volume"] = 0.0
        return {"candles": rows, "source": "delta", "symbol": requested_symbol}

    def get_historical_candles(self, symbol: str = "BTCUSD", resolution: str = "5m", limit: int = 100) -> List[Dict]:
        """Legacy list-shaped wrapper around metadata-bearing candle fetch."""
        return self.get_historical_candles_with_metadata(symbol, resolution, limit).get("candles", [])

    def fetch_deep_history(self, symbol: str, days: int = 30, resolution: str = "1h") -> List[Dict]:
        """
        [TIME-MACHINE SUPPORT]
        Fetch massive historical data using pagination.
        Used by: Smart Backtester
        """
        logger.info(f"[DELTA API] 🕰️ Fetching deep history for {symbol} ({days} days)...")
        all_candles = []
        end_time = int(time.time())
        start_time_final = end_time - (days * 86400)
        
        current_end = end_time
        
        # Paginate backwards
        while current_end > start_time_final:
            current_start = max(start_time_final, current_end - (2000 * 3600)) # Approx chunk
            
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "start": current_start,
                "end": current_end
            }
            res = self._request("GET", "/v2/history/candles", params)
            
            if res["success"]:
                chunk = res["data"].get("result", [])
                if not chunk:
                    break
                for r in chunk:
                    if r.get('volume') is None:
                        r['volume'] = 0.0
                all_candles = chunk + all_candles # Prepend (oldest first)
                # Update pointer
                current_end = int(chunk[-1]["time"]) - 1 
            else:
                break
            
            time.sleep(0.5) # Rate limit protection

        logger.info(f"[DELTA API] ✅ Fetched {len(all_candles)} historical candles")
        return all_candles

    # ==========================================
    # 3. OPTIONS CHAIN & GREEKS (Replaces Deribit)
    # ==========================================

    @staticmethod
    def _delta_ticker_page(response: Dict) -> tuple[list[dict], Any, bool, bool]:
        """Return rows, next cursor, marker-seen and terminal-page evidence."""
        if not isinstance(response, dict) or not response.get("success"):
            return [], None, False, False
        body = response.get("data")
        if not isinstance(body, dict):
            return [], None, False, False
        rows = body.get("result")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return [], None, False, False
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else None
        paging = body.get("pagination") if isinstance(body.get("pagination"), dict) else meta
        if paging is None:
            return rows, None, False, False
        cursor_key = next((k for k in ("next_cursor", "nextCursor", "next", "after") if k in paging), None)
        more_key = next((k for k in ("has_more", "hasMore", "has_next", "hasNextPage") if k in paging), None)
        marker_seen = cursor_key is not None or more_key is not None
        if not marker_seen:
            return rows, None, False, False
        cursor = paging.get(cursor_key) if cursor_key else None
        more = bool(paging.get(more_key)) if more_key else bool(cursor)
        return rows, cursor if more else None, True, not more

    def get_options_chain(self, underlying: str = "BTC") -> Dict:
        """Fetch current all-page BTC/ETH option chains from supported providers.

        A chain is filter-usable only when the provider schema, identity,
        freshness, and terminal-pagination/manifest evidence all validate.
        Missing fields stay null and are counted; Greeks are never modeled.
        """
        from datetime import datetime, timezone
        from options_snapshot_store import append_snapshot

        asset = str(underlying or "").upper().replace("USDT", "").replace("USD", "")
        cached = getattr(self, "_options_chain_cache", {}).get(asset)
        if cached and time.monotonic() - cached[0] <= 15:
            return copy.deepcopy(cached[1])
        request_started = datetime.now(timezone.utc)
        if not re.fullmatch(r"[A-Z0-9]{2,12}", asset):
            return combine_provider_chains("", {})
        pages: list[list[dict]] = []
        cursor = None
        marker_seen = False
        terminated = False
        provider_error = None
        max_pages = 100
        page_size = 500
        for page_number in range(max_pages):
            params = {
                "contract_types": "call_options,put_options",
                "underlying_asset_symbols": asset,
                "page_size": page_size,
            }
            if cursor is not None:
                params["after"] = cursor
            response = self._request("GET", "/v2/tickers", params)
            rows, next_cursor, has_marker, terminal = self._delta_ticker_page(response)
            if not isinstance(response, dict) or not response.get("success"):
                provider_error = (response.get("error") if isinstance(response, dict) else None) or "provider_request_failed"
                break
            if not isinstance(response.get("data"), dict) or not isinstance(response["data"].get("result"), list):
                provider_error = "provider_schema_invalid"
                break
            pages.append(rows)
            marker_seen = marker_seen or has_marker
            if terminal:
                terminated = True
                break
            if next_cursor is None:
                # A page without an explicit continuation is not completeness evidence.
                break
            if next_cursor == cursor:
                provider_error = "pagination_cursor_did_not_advance"
                break
            cursor = next_cursor
        else:
            provider_error = "pagination_page_limit_reached"
        delta_retrieved = datetime.now(timezone.utc)
        pagination = {
            "request_started_at": request_started.isoformat(),
            "complete": terminated,
            "truncated": bool(provider_error == "pagination_page_limit_reached" or (marker_seen and not terminated)),
            "marker_seen": marker_seen,
            "pages_received": len(pages),
            "page_size_requested": page_size,
            "termination_evidence": "provider_terminal_pagination_marker" if terminated else None,
            "cursor_used": cursor is not None,
        }
        delta_chain = build_provider_chain(
            asset, "Delta", pages, pagination, retrieved_at=delta_retrieved,
            provider_error=provider_error,
        )
        delta_chain["status"] = "ok" if delta_chain["validation"]["usable"] else ("provider_error" if provider_error else "incomplete_or_unverified")
        providers = {"Delta": delta_chain}
        if self._binance_options is not None:
            try:
                binance_chain = self._binance_options.get_options_chain(asset)
                if isinstance(binance_chain, dict):
                    providers["Binance Options"] = binance_chain
            except Exception as exc:
                providers["Binance Options"] = {
                    "venue": "Binance Options", "underlying": asset, "contracts": [], "status": "provider_error",
                    "validation": {"usable": False, "complete": False, "reasons": ["provider_request_failed"], "error_type": type(exc).__name__},
                }
        chain = combine_provider_chains(asset, providers)
        # Historical option observations cannot be reconstructed from OHLC.
        # Start a bounded local archive by default; deployments should override
        # this with a persistent volume path. Capture at a conservative cadence
        # because full option chains can be large.
        snapshot_path = os.environ.get(
            "JARVIS_OPTIONS_SNAPSHOT_PATH",
            os.path.join(".jarvis_state", "options_snapshots.jsonl"),
        ).strip()
        try:
            capture_interval = max(60.0, float(os.environ.get("JARVIS_OPTIONS_SNAPSHOT_INTERVAL_SECONDS", "300")))
        except (TypeError, ValueError, OverflowError):
            capture_interval = 300.0
        try:
            snapshot_max_bytes = max(1024, int(os.environ.get("JARVIS_OPTIONS_SNAPSHOT_MAX_BYTES", "134217728")))
        except (TypeError, ValueError, OverflowError):
            snapshot_max_bytes = 134217728
        capture_completed = datetime.now(timezone.utc)
        now_mono = time.monotonic()
        if not hasattr(self, '_options_snapshot_last_capture') or not isinstance(self._options_snapshot_last_capture, dict):
            self._options_snapshot_last_capture = {}
        last_capture = self._options_snapshot_last_capture.get(asset, 0.0)
        if snapshot_path and now_mono - last_capture >= capture_interval:
            written = append_snapshot(snapshot_path, chain, captured_at=capture_completed)
            self._options_snapshot_last_capture[asset] = now_mono
            chain["snapshot_persistence"] = {
                "enabled": True, "written": bool(written),
                "schema": "jarvis.options.snapshot.v1",
                "sample_interval_seconds": int(capture_interval),
                "max_bytes_per_generation": snapshot_max_bytes,
            }
        elif snapshot_path:
            chain["snapshot_persistence"] = {
                "enabled": True, "written": False, "reason": "sampling_interval_not_elapsed",
                "sample_interval_seconds": int(capture_interval),
            }
        else:
            chain["snapshot_persistence"] = {"enabled": False, "reason": "JARVIS_OPTIONS_SNAPSHOT_PATH_empty"}
        self._options_chain_cache[asset] = (time.monotonic(), copy.deepcopy(chain))
        return chain

    def get_institutional_bias(self, underlying: str = "BTC") -> Dict:
        """Apply the existing PCR heuristic only to one validated provider chain."""
        chain = self.get_options_chain(underlying)
        providers = chain.get("providers", {}) if isinstance(chain, dict) else {}
        provider = next((providers.get(name) for name in ("Delta", "Binance Options")
                         if isinstance(providers.get(name), dict)
                         and providers[name].get("validation", {}).get("usable")), None)
        if provider is None:
            validation = chain.get("validation", {}) if isinstance(chain, dict) else {}
            return {
                "bias": "NEUTRAL", "score": 0, "reasons": ["No Data"],
                "raw_data": {"options_validation": validation, "source_provider": None,
                             "provider_status": {name: item.get("status") for name, item in providers.items() if isinstance(item, dict)}},
            }
        all_opts = provider.get("contracts", [])
        metrics = provider.get("metrics", {})
        if not all_opts or not metrics.get("open_interest_complete") or metrics.get("put_call_ratio") is None:
            return {"bias": "NEUTRAL", "score": 0, "reasons": ["No Data"],
                    "raw_data": {"options_validation": provider.get("validation"), "source_provider": provider.get("venue"),
                                 "coverage": provider.get("coverage"), "provider_metrics": metrics}}

        expiries: dict[str, dict[str, float]] = {}
        for option in all_opts:
            exp = option.get("expiry")
            if not exp or option.get("open_interest") is None:
                continue
            row = expiries.setdefault(exp, {"calls": 0.0, "puts": 0.0, "call_volume": 0.0, "put_volume": 0.0})
            side = "calls" if option.get("type") == "call" else "puts"
            row[side] += option["open_interest"]
            vol = option.get("volume")
            if vol is not None:
                row["call_volume" if side == "calls" else "put_volume"] += vol
        reasons = []
        total_score = 0
        for exp in sorted(expiries):
            data = expiries[exp]
            if data["calls"] + data["puts"] < 100:
                continue
            if data["calls"] <= 0:
                continue
            pcr = data["puts"] / data["calls"]
            if pcr > 1.3:
                total_score -= 2
                reasons.append(f"[{exp}] High PCR ({pcr:.2f}); unsigned OI is descriptive, not dealer direction")
            elif pcr < 0.65:
                total_score += 2
                reasons.append(f"[{exp}] Low PCR ({pcr:.2f}); unsigned OI is descriptive, not dealer direction")
        final_bias = "BULLISH" if total_score >= 3 else ("BEARISH" if total_score <= -3 else "NEUTRAL")
        max_pain = metrics.get("max_pain", {})
        max_pain_strike = max_pain.get("strike") if max_pain.get("available") else None
        pcr = metrics.get("put_call_ratio")
        return {
            "bias": final_bias, "score": total_score, "reasons": reasons,
            "pcr": pcr, "max_pain": max_pain_strike,
            "raw_data": {
                "expiries": expiries, "total_oi": sum(c["open_interest"] for c in all_opts),
                "pcr": pcr, "max_pain": max_pain,
                "options_validation": provider.get("validation"),
                "source_provider": provider.get("venue"),
                "coverage": provider.get("coverage"),
                "provider_metrics": metrics,
            },
        }

    def _calculate_max_pain(self, chain: Dict) -> Optional[float]:
        """Return payout-minimizing strike, never the highest-OI strike."""
        try:
            providers = chain.get("providers", {}) if isinstance(chain, dict) else {}
            selected = chain.get("selected_provider") if isinstance(chain, dict) else None
            provider = providers.get(selected) if selected else None
            if isinstance(provider, dict):
                pain = provider.get("metrics", {}).get("max_pain", {})
                return pain.get("strike") if pain.get("available") else None
            pain = payout_max_pain(chain.get("calls", []) + chain.get("puts", []))
            return pain.get("strike") if pain.get("available") else None
        except Exception:
            return None


    # ==========================================
    # 4. OPTIONS EXECUTION & MANAGEMENT (NEW)
    # ==========================================

    def _usable_options_contracts(self, underlying: str) -> List[Dict]:
        """Return contracts from one complete, fresh, validated provider only."""
        chain = self.get_options_chain(underlying)
        providers = chain.get("providers", {}) if isinstance(chain, dict) else {}
        for name in ("Delta", "Binance Options"):
            provider = providers.get(name) if isinstance(providers, dict) else None
            validation = provider.get("validation", {}) if isinstance(provider, dict) else {}
            contracts = provider.get("contracts", []) if isinstance(provider, dict) else []
            if validation.get("usable") is True and isinstance(contracts, list):
                return [row for row in contracts if isinstance(row, dict)]
        return []

    def get_nearest_expiries(self, underlying: str = "BTC") -> List[str]:
        """List future expiries from the selected validated provider chain."""
        expiries = {
            str(opt.get("expiry")) for opt in self._usable_options_contracts(underlying)
            if opt.get("expiry") and opt.get("expiry") != "PERPETUAL"
        }
        return sorted(expiries)

    def get_option_positions(self) -> List[Dict]:
        """Fetch all currently open option positions"""
        res = self._request("GET", "/v2/positions", authorized=True)
        option_positions = []
        if res["success"]:
            for pos in res["data"].get("result", []):
                sym = pos.get("symbol", "")
                if "-C" in sym or "-P" in sym:  # Rough heuristic for option symbols
                    option_positions.append(pos)
        return option_positions

    def place_option_order(self, option_symbol: str, side: str, size: int) -> Dict:
        """Place an option order (wrapper around place_order)"""
        # Note: Delta options often require limit orders or have low liquidity
        # For scalp hedging, we try market first
        return self.place_order(option_symbol, side, size, order_type="market")

    def get_option_by_criteria(self, underlying: str, option_type: str, target_strike: float, expiry_preference: str = "nearest") -> Optional[Dict]:
        """Select from the nearest requested expiry in one usable provider chain.

        This is a discovery helper, not execution authorization: callers still
        need fresh quotes, product metadata, liquidity checks, and risk approval.
        """
        kind = str(option_type or "").strip().upper()
        if kind not in {"CALL", "PUT"}:
            return None
        try:
            strike = float(target_strike)
            if not math.isfinite(strike) or strike <= 0:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        options = [row for row in self._usable_options_contracts(underlying)
                   if row.get("type") == kind.lower()]
        if not options:
            return None
        if expiry_preference == "nearest":
            expiries = sorted({str(row.get("expiry")) for row in options if row.get("expiry")})
            if not expiries:
                return None
            selected_expiry = expiries[0]
        else:
            selected_expiry = str(expiry_preference or "")
        options = [row for row in options if str(row.get("expiry")) == selected_expiry]
        if not options:
            return None
        return min(options, key=lambda row: abs(float(row.get("strike")) - strike))

    # ==========================================
    # 5. ORDER EXECUTION & RISK MANAGEMENT (PHASE 3)
    # ==========================================

    def get_wallet_balance_checked(self) -> Optional[float]:
        """None = unavailable (API error); never conflate with a verified 0.0 balance."""
        res = self._request("GET", "/v2/wallet/balances", authorized=True)
        if not (isinstance(res, dict) and res.get("success")):
            return None
        v = self.get_wallet_balance()
        return v

    def get_wallet_balance(self) -> float:
        """Get Available Balance for 200x Calculation (USDT, USD, or DETO)."""
        res = self._request("GET", "/v2/wallet/balances", authorized=True)
        if res["success"]:
            try:
                # Find best collateral asset
                balances = res["data"].get("result", [])
                
                # Priority: USDT -> USD -> DETO
                for symbol in ["USDT", "USD", "DETO"]:
                    for asset in balances:
                        if asset["asset_symbol"] == symbol:
                            val = float(asset["available_balance"])
                            if val > 0:
                                return val
                
                # If none have balance, just return the first one found if any
                if balances:
                    return float(balances[0].get("available_balance", 0.0))
            except:
                pass
        return 0.0

    def get_open_positions(self, symbol: str = "BTCUSDT") -> List[Dict]:
        """Get open positions, scoping the private request to one product.

        Delta's positions endpoint uses ``product_id`` for this filter. Resolve
        the caller-facing symbol through the existing product lookup rather
        than sending an unsupported ``symbol`` query parameter or making a
        broad positions request.
        """
        self.last_positions_query_ok = False
        if not (getattr(self, 'api_key', '') and getattr(self, 'api_secret', '')):
            logger.info('[DELTA API] positions query skipped: no API credentials (unverified, NOT treated as flat)')
            return []
        resolved_symbol = None
        product_id = None
        if not symbol:
            # Preserve the explicit unscoped query for callers that intentionally
            # ask for all positions. Instrument-specific reconciliation below
            # must always resolve and scope by a known Delta product ID.
            res = self._request("GET", "/v2/positions", None, authorized=True)
        else:
            product = self._resolve_product(symbol)
            if not isinstance(product, dict):
                logger.error("[DELTA API] Product not found for requested positions")
                self.last_positions_query_ok = False; return []
            raw_symbol = product.get("symbol")
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                logger.error("[DELTA API] Invalid product symbol for requested positions")
                self.last_positions_query_ok = False; return []
            resolved_symbol = raw_symbol.strip().upper()
            requested = symbol.strip().upper() if isinstance(symbol, str) else ""
            accepted_symbols = {requested}
            if requested.endswith("USDT"):
                accepted_symbols.add(requested[:-4] + "USD")
            elif requested.endswith("USD"):
                accepted_symbols.add(requested[:-3] + "USDT")
            if resolved_symbol not in accepted_symbols:
                logger.error("[DELTA API] Resolved product does not match requested positions symbol")
                self.last_positions_query_ok = False; return []
            try:
                pid = int(product["id"])
                if pid <= 0:
                    raise ValueError("non-positive product ID")
            except (KeyError, TypeError, ValueError):
                logger.error("[DELTA API] Invalid product ID for requested positions")
                self.last_positions_query_ok = False; return []
            product_id = pid
            # Build signed URL with product_id as query param (Delta requirement).
            res = self._request("GET", f"/v2/positions?product_id={pid}", None, authorized=True)
        if not isinstance(res, dict) or not res.get("success"):
            self.last_positions_query_ok = False; return []
        data = res.get("data")
        positions = data.get("result") if isinstance(data, dict) else None
        if not isinstance(positions, list):
            self.last_positions_query_ok = False; return []
        self.last_positions_query_ok = True
        if resolved_symbol is None:
            return positions

        matched = []
        for position in positions:
            if not isinstance(position, dict):
                continue
            raw_product = position.get("product")
            if raw_product is not None and not isinstance(raw_product, dict):
                continue
            raw_product = raw_product if isinstance(raw_product, dict) else {}
            raw_position_id = raw_product.get("id", position.get("product_id"))
            if raw_position_id is not None:
                try:
                    if int(raw_position_id) != product_id:
                        continue
                except (TypeError, ValueError):
                    continue
            raw_position_symbol = (raw_product.get("symbol") or position.get("symbol")
                                   or position.get("product_symbol"))
            if raw_position_symbol is not None:
                if not isinstance(raw_position_symbol, str) or raw_position_symbol.strip().upper() != resolved_symbol:
                    continue
            elif raw_position_id is None:
                # The endpoint is scoped, but a record with no instrument identity
                # cannot safely be assigned to this caller's product.
                continue
            matched.append(position)
        return matched

    def close_all_positions(self, symbol: str = "BTCUSDT") -> List[Dict]:
        """
        EMERGENCY: Close all open positions at market.
        Returns list of close order results.
        """
        positions = self.get_open_positions(symbol)
        results = []
        for pos in positions:
            # Venue responses may encode size as a string, including signed
            # decimal strings.  Do not infer a close from malformed data.
            try:
                raw_size = float(pos.get("size", 0))
                size = int(abs(raw_size))
            except (TypeError, ValueError):
                logger.warning("[EMERGENCY] Skipping position with invalid size")
                continue
            if size <= 0:
                continue
            # If long (size > 0) → sell to close. If short → buy to close.
            close_side = "sell" if raw_size > 0 else "buy"
            # Emergency close shares the same process-local ownership guard as
            # normal monitors. A timeout remains claimed until an operator or
            # reconciliation confirms that the venue position is flat.
            try:
                from jarvis_close_coordinator import claim_close, release_close, reconcile_close
                position_id = pos.get("id") or pos.get("position_id") or "position"
                if not claim_close(symbol, position_id, owner="DeltaEmergencyClose"):
                    results.append({"success": False, "error": "close already claimed"})
                    continue
                client_id = f"jarvis-emergency-close-{symbol}-{position_id}"[:64]
                res = self.place_order(symbol, close_side, size, order_type="market",
                                       reduce_only=True, client_order_id=client_id)
                if isinstance(res, dict) and res.get("success"):
                    release_close(symbol, position_id)
                else:
                    recon = reconcile_close(self, symbol, close_side, size)
                    if recon.get("confirmed"):
                        release_close(symbol, position_id)
                    elif isinstance(res, dict):
                        res = dict(res, error="close unconfirmed; operator reconciliation required",
                                   reconciliation=recon)
                results.append(res)
                logger.warning(f"[EMERGENCY] Closed {size}x {symbol}: {res}")
            except Exception as exc:
                results.append({"success": False, "error": f"close unconfirmed: {type(exc).__name__}"})
        return results

    def _resolve_product(self, symbol: str) -> Optional[Dict]:
        """Resolve a caller-facing symbol to the matching Delta product.

        Scanners and defaults typically emit ``XXXUSDT`` while Delta India
        lists perps as ``XXXUSD``. Try the exact symbol first, then the
        alternate quote (USDT<->USD) so callers get the product they mean
        instead of a failed lookup.  The product list is cached briefly to
        avoid re-downloading the full catalogue on every call.
        Returns the raw product dict, or None when nothing matches.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        symbol = symbol.strip().upper()

        now = time.time()
        cache = getattr(self, "_products_cache", None)
        if not cache or now - cache.get("ts", 0) > 300:
            # CRITICAL FIX: Use authorized=True to force lookup on Testnet
            # (Private URL). Mainnet Product IDs are invalid on Testnet.
            res = self._request("GET", "/v2/products", authorized=True)
            if not res["success"]:
                return None
            products = res["data"].get("result", []) or []
            self._products_cache = {"ts": now, "products": products}
        products = self._products_cache["products"]

        candidates = [symbol]
        if symbol.endswith("USDT"):
            candidates.append(symbol[:-4] + "USD")
        elif symbol.endswith("USD"):
            candidates.append(symbol[:-3] + "USDT")

        for candidate in candidates:
            for p in products:
                if p.get("symbol") == candidate:
                    return p
        return None

    def get_product_id(self, symbol: str) -> Optional[str]:
        """Fetch Product ID for a given Symbol (required for Leverage)."""
        product = self._resolve_product(symbol)
        if product is None:
            return None
        try:
            return str(product["id"])
        except (KeyError, TypeError):
            return None

    def get_product_metadata(self, symbol: str) -> Optional[Dict]:
        """Return unique metadata from a complete public product snapshot.

        The legacy `_resolve_product` convenience lookup reads a single raw
        response page and can map USDT/USD aliases. That is unsuitable for
        policy-bound live execution: it could miss duplicate symbols or a
        product beyond the server's default page. The multicoin path requires
        one exact symbol in a terminally complete catalogue instead.
        """
        requested = str(symbol or "").strip().upper()
        if not requested:
            return None
        snapshot = self.get_available_products_snapshot()
        if not isinstance(snapshot, dict) or snapshot.get("complete") is not True:
            return None
        products = snapshot.get("products")
        if not isinstance(products, list):
            return None
        matches = [product for product in products if isinstance(product, dict)
                   and str(product.get("symbol") or "").strip().upper() == requested]
        if len(matches) != 1:
            return None
        product = matches[0]
        try:
            if int(product.get("id", product.get("product_id"))) <= 0:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        return dict(product)

    def set_leverage(self, symbol: str, leverage: int = None) -> bool:
        """Apply a caller-computed leverage only when live execution is enabled.

        There is intentionally no venue default: callers must derive leverage
        from balance, margin and stop-loss risk and pass a positive integer.
        """
        if os.environ.get("DELTA_ORDER_EXECUTION_ENABLED", "false").lower() != "true":
            logger.warning("[RISK] Leverage change blocked: order execution is disabled")
            return False
        if not isinstance(symbol, str) or not symbol.strip():
            return False
        try:
            leverage = int(leverage)
        except (TypeError, ValueError):
            return False
        if leverage <= 0:
            return False
        if leverage <= 0 or leverage > 1000:
            return False
        product = self.get_product_metadata(symbol)
        if not product:
            logger.error(f"[RISK] Output: Exact product metadata not found for {symbol}")
            return False
        try:
            product_id = int(product.get("id", product.get("product_id")))
        except (TypeError, ValueError, OverflowError):
            return False
        if product_id <= 0:
            return False
        if not self._set_and_confirm_product_leverage(product_id, leverage):
            logger.error(f"[RISK] Failed to set and confirm leverage for {symbol}")
            return False
        logger.info(f"[RISK] Leverage set to {leverage}x for {symbol}")
        return True

    def place_order(self, symbol: str, side: str, size: int, order_type: str = "market", limit_price: float = 0,
                    reduce_only: bool = False, client_order_id: str = None) -> Dict:
        """
        Execute Trade (Live or Paper).

        Delta supports ``reduce_only`` and ``client_order_id`` on order payloads.
        Close paths must set both so a retry cannot intentionally reverse a
        position and the venue can deduplicate a stable close identity.
        New entries cannot use this raw endpoint because it cannot attach and
        independently verify Jarvis-authorized stop/target protection; use
        place_protected_order with a fresh central authorization instead.
        """
        if not reduce_only:
            return {"success": False, "error": "Unprotected raw Delta entries are disabled; use Jarvis-authorized place_protected_order"}
        # An explicit process-level opt-in is required even on testnet.  This
        # prevents a caller or configuration mistake from turning analysis into
        # an order submission.
        if os.environ.get("DELTA_ORDER_EXECUTION_ENABLED", "false").lower() != "true":
            return {"success": False, "error": "Order execution is disabled"}
        if not isinstance(symbol, str) or not symbol.strip():
            return {"success": False, "error": "Invalid symbol"}
        if not isinstance(side, str) or side.lower() not in ("buy", "sell"):
            return {"success": False, "error": "Invalid order side"}
        try:
            size = int(size)
        except (TypeError, ValueError):
            return {"success": False, "error": "Invalid order size"}
        if size <= 0:
            return {"success": False, "error": "Invalid order size"}
        if not isinstance(order_type, str):
            return {"success": False, "error": "Invalid order type"}
        ot = order_type.lower().replace("_order", "").strip()
        if ot not in ("market", "limit"):
            return {"success": False, "error": "Invalid order type"}
        if ot == "limit":
            try:
                limit_price = float(limit_price)
            except (TypeError, ValueError):
                return {"success": False, "error": "Invalid limit price"}
            if limit_price <= 0:
                return {"success": False, "error": "Invalid limit price"}
        final_type = f"{ot}_order"  # Delta API expects: market_order or limit_order

        product_id = self.get_product_id(symbol.strip())
        if not product_id:
            logger.error(f"[EXECUTION] Product ID not found for {symbol}")
            return {"success": False, "error": "Product ID not found"}

        payload = {
            "product_id": int(product_id),
            "size": int(size),
            "side": side.lower(),
            "order_type": final_type,
        }

        if ot == "limit" and limit_price > 0:
            payload["limit_price"] = str(limit_price)
        if reduce_only:
            payload["reduce_only"] = True
        if client_order_id:
            client_order_id = str(client_order_id).strip()[:64]
            if client_order_id:
                payload["client_order_id"] = client_order_id

        logger.warning(f"[EXECUTION] Placing {side.upper()} {final_type} for {size} {symbol} (reduce_only={bool(reduce_only)})...")
        res = self._request("POST", "/v2/orders", payload, authorized=True)

        if res["success"]:
            order_data = res["data"].get("result", {})
            logger.info(f"[EXECUTION] Success! Order ID: {order_data.get('id')}")
            return {"success": True, "order_id": order_data.get("id"), "details": order_data}
        else:
            logger.error(f"[EXECUTION] Failed: {res.get('error')}")
            return {"success": False, "error": res.get("error")}

    def place_reduce_only_order(self, *, product_id: int, symbol: str, side: str, size: int,
                                client_order_id: str) -> Dict[str, Any]:
        """Submit an identity-exact, reduce-only market close order.

        Close orders are independently guarded just like exposure-creating
        entry orders: direct callers must not bypass any of the opt-in flags,
        mainnet selection, or the operator kill switch.
        """
        truthy = {"1", "true", "yes", "on"}
        required = ("JARVIS_MULTICOIN_DELTA_EXECUTION", "JARVIS_AUTO_TRADE", "JARVIS_LIVE_EXECUTION",
                    "DELTA_USE_MAINNET", "DELTA_ORDER_EXECUTION_ENABLED")
        if any(os.environ.get(name, "false").strip().lower() not in truthy for name in required):
            return {"success": False, "error": "All live execution flags are required for Delta close"}
        if not bool(getattr(self, "_USE_MAINNET", False)):
            return {"success": False, "error": "Delta mainnet client is not selected"}
        if os.environ.get("JARVIS_KILL_SWITCH", "1") != "0" and os.path.exists("C:\\jarvis\\STOP_JARVIS"):
            return {"success": False, "error": "Kill switch is active"}
        try:
            pid, qty = int(product_id), int(size)
            sym, order_side, client_id = str(symbol).strip().upper(), str(side).strip().lower(), str(client_order_id).strip()
            if pid <= 0 or qty <= 0 or order_side not in {"buy", "sell"} or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", client_id):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return {"success": False, "error": "Invalid reduce-only close request"}
        catalogue = self.get_available_products_snapshot()
        if catalogue.get("complete") is not True:
            return {"success": False, "error": "Complete Delta product catalogue unavailable"}
        matches = [p for p in catalogue["products"] if str(p.get("id", p.get("product_id"))) == str(pid)
                   and str(p.get("symbol") or "").strip().upper() == sym]
        if len(matches) != 1:
            return {"success": False, "error": "Exact Delta close identity not found"}
        positions = self._get_positions_for_product_id(str(pid))
        if positions is None:
            return {"success": False, "error": "Exact Delta position unavailable for close"}
        nonzero = []
        for position in positions:
            try:
                value = float(position.get("size", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                return {"success": False, "error": "Malformed exact Delta position"}
            if not math.isfinite(value):
                return {"success": False, "error": "Malformed exact Delta position"}
            if value:
                nonzero.append(value)
        if len(nonzero) != 1 or qty > abs(nonzero[0]) or order_side != ("sell" if nonzero[0] > 0 else "buy"):
            return {"success": False, "error": "Reduce-only close does not match the current owned position"}
        response = self._request("POST", "/v2/orders", {
            "product_id": pid, "product_symbol": sym, "size": qty, "side": order_side,
            "order_type": "market_order", "reduce_only": True, "client_order_id": client_id,
        }, authorized=True)
        order = self._order_result(response) if isinstance(response, dict) and response.get("success") else None
        if not order or str(order.get("client_order_id") or "") != client_id:
            # The request may have reached Delta despite an incomplete response.
            return {"success": False, "unknown": True, "client_order_id": client_id}
        try:
            if int(order.get("product_id")) != pid or order.get("reduce_only") is not True:
                return {"success": False, "unknown": True, "client_order_id": client_id}
        except (TypeError, ValueError):
            return {"success": False, "unknown": True, "client_order_id": client_id}
        return {"success": True, "order_id": str(order.get("id") or ""),
                "client_order_id": client_id, "details": order}

    def place_batch_orders(self, orders: List[Dict]) -> Dict:
        """
        Place Main Trade + Hedge Options simultaneously?
        Delta supports batch orders.
        """
        # Simplification: Loop for now. Delta has /v2/orders/batch if needed.
        results = []
        for order in orders:
            res = self.place_order(
                order["symbol"], order["side"], order["size"], 
                order.get("type", "market"), order.get("price", 0)
            )
            results.append(res)
        return {"results": results}
    
    def get_available_products(self) -> List[Dict[str, Any]]:
        """Return only a complete cursor-paginated public Delta product catalogue."""
        snapshot = self.get_available_products_snapshot()
        if snapshot.get("complete") is not True:
            logger.warning("[DELTA API] Product catalogue pagination incomplete")
            return []
        return [dict(product) for product in snapshot["products"]
                if isinstance(product, dict) and product.get("symbol")]

    def get_available_symbols(self) -> List[str]:
        """Get list of all available trading symbols."""
        return [str(product["symbol"]) for product in self.get_available_products()]
    
    def place_limit_order(self, symbol: str, side: str, quantity: int, price: float) -> Dict:
        """Place a limit order (wrapper for place_order)"""
        return self.place_order(symbol, side, quantity, order_type="limit_order", limit_price=price)
    
    def get_order(self, order_id: str) -> Dict:
        """Read one order for operator reconciliation; never retries or mutates."""
        if not order_id:
            return {"success": False, "error": "Invalid order id"}
        return self._request("GET", f"/v2/orders/{str(order_id)}", authorized=True)

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order"""
        payload = {"id": order_id}
        res = self._request("DELETE", f"/v2/orders/{order_id}", payload, authorized=True)
        return res

if __name__ == "__main__":
    # Quick Test
    client = DeltaExchangeData()
    print("Testing Delta Exchange Unified Client...")
    print(f"Live BTC Price: ${client.get_live_price()}")
    chain = client.get_options_chain()
    print(f"Options Chain PCR: {chain.get('pcr', 'N/A')}")
    hist = client.get_historical_candles(limit=5)
    print(f"History (5 candles): {len(hist)} fetched")

