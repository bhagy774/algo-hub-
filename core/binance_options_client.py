"""Read-only Binance Options public-data adapter (no account or order routes).

The eAPI can vary by product/region and is not live-verified in this offline
change. An unsupported or incomplete schema is returned as unavailable rather
than interpreted as an empty/complete chain.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

import requests

from options_chain import build_provider_chain, parse_instant

logger = logging.getLogger(__name__)


class BinanceOptionsClient:
    BASE_URL = "https://eapi.binance.com"

    def __init__(self, session: Any = None, timeout: float = 8.0):
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get(self, path: str) -> Any:
        response = self.session.get(self.BASE_URL + path, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _items(payload: Any) -> list[dict[str, Any]] | None:
        if isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            return payload
        if isinstance(payload, dict):
            for key in ("result", "data", "optionSymbols", "symbols"):
                value = payload.get(key)
                if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                    return value
        return None

    def get_options_chain(self, underlying: str) -> dict[str, Any]:
        target = str(underlying or "").upper().replace("USDT", "").replace("USD", "")
        if target not in {"BTC", "ETH"}:
            return {"venue": "Binance Options", "underlying": target, "status": "unsupported_underlying", "validation": {"usable": False, "reasons": ["only_BTC_ETH_supported_by_this_adapter"]}, "contracts": []}
        try:
            info = self._get("/eapi/v1/exchangeInfo")
            listing = self._items(info.get("optionSymbols") if isinstance(info, dict) else None)
            if listing is None and isinstance(info, dict):
                listing = self._items(info.get("symbols"))
            if listing is None:
                return {"venue": "Binance Options", "underlying": target, "status": "provider_schema_unverified", "validation": {"usable": False, "reasons": ["exchange_info_symbols_schema_missing"]}, "contracts": []}
            matching = [r for r in listing if str(r.get("underlying") or r.get("underlyingAsset") or "").upper() == target]
            if not matching:
                return {"venue": "Binance Options", "underlying": target, "status": "unsupported_or_no_listed_contracts", "contracts": [], "validation": {"usable": False, "reasons": ["no_matching_listed_instruments"], "complete": True}}
            marks = self._items(self._get("/eapi/v1/mark"))
            tickers = self._items(self._get("/eapi/v1/ticker"))
            if marks is None or tickers is None:
                return {"venue": "Binance Options", "underlying": target, "status": "provider_schema_unverified", "contracts": [], "validation": {"usable": False, "reasons": ["mark_or_ticker_schema_missing"], "complete": False}}
            by_mark = {str(x.get("symbol")): x for x in marks if x.get("symbol")}
            by_ticker = {str(x.get("symbol")): x for x in tickers if x.get("symbol")}
            expected = {str(r.get("symbol") or "") for r in matching}
            expected.discard("")
            if len(expected) != len(matching):
                return {"venue": "Binance Options", "underlying": target, "status": "provider_schema_unverified", "contracts": [], "validation": {"usable": False, "reasons": ["listing_rows_missing_unique_symbol"], "complete": False}}
            coverage_complete = expected.issubset(set(by_mark)) and expected.issubset(set(by_ticker))
            server_time = parse_instant(info.get("serverTime") if isinstance(info, dict) else None)
            retrieved = datetime.now(timezone.utc)
            stamp = server_time.isoformat() if server_time else None
            raw_rows = []
            for instrument in matching:
                symbol = str(instrument.get("symbol"))
                mark = by_mark.get(symbol, {})
                ticker = by_ticker.get(symbol, {})
                row = dict(instrument)
                row["timestamp"] = mark.get("time") or mark.get("timestamp") or ticker.get("closeTime") or stamp
                row["implied_volatility"] = mark.get("markIV") if mark.get("markIV") is not None else mark.get("markIv")
                row["greeks"] = {k: mark.get(k) for k in ("delta", "gamma", "theta", "vega", "rho") if mark.get(k) is not None}
                row["bid"] = ticker.get("bidPrice")
                row["ask"] = ticker.get("askPrice")
                row["openInterest"] = ticker.get("openInterest")
                row["volume"] = ticker.get("volume")
                raw_rows.append(row)
            return build_provider_chain(
                target, "Binance Options", [raw_rows],
                {"complete": coverage_complete, "truncated": False, "marker_seen": True,
                 "method": "exchangeInfo optionSymbols manifest matched against all mark and ticker symbols",
                 "listed_symbol_count": len(expected), "mark_symbol_count": len(expected.intersection(by_mark)),
                 "ticker_symbol_count": len(expected.intersection(by_ticker))},
                retrieved_at=retrieved,
            ) | {"status": "ok" if coverage_complete else "incomplete", "provider_server_time": stamp}
        except Exception as exc:
            # Do not echo response bodies, credentials, or request URLs.
            logger.warning("Binance Options public chain unavailable (%s)", type(exc).__name__)
            return {"venue": "Binance Options", "underlying": target, "status": "provider_error", "contracts": [], "validation": {"usable": False, "complete": False, "reasons": ["provider_request_failed"], "error_type": type(exc).__name__}}
