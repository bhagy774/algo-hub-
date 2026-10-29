"""Optional, side-effect-free multi-timeframe candle adapter for Delta data."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


class DeltaMultiTFDataFetcher:
    """Fetch and normalize candles only when an explicit fetch method is called."""

    DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h")

    def __init__(self, delta_client: Any = None, default_limit: int = 200):
        self.delta_client = delta_client
        self.default_limit = max(1, int(default_limit))

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        cleaned = str(symbol or "BTCUSDT").upper().replace("/", "").replace("_", "")
        return cleaned if cleaned.endswith(("USDT", "USD")) else f"{cleaned}USDT"

    @staticmethod
    def _normalise_candles(candles: Any) -> List[Dict[str, float]]:
        if not isinstance(candles, Iterable) or isinstance(candles, (str, bytes, dict)):
            return []
        normalized: List[Dict[str, float]] = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                timestamp = candle.get("time", candle.get("timestamp", candle.get("date", 0)))
                item = {
                    "time": int(float(timestamp)),
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle.get("volume") or 0.0),
                }
                if item["high"] < item["low"] or min(item["open"], item["high"], item["low"], item["close"]) <= 0:
                    continue
                normalized.append(item)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        # Stable de-duplication protects downstream ATR calculations.
        by_time = {item["time"]: item for item in normalized}
        return [by_time[key] for key in sorted(by_time)]

    def fetch_candles(self, symbol: str = "BTCUSDT", timeframe: str = "1m", limit: Optional[int] = None) -> List[Dict[str, float]]:
        """Fetch one timeframe.  Fail closed to an empty list on unavailable data."""
        if self.delta_client is None or not hasattr(self.delta_client, "get_historical_candles"):
            return []
        try:
            raw = self.delta_client.get_historical_candles(
                symbol=self.normalize_symbol(symbol), resolution=str(timeframe),
                limit=max(1, int(limit or self.default_limit)),
            )
            return self._normalise_candles(raw)
        except Exception as exc:  # Optional analysis must not interrupt the brain.
            logger.debug("Multi-TF candle fetch failed for %s/%s: %s", symbol, timeframe, type(exc).__name__)
            return []

    def fetch_multi_timeframe(self, symbol: str = "BTCUSDT", timeframes: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> Dict[str, List[Dict[str, float]]]:
        return {
            str(timeframe): candles
            for timeframe in (timeframes or self.DEFAULT_TIMEFRAMES)
            if (candles := self.fetch_candles(symbol, str(timeframe), limit))
        }

    # Friendly aliases for older callers.
    fetch_all_timeframes = fetch_multi_timeframe
    fetch_mtf_candles = fetch_multi_timeframe
