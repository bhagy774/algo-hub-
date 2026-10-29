"""Deterministic ATR/volatility take-profit and stop-loss calculations."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


class SmartTPSLCalculator:
    def __init__(self, trading_config: Optional[Mapping[str, Any]] = None):
        self.config = dict(trading_config or {})

    @staticmethod
    def _candles(candles: Any) -> List[Mapping[str, Any]]:
        if hasattr(candles, "to_dict"):  # pandas DataFrame, without requiring pandas.
            try:
                return list(candles.to_dict("records"))
            except Exception:
                return []
        return list(candles) if isinstance(candles, Iterable) and not isinstance(candles, (str, bytes, Mapping)) else []

    def calculate_atr(self, candles: Any, period: Optional[int] = None) -> float:
        rows = self._candles(candles)
        period = max(1, int(period or self.config.get("atr_period", 14)))
        ranges: List[float] = []
        previous_close: Optional[float] = None
        for row in rows:
            try:
                high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
                if high < low or low <= 0 or close <= 0:
                    continue
                true_range = high - low if previous_close is None else max(high - low, abs(high - previous_close), abs(low - previous_close))
                ranges.append(true_range)
                previous_close = close
            except (KeyError, TypeError, ValueError):
                continue
        if not ranges:
            return 0.0
        return sum(ranges[-period:]) / min(period, len(ranges))

    def calculate(self, candles: Any, direction: str, entry_price: Optional[float] = None, mode: str = "scalping") -> Dict[str, float]:
        rows = self._candles(candles)
        if entry_price is None:
            try:
                entry_price = float(rows[-1]["close"])
            except (IndexError, KeyError, TypeError, ValueError):
                entry_price = 0.0
        try:
            entry = float(entry_price)
        except (TypeError, ValueError):
            entry = 0.0
        if not math.isfinite(entry) or entry <= 0:
            return {"entry_price": 0.0, "take_profit": 0.0, "stop_loss": 0.0, "tp1": 0.0, "tp2": 0.0, "atr": 0.0, "risk_reward": 0.0}

        normalized_mode = "swing" if str(mode).lower() == "swing" else "scalping"
        stop_mult = float(self.config.get(f"{normalized_mode}_stop_atr", 1.0 if normalized_mode == "scalping" else 1.5))
        target_mult = float(self.config.get(f"{normalized_mode}_target_atr", 1.8 if normalized_mode == "scalping" else 3.0))
        atr = self.calculate_atr(rows)
        if not math.isfinite(atr) or atr <= 0:
            atr = entry * float(self.config.get("default_atr_pct", 0.003))
        # Prevent malformed candles/config from producing nonsensical levels.
        atr = min(max(atr, entry * 0.0001), entry * 0.20)
        is_long = str(direction).upper() in {"CALL", "BUY", "LONG"}
        risk = atr * max(stop_mult, 0.01)
        reward = max(atr * max(target_mult, 0.01), risk * float(self.config.get("min_risk_reward", 1.0)))
        stop = entry - risk if is_long else entry + risk
        target = entry + reward if is_long else entry - reward
        tp1 = entry + reward * 0.55 if is_long else entry - reward * 0.55
        return {
            "entry_price": round(entry, 8), "take_profit": round(target, 8), "stop_loss": round(stop, 8),
            "tp1": round(tp1, 8), "tp2": round(target, 8), "atr": round(atr, 8),
            "risk_reward": round(reward / risk, 4), "risk_distance": round(risk, 8),
            "reward_distance": round(reward, 8),
        }

    calculate_tpsl = calculate
    calculate_targets = calculate
