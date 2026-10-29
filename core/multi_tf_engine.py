"""On-demand, deterministic multi-timeframe confluence analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


class MultiTimeframeEngine:
    def __init__(self, jarvis_system: Any = None, delta_fetcher: Any = None, tpsl_calculator: Any = None, config: Optional[Mapping[str, Any]] = None):
        self.jarvis_system = jarvis_system
        self.delta_fetcher = delta_fetcher
        self.tpsl_calculator = tpsl_calculator
        self.config = dict(config or {})

    @staticmethod
    def _direction_for(candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            closes = [float(row["close"]) for row in candles if float(row["close"]) > 0]
        except (KeyError, TypeError, ValueError):
            closes = []
        if len(closes) < 2:
            return {"signal": "NEUTRAL", "strength": 0.0, "close": 0.0}
        lookback = min(10, len(closes) - 1)
        baseline = sum(closes[-lookback - 1:-1]) / lookback
        change = (closes[-1] - baseline) / baseline if baseline else 0.0
        # A tiny dead band avoids reporting precision noise as a trade view.
        signal = "CALL" if change > 0.0002 else "PUT" if change < -0.0002 else "NEUTRAL"
        return {"signal": signal, "strength": min(1.0, abs(change) / 0.01), "close": closes[-1]}

    def analyze(self, symbol: str = "BTC", mode: Optional[str] = None) -> Dict[str, Any]:
        mode = "swing" if str(mode or self.config.get("mode", "scalping")).lower() == "swing" else "scalping"
        timeframes = self.config.get(f"{mode}_timeframes", ("1m", "5m", "15m"))
        if not self.delta_fetcher:
            return self._no_trade(symbol, mode, "Candle fetcher unavailable")
        candle_sets = self.delta_fetcher.fetch_multi_timeframe(symbol, timeframes, self.config.get("candle_limit", 200))
        views = {tf: self._direction_for(candles) for tf, candles in candle_sets.items()}
        usable = {tf: view for tf, view in views.items() if view["signal"] != "NEUTRAL"}
        if len(candle_sets) < int(self.config.get("min_timeframes", 2)) or not usable:
            return self._no_trade(symbol, mode, "Insufficient multi-timeframe confluence", views)
        signed_strength = sum(view["strength"] if view["signal"] == "CALL" else -view["strength"] for view in usable.values())
        direction = "CALL" if signed_strength > 0 else "PUT"
        agreeing = [view for view in usable.values() if view["signal"] == direction]
        confluence = len(agreeing) / len(usable)
        if confluence < 0.60:
            return self._no_trade(symbol, mode, "Timeframes conflict", views)
        entry = next((view["close"] for tf, view in reversed(list(views.items())) if view["close"] > 0), 0.0)
        primary_candles = next((c for c in reversed(list(candle_sets.values())) if c), [])
        targets = self.tpsl_calculator.calculate(primary_candles, direction, entry, mode) if self.tpsl_calculator else {}
        confidence = min(100, round(50 + confluence * 30 + min(20, abs(signed_strength) * 20)))
        return {
            "signal": direction, "direction": direction, "confidence": confidence,
            "entry_price": entry, "take_profit_1": targets.get("tp1"), "take_profit_2": targets.get("tp2"),
            "stop_loss": targets.get("stop_loss"), "tp1": targets.get("tp1"), "tp2": targets.get("tp2"),
            "sl": targets.get("stop_loss"), "atr": targets.get("atr", 0.0), "risk_reward": targets.get("risk_reward", 0.0),
            "symbol": symbol, "mode": mode, "confluence": round(confluence, 3), "timeframe_signals": views,
        }

    def _no_trade(self, symbol: str, mode: str, reason: str, views: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"signal": "NO-TRADE", "direction": "NO-TRADE", "confidence": 0, "symbol": symbol, "mode": mode,
                "reason": reason, "confluence": 0.0, "timeframe_signals": views or {}}
