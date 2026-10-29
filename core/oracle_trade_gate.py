#!/usr/bin/env python3
"""
JARVIS Oracle Trade Gate — Layer 4 Auto-Trade Gate (Gate 0.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates all proposed live trades against the JARVIS Market Oracle.
Enforces:
  1. Direction Alignment: 5m & 30m timeframe consensus must match signal direction.
  2. Entry Zone Verification: Current price must fall inside Oracle entry zone.
  3. Dynamic TP/SL: Adopts Oracle exit target and stop loss when aligned.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("OracleTradeGate")

# Configuration
ORACLE_HARD_GATE = os.environ.get("ORACLE_HARD_GATE", "true").lower() == "true"
ORACLE_GATE_ENABLED = os.environ.get("ORACLE_GATE_ENABLED", "true").lower() == "true"
ORACLE_ENTRY_TOLERANCE = float(os.environ.get("ORACLE_ENTRY_TOLERANCE", "0.0025"))  # 0.25% tolerance margin


class OracleTradeGate:
    """
    Gate 0.5 in JarvisAutoTrader.
    Acts as a macro-market checkpoint before placing orders.
    """

    def __init__(self, oracle_ref=None, enabled: bool = None, hard_gate: bool = None):
        self._oracle_ref = oracle_ref
        self.enabled = ORACLE_GATE_ENABLED if enabled is None else enabled
        self.hard_gate = ORACLE_HARD_GATE if hard_gate is None else hard_gate
        self.tolerance = ORACLE_ENTRY_TOLERANCE

    def _get_oracle(self):
        """Retrieve Oracle instance."""
        if self._oracle_ref:
            return self._oracle_ref
        try:
            from jarvis_market_oracle import get_oracle
            return get_oracle()
        except Exception:
            return None

    def get_forecast(self) -> Dict[str, Any]:
        """Only a fresh, locally computed observed-data map may reach execution."""
        from datetime import datetime, timezone
        oracle = self._get_oracle()
        if not oracle or not hasattr(oracle, "get_latest_forecast"):
            return {}
        forecast = oracle.get_latest_forecast()
        if not isinstance(forecast, dict) or forecast.get("model_used") != "local_synthesizer":
            return {}
        try:
            timestamp = datetime.fromisoformat(forecast["generated_at"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if not 0 <= age <= 600:
                return {}
        except (ValueError, KeyError, TypeError):
            return {}
        return forecast

    # ──────────────────────────────────────────────────────────
    #  GATE CHECK: DIRECTION ALIGNMENT
    # ──────────────────────────────────────────────────────────

    def is_trade_aligned(self, direction: str, symbol: str = "BTCUSDT") -> Tuple[bool, str]:
        """
        Check if signal direction aligns with Oracle multi-timeframe forecast.
        Returns: (allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, "Oracle gate disabled (bypassed)"
        requested = str(symbol or "").upper().replace("-", "").replace("_", "")
        requested_base = requested[:-4] if requested.endswith("USDT") else (requested[:-3] if requested.endswith("USD") else requested)
        # Current oracle payload is explicitly BTC-only.  Never apply that
        # forecast as an altcoin execution gate.
        if requested_base and requested_base != "BTC":
            return True, f"Oracle ADVISORY-ONLY: BTC macro forecast not applicable to {requested_base}; not blocking"

        forecast = self.get_forecast()
        if not isinstance(direction, str) or direction.upper() not in ("CALL", "BUY", "LONG", "PUT", "SELL", "SHORT"):
            return False, "Invalid trade direction"
        if not forecast:
            return True, "Oracle ADVISORY-ONLY: fresh forecast unavailable; not blocking"

        is_call = direction.upper() in ("CALL", "BUY", "LONG")
        tf5 = forecast.get("5min")
        tf30 = forecast.get("30min")
        suggestion = forecast.get("trade_suggestion", "WAIT")
        if not isinstance(tf5, dict) or not isinstance(tf30, dict) or not isinstance(suggestion, str):
            return False, "Oracle forecast is incomplete"
        dir_5m = str(tf5.get("direction", "")).upper()
        dir_30m = str(tf30.get("direction", "")).upper()
        sugg = suggestion.upper()

        if self.hard_gate:
            if sugg == "WAIT":
                return False, "Oracle advises WAIT"
            expected = "BULLISH" if is_call else "BEARISH"
            # Hard gate requires exact consensus, rather than accepting one
            # favorable timeframe while the other is neutral or contradictory.
            if dir_5m != expected or dir_30m != expected:
                return False, f"Oracle lacks {expected} 5m/30m consensus ({dir_5m}/{dir_30m})"
            allowed_suggestions = ("CALL", "BUY", "LONG") if is_call else ("PUT", "SELL", "SHORT")
            if sugg not in allowed_suggestions:
                return False, "Oracle suggestion conflicts with trade direction"
            return True, f"Oracle aligned ({expected} on 5m and 30m)"

        if is_call and (dir_5m == "BEARISH" and dir_30m == "BEARISH"):
            return False, "Oracle bearish on both timeframes"
        if not is_call and (dir_5m == "BULLISH" and dir_30m == "BULLISH"):
            return False, "Oracle bullish on both timeframes"
        return True, f"Oracle advisory ({dir_5m} 5m / {dir_30m} 30m)"

    # ──────────────────────────────────────────────────────────
    #  GATE CHECK: ENTRY ZONE VERIFICATION
    # ──────────────────────────────────────────────────────────

    def is_price_in_entry_zone(self, current_price: float, symbol: str = "BTCUSDT") -> Tuple[bool, str]:
        """
        Verifies if current spot price is within Oracle's recommended entry zone.
        Returns: (in_zone: bool, reason: str)
        """
        if not self.enabled:
            return True, "Oracle gate disabled"
        requested = str(symbol or "").upper().replace("-", "").replace("_", "")
        requested_base = requested[:-4] if requested.endswith("USDT") else (requested[:-3] if requested.endswith("USD") else requested)
        if requested_base and requested_base != "BTC":
            return True, f"Oracle ADVISORY-ONLY: entry zone not applicable to {requested_base}; not blocking"

        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return False, "Invalid price"
        if current_price <= 0:
            return False, "Invalid price"

        forecast = self.get_forecast()
        if not forecast:
            return True, "Oracle ADVISORY-ONLY: fresh forecast unavailable; not blocking"

        ez = forecast.get("entry_zone")
        if not isinstance(ez, dict):
            return (False, "Oracle entry zone is missing") if self.hard_gate else (True, "Oracle entry zone missing; hard gate is off")
        try:
            p_from = float(ez.get("price_from", 0) or 0)
            p_to = float(ez.get("price_to", 0) or 0)
        except (TypeError, ValueError):
            return (False, "Oracle entry zone is invalid") if self.hard_gate else (True, "Oracle entry zone invalid; hard gate is off")

        if p_from <= 0 or p_to <= 0:
            return (False, "Oracle entry zone is invalid") if self.hard_gate else (True, "No specific entry zone defined by Oracle")

        low_bound  = min(p_from, p_to) * (1.0 - self.tolerance)
        high_bound = max(p_from, p_to) * (1.0 + self.tolerance)

        if low_bound <= current_price <= high_bound:
            return True, f"Price ${current_price:,.2f} is inside Oracle entry zone (${min(p_from, p_to):,.0f}-${max(p_from, p_to):,.0f})"

        # If outside entry zone
        if self.hard_gate:
            dist_pct = ((current_price - low_bound) / current_price) * 100 if current_price < low_bound else ((current_price - high_bound) / current_price) * 100
            return False, (
                f"Price ${current_price:,.2f} is outside Oracle entry zone "
                f"(${min(p_from, p_to):,.0f}-${max(p_from, p_to):,.0f}) by {abs(dist_pct):.2f}% — waiting for pullback"
            )

        return True, f"Price slightly outside entry zone (${min(p_from, p_to):,.0f}-${max(p_from, p_to):,.0f}) but hard gate is off"

    # ──────────────────────────────────────────────────────────
    #  DYNAMIC TAKE-PROFIT & STOP-LOSS
    # ──────────────────────────────────────────────────────────

    def get_oracle_tp_sl(self, current_price: float, direction: str, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Only same-asset, fresh deterministic Oracle exits can replace TP/SL."""
        requested = str(symbol or "").upper().replace("-", "").replace("_", "")
        if requested not in ("BTCUSDT", "BTCUSD"):
            return {"use_oracle": False, "reason": "BTC Oracle exits unavailable for selected asset"}
        forecast = self.get_forecast()
        if not isinstance(direction, str) or direction.upper() not in ("CALL", "BUY", "LONG", "PUT", "SELL", "SHORT"):
            return {"use_oracle": False, "reason": "Invalid direction"}
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return {"use_oracle": False, "reason": "Invalid price"}
        is_call = direction.upper() in ("CALL", "BUY", "LONG")

        if not isinstance(forecast, dict) or not forecast or forecast.get("model_used") == "startup_default":
            return {"use_oracle": False, "reason": "Oracle not ready"}

        try:
            tp = float(forecast.get("exit_target", 0) or 0)
            sl = float(forecast.get("stop_loss", 0) or 0)
            hold_min = int(forecast.get("hold_minutes", 15) or 15)
        except (TypeError, ValueError):
            return {"use_oracle": False, "reason": "Oracle TP/SL fields are invalid"}
        if current_price <= 0 or tp <= 0 or sl <= 0 or hold_min <= 0:
            return {"use_oracle": False, "reason": "Oracle TP/SL fields are invalid"}

        # Validate geometry
        if is_call:
            # For a CALL, TP must be above price and SL must be below price
            if tp > current_price and 0 < sl < current_price:
                # Check that TP is within a sane scalping/swing limit (< 4% away)
                if (tp - current_price) / current_price < 0.04 and (current_price - sl) / current_price < 0.03:
                    return {
                        "use_oracle": True,
                        "tp_price": tp,
                        "sl_price": sl,
                        "hold_minutes": hold_min,
                        "reason": f"Oracle CALL target ${tp:,.0f} / stop ${sl:,.0f}"
                    }
        else:
            # For a PUT, TP must be below price and SL must be above price
            if 0 < tp < current_price and sl > current_price:
                if (current_price - tp) / current_price < 0.04 and (sl - current_price) / current_price < 0.03:
                    return {
                        "use_oracle": True,
                        "tp_price": tp,
                        "sl_price": sl,
                        "hold_minutes": hold_min,
                        "reason": f"Oracle PUT target ${tp:,.0f} / stop ${sl:,.0f}"
                    }

        return {"use_oracle": False, "reason": "Oracle TP/SL geometry incompatible with current price"}

    def get_gate_status(self) -> Dict[str, Any]:
        """Return status snapshot for UI and logs."""
        forecast = self.get_forecast()
        return {
            "enabled": self.enabled,
            "hard_gate": self.hard_gate,
            "has_forecast": bool(forecast and forecast.get("model_used") != "startup_default"),
            "trade_suggestion": forecast.get("trade_suggestion", "WAIT"),
            "5m_direction": forecast.get("5min", {}).get("direction", "UNKNOWN"),
            "30m_direction": forecast.get("30min", {}).get("direction", "UNKNOWN"),
            "gemini_summary": forecast.get("gemini_summary", "Oracle Standby")
        }


# Global singleton instance
_trade_gate_instance: Optional[OracleTradeGate] = None


def get_oracle_trade_gate(oracle_ref=None) -> OracleTradeGate:
    global _trade_gate_instance
    if _trade_gate_instance is None:
        _trade_gate_instance = OracleTradeGate(oracle_ref=oracle_ref)
    elif oracle_ref and not _trade_gate_instance._oracle_ref:
        _trade_gate_instance._oracle_ref = oracle_ref
    return _trade_gate_instance
