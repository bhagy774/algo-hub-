"""Safe, central configuration for optional multi-timeframe analysis.

This module deliberately contains no client creation or network activity.  Analysis
may be enabled independently of order execution; its default is always paper mode.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TradingConfig:
    mode: str = "paper"
    paper_trading: bool = True
    auto_trade: bool = False
    live_execution: bool = False
    default_symbol: str = "BTCUSDT"
    scalping_timeframes: Tuple[str, ...] = ("1m", "5m", "15m")
    swing_timeframes: Tuple[str, ...] = ("15m", "1h", "4h")
    candle_limit: int = 200
    atr_period: int = 14
    scalping_stop_atr: float = 1.0
    scalping_target_atr: float = 1.8
    swing_stop_atr: float = 1.5
    swing_target_atr: float = 3.0
    min_risk_reward: float = 1.2
    min_timeframes: int = 2
    default_atr_pct: float = 0.003

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_active_config() -> Dict[str, Any]:
    """Return a fresh, safe configuration mapping.

    A request for a live mode is intentionally not an order-execution opt-in.  The
    execution layer has separate guards; this analysis config remains paper-only.
    """
    config = TradingConfig().to_dict()
    requested_mode = os.environ.get("JARVIS_ANALYSIS_MODE", "paper").lower()
    config["mode"] = requested_mode if requested_mode in {"paper", "backtest"} else "paper"
    return config


# Backwards-compatible mapping consumed by jarvis_FIXED.py.
TRADING_CONFIG: Dict[str, Any] = get_active_config()


# ── AI PROVIDER ENVIRONMENT VARIABLES (all safe defaults = local LegacyLLM only) ──
# LEGACY_LLM_BASE_URL            Local LegacyLLM server URL. Default: http://localhost:11434
# JARVIS_ENABLE_GEMINI       "1" to opt in to the external Gemini Supreme Advisor
#                            (also requires GEMINI_API_KEY). Default: off (LegacyLLM-only).
# JARVIS_ENABLE_EXTERNAL_AI  "1" to opt in to external AI clients (e.g. KIE GPT-6).
#                            Default: off — external clients return a "disabled" result
#                            and never make a network call.
# JARVIS_WATCHDOG            "0" to disable the crash-recovery watchdog. Default: on.
# JARVIS_DATA_VALIDATOR      "0" to disable the pre-brain data quality gate
#                            (completeness, price sanity, staleness, cross-source).
#                            Default: on.
AI_ENV_DEFAULTS: Dict[str, str] = {
    "LEGACY_LLM_BASE_URL": "http://localhost:11434",
    "JARVIS_ENABLE_GEMINI": "0",
    "JARVIS_ENABLE_EXTERNAL_AI": "0",
    "JARVIS_WATCHDOG": "1",
    "JARVIS_DATA_VALIDATOR": "1",
}


def external_ai_enabled() -> bool:
    """True only when the user explicitly opted in to external AI calls."""
    return os.environ.get("JARVIS_ENABLE_EXTERNAL_AI", "0") == "1"


def gemini_enabled() -> bool:
    """True only with explicit opt-in AND an API key present."""
    return (
        os.environ.get("JARVIS_ENABLE_GEMINI", "0") == "1"
        and bool(os.environ.get("GEMINI_API_KEY", "").strip())
    )
