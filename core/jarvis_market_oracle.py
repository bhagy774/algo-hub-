#!/usr/bin/env python3
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
"""
JARVIS Market Oracle — Multi-Timeframe AI Market Map
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every 5 minutes, gathers ALL market data sources (Binance, Deribit, Delta,
Parts 1-12 opinions, Local AI Board) and queries Gemini 3.6 Flash
to generate a comprehensive time-by-time Market Map forecast.

Layers:
  1. Data Collection : Binance (MTF OHLCV, Depth, Funding, OI, 24h Vol)
                       Deribit (Max Pain, PCR, Gamma Walls, Greeks, IV)
                       Delta Exchange (Options Chain, Expiries)
                       Parts 1-12 (Live signals & thoughts)
  2. AI Analysis     : Local AI Board (Parallel DeepSeek + Qwen + Mistral)
  3. Gemini Oracle   : gemini-3.6-flash Multi-Timeframe Market Map JSON
  4. Auto-Trade Gate : Direct integration with OracleTradeGate (Gate 0.5)
  5. Output          : Console Market Map, CognitiveBus ORACLE_FORECAST, Web HUD, Telegram
"""

import os
import json
import time
import logging
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("JarvisMarketOracle")

# ── Configuration ─────────────────────────────────────────────
GEMINI_API_KEY          = os.environ.get("GEMINI_API_KEY", "")
ORACLE_GEMINI_MODEL     = os.environ.get("ORACLE_GEMINI_MODEL", "gemini-3.6-flash")
ORACLE_INTERVAL_SEC     = int(os.environ.get("ORACLE_INTERVAL_SEC", "300"))  # 5 minutes default
ORACLE_ENABLED          = os.environ.get("ORACLE_ENABLED", "true").lower() == "true"
ORACLE_HARD_GATE        = os.environ.get("ORACLE_HARD_GATE", "true").lower() == "true"
ORACLE_TELEGRAM_ALERTS  = os.environ.get("ORACLE_TELEGRAM_ALERTS", "false").lower() == "true"

# LegacyLLM Models for Board
MODEL_ANALYST   = os.environ.get("MODEL_ANALYST", "deepseek-r1:14b")
MODEL_VALIDATOR = os.environ.get("MODEL_VALIDATOR", "qwen2.5:14b")
MODEL_RISK      = os.environ.get("MODEL_RISK", "mistral-nemo:12b")
MODEL_CHAIRMAN  = os.environ.get("MODEL_CHAIRMAN", "qwen2.5:14b")

# ANSI Colors
R  = '\033[91m'
G  = '\033[92m'
Y  = '\033[93m'
C  = '\033[96m'
W  = '\033[97m'
DG = '\033[90m'
M  = '\033[95m'
BD = '\033[1m'
RST= '\033[0m'

ORACLE_SYSTEM_PROMPT = """You are JARVIS MARKET ORACLE — an elite crypto derivatives market forecasting intelligence system.
Your job is to analyze comprehensive multi-timeframe market data, options chain positioning (Deribit + Delta), order book microstructure, institutional flow, and multi-AI board decisions to generate an ultra-precise, time-by-time MARKET MAP for BTC.

CRITICAL REQUIREMENTS:
- Output MUST be valid, strictly compliant JSON ONLY. No markdown, no commentary outside JSON.
- Multi-timeframe directions must be one of: "BULLISH", "BEARISH", "NEUTRAL"
- "day": provide realistic expected price ranges (bear low, base expected, bull high) based on options max pain, gamma walls, and ATR.
- "trade_suggestion": "CALL", "PUT", or "WAIT"
- "entry_zone": price range {"price_from": float, "price_to": float} for the optimal trade entry.
- "exit_target": realistic take-profit price.
- "stop_loss": invalidation price for the trade suggestion.
- "hold_minutes": expected duration for the trade (5 to 120 minutes).
- "gemini_summary": concise 2-3 sentence strategic executive summary with actionable insight.

JSON Schema format:
{
  "5min": {
    "direction": "BULLISH | BEARISH | NEUTRAL",
    "range": "$XX,XXX-$YY,YYY",
    "confidence": 75,
    "reason": "<short explanation>"
  },
  "30min": {
    "direction": "BULLISH | BEARISH | NEUTRAL",
    "range": "$XX,XXX-$YY,YYY",
    "confidence": 70,
    "reason": "<short explanation>"
  },
  "1hour": {
    "direction": "BULLISH | BEARISH | NEUTRAL",
    "range": "$XX,XXX-$YY,YYY",
    "confidence": 65,
    "reason": "<short explanation>"
  },
  "4hour": {
    "direction": "BULLISH | BEARISH | NEUTRAL",
    "range": "$XX,XXX-$YY,YYY",
    "confidence": 60,
    "reason": "<short explanation>"
  },
  "day": {
    "bear": 00000,
    "base": 00000,
    "bull": 00000,
    "direction": "BULLISH | BEARISH | NEUTRAL",
    "key_events": ["<event 1>", "<event 2>", "<event 3>"]
  },
  "options_intel": {
    "max_pain": 00000,
    "gamma_wall_call": 00000,
    "gamma_wall_put": 00000,
    "pcr": 0.00,
    "sentiment": "Bullish | Bearish | Neutral"
  },
  "trade_suggestion": "CALL | PUT | WAIT",
  "entry_zone": {
    "price_from": 00000,
    "price_to": 00000
  },
  "exit_target": 00000,
  "stop_loss": 00000,
  "hold_minutes": 15,
  "gemini_summary": "<2-3 sentence strategic insight>"
}"""


class JarvisMarketOracle:
    """
    JARVIS Market Oracle service.
    Collects all market data -> LegacyLLM Committee -> Gemini 3.6 Flash -> Market Map JSON.
    Publishes to CognitiveBus, saves to disk, powers Gate 0.5 in live trader.
    """

    def __init__(self, bus=None, live_trader=None):
        self.bus = bus
        self.live_trader = live_trader
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Clients
        self._binance = None
        self._deribit = None
        self._delta = None
        self._upstox = None      # ← Upstox Indian market (optional)
        self._gemini_client = None
        self._gemini_types = None

        # State
        self.last_forecast: Dict[str, Any] = self._get_fallback_forecast()
        self.last_raw_data: Dict[str, Any] = {}
        self.last_update_ts: Optional[datetime] = None
        self.cycle_count = 0

        self._init_clients()

    def _init_clients(self):
        """Initialize data clients and Gemini SDK."""
        # 1. Binance Data
        try:
            from binance_data import get_binance_data
            self._binance = get_binance_data()
            logger.info("[Oracle] Binance data client ready")
        except Exception as e:
            logger.warning(f"[Oracle] Binance client init failed: {e}")

        # 2. Deribit Client
        try:
            from deribit_options_client import DeribitOptionsClient
            self._deribit = DeribitOptionsClient(currency="BTC")
            logger.info("[Oracle] Deribit options client ready")
        except Exception as e:
            logger.warning(f"[Oracle] Deribit client init failed: {e}")

        # 3. Delta Exchange Wrapper
        try:
            from delta_api_wrapper import DeltaExchangeData
            self._delta = DeltaExchangeData()
            logger.info("[Oracle] Delta Exchange client ready")
        except Exception as e:
            logger.warning(f"[Oracle] Delta client init failed: {e}")

        # 4. Upstox Indian Market (OPTIONAL — silently skips if disabled/no token)
        try:
            from upstox_data import get_upstox_data
            ud = get_upstox_data()
            if ud._enabled:
                self._upstox = ud
                logger.info("[Oracle] Upstox Indian market client ready ✅")
            else:
                logger.info("[Oracle] Upstox disabled (UPSTOX_ENABLED=false or no token)")
        except Exception as e:
            logger.info(f"[Oracle] Upstox not loaded (optional): {e}")

        # 5. Gemini SDK
        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types as genai_types
                self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                self._gemini_types = genai_types
                logger.info(f"[Oracle] Gemini SDK initialized ({ORACLE_GEMINI_MODEL})")
            except Exception as e:
                logger.error(f"[Oracle] Gemini SDK init error: {e}")
        else:
            logger.warning("[Oracle] No GEMINI_API_KEY found in .env")

    # ──────────────────────────────────────────────────────────
    #  LIFECYCLE MANAGEMENT
    # ──────────────────────────────────────────────────────────

    def start(self):
        """Start background 5-minute oracle forecasting loop."""
        if not ORACLE_ENABLED:
            logger.info("[Oracle] Disabled via ORACLE_ENABLED=false")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._oracle_loop, daemon=True, name="JarvisMarketOracle"
        )
        self._thread.start()
        logger.info(f"[Oracle] Background loop started (interval={ORACLE_INTERVAL_SEC}s)")

    def stop(self):
        """Gracefully stop background loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[Oracle] Stopped.")

    def get_latest_forecast(self) -> Dict[str, Any]:
        """Return the latest Market Map forecast (thread-safe)."""
        with self._lock:
            return dict(self.last_forecast)

    def run_now(self) -> Dict[str, Any]:
        """Manually trigger an immediate oracle forecasting cycle."""
        return self._run_oracle_cycle()

    # ──────────────────────────────────────────────────────────
    #  BACKGROUND LOOP
    # ──────────────────────────────────────────────────────────

    def _oracle_loop(self):
        """Loop running every ORACLE_INTERVAL_SEC."""
        # Initial slight delay (10s) to allow network and exchange connections to settle
        time.sleep(10)
        logger.info("[Oracle] Executing initial market map forecast...")

        while self._running:
            try:
                forecast = self._run_oracle_cycle()
                if forecast:
                    # Publish on CognitiveBus
                    if self.bus:
                        self.bus.publish("ORACLE_FORECAST", "JarvisMarketOracle", forecast)

                    # Save to disk
                    self._persist_forecast(forecast)

                    # Print formatted console market map (opt-in; BTC-only context)
                    if os.getenv("JARVIS_SHOW_ORACLE", "0").strip().lower() in ("1", "true", "yes", "on"):
                        self.print_market_map(forecast)

                    # Send Telegram alert if enabled
                    if ORACLE_TELEGRAM_ALERTS:
                        self._send_telegram_alert(forecast)

            except Exception as e:
                logger.error(f"[Oracle] Cycle execution error: {e}", exc_info=True)

            # Sleep in 2s intervals for responsiveness
            elapsed = 0
            while self._running and elapsed < ORACLE_INTERVAL_SEC:
                time.sleep(2)
                elapsed += 2

    # ──────────────────────────────────────────────────────────
    #  LAYER 1: DATA COLLECTION
    # ──────────────────────────────────────────────────────────

    def _collect_data(self) -> Dict[str, Any]:
        """
        Collect multi-source intelligence:
        - Binance MTF candles, Funding rate, OI, Order Book Depth
        - Deribit Max Pain, PCR, Gamma Walls, ATM Greeks, Smart Money
        - Delta Options Chain
        - Parts 1-12 live thoughts
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "btc_spot": 0.0,
            "candles": {},
            "microstructure": {},
            "deribit": {},
            "delta": {},
            "parts_opinions": {},
        }

        # 1. Binance MTF Candles & Spot
        if self._binance:
            try:
                spot = self._binance.get_live_price("BTCUSDT")
                data["btc_spot"] = spot
                # 1m, 5m, 15m, 1h, 4h
                mtf = self._binance.fetch_mtf_candles("BTCUSDT", ["1m", "5m", "15m", "1h", "4h"], limit=50)
                data["candles"] = {
                    "1m": [c["close"] for c in mtf.get("1m", [])[-30:]],
                    "5m": [c["close"] for c in mtf.get("5m", [])[-24:]],
                    "15m": [c["close"] for c in mtf.get("15m", [])[-16:]],
                    "1h": [c["close"] for c in mtf.get("1h", [])[-12:]],
                    "4h": [c["close"] for c in mtf.get("4h", [])[-6:]],
                }

                funding = self._binance.get_funding_rate("BTCUSDT")
                oi_binance = self._binance.get_open_interest("BTCUSDT")
                depth = self._binance.get_order_book_depth("BTCUSDT", limit=30)
                stats24 = self._binance.get_24h_stats("BTCUSDT")

                data["microstructure"] = {
                    "funding_rate": funding.get("description", "+0.0100%"),
                    "funding_countdown": funding.get("countdown_str", "unknown"),
                    "binance_oi_contracts": oi_binance.get("open_interest", 0.0),
                    "imbalance_pct": depth.get("imbalance_pct"),
                    "imbalance_bias": depth.get("bias", "NEUTRAL"),
                    "bid_wall": depth.get("major_bid_wall", {}).get("price", 0.0),
                    "ask_wall": depth.get("major_ask_wall", {}).get("price", 0.0),
                    "volume_24h_usdt": stats24.get("volume_24h_usdt", 0.0),
                    "high_24h": stats24.get("high_24h", spot),
                    "low_24h": stats24.get("low_24h", spot),
                    "change_24h_pct": stats24.get("price_change_pct", 0.0),
                }
            except Exception as e:
                logger.warning(f"[Oracle] Binance data collection warning: {e}")

        # Fallback spot if 0
        # Never fabricate a live spot price. Unavailable data yields WAIT.
        if not isinstance(data["btc_spot"], (int, float)) or data["btc_spot"] <= 0:
            data["btc_spot"] = 0.0

        spot = data["btc_spot"]

        # 2. Deribit Options Chain Intelligence
        if self._deribit:
            try:
                chain = self._deribit.get_option_chain() or {}
                greeks = self._deribit.get_greeks_and_iv() or {}
                smart_money = self._deribit.detect_smart_money(spot) or {}

                strikes = chain.get("strikes", [])
                call_oi = chain.get("call_oi", [])
                put_oi = chain.get("put_oi", [])

                top_calls = [s for s, _ in sorted(zip(strikes, call_oi), key=lambda x: x[1], reverse=True)[:3]] if strikes and call_oi else []
                top_puts = [s for s, _ in sorted(zip(strikes, put_oi), key=lambda x: x[1], reverse=True)[:3]] if strikes and put_oi else []

                data["deribit"] = {
                    "max_pain": chain.get("max_pain"),
                    "pcr": chain.get("pcr"),
                    "gamma_wall_call": chain.get("resistance_wall", round(spot * 1.02 / 1000) * 1000),
                    "gamma_wall_put": chain.get("support_wall", round(spot * 0.98 / 1000) * 1000),
                    "call_iv": greeks.get("call_iv", 55.0),
                    "put_iv": greeks.get("put_iv", 55.0),
                    "call_gamma": greeks.get("call_gamma", 0.0001),
                    "call_delta": greeks.get("call_delta", 0.50),
                    "top_calls": top_calls,
                    "top_puts": top_puts,
                    "smart_money": smart_money.get("details", "No anomalous flow detected"),
                }
            except Exception as e:
                logger.warning(f"[Oracle] Deribit collection warning: {e}")

        # 3. Delta Exchange Options
        if self._delta:
            try:
                d_chain = self._delta.get_options_chain("BTC") or {}
                data["delta"] = {
                "total_oi": round(float(d_chain.get("total_oi") or 0.0), 2),
                "pcr": round(float(d_chain.get("pcr") or 0.0), 3),
                    "calls_count": len(d_chain.get("calls", [])),
                    "puts_count": len(d_chain.get("puts", [])),
                }
            except Exception as e:
                logger.warning(f"[Oracle] Delta collection warning: {e}")

        # 4. Upstox Indian Market (OPTIONAL — only if UPSTOX_ENABLED=true)
        data["upstox"] = {}
        if self._upstox:
            try:
                snap = self._upstox.get_market_snapshot()
                data["upstox"] = {
                    "market_open":   snap.get("market_open", False),
                    "nifty50":       snap.get("nifty50", 0.0),
                    "banknifty":     snap.get("banknifty", 0.0),
                    "default_stock": snap.get("default_stock", 0.0),
                    "instruments":   snap.get("instruments", {}),
                }
                if snap.get("nifty50", 0) > 0:
                    logger.debug(f"[Oracle] Upstox: Nifty50={snap['nifty50']:.1f}, BankNifty={snap['banknifty']:.1f}")
            except Exception as e:
                logger.warning(f"[Oracle] Upstox collection warning: {e}")

        # 5. Parts 1-12 Opinions & Signals
        data["parts_opinions"] = self._collect_parts_opinions()


        self.last_raw_data = data
        return data

    def _collect_parts_opinions(self) -> Dict[str, str]:
        """Collect latest opinions from CognitiveBus or disk log."""
        opinions = {}
        # Try from CognitiveBus
        if self.bus and hasattr(self.bus, "get_latest_by_sender"):
            try:
                latest = self.bus.get_latest_by_sender()
                for sender, msg in latest.items():
                    if sender.startswith("Part") or "part" in sender.lower():
                        payload = msg.get("payload", "")
                        if isinstance(payload, dict):
                            payload = json.dumps(payload)
                        opinions[sender] = str(payload)[:250]
            except Exception:
                pass

        # If sparse, read from logs/jarvis_thoughts.log
        if len(opinions) < 3:
            log_path = os.path.join("logs", "jarvis_thoughts.log")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    thought_lines = [l for l in lines[-800:] if "[THOUGHTS]" in l]
                    for line in reversed(thought_lines):
                        parts = line.split("]")
                        if len(parts) >= 3:
                            sender = parts[2].strip().lstrip("[")
                            if sender not in opinions:
                                opinions[sender] = "]".join(parts[3:]).strip()[:250]
                except Exception:
                    pass

        # No synthetic opinions: absent published diagnostics remain absent.
        return opinions

    # ──────────────────────────────────────────────────────────
    #  LAYER 2: LEGACY_LLM MULTI-AI BOARD (PARALLEL)
    # ──────────────────────────────────────────────────────────

    def _run_ai_board(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retired multi-model vote; leave a neutral, non-confirming board."""
        return {"status": "unavailable", "analyst": "Model committee retired",
                "validator": "Model committee retired", "risk_officer": "Model committee retired",
                "chairman": "CONSENSUS_WAIT", "verdict": "WAIT", "confidence": 0}

    def _run_ollama_board(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility alias for retired model board."""
        return self._run_ai_board(market_data)

    def _call_gemini_oracle(self, market_data: Dict[str, Any], ai_board: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility shim: derive from observed data; no model forecast."""
        return self._generate_local_forecast(market_data, ai_board)

    def _call_ollama_oracle(self, market_data: Dict[str, Any], ai_board: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility shim: derive from observed data; no model forecast."""
        return self._generate_local_forecast(market_data, ai_board)

    def _build_gemini_prompt(self, market_data: Dict[str, Any], ai_board: Dict[str, Any]) -> str:
        """Construct the prompt matching the exact user specification."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        candles = market_data.get("candles", {})
        deribit = market_data.get("deribit", {})
        delta = market_data.get("delta", {})
        micro = market_data.get("microstructure", {})
        parts = market_data.get("parts_opinions", {})
        spot = market_data.get("btc_spot", 94250.0)

        # Liquidation estimate clusters around spot
        liq_longs = round(spot * 0.988 / 100) * 100
        liq_shorts = round(spot * 1.012 / 100) * 100

        parts_text = "\n".join([f"{k}: {v}" for k, v in list(parts.items())[:12]])

        prompt = f"""JARVIS MARKET ORACLE REQUEST — {now_str}

=== MULTI-TIMEFRAME PRICE DATA ===
Current Spot: ${spot:,.2f}
1min:  {candles.get('1m', [])[-15:]}
5min:  {candles.get('5m', [])[-12:]}
15min: {candles.get('15m', [])[-8:]}
1h:    {candles.get('1h', [])[-6:]}
4h:    {candles.get('4h', [])[-4:]}

=== OPTIONS CHAIN INTELLIGENCE ===
Deribit Max Pain:    ${deribit.get('max_pain', 95000):,}
Put/Call Ratio:      {deribit.get('pcr', 0.72):.2f}
Gamma Wall (Call):   ${deribit.get('gamma_wall_call', 95500):,}
Gamma Wall (Put):    ${deribit.get('gamma_wall_put', 93000):,}
IV (ATM Mark):       {deribit.get('call_iv', 60.0):.1f}%
Top Call OI Strikes: {deribit.get('top_calls', [])}
Top Put OI Strikes:  {deribit.get('top_puts', [])}
Delta Exchange OI:   {delta.get('total_oi', 0)} BTC (PCR: {delta.get('pcr', 0)})
Smart Money Flow:    {deribit.get('smart_money', 'Normal')}

=== MARKET MICROSTRUCTURE ===
Binance Funding Rate: {micro.get('funding_rate', '+0.0100%')} (Reset in {micro.get('funding_countdown', 'unknown')})
Order Book Imbalance: {micro.get('imbalance_pct', 0.0):+.2f}% ({micro.get('imbalance_bias', 'NEUTRAL')})
Major Order Book Walls: Bid at ${micro.get('bid_wall', 0):,}, Ask at ${micro.get('ask_wall', 0):,}
24h Volume:           ${micro.get('volume_24h_usdt', 0)/1e9:.2f}B (24h Change: {micro.get('change_24h_pct', 0):+.2f}%)
Liquidation Zones:    Longs at ${liq_longs:,}, Shorts at ${liq_shorts:,}

{self._format_upstox_section(market_data.get('upstox', {}))}

=== PARTS AI OPINIONS (12 systems) ===
{parts_text}

=== LEGACY_LLM BOARD DECISION ===
Analyst:   {ai_board.get('analyst', 'N/A')}
Validator: {ai_board.get('validator', 'N/A')}
Risk:      {ai_board.get('risk_officer', 'N/A')}
Chairman:  {ai_board.get('chairman', 'N/A')}

Provide JSON market forecast for ALL timeframes matching the requested schema.
"""
        return prompt

    def _format_upstox_section(self, upstox: Dict[str, Any]) -> str:
        """Format Upstox Indian market data for the prompt. Returns empty string if no data."""
        if not upstox or upstox.get("nifty50", 0) <= 0:
            return ""
        lines = ["=== INDIAN MARKET CONTEXT (Upstox NSE) ==="]
        lines.append(f"NSE Market:  {'OPEN 🟢' if upstox.get('market_open') else 'CLOSED 🔴'}")
        if upstox.get("nifty50", 0) > 0:
            lines.append(f"Nifty 50:    ₹{upstox['nifty50']:,.2f}")
        if upstox.get("banknifty", 0) > 0:
            lines.append(f"Bank Nifty:  ₹{upstox['banknifty']:,.2f}")
        if upstox.get("default_stock", 0) > 0:
            lines.append(f"Stock Price: ₹{upstox['default_stock']:,.2f}")
        # Add any extra configured instruments
        for sym, price in (upstox.get("instruments") or {}).items():
            if price > 0 and sym not in ("NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank"):
                short_name = sym.split("|")[-1] if "|" in sym else sym
                lines.append(f"  {short_name:20s}: ₹{price:,.2f}")
        return "\n".join(lines)



    def _call_ai_oracle(self, market_data: Dict[str, Any], ai_board: Dict[str, Any]) -> Dict[str, Any]:
        """Observed-market local synthesizer; never use model or cached model result."""
        return self._generate_local_forecast(market_data, ai_board)

    def _generate_local_forecast(self, market_data: Dict[str, Any], ai_board: Dict[str, Any]) -> Dict[str, Any]:
        """Unvalidated deterministic heuristic; only observed BTC data may produce a map."""
        # Incomplete, invalid, or unobserved sources cannot manufacture a CALL/PUT.
        import math
        spot = market_data.get("btc_spot") if isinstance(market_data, dict) else None
        deribit = market_data.get("deribit") if isinstance(market_data, dict) else None
        micro = market_data.get("microstructure") if isinstance(market_data, dict) else None
        deribit = deribit if isinstance(deribit, dict) else {}
        micro = micro if isinstance(micro, dict) else {}
        pcr = deribit.get("pcr")
        imbalance = micro.get("imbalance_pct")
        observed = (all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in (spot, pcr, imbalance))
                    and spot > 0 and 0 < pcr < 5 and -100 <= imbalance <= 100
                    and isinstance(deribit.get("max_pain"), (int, float))
                    and not isinstance(deribit.get("max_pain"), bool)
                    and math.isfinite(deribit["max_pain"]) and deribit["max_pain"] > 0)
        if not observed:
            return self._get_fallback_forecast()
        max_pain = deribit.get("max_pain", round(spot / 500) * 500)
        g_call = deribit.get("gamma_wall_call", round(spot * 1.015 / 100) * 100)
        g_put = deribit.get("gamma_wall_put", round(spot * 0.985 / 100) * 100)

        # Basic multi-timeframe heuristics
        dir_5m = "BULLISH" if (pcr < 0.8 and imbalance > -5) else ("BEARISH" if pcr > 1.1 or imbalance < -15 else "NEUTRAL")
        dir_30m = "BULLISH" if (spot < max_pain and pcr < 0.85) else ("BEARISH" if spot > max_pain and pcr > 1.0 else "NEUTRAL")
        dir_1h = "BULLISH" if spot > g_put * 1.01 else "NEUTRAL"
        dir_4h = "NEUTRAL" if abs(spot - max_pain) / spot < 0.03 else ("BULLISH" if spot < max_pain else "BEARISH")

        suggestion = "CALL" if dir_5m == "BULLISH" and dir_30m == "BULLISH" else ("PUT" if dir_5m == "BEARISH" and dir_30m == "BEARISH" else "WAIT")

        spread_5m = spot * 0.003
        spread_30m = spot * 0.008
        spread_1h = spot * 0.012
        spread_4h = spot * 0.02

        tp = round(spot * 1.006, 1) if suggestion == "CALL" else (round(spot * 0.994, 1) if suggestion == "PUT" else 0)
        sl = round(spot * 0.996, 1) if suggestion == "CALL" else (round(spot * 1.004, 1) if suggestion == "PUT" else 0)

        return {
            "5min": {
                "direction": dir_5m,
                "range": f"${spot - spread_5m:,.0f}-${spot + spread_5m:,.0f}",
                "confidence": 76 if dir_5m != "NEUTRAL" else 62,
                "reason": f"Orderbook imbalance at {imbalance:+.1f}% with options PCR at {pcr:.2f}."
            },
            "30min": {
                "direction": dir_30m,
                "range": f"${spot - spread_30m:,.0f}-${spot + spread_30m:,.0f}",
                "confidence": 71,
                "reason": f"Price gravitating toward Max Pain at ${max_pain:,.0f}."
            },
            "1hour": {
                "direction": dir_1h,
                "range": f"${spot - spread_1h:,.0f}-${spot + spread_1h:,.0f}",
                "confidence": 65,
                "reason": f"Holding above key put gamma wall support at ${g_put:,.0f}."
            },
            "4hour": {
                "direction": dir_4h,
                "range": f"${spot - spread_4h:,.0f}-${spot + spread_4h:,.0f}",
                "confidence": 60,
                "reason": "Institutional macro positioning remains balanced within consolidation band."
            },
            "day": {
                "bear": round(spot * 0.975 / 100) * 100,
                "base": round(max_pain / 100) * 100,
                "bull": round(spot * 1.025 / 100) * 100,
                "direction": "NEUTRAL",
                "key_events": [
                    f"Funding Reset: {micro.get('funding_countdown', 'within 8h')} (Rate: {micro.get('funding_rate')})",
                    "US Market Session Volatility Band",
                    f"Deribit Major OI Expiry Cluster around ${max_pain:,.0f}"
                ]
            },
            "options_intel": {
                "max_pain": max_pain,
                "gamma_wall_call": g_call,
                "gamma_wall_put": g_put,
                "pcr": pcr,
                "sentiment": "Bullish" if pcr < 0.8 else ("Bearish" if pcr > 1.1 else "Neutral")
            },
            "trade_suggestion": suggestion,
            "entry_zone": {
                "price_from": round(spot * 0.9985, 1),
                "price_to": round(spot * 1.0015, 1)
            },
            "exit_target": tp,
            "stop_loss": sl,
            "hold_minutes": 15,
            "gemini_summary": f"Short-term momentum is {dir_5m.lower()} with PCR {pcr:.2f}. Gamma wall at ${g_call:,.0f} serves as major ceiling; watch support at ${g_put:,.0f}.",
            "model_used": "local_synthesizer",
            "generated_at": datetime.now().isoformat()
        }

    # ──────────────────────────────────────────────────────────
    #  EXECUTION CYCLE
    # ──────────────────────────────────────────────────────────

    def _run_oracle_cycle(self) -> Dict[str, Any]:
        """Complete 5-minute pipeline execution."""
        start_time = time.time()
        logger.info("[Oracle] Starting Layer 1 Data Collection...")
        data = self._collect_data()

        logger.info("[Oracle] Model committee retired; deriving from observed data only...")
        board = self._run_ai_board(data)

        logger.info("[Oracle] Building deterministic local market map...")
        forecast = self._call_ai_oracle(data, board)

        # Store spot & raw meta in forecast for display
        # This service currently collects BTC-only derivatives context. Keep
        # the scope explicit so callers cannot mistake it for an altcoin map.
        forecast["asset"] = "BTC"
        forecast["market_scope"] = "btc_macro_only"
        forecast["btc_spot"] = data.get("btc_spot", 0.0)
        forecast["cycle_number"] = self.cycle_count + 1
        forecast["duration_sec"] = round(time.time() - start_time, 2)

        with self._lock:
            self.last_forecast = forecast
            self.last_update_ts = datetime.now()
            self.cycle_count += 1

        return forecast

    def _persist_forecast(self, forecast: Dict[str, Any]):
        """Save latest forecast to logs/jarvis_market_map.json."""
        try:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", "jarvis_market_map.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(forecast, f, indent=2)
        except Exception as e:
            logger.debug(f"[Oracle] Could not persist market map to file: {e}")

    # ──────────────────────────────────────────────────────────
    #  LAYER 4: CONSOLE DISPLAY (MARKET MAP)
    # ──────────────────────────────────────────────────────────

    def print_market_map(self, forecast: Optional[Dict[str, Any]] = None):
        """
        Renders the exact formatted ASCII Market Map requested by the user.
        """
        fc = forecast or self.get_latest_forecast()
        now_str = datetime.now().strftime("%H:%M IST")
        spot = fc.get("btc_spot", 0.0)
        opt = fc.get("options_intel", {})
        max_pain = opt.get("max_pain", 0)
        pcr = opt.get("pcr", 0.72)
        sentiment = opt.get("sentiment", "Neutral")
        iv_val = opt.get("iv", "58%")
        if isinstance(iv_val, (int, float)):
            iv_str = f"{iv_val:.0f}%"
        else:
            iv_str = str(iv_val)

        tf5 = fc.get("5min", {})
        tf30 = fc.get("30min", {})
        tf1h = fc.get("1hour", {})
        tf4h = fc.get("4hour", {})
        day = fc.get("day", {})

        def _dir_col(d: str) -> str:
            d_u = str(d).upper()
            if "BULL" in d_u:
                return f"{G}{BD}{d_u:<8}{RST}"
            elif "BEAR" in d_u:
                return f"{R}{BD}{d_u:<8}{RST}"
            return f"{Y}{BD}{d_u:<8}{RST}"

        print("\n" + f"{C}╔════════════════════════════════════════════════════════════════════════════════════╗{RST}")
        print(f"{C}║{RST}{BD}                    JARVIS MARKET ORACLE  [{now_str}]                         {RST}{C}║{RST}")
        print(f"{C}╠════════════════════════════════════════════════════════════════════════════════════╣{RST}")
        print(f"  {DG}BTC Spot   :{RST} {BD}${spot:,.0f}{RST}        {DG}Options Max Pain:{RST} {BD}${max_pain:,.0f}{RST}")
        print(f"  {DG}IV (30d)   :{RST} {W}{iv_str}{RST}             {DG}Put/Call Ratio  :{RST} {W}{pcr:.2f} ({sentiment}){RST}")
        print(f"{C}╟────────────────────────────────────────────────────────────────────────────────────╢{RST}")
        print(f"  {BD}TIMEFRAME     DIRECTION   RANGE                 CONFIDENCE{RST}")
        print(f"  {DG}Next 5 min   {RST} {_dir_col(tf5.get('direction', 'NEUTRAL'))}  {W}{tf5.get('range', 'N/A'):<20}{RST}  {BD}{tf5.get('confidence', 0)}%{RST}")
        print(f"  {DG}Next 30 min  {RST} {_dir_col(tf30.get('direction', 'NEUTRAL'))}  {W}{tf30.get('range', 'N/A'):<20}{RST}  {BD}{tf30.get('confidence', 0)}%{RST}")
        print(f"  {DG}Next 1 hour  {RST} {_dir_col(tf1h.get('direction', 'NEUTRAL'))}  {W}{tf1h.get('range', 'N/A'):<20}{RST}  {BD}{tf1h.get('confidence', 0)}%{RST}")
        print(f"  {DG}Next 4 hour  {RST} {_dir_col(tf4h.get('direction', 'NEUTRAL'))}  {W}{tf4h.get('range', 'N/A'):<20}{RST}  {BD}{tf4h.get('confidence', 0)}%{RST}")

        day_dir = day.get('direction', 'NEUTRAL')
        print(f"  {DG}Full Day     {RST} {_dir_col(day_dir)}  {DG}BEAR:{RST}${day.get('bear', 0):,.0f}  {DG}BASE:{RST}${day.get('base', 0):,.0f}  {DG}BULL:{RST}${day.get('bull', 0):,.0f}")
        print(f"{C}╟────────────────────────────────────────────────────────────────────────────────────╢{RST}")
        print(f"  {BD}KEY EVENTS TODAY{RST}")
        events = day.get("key_events", [])
        if events:
            for ev in events[:4]:
                print(f"  {Y}•{RST} {W}{ev}{RST}")
        else:
            print(f"  {DG}• Funding rate resets every 8 hours | Monitor US market session volume{RST}")

        print(f"{C}╟────────────────────────────────────────────────────────────────────────────────────╢{RST}")
        print(f"  {BD}OPTIONS INTEL{RST}")
        print(f"  {DG}Max Pain   :{RST} ${max_pain:,.0f} (market gravitates toward this level)")
        raw_deribit = self.last_raw_data.get("deribit", {})
        top_c = raw_deribit.get("top_calls", [round(spot*1.02/1000)*1000, round(spot*1.04/1000)*1000])
        top_p = raw_deribit.get("top_puts", [round(spot*0.98/1000)*1000, round(spot*0.96/1000)*1000])
        print(f"  {DG}Big Call OI:{RST} {', '.join(f'${c:,.0f}' for c in top_c[:3])}")
        print(f"  {DG}Big Put OI :{RST} {', '.join(f'${p:,.0f}' for p in top_p[:3])}")
        g_call = opt.get("gamma_wall_call", 0)
        g_put = opt.get("gamma_wall_put", 0)
        print(f"  {DG}Gamma Walls:{RST} Call: ${g_call:,.0f} (Resistance) | Put: ${g_put:,.0f} (Support)")

        summary = fc.get("gemini_summary", "Market Map calculated. Monitor entry zones and key walls.")
        print(f"{C}╟────────────────────────────────────────────────────────────────────────────────────╢{RST}")
        trade_sugg = fc.get("trade_suggestion", "WAIT")
        ez = fc.get("entry_zone", {})
        ez_from = ez.get("price_from", 0) or day.get("bear", 0) or g_put
        ez_to = ez.get("price_to", 0) or day.get("bull", 0) or g_call
        tp_val = fc.get("exit_target", 0) or g_call or day.get("bull", 0)
        sl_val = fc.get("stop_loss", 0) or g_put or day.get("bear", 0)

        sugg_col = G if trade_sugg == "CALL" else (R if trade_sugg == "PUT" else Y)
        print(f"  {BD}BTC MACRO TREND:{RST} {sugg_col}{BD}[{trade_sugg}]{RST} | Focus Zone: ${ez_from:,.0f}-${ez_to:,.0f} | Key Res: ${tp_val:,.0f} | Key Sup: ${sl_val:,.0f}")
        print(f"\n  {M}{BD}MACRO AI SAYS:{RST} \"{W}{summary}{RST}\"")

        # ── Indian Market Context (Upstox) ──────────────────────────────
        upstox_data = self.last_raw_data.get("upstox", {})
        if upstox_data.get("nifty50", 0) > 0:
            mkt_str = f"{G}OPEN 🟢{RST}" if upstox_data.get("market_open") else f"{DG}CLOSED{RST}"
            print(f"{C}╟────────────────────────────────────────────────────────────────────────────────────╢{RST}")
            print(f"  {BD}INDIAN MARKET (NSE via Upstox){RST}  {mkt_str}")
            print(f"  {DG}Nifty 50   :{RST} {W}₹{upstox_data['nifty50']:,.2f}{RST}  {DG}Bank Nifty:{RST} {W}₹{upstox_data.get('banknifty', 0):,.2f}{RST}")
            if upstox_data.get("default_stock", 0) > 0:
                print(f"  {DG}Stock Price:{RST} {W}₹{upstox_data['default_stock']:,.2f}{RST}")

        print(f"{C}╚════════════════════════════════════════════════════════════════════════════════════╝{RST}\n")

    # ──────────────────────────────────────────────────────────
    #  TELEGRAM NOTIFICATION
    # ──────────────────────────────────────────────────────────

    def _send_telegram_alert(self, forecast: Dict[str, Any]):
        """Send a formatted telegram message with the market forecast."""
        try:
            from telegram_notifier import send_message
            now_str = datetime.now().strftime("%H:%M IST")
            spot = forecast.get("btc_spot", 0)
            tf5 = forecast.get("5min", {})
            tf30 = forecast.get("30min", {})
            sugg = forecast.get("trade_suggestion", "WAIT")
            summary = forecast.get("gemini_summary", "")

            msg = (
                f"🔮 *JARVIS MARKET ORACLE* [{now_str}]\n\n"
                f"• *BTC Spot*: `${spot:,.0f}`\n"
                f"• *Next 5m*: `{tf5.get('direction')}` ({tf5.get('confidence')}%) | `{tf5.get('range')}`\n"
                f"• *Next 30m*: `{tf30.get('direction')}` ({tf30.get('confidence')}%) | `{tf30.get('range')}`\n"
                f"• *Trade Signal*: *{sugg}*\n\n"
                f"💡 _{summary}_"
            )
            send_message(msg)
        except Exception as e:
            logger.debug(f"[Oracle] Telegram alert error: {e}")

    def _get_fallback_forecast(self) -> Dict[str, Any]:
        """Default baseline forecast used on startup before first cycle completes."""
        return {
            "5min": {"direction": "NEUTRAL", "range": "Calculating...", "confidence": 50, "reason": "Oracle initializing"},
            "30min": {"direction": "NEUTRAL", "range": "Calculating...", "confidence": 50, "reason": "Oracle initializing"},
            "1hour": {"direction": "NEUTRAL", "range": "Calculating...", "confidence": 50, "reason": "Oracle initializing"},
            "4hour": {"direction": "NEUTRAL", "range": "Calculating...", "confidence": 50, "reason": "Oracle initializing"},
            "day": {"bear": 0, "base": 0, "bull": 0, "direction": "NEUTRAL", "key_events": ["Oracle initializing"]},
            "options_intel": {"max_pain": 0, "gamma_wall_call": 0, "gamma_wall_put": 0, "pcr": 0.0, "sentiment": "Neutral"},
            "trade_suggestion": "WAIT",
            "entry_zone": {"price_from": 0, "price_to": 0},
            "exit_target": 0,
            "stop_loss": 0,
            "hold_minutes": 15,
            "gemini_summary": "JARVIS Market Oracle initializing first cycle...",
            "asset": "BTC",
            "market_scope": "btc_macro_only",
            "btc_spot": 0.0,
            "model_used": "startup_default"
        }


# ── Global Singleton ──────────────────────────────────────────
_oracle_instance: Optional[JarvisMarketOracle] = None


def get_oracle() -> Optional[JarvisMarketOracle]:
    """Retrieve global singleton Oracle instance."""
    return _oracle_instance


def init_oracle(bus=None, live_trader=None) -> JarvisMarketOracle:
    """Initialize and start the global JARVIS Market Oracle singleton."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = JarvisMarketOracle(bus=bus, live_trader=live_trader)
        _oracle_instance.start()
    return _oracle_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Testing JARVIS Market Oracle...")
    oracle = JarvisMarketOracle()
    fc = oracle.run_now()
    oracle.print_market_map(fc)
