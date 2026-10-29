#!/usr/bin/env python3
"""
AI Hedge Advisor - Brain for Options Hedged Scalp Strategy
Uses LegacyLLM (deepseek-r1:14b) with memory to make hedging decisions.
"""

import os
import re
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

LEGACY_LLM_BASE_URL = os.environ.get("LEGACY_LLM_BASE_URL", "http://localhost:11434")
LEGACY_LLM_MODEL    = os.environ.get("LEGACY_LLM_MODEL", "deepseek-r1:14b")

class AIHedgeAdvisor:
    """
    AI brain that decides how to hedge futures scalp trades with options.
    Uses rolling chat history to maintain market context memory.
    """

    SYSTEM_PROMPT = (
        "You are JARVIS HEDGE ADVISOR, an elite BTC options hedging AI.\n"
        "Your job: Protect scalp trades using options as insurance.\n\n"
        "You receive:\n"
        "- Main trade direction and confidence\n"
        "- Full options chain with Greeks (Delta, IV, OI)\n"
        "- ATR-based volatility context\n"
        "- Institutional positioning (whale walls)\n\n"
        "HEDGE RULES:\n"
        "- ALWAYS hedge if confidence < 75%\n"
        "- SMART hedge if volatility HIGH (ATR > 1.5%)\n"
        "- SKIP hedge if confidence >= 90% and volatility LOW\n"
        "- Pick strikes 0.3-0.7 ATR from entry (OTM = cheaper premium)\n"
        "- Prefer nearest expiry for scalps (cheaper time value)\n"
        "- Premium budget: max 20% of expected profit\n"
        "- If IV > 80th percentile: SELL option instead (premium harvesting)\n\n"
        "RESPONSE FORMAT - Reply ONLY in valid JSON:\n"
        '{"hedge": "YES", "strike": 86500, "expiry": "nearest", "premium_budget_pct": 15, "hedge_ratio": 0.75, "reason": "High IV selling opportunity, protective put at 0.5 ATR below entry"}\n'
        'hedge: exactly "YES", "NO", or "SELL_PREMIUM"\n'
        "strike: integer representing best strike price to use\n"
        'expiry: string like "nearest", "weekly", "monthly"\n'
        "premium_budget_pct: 0-100 integer representing max % of expected profit to spend\n"
        "hedge_ratio: float between 0.1 and 1.0 representing size of hedge relative to main position\n"
        "reason: concise explanation (max 40 words)"
    )

    def __init__(self):
        self.chat_history: List[Dict] = []
        self.max_history_pairs = 10
        self.model = getattr(self, "LEGACY_LLM_MODEL", "laya-retired")
        self._call_count = 0
        logger.info(f"[HEDGE-AI] Initialized -- model={self.model} memory={self.max_history_pairs} trades")

    def evaluate_hedge_setup(self, signal_direction: str, signal_confidence: int,
                             current_price: float, atr: float, options_chain: Dict,
                             expected_profit: float, jarvis_result: dict = None) -> Dict:
        """No model-selected strike, budget, ratio or hedge direction."""
        return {"hedge": "NO", "strike": None, "hedge_ratio": 0,
                "reason": "Model hedge adviser retired; no validated deterministic hedge policy"}

    def monitor_active_hedge(self, hedge_position: Dict, current_price: float,
                             main_trade_pnl: float, hedge_pnl: float) -> Dict:
        """Protective futures TP/SL remains in the paper monitor; no model exit."""
        return {"action": "HOLD", "reason": "Model exit adviser retired"}

    def _build_pre_trade_report(self, direction, confidence, price, atr, options_chain, expected_profit, jarvis_result) -> str:
        now = datetime.now().strftime('%H:%M:%S')
        
        # Summarize options chain to avoid context overflow
        calls_summary = []
        puts_summary = []
        
        # Just grab the 3 closest strikes above and below
        if options_chain and "calls" in options_chain and "puts" in options_chain:
            sorted_calls = sorted(options_chain.get("calls", []), key=lambda x: abs(x.get("strike", 0) - price))[:3]
            sorted_puts = sorted(options_chain.get("puts", []), key=lambda x: abs(x.get("strike", 0) - price))[:3]
            
            calls_summary = [f"CALL {c.get('strike')} | Premium: {c.get('price')} | IV: {c.get('iv', 0):.2f}" for c in sorted_calls]
            puts_summary = [f"PUT {p.get('strike')} | Premium: {p.get('price')} | IV: {p.get('iv', 0):.2f}" for p in sorted_puts]

        # Extract intelligence from Jarvis Parts
        intelligence_str = "No additional AI context available."
        if jarvis_result:
            board = jarvis_result.get('intelligence_board', [])
            if board:
                intelligence_str = "\n".join([f"- {thought[:100]}" for thought in board[:8]])
            
            ctx = jarvis_result.get('market_context', {})
            if ctx:
                intelligence_str += f"\nMarket Context: Regime={ctx.get('regime', 'N/A')} | Volatility={ctx.get('volatility', 'N/A')}"

        report = (
            f"\n{now} | PRE-TRADE HEDGE EVALUATION | Call #{self._call_count}\n"
            f"Trade: {direction} BTCUSDT @ ${price:,.2f}\n"
            f"Confidence: {confidence}%\n"
            f"Expected Scalp Profit: ${expected_profit:,.2f}\n"
            f"ATR(14): ${atr:,.2f}\n\n"
            f"INTELLIGENCE BOARD (From 14 Parts):\n"
            f"{intelligence_str}\n\n"
            f"OPTIONS CHAIN CONTEXT:\n"
            f"Nearby Calls:\n" + "\n".join(calls_summary) + "\n"
            f"Nearby Puts:\n" + "\n".join(puts_summary) + "\n\n"
            f"DECIDE: Should we hedge this {direction} trade? With what strike?\n"
            f"Reply JSON ONLY per the system prompt format."
        )
        return report

    
    def _parse_hedge_decision(self, raw: Optional[str], current_price: float) -> Dict:
        if not raw:
            return self._fallback_decision("NO_TRADE", 0, current_price, 0)
            
        try:
            data = json.loads(raw)
            hedge = str(data.get("hedge", "NO")).upper()
            if hedge not in ("YES", "NO", "SELL_PREMIUM"):
                hedge = "NO"
                
            try:
                strike = int(data.get("strike", 0))
            except ValueError:
                strike = 0
                
            try:
                budget = int(data.get("premium_budget_pct", 15))
            except ValueError:
                budget = 15
                
            try:
                ratio = float(data.get("hedge_ratio", 0.5))
            except ValueError:
                ratio = 0.5
                
            return {
                "hedge": hedge,
                "strike": strike,
                "expiry": str(data.get("expiry", "nearest")),
                "premium_budget_pct": max(0, min(100, budget)),
                "hedge_ratio": max(0.1, min(1.0, ratio)),
                "reason": str(data.get("reason", ""))[:100]
            }
        except json.JSONDecodeError:
            # Regex fallback
            logger.warning(f"[HEDGE-AI] Failed to parse JSON: {raw[:100]}")
            return self._fallback_decision("UNKNOWN", 50, current_price, 0)

    def _fallback_decision(self, direction: str, confidence: int, price: float, atr: float) -> Dict:
        """Math fallback if AI fails"""
        hedge = "YES" if confidence < 80 else "NO"
        
        # Simple math strike: 0.5 ATR OTM
        strike_offset = (atr * 0.5) if atr > 0 else (price * 0.005)
        if direction == "CALL" or direction == "BUY":
            strike = int((price - strike_offset) / 100) * 100  # Round to nearest 100
        else:
            strike = int((price + strike_offset) / 100) * 100
            
        return {
            "hedge": hedge,
            "strike": strike,
            "expiry": "nearest",
            "premium_budget_pct": 15,
            "hedge_ratio": 0.5,
            "reason": "Math fallback due to AI parse error"
        }
