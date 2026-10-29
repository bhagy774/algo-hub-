"""Retired model roundtable: no model result can confirm a trade."""
from typing import Any, Dict

def run_ai_roundtable(market_context: Dict[str, Any], signal_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"approved": False, "final_verdict": "NO_TRADE",
            "chairman_summary": "Model roundtable retired; no confirmation",
            "approve_votes": 0, "opinions": {}}
