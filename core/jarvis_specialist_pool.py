"""Retired specialist-model committee compatibility API.

Model votes are neither a gate nor a deterministic score. Callers receive
explicit no-vote/NO_TRADE; the independent Laya adapter is audit-only.
"""
from typing import Any, Dict

class SpecialistPool:
    def __init__(self, max_workers: int = 3):
        pass

    def _ask_specialist(self, role: str, model: str,
                        context_packet: Dict[str, Any], prompt_instruction: str) -> str:
        return "[UNAVAILABLE] Model committee retired"

    def run_parallel_evaluation(self, context_packet: Dict[str, Any]) -> Dict[str, str]:
        return {key: "[UNAVAILABLE] Model committee retired"
                for key in ("analyst", "validator", "risk_officer")}

    def synthesize_chairman_decision(self, context_packet: Dict[str, Any], opinions: Dict[str, str]) -> Dict[str, Any]:
        return {"approved": False, "final_verdict": "NO_TRADE",
                "chairman_summary": "Model committee retired; no confirmation",
                "approve_votes": 0, "opinions": opinions}

    def shutdown(self) -> None:
        """No-op: pool has no live resources to release."""
        pass
