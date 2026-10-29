#!/usr/bin/env python3
"""Shared balance/risk/margin-aware position sizing for JARVIS."""
import logging
from typing import Dict, Optional

from jarvis_risk import (
    BASE_RISK_PCT, MAX_RISK_PCT, MIN_MARGIN_USDT, COMPOUND_RATIO,
    MAX_LEVERAGE_CAP, CONF_MULTIPLIERS, calculate_trade_size,
)

logger = logging.getLogger("JarvisSizer")
# Compatibility name: this is a policy cap, never an operating leverage value.
LEVERAGE_CAP = MAX_LEVERAGE_CAP

class JarvisSizer:
    def __init__(self, delta_client):
        self.delta = delta_client
        self._cached_balance: float = 0.0
        self._compound_pool: float = 0.0

    def set_compound_pool(self, amount: float):
        try:
            self._compound_pool = max(0.0, float(amount))
        except (TypeError, ValueError):
            self._compound_pool = 0.0

    def get_live_balance(self) -> float:
        try:
            _chk = getattr(self.delta, 'get_wallet_balance_checked', None)
            bal = _chk() if callable(_chk) else self.delta.get_wallet_balance()
            if bal is None:
                logger.warning("[Sizer] API_UNAVAILABLE: Delta balance unreadable; sizing fails closed (stale cache NOT used, not a verified 0.0)")
                return 0.0
            if bal is not None and float(bal) > 0:
                self._cached_balance = float(bal)
                return self._cached_balance
        except Exception as e:
            logger.warning("[Sizer] Balance fetch failed: %s", e)
        return self._cached_balance

    def calculate_size(self, confidence: int, symbol: str = "BTCUSDT",
                       force_balance: float = None, stop_distance_pct: float = 0.002,
                       max_trade_risk_usdt: float = None,
                       product_max_leverage: float = None,
                       contract_value_usdt: float = None,
                       require_contract_value: bool = False) -> Dict:
        balance = force_balance if force_balance is not None else self.get_live_balance()
        result = calculate_trade_size(
            balance, confidence, stop_distance_pct,
            max_trade_risk_usdt=max_trade_risk_usdt,
            compound_pool=self._compound_pool,
            product_max_leverage=product_max_leverage,
            contract_value_usdt=contract_value_usdt,
            require_contract_value=require_contract_value,
        )
        result.setdefault("balance", float(balance or 0))
        result.setdefault("symbol", symbol)
        result.setdefault("compound_used", 0.0)
        result.setdefault("multiplier", 0.0)
        result.setdefault("margin_usdt", 0.0)
        result.setdefault("notional_usdt", 0.0)
        result.setdefault("contracts", 0)
        result.setdefault("leverage", 0)
        result.setdefault("confidence", int(confidence) if str(confidence).lstrip('-').isdigit() else 0)
        if not result.get("ok"):
            result.setdefault("sizing_note", result.get("reason", "Sizing blocked"))
        return result

    def print_sizing(self, result: Dict):
        print(
            f"  SIZER: Conf={result.get('confidence', 0)}% | "
            f"Margin=${result.get('margin_usdt', 0):.4f} | "
            f"{result.get('contracts', 0)} contracts | "
            f"Notional=${result.get('notional_usdt', 0):.2f} | "
            f"Auto leverage={result.get('leverage', 0)}x"
        )

_sizer_instance: Optional[JarvisSizer] = None

def get_sizer(delta_client=None) -> Optional[JarvisSizer]:
    global _sizer_instance
    if _sizer_instance is None and delta_client is not None:
        _sizer_instance = JarvisSizer(delta_client)
    return _sizer_instance

if __name__ == "__main__":
    print("JarvisSizer module loaded OK; leverage is derived per trade")
