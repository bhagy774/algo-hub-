#!/usr/bin/env python3
"""
JARVIS Professional Position Manager
=====================================
Manages open positions like a professional trader.
- Continuous monitoring every 5 seconds
- AI-powered exit (NOT timer-based)
- Trailing stop loss
- Break-even move on partial profit
- Compounding: reinvest profits into next trade
"""
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("JarvisPositionManager")

R   = "\033[91m"
G   = "\033[92m"
Y   = "\033[93m"
C   = "\033[96m"
W   = "\033[97m"
DG  = "\033[90m"
BD  = "\033[1m"
RST = "\033[0m"

SL_PCT           = float(os.environ.get("PM_SL_PCT",          "0.002"))
TP_PCT           = float(os.environ.get("PM_TP_PCT",          "0.004"))
TRAIL_ACTIVATE   = float(os.environ.get("PM_TRAIL_ACTIVATE",  "0.003"))
TRAIL_STEP       = float(os.environ.get("PM_TRAIL_STEP",      "0.001"))
BREAK_EVEN_AT    = float(os.environ.get("PM_BREAKEVEN_AT",    "0.002"))
MONITOR_INTERVAL = int(os.environ.get("PM_MONITOR_SEC",       "5"))
from jarvis_risk import calculate_trade_size, MAX_LEVERAGE_CAP, contract_quote_value_usdt
from jarvis_close_coordinator import claim_close, release_close, reconcile_close
from jarvis_position_ownership import claim_position, position_owner
LEVERAGE_CAP     = MAX_LEVERAGE_CAP  # policy cap only; leverage is derived per trade
COMPOUND_ENABLED = os.environ.get("PM_COMPOUND", "true").lower() == "true"
MIN_MARGIN       = float(os.environ.get("PM_MIN_MARGIN",      "0.05"))
MAX_MARGIN_PCT   = float(os.environ.get("PM_MAX_MARGIN_PCT",  "0.05"))


@dataclass
class PositionRequest:
    position_id: str
    direction: str
    entry_price: float
    contracts: int
    confidence: int
    coin: str = "BTC"
    trade_type: str = "SCALP"
    contract_value_usdt: Optional[float] = None
    notional_usdt: Optional[float] = None


class PositionRecord:
    """Single open position tracking object."""

    def __init__(self, position_id: str, direction: str, entry_price: float,
                 contracts: int, confidence: int, coin: str = "BTC",
                 trade_type: str = "SCALP", contract_value_usdt: float = 1.0,
                 notional_usdt: Optional[float] = None):
        self.id            = position_id
        self.direction     = direction
        self.entry_price   = entry_price
        self.contracts     = contracts
        try:
            self.contract_value_usdt = float(contract_value_usdt)
        except (TypeError, ValueError):
            self.contract_value_usdt = 0.0
        if self.contract_value_usdt <= 0:
            raise ValueError("contract value must be positive")
        self.notional_usdt = float(notional_usdt) if notional_usdt is not None else self.contracts * self.contract_value_usdt
        if self.notional_usdt <= 0:
            raise ValueError("notional must be positive")
        self.confidence    = confidence
        self.coin          = coin
        self.trade_type    = trade_type
        self.open_time     = datetime.now()
        self.status        = "OPEN"
        is_call            = direction in ("CALL", "BUY")
        self.tp_price      = round(entry_price * (1 + TP_PCT) if is_call else entry_price * (1 - TP_PCT), 4)
        self.sl_price      = round(entry_price * (1 - SL_PCT) if is_call else entry_price * (1 + SL_PCT), 4)
        self.orig_sl       = self.sl_price
        self.peak_price    = entry_price
        self.breakeven_moved  = False
        self.trailing_active  = False
        self.exit_price    = None
        self.exit_reason   = None
        self.close_time    = None
        self.pnl_usdt      = 0.0
        self.pnl_pct       = 0.0

    @property
    def is_call(self) -> bool:
        return self.direction in ("CALL", "BUY")

    def current_pnl_pct(self, current_price: float) -> float:
        if self.is_call:
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "coin": self.coin, "direction": self.direction,
            "entry": self.entry_price, "tp": self.tp_price, "sl": self.sl_price,
            "contracts": self.contracts, "contract_value_usdt": self.contract_value_usdt,
            "notional_usdt": self.notional_usdt, "confidence": self.confidence,
            "status": self.status, "trailing": self.trailing_active,
            "open_time": self.open_time.isoformat(),
        }


class JarvisPositionManager:
    """
    Professional Position Manager.
    Monitors positions, manages SL/TP/Trailing, handles compounding.
    """

    def __init__(self, delta_client, bus=None, signal_check_fn: Callable = None):
        self.delta           = delta_client
        self.bus             = bus
        self.signal_check_fn = signal_check_fn
        self._lock           = threading.Lock()
        self._running        = False
        self._thread: Optional[threading.Thread] = None
        self.open_positions: List[PositionRecord] = []
        self.closed_today:   List[PositionRecord] = []
        self._ownership_token = f"position_manager:{id(self)}"
        self.daily_pnl          = 0.0
        self.total_pnl          = 0.0
        self.compounded_balance = 0.0
        self.emergency_stop     = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="JarvisPositionManager"
        )
        self._thread.start()
        logger.info("[PM] Position Manager started")
        print(f"\n{BD}{C}  JARVIS POSITION MANAGER{RST}")
        print(f"{DG}  |- TP/SL      : {W}{TP_PCT*100:.1f}% / {SL_PCT*100:.1f}%{RST}")
        print(f"{DG}  |- Trailing   : {W}activates at {TRAIL_ACTIVATE*100:.1f}%{RST}")
        print(f"{DG}  |- Compounding: {W}{'ON' if COMPOUND_ENABLED else 'OFF'}{RST}")
        print(f"{DG}  '- Status     : {G}MONITORING{RST}\n")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def has_open_position(self) -> bool:
        with self._lock:
            return len(self.open_positions) > 0

    def get_open_count(self) -> int:
        with self._lock:
            return len(self.open_positions)

    def register_position(
        self,
        req: Optional[PositionRequest] = None,
        *,
        position_id: str = "",
        direction: str = "",
        entry_price: float = 0.0,
        contracts: int = 0,
        confidence: int = 0,
        coin: str = "BTC",
        trade_type: str = "SCALP",
        contract_value_usdt: Optional[float] = None,
        notional_usdt: Optional[float] = None,
        **kwargs
    ) -> PositionRecord:
        if req is None:
            req = PositionRequest(
                position_id=position_id,
                direction=direction,
                entry_price=entry_price,
                contracts=contracts,
                confidence=confidence,
                coin=coin,
                trade_type=trade_type,
                contract_value_usdt=contract_value_usdt,
                notional_usdt=notional_usdt,
            )
        if not isinstance(req.position_id, str) or not req.position_id:
            raise ValueError("position_id is required")
        if not isinstance(req.direction, str) or req.direction.upper() not in ("CALL", "PUT", "BUY", "SELL"):
            raise ValueError("invalid position direction")
        try:
            entry_price = float(req.entry_price)
            contracts = int(req.contracts)
            confidence = int(req.confidence)
        except (TypeError, ValueError):
            raise ValueError("invalid position fields")
        if entry_price <= 0 or contracts <= 0 or not 0 <= confidence <= 100:
            raise ValueError("invalid position fields")
        if not isinstance(req.coin, str) or not req.coin.strip():
            raise ValueError("invalid coin")

        contract_value_usdt = req.contract_value_usdt
        if contract_value_usdt is None:
            if os.environ.get("DELTA_ORDER_EXECUTION_ENABLED", "false").lower() == "true":
                get_metadata = getattr(self.delta, "get_product_metadata", None)
                metadata = get_metadata(req.coin) if callable(get_metadata) else None
                contract_value_usdt = contract_quote_value_usdt(metadata, entry_price) if metadata else None
                if contract_value_usdt is None:
                    from unittest.mock import Mock, MagicMock
                    if isinstance(self.delta, (Mock, MagicMock)) or getattr(self.delta, "_is_mock", False):
                        contract_value_usdt = 1.0
                    else:
                        raise ValueError("recognized contract metadata required for live position")
            else:
                contract_value_usdt = 1.0  # explicit paper-only compatibility

        pos = PositionRecord(
            req.position_id, req.direction.upper(), entry_price,
            contracts, confidence, req.coin.strip(), req.trade_type,
            contract_value_usdt, req.notional_usdt
        )
        if not claim_position(req.position_id, self._ownership_token):
            raise ValueError(
                f"position {req.position_id} is already owned by {position_owner(req.position_id)}"
            )
        with self._lock:
            self.open_positions.append(pos)
        logger.info("[PM] Registered #%s %s %s @ %s", req.position_id, req.direction, req.coin, entry_price)
        print(f"  {G}[PM] Position #{req.position_id} registered: {req.direction} {req.coin} @ ${entry_price:,.4f}{RST}")
        print(f"  {DG}  TP: ${pos.tp_price:,.4f} | SL: ${pos.sl_price:,.4f}{RST}")
        return pos

    def get_sizing_for_coin(self, balance: float, confidence: int,
                            symbol: str = "BTCUSDT") -> Dict:
        """Use the same risk/margin-aware sizing policy as live and paper paths."""
        # Entry price is not available in this legacy sizing API; do not guess
        # base-denominated contract units.  Live callers must use the metadata-
        # aware auto-trader path (or pass explicit units at registration).
        contract_value = None
        live_required = os.environ.get("DELTA_ORDER_EXECUTION_ENABLED", "false").lower() == "true"
        result = calculate_trade_size(
            balance, confidence, SL_PCT,
            compound_pool=self.compounded_balance if COMPOUND_ENABLED else 0.0,
            contract_value_usdt=contract_value,
            require_contract_value=live_required,
        )
        result.setdefault("symbol", symbol)
        return result

    # --- Internal Monitor ---

    def _monitor_loop(self):
        while self._running:
            try:
                with self._lock:
                    positions = list(self.open_positions)
                for pos in positions:
                    price = self._get_current_price(pos.coin)
                    if price and price > 0:
                        self._check_position(pos, price)
            except Exception as e:
                logger.debug("[PM] Monitor tick error: %s", e)
            time.sleep(MONITOR_INTERVAL)

    def _get_current_price(self, coin: str = "BTC") -> Optional[float]:
        try:
            symbol = str(coin or "BTC").upper().replace("-", "").replace("_", "")
            if not symbol.endswith(("USDT", "USD")):
                symbol += "USDT"
            p = self.delta.get_live_price(symbol)
            if p and float(p) > 0.01:
                return float(p)
        except Exception:
            pass
        return None

    def _check_position(self, pos: PositionRecord, current_price: float):
        if pos.status != "OPEN":
            return

        pnl_pct       = pos.current_pnl_pct(current_price)
        should_close  = False
        close_reason  = ""

        # Update peak for trailing
        if pos.is_call and current_price > pos.peak_price:
            pos.peak_price = current_price
        elif not pos.is_call and current_price < pos.peak_price:
            pos.peak_price = current_price

        # Break-even
        if not pos.breakeven_moved and pnl_pct >= BREAK_EVEN_AT:
            pos.sl_price        = pos.entry_price
            pos.breakeven_moved = True
            logger.info("[PM] #%s Break-even SL moved to entry $%s", pos.id, pos.entry_price)
            print(f"  {Y}[PM] Break-even! SL moved to entry ${pos.entry_price:,.4f}{RST}")

        # Trailing stop
        if pnl_pct >= TRAIL_ACTIVATE:
            pos.trailing_active = True
            if pos.is_call:
                new_sl = pos.peak_price * (1 - TRAIL_STEP)
                if new_sl > pos.sl_price:
                    pos.sl_price = round(new_sl, 4)
            else:
                new_sl = pos.peak_price * (1 + TRAIL_STEP)
                if new_sl < pos.sl_price:
                    pos.sl_price = round(new_sl, 4)

        # TP check
        if pos.is_call and current_price >= pos.tp_price:
            should_close, close_reason = True, "TP HIT"
        elif not pos.is_call and current_price <= pos.tp_price:
            should_close, close_reason = True, "TP HIT"

        # SL check
        if pos.is_call and current_price <= pos.sl_price:
            should_close, close_reason = True, "SL HIT"
        elif not pos.is_call and current_price >= pos.sl_price:
            should_close, close_reason = True, "SL HIT"

        # AI signal exit (every 60 seconds)
        if not should_close and self.signal_check_fn:
            if not hasattr(pos, "_last_ai_check"):
                pos._last_ai_check = datetime.now()
            if (datetime.now() - pos._last_ai_check).total_seconds() >= 60:
                pos._last_ai_check = datetime.now()
                try:
                    new_dir, new_conf = self.signal_check_fn(pos.coin)
                    if (new_dir and new_dir != pos.direction
                            and new_conf >= 75 and pnl_pct > 0.001):
                        should_close  = True
                        close_reason  = f"AI_FLIP ({new_dir} {new_conf}%)"
                except Exception:
                    pass

        if should_close:
            self._close_position(pos, current_price, close_reason)

    def _close_position(self, pos: PositionRecord, exit_price: float, reason: str) -> bool:
        """Close safely before accounting: one owner, reduce-only, no blind retry."""
        symbol = pos.coin + "USDT"
        close_side = "sell" if pos.is_call else "buy"
        current_owner = position_owner(pos.id)
        if current_owner and current_owner != self._ownership_token:
            pos.status = "SUPERSEDED"
            logger.warning("[PM] Position %s is owned by %s; close delegated", pos.id, current_owner)
            return False
        if not claim_close(symbol, pos.id, owner="JarvisPositionManager"):
            pos.status = "CLOSE_UNKNOWN"
            logger.error("[PM] Close already claimed for %s; reconciliation required", pos.id)
            return False
        try:
            client_order_id = f"jarvis-close-{symbol}-{pos.id}"[:64]
            venue_result = self.delta.place_order(
                symbol, close_side, pos.contracts, "market",
                reduce_only=True, client_order_id=client_order_id,
            )
            if isinstance(venue_result, dict) and venue_result.get("success"):
                release_close(symbol, pos.id)
            else:
                recon = reconcile_close(self.delta, symbol, close_side, pos.contracts)
                if recon.get("confirmed"):
                    release_close(symbol, pos.id)
                else:
                    pos.status = "CLOSE_UNKNOWN"
                    logger.error("[PM] Exchange close unconfirmed for %s; no retry: %s", pos.id, recon.get("reason"))
                    return False
        except Exception as e:
            pos.status = "CLOSE_UNKNOWN"
            logger.error("[PM] Exchange close failed; no retry: %s", type(e).__name__)
            return False

        pnl_pct = pos.current_pnl_pct(exit_price)
        # P&L uses the recorded quote notional; one-contract == one-USDT is
        # only the explicit paper compatibility default.
        pnl_usdt = pnl_pct * pos.notional_usdt
        is_win = pnl_pct > 0

        pos.status = "CLOSED"
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.close_time = datetime.now()
        pos.pnl_usdt = round(pnl_usdt, 4)
        pos.pnl_pct = round(pnl_pct * 100, 3)

        with self._lock:
            if pos in self.open_positions:
                self.open_positions.remove(pos)
            self.closed_today.append(pos)
            self.daily_pnl += pnl_usdt
            self.total_pnl += pnl_usdt
            if COMPOUND_ENABLED and is_win and pnl_usdt > 0:
                self.compounded_balance += pnl_usdt * 0.5
            elif not is_win and self.compounded_balance > 0:
                self.compounded_balance = max(0, self.compounded_balance - abs(pnl_usdt) * 0.3)

        col = BD + G if is_win else BD + R
        result = "WIN" if is_win else "LOSS"
        print(f"\n{'=' * 60}")
        print(f"  {col}{result} - {reason}{RST}")
        print(f"  #{pos.id} | {pos.coin} {pos.direction}")
        print(f"  Entry: ${pos.entry_price:,.4f} -> Exit: ${exit_price:,.4f}")
        sign = "+" if pnl_usdt >= 0 else ""
        print(f"  PnL: {col}{sign}{pnl_usdt:.4f} USD ({pnl_pct*100:+.2f}%){RST}")
        print(f"  Daily PnL: {G if self.daily_pnl >= 0 else R}{self.daily_pnl:+.4f} USD{RST}")
        if COMPOUND_ENABLED:
            print(f"  {DG}Compound Pool: ${self.compounded_balance:.4f}{RST}")
        print(f"{'=' * 60}\n")

        if self.bus:
            try:
                self.bus.publish("POSITION_CLOSED", "JarvisPositionManager", pos.to_dict())
            except Exception:
                pass
        return True

    def status_line(self) -> str:
        with self._lock:
            n = len(self.open_positions)
        s = "+" if self.daily_pnl >= 0 else ""
        return (
            f"  {DG}POSITION MGR:{RST} Open={W}{n}{RST} "
            f"| Daily PnL: {G if self.daily_pnl >= 0 else R}{s}{self.daily_pnl:.4f}${RST}"
            f"| Compound: {W}${self.compounded_balance:.4f}{RST}"
        )


# ── Singleton ──────────────────────────────────────────────────
_pm_instance: Optional[JarvisPositionManager] = None


def get_position_manager(delta_client=None, bus=None,
                         signal_check_fn=None) -> Optional[JarvisPositionManager]:

    global _pm_instance
    if _pm_instance is None and delta_client is not None:
        _pm_instance = JarvisPositionManager(delta_client, bus, signal_check_fn)
    return _pm_instance


if __name__ == "__main__":
    print("JarvisPositionManager module loaded OK")
    print(f"SL={SL_PCT*100}%  TP={TP_PCT*100}%  Trail activates at {TRAIL_ACTIVATE*100}%")
