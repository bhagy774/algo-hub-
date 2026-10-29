#!/usr/bin/env python3
"""JARVIS historical replay runner.

This runner feeds only locally supplied, completed OHLCV candles to the same
``JarvisElite.analyze_trade_setup`` math/GPU decision path used by live mode.
It has no market-data downloader, model endpoint, or trading-client code.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    net: float
    gross: float
    fees: float
    wr: float
    dd: float
    sharpe: float

# The replay safety mode is set immediately before importing the brain in
# ``_init_jarvis``.  Do not mutate the process environment at module import:
# importing this optional utility from the live brain must not silently turn the
# live brain into a historical replay.

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

RESULT_DIR = Path("data/backtest_results")
LEVERAGE = int(os.environ.get("JARVIS_LEVERAGE", "100"))
MAX_RISK_USDT = float(os.environ.get("JARVIS_MAX_RISK_USDT", "10"))
MAX_DAILY_LOSS_USDT = float(os.environ.get("JARVIS_MAX_DAILY_LOSS", "30"))
MAX_OPEN_POSITIONS = int(os.environ.get("JARVIS_MAX_OPEN", "2"))
MIN_CONFIDENCE = int(os.environ.get("JARVIS_MIN_CONFIDENCE", "70"))
COOLDOWN_SECONDS = int(os.environ.get("JARVIS_COOLDOWN_SEC", "180"))
CONSEC_LOSS_LIMIT = int(os.environ.get("JARVIS_CONSEC_LOSS", "3"))
SCALP_TP_PCT = float(os.environ.get("JARVIS_SCALP_TP", "0.008"))
SCALP_SL_PCT = float(os.environ.get("JARVIS_SCALP_SL", "0.004"))
SWING_TP_PCT = float(os.environ.get("JARVIS_SWING_TP", "0.020"))
SWING_SL_PCT = float(os.environ.get("JARVIS_SWING_SL", "0.008"))

logger = logging.getLogger("JarvisBacktest")

# These are the 12 local core adapters actually wired into JarvisElite.  The
# mapping makes the report explicit about what candle-only replay can prove.
CORE_PART_DATA_CONTRACT = {
    'part1_breakout': 'OHLCV', 'part2_zone': 'OHLCV', 'part3_psychology': 'OHLCV',
    'part4_volume': 'OHLCV volume (institutional feed unavailable)',
    'part5_ml': 'OHLCV-derived features', 'part6_trend': 'OHLCV',
    'part7_volatility': 'OHLCV', 'part8_structure': 'OHLCV',
    'part9_orderflow': 'OHLCV volume proxy (not tick/order-book flow)',
    'part10_candlestats': 'OHLCV', 'part11_fusion': 'outputs of Parts 1-10',
    'part12_confidence': 'outputs of Parts 1-11',
}
HISTORICAL_DATA_NOT_IN_OHLCV = (
    'timestamp-matched options chain/OI/PCR', 'funding-rate history',
    'order-book/tick trades', 'liquidations', 'dated cross-exchange snapshots',
)

class HistoricalCandleLoader:
    """Read a local, immutable historical OHLCV export; never fetches a feed."""

    REQUIRED = ("open", "high", "low", "close", "volume")

    def __init__(self, source: Path):
        self.source = source

    def load(self) -> pd.DataFrame:
        if not self.source.is_file():
            raise FileNotFoundError(f"Historical data file not found: {self.source}")
        suffix = self.source.suffix.lower()
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(self.source)
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(self.source)
        else:
            raise ValueError("Historical input must be CSV or Parquet")
        df.columns = [str(c).strip().lower() for c in df.columns]
        time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), None)
        if time_col is None:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Historical input needs timestamp/time/datetime/date or a DatetimeIndex")
        else:
            raw = df.pop(time_col)
            # Numeric exports are conventionally seconds or milliseconds.
            if pd.api.types.is_numeric_dtype(raw):
                unit = "ms" if raw.dropna().abs().median() > 10_000_000_000 else "s"
                df.index = pd.to_datetime(raw, unit=unit, utc=True).dt.tz_localize(None)
            else:
                df.index = pd.to_datetime(raw, utc=True).dt.tz_localize(None)
        missing = set(self.REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"Historical input missing columns: {sorted(missing)}")
        df = df.loc[:, self.REQUIRED].copy()
        for col in self.REQUIRED:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_index()
        if df.index.has_duplicates:
            raise ValueError("Historical input contains duplicate timestamps")
        if len(df) < 2 or not df.index.is_monotonic_increasing:
            raise ValueError("Historical input must contain ordered completed candles")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("OHLC prices must be positive")
        if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError("Invalid OHLC range in historical input")
        return df


def extract_signal(result: dict) -> Tuple[str, int]:
    """Parse a direction and confidence from the central decision result."""
    signal = result.get("trade_signal", {})
    direction = str(signal.get("direction", "NO_TRADE")).upper()
    raw = signal.get("confidence_score", "0/100")
    try:
        confidence = int(float(str(raw).split("/")[0]))
    except (TypeError, ValueError):
        confidence = 0
    return {"BUY": "CALL", "SELL": "PUT"}.get(direction, direction), confidence


class BacktestPosition:
    """A filled futures position using deterministic OHLC execution assumptions."""

    _ctr = 0

    def __init__(self, direction: str, entry_price: float, entry_time: datetime,
                 contracts: float, trade_type: str = "SCALP", confidence: int = 0,
                 slippage_bps: float = 5.0, fee_bps: float = 10.0,
                 part_snapshot: Optional[Dict[str, dict]] = None,
                 scalp_tp: Optional[float] = None, scalp_sl: Optional[float] = None,
                 swing_tp: Optional[float] = None, swing_sl: Optional[float] = None):
        BacktestPosition._ctr += 1
        self.id = BacktestPosition._ctr
        self.direction, self.entry_price, self.entry_time = direction, entry_price, entry_time
        self.contracts, self.trade_type, self.confidence = contracts, trade_type, confidence
        self.slippage_bps, self.fee_bps = slippage_bps, fee_bps
        self.part_snapshot = part_snapshot or {}
        tp_ref = swing_tp if swing_tp is not None else SWING_TP_PCT
        sl_ref = swing_sl if swing_sl is not None else SWING_SL_PCT
        s_tp_ref = scalp_tp if scalp_tp is not None else SCALP_TP_PCT
        s_sl_ref = scalp_sl if scalp_sl is not None else SCALP_SL_PCT
        tp_pct = tp_ref if trade_type == "SWING" else s_tp_ref
        sl_pct = sl_ref if trade_type == "SWING" else s_sl_ref
        self.is_call = direction in ("CALL", "BUY")
        self.tp_price = entry_price * (1 + tp_pct if self.is_call else 1 - tp_pct)
        self.sl_price = entry_price * (1 - sl_pct if self.is_call else 1 + sl_pct)
        self.expiry_time = entry_time + timedelta(hours=24 if trade_type == "SWING" else 1)
        self.status = "OPEN"
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.close_reason: Optional[str] = None
        self.result: Optional[str] = None
        self.gross_pnl = 0.0
        self.fees_usdt = 0.0
        self.pnl_usdt = 0.0

    def get_part_attribution(self) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]], List[str]]:
        """Return (voted_for, voted_against, neutral) parts based on trade direction."""
        target = 1 if self.is_call else -1
        voted_for, voted_against, neutral = [], [], []
        for name, info in self.part_snapshot.items():
            if not isinstance(info, dict):
                continue
            sig = info.get("signal", 0)
            try:
                sig = float(sig)
            except (ValueError, TypeError):
                sig = 0.0
            thought = str(info.get("thought", "")).strip()
            if (target > 0 and sig > 0) or (target < 0 and sig < 0):
                voted_for.append((name, sig, thought))
            elif (target > 0 and sig < 0) or (target < 0 and sig > 0):
                voted_against.append((name, sig, thought))
            else:
                neutral.append(name)
        return voted_for, voted_against, neutral

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    @property
    def fee_rate(self) -> float:
        # ``fee_bps`` is explicitly a round-trip assumption, split across fills.
        return self.fee_bps / 20_000.0

    def adverse_exit_fill(self, raw_price: float) -> float:
        """Long exits receive less; short exits pay more."""
        return raw_price * (1 - self.slippage_rate if self.is_call else 1 + self.slippage_rate)

    def check_candle(self, candle: pd.Series, dt: datetime, at_entry: bool = False) -> Optional[Tuple[str, float]]:
        """Return a reason/raw exit price. Same-bar TP+SL resolves to SL first."""
        high, low, close = float(candle["high"]), float(candle["low"]), float(candle["close"])
        tp_hit = high >= self.tp_price if self.is_call else low <= self.tp_price
        sl_hit = low <= self.sl_price if self.is_call else high >= self.sl_price
        if tp_hit and sl_hit:
            return "SL HIT (same-bar conservative)", self.sl_price
        if sl_hit:
            return "SL HIT", self.sl_price
        if tp_hit:
            return "TP HIT", self.tp_price
        # Time expiry removed per user request: only close on SL/TP
        return None

    def close(self, raw_exit_price: float, exit_time: datetime, reason: str) -> float:
        self.exit_price = self.adverse_exit_fill(raw_exit_price)
        self.exit_time, self.close_reason, self.status = exit_time, reason, "CLOSED"
        signed_move = (self.exit_price - self.entry_price) if self.is_call else (self.entry_price - self.exit_price)
        self.gross_pnl = signed_move * self.contracts
        self.fees_usdt = (self.entry_price + self.exit_price) * self.contracts * self.fee_rate
        self.pnl_usdt = self.gross_pnl - self.fees_usdt
        self.result = "WIN" if self.pnl_usdt > 0 else "LOSS"
        return self.pnl_usdt


class BacktestRiskGate:
    def __init__(self):
        self.daily_pnl = 0.0
        self.consec_losses = 0
        self.last_trade_dt: Optional[datetime] = None
        self.open_positions: List[BacktestPosition] = []
        self.today_date = None

    def reset_daily(self, day) -> None:
        if day != self.today_date:
            self.daily_pnl, self.consec_losses, self.today_date = 0.0, 0, day

    def can_trade(self, confidence: int, dt: datetime) -> Tuple[bool, str]:
        self.reset_daily(dt.date())
        if confidence < MIN_CONFIDENCE:
            return False, f"confidence {confidence}% < {MIN_CONFIDENCE}%"
        if self.daily_pnl <= -MAX_DAILY_LOSS_USDT:
            return False, f"daily loss ${self.daily_pnl:.2f}"
        if self.consec_losses >= CONSEC_LOSS_LIMIT:
            return False, f"{self.consec_losses} consecutive losses"
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"max {MAX_OPEN_POSITIONS} open positions"
        if self.last_trade_dt and (dt - self.last_trade_dt).total_seconds() < COOLDOWN_SECONDS:
            return False, "cooldown"
        return True, "OK"

    def record_open(self, position: BacktestPosition, dt: datetime) -> None:
        self.open_positions.append(position)
        self.last_trade_dt = dt

    def record_close(self, position: BacktestPosition) -> None:
        if position in self.open_positions:
            self.open_positions.remove(position)
        self.daily_pnl += position.pnl_usdt
        self.consec_losses = 0 if position.result == "WIN" else self.consec_losses + 1

    def calc_contracts(self, position: BacktestPosition, cash_balance: float) -> float:
        """Size base-asset quantity by worst stop loss, then cap gross exposure."""
        if position.entry_price <= 0 or cash_balance <= 0:
            return 0.0
        is_call = position.direction in ("CALL", "BUY")
        adverse_stop = position.sl_price * (1 - position.slippage_bps / 10_000 if is_call else 1 + position.slippage_bps / 10_000)
        loss_per_unit = abs(position.entry_price - adverse_stop) + (position.entry_price + adverse_stop) * position.fee_bps / 20_000
        if loss_per_unit <= 0:
            return 0.0
        used_notional = sum(p.entry_price * p.contracts for p in self.open_positions)
        exposure_room = max(0.0, cash_balance * LEVERAGE - used_notional)
        quantity = min(MAX_RISK_USDT / loss_per_unit, exposure_room / position.entry_price)
        # Deterministic quantity precision without forcing an unaffordable one-unit trade.
        return math.floor(max(0.0, quantity) * 100_000_000) / 100_000_000


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    years: int = 3
    starting_capital: float = 1000.0
    warmup: int = 100
    slippage_bps: float = 5.0
    fee_bps: float = 10.0
    scalp_tp: float = SCALP_TP_PCT
    scalp_sl: float = SCALP_SL_PCT
    swing_tp: float = SWING_TP_PCT
    swing_sl: float = SWING_SL_PCT
    live_audit: bool = True


class JarvisFullBacktester:
    """Historical-only replay with central math/GPU decision analysis in isolation mode."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        config = config or BacktestConfig()
        self.symbol, self.timeframe, self.years = config.symbol.upper(), config.timeframe, config.years
        self.starting_capital, self.warmup = config.starting_capital, config.warmup
        self.slippage_bps, self.fee_bps = config.slippage_bps, config.fee_bps
        self.scalp_tp, self.scalp_sl = config.scalp_tp, config.scalp_sl
        self.swing_tp, self.swing_sl = config.swing_tp, config.swing_sl
        self.live_audit = config.live_audit
        self.gate = BacktestRiskGate()
        self.all_trades: List[BacktestPosition] = []
        self.balance = self.peak = config.starting_capital
        self.equity_curve: List[Dict[str, object]] = []
        self.n_no_signal = self.n_skip_gate = self.n_unaffordable = 0
        self.part_activity = {name: {'decisions_seen': 0, 'non_neutral': 0, 'offline': 0}
                              for name in CORE_PART_DATA_CONTRACT}
        self.part_fault_tracker = {
            name: {'total_votes': 0, 'wins': 0, 'losses': 0, 'fault_count': 0, 'correct_dissent': 0}
            for name in CORE_PART_DATA_CONTRACT
        }

    def _init_jarvis(self):
        print("  Initializing JarvisElite central math/GPU decision path (isolated replay)...")
        # Set the isolation floor before importing the engine.  This is scoped
        # to an actual replay invocation rather than an optional-module import.
        os.environ["JARVIS_BACKTEST_MODE"] = "1"
        from jarvis_FIXED import JarvisElite, pro_display
        # This must remain true: it preserves analysis engines while blocking live-only inputs.
        brain = JarvisElite(backtest_mode=True)
        brain.print_dashboard = lambda *args, **kwargs: None
        pro_display.display_full_signal = lambda *args, **kwargs: None
        return brain

    def _record_part_activity(self, brain) -> None:
        """Keep auditable proof that each core adapter participated in replay."""
        results = getattr(brain, 'latest_part_results', {}) or {}
        for name in CORE_PART_DATA_CONTRACT:
            result = results.get(name, {})
            if not isinstance(result, dict):
                self.part_activity[name]['offline'] += 1
                continue
            self.part_activity[name]['decisions_seen'] += 1
            if str(result.get('thought', '')).lower() == 'part offline':
                self.part_activity[name]['offline'] += 1
            try:
                if float(result.get('signal', 0)) != 0:
                    self.part_activity[name]['non_neutral'] += 1
            except (TypeError, ValueError):
                self.part_activity[name]['offline'] += 1

    def _part_coverage(self) -> dict:
        coverage = {}
        for name, source in CORE_PART_DATA_CONTRACT.items():
            stats = self.part_activity[name]
            coverage[name] = {
                'historical_input': source,
                **stats,
                'active': stats['decisions_seen'] > 0 and stats['offline'] == 0,
            }
        return coverage

    def _entry_fill(self, direction: str, opening_price: float) -> float:
        adverse = self.slippage_bps / 10_000.0
        return opening_price * (1 + adverse if direction == "CALL" else 1 - adverse)

    def _close_position(self, position: BacktestPosition, raw_exit: float, dt: datetime, reason: str) -> None:
        self.balance += position.close(raw_exit, dt, reason)
        self.peak = max(self.peak, self.balance)
        self.gate.record_close(position)
        self._audit_trade_close(position, dt, reason)

    def _audit_trade_close(self, position: BacktestPosition, dt: datetime, reason: str) -> None:
        voted_for, voted_against, neutral = position.get_part_attribution()
        ts = dt.strftime("%Y-%m-%d %H:%M")
        if position.result == "WIN":
            for name, _, _ in voted_for:
                if name in self.part_fault_tracker:
                    self.part_fault_tracker[name]['wins'] += 1
                    self.part_fault_tracker[name]['total_votes'] += 1
            if self.live_audit:
                drivers = ", ".join(f"{p[0]}({p[1]:+.1f})" for p in voted_for[:4]) or "baseline rules"
                print(f"\n[WIN #{position.id}] {ts} | {position.direction} ({position.trade_type}) | Exit: ${position.exit_price:,.2f} | PnL: +${position.pnl_usdt:.2f} (fees: ${position.fees_usdt:.2f}) | Balance: ${self.balance:,.2f} | Reason: {reason}", flush=True)
                print(f"  Drivers: {drivers}", flush=True)
        else:
            for name, _, _ in voted_for:
                if name in self.part_fault_tracker:
                    self.part_fault_tracker[name]['losses'] += 1
                    self.part_fault_tracker[name]['fault_count'] += 1
                    self.part_fault_tracker[name]['total_votes'] += 1
            for name, _, _ in voted_against:
                if name in self.part_fault_tracker:
                    self.part_fault_tracker[name]['correct_dissent'] += 1
            if self.live_audit:
                print(f"\n[LOSS #{position.id}] {ts} | {position.direction} ({position.trade_type}) | Exit: ${position.exit_price:,.2f} | PnL: -${abs(position.pnl_usdt):.2f} (fees: ${position.fees_usdt:.2f}) | Balance: ${self.balance:,.2f} | Reason: {reason}", flush=True)
                print(f"  FAULT AUDIT (Kaya part no vak hato?):", flush=True)
                if voted_for:
                    print(f"    Parts that voted {position.direction} (Misled entry / Vak Hato):", flush=True)
                    for name, sig, thought in voted_for:
                        t_clean = (thought[:85] + '...') if len(thought) > 85 else thought
                        print(f"      * [{name}] sig={sig:+.1f} | {t_clean}", flush=True)
                else:
                    print(f"      * (No core part gave directional vote; entered on baseline aggregate)", flush=True)
                if voted_against:
                    print(f"    Dissenting parts (Correctly warned opposite):", flush=True)
                    for name, sig, thought in voted_against:
                        t_clean = (thought[:85] + '...') if len(thought) > 85 else thought
                        print(f"      * [{name}] sig={sig:+.1f} | {t_clean}", flush=True)
                if neutral:
                    print(f"    Neutral parts: {', '.join(neutral)}", flush=True)

    def _check_open_positions(self, candle: pd.Series, dt: datetime) -> None:
        for position in list(self.gate.open_positions):
            outcome = position.check_candle(candle, dt)
            if outcome:
                reason, raw_exit = outcome
                self._close_position(position, raw_exit, dt, reason)

    def run(self, data_file: Path) -> None:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        df = HistoricalCandleLoader(data_file).load()
        if len(df) < self.warmup + 2:
            raise ValueError("Not enough historical candles after warmup")
        print("\n" + "=" * 70)
        print("JARVIS HISTORICAL REPLAY")
        print("=" * 70)
        print(f"Symbol: {self.symbol} | TF label: {self.timeframe} | candles: {len(df):,}")
        print(f"Capital: ${self.starting_capital:,.2f} | leverage cap: {LEVERAGE}x | risk budget: ${MAX_RISK_USDT:.2f}")
        print(f"Execution assumptions: entry/exit slippage {self.slippage_bps:.2f} bps adverse each side; round-trip fee {self.fee_bps:.2f} bps")
        print(f"Targets (Option B): Scalp TP={self.scalp_tp*100:.2f}%%, SL={self.scalp_sl*100:.2f}%% | Swing TP={self.swing_tp*100:.2f}%%, SL={self.swing_sl*100:.2f}%%")
        print("Data isolation: local historical input only. No live feed, client, or remote model path.")
        print("=" * 70)
        brain = self._init_jarvis()

        # At i, only candles ending before i are visible to the decision. A qualifying
        # signal then fills at candle i OPEN and faces candle i HIGH/LOW thereafter.
        total_steps = len(df) - self.warmup
        for i in range(self.warmup, len(df)):
            step = i - self.warmup + 1
            if step in (1, 10, 50, 100, 250) or step % 500 == 0 or step == total_steps:
                pct = (step / total_steps) * 100
                print(f"  Replay progress: {step:,}/{total_steps:,} candles ({pct:.1f}%) | Open: {len(self.gate.open_positions)} | Closed: {len(self.all_trades)} | Balance: ${self.balance:,.2f}", flush=True)
            dt = df.index[i].to_pydatetime()
            candle = df.iloc[i]
            self._check_open_positions(candle, dt)
            self.equity_curve.append({"time": dt.isoformat(), "balance": self.balance,
                                      "price": float(candle["close"]), "open": len(self.gate.open_positions),
                                      "trades": len(self.all_trades)})
            # Provide 5000 candles to allow 1h and 4h MTF generation (needs >20 candles)
            ws = max(0, i - 5000)
            df_win = df.iloc[ws:i].copy()  # excludes candle i: no signal lookahead
            try:
                result = brain.analyze_trade_setup(df_win)
                self._record_part_activity(brain)
            except Exception as exc:
                logger.debug("analysis failure at %s: %s", dt, exc)
                continue
            direction, confidence = extract_signal(result)
            if direction not in ("CALL", "PUT"):
                self.n_no_signal += 1
                continue
            allowed, _ = self.gate.can_trade(confidence, dt)
            if not allowed:
                self.n_skip_gate += 1
                continue
            opening_price = float(df.iloc[i]["open"])
            entry_price = self._entry_fill(direction, opening_price)
            trade_type = str(result.get("trade_signal", {}).get("recommended_expiry", "SCALP")).upper()
            if self.timeframe in ("1h", "2h", "4h", "6h", "8h", "12h", "1d"):
                trade_type = "SWING"
            else:
                trade_type = "SWING" if trade_type in ("SWING", "DAY_TRADE") else "SCALP"
            provisional = BacktestPosition(direction, entry_price, dt, 0.0, trade_type, confidence,
                                            self.slippage_bps, self.fee_bps,
                                            scalp_tp=self.scalp_tp, scalp_sl=self.scalp_sl,
                                            swing_tp=self.swing_tp, swing_sl=self.swing_sl)
            quantity = self.gate.calc_contracts(provisional, self.balance)
            if quantity <= 0:
                self.n_unaffordable += 1
                continue
            part_snapshot = dict(getattr(brain, 'latest_part_results', {}) or {})
            position = BacktestPosition(direction, entry_price, dt, quantity, trade_type, confidence,
                                        self.slippage_bps, self.fee_bps,
                                        part_snapshot=part_snapshot,
                                        scalp_tp=self.scalp_tp, scalp_sl=self.scalp_sl,
                                        swing_tp=self.swing_tp, swing_sl=self.swing_sl)
            self.gate.record_open(position, dt)
            self.all_trades.append(position)
            if self.live_audit:
                ts = dt.strftime("%Y-%m-%d %H:%M")
                v_for, _, _ = position.get_part_attribution()
                v_str = ", ".join(f"{p[0]}({p[1]:+.1f})" for p in v_for[:4]) if v_for else "baseline rules"
                print(f"\n[ENTRY #{position.id}] {ts} | {position.direction} ({position.trade_type}) @ ${position.entry_price:,.2f} | Qty: {position.contracts:.4f} BTC | Conf: {position.confidence}% | TP: ${position.tp_price:,.2f} | SL: ${position.sl_price:,.2f} | Voted: {v_str}", flush=True)
            # Entry is at this open, so only this candle's subsequent range can trigger it.
            outcome = position.check_candle(candle, dt, at_entry=True)
            if outcome:
                reason, raw_exit = outcome
                self._close_position(position, raw_exit, dt, reason)

        final_dt = df.index[-1].to_pydatetime()
        final_close = float(df.iloc[-1]["close"])
        for position in list(self.gate.open_positions):
            self._close_position(position, final_close, final_dt, "END OF DATA")
        self._report(df)

    def _maxdd(self) -> float:
        peak, worst = self.starting_capital, 0.0
        for point in self.equity_curve:
            peak = max(peak, float(point["balance"]))
            worst = max(worst, (peak - float(point["balance"])) / peak * 100 if peak else 0.0)
        return worst

    @staticmethod
    def _sharpe(closed: List[BacktestPosition]) -> float:
        if len(closed) < 2:
            return 0.0
        values = np.array([trade.pnl_usdt for trade in closed])
        std = values.std(ddof=1)
        return float(values.mean() / std * math.sqrt(365 * 5)) if std else 0.0

    def _report(self, df: pd.DataFrame) -> None:
        closed = [trade for trade in self.all_trades if trade.status == "CLOSED"]
        wins = [trade for trade in closed if trade.result == "WIN"]
        losses = [trade for trade in closed if trade.result == "LOSS"]
        net = sum(trade.pnl_usdt for trade in closed)
        gross = sum(trade.gross_pnl for trade in closed)
        fees = sum(trade.fees_usdt for trade in closed)
        wr = 100 * len(wins) / len(closed) if closed else 0.0
        loss_sum = sum(trade.pnl_usdt for trade in losses)
        pf = abs(sum(trade.pnl_usdt for trade in wins) / loss_sum) if loss_sum else float("inf")
        dd, sharpe = self._maxdd(), self._sharpe(closed)
        print("\n" + "=" * 70)
        print("HISTORICAL REPLAY RESULTS")
        print("=" * 70)
        print(f"Trades: {len(closed)} | wins/losses: {len(wins)}/{len(losses)} | net win rate: {wr:.1f}%")
        print(f"Gross PnL: ${gross:.2f} | modeled fees: ${fees:.2f} | net PnL: ${net:.2f}")
        print(f"Final balance: ${self.balance:.2f} | profit factor: {pf:.2f} | max drawdown: {dd:.2f}% | Sharpe: {sharpe:.2f}")
        print(f"Signals excluded: no-decision {self.n_no_signal}; risk-gated {self.n_skip_gate}; zero-size {self.n_unaffordable}")
        print("Parity: decisions use the central JarvisElite core math/GPU path in backtest isolation mode.")
        print("Parity limitation: historical replay does not reproduce historical options-chain or cross-exchange snapshots unless a dated historical dataset is supplied to that decision pipeline.")
        print("Those omitted live-only inputs are excluded; they are never treated as neutral or positive confirmation.")
        print("=" * 70)

        total_losses = len(losses)
        print("\n" + "=" * 80)
        print("PART-BY-PART ACCURACY & FAULT SCOREBOARD (KAYA PART NO KETLO VAK HATO?)")
        print("=" * 80)
        print(f"{'Part Name':<20} | {'Votes':<6} | {'Wins':<5} | {'Losses':<6} | {'Win Rate':<9} | {'Fault Share':<11} | Assessment")
        print("-" * 80)
        fault_summary = {}
        for name in CORE_PART_DATA_CONTRACT:
            stats = self.part_fault_tracker.get(name, {})
            votes = stats.get('total_votes', 0)
            pwins = stats.get('wins', 0)
            plosses = stats.get('losses', 0)
            wr_part = (pwins / votes * 100) if votes > 0 else 0.0
            fault_pct = (plosses / total_losses * 100) if total_losses > 0 else 0.0
            if votes == 0:
                eval_tag = "PASSIVE (No direct votes)"
            elif wr_part >= 65.0:
                eval_tag = "RELIABLE (Strong edge)"
            elif wr_part >= 50.0:
                eval_tag = "MODERATE (Acceptable balance)"
            elif plosses >= 3:
                eval_tag = "HIGH FAULT (Sau thi vadhu bhul)"
            else:
                eval_tag = "LOW SAMPLE"
            fault_summary[name] = {
                'total_votes': votes,
                'wins': pwins,
                'losses': plosses,
                'win_rate_pct': round(wr_part, 2),
                'fault_share_pct': round(fault_pct, 2),
                'assessment': eval_tag,
            }
            print(f"{name:<20} | {votes:<6} | {pwins:<5} | {plosses:<6} | {wr_part:>7.1f}% | {fault_pct:>9.1f}% | {eval_tag}")
        print("=" * 80)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = RESULT_DIR / f"backtest_{self.symbol}_{self.timeframe}_{stamp}"
        self._csv(closed, prefix)
        metrics = BacktestMetrics(net=net, gross=gross, fees=fees, wr=wr, dd=dd, sharpe=sharpe)
        self._html(prefix, closed, metrics, stamp, fault_summary)
        coverage = self._part_coverage()
        coverage_path = prefix.with_name(prefix.name + '_part_coverage.json')
        coverage_path.write_text(json.dumps({
            'core_parts': coverage,
            'part_fault_audit': fault_summary,
            'not_replayed_without_timestamped_datasets': HISTORICAL_DATA_NOT_IN_OHLCV,
            'options_chain_vote_included': False,
        }, indent=2), encoding='utf-8')
        active = sum(1 for item in coverage.values() if item['active'])
        print(f"Core Part coverage: {active}/12 active; detailed audit: {coverage_path.name}")
        print("Historical options/OI/funding/order-book are not in OHLCV and do not vote in this replay.")
        print(f"Reports: {prefix.name}.csv, {prefix.name}.html and {coverage_path.name}")

    def _csv(self, closed: List[BacktestPosition], prefix: Path) -> None:
        rows = [{"id": t.id, "direction": t.direction, "type": t.trade_type, "confidence": t.confidence,
                 "entry_time": t.entry_time, "exit_time": t.exit_time, "entry_fill": t.entry_price,
                 "exit_fill": t.exit_price, "quantity_base": t.contracts, "tp_trigger": t.tp_price,
                 "sl_trigger": t.sl_price, "reason": t.close_reason, "result": t.result,
                 "gross_pnl": round(t.gross_pnl, 8), "fees_usdt": round(t.fees_usdt, 8),
                 "net_pnl": round(t.pnl_usdt, 8),
                 "parts_voted_for": ", ".join(f"{p[0]}({p[1]:+.1f})" for p in t.get_part_attribution()[0]),
                 "parts_dissenting": ", ".join(f"{p[0]}({p[1]:+.1f})" for p in t.get_part_attribution()[1]),
                } for t in closed]
        pd.DataFrame(rows).to_csv(f"{prefix}.csv", index=False)

    def _html(self, prefix: Path, closed: List[BacktestPosition], metrics: BacktestMetrics, stamp: str,
              fault_summary: Optional[dict] = None) -> None:
        rows = "".join(
            f"<tr><td>{t.id}</td><td>{t.direction}</td><td>{t.confidence}%</td><td>{t.entry_price:.2f}</td><td>{t.exit_price:.2f}</td><td>{t.close_reason}</td><td>{t.pnl_usdt:.4f}</td></tr>"
            for t in closed[-100:])
        fault_rows = ""
        if fault_summary:
            fault_rows = "".join(
                f"<tr><td>{k}</td><td>{v['total_votes']}</td><td>{v['wins']}</td><td>{v['losses']}</td><td>{v['win_rate_pct']:.1f}%</td><td>{v['fault_share_pct']:.1f}%</td><td>{v['assessment']}</td></tr>"
                for k, v in fault_summary.items()
            )
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>JARVIS historical replay</title>
<style>body{{font-family:system-ui;background:#10151c;color:#e8edf2;margin:2rem}}section,table{{background:#19212b;padding:1rem;margin:1rem 0;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;text-align:left;border-bottom:1px solid #334}}</style></head><body>
<h1>JARVIS Historical Replay</h1><p>{stamp} · {self.symbol} · {self.timeframe}</p>
<section><h2>Results</h2><p>Net PnL: ${metrics.net:.2f}; gross PnL: ${metrics.gross:.2f}; modeled fees: ${metrics.fees:.2f}; net win rate: {metrics.wr:.1f}%; max drawdown: {metrics.dd:.2f}%; Sharpe: {metrics.sharpe:.2f}</p></section>
<section><h2>Execution assumptions (Option B)</h2><p>Conservative, user-configurable assumptions: adverse entry/exit slippage {self.slippage_bps:.2f} bps each side; round-trip trading fee {self.fee_bps:.2f} bps. Scalp TP: {self.scalp_tp*100:.2f}%, SL: {self.scalp_sl*100:.2f}%. Swing TP: {self.swing_tp*100:.2f}%, SL: {self.swing_sl*100:.2f}%. Entries fill at next candle open; TP/SL use high/low; same-candle dual touch resolves to SL first.</p></section>
<section><h2>Core Part Accuracy & Fault Breakdown (Kaya Part no Vak Hato)</h2>
<table><thead><tr><th>Part Name</th><th>Total Votes</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Fault Share</th><th>Assessment</th></tr></thead><tbody>{fault_rows}</tbody></table>
</section>
<section><h2>Decision-path parity and limitation</h2><p>Decisions use the central JarvisElite core math/GPU decision path in isolation mode. Historical options-chain/OI/PCR, funding, order-book, liquidation, and cross-exchange snapshots require timestamp-matched datasets. They are excluded from OHLCV-only replay and cannot vote as neutral or positive confirmation.</p></section>
<table><thead><tr><th>#</th><th>Direction</th><th>Confidence</th><th>Entry fill</th><th>Exit fill</th><th>Reason</th><th>Net PnL</th></tr></thead><tbody>{rows}</tbody></table>
<footer>JARVIS Backtest Engine | Core math/GPU decision path | Historical replay only</footer></body></html>"""
        prefix.with_suffix(".html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS isolated historical replay")
    parser.add_argument("--data-file", required=True, type=Path, help="Local completed OHLCV CSV or Parquet")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", default="5m", help="Label only; input timestamps remain authoritative")
    parser.add_argument("--years", default=3, type=int, help="Label only; local input remains authoritative")
    parser.add_argument("--capital", default=1000.0, type=float)
    parser.add_argument("--warmup", default=100, type=int)
    parser.add_argument("--slippage-bps", default=5.0, type=float,
                        help="Adverse entry and exit slippage per side; conservative assumption, default 5")
    parser.add_argument("--fee-bps", default=10.0, type=float,
                        help="Round-trip trading fee; conservative assumption, default 10")
    parser.add_argument("--scalp-tp", default=SCALP_TP_PCT, type=float,
                        help="Scalp Take-Profit fraction (default 0.008 = 0.8%%)")
    parser.add_argument("--scalp-sl", default=SCALP_SL_PCT, type=float,
                        help="Scalp Stop-Loss fraction (default 0.004 = 0.4%%)")
    parser.add_argument("--swing-tp", default=SWING_TP_PCT, type=float,
                        help="Swing Take-Profit fraction (default 0.020 = 2.0%%)")
    parser.add_argument("--swing-sl", default=SWING_SL_PCT, type=float,
                        help="Swing Stop-Loss fraction (default 0.008 = 0.8%%)")
    parser.add_argument("--no-audit", action="store_true", default=False,
                        help="Disable live trade and fault audit streaming")
    args = parser.parse_args()
    if args.capital <= 0 or args.warmup < 20 or args.slippage_bps < 0 or args.fee_bps < 0:
        parser.error("capital must be positive; warmup >=20; costs must be non-negative")

    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.tf,
        years=args.years,
        starting_capital=args.capital,
        warmup=args.warmup,
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
        scalp_tp=args.scalp_tp,
        scalp_sl=args.scalp_sl,
        swing_tp=args.swing_tp,
        swing_sl=args.swing_sl,
        live_audit=not args.no_audit
    )
    runner = JarvisFullBacktester(config)
    runner.run(args.data_file)


if __name__ == "__main__":
    main()
