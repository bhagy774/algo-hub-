"""Fast source/isolated-unit checks for the historical replay safety contract."""
import ast
import math
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parents[1]
BACKTESTER_PATH = HERE / "jarvis_backtester.py"
BRAIN_PATH = HERE / "jarvis_FIXED.py"
BACKTESTER = BACKTESTER_PATH.read_text(encoding="utf-8")
BRAIN = BRAIN_PATH.read_text(encoding="utf-8")


def _isolated_classes():
    """Compile only the small simulation classes, never import the large brain."""
    module = ast.parse(BACKTESTER)
    wanted = {"BacktestPosition", "BacktestRiskGate"}
    nodes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name in wanted]
    ns = {
        "math": math, "timedelta": timedelta, "datetime": datetime,
        "List": List, "Optional": Optional, "Tuple": Tuple, "Dict": Dict,
        "pd": SimpleNamespace(Series=object),
        "LEVERAGE": 100, "MAX_RISK_USDT": 10.0,
        "MAX_DAILY_LOSS_USDT": 30.0, "MAX_OPEN_POSITIONS": 2,
        "MIN_CONFIDENCE": 70, "COOLDOWN_SECONDS": 180, "CONSEC_LOSS_LIMIT": 3,
        "SCALP_TP_PCT": 0.008, "SCALP_SL_PCT": 0.004,
        "SWING_TP_PCT": 0.020, "SWING_SL_PCT": 0.008,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BACKTESTER_PATH), "exec"), ns)
    return ns["BacktestPosition"], ns["BacktestRiskGate"]


def test_backtester_has_no_remote_model_path_or_network_library():
    lower = BACKTESTER.lower()
    for prohibited in ("legacy_llm", "requests", "http://", "https://", "api/generate"):
        assert prohibited not in lower
    assert 'p.add_argument("--legacy_llm' not in BACKTESTER


def test_backtester_has_no_execution_client_or_order_submission_path():
    lower = BACKTESTER.lower()
    for prohibited in ("delta_api_wrapper", "place_order", "create_order", "submit_order", "execute_trade"):
        assert prohibited not in lower
    assert "JarvisElite(backtest_mode=True)" in BACKTESTER


def test_brain_backtest_mode_keeps_core_engines_but_does_not_construct_live_sources():
    assert "self.is_backtest_mode = bool(backtest_mode or env_backtest)" in BRAIN
    assert "self.deepseek_enabled = False" in BRAIN
    assert "# Initialize External GPU Engines" in BRAIN
    assert "if not self.is_backtest_mode:\n                    try:\n                        from part7_FIXED" in BRAIN
    assert "# Historical replay must not even construct a live data client." in BRAIN
    assert "# Historical replay has no asynchronous/live event bus." in BRAIN
    assert "if not self.is_backtest_mode and self.delta_data" in BRAIN
    # Deribit is allowed only for the selected BTC asset, never as an altcoin
    # substitute during replay or execution.
    assert "if selected_base == 'BTC' and not self.is_backtest_mode and self.deribit" in BRAIN


def test_signal_window_excludes_entry_bar_and_entry_uses_next_open():
    assert "df_win = df.iloc[ws:i].copy()  # excludes candle i: no signal lookahead" in BACKTESTER
    assert 'opening_price = float(df.iloc[i]["open"])' in BACKTESTER
    assert BACKTESTER.index("brain.analyze_trade_setup(df_win)") < BACKTESTER.index('opening_price = float(df.iloc[i]["open"])')


def test_both_targets_in_one_bar_resolve_conservatively_to_stop_first():
    Position, _ = _isolated_classes()
    position = Position("CALL", 100.0, datetime(2024, 1, 1), 1.0, slippage_bps=0, fee_bps=0)
    outcome = position.check_candle({"high": 101.0, "low": 99.0, "close": 100.0}, datetime(2024, 1, 1))
    assert outcome[0] == "SL HIT (same-bar conservative)"
    assert outcome[1] == position.sl_price


def test_adverse_slippage_and_round_trip_fee_reduce_pnl():
    Position, _ = _isolated_classes()
    when = datetime(2024, 1, 1)
    frictionless = Position("CALL", 100.0, when, 1.0, slippage_bps=0, fee_bps=0)
    costly = Position("CALL", 100.0, when, 1.0, slippage_bps=10, fee_bps=10)
    assert frictionless.close(101.0, when, "EXPIRY") == 1.0
    assert costly.close(101.0, when, "EXPIRY") < frictionless.pnl_usdt
    assert costly.fees_usdt > 0


def test_size_is_stop_risk_based_and_exposure_capped():
    Position, Gate = _isolated_classes()
    gate = Gate()
    # A $1 account at 100x can expose at most $100, i.e. one unit at a $100 fill.
    position = Position("CALL", 100.0, datetime(2024, 1, 1), 0.0, slippage_bps=5.0, fee_bps=10.0)
    position.sl_price = 99.8
    quantity = gate.calc_contracts(position, 1.0)
    assert 0 < quantity <= 1.0
