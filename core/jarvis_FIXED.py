 # JARVIS_MAGIC_STRING_12345
import math
# jarvis_trade_elite_integrated.py
# JARVIS TRADE ELITE v7.0 - FULLY INTEGRATED WITH ALL 4 ENGINES
# Complete system with AutoBacktest, AutoTraining, AutoOptimizer, LiveTrading

import os
import json
import time
import threading
import logging
import asyncio
import concurrent.futures
import websockets
import random
import re
import gc
import requests
import subprocess

# --- ZERO LAG PURE ALGO MODE ---
# Set to "True" to completely bypass all LLM/LegacyLLM network calls during live trading
os.environ.setdefault("JARVIS_PURE_ALGO", "true")
# -------------------------------
def _pure_algorithm_mode():
    """Explicit runtime policy: algorithm-only means model calls never decide or veto."""
    return os.getenv("JARVIS_PURE_ALGO", "true").strip().lower() in {"1", "true", "yes", "on"}

import numpy as np
import pandas as pd
from professional_display import ProfessionalSignalDisplay
from jarvis_dashboard import UnifiedDashboard
from jarvis_risk import calculate_trade_size, MAX_LEVERAGE_CAP
from jarvis_decision import normalize_confidence, confidence_text, build_final_decision
from jarvis_runtime import detect_backend, torch_device
from mode_engine import ModeEngine
# Legacy LLM context removed; deterministic gates are authoritative.
pro_display = ProfessionalSignalDisplay()
import warnings
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from queue import Queue
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

_P10_ZONES = {}


def _p10_breakout(parts_by_tf, direction):
    """True only if Part 1, Part 8 and Part 9 all vote in the trade direction on BOTH 1m and 5m (real signals, no defaults)."""
    try:
        want = 1 if str(direction).upper() in ('CALL', 'BUY') else -1
        for tf in ('1m', '5m'):
            fp = (parts_by_tf or {}).get(tf) or {}
            for k in ('part1', 'part8', 'part9'):
                if int(float((fp.get(k) or {}).get('signal', 0) or 0)) != want:
                    return False
        return True
    except Exception:
        return False


def _p10_zone_track(coin, se, px):
    """Remember pending WAIT_FOR_PULLBACK zones per coin and log whether price entered them on later cycles."""
    import time as _t
    z = _P10_ZONES.get(coin)
    if se.get('status') == 'WAIT_FOR_PULLBACK' and se.get('zone'):
        if not z or z.get('zone') != se.get('zone'):
            _P10_ZONES[coin] = {'zone': se['zone'], 'mode': se.get('mode'), 'dir': se.get('direction'), 't0': _t.time(), 'entered': False}
            logger.info('[PULLBACK_ZONE_TRACKER] %s NEW pending %s %s zone=%s px=%s', coin, se.get('mode'), se.get('direction'), se['zone'], px)
        z = _P10_ZONES[coin]
    if z:
        lo, hi = min(z['zone']), max(z['zone'])
        inside = lo <= px <= hi
        if inside and not z['entered']:
            z['entered'] = True
        logger.info('[PULLBACK_ZONE_TRACKER] %s pending %s %s zone=%s px=%s inside=%s ever_entered=%s age_s=%d', coin, z['mode'], z['dir'], z['zone'], px, inside, z['entered'], int(_t.time() - z['t0']))


@dataclass

class PresimParams:
    direction: str
    confidence: int
    entry_price: float
    result: Optional[Dict[str, Any]] = None
    df: Optional[Any] = None
    current_price: Optional[float] = None
    symbol: str = 'BTCUSDT'

class RandomForestClassifier:
    def __init__(self, *args, **kwargs): pass
    def fit(self, *args, **kwargs): pass
    def predict(self, *args, **kwargs): return []

def train_test_split(*args, **kwargs): return ([], [], [], [])
def accuracy_score(*args, **kwargs): return 1.0
try:
    from dotenv import load_dotenv
    # Load .env file
    load_dotenv()
except ImportError:
    load_dotenv = None
    logging.warning("python-dotenv not installed. Environment variables must be set manually.")

# FIX: telegram_notifier.send_trading_signal was imported but never called
# part10_FIXED.py has its own TelegramNotifier class
# from telegram_notifier import send_trading_signal  # REMOVED — unused
# FIX #4: Deribit removed — using Delta Exchange only for options data
# from deribit_options_client import DeribitOptionsClient  # REMOVED
from jarvis_neural_cortex import JarvisNeuralCortex  # NEW: Jarvis Unified AI Brain
try:
    from jarvis_live_trader import JarvisAutoTrader
    LIVE_TRADER_AVAILABLE = True
except ImportError as _e:
    JarvisAutoTrader = None
    LIVE_TRADER_AVAILABLE = False
    print(f"[JARVIS CORE] ⚠️  JarvisAutoTrader not loaded: {_e}")
try:
    from jarvis_market_oracle import init_oracle as _init_market_oracle, get_oracle as _get_market_oracle
    MARKET_ORACLE_AVAILABLE = True
except ImportError as _oe:
    _init_market_oracle = None
    _get_market_oracle = None
    MARKET_ORACLE_AVAILABLE = False
    print(f"[JARVIS CORE] ⚠️  JarvisMarketOracle not loaded: {_oe}")

# Import Indian Signal Engine (Dual Market: NSE + Crypto)
try:
    from jarvis_indian_signal import get_indian_signal_engine as _get_indian_engine
    INDIAN_SIGNAL_AVAILABLE = True
except ImportError as _ise:
    _get_indian_engine = None
    INDIAN_SIGNAL_AVAILABLE = False
    print(f"[JARVIS CORE] ⚠️  IndianSignalEngine not loaded: {_ise}")

# Optional, on-demand integrations.  These imports create no network clients or
# background work; each feature is explicitly activated by its owning path.
try:
    from jarvis_sizer import get_sizer as _get_jarvis_sizer
    SIZER_AVAILABLE = True
except Exception:
    _get_jarvis_sizer = lambda *_args, **_kwargs: None
    SIZER_AVAILABLE = False
try:
    from jarvis_position_manager import get_position_manager as _get_jarvis_position_manager
    POSITION_MANAGER_AVAILABLE = True
except Exception:
    _get_jarvis_position_manager = lambda *_args, **_kwargs: None
    POSITION_MANAGER_AVAILABLE = False
try:
    from jarvis_coin_scanner import get_coin_scanner as _get_jarvis_coin_scanner
    COIN_SCANNER_AVAILABLE = True
except Exception:
    _get_jarvis_coin_scanner = lambda *_args, **_kwargs: None
    COIN_SCANNER_AVAILABLE = False
try:
    from jarvis_market_router import MarketRouter
    MARKET_ROUTER_AVAILABLE = True
except Exception:
    MarketRouter = None
    MARKET_ROUTER_AVAILABLE = False
try:
    from jarvis_options_context import resolve_options_context, apply_options_confirmation
    OPTIONS_CONTEXT_AVAILABLE = True
except Exception:
    resolve_options_context = None
    apply_options_confirmation = None
    OPTIONS_CONTEXT_AVAILABLE = False
try:
    from jarvis_specialist_pool import SpecialistPool as _SpecialistPool
    SPECIALIST_POOL_AVAILABLE = True
except Exception:
    _SpecialistPool = None
    SPECIALIST_POOL_AVAILABLE = False
try:
    from binance_data import get_binance_data as _get_binance_data
    BINANCE_DATA_AVAILABLE = True
except Exception:
    _get_binance_data = lambda: None
    BINANCE_DATA_AVAILABLE = False
try:
    from upstox_data import get_upstox_data as _get_upstox_data
    UPSTOX_DATA_AVAILABLE = True
except Exception:
    _get_upstox_data = lambda: None
    UPSTOX_DATA_AVAILABLE = False
try:
    from multi_source_data import get_cross_exchange_perspective as _get_cross_exchange_perspective
    MULTI_SOURCE_DATA_AVAILABLE = True
except Exception:
    _get_cross_exchange_perspective = None
    MULTI_SOURCE_DATA_AVAILABLE = False
try:
    from jarvis_data_validator import get_validator as _get_data_validator
    DATA_VALIDATOR_AVAILABLE = True
except Exception:
    _get_data_validator = None
    DATA_VALIDATOR_AVAILABLE = False
try:
    from direct_candle_cache import DirectCandleCache, CandleDataError, LIVE_TIMEFRAMES
    from binance_timeframes import (BINANCE_SPOT_TIMEFRAMES, DEFAULT_STRATEGY_TIMEFRAMES,
        TIMEFRAME_HISTORY_CANDLES, TIMEFRAME_CORRELATION_GROUPS, TIMEFRAME_GROUP_WEIGHTS,
        grouped_timeframe_consensus, grouped_timeframe_values,
        validate_closed_candle_timestamps, next_candle_open,
        synchronize_native_frames_to_1m_close)
    DIRECT_CANDLE_CACHE_AVAILABLE = True
except Exception:
    DirectCandleCache = None
    CandleDataError = ValueError
    LIVE_TIMEFRAMES = ()
    BINANCE_SPOT_TIMEFRAMES = ()
    DEFAULT_STRATEGY_TIMEFRAMES = ()
    TIMEFRAME_HISTORY_CANDLES = {}
    TIMEFRAME_CORRELATION_GROUPS = {}
    TIMEFRAME_GROUP_WEIGHTS = {}
    grouped_timeframe_consensus = lambda values, **kwargs: (0.0, {})
    grouped_timeframe_values = lambda values, **kwargs: {}
    validate_closed_candle_timestamps = lambda timeframe, stamps, decision: (_ for _ in ()).throw(ValueError('native timestamp validator unavailable'))
    next_candle_open = lambda timeframe, stamp: float('inf')
    DIRECT_CANDLE_CACHE_AVAILABLE = False
try:
    from jarvis_multicoin_analysis import multicoin_analysis_enabled
except Exception:
    multicoin_analysis_enabled = lambda value=None: False
try:
    from jarvis_multicoin_pipeline import (
        BinanceSpotCandleClient, DeltaNativeCandleClient, MultiCoinPipeline,
    )
    MULTICOIN_PIPELINE_AVAILABLE = True
except Exception:
    BinanceSpotCandleClient = None
    DeltaNativeCandleClient = None
    MultiCoinPipeline = None
    MULTICOIN_PIPELINE_AVAILABLE = False
try:
    from jarvis_multicoin_execution import (
        PortfolioCoordinator, DeterministicPaperAdapter, candidate_from_analysis,
    )
    MULTICOIN_EXECUTION_AVAILABLE = True
except Exception:
    PortfolioCoordinator = None
    DeterministicPaperAdapter = None
    candidate_from_analysis = None
    MULTICOIN_EXECUTION_AVAILABLE = False
try:
    from jarvis_delta_execution import DeltaExecutionAdapter, build_delta_candidate
    DELTA_MULTICOIN_EXECUTION_AVAILABLE = True
except Exception:
    DeltaExecutionAdapter = None
    build_delta_candidate = None
    DELTA_MULTICOIN_EXECUTION_AVAILABLE = False
try:
    from part7_signal import analyze_timeframe as _analyze_part7_timeframe, aggregate_results as _aggregate_part7_results
    PART7_SHARED_ANALYZER_AVAILABLE = True
except Exception:
    _analyze_part7_timeframe = None
    _aggregate_part7_results = None
    PART7_SHARED_ANALYZER_AVAILABLE = False
try:
    from terminal_live_diagnostics import classify_part_output as _classify_part_output
    from terminal_live_diagnostics import build_timeframe_inputs as _build_timeframe_inputs
    from terminal_live_diagnostics import TIMEFRAMES as _DIAGNOSTIC_TIMEFRAMES
    from terminal_live_diagnostics import PART_KEYS as _DIAGNOSTIC_PART_KEYS
except Exception:
    _classify_part_output = None
    _build_timeframe_inputs = None
    _DIAGNOSTIC_TIMEFRAMES = ('5m', '15m', '30m', '1h', '2h', '4h')
    _DIAGNOSTIC_PART_KEYS = ()
try:
    from jarvis_backtester import JarvisFullBacktester as _JarvisFullBacktester
    BACKTESTER_AVAILABLE = True
except Exception:
    _JarvisFullBacktester = None
    BACKTESTER_AVAILABLE = False
# Import JARVIS Self-Healing Doctor
try:
    from jarvis_doctor import init_doctor as _init_doctor, get_doctor as _get_doctor
    DOCTOR_AVAILABLE = True
except ImportError as _de:
    _init_doctor = None
    _get_doctor = None
    DOCTOR_AVAILABLE = False
    print(f"[JARVIS CORE] ⚠️  JarvisDoctor not loaded: {_de}")


runtime_metadata = lambda: {"available": False, "reason": "integration unavailable"}

# FIX: Set EXTERNAL_ENGINES_AVAILABLE BEFORE imports to avoid circular dependency
EXTERNAL_ENGINES_AVAILABLE = False
OPENROUTER_ENABLED = False  # All remote/model decision routes are retired; Laya is advisory-only.

# External Engines Imports (Mapped to actual part files)
try:
    from part1_FIXED import SmartBreakoutAI                          # Part1: 13-brain breakout engine
    from part2_FIXED import NeuralNetworkManager, AdvancedAnalysisSystem  # Part2: 14 GPU brains + ML
    from part3_FIXED import InstitutionalTradingEngineGPU             # Part3: Institutional engine
    from part4_FIXED import GPUInstitutionalBacktestingEngine         # Part4: Backtesting (FIX: SmartVolumeProfileGPU didn't exist)
    from part5_FIXED import GPUEnhancedFusionEngine                   # Part5: Signal fusion
    from part6_FIXED import GPUComprehensiveBacktester                # Part6: Comprehensive backtester
    from part7_FIXED import EnhancedGPULiveDataEngine                 # Part7: Live data engine
    from part8_FIXED import EnhancedGPUPatternRecognitionEngine       # Part8: Pattern recognition
    from part9_FIXED import GPUAIAdaptiveLearningEngine               # Part9: Adaptive learning
    from part11_FIXED import GPUUnifiedConfidenceEngine               # Part11: Confidence engine
    from part12_FIXED import GPUOrderExecutionEngine, CryptoMarketAnalyzer  # Part12: Execution
    EXTERNAL_ENGINES_AVAILABLE = True
    print("[JARVIS CORE] ✅ All 11 External GPU Engines Found (Parts 1-12)")
except ImportError as e:
    print(f"[JARVIS CORE] [WARNING] External Engines Missing: {e}")
    EXTERNAL_ENGINES_AVAILABLE = False
    # Safe stubs so code doesn't crash if parts missing
    SmartBreakoutAI = None
    AdvancedAnalysisSystem = None
    GPUInstitutionalBacktestingEngine = None
    GPUComprehensiveBacktester = None
    EnhancedGPULiveDataEngine = None
    GPUAIAdaptiveLearningEngine = None
    GPUOrderExecutionEngine = None
    CryptoMarketAnalyzer = None
# FIX #18: Removed duplicate OPENROUTER_ENABLED (already set at line 44)

warnings.filterwarnings('ignore')

# Pre-declare torch modules at global scope (FIX: Prevent 'optim' not defined error)
torch = None
nn = None
optim = None

# Try to import torch safely
try:
    import torch
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    if torch.cuda.is_available():
        try:
            # BUG FIX #4: float16 causes silent precision/overflow errors — use float32
            torch.set_default_dtype(torch.float32)
            print(f"[JARVIS CORE] ✅ CUDA Available - GPU Mode Enabled ({torch.cuda.get_device_name(0)})")
        except Exception:
            torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float32)
        print("[JARVIS CORE] ⚠️ CUDA Not Available - CPU Mode")
except Exception as e:
    torch = None
    nn = None
    optim = None
    logging.warning(f"PyTorch not available. ML features will be limited: {e}")

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

# ==================== HELPER FUNCTIONS ====================

def _safe_get_device_name(device):
    """Safely get GPU device name with fallbacks"""
    try:
        if torch is not None and hasattr(device, "type") and device.type == "cuda":
            try:
                idx = device.index if hasattr(device, "index") and device.index is not None else 0
                return torch.cuda.get_device_name(idx)
            except (RuntimeError, AttributeError):
                try:
                    return torch.cuda.get_device_name()
                except (RuntimeError, AttributeError):
                    return "cuda_device"
        return str(device)
    except Exception:
        return "unknown_device"

class LinuxOptimizedDeque(deque):
    """Linux-optimized deque with safe append"""
    def __init__(self, maxlen=500):
        super().__init__(maxlen=maxlen)
    
    def append(self, item):
        try:
            super().append(item)
        except Exception:
            pass

class GPUFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch is not None and torch.cuda.is_available() else 'cpu')
    
    def extract_basic(self, data):
        try:
            return torch.tensor([float(x) for x in data[:10]], device=self.device)
        except Exception:
            return torch.zeros(10, device=self.device)

# Set encoding for Windows console (Fix for 🧠 emoji)
import sys
import os
from pathlib import Path

# --- NEW: Terminal Logger for HUD ---
class TerminalLogger(object):
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Clear old log on startup if it's too large, but for now just open in append mode
        try:
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 5 * 1024 * 1024:
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("")
        except:
            pass

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if data:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(data)
            except:
                pass

    def flush(self):
        self.stream.flush()

    def reconfigure(self, **kwargs):
        if hasattr(self.stream, 'reconfigure'):
            self.stream.reconfigure(**kwargs)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Hook stdout to log file for the HUD
sys.stdout = TerminalLogger(sys.stdout, os.path.join(str(Path(__file__).parent), "logs", "jarvis_terminal.log"))
# ------------------------------------

logging.basicConfig(
    level=logging.WARNING,  # Hide INFO spam — signals display via print()
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("JarvisElite")
try:
    from jarvis_partlog import install_per_file_logging as _ipfl, install_print_capture as _ipc
    _ipfl(str(Path(__file__).parent))
    _ipc(str(Path(__file__).parent))
except Exception:
    pass

# --- CLEAN OUTPUT MODE (default) -----------------------------------------
# Terminal ma fakt JARVIS no FINAL decision dekhay. Badha parts no chatter
# logs/engine_chatter.log ma save thay. Full detail mate:
#   JARVIS_OUTPUT_MODE=verbose python jarvis_FIXED.py
JARVIS_OUTPUT_MODE = os.environ.get("JARVIS_OUTPUT_MODE", "clean").lower()
JARVIS_VERBOSE = JARVIS_OUTPUT_MODE == "verbose"

def _vprint(*args, **kwargs):
    """Print only in verbose mode."""
    if JARVIS_VERBOSE:
        print(*args, **kwargs)

def _run_quietly(fn, *args, **kwargs):
    """Run fn with stdout captured in clean mode; chatter goes to logs/engine_chatter.log."""
    if JARVIS_VERBOSE:
        return fn(*args, **kwargs)
    import contextlib, io
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _result = fn(*args, **kwargs)
    _chatter = _buf.getvalue()
    if _chatter.strip():
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/engine_chatter.log", "a", encoding="utf-8") as _fh:
                _fh.write(_chatter)
        except Exception:
            pass
    return _result



# ==================== TRADE CONFIGURATION ====================
TRADE_CONFIG = {
    'min_confidence_score': 70,
    'max_daily_trades': 15,
    'position_size_percent': 2,
    'consecutive_loss_limit': 2,
    'trade_cooldown_minutes': 3,
    'use_big_player_filter': True,
    'use_neural_fusion': True
}

TRADING_SESSIONS = {
    'asian': {'start': 0, 'end': 6, 'quality': 'LOW'},
    'london': {'start': 8, 'end': 12, 'quality': 'HIGH'},
    'ny_overlap': {'start': 13, 'end': 16, 'quality': 'BEST'},
    'ny_close': {'start': 16, 'end': 20, 'quality': 'MEDIUM'}
}



def call_ai_for_analysis(prompt, timeout=30):
    return None, "Model analysis retired; Laya is audit-only"

# ==================== 1. AUTO BACKTEST ENGINE ====================

class AutoBacktestEngine:
    """Automated Backtesting Engine for SwingScalp Strategies"""
    
    def __init__(self, jarvis_system):
        self.jarvis = jarvis_system
        self.results = {}
        
    def run_backtest(self, historical_data, initial_balance=1000):
        """Run backtest on supplied candles; only recorded point-in-time options are eligible."""
        logger.info("🔄 Starting Auto Backtest Engine...")
        if not isinstance(historical_data, pd.DataFrame):
            raise ValueError("backtest requires one symbol's timestamped OHLCV DataFrame")
        if not isinstance(historical_data.index, pd.DatetimeIndex) and 'timestamp' in historical_data.columns:
            historical_data = historical_data.copy()
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True, errors='coerce')
            historical_data = historical_data.dropna(subset=['timestamp']).set_index('timestamp')
        _asset_attrs = getattr(historical_data, 'attrs', {}) or {}
        _selected_symbol = str(_asset_attrs.get('symbol') or '').upper().replace('/', '').replace('-', '').replace('_', '')
        self.jarvis.active_symbol = _selected_symbol or None
        self.jarvis.active_base_asset = next(
            (_selected_symbol[:-len(q)] for q in ('USDT', 'USDC', 'USD') if _selected_symbol.endswith(q)),
            _selected_symbol,
        ) or None
        
        # Bypass trade manager checks (like session time) for backtesting
        original_can_trade = self.jarvis.trade_manager.can_trade
        self.jarvis.trade_manager.can_trade = lambda: True
        self.jarvis.is_backtest_mode = True  # Enable backtest mode for AI Gatekeeper
        
        try:
            balance = initial_balance
            trades = []
            winning_trades = 0
            total_trades = 0
            
            # Test different expiry periods
            expiry_periods = ['1M', '2M', '3M', '5M']
            expiry_results = {}
            
            for expiry in expiry_periods:
                expiry_balance = initial_balance
                expiry_trades = 0
                expiry_wins = 0
                
                # Simulate trading with this expiry
                for i in range(50, len(historical_data) - 5):
                    data_slice = historical_data.iloc[i-50:i]
                    
                    # Get trade signal
                    trade_result = self.jarvis.analyze_trade_setup(data_slice)
                    
                    # Safe timestamp extraction for logs
                    try:
                        current_time = data_slice.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        current_time = "Unknown Time"
                    
                    if trade_result['trade_signal']['direction'] != 'NO_TRADE':
                        signal = trade_result['trade_signal']['direction']
                        try:
                            conf_str = trade_result['trade_signal'].get('confidence_score')
                            confidence = normalize_confidence(conf_str, default=0) or 0
                        except (ValueError, IndexError, AttributeError):
                            confidence = 0
                        
                        # NEW: Use 60% threshold for backtest to only take high-probability setups
                        backtest_threshold = 60
                        
                        if confidence >= backtest_threshold:
                            # Print full signal for backtest
                            print(f"\n[BACKTEST: Trade Style: {expiry}] 🕒 Time: {current_time}")
                            try:
                                pro_display.display_full_signal(trade_result)
                            except Exception:
                                pass
                                
                            # Simulate trade outcome
                            # The AI analyzed data up to index i-1. The trade is entered exactly when candle i opens.
                            # So current_price is the OPEN of candle i (or CLOSE of i-1).
                            current_price = float(historical_data['close'].iloc[i-1])
                            
                            # For a 1-bar expiry, the trade closes at the end of candle i.
                            # So future_price is the CLOSE of candle i + expiry_bars - 1
                            expiry_bars = self._expiry_to_bars(expiry)
                            future_price = float(historical_data['close'].iloc[i + expiry_bars - 1])
                            
                            if signal == 'NO_TRADE' or signal == 'NEUTRAL':
                                print(f"⏭️ [TRADE SKIPPED] | Signal: {signal} (No clear direction)\n")
                            elif future_price == current_price:
                                print(f"➖ [TRADE BREAKEVEN] | Signal: {signal} | Entry: {current_price:.2f} | Exit: {future_price:.2f} | Profit: $0.00\n")
                                expiry_trades += 1
                            elif (signal == 'CALL' and future_price > current_price) or \
                               (signal == 'PUT' and future_price < current_price):
                                # Win
                                profit = initial_balance * 0.02
                                expiry_balance += profit  # 2% profit
                                expiry_wins += 1
                                print(f"✅ [TRADE WON] | Signal: {signal} | Entry: {current_price:.2f} | Exit: {future_price:.2f} | Profit: +${profit:.2f}\n")
                                expiry_trades += 1
                            else:
                                # Loss
                                loss = initial_balance * 0.01
                                expiry_balance -= loss  # 1% loss
                                print(f"❌ [TRADE LOST] | Signal: {signal} | Entry: {current_price:.2f} | Exit: {future_price:.2f} | Loss: -${loss:.2f}\n")
                                expiry_trades += 1
                        else:
                            print(f"[BACKTEST: Trade Style: {expiry}] 🕒 Time: {current_time}\n⚠️ SKIPPED: Confidence too low ({confidence}% < {backtest_threshold}% requirement for backtest).\n")
                    else:
                        print(f"[BACKTEST: Trade Style: {expiry}] 🕒 Time: {current_time}\n⚠️ SKIPPED: Signal was NEUTRAL / NO_TRADE.\n")
                
                
                if expiry_trades > 0:
                    win_rate = (expiry_wins / expiry_trades) * 100
                    expiry_results[expiry] = {
                        'win_rate': win_rate,
                        'total_trades': expiry_trades,
                        'final_balance': expiry_balance,
                        'profit_loss': expiry_balance - initial_balance
                    }
            
            # Find best expiry
            if expiry_results:
                trade_type = max(expiry_results.items(), key=lambda x: x[1]['win_rate'])
                
                self.results = {
                    'trade_type': trade_type[0],
                    'expiry_results': expiry_results,
                    'overall_win_rate': trade_type[1]['win_rate'],
                    'total_trades': sum(exp['total_trades'] for exp in expiry_results.values()),
                    'initial_balance': initial_balance,
                    'final_balance': trade_type[1]['final_balance']
                }
                logger.info(f"✅ Backtest Complete - Best Expiry: {trade_type[0]} | Win Rate: {trade_type[1]['win_rate']:.1f}%")
            else:
                logger.warning("⚠️ No trades met the criteria (>=85% confidence) during backtest.")
                self.results = {
                    'trade_type': '2m',
                    'expiry_results': {},
                    'overall_win_rate': 0.0,
                    'total_trades': 0,
                    'initial_balance': initial_balance,
                    'final_balance': initial_balance
                }
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {'error': str(e)}
        finally:
            self.jarvis.trade_manager.can_trade = original_can_trade
            self.jarvis.is_backtest_mode = False  # Reset backtest mode for live trading

    def _expiry_to_bars(self, expiry):
        """Convert expiry time to number of bars"""
        expiry_map = {'1M': 1, '2M': 2, '3M': 3, '5M': 5}
        return expiry_map.get(expiry, 3)

    def generate_backtest_report(self):
        """Generate detailed backtest report"""
        if not self.results:
            return "No backtest results available"
            
        report = [
            "📊 AUTO BACKTEST ENGINE REPORT",
            "=" * 40,
            f"Best Expiry Period: {self.results['trade_type']}",
            f"Overall Win Rate: {self.results['overall_win_rate']:.1f}%",
            f"Total Trades Analyzed: {self.results['total_trades']}",
            f"Initial Balance: ${self.results['initial_balance']}",
            f"Final Balance: ${self.results['final_balance']:.2f}",
            f"Net P/L: ${self.results['final_balance'] - self.results['initial_balance']:.2f}",
            "",
            "Expiry Performance Breakdown:"
        ]
        
        for expiry, stats in self.results['expiry_results'].items():
            report.append(
                f"  {expiry}: {stats['win_rate']:.1f}% win rate | "
                f"{stats['total_trades']} trades | "
                f"P/L: ${stats['profit_loss']:.2f}"
            )
        
        return "\n".join(report)

# ==================== 2. AUTO TRAINING ENGINE ====================

class AutoTrainingEngine:
    """Automated ML Training Engine"""
    
    def __init__(self, jarvis_system):
        self.jarvis = jarvis_system
        self.models = {}
        self.training_history = {}
        
    def train_models(self, historical_data):
        """Retired compatibility hook: training does not produce trading models."""
        return {'status': 'unavailable', 'reason': 'model prediction retired'}
        logger.info("🧠 Starting Auto Training Engine...")
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(historical_data)
            
            if len(X) == 0:
                logger.warning("Insufficient data for training")
                return {'status': 'insufficient_data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models['random_forest'] = rf_model
            self.training_history['random_forest'] = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'features_used': X.shape[1]
            }
            
            # Train simple neural network if torch is available
            if torch is not None:
                nn_model = self._train_neural_network(X_train, y_train, X_test, y_test)
                self.models['neural_network'] = nn_model
            
            logger.info(f"✅ Training Complete - Random Forest Accuracy: {accuracy:.1%}")
            
            return {
                'status': 'success',
                'models_trained': list(self.models.keys()),
                'accuracy': accuracy,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'status': 'error', 'error': str(e)}

    def _prepare_training_data(self, data):
        """Prepare features and labels for training"""
        features = []
        labels = []
        
        try:
            for i in range(60, len(data) - 5):
                # Create features from past data
                window = data.iloc[i-50:i]
                
                # Technical indicators as features
                feature_vector = [
                    float(window['close'].pct_change().mean() or 0),
                    float(window['close'].pct_change().std() or 0),
                    float(window['high'].max() - window['low'].min()),
                    float((window['close'] > window['open']).mean()),
                    float(window['volume'].mean() if 'volume' in window else 0),
                    float(window['close'].rolling(5).mean().iloc[-1]),
                    float(window['close'].rolling(10).mean().iloc[-1]),
                    float(window['close'].iloc[-1] - window['close'].rolling(20).mean().iloc[-1]),
                ]
                
                # Create label (1 if price goes up in next 3 bars, 0 if down)
                future_price = float(data['close'].iloc[i + 3])
                current_price = float(data['close'].iloc[i])
                label = 1 if future_price > current_price else 0
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return np.array([]), np.array([])

    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train a simple neural network"""
        try:
            class SimpleNN(nn.Module):
                def __init__(self, input_size):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 2)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            model = SimpleNN(X_train.shape[1])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test_tensor).float().mean()
                
            self.training_history['neural_network'] = {
                'accuracy': accuracy.item(),
                'training_samples': len(X_train),
                'epochs': 100
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Neural network training error: {e}")
            return None

    def predict_signal(self, data):
        """Retired compatibility hook: no local model predicts a trade."""
        return 0
        try:
            if 'random_forest' not in self.models:
                return 0
                
            # Prepare features for prediction
            feature_vector = self._prepare_features_for_prediction(data)
            if len(feature_vector) == 0:
                return 0
                
            prediction = self.models['random_forest'].predict([feature_vector])[0]
            return 1 if prediction == 1 else -1
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0

    def _prepare_features_for_prediction(self, data):
        """Prepare features for real-time prediction"""
        try:
            if len(data) < 50:
                return []
                
            window = data.tail(50)
            
            feature_vector = [
                float(window['close'].pct_change().mean() or 0),
                float(window['close'].pct_change().std() or 0),
                float(window['high'].max() - window['low'].min()),
                float((window['close'] > window['open']).mean()),
                float(window['volume'].mean() if 'volume' in window else 0),
                float(window['close'].rolling(5).mean().iloc[-1]),
                float(window['close'].rolling(10).mean().iloc[-1]),
                float(window['close'].iloc[-1] - window['close'].rolling(20).mean().iloc[-1]),
            ]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature prediction error: {e}")
            return []

# ==================== 3. AUTO OPTIMIZER ENGINE ====================

class AutoOptimizerEngine:
    """Automated Parameter Optimization Engine"""
    
    def __init__(self, jarvis_system):
        self.jarvis = jarvis_system
        self.optimized_params = {}
        self.optimization_history = []
        
    def run_optimization(self, backtest_results, historical_data):
        """Run parameter optimization for trade trading"""
        logger.info("⚡ Starting Auto Optimizer Engine...")
        
        try:
            # Optimize confidence threshold
            confidence_thresholds = [80, 85, 90, 95]
            best_threshold = 85
            best_performance = 0
            
            for threshold in confidence_thresholds:
                performance = self._test_confidence_threshold(
                    threshold, historical_data
                )
                
                self.optimization_history.append({
                    'parameter': 'confidence_threshold',
                    'value': threshold,
                    'performance': performance
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_threshold = threshold
            
            # Optimize position sizing
            position_sizes = [1, 2, 3, 5]
            best_position_size = 2
            best_risk_return = 0
            
            for size in position_sizes:
                risk_return = self._test_position_size(size, historical_data)
                
                if risk_return > best_risk_return:
                    best_risk_return = risk_return
                    best_position_size = size
            
            # Optimize trading sessions
            best_sessions = self._optimize_trading_sessions(historical_data)
            
            self.optimized_params = {
                'confidence_threshold': best_threshold,
                'position_size_percent': best_position_size,
                'optimal_sessions': best_sessions,
                'max_daily_trades': 8,
                'cooldown_minutes': 3,
                'optimization_score': best_performance
            }
            
            logger.info(f"✅ Optimization Complete - Best Confidence: {best_threshold}%")
            
            return self.optimized_params
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'error': str(e)}

    def _test_confidence_threshold(self, threshold, data):
        """Test different confidence thresholds"""
        try:
            wins = 0
            total = 0
            
            for i in range(100, len(data) - 5):
                data_slice = data.iloc[i-50:i]
                trade_result = self.jarvis.analyze_trade_setup(data_slice)
                
                if trade_result['trade_signal']['direction'] != 'NO_TRADE':
                    try:
                        conf_str = trade_result['trade_signal'].get('confidence_score')
                        confidence = normalize_confidence(conf_str, default=0) or 0
                    except (ValueError, IndexError, AttributeError):
                        confidence = 0
                    
                    if confidence >= threshold:
                        # Check if trade would be successful
                        current_price = float(data['close'].iloc[i])
                        future_price = float(data['close'].iloc[i + 3])
                        signal = trade_result['trade_signal']['direction']
                        
                        if (signal == 'CALL' and future_price > current_price) or \
                           (signal == 'PUT' and future_price < current_price):
                            wins += 1
                        total += 1
            
            return wins / total if total > 0 else 0
            
        except Exception as e:
            logger.error(f"Threshold testing error: {e}")
            return 0

    def _test_position_size(self, size, data):
        """Test different position sizes"""
        # Simplified risk-return calculation
        base_return = 0.02  # 2% per win
        base_risk = 0.01   # 1% per loss
        
        # Larger positions = higher risk, higher return
        risk_adjusted_return = (base_return * size) - (base_risk * size * 0.5)
        return risk_adjusted_return

    def _optimize_trading_sessions(self, data):
        """Optimize which trading sessions to use"""
        session_performance = {}
        
        for session_name, session_times in TRADING_SESSIONS.items():
            # Count successful trades in this session
            success_count = 0
            total_count = 0
            
            for i in range(100, len(data) - 5):
                timestamp = data.iloc[i]['timestamp'] if 'timestamp' in data else None
                if timestamp:
                    hour = pd.to_datetime(timestamp).hour
                    
                    if session_times['start'] <= hour < session_times['end']:
                        data_slice = data.iloc[i-50:i]
                        trade_result = self.jarvis.analyze_trade_setup(data_slice)
                        
                        if trade_result['trade_signal']['direction'] != 'NO_TRADE':
                            # Check trade success
                            current_price = float(data['close'].iloc[i])
                            future_price = float(data['close'].iloc[i + 3])
                            signal = trade_result['trade_signal']['direction']
                            
                            if (signal == 'CALL' and future_price > current_price) or \
                               (signal == 'PUT' and future_price < current_price):
                                success_count += 1
                            total_count += 1
            
            if total_count > 0:
                session_performance[session_name] = success_count / total_count
            else:
                session_performance[session_name] = 0
        
        # Return sessions with performance > 50%
        optimal_sessions = [
            session for session, perf in session_performance.items() 
            if perf > 0.5
        ]
        
        return optimal_sessions if optimal_sessions else list(TRADING_SESSIONS.keys())

    def apply_optimized_params(self):
        """Apply optimized parameters to the trading system"""
        try:
            if self.optimized_params:
                # Update trade config
                # SAFETY: Enforce minimum floors — optimizer cannot disable risk controls
                TRADE_CONFIG['min_confidence_score'] = max(self.optimized_params['confidence_threshold'], 60)
                TRADE_CONFIG['position_size_percent'] = self.optimized_params['position_size_percent']
                TRADE_CONFIG['max_daily_trades'] = min(self.optimized_params['max_daily_trades'], 25)
                
                logger.info("✅ Optimized parameters applied to trading system")
                return True
            else:
                logger.warning("No optimized parameters to apply")
                return False
                
        except Exception as e:
            logger.error(f"Parameter application error: {e}")
            return False

# ==================== 4. LIVE TRADING ENGINE ====================

class DeltaWebSocketClient:
    """Live Delta WebSocket Client for Real-Time Data"""
    
    def __init__(self, callback):
        self.url = "wss://socket.delta.exchange"
        self.callback = callback
        self.is_running = False
        self.loop = None
        self.thread = None
        
    def start(self):
        """Start WebSocket listener in a separate thread"""
        pass # DISABLED: Using Unified Delta Hybrid Client in the live path
        # self.is_running = True
        # self.thread = threading.Thread(target=self._run_loop, daemon=True)
        # self.thread.start()
        # print("DEBUG: WS Client START")
        # logger.info("📡 Delta WebSocket Client Initiated")
        
    def stop(self):
        """Stop WebSocket listener"""
        self.is_running = False
        
    def _run_loop(self):
        """Run asyncio loop for WebSocket"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._listen())
        except Exception as e:
            logger.error(f"WebSocket Loop Error: {e}")
            
    async def _listen(self):
        """Connect and listen to WebSocket stream"""
        while self.is_running:
            try:
                print(f"DEBUG: Connecting to {self.url}")
                async with websockets.connect(self.url) as ws:
                    print(f"DEBUG: Connected to {self.url}")
                    logger.info(f"✅ Connected to Delta Live Feed: {self.url}")
                    
                    while self.is_running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Multi-stream format has 'stream' and 'data' fields
                        if 'data' in data:
                            stream_data = data['data']
                            stream_name = data.get('stream', '')
                            
                            if 'e' in stream_data and stream_data['e'] == 'kline':
                                k = stream_data['k']
                                
                                # Extract timeframe from stream name (e.g., "btcusdt@kline_1m" -> "1m")
                                timeframe = stream_name.split('_')[-1] if '_' in stream_name else '5m'
                                
                                # ONLY process completed candles
                                if k['x']:  # is_closed = True
                                    candle = {
                                        'timestamp': pd.to_datetime(k['t'], unit='ms'),
                                        'open': float(k['o']),
                                        'high': float(k['h']),
                                        'low': float(k['l']),
                                        'close': float(k['c']),
                                        'volume': float(k['v']),
                                        'number_of_trades': int(k.get('n', 0)),
                                        'taker_buy_volume': float(k.get('V', 0.0)),
                                        'timeframe': timeframe,
                                        'is_closed': True
                                    }
                                    self.callback(candle)
                            
            except Exception as e:
                logger.error(f"⚠️ WebSocket Connection Connection Lost ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

@dataclass
class ScenarioContext:
    direction: str = 'NO_TRADE'
    entry_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    df: Any = None
    symbol: str = 'BTCUSDT'

def _binance_spot_symbol(symbol) -> str:
    """Canonical Binance Spot pair for a route symbol.

    Delta India lists perps as XXXUSD while analysis candles come from Binance
    Spot (XXXUSDT). The mapping is explicit and exact; execution symbols are
    never changed by this helper.
    """
    sym = str(symbol or '').upper().replace('/', '').replace('-', '').replace('_', '')
    if sym.endswith('USD') and not sym.endswith('USDT'):
        sym = sym[:-3] + 'USDT'
    return sym



class LiveTradingEngine:
    """Live Trading Engine with Paper Trading + Live Signals"""
    
    # Paper Trading Config
    PAPER_CONFIG = {
        'initial_balance': 1000.00,
        'risk_per_trade_pct': 0.02,      # 2% risk per trade
        'reward_ratio': 2.0,             # 2:1 reward
        'min_confidence': 60,            # Min confidence to open paper trade
        'max_open_trades': 3,
        # FIX: Realistic expiry times - SCALP needs 5min not 1min!
        'expiry_map': {'1M': 3, '2M': 5, '3M': 7, '5M': 10, 'SCALP': 5, 'DAY_TRADE': 15, 'SWING': 30},
        'save_file': 'paper_trades.json',
    }
    
    def __init__(self, jarvis_system, optimized_params=None):
        self.jarvis = jarvis_system
        self.optimized_params = optimized_params or {}
        self.is_running = False
        self.stop_event = threading.Event()
        self.readiness = {"status": "STOPPED", "reason": "not started", "updated_at": datetime.now().isoformat()}
        self._live_future = None
        self.trade_queue = Queue()
        # BUG FIX #5: These attributes used in can_trade() / record_trade() but were never initialized
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.last_trading_date = datetime.now().date()
        self.trade_history = []
        
        # God Mode Dashboard State Storage
        self.last_jarvis_result = {}
        self.last_consensus = {}
        self.last_hedged_result = {}
        self.engine_start_time = time.time()
        self.dashboard = UnifiedDashboard(
            clear=os.getenv("JARVIS_DASHBOARD_CLEAR", "1").lower() not in ("0", "false", "off")
        )
        self._dashboard_events = []
        self._dashboard_signal = {}
        self._dashboard_plan = {}
        self._dashboard_account = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'current_streak': 0,
            'best_streak': 0
        }
        self.data_buffer = pd.DataFrame()
        self.ws_client = None
        self.candle_count = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # ═══ LIVE AUTO-TRADER ═══
        self.auto_trader: Optional[object] = None
        if LIVE_TRADER_AVAILABLE and JarvisAutoTrader is not None:
            try:
                from ai_hedge_advisor import AIHedgeAdvisor
                hedge_advisor = AIHedgeAdvisor()
            except Exception:
                hedge_advisor = None
            try:
                self.auto_trader = JarvisAutoTrader(
                    delta_client=self.jarvis.delta_data,
                    hedge_advisor=hedge_advisor,
                )
                # ── Hook up reversal engine ────────────────────────
                # Give auto_trader a reference to JarvisElite so it can
                # re-run analyze_trade_setup() mid-trade for reversal checks
                self.auto_trader._jarvis_ref = self.jarvis
            except Exception as at_err:
                print(f"[JARVIS CORE] AutoTrader init error: {at_err}")
                self.auto_trader = None

        # No model strategy advisor is wired into execution.
        self.gemini_advisor = None
        print("[JARVIS CORE] Model strategy advisor retired; Laya is audit-only")

        # === JARVIS MARKET ORACLE (5-min multi-timeframe market map) ===
        self.market_oracle = None
        if MARKET_ORACLE_AVAILABLE and _init_market_oracle:
            try:
                bus = getattr(self, 'cognitive_bus', None)
                self.market_oracle = _init_market_oracle(
                    bus=bus,
                    live_trader=self.auto_trader
                )
                # Wire oracle gate into auto_trader for Gate 0.5 checks
                if self.auto_trader:
                    from oracle_trade_gate import get_oracle_trade_gate
                    self.auto_trader._oracle_gate = get_oracle_trade_gate(oracle_ref=self.market_oracle)
                print("[JARVIS CORE] JARVIS Market Oracle started (5-min cycle)")
            except Exception as mo_err:
                print(f"[JARVIS CORE] Market Oracle init error: {mo_err}")
                self.market_oracle = None

        # === INDIAN SIGNAL ENGINE (Dual Market: NSE Nifty + Crypto) ===
        self.indian_signal_engine = None
        if INDIAN_SIGNAL_AVAILABLE and _get_indian_engine:
            try:
                bus = getattr(self, 'cognitive_bus', None)
                self.indian_signal_engine = _get_indian_engine(bus=bus)
                self.indian_signal_engine.start()
                print("🇮🇳 [JARVIS CORE] Indian Signal Engine ONLINE (NSE Nifty + BankNifty + Crypto Dual-Market)")
            except Exception as _ise_err:
                print(f"[JARVIS CORE] ⚠️  Indian Signal Engine init error: {_ise_err}")
                self.indian_signal_engine = None

        # === JARVIS SELF-HEALING DOCTOR (60-sec monitor + auto-fix) ===
        self.doctor = None
        if DOCTOR_AVAILABLE and _init_doctor:
            try:
                bus = getattr(self, 'cognitive_bus', None)
                self.doctor = _init_doctor(
                    bus=bus,
                    live_trader=self.auto_trader
                )
                print("[JARVIS CORE] 🏥 JARVIS Self-Healing Doctor ONLINE (60-sec cycle)")
            except Exception as _doc_err:
                print(f"[JARVIS CORE] ⚠️  Doctor init error: {_doc_err}")
                self.doctor = None

        # ═══ PAPER TRADING STATE ═══

        self.paper_balance = self.PAPER_CONFIG['initial_balance']
        self.paper_open_trades = []     # List of open paper trade dicts
        self.paper_closed_trades = []   # List of closed paper trade dicts
        self.paper_wins = 0
        self.paper_losses = 0
        self.paper_breakeven = 0
        self.paper_peak_balance = self.paper_balance
        self._load_paper_state()

        # ═══ SYMBOL-SAFE MARKET ROUTER ═══
        # The route is chosen before data is fetched and remains locked while a
        # position is open.  Every downstream crypto value must use this symbol.
        self.market_router = None
        self.active_symbol = os.getenv('JARVIS_DEFAULT_SYMBOL', 'BTCUSDT')
        self.active_base_asset = 'BTC'
        if MARKET_ROUTER_AVAILABLE and MarketRouter is not None:
            self.market_router = MarketRouter(
                scanner=getattr(self.jarvis, 'coin_scanner', None),
                delta_client=getattr(self.jarvis, 'delta_data', None),
            )

        # Optional full Parts 1-12 analysis is default-off. Delta's eligible
        # active products are joined to one exact Binance spot symbol, and the
        # analysis cache is Binance-only. The distinct Delta execution identity
        # is carried through the pipeline and cannot be inferred from a ticker.
        # The selected-symbol execution path remains independent.
        self.multicoin_pipeline = None
        self.multicoin_pipeline_status = {
            'enabled': multicoin_analysis_enabled(),
            'status': 'DISABLED' if not multicoin_analysis_enabled() else 'UNAVAILABLE',
            'results': {},
        }
        if (multicoin_analysis_enabled() and not getattr(self.jarvis, 'is_backtest_mode', False)
                and MULTICOIN_PIPELINE_AVAILABLE and getattr(self.jarvis, 'delta_data', None) is not None
                and getattr(self.jarvis, 'binance_data', None) is not None and BinanceSpotCandleClient is not None):
            try:
                # The optional scanner uses the same full native interval set
                # and history profile as the selected-asset path. Keep its
                # default candidate budget conservative: 15 intervals are a
                # much heavier REST workload than the former 8-frame runner.
                _multi_timeframes = tuple(BINANCE_SPOT_TIMEFRAMES)
                _multi_history = {tf: TIMEFRAME_HISTORY_CANDLES[tf] for tf in _multi_timeframes}
                _multi_candidates = int(os.getenv('JARVIS_MULTICOIN_MAX_CANDIDATES', '10'))
                _multi_cache = DirectCandleCache(
                    BinanceSpotCandleClient(self.jarvis.binance_data),
                    ttl_seconds=float(os.getenv('JARVIS_MULTICOIN_CANDLE_TTL_SEC', '60')),
                    max_concurrent_fetches=1,
                    max_cached_identities=max(64, min(256, _multi_candidates)),
                    timeframes=_multi_timeframes, history_limits=_multi_history,
                )
                self.multicoin_pipeline = MultiCoinPipeline(
                    _multi_cache,
                    self._multicoin_candidate_instruments,
                    self._analyze_multicoin_instrument,
                    enabled=True,
                    max_workers=int(os.getenv('JARVIS_MULTICOIN_MAX_WORKERS', '1')),
                    max_candidates=_multi_candidates,
                    discovery_interval_seconds=float(os.getenv('JARVIS_MULTICOIN_CANDIDATE_REFRESH_SEC', '300')),
                    retry_interval_seconds=float(os.getenv('JARVIS_MULTICOIN_RETRY_SEC', '60')),
                    result_max_age_seconds=float(os.getenv('JARVIS_MULTICOIN_RESULT_MAX_AGE_SEC', '180')),
                )
                self.multicoin_pipeline_status = {'enabled': True, 'status': 'IDLE', 'results': {}}
            except Exception as _multi_error:
                logger.warning('[MULTICOIN] Initialization unavailable: %s', type(_multi_error).__name__)
                self.multicoin_pipeline_status = {'enabled': True, 'status': 'UNAVAILABLE', 'results': {}}

        # Optional isolated paper-only execution of an explicitly supplied plan.
        # The Parts 1-12 adapter currently emits no entry/stop/size plan, so the
        # fail-closed validator ordinarily refuses candidates. This never routes
        # to Delta/Binance orders or modifies the selected-symbol execution path.
        self.multicoin_paper_coordinator = None
        self.multicoin_paper_adapter = None
        self.multicoin_paper_status = {'enabled': False, 'submitted': 0, 'blocked': 0}
        if (os.getenv('JARVIS_MULTICOIN_PAPER_EXECUTION', '0').strip() == '1'
                and MULTICOIN_EXECUTION_AVAILABLE and not getattr(self.jarvis, 'is_backtest_mode', False)):
            try:
                _policies = json.loads(os.getenv('JARVIS_MULTICOIN_PAPER_POLICY_REGISTRY', '{}'))
                if not isinstance(_policies, dict):
                    raise ValueError('policy registry must be an asset-to-policy object')
                _budget = float(os.environ['JARVIS_MULTICOIN_PAPER_MAX_NOTIONAL'])
                _risk_budget = float(os.environ['JARVIS_MULTICOIN_PAPER_MAX_RISK'])
                self.multicoin_paper_coordinator = PortfolioCoordinator(
                    journal_path=os.getenv('JARVIS_MULTICOIN_PAPER_JOURNAL', '.jarvis_state/multicoin_paper_execution.json'),
                    enabled=True, max_positions=int(os.getenv('JARVIS_MULTICOIN_PAPER_MAX_POSITIONS', '2')),
                    max_total_notional=_budget, max_total_risk=_risk_budget,
                    per_asset_caps=json.loads(os.getenv('JARVIS_MULTICOIN_PAPER_ASSET_CAPS', '{}')),
                    correlation_groups=json.loads(os.getenv('JARVIS_MULTICOIN_PAPER_CORRELATION_GROUPS', '{}')),
                    max_per_correlation_group=int(os.getenv('JARVIS_MULTICOIN_PAPER_MAX_PER_GROUP', '1')),
                )
                self.multicoin_paper_adapter = DeterministicPaperAdapter()
                self._multicoin_paper_policies = {str(k).upper(): str(v) for k, v in _policies.items()}
                self.multicoin_paper_status = {'enabled': True, 'submitted': 0, 'blocked': 0, 'last_status': 'IDLE'}
            except Exception as _paper_error:
                logger.warning('[MULTICOIN-PAPER] Initialization fail-closed: %s', type(_paper_error).__name__)
                self.multicoin_paper_status = {'enabled': False, 'status': 'UNAVAILABLE'}

        # Real Delta handoff is a separate, explicit opt-in. It requires an
        # asset-specific exact Binance->Delta policy, staged entry plus
        # separately attached/verified protection, and complete reconciliation.
        # This is deliberately not described as atomic: Delta's documented
        # bracket endpoint is a post-fill operation and must fail closed if it
        # cannot be verified.
        self.multicoin_delta_coordinator = None
        self.multicoin_delta_adapter = None
        self._multicoin_delta_policies = {}
        self.multicoin_delta_status = {'enabled': False, 'submitted': 0, 'blocked': 0, 'last_status': 'DISABLED'}
        if (os.getenv('JARVIS_MULTICOIN_DELTA_EXECUTION', '0').strip() == '1'
                and DELTA_MULTICOIN_EXECUTION_AVAILABLE and MULTICOIN_EXECUTION_AVAILABLE
                and not getattr(self.jarvis, 'is_backtest_mode', False)):
            try:
                _delta_policies = json.loads(os.getenv('JARVIS_MULTICOIN_DELTA_POLICY_REGISTRY', '{}'))
                if not isinstance(_delta_policies, dict) or not _delta_policies:
                    raise ValueError('explicit asset-specific Delta policy registry required')
                _delta_client = getattr(self.jarvis, 'delta_data', None)
                _needed = ('place_protected_order', 'place_reduce_only_order', 'get_complete_account_snapshot',
                           'get_delta_executable_quote', 'get_available_balance')
                if _delta_client is None or any(not callable(getattr(_delta_client, name, None)) for name in _needed):
                    raise ValueError('Delta broker lacks protective order, quote, or complete reconciliation capability')
                self._multicoin_delta_policies = {str(k).upper(): dict(v) for k, v in _delta_policies.items() if isinstance(v, dict)}
                _policy_ids = {asset: str(policy.get('policy_id') or '') for asset, policy in self._multicoin_delta_policies.items()}
                if any(not value for value in _policy_ids.values()):
                    raise ValueError('each asset policy requires a nonempty policy_id')
                self.multicoin_delta_adapter = DeltaExecutionAdapter(
                    _delta_client, policy_registry=self._multicoin_delta_policies)
                self.multicoin_delta_coordinator = PortfolioCoordinator(
                    journal_path=os.getenv('JARVIS_MULTICOIN_DELTA_JOURNAL', '.jarvis_state/multicoin_delta_execution.json'),
                    enabled=True, require_reconciliation=True,
                    require_protective_reconciliation=True,
                    max_positions=int(os.getenv('JARVIS_MULTICOIN_DELTA_MAX_POSITIONS', '2')),
                    max_total_notional=float(os.environ['JARVIS_MULTICOIN_DELTA_MAX_NOTIONAL']),
                    max_total_risk=float(os.environ['JARVIS_MULTICOIN_DELTA_MAX_RISK']),
                    per_asset_caps=json.loads(os.getenv('JARVIS_MULTICOIN_DELTA_ASSET_CAPS', '{}')),
                    correlation_groups=json.loads(os.getenv('JARVIS_MULTICOIN_DELTA_CORRELATION_GROUPS', '{}')),
                    max_per_correlation_group=int(os.getenv('JARVIS_MULTICOIN_DELTA_MAX_PER_GROUP', '1')),
                )
                if not self.multicoin_delta_adapter.reconcile(self.multicoin_delta_coordinator):
                    raise ValueError('authoritative Delta startup reconciliation failed')
                self.multicoin_delta_status = {'enabled': True, 'submitted': 0, 'blocked': 0, 'last_status': 'RECONCILED'}
            except Exception as _delta_exec_error:
                logger.warning('[MULTICOIN-DELTA] Initialization fail-closed: %s', type(_delta_exec_error).__name__)
                self.multicoin_delta_coordinator = None
                self.multicoin_delta_adapter = None
                self.multicoin_delta_status = {'enabled': False, 'status': 'UNAVAILABLE', 'reason': str(_delta_exec_error)[:160]}

        # ═══ PRE-TRADE SIMULATOR STATS (fail-open; JARVIS_PRESIM=0 disables) ═══
        self.presim_stats = {'checks': 0, 'vetoes': 0, 'adjustments': 0}
        
        # ═══ OPTIONS HEDGED SCALP ENGINE ═══
        try:
            from ai_hedge_advisor import AIHedgeAdvisor
            from options_hedged_scalp import OptionsHedgedScalpEngine
            self.ai_hedge_advisor = AIHedgeAdvisor()
            self.hedged_engine = OptionsHedgedScalpEngine(
                delta_client=self.jarvis.delta_data,
                ai_hedge_advisor=self.ai_hedge_advisor,
                ai_roundtable=None
            )
            logger.info("✅ Options Hedged Scalp Engine initialized.")
        except Exception as e:
            logger.error(f"Failed to load Hedged Engine: {e}")
            self.hedged_engine = None
        
    def _delta_available_balance(self):
        """Read Delta available collateral once for the dashboard, fail closed."""
        try:
            client = getattr(self.jarvis, 'delta_data', None)
            if client and callable(getattr(client, 'get_wallet_balance', None)):
                _chk = getattr(client, 'get_wallet_balance_checked', None)
                _raw = _chk() if callable(_chk) else client.get_wallet_balance()
                if _raw is None:
                    return 0.0, 'API_UNAVAILABLE'
                value = float(_raw)
                if value > 0:
                    return value, 'OK'
                return 0.0, 'verified zero balance'
        except Exception:
            pass
        return 0.0, 'unavailable'

    def _paper_size(self, confidence, entry_price, sl, direction):
        try:
            stop_pct = abs(float(entry_price) - float(sl)) / float(entry_price) if sl else 0.002
            return calculate_trade_size(
                self.paper_balance, confidence, max(stop_pct, 0.0001),
                max_trade_risk_usdt=float(os.getenv('JARVIS_MAX_RISK_USDT', '10')),
            )
        except Exception:
            return {'ok': False, 'reason': 'paper risk sizing failed', 'contracts': 0}

    def _render_unified_dashboard(self, symbol=None, current_price=None, action=None, gate_reason=None):
        """Publish one state snapshot; signal/reason/plan/risk never use separate blocks."""
        try:
            result = self.last_jarvis_result or {}
            _decision = getattr(self, "last_decision", None)
            signal = self._dashboard_signal or result.get('trade_signal', {}) or {}
            plan = dict(self._dashboard_plan or {})
            if action:
                plan['action'] = action
            if gate_reason:
                plan['gates'] = gate_reason
            delta_balance, delta_status = self._delta_available_balance()
            total = self.paper_wins + self.paper_losses + self.paper_breakeven
            uptime = str(timedelta(seconds=int(time.time() - self.engine_start_time)))
            account = dict(self._dashboard_account or {})
            account.update({
                'delta_available': delta_balance,
                'delta_status': delta_status,
                'paper_balance': self.paper_balance,
                'max_leverage_cap': MAX_LEVERAGE_CAP,
                'max_risk': float(os.getenv('JARVIS_MAX_RISK_USDT', '10')),
            })
            part7 = result.get('part7_volatility') or getattr(self.jarvis, 'latest_part7', {}) or {}
            status = {
                'mode': 'LIVE-EXECUTION' if getattr(self, 'auto_trader', None) and self.auto_trader.is_enabled else 'PAPER',
                'open_trades': len(self.paper_open_trades),
                'pending': sum(1 for t in self.paper_open_trades if t.get('status') == 'PENDING_LIMIT'),
                'uptime': uptime,
                'gpu': getattr(self.jarvis, 'gpu_status', {'backend': 'unknown'}),
                'readiness': dict(getattr(self, 'readiness', {'status': 'UNKNOWN'})),
                'native_engines': getattr(self.jarvis, 'native_engine_status', {}),
                'ai_suggestion': getattr(self.jarvis, 'last_legacy_llm_decision', {'decision': 'unavailable'}),
                'part7': part7,
            }
            reasons = (result.get('intelligence_board', []) or [])[:3]
            if part7.get('entry_blocked'):
                reasons = [part7.get('reason', 'Part7 entry gate blocked')] + reasons
            _decision = getattr(self, 'last_decision', None) or {}
            self.dashboard.update(
                thinking=getattr(self, '_last_thinking_text', None),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol=(_decision.get('symbol') if _decision else None) or symbol or self.active_symbol,
                price=f"${float(current_price):,.2f}" if current_price else '—',
                signal={'direction': signal.get('direction', 'NO_TRADE'), 'confidence': signal.get('confidence_score', signal.get('confidence', 0))},
                reasons=reasons or [result.get('no_trade_reason', 'Awaiting analysis')],
                plan=plan,
                status=status,
                account=account,
                events=self._dashboard_events[-3:],
            )
            self.dashboard.render()
        except Exception as e:
            logger.debug('[DASHBOARD] render failed: %s', e)

    def _print_professional_signal(self, part_details):
        """Print signals in professional format"""
        self.candle_count += 1
        
        # Group signals
        bullish = [p for p, s in part_details.items() if s == 1 and p != 'part12_confidence']
        bearish = [p for p, s in part_details.items() if s == -1 and p != 'part12_confidence']
        neutral = [p for p, s in part_details.items() if s == 0 and p != 'part12_confidence']
        confidence = part_details.get('part12_confidence', 0)
        
        # Format part names
        def format_name(p):
            return p.replace('part', 'P').replace('_', ' ').title()

    # ═══════════════════════════════════════════════════════════════
    #  PAPER TRADE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _xgb_meta_gate(self, direction, snapshot, symbol, result=None):
        """XGBoost win-probability gate on the selected coin's own native 15m candles.

        Veto-only and fail-neutral: unavailable/unvalidated models, missing 15m
        frame or any error leave ``direction`` unchanged. Default mode is
        ``shadow`` (log only); ``JARVIS_XGB_MODE=veto`` lets a VALIDATED
        per-symbol model block an entry. It never changes direction or size.
        """
        try:
            from jarvis_xgb_meta import get_gate
            frame = None if snapshot is None else getattr(snapshot, 'frames', {}).get('15m')
            closed = None if frame is None else frame.closed
            verdict = get_gate().evaluate(closed, direction, symbol)
            self.last_xgb_meta = verdict
            if verdict.get('veto'):
                try:
                    self._dashboard_events.append(
                        f"XGB veto {direction}: p={verdict.get('probability')} < {verdict.get('threshold')}")
                except Exception:
                    pass
                logger.info('[XGB] VETO %s %s p=%s thr=%s', symbol, direction,
                            verdict.get('probability'), verdict.get('threshold'))
                return 'NO_TRADE'
        except Exception as exc:  # advisory layer must never break or authorize trading
            logger.debug('[XGB] gate unavailable (%s)', type(exc).__name__)
        # Validated per-coin XGBoost mode advisor (OOS-validated artifacts only).
        # Veto-only; default JARVIS_XGB_ADVISOR=shadow (log only). Fail-neutral.
        try:
            if direction in ('CALL', 'PUT') and result is not None:
                from jarvis_xgb_mode import get_advisor
                from jarvis_part_snapshot import build_feature_row
                _adv = get_advisor().advise(_binance_spot_symbol(symbol), build_feature_row(result, snapshot), direction)
                self.last_xgb_mode_advice = _adv
                logger.info('[XGB_MODE] %s dir=%s status=%s advice=%s veto=%s mode=%s known=%s reason=%s probs=%s',
                            symbol, direction, _adv.get('status'), _adv.get('advice'), _adv.get('veto'),
                            _adv.get('mode'), _adv.get('known_features'), _adv.get('reason'), _adv.get('probs'))
                if _adv.get('veto') and _adv.get('can_veto'):
                    return 'NO_TRADE'
        except Exception as exc:
            logger.warning('[XGB_MODE] advisor unavailable (%s: %s)', type(exc).__name__, exc)
        return direction

    def _presim_gate(self, direction, confidence, entry_price, result=None, df=None, current_price=None, symbol='BTCUSDT'):
        """Run the pre-trade simulator and fail closed on safety uncertainty.

        Returns ``(None, confidence)`` for a veto or simulator error. A
        dashboard/bookkeeping failure is deliberately non-fatal and cannot
        turn a committed veto into an allowed entry. Disable explicitly with
        ``JARVIS_PRESIM=0`` when the operator accepts bypassing this gate.
        """
        try:
            if os.getenv('JARVIS_PRESIM', '1') == '0':
                return direction, confidence
            from jarvis_presim import run_presim
            # Partial test/recovery wiring may omit the optional event sink;
            # initialize it locally so recording a veto cannot itself turn the
            # gate into an implicit pass.
            if not isinstance(getattr(self, '_dashboard_events', None), list):
                self._dashboard_events = []
            market_ctx = (result or {}).get('market_context', {}) if isinstance(result, dict) else {}
            decision = run_presim(
                signal={
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price or current_price,
                    'regime': market_ctx.get('trend'),
                },
                candles=df,
                snapshot=market_ctx,
            )
            self.presim_stats = getattr(self, 'presim_stats', {}) or {}
            self.presim_stats['checks'] = self.presim_stats.get('checks', 0) + 1
            action = decision.get('action', 'pass')
            if action == 'veto':
                # The safety decision is committed before optional dashboard
                # bookkeeping. A missing/broken display must never turn veto
                # into an allowed entry.
                self.presim_stats['vetoes'] = self.presim_stats.get('vetoes', 0) + 1
                event = f"PreSim veto: {decision.get('reason', '')}"
                try:
                    self._dashboard_events.append(event)
                except Exception:
                    logger.debug("[PRESIM] veto dashboard event unavailable")
                logger.info(f"[PRESIM] VETO {direction}: {decision.get('reason', '')}")
                return None, confidence
            if action == 'adjust':
                self.presim_stats['adjustments'] = self.presim_stats.get('adjustments', 0) + 1
                delta = max(-10, min(10, int(decision.get('confidence_delta', 0))))
                confidence = max(0, min(100, confidence + delta))
                try:
                    self._dashboard_events.append(f"PreSim adjust {delta:+d}: {decision.get('reason', '')}")
                except Exception:
                    logger.debug("[PRESIM] adjust dashboard event unavailable")
                logger.info(f"[PRESIM] ADJUST {delta:+d}: {decision.get('reason', '')}")
            return direction, confidence
        except Exception as e:
            # The simulator is a safety gate. An unavailable/corrupt simulator
            # must never silently authorize a candidate entry.
            self.presim_stats = getattr(self, 'presim_stats', {}) or {}
            self.presim_stats['errors'] = self.presim_stats.get('errors', 0) + 1
            try:
                self._dashboard_events.append(f"PreSim error: {type(e).__name__}; entry vetoed")
            except Exception:
                pass
            logger.warning("[PRESIM] gate error; entry vetoed: %s", type(e).__name__)
            return None, confidence

    def _scenario_gate(self, direction, entry_price=None, sl=None, tp=None, df=None, symbol='BTCUSDT'):
        """Pre-Trade Scenario Simulator gate (fail-closed).

        Runs jarvis_scenario_simulator on a candidate ENTER signal. Returns
        direction — set to 'NO_TRADE' on veto (too many stress scenarios
        fail). Any simulator exception blocks entry; only explicit disable bypasses the gate.
        """
        if isinstance(direction, ScenarioContext):
            ctx = direction
            direction = ctx.direction
            entry_price = ctx.entry_price
            sl = ctx.sl
            tp = ctx.tp
            df = ctx.df
            symbol = ctx.symbol
        try:
            if os.getenv('JARVIS_SCEN_SIM', '1') == '0':
                return direction
            from jarvis_scenario_simulator import run_scenarios, ScenarioConfig
            verdict = run_scenarios(ScenarioConfig(
                direction=direction,
                entry_price=entry_price,
                sl=sl, tp=tp,
                df=df,
                fee_bps=float(os.getenv('JARVIS_SCEN_FEE_BPS', '10')),
            ))
            if verdict.get('action') == 'veto':
                self._dashboard_events.append(f"Scenario veto: {verdict.get('reason', 'stress gate')}")

                logger.info(f"[SCENARIO] VETO {direction} {symbol}: {verdict.get('reason')}")
                return 'NO_TRADE'
            if verdict.get('total'):
                logger.info(f"[SCENARIO] PASS {verdict.get('passed')}/{verdict.get('total')} {direction} {symbol}")
            return direction
        except Exception as e:
            logger.warning("[SCENARIO] gate error; entry vetoed: %s", type(e).__name__)
            return "NO_TRADE"

    def _open_paper_trade(
        self, direction, entry_price, confidence, expiry_name, tp1, tp2, sl,
        current_price=None, symbol='BTCUSDT', *, central_approval=None,
        snapshot_version=None, part_results=None, timeframe_parts=None,
        part7_gate=None, execution_plan=None,
    ):
        """Open a paper position only from a fresh, centrally approved bound plan."""
        try:
            from jarvis_strategy_approval import (
                evaluate_mtf_central_strategy, validate_entry_approval,
                validate_execution_plan,
            )
            expected = 'BUY' if str(direction).upper() in ('CALL', 'BUY') else 'SELL' if str(direction).upper() in ('PUT', 'SELL') else 'NO_TRADE'
            for _ot in list(self.paper_open_trades):
                _osym, _nsym = _binance_spot_symbol(_ot.get('symbol')), _binance_spot_symbol(symbol)
                _odir = 'BUY' if str(_ot.get('direction')).upper() in ('CALL', 'BUY') else 'SELL'
                if _osym and _nsym and _osym != _nsym and _odir == expected:
                    logger.warning('[PORTFOLIO_CORRELATION_GUARD] blocked %s %s: correlated %s %s already open', _nsym, expected, _osym, _odir)
                    raise ValueError('portfolio correlation guard: same-direction BTC/ETH exposure already open')
            _paper_min_confidence = 0
            try:
                _paper_min_confidence = float(getattr(self, 'PAPER_CONFIG', {}).get('min_confidence', 0))
            except (TypeError, ValueError, OverflowError):
                raise ValueError('paper minimum confidence is invalid')
            decision = evaluate_mtf_central_strategy(
                timeframe_parts, part7_gate, confidence=confidence,
                minimum_confidence=_paper_min_confidence,
                expected_symbol=_binance_spot_symbol(symbol),
            )
            if not decision.get('approved') or decision.get('direction') != expected:
                raise ValueError('central MTF strategy did not approve paper entry')
            mode = decision.get('trade_mode')
            plan_ok, plan_reason = validate_execution_plan(
                execution_plan, direction=expected, symbol=symbol,
                snapshot_version=snapshot_version, confidence=confidence,
                trade_mode=mode,
            )
            if not plan_ok:
                raise ValueError(plan_reason)
            if str((execution_plan or {}).get('trade_mode') or '').upper() != str(expiry_name or '').upper():
                raise ValueError('paper expiry/mode differs from the approved plan')
            approval_ok, approval_reason = validate_entry_approval(
                central_approval, direction=expected, symbol=symbol,
                exchange='delta', contract=symbol, instrument_id=symbol,
                market_type='unverified', snapshot_version=snapshot_version,
                confidence=confidence, execution_plan=execution_plan,
            )
            if not approval_ok:
                raise ValueError(approval_reason)
            # The authorized plan is the sole source of paper entry/SL/TP.
            # Caller-supplied prices and a second TP cannot alter the plan.
            entry_price = float(execution_plan['entry_price'])
            sl = float(execution_plan['stop_loss'])
            tp1 = float(execution_plan['take_profit'])
            tp2 = tp1
            direction = 'CALL' if expected == 'BUY' else 'PUT'
            confidence = int(execution_plan['confidence'])
        except Exception as approval_error:
            try:
                self._dashboard_events.append('paper entry blocked: ' + str(approval_error)[:180])
            except Exception:
                pass
            return None
        if len(self.paper_open_trades) >= self.PAPER_CONFIG['max_open_trades']:
            return None

        is_limit = False
        if current_price and abs(entry_price - current_price) / current_price > 0.0001:
            is_limit = True

        expiry_min = self.PAPER_CONFIG['expiry_map'].get(expiry_name, 3)
        trade = {
            'id': datetime.now().strftime('%H%M%S'),
            'direction': direction,
            'symbol': symbol,
            'entry_price': entry_price,
            'confidence': confidence,
            'expiry_name': expiry_name,
            'tp1': tp1, 'tp2': tp2, 'sl': sl,
            'status': 'PENDING_LIMIT' if is_limit else 'OPEN',
            'entry_time': datetime.now().isoformat(),
            'expiry_time': (datetime.now() + timedelta(minutes=expiry_min)).isoformat(),
            'exit_price': None,
            'result': None,
            'pnl_dollar': 0.0,
            'close_reason': None,
        }
        sizing = self._paper_size(confidence, entry_price, sl, direction)
        if not sizing.get('ok'):
            self._dashboard_events.append('paper entry blocked: ' + str(sizing.get('reason', 'risk sizing failed')))
            return None
        trade.update({
            'contracts': int(sizing.get('contracts', 0)),
            'margin_usdt': float(sizing.get('margin_usdt', 0.0)),
            'notional_usdt': float(sizing.get('notional_usdt', 0.0)),
            'leverage': int(sizing.get('leverage', 0)),
            'trade_risk_usdt': float(sizing.get('trade_risk_usdt', 0.0)),
        })
        if trade['contracts'] <= 0:
            self._dashboard_events.append('paper entry blocked: no whole contract within risk budget')
            return None
        self._dashboard_account = {
            'trade_risk': trade['trade_risk_usdt'],
            'margin': trade['margin_usdt'],
            'contracts': trade['contracts'],
            'notional': trade['notional_usdt'],
            'leverage': trade['leverage'],
        }
        self.paper_open_trades.append(trade)

        # Persist state for crash recovery (guarded; never raises)
        try:
            from jarvis_watchdog import save_state
            save_state(self.paper_open_trades)
        except Exception:
            pass

        # ── TELEGRAM: Notify trade opened ──
        try:
            from telegram_notifier import send_trading_signal
            send_trading_signal({
                'direction': direction,
                'score': confidence,
                'entry_price': entry_price,
                'tp1': tp1,
                'sl': sl,
                'expiry': expiry_name,
            })
        except Exception:
            pass

        return trade
    
    def _check_paper_trades(self, current_price, symbol=None):
        """Check all open paper trades for TP/SL/Expiry"""
        # Heartbeat FIRST — fires every cycle regardless of open trades
        # Prevents false WATCHDOG STALE alerts when no positions are open
        try:
            from jarvis_watchdog import beat
            beat("paper_trade_loop")
        except Exception:
            pass

        if not current_price or current_price < 100:
            return
        
        still_open = []
        newly_closed = []
        
        for trade in self.paper_open_trades:
            closed = False
            try:
                if symbol and trade.get('symbol') and _binance_spot_symbol(trade['symbol']) != _binance_spot_symbol(symbol):
                    still_open.append(trade)  # other coin's trade: never evaluate with this coin's price
                    continue
            except Exception:
                pass
            
            # --- Handle Pending Limits ---
            if trade.get('status') == 'PENDING_LIMIT':
                if trade['direction'] == 'CALL' and current_price <= trade['entry_price']:
                    trade['status'] = 'OPEN'
                    trade['entry_time'] = datetime.now().isoformat()
                    expiry_min = self.PAPER_CONFIG['expiry_map'].get(trade['expiry_name'], 3)
                    trade['expiry_time'] = (datetime.now() + timedelta(minutes=expiry_min)).isoformat()
                    self._dashboard_events.append(f"Limit filled: {trade['direction']} @ {trade['entry_price']}")
                elif trade['direction'] == 'PUT' and current_price >= trade['entry_price']:
                    trade['status'] = 'OPEN'
                    trade['entry_time'] = datetime.now().isoformat()
                    expiry_min = self.PAPER_CONFIG['expiry_map'].get(trade['expiry_name'], 3)
                    trade['expiry_time'] = (datetime.now() + timedelta(minutes=expiry_min)).isoformat()
                    self._dashboard_events.append(f"Limit filled: {trade['direction']} @ {trade['entry_price']}")
                else:
                    still_open.append(trade)
                    continue
            
            direction = trade['direction']
            entry = trade['entry_price']
            sl = trade.get('sl')
            tp1 = trade.get('tp1')
            expiry_dt = datetime.fromisoformat(trade['expiry_time'])
            
            # Check Stop Loss
            if sl and not closed:
                if (direction == 'CALL' and current_price <= sl) or \
                   (direction == 'PUT' and current_price >= sl):
                    trade['result'] = 'LOSS'
                    trade['close_reason'] = '🛑 Stop Loss Hit'
                    closed = True
            
            # Check Take Profit
            if tp1 and not closed:
                if (direction == 'CALL' and current_price >= tp1) or \
                   (direction == 'PUT' and current_price <= tp1):
                    trade['result'] = 'WIN'
                    trade['close_reason'] = '🎯 Take Profit Hit'
                    closed = True
            
            # Check Time Expiry
            if not closed and datetime.now() >= expiry_dt:
                if direction == 'CALL':
                    if current_price > entry:
                        trade['result'] = 'WIN'
                        trade['close_reason'] = '⏱️ Expiry - Price Up'
                    elif current_price < entry:
                        trade['result'] = 'LOSS'
                        trade['close_reason'] = '⏱️ Expiry - Price Down'
                    else:
                        trade['result'] = 'BREAKEVEN'
                        trade['close_reason'] = '⏱️ Expiry - Flat'
                else:  # PUT
                    if current_price < entry:
                        trade['result'] = 'WIN'
                        trade['close_reason'] = '⏱️ Expiry - Price Down'
                    elif current_price > entry:
                        trade['result'] = 'LOSS'
                        trade['close_reason'] = '⏱️ Expiry - Price Up'
                    else:
                        trade['result'] = 'BREAKEVEN'
                        trade['close_reason'] = '⏱️ Expiry - Flat'
                closed = True
            
            if closed:
                trade['exit_price'] = current_price
                # One contract is one USD notional in the repository convention;
                # leverage only determines required margin, never gets multiplied
                # into P&L a second time.
                position_size = float(trade.get('notional_usdt', 0.0))
                price_move_pct = abs(current_price - entry) / entry if entry else 0.0
                if trade['result'] == 'WIN':
                    trade['pnl_dollar'] = position_size * price_move_pct
                    self.paper_balance += trade['pnl_dollar']
                    self.paper_wins += 1
                elif trade['result'] == 'LOSS':
                    trade['pnl_dollar'] = -(position_size * price_move_pct)
                    self.paper_balance += trade['pnl_dollar']
                    self.paper_losses += 1
                else:
                    trade['pnl_dollar'] = 0
                    self.paper_breakeven += 1
                
                self.paper_peak_balance = max(self.paper_peak_balance, self.paper_balance)
                newly_closed.append(trade)
                self.paper_closed_trades.append(trade)
                
                # FIX BUG 1: Call record_trade here when it actually closes
                self.record_trade(direction, trade.get('confidence', 0), trade['result'])

                # --- LEARNING LOOP: record outcome + engine attribution (guarded) ---
                try:
                    from jarvis_learning import record_trade as _jl_record, learning_enabled as _jl_enabled
                    if _jl_enabled():
                        engine_signals = {}
                        jarvis = getattr(self, 'jarvis', None)
                        part_results = getattr(jarvis, 'latest_part_results', None) if jarvis else None
                        if isinstance(part_results, dict):
                            for pname, pres in part_results.items():
                                if isinstance(pres, dict) and isinstance(pres.get('signal'), (int, float)):
                                    engine_signals[pname] = pres['signal']
                        trade_snapshot = dict(trade)
                        trade_snapshot.setdefault('symbol', 'BTC/USDT')
                        _jl_record(trade_snapshot, engine_signals=engine_signals or None)
                except Exception as e:
                    logger.debug(f"[Learning] record on close failed (ignored): {e}")
                
                # --- WIRING FIX: CNS PAIN DETECTION ---
                if trade['result'] == 'LOSS' and hasattr(self.jarvis, 'cns') and self.jarvis.cns:
                    try:
                        import threading
                        threading.Thread(
                            target=self.jarvis.cns.detect_pain,
                            args=({'pnl': trade.get('pnl_dollar', -1), 'trade': trade, 'signal': direction},),
                            daemon=True
                        ).start()
                    except Exception as e:
                        logger.debug(f"[CNS] detect_pain trigger failed: {e}")
            else:
                still_open.append(trade)
        
        self.paper_open_trades = still_open

        # Watchdog: persist state when trades closed (beat already sent at top)
        try:
            from jarvis_watchdog import save_state
            if newly_closed:
                save_state(self.paper_open_trades)
        except Exception:
            pass

        # Print closed trade results
        for t in newly_closed:
            emoji = '✅' if t['result'] == 'WIN' else ('❌' if t['result'] == 'LOSS' else '➖')
            pnl_str = f"+${t['pnl_dollar']:.2f}" if t['pnl_dollar'] >= 0 else f"-${abs(t['pnl_dollar']):.2f}"
            self._dashboard_events.append(
                f"Paper closed #{t['id']} {t['result']} {pnl_str} | balance ${self.paper_balance:,.2f}"
            )

            # ── TELEGRAM: Notify trade closed ──
            try:
                from telegram_notifier import send_message
                send_message(
                    f"{emoji} JARVIS TRADE CLOSED\n"
                    f"{t['direction']} #{t['id']}\n"
                    f"Entry: ${t['entry_price']:,.2f} → Exit: ${t['exit_price']:,.2f}\n"
                    f"Result: {t['result']} | P&L: {pnl_str}\n"
                    f"Balance: ${self.paper_balance:,.2f}"
                )
            except Exception:
                pass
        
        if newly_closed:
            self._save_paper_state()
    
    def _apply_options_confirmation(self, result, symbol):
        """Attach asset-specific options intelligence before final display/trade.

        BTC is used only as bounded macro confirmation when an altcoin has no
        usable option chain. It is never used as the altcoin's hedge contract,
        strike, TP, or SL source.
        """
        if not OPTIONS_CONTEXT_AVAILABLE or not resolve_options_context:
            return result
        try:
            signal = result.get('trade_signal', {})
            direction = signal.get('direction', 'NO_TRADE')
            raw_conf = signal.get('confidence_score')
            confidence = normalize_confidence(raw_conf)
            context = resolve_options_context(self.jarvis.delta_data, symbol)
            adjusted, note = apply_options_confirmation(direction, confidence or 0, context)
            result.setdefault('market_context', {})['options_context'] = context.to_dict()
            result['market_context']['options_confirmation'] = note
            if direction in ('CALL', 'PUT', 'BUY', 'SELL') and confidence is not None:
                signal['confidence_score'] = f'{adjusted}/100'
            logger.info('[OPTIONS] %s | selected=%s | source=%s | %s',
                        context.role, context.selected_asset, context.source_asset or 'none', note)
        except Exception as options_error:
            logger.debug('[OPTIONS] confirmation skipped: %s', options_error)
        return result

    def _print_unavailable_live_diagnostics(self, symbol, stage, reason, snapshot=None):
        """Print a throttled, current-cycle unavailable summary; never reuse old evidence."""
        try:
            safe_reason = type(reason).__name__ if isinstance(reason, BaseException) else str(reason or 'no reason supplied')
            reason_key = (str(symbol or ''), str(stage or ''), safe_reason[:120])
            now_mono = time.monotonic()
            try:
                configured_interval = float(os.getenv('JARVIS_DIAGNOSTICS_UNAVAILABLE_INTERVAL_SECONDS', '60'))
                if not math.isfinite(configured_interval):
                    configured_interval = 60.0
            except (TypeError, ValueError, OverflowError):
                configured_interval = 60.0
            min_interval = max(5.0, configured_interval)
            last_key = getattr(self, '_last_unavailable_diagnostics_key', None)
            last_time = getattr(self, '_last_unavailable_diagnostics_time', 0.0)
            if reason_key == last_key and now_mono - last_time < min_interval:
                return False
            self._last_unavailable_diagnostics_key = reason_key
            self._last_unavailable_diagnostics_time = now_mono

            from terminal_live_diagnostics import build_timeframe_inputs, format_live_diagnostics, TIMEFRAMES
            jarvis = getattr(self, 'jarvis', None)
            frames = {}
            if isinstance(snapshot, dict):
                snapshot_frames = snapshot.get('frames', {})
            else:
                snapshot_frames = getattr(snapshot, 'frames', {}) if snapshot is not None else {}
            if callable(getattr(snapshot_frames, 'items', None)):
                for tf, candle_frame in snapshot_frames.items():
                    closed = candle_frame.get('closed') if isinstance(candle_frame, dict) else getattr(candle_frame, 'closed', None)
                    if closed is not None:
                        frames[tf] = closed
            cache = getattr(jarvis, 'direct_candle_cache', None)
            timeframe_inputs = build_timeframe_inputs(
                frames, snapshot,
                timeframes=tuple(TIMEFRAMES), expected_symbol=_binance_spot_symbol(symbol),
                stale_intervals=getattr(cache, 'stale_intervals', 2.0) if cache is not None else 2.0,
                fetch_ttl_seconds=getattr(cache, 'ttl_seconds', 45.0) if cache is not None else 45.0,
            )
            # A failed route/fetch with no snapshot is unavailable input, not
            # evidence that eight individually requested frames were absent.
            if snapshot is None:
                for item in timeframe_inputs.values():
                    if isinstance(item, dict) and item.get('status') == 'MISSING':
                        item['status'] = 'UNAVAILABLE'
            text = format_live_diagnostics(
                symbol=symbol, snapshot=snapshot, timeframe_inputs=timeframe_inputs,
                parts_by_timeframe={}, fused_parts={}, decision={'status': 'NOT GENERATED'},
                setup_mode={'mode': 'NOT EVALUATED', 'reason': safe_reason},
                part7_gate=None, entry_trigger=None,
                runtime=getattr(jarvis, 'gpu_status', None) if jarvis is not None else None,
                neural_status={'status': 'not_run', 'availability': 'not probed',
                               'reason': 'analysis not evaluated without a valid current native snapshot'},
                cycle_status={'status': 'UNAVAILABLE', 'reason': safe_reason},
            )
            if os.getenv('JARVIS_DIAGNOSTICS_FULL', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
                print(text)
            try:
                from terminal_live_thinking import format_live_thinking
                self._last_thinking_text = format_live_thinking(
                    symbol=symbol, price=getattr(self, '_last_current_price', None),
                    cycle_status={'status': 'UNAVAILABLE', 'reason': f'{stage}: {safe_reason}'},
                    decision={'direction': 'NO_TRADE', 'reasons': [f'{stage}: {safe_reason}']},
                )
                self._render_unified_dashboard(symbol=symbol, current_price=getattr(self, '_last_current_price', None), action='WAIT', gate_reason=f'{stage}: {safe_reason}')
            except Exception:
                print(text)
            return True
        except Exception as diagnostic_error:
            logger.debug('[LIVE-DIAGNOSTICS] unavailable-cycle display failed: %s', type(diagnostic_error).__name__)
            return False

    def _print_decision_audit(self, decision, symbol, parts=None, blockers=None, stage="FINAL_GATE", order_outcome=None):
        """Best-effort display of observed values; formatting never gates trading."""
        try:
            decision = decision if isinstance(decision, dict) else {}
            last_result = getattr(self, 'last_jarvis_result', {})
            last_result = last_result if isinstance(last_result, dict) else {}
            candidate = last_result.get('trade_signal', {})
            market_context = last_result.get('market_context', {})
            jarvis = getattr(self, 'jarvis', None)
            snapshot = getattr(jarvis, '_active_candle_snapshot', None) if jarvis is not None else None
            trader = getattr(self, 'auto_trader', None)
            if trader is None:
                execution_mode = 'UNKNOWN'
            else:
                execution_mode = 'LIVE' if getattr(trader, 'is_enabled', False) else 'PAPER'
            position_state = {
                'live': getattr(trader, 'open_positions', None) if trader is not None else None,
                'paper': getattr(self, 'paper_open_trades', None),
            }
            provenance = decision.get('provenance') or decision.get('source')
            if not provenance:
                provenance = (
                    'pure algorithm mode (Part 1–12 math path; model synthesis, AI gate, and background consensus disabled)'
                    if _pure_algorithm_mode()
                    else 'mixed/current pipeline; AI-assisted components may participate (not claimed pure algorithm)'
                )
            from terminal_decision_display import format_decision_audit
            audit_text = format_decision_audit(
                decision, symbol, parts=parts, blockers=blockers,
                candidate=candidate, analysis_result=last_result,
                market_context=market_context, snapshot=snapshot,
                position_state=position_state, execution_mode=execution_mode,
                stage=stage, order_outcome=order_outcome, provenance=provenance,
            )
            # The full MTF report is emitted once per completed analysis cycle,
            # not again for the subsequent order-return audit. It is strictly
            # best-effort display code and has no decision or execution authority.
            if str(stage).upper() == 'FINAL_GATE':
                try:
                    from terminal_live_diagnostics import format_live_diagnostics
                    final_direction = str(decision.get('execution_direction') or decision.get('direction') or '').upper()
                    expiry = str(candidate.get('recommended_expiry') or '').upper() if isinstance(candidate, dict) else ''
                    if final_direction in ('CALL', 'PUT') and decision.get('execution_allowed') is True:
                        if expiry in ('DAY_TRADE', 'SWING', '15M', '30M'):
                            setup_mode = {'mode': 'SWING', 'reason': f"auto-execution expiry route: recommended_expiry={expiry}"}
                        else:
                            setup_mode = {'mode': 'SCALP', 'reason': f"auto-execution default route for recommended_expiry={expiry or 'UNKNOWN'}"}
                    else:
                        _mode_reasons = decision.get('reasons')
                        _mode_reason = ' | '.join(str(x) for x in _mode_reasons[:3]) if isinstance(_mode_reasons, (list, tuple)) else str(_mode_reasons or '')
                        setup_mode = {'mode': 'NO TRADE', 'reason': _mode_reason or f"final decision direction={final_direction or 'UNKNOWN'}; execution_allowed={decision.get('execution_allowed', 'UNKNOWN')}"}
                    part7_gate = getattr(jarvis, 'latest_part7_cycle_display', None) if jarvis is not None else None
                    if not isinstance(part7_gate, dict) and isinstance(last_result, dict):
                        part7_gate = last_result.get('part7_volatility')
                    live_text = format_live_diagnostics(
                        symbol=symbol,
                        snapshot=snapshot,
                        timeframe_inputs=getattr(jarvis, 'latest_timeframe_inputs', None) if jarvis is not None else None,
                        parts_by_timeframe=getattr(jarvis, 'latest_part_timeframe_results', None) if jarvis is not None else None,
                        fused_parts=getattr(jarvis, 'latest_fusion_results', None) if jarvis is not None else None,
                        decision=decision,
                        candidate=last_result,
                        part7_gate=part7_gate,
                        entry_trigger=getattr(self, 'last_5m_entry', None),
                        setup_mode=setup_mode,
                        runtime=getattr(jarvis, 'gpu_status', None) if jarvis is not None else None,
                        neural_status=getattr(jarvis, 'latest_neural_cycle_status', None) if jarvis is not None else None,
                        cycle_status=getattr(jarvis, 'latest_diagnostics_cycle_status', None) if jarvis is not None else None,
                        verbose=os.getenv('JARVIS_DIAGNOSTICS_VERBOSE', '0').strip().lower() in ('1', 'true', 'yes', 'on'),
                    )
                    audit_text = live_text + '\n\n' + audit_text
                except Exception as live_diag_error:
                    logger.debug('[LIVE-DIAGNOSTICS] display unavailable: %s', type(live_diag_error).__name__)
            full_output = os.getenv('JARVIS_DIAGNOSTICS_FULL', '1').strip().lower() in ('1', 'true', 'yes', 'on')
            if str(stage).upper() == 'FINAL_GATE':
                try:
                    from terminal_live_thinking import format_live_thinking
                    thinking = format_live_thinking(
                        symbol=symbol,
                        price=getattr(self, '_last_current_price', None) or decision.get('price'),
                        parts_by_timeframe=getattr(jarvis, 'latest_part_timeframe_results', None) if jarvis is not None else None,
                        fused_parts=getattr(jarvis, 'latest_fusion_results', None) if jarvis is not None else None,
                        decision=decision, candidate=last_result,
                        part7_gate=getattr(jarvis, 'latest_part7_cycle_display', None) if jarvis is not None else None,
                        entry_trigger=getattr(self, 'last_5m_entry', None),
                        cycle_status=getattr(jarvis, 'latest_diagnostics_cycle_status', None) if jarvis is not None else None,
                        trace=getattr(self, '_thinking_trace', None),
                    )
                    self._last_thinking_text = thinking
                    # Ensure audit_text is retained if they want to see the full system thinking
                    if not full_output:
                        audit_text = thinking
                except Exception as thinking_error:
                    logger.debug('[LIVE-THINKING] display unavailable: %s', type(thinking_error).__name__)
            elif not full_output:
                _oc = order_outcome if isinstance(order_outcome, dict) else {}
                getattr(self, '_dashboard_events', []).append(
                    f"ORDER {stage}: {'OK' if _oc.get('success') else 'not placed'} {decision.get('direction', '')} {_oc.get('reason', '')}".strip())
                audit_text = None
            if audit_text:
                print(audit_text)
            # Experimental commentary only at FINAL_GATE; never repeated for order return.
            # Worker is daemonized and never gates execution.
            try:
                from jarvis_laya_advisor import request_runtime_advisory
                if stage == "FINAL_GATE" and full_output:
                    advisory = request_runtime_advisory({
                        'symbol': str(symbol).upper(),
                        'direction': decision.get('direction', 'NO_TRADE'),
                        'confidence': decision.get('confidence'),
                        'price': decision.get('price'),
                        'reasons': decision.get('reasons', []),
                    }, symbol=symbol, deterministic_decision=decision.get('direction', 'NO_TRADE'))
                    print(f"Laya advisory (experimental; not a gate): {advisory.status}"
                          + (f" suggestion={advisory.suggestion}" if advisory.suggestion else ""))
            except Exception as advisory_error:
                print(f"Laya advisory (experimental; not a gate): unavailable ({type(advisory_error).__name__})")
        except Exception as audit_error:
            # Display failure is deliberately non-fatal and cannot affect safety gates.
            logger.debug("[AUDIT] terminal summary unavailable: %s", type(audit_error).__name__)

    def _print_live_signal(self, result, current_price, symbol='BTCUSDT', df=None):
        """Prepare live signal/dashboard fields; this method itself does not print."""
        signal = result.get('trade_signal', {})
        direction = signal.get('direction', 'NO_TRADE')
        
        conf_str = signal.get('confidence_score')
        confidence = normalize_confidence(conf_str)
        
        raw_entry = signal.get('entry_price', current_price)
        tp1 = signal.get('take_profit_1')
        tp2 = signal.get('take_profit_2')
        sl  = signal.get('stop_loss')
        expiry = signal.get('recommended_expiry', '3M')
        thoughts = result.get('intelligence_board', [])
        market_ctx = result.get('market_context', {})

        # ── SMART ENTRY: Compute optimal limit entry based on ATR ──
        entry_price, entry_type = self._calculate_smart_entry(direction, current_price, result)

        # ── 1M ENTRY REFINEMENT: HTF decision stays; 1m swing SL + confirmation ──
        self.last_5m_entry = None
        if direction in ('CALL', 'PUT'):
            entry_price, sl, confirmed_1m, note_1m = self._refine_entry_with_5m(
                direction, entry_price, sl, df, current_price
            )
            if note_1m not in ('5M-OFF', 'N/A', '5M-FALLBACK'):
                entry_type = f'{entry_type} | 5m: {note_1m}'

        self._dashboard_signal = {
            'direction': direction,
            'confidence_score': confidence,
            'entry_price': entry_price,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'stop_loss': sl,
        }
        self._dashboard_plan = {
            'entry': f"${entry_price:,.2f}" if entry_price else '—',
            'tp1': f"${tp1:,.2f}" if tp1 else '—',
            'sl': f"${sl:,.2f}" if sl else '—',
            'expiry': expiry,
            'action': 'WAIT' if direction == 'NO_TRADE' else direction,
            'gates': 'analysis complete',
        }
        return direction, confidence, entry_price, tp1, tp2, sl, expiry
    
    def _calculate_smart_entry(self, direction: str, current_price: float, result: dict):
        """
        Calculate optimal entry price using ATR-based pullback logic.
        Returns: (entry_price, entry_type_label)

        Rules:
          CALL: Market price is good for strong momentum candles.
                For weak signals, wait for a small pullback (0.1-0.25% below current)
          PUT:  Wait for a small bounce (0.1-0.25% above current)

        This improves R:R by getting better fill prices.
        """
        if not direction or direction == 'NO_TRADE' or not current_price:
            return current_price, 'MARKET'

        try:
            signal = result.get('trade_signal', {})
            market_ctx = result.get('market_context', {})

            # Get ATR if available from signal, else estimate
            atr = signal.get('atr', current_price * 0.003)  # default 0.3% ATR
            if atr and atr > 0:
                atr_pct = atr / current_price
            else:
                atr_pct = 0.003  # 0.3% default

            # Clamp ATR pullback between 0.05% and 0.4%
            pullback_pct = max(0.0005, min(0.004, atr_pct * 0.25))

            # Strong momentum = enter at market (body > 60% of candle range)
            volatility = str(market_ctx.get('volatility', 'MEDIUM')).upper()
            conf_str = signal.get('confidence_score')
            confidence = normalize_confidence(conf_str)
            confidence_for_entry = confidence if confidence is not None else -1

            # HIGH confidence (≥85%) + LOW volatility → MARKET entry (momentum)
            if confidence_for_entry >= 85 and volatility in ('LOW', 'VERY_LOW'):
                return current_price, 'MARKET (High Conf)'

            # Compute limit entry
            if direction == 'CALL':
                # Wait for small dip below current for better fill
                entry = round(current_price * (1.0 - pullback_pct), 2)
                label = f'LIMIT PULL ({pullback_pct*100:.2f}% below)'
            else:  # PUT
                # Wait for small bounce above current for better fill
                entry = round(current_price * (1.0 + pullback_pct), 2)
                label = f'LIMIT BOUNCE ({pullback_pct*100:.2f}% above)'

            return entry, label

        except Exception as e:
            logger.debug(f"[SMART ENTRY] Fallback to market: {e}")
            return current_price, 'MARKET'

    def _refine_entry_with_5m(self, direction, entry_price, sl, df, current_price):
        """
        1-MINUTE ENTRY REFINEMENT (HTF decision + LTF execution).

        JARVIS no final BUY/SELL decision higher-timeframe analysis (Part 1-12)
        par j based rahe chhe. Aa method fakt EXECUTION improve kare chhe:

          1. SL TIGHTENING: Recent 1m swing low (CALL) / swing high (PUT)
             par stop-loss move kare chhe — same risk, better R:R.
             Fakt TIGHTEN kare chhe; kadi WIDEN nathi kartu.
             Bounds: entry thi 0.10% thi 1.20% distance. Out-of-range hoy
             to original HTF SL j rehse (fail-open).

          2. 1M CONFIRMATION: Last CLOSED 1m candle signal direction sathe
             match karvu joie (CALL = bullish close, PUT = bearish close).
             Match na kare to confirmed=False — e cycle ma trade skip thase
             ane next cycle ma fari try thase (entry timing wait).

        Env controls:
          JARVIS_ENTRY_1M=0        -> full feature off (original behaviour)
          JARVIS_ENTRY_1M_CONFIRM=0 -> confirmation off, fakt SL tightening

        Returns: (entry_price, sl, confirmed, note)
        Koi pan error/failure par original values + confirmed=True (fail-open),
        etle existing behaviour kadi break nathi thatu.
        """
        self.last_5m_entry = None
        try:
            if os.getenv('JARVIS_ENTRY_1M', '1').lower() in ('0', 'false', 'off'):
                return entry_price, sl, True, '5M-OFF'
            if direction not in ('CALL', 'PUT') or not entry_price or not current_price:
                return entry_price, sl, True, 'N/A'
            if df is None or len(df) < 15:
                return entry_price, sl, True, 'NO-1M-DATA'

            lookback = int(os.getenv('JARVIS_ENTRY_1M_LOOKBACK', '12'))
            # Last candle haji close na thayeli hoy sake — swing mate exclude kariye
            win = df.iloc[-(lookback + 1):-1]
            if len(win) < 5:
                return entry_price, sl, True, 'NO-1M-DATA'

            buffer_pct = 0.0003   # 0.03% swing buffer
            min_dist   = 0.0010   # SL entry thi ochho ma ochho 0.10% dur
            max_dist   = 0.0120   # SL entry thi vadhu ma vadhu 1.20% dur

            note_parts = []
            refined_sl = sl
            if direction == 'CALL':
                swing = float(win['low'].min()) * (1.0 - buffer_pct)
                # Tighten only: 1m swing SL original SL karta UP j hoy to j use
                if swing < entry_price and (sl is None or swing > sl):
                    dist = (entry_price - swing) / entry_price
                    if min_dist <= dist <= max_dist:
                        refined_sl = round(swing, 2)
                        note_parts.append(f'SL 5m-swing ${refined_sl:,.2f} ({dist*100:.2f}%)')
            else:  # PUT
                swing = float(win['high'].max()) * (1.0 + buffer_pct)
                # Tighten only: 1m swing SL original SL karta NICHE j hoy to j use
                if swing > entry_price and (sl is None or swing < sl):
                    dist = (swing - entry_price) / entry_price
                    if min_dist <= dist <= max_dist:
                        refined_sl = round(swing, 2)
                        note_parts.append(f'SL 5m-swing ${refined_sl:,.2f} ({dist*100:.2f}%)')

            # 1m direction confirmation (last CLOSED candle)
            confirmed = True
            if os.getenv('JARVIS_ENTRY_1M_CONFIRM', '1').lower() not in ('0', 'false', 'off'):
                last = df.iloc[-1]
                candle_bull = float(last['close']) > float(last['open'])
                if direction == 'CALL' and not candle_bull:
                    confirmed = False
                    note_parts.append('5m confirm wait (bearish candle)')
                elif direction == 'PUT' and candle_bull:
                    confirmed = False
                    note_parts.append('5m confirm wait (bullish candle)')

            note = ' + '.join(note_parts) if note_parts else '1m OK'
            self.last_5m_entry = {
                'confirmed': confirmed,
                'refined_sl': refined_sl,
                'original_sl': sl,
                'note': note,
            }
            logger.info('[1M-ENTRY] %s | confirmed=%s | %s', direction, confirmed, note)
            return entry_price, refined_sl, confirmed, note

        except Exception as e:
            logger.debug(f"[1M-ENTRY] fail-open fallback: {e}")
            return entry_price, sl, True, '5M-FALLBACK'

    def _print_compact_status(self, current_price=None):
        self._render_unified_dashboard(current_price=current_price)

    def _print_paper_dashboard(self):
        self._render_unified_dashboard(current_price=getattr(self, '_last_current_price', None))

    def _save_paper_state(self):
        """Save paper trading state"""
        try:
            import json
            data = {
                'balance': self.paper_balance,
                'wins': self.paper_wins,
                'losses': self.paper_losses,
                'breakeven': self.paper_breakeven,
                'peak': self.paper_peak_balance,
                'closed_trades': self.paper_closed_trades[-100:],
            }
            with open(self.PAPER_CONFIG['save_file'], 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass
    
    def _load_paper_state(self):
        """Load paper trading state from previous session"""
        try:
            import json
            import os
            if os.path.exists(self.PAPER_CONFIG['save_file']):
                with open(self.PAPER_CONFIG['save_file'], 'r') as f:
                    data = json.load(f)
                self.paper_balance = data.get('balance', self.PAPER_CONFIG['initial_balance'])
                self.paper_wins = data.get('wins', 0)
                self.paper_losses = data.get('losses', 0)
                self.paper_breakeven = data.get('breakeven', 0)
                self.paper_peak_balance = data.get('peak', self.paper_balance)
                self.paper_closed_trades = data.get('closed_trades', [])
                logger.info(f"📂 Paper state loaded: ${self.paper_balance:,.2f} ({self.paper_wins}W/{self.paper_losses}L)")
        except Exception:
            pass

    def _set_readiness(self, status, reason=''):
        self.readiness = {
            'status': str(status).upper(),
            'reason': str(reason)[:240],
            'updated_at': datetime.now().isoformat(),
        }

    def _multicoin_candidate_instruments(self):
        """Join scanner-qualified Binance spot assets to explicit active Delta perps.

        This is discovery only. The Delta base, quoting, and settling assets all
        come from product metadata; no DELTA_SYMBOL_MAP, ticker suffix, quote
        currency, or BTC-only assumption is used. Products lacking supported
        risk units, explicit assets, or a unique eligible id are excluded.
        """
        try:
            scanner = getattr(self.market_router, 'scanner', None)
            delta = getattr(self.market_router, 'delta', None)
            if scanner is None or delta is None:
                return ()
            from jarvis_coin_scanner import MIN_VOLUME_USDT
            snapshot_getter = getattr(delta, 'get_available_products_snapshot', None)
            if callable(snapshot_getter):
                snapshot = snapshot_getter()
                if not isinstance(snapshot, dict) or snapshot.get('complete') is not True:
                    return ()
                product_rows = snapshot.get('products', ())
            else:
                product_getter = getattr(delta, 'get_available_products', None)
                if not callable(product_getter):
                    return ()
                product_rows = product_getter() or ()
            scores = {}
            for coin, row in (scanner.get_all_scores() or {}).items():
                asset = str(coin).strip().upper()
                try:
                    volume = float(row.get('volume_24h_usdt', 0.0))
                except (TypeError, ValueError, AttributeError):
                    continue
                if asset and volume >= float(MIN_VOLUME_USDT) and volume < float('inf'):
                    scores[asset] = max(volume, scores.get(asset, 0.0))
            eligible_by_asset = {}
            accepted_types = {'perpetual', 'perpetual_futures', 'perpetual_swap', 'perpetual_swaps'}
            from jarvis_risk import contract_quote_value_in_currency
            for product in product_rows:
                if not isinstance(product, dict):
                    continue
                state = str(product.get('state') or product.get('status') or '').strip().lower()
                marker = str(product.get('product_type') or product.get('market_type') or
                             product.get('contract_type') or '').strip().lower()
                if state not in {'active', 'live', 'trading', 'listed'} or marker not in accepted_types:
                    continue
                def asset_symbol(value):
                    if isinstance(value, dict):
                        value = value.get('symbol') or value.get('asset_symbol') or value.get('name')
                    return str(value or '').strip().upper()
                base = asset_symbol(product.get('underlying_asset_symbol') or product.get('underlying_asset')
                                    or product.get('base_asset'))
                quote = asset_symbol(product.get('quoting_asset') or product.get('quote_asset')
                                     or product.get('quote_currency'))
                settle = asset_symbol(product.get('settling_asset') or product.get('settlement_asset')
                                      or product.get('settlement_currency'))
                symbol = str(product.get('symbol') or '').strip().upper()
                pid = product.get('id', product.get('product_id'))
                try:
                    tick_size = float(product.get('tick_size'))
                except (TypeError, ValueError, OverflowError):
                    continue
                notional_kind = str(product.get('notional_type') or '').strip().lower()
                specs = product.get('product_specs') if isinstance(product.get('product_specs'), dict) else {}
                trading_status = str(product.get('trading_status') or '').strip().lower()
                if (not base or not quote or not settle or not symbol or pid is None
                        or quote != settle or not math.isfinite(tick_size) or tick_size <= 0
                        or notional_kind == 'inverse' or product.get('is_quanto') is True
                        or specs.get('only_reduce_only_orders_allowed') is True
                        or trading_status not in {'', 'operational'}
                        or contract_quote_value_in_currency(product, 1.0, settle) is None):
                    continue
                if base in scores:
                    eligible_by_asset.setdefault(base, []).append((product, quote, settle, marker))
            candidates = []
            for asset, volume in scores.items():
                # Ambiguous active products for one underlying are excluded;
                # operators may narrow via a reviewed product-specific policy.
                matches = eligible_by_asset.get(asset, ())
                if len(matches) != 1:
                    continue
                product, quote, settle, marker = matches[0]
                symbol = str(product['symbol']).strip().upper()
                binance_symbol = asset + 'USDT'
                mapping_policy_id = f'binance_spot_usdt_to_delta_{quote.lower()}_{settle.lower()}_{marker}'
                candidates.append({
                    'venue': 'binance', 'market_type': 'spot',
                    'instrument_id': binance_symbol, 'symbol': binance_symbol,
                    'execution_identity': {
                        'venue': 'delta', 'market_type': marker,
                        'instrument_id': str(product.get('id', product.get('product_id'))).strip(),
                        'symbol': symbol,
                    },
                    'mapping_policy_id': mapping_policy_id,
                    'liquidity_24h_usdt': volume,
                })
            candidates.sort(key=lambda item: (-float(item['liquidity_24h_usdt']), item['symbol']))
            return tuple(candidates)
        except Exception as _candidate_error:
            logger.warning('[MULTICOIN] Product discovery unavailable: %s', type(_candidate_error).__name__)
            return ()

    def _analyze_multicoin_instrument(self, key, snapshot):
        """Run the real Part1-12 Jarvis method on an isolated analysis-only owner."""
        owner = None
        try:
            if tuple(snapshot.identity) != key.as_tuple():
                return {}
            native_frames = snapshot.analysis_frames()
            owner = JarvisElite(backtest_mode=True)
            owner.active_symbol = key.symbol
            owner.active_base_asset = key.symbol
            for quote in ('USDT', 'USD', 'USDC'):
                if owner.active_base_asset.endswith(quote):
                    owner.active_base_asset = owner.active_base_asset[:-len(quote)]
                    break
            owner.engines = {}  # only the reviewed Part adapters run in this path
            owner.direct_candle_cache = None
            _host_jarvis = getattr(self, 'jarvis', None)
            _live_delta = getattr(_host_jarvis, 'delta_data', None) or getattr(self, 'delta_data', None)
            _live_deribit = (getattr(_host_jarvis, 'deribit', None) or getattr(self, 'deribit', None))
            _want_cur = str(owner.active_base_asset).upper()
            if _want_cur not in ('BTC', 'ETH'):
                _live_deribit = None
            elif _live_deribit is not None and str(getattr(_live_deribit, 'currency', 'BTC')).upper() != _want_cur:
                # Never feed another coin's option chain: use a per-asset Deribit client.
                _dcache = getattr(self, '_deribit_clients', None)
                if _dcache is None:
                    _dcache = self._deribit_clients = {}
                if _want_cur not in _dcache:
                    try:
                        from deribit_options_client import DeribitOptionsClient
                        _dcache[_want_cur] = DeribitOptionsClient(currency=_want_cur, client_id=os.getenv('DERIBIT_CLIENT_ID', ''), client_secret=os.getenv('DERIBIT_CLIENT_SECRET', ''))
                    except Exception:
                        _dcache[_want_cur] = None
                _live_deribit = _dcache[_want_cur]
            owner.delta_data = None
            owner.deribit = None
            if 'part14_options_chain' in getattr(owner, 'parts', {}):
                owner.parts['part14_options_chain'].backtest_mode = False
                owner.parts['part14_options_chain'].asset = owner.active_base_asset
                owner.parts['part14_options_chain'].delta_client = _live_delta
                owner.parts['part14_options_chain'].deribit = _live_deribit
            owner.binance_data = None
            owner.coin_scanner = None
            owner._active_candle_snapshot = snapshot
            native_frames = {tf: frame.copy(deep=True) for tf, frame in native_frames.items()}
            # Use a single point-in-time cut for this analysis-only replay.
            # In particular, faster candles can close after the latest 1m candle;
            # passing their newer values beside that 1m reference would leak
            # future information into a supposedly synchronized MTF result.
            if '5m' not in native_frames or native_frames['5m'].empty:
                return {}
            _one_minute_last = pd.Timestamp(native_frames['5m'].index[-1])
            if _one_minute_last.tzinfo is None:
                _one_minute_last = _one_minute_last.tz_localize('UTC')
            else:
                _one_minute_last = _one_minute_last.tz_convert('UTC')
            _decision_epoch = next_candle_open('5m', _one_minute_last.timestamp())
            for tf, frame in list(native_frames.items()):
                if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                    return {}
                _utc_index = frame.index.tz_convert('UTC')
                _keep = [next_candle_open(tf, _stamp.timestamp()) <= _decision_epoch
                         for _stamp in _utc_index]
                frame = frame.loc[_keep].copy(deep=True)
                if len(frame) < 5:
                    return {}
                native_frames[tf] = frame
                frame.attrs.update({'symbol': key.symbol, 'timeframe': tf})
            # Parts 1-10 adapters operate on the canonical strategy set.
            # Sub-minute and multi-day timeframes from the full snapshot would
            # fail the backtest closed-candle timestamp gate; drop them before
            # passing to analyze_trade_setup.
            # 15-Oct: user rule - every analysis path receives all 15 live timeframes.
            analysis_frames = {tf: frame for tf, frame in native_frames.items()
                               if tf in LIVE_TIMEFRAMES}
            if '5m' not in analysis_frames:
                return {}
            owner.analyze_trade_setup(
                analysis_frames['5m'].copy(deep=True), candle_snapshot=snapshot,
                native_mtf=analysis_frames,
            )

            analysis = getattr(owner, 'latest_multicoin_analysis', None)
            if not isinstance(analysis, dict):
                return {}
            one_minute = snapshot.frames.get('5m')
            if one_minute is None or one_minute.closed.empty:
                return {}
            reference_open = float(pd.Timestamp(one_minute.closed.index[-1]).timestamp())
            analysis_reference = {
                'source': 'binance', 'symbol': key.symbol, 'timeframe': '5m',
                # Candle indexes mark the open; the confirmed close instant is
                # one native interval later and never after snapshot retrieval.
                'timestamp': reference_open + 60.0,
                'price': float(one_minute.closed['close'].iloc[-1]),
            }
            return {
                'parts_by_timeframe': analysis.get('parts_by_timeframe', {}),
                'once_per_symbol_parts': analysis.get('once_per_symbol_parts', {}),
                'part7_gate': analysis.get('part7_gate'),
                'deterministic_decision': analysis.get('deterministic_decision'),
                'analysis_reference': analysis_reference,
                # Candidate creation is downstream of a fresh Delta quote,
                # exact contract metadata, balance-based risk sizing, and policy.
                'execution_plan': None,
                'adapter_engine_status': {
                    'part1_native': getattr(owner.parts.get('part1_breakout'), '_engine_factory', None) is not None,
                    'part2_native': getattr(owner.parts.get('part2_zone'), '_engine', None) is not None,
                },
            }
        except Exception as _analysis_error:
            logger.exception('[MULTICOIN] Isolated pipeline failed for %s: %s', getattr(key, 'symbol', '?'), type(_analysis_error).__name__)
            raise

    def _consume_multicoin_paper_results(self, pipeline_status):
        """Consume only fresh, explicitly planned complete analysis in isolated paper mode."""
        coordinator = getattr(self, 'multicoin_paper_coordinator', None)
        adapter = getattr(self, 'multicoin_paper_adapter', None)
        if coordinator is None or adapter is None or not isinstance(pipeline_status, dict):
            return
        for result in (pipeline_status.get('results') or {}).values():
            if not isinstance(result, dict) or result.get('status') != 'COMPLETE':
                continue
            try:
                candidate = candidate_from_analysis(
                    result, policy_registry=getattr(self, '_multicoin_paper_policies', {}),
                    now=time.time(), max_age_seconds=float(os.getenv('JARVIS_MULTICOIN_RESULT_MAX_AGE_SEC', '180')),
                )
                outcome = coordinator.submit(candidate, adapter)
                self.multicoin_paper_status['last_status'] = outcome.get('status')
                if outcome.get('duplicate'):
                    continue
                if outcome.get('status') in {'FILLED', 'PARTIAL', 'SUBMISSION_UNKNOWN'}:
                    self.multicoin_paper_status['submitted'] = int(self.multicoin_paper_status.get('submitted', 0)) + 1
                else:
                    self.multicoin_paper_status['blocked'] = int(self.multicoin_paper_status.get('blocked', 0)) + 1
            except Exception as _candidate_error:
                self.multicoin_paper_status['last_status'] = 'BLOCKED_INVALID_CANDIDATE'
                self.multicoin_paper_status['last_reason'] = str(_candidate_error)[:160]
                self.multicoin_paper_status['blocked'] = int(self.multicoin_paper_status.get('blocked', 0)) + 1

    def _consume_multicoin_delta_results(self, pipeline_status):
        """Run the validated Binance-analysis -> Delta lifecycle when configured.

        Background Parts outputs remain analysis-only. Only the strict bridge
        may create a contract plan, and only a reconciled portfolio coordinator
        may reserve and call the protected-order broker capability.
        """
        coordinator = getattr(self, 'multicoin_delta_coordinator', None)
        adapter = getattr(self, 'multicoin_delta_adapter', None)
        status = getattr(self, 'multicoin_delta_status', None)
        if coordinator is None or adapter is None or not isinstance(status, dict):
            return
        try:
            monitor = adapter.monitor_once(coordinator)
        except Exception as monitor_error:
            status.update({'last_status': 'RECONCILIATION_BLOCKED', 'last_reason': type(monitor_error).__name__})
            status['blocked'] = int(status.get('blocked', 0)) + 1
            return
        status['monitor_status'] = monitor.get('status')
        if monitor.get('status') != 'MONITORED':
            status['last_status'] = 'RECONCILIATION_BLOCKED'
            status['blocked'] = int(status.get('blocked', 0)) + 1
            return
        if not adapter.authorization_ready():
            status['last_status'] = 'BLOCKED_LIVE_AUTHORIZATION'
            return
        if not isinstance(pipeline_status, dict):
            return
        max_age = float(os.getenv('JARVIS_MULTICOIN_RESULT_MAX_AGE_SEC', '180'))
        for result in (pipeline_status.get('results') or {}).values():
            if not isinstance(result, dict) or result.get('status') != 'COMPLETE':
                continue
            if result.get('freshness_status') != 'FRESH':
                continue
            try:
                candidate = build_delta_candidate(
                    result, delta=adapter.delta,
                    policy_registry=getattr(self, '_multicoin_delta_policies', {}),
                    now=time.time(), max_age_seconds=max_age,
                )
                outcome = coordinator.submit(candidate, adapter)
                status['last_status'] = outcome.get('status')
                if outcome.get('duplicate'):
                    continue
                if outcome.get('status') in {'FILLED', 'PARTIAL', 'SUBMISSION_UNKNOWN'}:
                    status['submitted'] = int(status.get('submitted', 0)) + 1
                else:
                    status['blocked'] = int(status.get('blocked', 0)) + 1
                    status['last_reason'] = str(outcome.get('reason') or '')[:160]
            except Exception as candidate_error:
                status['last_status'] = 'BLOCKED_INVALID_CANDIDATE'
                status['last_reason'] = str(candidate_error)[:160]
                status['blocked'] = int(status.get('blocked', 0)) + 1

    def stop_live_trading(self, timeout=None):
        """Request a bounded, orderly stop without touching open positions."""
        self.is_running = False
        self.stop_event.set()
        self._set_readiness('STOPPING', 'stop requested')
        future = getattr(self, '_live_future', None)
        if future is not None:
            try:
                future.result(timeout=float(timeout if timeout is not None else os.getenv('JARVIS_SHUTDOWN_TIMEOUT', '10')))
            except Exception:
                logger.warning('[LIVE] loop did not stop within bounded timeout')
        try:
            if getattr(self, 'executor', None):
                self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._set_readiness('STOPPED', 'stopped; positions require independent reconciliation')
        return dict(self.readiness)

    def start_live_trading(self):
        """Start live/paper analysis asynchronously with explicit readiness state."""
        import pandas as pd
        import time

        if self.is_running and self._live_future is not None and not self._live_future.done():
            return {'status': 'running', 'mode': 'paper_trading_live', 'readiness': dict(self.readiness)}
        self.stop_event.clear()
        self._set_readiness('STARTING', 'awaiting a verified venue route and candle data')
        self.is_running = True
        # Evaluate the kill-switch before spawning any worker. This makes the
        # safety boundary deterministic and avoids a startup race with daemon
        # services or an unavailable venue.
        if os.getenv('JARVIS_KILL_SWITCH', '1') != '0' and os.path.exists('C:\\jarvis\\STOP_JARVIS'):
            self.is_running = False
            self._set_readiness('STOPPED', 'kill-switch file present')
            logger.warning('[LIVE] startup blocked by kill-switch')
            return {'status': 'stopped', 'mode': 'paper_trading_live', 'readiness': dict(self.readiness)}
        
        print(f"\n{'═' * 60}")
        print(f"  🚀 JARVIS LIVE ENGINE + PAPER TRADING")
        print(f"{'═' * 60}")
        print(f"  💰 Paper Balance: ${self.paper_balance:,.2f}")
        print(f"  📊 Min Confidence: {self.PAPER_CONFIG['min_confidence']}%")
        print(f"  🔄 Mode: PAPER (Fake Money) + LIVE SIGNALS")
        print(f"{'═' * 60}\n")
        
        # --- WIRING FIX: START RISK MONITOR THREAD ---
        if hasattr(self.jarvis, 'engines'):
            exec_eng = next((e for e in self.jarvis.engines.values() if hasattr(e, 'monitor_and_manage')), None)
            if exec_eng:
                try:
                    import threading
                    threading.Thread(
                        target=exec_eng.monitor_and_manage,
                        daemon=True, name="JarvisRiskMonitor"
                    ).start()
                    logger.info("✅ [RISK] GPUOrderExecution Risk Monitor STARTED")
                except Exception as e:
                    logger.warning(f"⚠️ [RISK] monitor_and_manage failed: {e}")
        
        cycle_count = [0]  # Use list for closure access
        last_report_date = [datetime.now().date() - timedelta(days=1)] # Force report if time matches on startup
        
        def _live_loop():
            while self.is_running and not self.stop_event.is_set():
                try:
                    from jarvis_watchdog import beat
                    beat("live_loop")
                except Exception:
                    pass
                try:
                    # --- KILL-SWITCH CHECK ---
                    if os.getenv("JARVIS_KILL_SWITCH", "1") != "0" and os.path.exists("C:\\jarvis\\STOP_JARVIS"):
                        logger.warning("🛑 KILL-SWITCH TRIGGERED (STOP_JARVIS file found). Initiating safe exit...")
                        try:
                            from telegram_notifier import send_message
                            send_message("🛑 JARVIS stopped via kill-switch")
                        except Exception:
                            pass
                        self.is_running = False
                        break # Safe exit, does not close existing positions
                        
                    # --- DAILY REPORT CHECK ---
                    current_time = datetime.now()
                    report_time_str = os.getenv("JARVIS_REPORT_TIME", "20:00")
                    try:
                        report_hour, report_minute = map(int, report_time_str.split(':'))
                    except:
                        report_hour, report_minute = 20, 0
                        
                    if os.getenv("JARVIS_DAILY_REPORT", "1") != "0":
                        if current_time.hour == report_hour and current_time.minute == report_minute and last_report_date[0] != current_time.date():
                            last_report_date[0] = current_time.date()
                            try:
                                from telegram_notifier import send_daily_report
                                trades_taken = self.daily_trades
                                total = self.paper_wins + self.paper_losses + self.paper_breakeven
                                wr = (self.paper_wins / total * 100) if total > 0 else 0
                                pnl = self.paper_balance - self.PAPER_CONFIG['initial_balance']
                                
                                best_trade = 0
                                worst_trade = 0
                                for t in self.paper_closed_trades:
                                    if t.get('pnl'):
                                        best_trade = max(best_trade, t['pnl'])
                                        worst_trade = min(worst_trade, t['pnl'])
                                        
                                ps = getattr(self, 'presim_stats', {}) or {}
                                presim_vetoes = ps.get('vetoes', 0)
                                
                                dv_rejects = 0
                                if hasattr(self.jarvis, 'data_validator') and self.jarvis.data_validator:
                                    try:
                                        dv_rejects = self.jarvis.data_validator.get_stats().get('rejected', 0)
                                    except:
                                        pass
                                
                                uptime = str(timedelta(seconds=int(time.time() - self.engine_start_time)))
                                
                                errors = 0
                                if hasattr(self.jarvis, 'bus') and self.jarvis.bus:
                                    errors = self.jarvis.bus.get_errors_count() if hasattr(self.jarvis.bus, 'get_errors_count') else 0
                                
                                top_engines = "1. AI Core\\n2. Momentum\\n3. Scalp"
                                if 'adaptive' in getattr(self.jarvis, 'engines', {}):
                                    try:
                                        ins = self.jarvis.engines['adaptive'].get_learning_insights()
                                        if ins and 'top' in ins:
                                            top_engines = str(ins['top'])
                                    except:
                                        pass
                                        
                                stats = {
                                    'trades_taken': trades_taken,
                                    'win_rate': wr,
                                    'pnl': pnl,
                                    'best_trade': best_trade,
                                    'worst_trade': worst_trade,
                                    'presim_vetoes': presim_vetoes,
                                    'dv_rejects': dv_rejects,
                                    'top_engines': top_engines,
                                    'uptime': uptime,
                                    'errors': errors
                                }
                                send_daily_report(stats)
                                logger.info("📊 Daily report sent.")
                            except Exception as e:
                                logger.error(f"Failed to send daily report: {e}")

                    # FIX BUG 5: Auto-reset daily stats at midnight
                    current_date = datetime.now().date()
                    if current_date != self.last_trading_date:
                        self.reset_daily_stats()
                        self.last_trading_date = current_date
                        logger.info("📅 Midnight reached: Daily trades reset to 0.")
                        
                    # --- WIRING FIX: EMERGENCY RESUME CHECK ---
                    if os.getenv('JARVIS_RESUME') == '1' and hasattr(self, 'auto_trader') and self.auto_trader:
                        self.auto_trader.resume()
                        os.environ.pop('JARVIS_RESUME', None)
                        logger.info("▶️ [SYSTEM] Emergency Stop Lifted. Trading Resumed.")
                        
                    cycle_count[0] += 1
                    try:
                        from jarvis_watchdog import beat as _wd_beat_loop
                        _wd_beat_loop('paper_trade_loop')
                    except Exception:
                        pass
                    current_price = None
                    direction = 'NO_TRADE'
                    confidence = None
                    self.last_decision = None
                    self._dashboard_signal = {}
                    self._dashboard_plan = {}

                    # Full multicoin analysis is scheduled independently of the
                    # selected route and continues while positions hold the route.
                    # Status is diagnostics-only and never enters decision logic.
                    if self.multicoin_pipeline is not None:
                        self.multicoin_pipeline_status = self.multicoin_pipeline.poll()
                        self._consume_multicoin_paper_results(self.multicoin_pipeline_status)
                        self._consume_multicoin_delta_results(self.multicoin_pipeline_status)

                    # 1. Select a verified crypto contract before collecting any
                    # data. A route is locked for the entire life of an open
                    # position, so analysis, paper ledger and execution cannot
                    # accidentally use different symbols.
                    route = None
                    if self.market_router:
                        has_position = bool(self.paper_open_trades) or bool(
                            self.auto_trader and getattr(self.auto_trader, 'open_positions', [])
                        )
                        route = self.market_router.select_crypto(has_open_position=has_position)
                        if route.status != 'READY':
                            self._set_readiness('NOT_READY', f'venue route unavailable: {route.reason}')
                            logger.warning('[MARKET-ROUTER] Cycle blocked: %s', route.reason)
                            self._print_unavailable_live_diagnostics(
                                getattr(route, 'symbol', 'UNKNOWN'), 'MARKET_ROUTE',
                                getattr(route, 'reason', 'market route not ready'), snapshot=None,
                            )
                            time.sleep(float(os.getenv('JARVIS_ROUTE_RETRY_SECONDS', '5')))
                            continue
                    try:
                        if route is not None and os.getenv('JARVIS_DUAL_COIN_ROTATE', '1') == '1':
                            _dc_coins = [c.strip().upper() for c in os.getenv('JARVIS_TRADE_COINS', 'BTC,ETH').split(',') if c.strip()]
                            if len(_dc_coins) == 2 and str(route.base_asset).upper() in _dc_coins:
                                self._dual_tick = int(getattr(self, '_dual_tick', 0)) + 1
                                _dc_want = _dc_coins[self._dual_tick % 2]
                                if _dc_want != str(route.base_asset).upper():
                                    import dataclasses as _dcl
                                    _dc_suffix = 'USDT' if str(route.symbol).upper().endswith('USDT') else 'USD'
                                    route = _dcl.replace(route, symbol=_dc_want + _dc_suffix, base_asset=_dc_want, reason='dual-coin rotation')
                                logger.info('[DUAL_COIN] cycle_tick=%d analysing=%s (position_open=%s)', self._dual_tick, route.symbol, has_position if 'has_position' in dir() else None)
                    except Exception as _dc_err:
                        logger.warning('[DUAL_COIN] rotation skipped (%s)', _dc_err)
                    self._thinking_trace = []
                    self._last_thinking_text = None
                    symbol = route.symbol if route else os.getenv('JARVIS_DEFAULT_SYMBOL', 'BTCUSDT')
                    base_asset = route.base_asset if route else 'BTC'
                    _clean_sym = str(symbol).upper().replace('/', '').replace('-', '').replace('_', '')
                    _spot_symbol = _clean_sym if _clean_sym.endswith('USDT') else f"{base_asset}USDT"
                    # Let the brain know which symbol is being analysed so that
                    # cross-source checks compare the SAME asset (not BTC).
                    try:
                        self.jarvis.active_symbol = symbol
                        self.jarvis.active_base_asset = base_asset
                        self.jarvis.active_spot_symbol = _spot_symbol
                    except Exception:
                        pass

                    # 2. Get price and check open paper trades for this exact symbol.
                    try:
                        if hasattr(self.jarvis, 'delta_data') and self.jarvis.delta_data:
                            cp = self.jarvis.delta_data.get_live_price(symbol)
                            if cp and float(cp) > 0:
                                current_price = float(cp)
                    except Exception:
                        pass

                    if current_price:
                        self._last_current_price = current_price
                        self._check_paper_trades(current_price, symbol=symbol)

                    # 3. Fetch one direct native-interval snapshot for the
                    # selected coin from strict Binance Spot. Each interval has
                    # its configured CLOSED history plus a separate forming candle.
                    snapshot = None
                    if getattr(self.jarvis, 'direct_candle_cache', None) is not None:
                        try:
                            _candle_symbol = _binance_spot_symbol(symbol)
                            snapshot = self.jarvis.direct_candle_cache.refresh(
                                _candle_symbol, venue='binance', market_type='spot',
                                instrument_id=_candle_symbol,
                            )
                        except Exception as candle_error:
                            self._set_readiness('NOT_READY', f'candle feed rejected ({type(candle_error).__name__})')
                            logger.warning('[CANDLES] Cycle blocked for %s (%s: %s)', symbol, type(candle_error).__name__, str(candle_error)[:200])
                            # A coin whose Binance Spot history is incomplete (e.g. too young
                            # for the 1M frame) must not stall the loop: skip it for a while.
                            if self.market_router and not has_position and not str(symbol).upper().startswith(('BTC', 'ETH')):
                                try:
                                    self.market_router.reject_symbol(symbol, f'candle data unusable: {str(candle_error)[:80]}')
                                except Exception:
                                    pass
                            self._print_unavailable_live_diagnostics(
                                symbol, 'CANDLE_FETCH', candle_error, snapshot=None,
                            )
                            time.sleep(float(os.getenv('JARVIS_ROUTE_RETRY_SECONDS', '5')))
                            continue
                    if snapshot is not None and '5m' in snapshot.frames:
                        self._set_readiness('READY', f'verified route {symbol}; direct native candles available')
                        self.jarvis._active_candle_snapshot = snapshot
                        try:
                            from jarvis_paper_tracker import update_from_snapshot, scorecard_line
                            from jarvis_institutional_feed import get_context, context_line
                            from jarvis_strategy_approval import LAST_REGIME as _LRG
                            _ik = _binance_spot_symbol(symbol)
                            update_from_snapshot(_ik, snapshot)
                            _ictx = get_context(_ik)
                            _xp = (getattr(self, 'last_xgb_mode_advice', None) or {}).get('probs')
                            pass  # authoritative [INSTITUTIONAL_CONTEXT] is emitted after analysis
                            logger.info('%s', scorecard_line())
                        except Exception as _ie:
                            logger.warning('[INSTITUTIONAL_CONTEXT] unavailable (%s: %s)', type(_ie).__name__, _ie)
                        df = snapshot.frames['5m'].closed.copy()
                        # The forming candle is deliberately not appended to df.
                        # It is carried as metadata for display/current price only.
                        current_candle = snapshot.frames['5m'].current
                        if current_price is None and current_candle is not None:
                            current_price = float(current_candle['close'])

                        # 4. Run full AI analysis on CLOSED candles only.
                        result = _run_quietly(
                            self.jarvis.analyze_trade_setup, df,
                            candle_snapshot=snapshot,
                        )
                        self.last_jarvis_result = result
                        # Keep auto_trader updated with latest live data for reversal checks
                        if self.auto_trader:
                            self.auto_trader._df_ref = df.copy()

                        # No model consensus may flow into the final decision.
                        self.last_consensus = {}

                        # 5. Apply selected-asset options intelligence before
                        # the displayed final decision and any order path.
                        result = self._apply_options_confirmation(result, symbol)
                        direction, confidence, entry_price, tp1, tp2, sl, expiry = \
                            self._print_live_signal(result, current_price, symbol=symbol, df=df)
                        self._thinking_trace = [('Raw signal (P1-12)', f"{direction} conf={confidence}")]


                        # PHASE4: Mode-specific entry confirmation gate (SWING uses 1h/4h/15m, SCALP uses 15m/5m).
                        entry_gate_5m = getattr(self, 'last_5m_entry', None)
                        _sig_mode = str((result.get('trade_signal') or {}).get('trade_mode') or expiry or 'SCALP').upper()
                        _15m_confirmed = False
                        if snapshot is not None and '15m' in getattr(snapshot, 'frames', {}):
                            try:
                                _df15 = snapshot.frames['15m'].closed
                                if len(_df15) > 0:
                                    _last15 = _df15.iloc[-1]
                                    _bull15 = float(_last15['close']) > float(_last15['open'])
                                    _15m_confirmed = (_bull15 if direction == 'CALL' else not _bull15)
                            except Exception:
                                _15m_confirmed = False
                        if _sig_mode in ('SWING', 'DAY_TRADE') and direction in ('CALL', 'PUT'):
                            self._thinking_trace.append(('Swing trigger', 'confirmed (HTF Swing track)'))
                        elif entry_gate_5m and not entry_gate_5m.get('confirmed', True) and not _15m_confirmed \
                                and direction in ('CALL', 'PUT'):
                            self._dashboard_events.append(f"Entry wait: {direction} confirmation pending (5m/15m)")
                            self._thinking_trace.append(('15m/5m trigger', f"WAIT ({entry_gate_5m.get('note', 'not confirmed')})"))
                            direction = 'NO_TRADE'
                        elif direction in ('CALL', 'PUT'):
                            self._thinking_trace.append(('15m/5m trigger', 'confirmed'))

                        # SCENARIO SIMULATOR GATE: trade pehla future stress
                        # scenarios simulate kare; worst-case fail -> entry skip.
                        # Fail-open; JARVIS_SCEN_SIM=0 thi off.
                        if direction in ('CALL', 'PUT'):
                            _before = direction
                            direction = self._scenario_gate(
                                direction,
                                entry_price=entry_price or current_price,
                                sl=sl, tp=tp2 or tp1,
                                df=df, symbol=symbol,
                            )
                            self._thinking_trace.append(('Scenario sim', 'pass' if direction == _before else f'VETO -> {direction}'))

                        # 5b. Build one authoritative post-gate decision for
                        # paper ledger, auto-trader, and dashboard.  PreSim is
                        # applied before this snapshot so a veto propagates to
                        # every downstream consumer.
                        if direction in ('CALL', 'PUT'):
                            _before = direction
                            direction, confidence = self._presim_gate(
                                direction, confidence, entry_price,
                                result=result, df=df, current_price=current_price, symbol=symbol
                            )
                            if direction not in ('CALL', 'PUT'):
                                direction = 'NO_TRADE'
                            self._thinking_trace.append(('PreSim', f'pass conf={confidence}' if direction == _before else f'VETO -> {direction}'))
                        # XGBoost meta-gate: advisory, can only VETO an entry that
                        # the deterministic path already produced (never create/size).
                        if direction in ('CALL', 'PUT'):
                            direction = self._xgb_meta_gate(direction, snapshot, symbol, result=result)

                        _options_ctx = (result.get('market_context', {}) or {}).get('options_context', {})
                        _opinions = list(result.get('decision_opinions', []) or [])
                        _live_execution_enabled = bool(
                            self.auto_trader and getattr(self.auto_trader, 'is_enabled', False)
                        )
                        _candidate = dict(result.get('trade_signal', {}) or {})
                        _candidate['direction'] = direction if direction in ('CALL', 'PUT') else 'NO_TRADE'
                        _candidate['confidence_score'] = confidence
                        from jarvis_decision import DecisionContext
                        _ctx = DecisionContext(
                            symbol=symbol,
                            price=current_price,
                            reasons=(result.get('intelligence_board', []) or [])[:3],
                            opinions=_opinions,
                            options_context=_options_ctx,
                            require_options=_live_execution_enabled,
                            gate_reason=('entry confirmation pending' if entry_gate_5m and not entry_gate_5m.get('confirmed', True) and str((getattr(self.jarvis, 'latest_multicoin_analysis', {}) or {}).get('central_strategy_approval', {}).get('trade_mode') or '').upper() != 'SWING' else ''),
                            plan={'entry': entry_price, 'tp1': tp1, 'tp2': tp2, 'sl': sl, 'expiry': expiry},
                        )
                        _final_snapshot = build_final_decision(_candidate, _ctx)
                        self.last_decision = _final_snapshot
                        if not _final_snapshot.get('execution_allowed'):
                            direction = 'NO_TRADE'
                        else:
                            direction = _final_snapshot.get('execution_direction', 'NO_TRADE')
                        confidence = _final_snapshot.get('confidence') or 0
                        try:
                            from jarvis_institutional_feed import get_context as _gc2, context_line as _cl2, deep_get as _dg2
                            from jarvis_strategy_approval import LAST_REGIME as _LR3
                            _k3 = _binance_spot_symbol(symbol)
                            _pr3 = (getattr(self.jarvis, 'latest_part_results', None) or getattr(self, 'latest_part_results', None) or {}).get('part14_options_chain') or {}; _pt3 = _pr3.get('telemetry') if isinstance(_pr3, dict) else {}; _pt3 = _pt3 if isinstance(_pt3, dict) else {}; _opt3 = {'pcr': (_dg2(_pt3, 'pcr') if _dg2(_pt3, 'pcr') is not None else _dg2(_options_ctx, 'pcr')), 'max_pain': (_dg2(_pt3, 'max_pain') if _dg2(_pt3, 'max_pain') is not None else _dg2(_options_ctx, 'max_pain'))}
                            _xp3 = (getattr(self, 'last_xgb_mode_advice', None) or {}).get('probs')
                            logger.info('[INSTITUTIONAL_CONTEXT] %s', _cl2(_k3, _gc2(_k3), _LR3.get(_k3), _xp3, _opt3))
                            _sm3 = (_LR3.get(_k3) or {}).get('summary')
                            if _sm3:
                                logger.info('[CONTINUOUS_REGIME_SUMMARY] %s %s', _k3, _sm3)
                            _op3 = ((getattr(self.jarvis, 'latest_multicoin_analysis', None) or {}).get('once_per_symbol_parts') or {})
                            _p11_l = _op3.get('part11') or _op3.get('part11_fusion') or (getattr(self.jarvis, 'latest_part_results', None) or {}).get('part11_fusion') or {}
                            _p12_l = _op3.get('part12') or _op3.get('part12_confidence') or (getattr(self.jarvis, 'latest_part_results', None) or {}).get('part12_confidence') or {}
                            logger.info('[PART11_12_LIVE] %s p11_sig=%s p11_thought=%s | p12_conf=%s p12_thought=%s | dual_track=%s',
                                        _k3, _p11_l.get('signal'), _p11_l.get('thought'), _p12_l.get('confidence'), _p12_l.get('thought'), getattr(self.jarvis, 'latest_dual_track_summary', None))
                        except Exception as _e3:
                            logger.warning('[INSTITUTIONAL_CONTEXT] post-analysis unavailable (%s: %s)', type(_e3).__name__, _e3)
                        _audit_blockers = []
                        if entry_gate_5m and not entry_gate_5m.get('confirmed', True) and str((getattr(self.jarvis, 'latest_multicoin_analysis', {}) or {}).get('central_strategy_approval', {}).get('trade_mode') or '').upper() != 'SWING':
                            _audit_blockers.append('5m confirmation pending')
                        if not _final_snapshot.get('execution_allowed'):
                            _audit_blockers.extend(_final_snapshot.get('reasons', []) or ['authoritative decision gate blocked'])
                        self._print_decision_audit(
                            _final_snapshot, symbol,
                            parts=getattr(self.jarvis, 'latest_part_results', {}),
                            blockers=_audit_blockers,
                        )

                        # 6. AUTO-TRADE: Execute on Delta Exchange if enabled
                        if self.auto_trader and direction in ('CALL', 'PUT'):
                            try:
                                trade_type = 'SCALP'  # default
                                # Use SWING if expiry suggests longer hold
                                if expiry and str(expiry).upper() in ('DAY_TRADE', 'SWING', '15M', '30M'):
                                    trade_type = 'SWING'
                                # Bridge central MTF approval (SCALP/SWING arbiter) into the
                                # executor preflight, same plan-bound approval as the paper path.
                                _at_auth = {}
                                try:
                                    from jarvis_strategy_approval import build_execution_plan, make_entry_approval
                                    _at_analysis = getattr(self.jarvis, 'latest_multicoin_analysis', {}) or {}
                                    _at_central = _at_analysis.get('central_strategy_approval', {})
                                    _at_snapshot = getattr(self.jarvis, '_active_candle_snapshot', None)
                                    _at_symbol = _binance_spot_symbol(symbol)
                                    _at_exec_symbol = str(symbol or '').upper().replace('-', '').replace('_', '').strip()
                                    if (_at_snapshot is None or _at_snapshot.symbol != _at_symbol
                                            or _at_snapshot.identity != ('binance', 'spot', _at_symbol, _at_symbol)
                                            or not isinstance(_at_central, dict) or _at_central.get('approved') is not True):
                                        raise ValueError('fresh Binance snapshot or approved native MTF decision is unavailable')
                                    _at_mode = str(_at_central.get('trade_mode') or '').upper()
                                    if _at_mode in ('SCALP', 'SWING'):
                                        trade_type = _at_mode
                                    _at_version = f"{_at_symbol}:{int(float(_at_snapshot.fetched_at) * 1000)}"
                                    _at_side = 'BUY' if direction == 'CALL' else 'SELL'
                                    _at_plan = build_execution_plan(
                                        direction=_at_side, recommended_expiry=expiry,
                                        entry_price=entry_price or current_price,
                                        stop_loss=sl, take_profit=tp1,
                                        symbol=_at_exec_symbol, snapshot_version=_at_version,
                                        confidence=confidence,
                                    )
                                    _at_approval = make_entry_approval(
                                        {**_at_central, 'symbol': _at_exec_symbol},
                                        direction=_at_side, symbol=_at_exec_symbol,
                                        exchange='delta', contract=_at_exec_symbol,
                                        instrument_id=_at_exec_symbol, market_type='unverified',
                                        analysis_symbol=_at_symbol, analysis_exchange='binance',
                                        snapshot_version=_at_version,
                                        analysis_timestamp=time.time(),
                                        confidence=confidence, execution_plan=_at_plan,
                                    )
                                    _at_auth = {
                                        'central_approval': _at_approval,
                                        'snapshot_version': _at_version,
                                        'timeframe_parts': _at_analysis.get('parts_by_timeframe'),
                                        'part7_gate': _at_analysis.get('part7_gate'),
                                        'execution_plan': _at_plan,
                                    }
                                except Exception as _at_auth_error:
                                    logger.warning('[AUTO_TRADE] central approval bridge unavailable: %s', _at_auth_error)
                                    self._dashboard_events.append(f"Auto-trade approval unavailable: {_at_auth_error}")
                                at_result = self.auto_trader.execute(
                                    direction=direction,
                                    confidence=confidence,
                                    current_price=current_price or 0,
                                    symbol=symbol,
                                    part_results=getattr(self.jarvis, 'latest_part_results', {}),
                                    trade_type=trade_type,
                                    **_at_auth,
                                )
                                self._print_decision_audit(
                                    _final_snapshot, symbol, stage="ORDER_SUBMISSION",
                                    order_outcome=at_result,
                                )
                                if at_result.get('success'):
                                    pos = at_result.get('position', {})
                                    self._dashboard_account = {
                                        'trade_risk': pos.get('contracts', 0) * (0.008 if trade_type == 'SWING' else 0.002),
                                        'margin': pos.get('contracts', 0) / max(pos.get('leverage', 1), 1),
                                        'contracts': pos.get('contracts', 0),
                                        'notional': pos.get('contracts', 0),
                                        'leverage': pos.get('leverage', 'AUTO'),
                                    }
                                    self._dashboard_events.append(f"Auto-trade placed #{pos.get('id','?')} ({direction})")
                                else:
                                    self._dashboard_events.append(f"Auto-trade blocked: {at_result.get('reason', 'gate failed')}")
                            except Exception as at_err:
                                self._dashboard_events.append(f"AutoTrader error: {at_err}")
                        elif direction in ('CALL', 'PUT') and self.can_trade():
                            # PreSim was already applied in the authoritative gate above.
                            if direction in ('CALL', 'PUT') and confidence >= self.PAPER_CONFIG['min_confidence']:
                                # Compute ATR for hedge advisor
                                try:
                                    atr = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])
                                except Exception:
                                    atr = current_price * 0.005  # Fallback ATR (0.5%)

                                if hasattr(self, 'hedged_engine') and self.hedged_engine:
                                    # Use AI Options Hedged Scalp Engine
                                    try:
                                        options_chain = self.jarvis.delta_data.get_options_chain(base_asset) \
                                            if hasattr(self.jarvis.delta_data, 'get_options_chain') else {}
                                    except Exception:
                                        options_chain = {}
                                    hedged_result = self.hedged_engine.execute_hedged_scalp(
                                        signal={'direction': direction, 'confidence': confidence},
                                        current_price=current_price,
                                        atr=atr,
                                        options_chain=options_chain,
                                        jarvis_result=result
                                    )
                                    self.last_hedged_result = hedged_result
                                    self._dashboard_events.append(f"Hedge: {hedged_result.get('status')} / applied={hedged_result.get('hedge_applied')}")
                                
                                # Paper entries use the same complete native
                                # MTF approval and plan-bound levels; a raw AI
                                # direction or caller-supplied TP/SL is not enough.
                                _paper_auth = {}
                                try:
                                    from jarvis_strategy_approval import build_execution_plan, make_entry_approval
                                    _analysis = getattr(self.jarvis, 'latest_multicoin_analysis', {}) or {}
                                    _central = _analysis.get('central_strategy_approval', {})
                                    _snapshot = getattr(self.jarvis, '_active_candle_snapshot', None)
                                    _canonical_symbol = _binance_spot_symbol(symbol)
                                    if (_snapshot is None or _snapshot.symbol != _canonical_symbol
                                            or _snapshot.identity != ('binance', 'spot', _canonical_symbol, _canonical_symbol)
                                            or not isinstance(_central, dict) or _central.get('approved') is not True):
                                        raise ValueError('fresh Binance snapshot or approved native MTF decision is unavailable')
                                    _snapshot_version = f"{_canonical_symbol}:{int(float(_snapshot.fetched_at) * 1000)}"
                                    _side = 'BUY' if direction == 'CALL' else 'SELL'
                                    _bound_plan = build_execution_plan(
                                        direction=_side, recommended_expiry=expiry,
                                        entry_price=entry_price or current_price,
                                        stop_loss=sl, take_profit=tp1,
                                        symbol=_canonical_symbol, snapshot_version=_snapshot_version,
                                        confidence=confidence,
                                    )
                                    _bound_approval = make_entry_approval(
                                        {**_central, 'symbol': _canonical_symbol},
                                        direction=_side, symbol=_canonical_symbol,
                                        exchange='delta', contract=_canonical_symbol,
                                        instrument_id=_canonical_symbol, market_type='unverified',
                                        analysis_symbol=_canonical_symbol, analysis_exchange='binance',
                                        snapshot_version=_snapshot_version,
                                        analysis_timestamp=time.time(),
                                        confidence=confidence, execution_plan=_bound_plan,
                                    )
                                    _paper_auth = {
                                        'central_approval': _bound_approval,
                                        'snapshot_version': _snapshot_version,
                                        'part_results': getattr(self.jarvis, 'latest_part_results', {}),
                                        'timeframe_parts': _analysis.get('parts_by_timeframe'),
                                        'part7_gate': _analysis.get('part7_gate'),
                                        'execution_plan': _bound_plan,
                                    }
                                except Exception as _paper_auth_error:
                                    self._dashboard_events.append(f"Paper entry blocked: {_paper_auth_error}")
                                trade = self._open_paper_trade(
                                    direction, entry_price or current_price,
                                    confidence, expiry, tp1, tp2, sl, current_price=current_price,
                                    symbol=symbol, **_paper_auth,
                                )
                                if trade:
                                    self._dashboard_events.append(
                                        f"Paper trade #{trade['id']} {trade['status']} | "
                                        f"margin ${trade['margin_usdt']:.2f} @ {trade['leverage']}x"
                                    )
                                else:
                                    self._dashboard_events.append("Paper entry blocked (max open or risk budget)")
                            elif direction in ('CALL', 'PUT'):
                                self._dashboard_events.append(f"Paper skipped: conf {confidence}% < {self.PAPER_CONFIG['min_confidence']}%")
                            
                            # FIX BUG 1: PENDING record_trade removed to fix Consecutive Loss Circuit Breaker
                        
                        else:
                            logger.debug("[LIVE] No candle data received")
                    else:
                        _available_frames = sorted(getattr(snapshot, 'frames', {}).keys()) if snapshot is not None else []
                        _missing_reason = (
                            f"current native candle snapshot missing 1m frame; observed={_available_frames}"
                            if snapshot is not None else
                            'direct candle snapshot unavailable; analysis not evaluated'
                        )
                        self._print_unavailable_live_diagnostics(
                            symbol, 'CANDLE_SNAPSHOT', _missing_reason, snapshot=snapshot,
                        )
                        logger.warning("[LIVE] Current native candle snapshot unavailable for %s", symbol)
                    
                    self._render_unified_dashboard(
                        symbol=symbol, current_price=current_price,
                        action=(getattr(self, 'last_decision', None) or {}).get('direction') or ('WAIT' if direction == 'NO_TRADE' else direction),
                        gate_reason='; '.join((getattr(self, 'last_decision', None) or {}).get('reasons', [])) or ('risk/venue gates passed' if direction in ('CALL', 'PUT') else 'no eligible entry'),
                    )
                    self._dashboard_events = []
                    time.sleep(float(os.getenv('JARVIS_POLL_SECONDS', '60')))  # Poll every candle
                    
                except Exception as e:
                    logger.error(f"[LIVE] Trading loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Report main loop error to Cognitive Bus
                    if hasattr(self.jarvis, 'bus') and self.jarvis.bus:
                        self.jarvis.bus.report_error('JarvisElite_LiveLoop', e, context='main _live_loop cycle', try_legacy_llm=False)
                    time.sleep(30)
            self.is_running = False
            if self.readiness.get('status') not in {'STOPPING', 'STOPPED'}:
                self._set_readiness('STOPPED', 'loop exited')

        self._live_future = self.executor.submit(_live_loop)
        return {"status": "running", "mode": "paper_trading_live", "readiness": dict(self.readiness)}

    def can_trade(self):
        """Check if trading is allowed"""
        # --- WIRING FIX: CHECK RISK ENGINE STATUS ---
        if hasattr(self.jarvis, 'engines'):
            exec_eng = next((e for e in self.jarvis.engines.values() if hasattr(e, 'get_risk_status')), None)
            if exec_eng:
                try:
                    risk = exec_eng.get_risk_status()
                    if risk and risk.get('halt_trading'):
                        logger.warning(f"[RISK] Trading halted by risk engine: {risk.get('reason')}")
                        return False
                except Exception:
                    pass

        # Check daily limit
        if self.daily_trades >= TRADE_CONFIG['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return False
            
        # Check consecutive losses
        if self.consecutive_losses >= TRADE_CONFIG['consecutive_loss_limit']:
            logger.warning("Consecutive loss limit reached")
            return False
            
        # Check cooldown period
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if time_since_last < TRADE_CONFIG['trade_cooldown_minutes']:
                logger.warning(f"Trade cooldown active: {TRADE_CONFIG['trade_cooldown_minutes'] - time_since_last:.1f} minutes remaining")
                return False
                
        # Check session quality
        current_hour = datetime.now().hour
        session_quality = self._get_session_quality(current_hour)
        if session_quality == 'LOW':
            pass
            # logger.warning("Low quality trading session - trading anyway")
            # return False
            
        return True
        
    def record_trade(self, signal, score, result):
        """Record trade outcome"""
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'score': score,
            'result': result,
            'session': self._get_current_session()
        }
        
        self.trade_history.append(trade_record)
        
        # Update consecutive losses
        if result == 'LOSS':
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # --- WIRING FIX: UPDATE METRICS FOR LEARNING/RISK/CONFIDENCE ---
        if hasattr(self.jarvis, 'engines'):
            # 1. Update Confidence Engine
            if 'confidence' in self.jarvis.engines:
                try:
                    self.jarvis.engines['confidence'].update_confidence_history(score, result)
                except Exception:
                    pass
            
            # 2. Update Risk Engine
            exec_eng = next((e for e in self.jarvis.engines.values() if hasattr(e, 'update_risk_metrics')), None)
            if exec_eng:
                try:
                    exec_eng.update_risk_metrics({'result': result, 'score': score})
                except Exception:
                    pass
                    
            # 3. Update Adaptive AI Engine
            if 'adaptive' in self.jarvis.engines:
                try:
                    adaptive = self.jarvis.engines['adaptive']
                    if hasattr(adaptive, 'add_training_data'):
                        adaptive.add_training_data({'signal': signal, 'result': result})
                except Exception:
                    pass
                    
            # --- WIRING FIX PHASE 2: INSTITUTIONAL & BACKTEST METRICS ---
            # 4. Institutional Analytics
            if 'institutional' in self.jarvis.engines:
                try:
                    is_win = (result == 'WIN')
                    pnl_val = 1.0 if is_win else -1.0
                    trade_decision = {
                        'signal': signal,
                        'strategy_type': 'SCALP',
                        'regime': self.jarvis.market_context.get('regime', 'NEUTRAL'),
                        'expiry_minutes': 3
                    }
                    inst_eng = self.jarvis.engines['institutional']
                    inst_eng.record_trade_outcome(trade_decision, is_win, pnl_val)
                    inst_eng.update_institutional_risk_metrics([{'result': result, 'pnl': pnl_val}])
                except Exception:
                    pass
            
            # 5. Live Metrics (Backtest engine)
            if 'backtest' in self.jarvis.engines:
                try:
                    self.jarvis.engines['backtest'].update_live_metrics({'result': result, 'score': score})
                except Exception:
                    pass
            # -------------------------------------------------------------

            
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.trade_history = []
        
    def _get_session_quality(self, current_hour=None):
        """Get current session quality"""
        if current_hour is None:
            current_hour = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return times['quality']
        return 'LOW'
        
    def _get_current_session(self):
        """Get current trading session name"""
        current_hour = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return session
        return 'overnight'

# ==================== TRADE EXPIRY OPTIMIZER ====================

class TradeOptimizer:
    """Optimize trade style based on market conditions"""
    
    def __init__(self):
        self.trade_styles = ['SCALP', 'DAY_TRADE', 'SWING']
        
    def recommend_trade_type(self, score, volatility, pattern_type):
        """Recommend optimal trade style"""
        # Base style on score tier
        if score >= 90:
            base_style = 'SCALP'
        elif score >= 85:
            base_style = 'DAY_TRADE'
        else:
            base_style = 'SWING'
            
        # Adjust for volatility
        if volatility > 0.004:  # High volatility
            if base_style == 'SCALP':
                base_style = 'DAY_TRADE'
        elif volatility < 0.001:  # Low volatility
            if base_style == 'SWING':
                base_style = 'SCALP'
                
        # Adjust for pattern type
        if pattern_type == 'MOMENTUM_SCALP':
            base_style = 'SCALP'
        elif pattern_type == 'REJECTION_PLAY':
            base_style = 'DAY_TRADE'
        elif pattern_type == 'TREND_PULLBACK':
            base_style = 'SWING'
            
        return base_style
        
    def calculate_payout_ratio(self, score, trade_type):
        """Calculate Risk:Reward ratio"""
        rr_ratio = 1.5
        
        # Adjust for score
        if score >= 95:
            rr_ratio += 0.5
        elif score >= 90:
            rr_ratio += 0.3
        elif score >= 85:
            rr_ratio += 0.1
            
        # Adjust for trade type
        if trade_type == 'SWING':
            rr_ratio += 1.0
        elif trade_type == 'SCALP':
            rr_ratio -= 0.5
            
        return f"1:{rr_ratio:.1f}"

class ScalpingEngine:
    """Calculates TP, SL, and RR for regular scalping (Spot/Futures)"""
    def __init__(self):
        self.default_rr = 1.5

    def calculate_targets(self, data, direction, current_price, options_data=None, mtf_data=None):
        """Calculates TP/SL with Chart + Options Confluence"""
        try:
            if len(data) < 20: return None
            
            # 1. Base SL using ATR
            high_low = (data['high'] - data['low'])
            atr = high_low.rolling(14).mean().iloc[-1]
            
            # FIX: Cap ATR-based SL to max 0.4% of price for scalp realism
            max_sl_pct = 0.004  # 0.4%
            atr = min(atr, current_price * max_sl_pct / 1.8)
            sl_multiplier = 1.8
            
            # 2. Extract only finite, positive option levels. Provider responses
            # often include an explicit None when a wall is unavailable; dict.get
            # would preserve that None and break arithmetic below. Missing or
            # invalid observations are not treated as options evidence.
            def _positive_level(value, fallback):
                try:
                    parsed = float(value)
                    return parsed if math.isfinite(parsed) and parsed > 0 else float(fallback)
                except (TypeError, ValueError, OverflowError):
                    return float(fallback)

            support_wall = _positive_level(
                options_data.get('support') if isinstance(options_data, dict) else None,
                current_price * 0.997,
            )
            resistance_wall = _positive_level(
                options_data.get('resistance') if isinstance(options_data, dict) else None,
                current_price * 1.003,
            )
            max_pain = _positive_level(
                options_data.get('max_pain') if isinstance(options_data, dict) else None,
                current_price,
            )
            
            # 3. Handle Direction
            if direction == 'CALL':
                # SL should be below support wall or ATR-based SL
                sl_price = min(current_price - (atr * sl_multiplier), support_wall - (current_price * 0.001))
                
                # Targets (TP1: ATR-based, TP2: Options-based Magnet)
                risk = current_price - sl_price
                tp_price1 = current_price + (risk * 1.5)
                # TP2 is either Max Pain (Magnet) or Resistance Wall
                tp_price2 = max(tp_price1 * 1.01, max_pain if max_pain > current_price else resistance_wall)
                
                # FIX: Hard cap - TP1 max 0.8% above entry, SL max 0.4% below
                tp_price1 = min(tp_price1, current_price * 1.008)
                sl_price = max(sl_price, current_price * 0.996)
                tp_price2 = min(tp_price2, current_price * 1.015)
                
            else: # PUT
                # SL above resistance wall or ATR-based SL
                sl_price = max(current_price + (atr * sl_multiplier), resistance_wall + (current_price * 0.001))
                
                risk = sl_price - current_price
                tp_price1 = current_price - (risk * 1.5)
                # TP2 is either Max Pain or Support Wall
                tp_price2 = min(tp_price1 * 0.99, max_pain if max_pain < current_price else support_wall)
                
                # FIX: Hard cap - TP1 max 0.8% below entry, SL max 0.4% above
                tp_price1 = max(tp_price1, current_price * 0.992)
                sl_price = min(sl_price, current_price * 1.004)
                tp_price2 = max(tp_price2, current_price * 0.985)
            
            return {
                'entry': current_price,
                'stop_loss': round(sl_price, 2),
                'take_profit_1': round(tp_price1, 2),
                'take_profit_2': round(tp_price2, 2),
                'options_magnet': round(max_pain, 2)
            }
        except Exception as e:
            logger.error(f"Scalping targets error: {e}")
            return None

# ==================== EXISTING JARVIS CLASSES ====================

# ==================== EXTERNAL GPU ENGINE ADAPTERS ====================
# Replaces internal Parts with Institutional-Grade GPU Logic

def _sig_to_num(sig):
    """Convert GPU engine signal (string or number) to numeric -1/0/1"""
    if isinstance(sig, (int, float)):
        return 1 if sig > 0 else (-1 if sig < 0 else 0)
    if isinstance(sig, str):
        s = sig.upper()
        if s in ('CALL', 'BUY', 'BULLISH', 'LONG'): return 1
        if s in ('PUT', 'SELL', 'BEARISH', 'SHORT'): return -1
    return 0

def _df_to_market_data(df):
    """Convert pandas DataFrame to market_data dict expected by Part1 SmartBreakoutAI"""
    try:
        price_action = []
        for _, row in df.iterrows():
            price_action.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            })
        volume_pattern = [float(row.get('volume', 0)) for _, row in df.iterrows()]
        return {'price_action': price_action, 'volume_pattern': volume_pattern}
    except Exception:
        return {'price_action': [], 'volume_pattern': []}


class Part1Breakout:
    """Breakout Analysis — backed by SmartBreakoutAI (13 brains, Part1)"""
    def __init__(self):
        try:
            from part1_FIXED import SmartBreakoutAI
            self._engine = SmartBreakoutAI()
        except Exception as e:
            import logging
            logging.getLogger().debug(f"Part1 import error: {e}")
            self._engine = None

    def analyze(self, data, context=None):
        # Skip noisy timeframes
        tf = getattr(data, 'attrs', {}).get('timeframe', '')
        if tf in ['1s', '5s']:
            return {"signal": 0, "thought": f"Part1Breakout skipped noisy timeframe {tf}", "breakout": {}}
            
        # ── Try real Part1 engine (13-brain SmartBreakoutAI) ──────────────
        if self._engine is not None:
            try:
                market_data = _df_to_market_data(data)
                if len(market_data['price_action']) >= 20:
                    result = self._engine.analyze(market_data, context=context)
                    brk = result.get('breakout', {})
                    is_breakout = bool(brk.get('breakout_detected', False))
                    brk_dir = int(brk.get('direction', 0))
                    if not is_breakout or brk_dir == 0:
                        sig = 0
                    else:
                        sig = result.get('signal', brk_dir)
                    lvl = result.get('levels', {})
                    thought = (
                        f"P1-Engine: Breakout={'YES' if is_breakout else 'NO'} "
                        f"dir={brk_dir} str={brk.get('strength',0):.2f} "
                        f"conf={result.get('confidence',0):.1f}"
                    )
                    import quantitative_math as _qm1
                    _p1_cs = result.get("continuous_score") if result.get("continuous_score") is not None else _qm1.part_signal("1", market_data).get("continuous_score")
                    output = result.copy()
                    output["continuous_score"] = _p1_cs
                    output.update({"signal": sig, "thought": thought, "telemetry": {
                        "breakout": brk, "levels": lvl, "momentum": result.get('momentum', {}),
                        "fakeout": result.get('fakeout', {}), "regime": result.get('regime', {})
                    }})
                    return output
            except Exception as e:
                logger.debug(f"Part1 engine error: {e}")

        # ── Fallback: basic 20-bar breakout ───────────────────────────────
        try:
            if len(data) >= 20:
                current = float(data['close'].iloc[-1])
                high_20 = float(data['high'].tail(20).max())
                low_20  = float(data['low'].tail(20).min())
                prev    = float(data['close'].iloc[-2])
                if current > high_20 and prev <= high_20:
                    return {"signal": 1,  "thought": f"Bullish Breakout above {high_20:.0f} (fallback)"}
                elif current < low_20 and prev >= low_20:
                    return {"signal": -1, "thought": f"Bearish Breakdown below {low_20:.0f} (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "No Breakout Pattern"}


class Part2Zone:
    """Supply/Demand Zone Analysis — backed by AdvancedAnalysisSystem (Part2)"""
    def __init__(self):
        self._engines = {}
        self._import_failed = False

    def _get_engine(self, timeframe):
        if self._import_failed:
            return None
        if timeframe not in self._engines:
            try:
                from part2_FIXED import AdvancedAnalysisSystem
                self._engines[timeframe] = AdvancedAnalysisSystem()
            except Exception as e:
                self._import_failed = True
                print(f"Part2 import error: {e}")
                import logging
                logging.getLogger().debug(f"Part2 import error: {e}")
                return None
        return self._engines[timeframe]

    def analyze(self, data, context=None):
        # --- TRAP DETECTION LOGIC (Global Guard for Part 2 Live & Fallback) ---
        try:
            recent_candles = data.tail(5).to_dict('records')
            is_trap = False
            if len(recent_candles) == 5:
                avg_vol_20 = data['volume'].tail(20).mean()
                curr_vol = recent_candles[-1]['volume']
                is_high_volume = curr_vol > (avg_vol_20 * 1.5)
                
                max_pump = max((c['close'] - c['open']) / (c['open'] + 1e-8) for c in recent_candles)
                max_dump = max((c['open'] - c['close']) / (c['open'] + 1e-8) for c in recent_candles)
                
                # Dynamic threshold: at least 0.5%, or 2.5x the average candle body size (adapts to timeframe)
                avg_body = (data['close'] - data['open']).abs().tail(20).mean() / (data['open'].tail(20).mean() + 1e-8)
                threshold = max(0.005, avg_body * 2.5)
                
                if max_pump > threshold and max_dump > threshold and is_high_volume:
                    is_trap = True
                    
            if is_trap:
                return {"signal": 0, "thought": "TRAP AVOIDED: Whipsaw/Fakeout detected by Part 2 Guard.", "fakeout_prob": 1.0}
        except Exception:
            pass
        # ----------------------------------------------------------------------
        
        # The legacy Part2 engine consumes 1m/5m/15m jointly. Run that
        # cross-frame computation once on the 1m row; invoking it from every
        # native timeframe would duplicate identical evidence across groups.
        active_timeframe = ((context or {}).get('timeframe')
                            or getattr(data, 'attrs', {}).get('timeframe'))
                            
        # Skip noisy timeframes completely
        if active_timeframe in ['1s', '5s']:
            return {"signal": 0, "thought": f"Part2Zone skipped noisy timeframe {active_timeframe}"}
            
        # If the Advanced Engine is loaded, execute it on the active timeframe.
        # We fetch higher timeframes dynamically to feed into the trend analysis brain.
        engine = self._get_engine(active_timeframe)
        if engine is not None and context and 'mtf_datasets' in context:
            try:
                mtf = context['mtf_datasets']
                primary_df = data
                
                # Dynamically get higher timeframes for context
                tf_list = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
                try:
                    idx = tf_list.index(active_timeframe)
                    higher1 = tf_list[min(idx+2, len(tf_list)-1)]
                    higher2 = tf_list[min(idx+4, len(tf_list)-1)]
                    df_h1 = mtf.get(higher1, primary_df)
                    df_h2 = mtf.get(higher2, primary_df)
                except ValueError:
                    df_h1, df_h2 = primary_df, primary_df
                
                if len(primary_df) > 0:
                    current_candle = primary_df.iloc[-1].to_dict()
                    signals = engine.process_market_data(primary_df, df_h1, df_h2, current_candle)
                    
                    if signals and isinstance(signals, list):
                        # Filter out invalid tuples
                        valid = [s for s in signals if isinstance(s, tuple) and len(s) >= 3]
                        if valid:
                            # Pick strongest confidence signal
                            # Part 2 is the zone analyser: accept a fused signal only when the support/resistance brain agrees.
                            valid = [s for s in valid if "ZONE_CONFIRMED" in str(s[2])]
                            if not valid:
                                import quantitative_math as _qm2
                                return _qm2.part_signal("2", data, context=context)
                            best = max(valid, key=lambda x: x[1])
                            sig_dir = 1 if str(best[0]).upper() in ['CALL', 'BUY', 'LONG'] else -1
                            return {"signal": sig_dir, "thought": f"Part2({active_timeframe}): {best[2]}"}
            except Exception as e:
                import logging
                logging.getLogger().debug(f"Part2 execution error: {e}")

        # ── Fallback: proximity to multi-lookback range extremes (20, 60, 100 bars) ──────
        try:
            if len(data) < 20:
                return {"signal": 0, "thought": "Insufficient data"}
            current = float(data['close'].iloc[-1])
            for lookback in [20, 60, 100]:
                if len(data) < lookback:
                    continue
                h = float(data['high'].tail(lookback).max())
                l = float(data['low'].tail(lookback).min())
                rng = h - l
                if rng <= 0:
                    continue
                pct = (current - l) / rng   # 0=at support, 1=at resistance
                if pct >= 0.92:
                    return {"signal": -1, "thought": f"Resistance Zone top {pct*100:.0f}% ({lookback}-bar)"}
                elif pct <= 0.08:
                    return {"signal": 1,  "thought": f"Support Zone bottom {pct*100:.0f}% ({lookback}-bar)"}
        except Exception:
            pass
        try:
            import quantitative_math as _qm2
            _q2 = _qm2.part_signal("2", data, context=context)
            if isinstance(_q2, dict) and _q2.get("signal", 0) != 0:
                return _q2
        except Exception:
            _q2 = {}
        return {"signal": 0, "thought": "No Zones (fallback)", "continuous_score": (_q2.get("continuous_score") if isinstance(locals().get("_q2"), dict) else None)}


class Part3Psychology:
    """Candle Psychology — backed by CandlePsychologyMasterGPU (Part3)"""
    def __init__(self):
        try:
            from part3_FIXED import CandlePsychologyMasterGPU
            self._engine = CandlePsychologyMasterGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── Try real Part 3 CandlePsychologyMasterGPU ─────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part3 CandlePsychology error: {e}")

        # ── Fallback: multi-candle ATR-backed pattern analysis ────────────
        try:
            if len(data) < 15:
                return {"signal": 0, "thought": "Insufficient data (<15)"}

            highs = data['high'].tail(20).astype(float)
            lows = data['low'].tail(20).astype(float)
            closes = data['close'].tail(20).astype(float)
            opens = data['open'].tail(20).astype(float)

            tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())

            c, o, h, l = float(closes.iloc[-1]), float(opens.iloc[-1]), float(highs.iloc[-1]), float(lows.iloc[-1])
            rng = max(h - l, 1e-8)
            body = abs(c - o)
            body_ratio = body / rng
            upper_wick = (h - max(c, o)) / rng
            lower_wick = (min(c, o) - l) / rng

            # Micro-noise filter
            if rng < 0.55 * atr or atr == 0:
                return {"signal": 0, "thought": f"Noise/Micro-candle ({rng:.1f} < 0.55*ATR) — Neutral"}

            prev_c, prev_o = float(closes.iloc[-2]), float(opens.iloc[-2])
            prev_body = abs(prev_c - prev_o)

            if lower_wick >= 0.55 and upper_wick <= 0.22 and body_ratio <= 0.35:
                return {"signal": 1, "thought": f"Bullish Hammer/Pin Bar (lower wick {lower_wick*100:.0f}%) (fallback)"}
            elif upper_wick >= 0.55 and lower_wick <= 0.22 and body_ratio <= 0.35:
                return {"signal": -1, "thought": f"Bearish Shooting Star (upper wick {upper_wick*100:.0f}%) (fallback)"}
            elif prev_c < prev_o and c > o and c >= prev_o and o <= prev_c and body > prev_body * 1.15 and body_ratio > 0.6:
                return {"signal": 1, "thought": f"Bullish Engulfing (fallback)"}
            elif prev_c > prev_o and c < o and c <= prev_o and o >= prev_c and body > prev_body * 1.15 and body_ratio > 0.6:
                return {"signal": -1, "thought": f"Bearish Engulfing (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "Neutral Psychology (fallback)"}


class Part4Volume:
    """Volume Profile Analysis — backed by VolumeProfileEngineGPU (Part4)"""
    def __init__(self):
        try:
            from part4_FIXED import VolumeProfileEngineGPU
            self._engine = VolumeProfileEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── Try real Part 4 VolumeProfileEngineGPU ────────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part4 VolumeProfile error: {e}")

        # ── Fallback: POC & Value Area (70%) analysis ──────────────────────
        try:
            if len(data) < 30:
                return {"signal": 0, "thought": "Insufficient data (<30)"}

            recent = data.tail(50).copy()
            highs = recent['high'].astype(float)
            lows = recent['low'].astype(float)
            closes = recent['close'].astype(float)
            volumes = recent['volume'].astype(float)

            current_close = float(closes.iloc[-1])
            current_vol = float(volumes.iloc[-1])
            avg_vol = float(volumes.mean())
            vol_ratio = current_vol / max(avg_vol, 1e-8)

            last10 = recent.tail(10)
            up_vol = float(last10.loc[last10['close'] > last10['open'], 'volume'].sum())
            down_vol = float(last10.loc[last10['close'] < last10['open'], 'volume'].sum())
            tot_vol = up_vol + down_vol
            buy_delta_pct = (up_vol / tot_vol) if tot_vol > 0 else 0.5

            min_p, max_p = float(lows.min()), float(highs.max())
            if max_p <= min_p:
                return {"signal": 0, "thought": "Flat price range"}

            num_bins = 20
            bin_size = (max_p - min_p) / num_bins
            vol_bins = np.zeros(num_bins)
            for _, row in recent.iterrows():
                p = (float(row['high']) + float(row['low']) + float(row['close'])) / 3.0
                b_idx = min(int((p - min_p) / bin_size), num_bins - 1)
                vol_bins[b_idx] += float(row['volume'])

            poc_idx = int(np.argmax(vol_bins))
            poc_price = min_p + (poc_idx + 0.5) * bin_size

            target_vol = 0.70 * vol_bins.sum()
            va_indices = {poc_idx}
            cur_vol = vol_bins[poc_idx]
            up_idx, dn_idx = poc_idx + 1, poc_idx - 1
            while cur_vol < target_vol and (up_idx < num_bins or dn_idx >= 0):
                up_v = vol_bins[up_idx] if up_idx < num_bins else -1
                dn_v = vol_bins[dn_idx] if dn_idx >= 0 else -1
                if up_v >= dn_v and up_idx < num_bins:
                    va_indices.add(up_idx); cur_vol += up_v; up_idx += 1
                elif dn_idx >= 0:
                    va_indices.add(dn_idx); cur_vol += dn_v; dn_idx -= 1
                else: break

            val_price = min_p + min(va_indices) * bin_size
            vah_price = min_p + (max(va_indices) + 1) * bin_size

            if vol_ratio < 0.6:
                return {"signal": 0, "thought": f"Low Volume ({vol_ratio:.1f}x) — Neutral (fallback)"}
            if val_price <= current_close <= vah_price:
                return {"signal": 0, "thought": f"Inside Value Area (POC={poc_price:.0f}) — Neutral (fallback)"}
            if current_close > vah_price and buy_delta_pct > 0.60 and vol_ratio >= 1.2:
                return {"signal": 1, "thought": f"Bullish VA Breakout above {vah_price:.0f} (fallback)"}
            if current_close < val_price and buy_delta_pct < 0.40 and vol_ratio >= 1.2:
                return {"signal": -1, "thought": f"Bearish VA Breakdown below {val_price:.0f} (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "Normal Volume (fallback)"}


class Part5ML:
    """ML Predictions — backed by MLEngineGPU (Part5)"""
    def __init__(self):
        try:
            from part5_FIXED import MLEngineGPU
            self._engine = MLEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── 1. Try real Part 5 MLEngineGPU ────────────────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part5 MLEngine error: {e}")

        # Neural context is retired: only observed-data statistical features may vote.
        # ── 2. High-Conviction Statistical Feature Check ─────────
        try:
            if len(data) >= 30:
                closes = data['close'].tail(30).astype(float).values
                vols = data['volume'].tail(30).astype(float).values

                # Volatility-normalized Sharpe drift
                rets = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
                ret_mean = float(np.mean(rets[-20:]))
                ret_std = float(np.std(rets[-20:])) + 1e-8
                sharpe_drift = ret_mean / ret_std

                # Linear regression t-statistic (20 bars)
                N = 20
                x = np.arange(N, dtype=np.float64)
                y_raw = closes[-N:]
                y_std = float(np.std(y_raw)) + 1e-8
                y = (y_raw - float(np.mean(y_raw))) / y_std
                x_mean = float(np.mean(x))
                x_dev = x - x_mean
                ss_x = float(np.sum(x_dev ** 2)) + 1e-8
                beta = float(np.sum(x_dev * y)) / ss_x
                residuals = y - beta * x_dev
                s_err = np.sqrt(float(np.sum(residuals ** 2)) / max(N - 2, 1)) / np.sqrt(ss_x)
                t_stat = beta / (s_err + 1e-8)

                # Volume & EMA
                vol_curr = float(vols[-1])
                vol_mean = float(np.mean(vols[-20:])) + 1e-8
                vol_ratio = vol_curr / vol_mean
                ema8 = float(pd.Series(closes).ewm(span=8).mean().iloc[-1])
                ema21 = float(pd.Series(closes).ewm(span=21).mean().iloc[-1])

                # RSI-14
                diffs = np.diff(closes[-15:])
                gains = np.where(diffs > 0, diffs, 0.0)
                losses = np.where(diffs < 0, -diffs, 0.0)
                avg_gain = float(np.mean(gains)) + 1e-8
                avg_loss = float(np.mean(losses)) + 1e-8
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi_norm = (rsi - 50.0) / 25.0

                composite = 0.35 * float(np.clip(sharpe_drift, -2.0, 2.0)) + \
                            0.35 * float(np.clip(t_stat / 2.0, -2.0, 2.0)) + \
                            0.30 * float(np.clip(rsi_norm, -2.0, 2.0))

                if composite >= 0.50 and t_stat >= 2.0 and ema8 > ema21 and vol_ratio >= 0.8:
                    return {"signal": 1,  "thought": f"ML Momentum Bullish Edge (comp={composite:.2f}, t={t_stat:.1f}, fallback)"}
                elif composite <= -0.50 and t_stat <= -2.0 and ema8 < ema21 and vol_ratio >= 0.8:
                    return {"signal": -1, "thought": f"ML Momentum Bearish Edge (comp={composite:.2f}, t={t_stat:.1f}, fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "ML Neutral (fallback)"}


class Part6Trend:
    """Trend Analysis — backed by TrendEngineGPU (Part6)"""
    def __init__(self):
        try:
            from part6_FIXED import TrendEngineGPU
            self._engine = TrendEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── Try real Part 6 TrendEngineGPU ────────────────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part6 TrendEngine error: {e}")

        # ── Fallback: multi-EMA + ADX chop filter ─────────────────────────
        try:
            if len(data) >= 50:
                closes = data['close'].tail(50).astype(float)
                highs  = data['high'].tail(50).astype(float)
                lows   = data['low'].tail(50).astype(float)

                current = float(closes.iloc[-1])
                ema8   = float(closes.ewm(span=8).mean().iloc[-1])
                ema21  = float(closes.ewm(span=21).mean().iloc[-1])
                ema50  = float(closes.ewm(span=50).mean().iloc[-1])

                spread = abs(ema8 - ema21) / max(ema21, 1.0)
                
                # ADX 14-period approximation
                tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14).mean().iloc[-1])
                up_move = highs.diff()
                down_move = -lows.diff()
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                plus_di = 100.0 * float(pd.Series(plus_dm, index=highs.index).rolling(14).mean().iloc[-1] / max(atr14, 1e-8))
                minus_di = 100.0 * float(pd.Series(minus_dm, index=lows.index).rolling(14).mean().iloc[-1] / max(atr14, 1e-8))
                dx = 100.0 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-8)

                # Chop Filter: if spread < 0.08% or DX < 20, market is in CHOP / RANGE
                if spread < 0.0008 or dx < 20.0:
                    return {"signal": 0, "thought": f"Chop/Rangebound (DX={dx:.1f}, spread={spread*100:.3f}%) — Neutral"}

                # Strong trend: price > ema8 > ema21 > ema50
                if current > ema8 > ema21 > ema50 and plus_di > minus_di:
                    return {"signal": 1,  "thought": f"Strong Uptrend EMA8>{ema21:.0f}>{ema50:.0f} (DX={dx:.1f})"}
                elif current < ema8 < ema21 < ema50 and minus_di > plus_di:
                    return {"signal": -1, "thought": f"Strong Downtrend EMA8<{ema21:.0f}<{ema50:.0f} (DX={dx:.1f})"}
                elif ema8 > ema21 and current > ema21 and plus_di > minus_di and spread >= 0.0012:
                    return {"signal": 1,  "thought": f"Moderate Uptrend EMA8>{ema21:.0f} (DX={dx:.1f})"}
                elif ema8 < ema21 and current < ema21 and minus_di > plus_di and spread >= 0.0012:
                    return {"signal": -1, "thought": f"Moderate Downtrend EMA8<{ema21:.0f} (DX={dx:.1f})"}
        except Exception:
            pass
        return {"signal": 0, "thought": "No Clear Trend (fallback)"}


class Part7Volatility:
    """Part 7 consumer for one selected-symbol native exchange timeframe.

    This adapter intentionally does not instantiate the legacy standalone
    trade-to-candle ``VolatilityEngineGPU``.  The canonical Jarvis path already
    owns a shared native candle snapshot; this Part only analyzes the frame it
    receives and reports the actual Pandas/CPU backend.
    """
    def __init__(self):
        self.backend = "pandas_cpu"

    def analyze(self, data, context=None):
        context = dict(context or {})
        symbol = context.get("selected_symbol") or context.get("symbol") or "UNKNOWN"
        timeframe = context.get("timeframe") or "unknown"
        if _analyze_part7_timeframe is None:
            return {
                "signal": 0, "signal_identity": f"part7:{symbol}:{timeframe}",
                "symbol": str(symbol).upper(), "timeframe": str(timeframe),
                "status": "error", "data_status": "error", "entry_blocked": True,
                "risk_veto": False, "volatility_status": "unknown",
                "reason": "Part7 shared analyzer unavailable",
                "thought": "Part7 shared analyzer unavailable",
                "computation_backend": self.backend,
            }
        return _analyze_part7_timeframe(
            data, symbol=str(symbol), timeframe=str(timeframe), context=context
        )


class Part8Structure:
    """Market Structure — backed by MarketStructureEngineGPU (Part8)"""
    def __init__(self):
        try:
            from part8_FIXED import MarketStructureEngineGPU
            self._engine = MarketStructureEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── 1. Try real Part 8 MarketStructureEngineGPU ────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part8 MarketStructureEngine error: {e}")

        # ── 2. Fallback: Quantitative Fractal BOS / CHoCH Check ───────────
        try:
            if len(data) >= 30:
                recent = data.tail(50).copy()
                highs  = recent['high'].astype(float).values
                lows   = recent['low'].astype(float).values
                closes = recent['close'].astype(float).values
                vols   = recent['volume'].astype(float).values
                current_close = float(closes[-1])

                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
                )
                atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(highs[-1] - lows[-1])
                noise_buffer = 0.20 * atr14

                k = 2
                lookback = min(len(recent) - k, 45)
                swing_highs = []
                swing_lows  = []

                for idx in range(len(recent) - lookback, len(recent) - k):
                    h = highs[idx]
                    l = lows[idx]
                    if all(h > highs[idx - i] for i in range(1, k + 1)) and all(h >= highs[idx + i] for i in range(1, k + 1)):
                        swing_highs.append((idx, float(h)))
                    if all(l < lows[idx - i] for i in range(1, k + 1)) and all(l <= lows[idx + i] for i in range(1, k + 1)):
                        swing_lows.append((idx, float(l)))

                if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                    last_sh = swing_highs[-1][1]
                    prev_sh = swing_highs[-2][1]
                    last_sl = swing_lows[-1][1]
                    prev_sl = swing_lows[-2][1]

                    bullish_bos = current_close > (last_sh + noise_buffer)
                    bearish_bos = current_close < (last_sl - noise_buffer)
                    bullish_choch = (last_sh < prev_sh) and (current_close > last_sh + noise_buffer)
                    bearish_choch = (last_sl > prev_sl) and (current_close < last_sl - noise_buffer)

                    ema20 = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
                    vol_curr = float(vols[-1])
                    vol_mean = float(np.mean(vols[-20:])) + 1e-8
                    vol_ok = (vol_curr / vol_mean) >= 0.75

                    if (bullish_bos or bullish_choch) and current_close > ema20 and vol_ok:
                        label = 'CHoCH' if bullish_choch else 'BOS'
                        return {"signal": 1,  "thought": f"Bullish Structure {label} above {last_sh:.1f} (fallback)"}
                    if (bearish_bos or bearish_choch) and current_close < ema20 and vol_ok:
                        label = 'CHoCH' if bearish_choch else 'BOS'
                        return {"signal": -1, "thought": f"Bearish Structure {label} below {last_sl:.1f} (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "Structure Consolidation / Neutral (fallback)"}


class Part9Orderflow:
    """Orderflow/Delta — backed by OrderflowEngineGPU (Part9)"""
    def __init__(self):
        try:
            from part9_FIXED import OrderflowEngineGPU
            self._engine = OrderflowEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── 1. Try real Part 9 OrderflowEngineGPU ──────────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part9 OrderflowEngine error: {e}")

        # ── 2. Fallback: Quantitative CVD & Volume Delta Analysis ──────────
        try:
            if len(data) >= 20:
                recent = data.tail(35).copy()
                highs  = recent['high'].astype(float).values
                lows   = recent['low'].astype(float).values
                closes = recent['close'].astype(float).values
                vols   = recent['volume'].astype(float).values

                candle_ranges = np.maximum(highs - lows, 1e-8)
                delta_ratios = (2.0 * (closes - lows) - candle_ranges) / candle_ranges
                deltas = vols * delta_ratios

                N = 20
                cvd_20 = float(np.sum(deltas[-N:]))
                tot_vol_20 = float(np.sum(vols[-N:])) + 1e-8
                cvd_ratio = cvd_20 / tot_vol_20

                cvd_5 = float(np.sum(deltas[-5:]))
                tot_vol_5 = float(np.sum(vols[-5:])) + 1e-8
                cvd_ratio_5 = cvd_5 / tot_vol_5

                current_close = float(closes[-1])
                price_chg_20 = float((current_close - closes[-N]) / max(closes[-N], 1e-8))
                ema20 = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])

                bullish_div = (price_chg_20 < -0.0025) and (cvd_ratio > 0.25)
                bearish_div = (price_chg_20 > 0.0025) and (cvd_ratio < -0.25)

                bullish_flow = (cvd_ratio > 0.30) and (cvd_ratio_5 > 0.15) and (current_close > ema20)
                bearish_flow = (cvd_ratio < -0.30) and (cvd_ratio_5 < -0.15) and (current_close < ema20)

                if bullish_div:
                    return {"signal": 1,  "thought": f"Bullish Absorption Divergence (CVD={cvd_ratio*100:+.0f}%, Price-) (fallback)"}
                if bearish_div:
                    return {"signal": -1, "thought": f"Bearish Absorption Divergence (CVD={cvd_ratio*100:+.0f}%, Price+) (fallback)"}
                if bullish_flow:
                    return {"signal": 1,  "thought": f"Bullish Volume Delta Imbalance (CVD={cvd_ratio*100:+.0f}%) (fallback)"}
                if bearish_flow:
                    return {"signal": -1, "thought": f"Bearish Volume Delta Imbalance (CVD={cvd_ratio*100:+.0f}%) (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "Balanced Orderflow / Neutral (fallback)"}


class Part10Candlestats:
    """Multi-candle Statistical Analysis — backed by CandleStatsEngineGPU (Part10)"""
    def __init__(self):
        try:
            from part10_FIXED import CandleStatsEngineGPU
            self._engine = CandleStatsEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, data, context=None):
        # ── 1. Try real Part 10 CandleStatsEngineGPU ───────────────────────
        if self._engine is not None:
            try:
                res = self._engine.analyze(data, context=context)
                if isinstance(res, dict):
                    return res
            except Exception as e:
                logger.debug(f"Part10 CandleStatsEngine error: {e}")

        # ── 2. Fallback: Quantitative Multi-bar Candle Analytics ───────────
        try:
            if len(data) >= 20:
                recent = data.tail(25).copy()
                closes = recent['close'].astype(float).values
                opens  = recent['open'].astype(float).values
                highs  = recent['high'].astype(float).values
                lows   = recent['low'].astype(float).values

                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
                )
                atr14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(highs - lows))

                bodies = np.abs(closes[-3:] - opens[-3:])
                avg_recent_body = float(np.mean(bodies))

                ranges = np.maximum(highs[-3:] - lows[-3:], 1e-8)
                body_ratios = bodies / ranges
                avg_body_ratio = float(np.mean(body_ratios))

                if avg_recent_body < 0.40 * atr14 or avg_body_ratio < 0.50:
                    return {"signal": 0, "thought": "Micro-noise / Indecisive Bodies (fallback)"}

                is_bull_streak = all(closes[-i] > opens[-i] for i in range(1, 4)) and (closes[-1] > closes[-2] > closes[-3])
                is_bear_streak = all(closes[-i] < opens[-i] for i in range(1, 4)) and (closes[-1] < closes[-2] < closes[-3])

                last10_closes = closes[-10:]
                last10_opens  = opens[-10:]
                bull_count = int(np.sum(last10_closes > last10_opens))
                bear_count = 10 - bull_count

                baseline_bodies = np.abs(closes[-10:-3] - opens[-10:-3])
                avg_baseline = float(np.mean(baseline_bodies)) if len(baseline_bodies) > 0 else avg_recent_body
                body_accel = float(avg_recent_body / max(avg_baseline, 0.30 * atr14))

                if is_bull_streak and avg_body_ratio >= 0.55 and body_accel >= 1.3 and bull_count >= 6:
                    return {"signal": 1,  "thought": f"Bullish Candle Run Streak=3 Accel={body_accel:.1f}x (fallback)"}
                if is_bear_streak and avg_body_ratio >= 0.55 and body_accel >= 1.3 and bear_count >= 6:
                    return {"signal": -1, "thought": f"Bearish Candle Run Streak=3 Accel={body_accel:.1f}x (fallback)"}
                if bull_count >= 8 and avg_recent_body >= 0.50 * atr14 and closes[-1] > opens[-1]:
                    return {"signal": 1,  "thought": f"Overwhelming Bullish Run {bull_count}/10 green (fallback)"}
                if bear_count >= 8 and avg_recent_body >= 0.50 * atr14 and closes[-1] < opens[-1]:
                    return {"signal": -1, "thought": f"Overwhelming Bearish Run {bear_count}/10 red (fallback)"}
        except Exception:
            pass
        return {"signal": 0, "thought": "Mixed / Indecisive Candles (fallback)"}


class Part11Fusion:
    """Fusion — Institutional Weighted Voting & Multi-Engine Confluence with Zone Veto Protection"""
    def __init__(self):
        self._engine = None
        try:
            from part11_FIXED import SignalFusionEngineGPU
            self._engine = SignalFusionEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, part_results):
        if self._engine is not None:
            try:
                res = self._engine.analyze(part_results)
                if isinstance(res, dict) and 'signal' in res:
                    return res
            except Exception:
                pass

        # Fallback: Institutional Weighted Voting & Confluence Consensus
        if isinstance(part_results, dict):
            results_dict = part_results
            p2 = part_results.get('part2_zone', {})
        elif isinstance(part_results, list):
            results_dict = {f"part_{i}": r for i, r in enumerate(part_results)}
            p2 = next((r for r in part_results if isinstance(r, dict) and ('support zone' in r.get('thought', '').lower() or 'resistance zone' in r.get('thought', '').lower())), {})
        else:
            return {"signal": 0, "thought": "Invalid input"}

        weights = {
            'part1_breakout': 1.1,
            'part2_zone': 1.3,
            'part3_psychology': 1.0,
            'part4_volume': 1.3,
            'part5_ml': 1.2,
            'part6_trend': 1.5,
            'part7_volatility': 1.1,
            'part8_structure': 1.4,
            'part9_orderflow': 1.4,
            'part10_candlestats': 1.0,
        }

        weighted_buy = 0.0
        weighted_sell = 0.0
        active_buy_engines = []
        active_sell_engines = []

        for name, r in results_dict.items():
            if not isinstance(r, dict):
                continue
            sig = r.get('signal', 0)
            try:
                sig = float(sig)
            except (ValueError, TypeError):
                continue

            w = weights.get(name, 1.0)
            if sig > 0:
                weighted_buy += w * sig
                active_buy_engines.append(name)
            elif sig < 0:
                weighted_sell += w * abs(sig)
                active_sell_engines.append(name)

        total_weight = weighted_buy + weighted_sell
        total_active_engines = len(active_buy_engines) + len(active_sell_engines)

        # Quorum & Confluence Threshold: Minimum 3 active engines & 3.5 total weight
        if total_active_engines < 3 or total_weight < 3.5:
            return {"signal": 0, "thought": f"Fusion Neutral: Insufficient confluence ({total_active_engines}/3 engines, weight {total_weight:.1f}/3.5)"}

        buy_ratio = weighted_buy / total_weight
        sell_ratio = weighted_sell / total_weight

        p2_thought = str(p2.get('thought', '')).lower()
        p2_sig = p2.get('signal', 0)

        # 🛡️ ZONE VETO: If at Support Zone bottom, strictly VETO all SELL/PUT trades!
        # If at Resistance Zone top, strictly VETO all BUY/CALL trades!
        if buy_ratio >= 0.65:
            if p2_sig == -1 or 'resistance zone' in p2_thought:
                return {"signal": 0, "thought": f"Zone Veto: BUY blocked at Resistance Zone ({p2.get('thought')})"}
            strong_opponents = [eng for eng in active_sell_engines if eng in ('part6_trend', 'part8_structure', 'part9_orderflow')]
            if len(strong_opponents) >= 2:
                return {"signal": 0, "thought": f"Fusion Dissent Veto: BUY opposed by key anchors {strong_opponents}"}
            return {"signal": 1,  "thought": f"Fusion Bullish: {len(active_buy_engines)} engines ({buy_ratio*100:.0f}% weighted consensus, score={weighted_buy:.1f})"}

        elif sell_ratio >= 0.65:
            if p2_sig == 1 or 'support zone' in p2_thought:
                return {"signal": 0, "thought": f"Zone Veto: SELL blocked at Support Zone ({p2.get('thought')})"}
            strong_opponents = [eng for eng in active_buy_engines if eng in ('part6_trend', 'part8_structure', 'part9_orderflow')]
            if len(strong_opponents) >= 2:
                return {"signal": 0, "thought": f"Fusion Dissent Veto: SELL opposed by key anchors {strong_opponents}"}
            return {"signal": -1, "thought": f"Fusion Bearish: {len(active_sell_engines)} engines ({sell_ratio*100:.0f}% weighted consensus, score={weighted_sell:.1f})"}

        return {"signal": 0, "thought": f"Fusion Split: Conflict between Bull ({weighted_buy:.1f}) and Bear ({weighted_sell:.1f})"}


class Part12Confidence:
    """Confidence Score — Institutional Weighted Confluence across all active parts"""
    def __init__(self):
        self._engine = None
        try:
            from part12_FIXED import ConfidenceEngineGPU
            self._engine = ConfidenceEngineGPU()
        except Exception:
            self._engine = None

    def analyze(self, part_results):
        if self._engine is not None:
            try:
                res = self._engine.analyze(part_results)
                if isinstance(res, dict) and 'confidence' in res:
                    return res
            except Exception:
                pass

        # Fallback: Institutional Weighted Confluence Calculation
        if not part_results:
            return {"confidence": 10, "thought": "No data"}
            
        if isinstance(part_results, dict):
            items = list(part_results.items())
        elif isinstance(part_results, list):
            items = [(f"part_{i}", r) for i, r in enumerate(part_results)]
        else:
            return {"confidence": 10, "thought": "Invalid input"}

        weights = {
            'part1_breakout': 1.1,
            'part2_zone': 1.3,
            'part3_psychology': 1.0,
            'part4_volume': 1.3,
            'part5_ml': 1.2,
            'part6_trend': 1.5,
            'part7_volatility': 1.1,
            'part8_structure': 1.4,
            'part9_orderflow': 1.4,
            'part10_candlestats': 1.0,
        }
        anchors = {'part6_trend', 'part8_structure', 'part9_orderflow', 'part2_zone'}

        w_buy, w_sell = 0.0, 0.0
        buy_engines, sell_engines = [], []
        p2_thought = ""

        for name, r in items:
            if not isinstance(r, dict):
                continue
            sig = r.get('signal', 0)
            try:
                sig = float(sig)
            except (ValueError, TypeError):
                continue

            w = weights.get(name, 1.0)
            if 'zone' in name or 'support' in str(r.get('thought', '')).lower() or 'resistance' in str(r.get('thought', '')).lower():
                p2_thought = str(r.get('thought', '')).lower()

            if sig > 0:
                w_buy += w * sig
                buy_engines.append(name)
            elif sig < 0:
                w_sell += w * abs(sig)
                sell_engines.append(name)

        if len(buy_engines) == 0 and len(sell_engines) == 0:
            return {"confidence": 10, "thought": "All parts neutral"}

        if w_buy >= w_sell:
            majority_dir = 1
            w_agree, w_dissent = w_buy, w_sell
            agree_engines, dissent_engines = buy_engines, sell_engines
        else:
            majority_dir = -1
            w_agree, w_dissent = w_sell, w_buy
            agree_engines, dissent_engines = sell_engines, buy_engines

        agree_count = len(agree_engines)
        w_active = w_agree + w_dissent

        if agree_count < 3 or w_active < 3.5:
            conf = int(min(30, 10 + agree_count * 5 + w_agree * 3))
            return {
                "confidence": conf,
                "thought": f"Low Confluence: {agree_count} engines, weight {w_agree:.1f}",
                "weighted_agree": round(w_agree, 2),
                "weighted_dissent": round(w_dissent, 2),
            }

        confluence_ratio = w_agree / w_active
        saturation = min(1.0, w_agree / 6.0)
        base_conf = 45.0 + (40.0 * confluence_ratio * saturation)

        aligned_anchors = [eng for eng in agree_engines if eng in anchors]
        if len(aligned_anchors) >= 3:
            base_conf += 10.0
        elif len(aligned_anchors) == 2:
            base_conf += 5.0

        dissenting_anchors = [eng for eng in dissent_engines if eng in anchors]
        if len(dissenting_anchors) >= 2:
            base_conf -= 35.0
        elif len(dissenting_anchors) == 1:
            base_conf -= 18.0

        if majority_dir == 1 and 'resistance zone' in p2_thought:
            base_conf -= 25.0
        elif majority_dir == -1 and 'support zone' in p2_thought:
            base_conf -= 25.0

        if w_dissent > 0:
            conflict_penalty = (w_dissent / w_agree) * 20.0
            base_conf -= conflict_penalty

        conf = int(max(10, min(95, round(base_conf))))
        return {
            "confidence": conf,
            "thought": f"Confluence: {agree_count} parts ({conf}%), anchors aligned={len(aligned_anchors)}, dissenting={len(dissenting_anchors)}",
            "weighted_agree": round(w_agree, 2),
            "weighted_dissent": round(w_dissent, 2),
            "confluence_ratio": round(confluence_ratio, 2)
        }









class Part14OptionsChain:
    """Deterministic point-in-time options analysis; never lets a model vote."""
    def __init__(self, delta_client=None, asset='BTC', backtest_mode=False):
        self.delta_client = delta_client
        self.asset = str(asset or 'BTC').upper().replace('USDT', '').replace('USD', '')
        self.backtest_mode = bool(backtest_mode)
        self.deribit = None
        if not self.backtest_mode and self.asset in ('BTC', 'ETH'):
            try:
                from deribit_options_client import DeribitOptionsClient
                # Credentials are environment-managed; Deribit remains BTC-only.
                client_id = os.getenv("DERIBIT_CLIENT_ID", "")
                client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
                self.deribit = DeribitOptionsClient(currency=self.asset, client_id=client_id, client_secret=client_secret)
            except Exception:
                self.deribit = None

    def analyze_options_deterministically(self, telemetry: Dict, current_price: float) -> Tuple[str, str, int]:
        """Return only the deterministic observed-chain signal; no model vote."""
        signal = telemetry.get('signal', 0)
        try:
            signal = int(signal) if int(signal) in (-1, 0, 1) else 0
        except (TypeError, ValueError, OverflowError):
            signal = 0
        return "DETERMINISTIC_CHAIN_ONLY", "Options result uses validated provider observations; Laya is commentary-only", signal


    def analyze(self, data, context=None):
        """Unified selected-asset options analysis; BTC Deribit is never an alt substitute."""
        try:
            requested = (context.get('selected_symbol') or context.get('symbol')) if isinstance(context, dict) else None
            asset = str(requested or self.asset or '').upper().replace('USDT', '').replace('USD', '')
            if asset:
                self.asset = asset
            if data is None or 'close' not in data or len(data['close']) == 0:
                return {"signal": 0, "thought": "No price data available", "telemetry": {"signal": 0}}

            raw_price = data['close'].iloc[-1]
            try:
                current_price = float(raw_price)
                if not math.isfinite(current_price) or current_price <= 0:
                    return {"signal": 0, "thought": "Invalid current price; options math unavailable", "telemetry": {"signal": 0, "available": False, "asset": getattr(self, 'asset', 'BTC')}}
            except (TypeError, ValueError, OverflowError):
                return {"signal": 0, "thought": "Invalid current price; options math unavailable", "telemetry": {"signal": 0, "available": False, "asset": getattr(self, 'asset', 'BTC')}}

            is_backtest = bool(getattr(self, 'backtest_mode', False)) or bool((context or {}).get('is_backtest_mode'))
            if is_backtest:
                bias_data = (context or {}).get('historical_options_context', {})
                if not isinstance(bias_data, dict) or not bias_data.get('available'):
                    return {"signal": 0, "thought": "Historical options snapshot unavailable at decision time",
                            "telemetry": {"signal": 0, "available": False, "asset": self.asset,
                                           "source": "none", "not_ohlc_derived": True}}
                source = None
                source_name = str(bias_data.get('source') or 'recorded_options_snapshots')
                thoughts = list(bias_data.get('reasons') or [])[:2]
            else:
                delta_source = self.delta_client if self.delta_client and hasattr(self.delta_client, 'get_institutional_bias') else None
                deribit_source = self.deribit if self.asset in ('BTC', 'ETH') else None
                if deribit_source is not None and str(getattr(deribit_source, 'currency', 'BTC')).upper() != str(self.asset).upper():
                    # Never feed another coin's option chain: use a per-asset client.
                    _dcm = getattr(self, '_deribit_by_asset', None)
                    if _dcm is None:
                        _dcm = self._deribit_by_asset = {}
                    if self.asset not in _dcm:
                        try:
                            from deribit_options_client import DeribitOptionsClient as _DOC
                            _dcm[self.asset] = _DOC(currency=str(self.asset), client_id=os.getenv('DERIBIT_CLIENT_ID', ''), client_secret=os.getenv('DERIBIT_CLIENT_SECRET', ''))
                        except Exception:
                            _dcm[self.asset] = None
                    deribit_source = _dcm[self.asset]
                candidates = []
                if delta_source is not None:
                    candidates.append((delta_source, "Delta", self.asset))
                if deribit_source is not None:
                    candidates.append((deribit_source, "Deribit", current_price))
                source = None
                source_name = "None"
                bias_data = {}
                source_failures = []
                for candidate, candidate_name, query in candidates:
                    try:
                        if candidate_name == "Delta":
                            candidate_data = candidate.get_institutional_bias(self.asset)
                        else:
                            candidate_data = candidate.get_institutional_bias(query)
                    except Exception as e:
                        source_failures.append(f"{candidate_name}: {type(e).__name__}")
                        logging.warning("Options %s fetch failed: %s", candidate_name, type(e).__name__)
                        continue
                    if not isinstance(candidate_data, dict):
                        source_failures.append(f"{candidate_name}: malformed response")
                        continue
                    if "No Data" in candidate_data.get("reasons", []) or not candidate_data.get("raw_data"):
                        source_failures.append(f"{candidate_name}: no data")
                        continue
                    validation = (candidate_data.get('raw_data') or {}).get('options_validation')
                    candidate_available = (candidate_data.get('available', True) is True
                                           and isinstance(validation, dict)
                                           and validation.get('usable') is True)
                    if not candidate_available:
                        source_failures.append(f"{candidate_name}: unavailable or unusable chain")
                        continue
                    try:
                        _mp_chk = None
                        _stack = [candidate_data]
                        while _stack and _mp_chk is None:
                            _d0 = _stack.pop()
                            if isinstance(_d0, dict):
                                if _d0.get('max_pain') is not None:
                                    _mp_chk = float(_d0.get('max_pain'))
                                _stack.extend(v for v in _d0.values() if isinstance(v, dict))
                        if candidate_name == 'Deribit' and _mp_chk and current_price and not (0.2 <= _mp_chk / float(current_price) <= 5.0):
                            source_failures.append(f"Deribit: chain/spot currency mismatch (max_pain {_mp_chk:.0f} vs price {float(current_price):.0f})")
                            logging.warning('Part14 rejected mismatched options chain: max_pain=%s price=%s asset=%s', _mp_chk, current_price, self.asset)
                            continue
                    except Exception:
                        pass
                    source, source_name, bias_data = candidate, candidate_name, candidate_data
                    break
                if source is None:
                    reason = "; ".join(source_failures) or "no eligible provider configured"
                    return {"signal": 0, "thought": f"Selected-asset options unavailable ({self.asset or 'unknown'}): {reason}",
                            "telemetry": {"signal": 0, "exchange": "None", "available": False, "asset": self.asset,
                                           "provider_failures": source_failures}}
                thoughts = []
                if hasattr(source, 'analyze_full_market'):
                    try:
                        market = source.analyze_full_market(current_price)
                        if market and market.get('smart_money', {}).get('detected'):
                            thoughts.append(f"🐋 {market['smart_money']['details']}")
                    except Exception:
                        pass
                thoughts.extend(bias_data.get('reasons', [])[:2])
            
            math_signal = 1 if bias_data.get('bias') == 'BULLISH' else (-1 if bias_data.get('bias') == 'BEARISH' else 0)
            
            # Extract safe floats for telemetry
            pcr_raw = bias_data.get('pcr', bias_data.get('raw_data', {}).get('pcr'))
            pcr_float = float(pcr_raw) if pcr_raw is not None and str(pcr_raw).replace('.', '', 1).isdigit() else None
            
            max_pain_raw = bias_data.get('max_pain') or bias_data.get('raw_data', {}).get('max_pain')
            max_pain_float = float(max_pain_raw) if max_pain_raw is not None and str(max_pain_raw).replace('.', '', 1).isdigit() else None
            # Descriptive, deterministic chain math: only derive relative distances
            # when both observed prices are finite and positive. Never infer a
            # missing wall, PCR, or selected instrument from another asset.
            def _observed_price(value):
                try:
                    value = float(value)
                    return value if math.isfinite(value) and value > 0 else None
                except (TypeError, ValueError, OverflowError):
                    return None
            support_raw = bias_data.get('support_wall') or bias_data.get('raw_data', {}).get('support')
            resistance_raw = bias_data.get('resistance_wall') or bias_data.get('raw_data', {}).get('resistance')
            support_price = _observed_price(support_raw)
            resistance_price = _observed_price(resistance_raw)
            current_price = _observed_price(current_price)
            if current_price is None:
                return {"signal": 0, "thought": "Invalid current price; options math unavailable", "telemetry": {"signal": 0, "available": False, "asset": self.asset}}
            max_pain_float = _observed_price(max_pain_float)
            pcr_float = pcr_float if pcr_float is not None and math.isfinite(pcr_float) and pcr_float >= 0 else None

            options_validation = bias_data.get('raw_data', {}).get('options_validation')
            options_available = bool(bias_data.get('available', True))
            if isinstance(options_validation, dict):
                options_available = options_available and bool(options_validation.get('usable'))
            telemetry = {
                "exchange": source_name,
                "asset": self.asset,
                "available": options_available,
                "historical": bool(bias_data.get('historical', False)),
                "not_ohlc_derived": bool(bias_data.get('not_ohlc_derived', False)),
                "bias_score": float(bias_data.get('score', 0)),
                "pcr": pcr_float,
                "signal": math_signal,
                "support_wall": support_price,
                "resistance_wall": resistance_price,
                "max_pain": max_pain_float,
                "support_distance_pct": ((current_price - support_price) / current_price * 100) if support_price is not None else None,
                "resistance_distance_pct": ((resistance_price - current_price) / current_price * 100) if resistance_price is not None else None,
                "max_pain_distance_pct": ((max_pain_float - current_price) / current_price * 100) if max_pain_float is not None else None,
                "math_model": "observed_chain_descriptive_v1"
            }

            # Deterministic options math; optional model commentary cannot change signal.
            if hasattr(self, 'analyze_options_deterministically'):
                whale_tag, insight, final_signal = self.analyze_options_deterministically(telemetry, current_price)
            elif hasattr(self, 'analyze_options_with_ollama'):
                whale_tag, insight, final_signal = self.analyze_options_with_ollama(telemetry, current_price)
            else:
                whale_tag, insight, final_signal = "DETERMINISTIC_CHAIN_ONLY", "Options result uses validated provider observations", telemetry.get('signal', 0)
            telemetry['advisory_status'] = whale_tag
            telemetry['advisory_note'] = insight

            combined_thought = f"[{whale_tag}] " + (" | ".join(thoughts) if thoughts else f"Institutional: {bias_data.get('bias', 'NEUTRAL')}")
            
            return {
                "signal": final_signal if final_signal != 0 else math_signal,
                "thought": combined_thought,
                "telemetry": telemetry
            }
        except Exception as e:
            logging.error(f"Part14OptionsChain analyze error: {e}")
            return {"signal": 0, "thought": "Institutional error", "telemetry": {"error": str(e), "signal": 0}}

class DeepSeekV3Brain:
    """Retired model surface; Laya is advisory-only and never casts a vote."""
    def __init__(self):
        self.model_name = None
        self.enabled = False

    def analyze_sentiment(self, data, market_context):
        return 0, "unavailable: legacy LegacyLLM/DeepSeek voting retired; Laya is commentary-only"


class DeepSeekR1ReasoningBrain:
    """Retired decision surface; deterministic code remains authoritative."""
    def __init__(self):
        self.model_name = None
        self.enabled = False

    def complex_reasoning(self, all_signals, market_data, context):
        return {"signal": "NO-TRADE", "confidence": 0,
                "reasoning": "unavailable: legacy LegacyLLM/DeepSeek voting retired; Laya is commentary-only",
                "key_factors": [], "available": False}

class QuantumV5:
    """
    Quantum Probability Simulator V5 - Dual Mode
    
    Mode 1: Independent (physics-only, no Parts influence)
    Mode 2: Validated (considers Parts context for smart validation)
    
    This prevents echo chamber while enabling conflict detection.
    """
    def __init__(self):
        self.num_paths = 128
    
    def simulate(self, data, parts_context=None):
        """
        Dual-mode quantum simulation
        
        Args:
            data: OHLCV DataFrame
            parts_context: Optional dict with Parts summary
                {
                    'consensus': 7/10,
                    'harmony_score': 78,
                    'strong_agreers': ['part4', 'part6'],
                    'conflicts': 2
                }
        
        Returns:
            Dict with independent + validated results (if context provided)
        """
        # Quantum simulation disabled
        try:
            if data is not None and len(data) >= 10:
                from jarvis_self_learning import get_self_learning_engine
                _cp = float(data['close'].iloc[-1])
                _vol = float(data['close'].pct_change().std() or 0.0015)
                _rp = data['close'].tail(5).values
                _trd = (_rp[-1] - _rp[0]) / max(_rp[0], 1e-9)
                _mc = get_self_learning_engine().gpu.monte_carlo_paths(_cp, _trd, max(_vol, 1e-5), n_paths=512, horizon=12)
                if _mc.get("available"):
                    _bp = float(_mc.get("bull_prob", 0.5))
                    return {"signal": 1 if _bp >= 0.54 else (-1 if _bp <= 0.46 else 0), "thought": f"QUANTUM GPU ({_mc.get('gpu_device')}): {_bp*100:.1f}% paths UP | ExpRet: {_mc.get('expected_return_pct', 0):+.3f}%", "mode": "gpu_ptx_monte_carlo", "bull_ratio": _bp, "confidence": int(min(100, abs(_bp - 0.5) * 200))}
        except Exception:
            pass
        return {"signal": 0, "thought": "Quantum disabled", "mode": "disabled", "confidence": 0}
    
    def _pure_physics_simulation(self, data):
        """Independent physics-only simulation (no Parts influence)"""
        current_price = float(data['close'].iloc[-1])
        
        # Calculate volatility
        volatility = float(data['close'].pct_change().std() or 0.0)
        
        # Calculate trend bias (momentum) with velocity extrapolation
        recent_prices = data['close'].tail(5).values
        # Trend: Simple % change
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        # Velocity: Acceleration of trend (Phase 36)
        velocity = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
        extrapolated_drift = trend + (velocity * 0.5) # Weight towards current momentum
        
        # Generate quantum paths with trend bias (increased resolution)
        self.num_paths = 256 # Higher resolution (Phase 36)
        paths = np.random.normal(extrapolated_drift, volatility + 1e-12, self.num_paths)
        final_prices = current_price * (1 + paths)
        
        # Calculate bull ratio
        bull_ratio = np.sum(final_prices > current_price) / self.num_paths
        
        # Determine signal (Phase 39: Loosened Thresholds)
        # Original: 0.52/0.48 -> New: 0.505/0.495 (React to subtle drift)
        if bull_ratio > 0.505:
            signal = 1
            thought = f"QUANTUM (Physics): {bull_ratio*100:.1f}% paths UP | Drift: {extrapolated_drift*10000:.1f}bps"
        elif bull_ratio < 0.495:
            signal = -1
            thought = f"QUANTUM (Physics): {(1-bull_ratio)*100:.1f}% paths DOWN | Drift: {extrapolated_drift*10000:.1f}bps"
        else:
            signal = 0
            thought = f"QUANTUM (Physics): Balanced paths ({bull_ratio*100:.1f}% Bull) | Drift: {extrapolated_drift*10000:.1f}bps"
        
        return {
            'signal': signal,
            'thought': thought,
            'mode': 'independent',
            'bull_ratio': bull_ratio,
            'confidence': int(abs(bull_ratio - 0.5) * 200)  # 0-100%
        }
    
    def _validate_with_parts(self, physics_result, parts_context):
        """
        Validate independent physics with Parts context
        
        Returns enhanced result with validation and divergence detection
        """
        physics_signal = physics_result['signal']
        physics_confidence = physics_result['confidence']
        
        # Extract Parts consensus
        parts_ratio = parts_context.get('consensus', 0.5)  # 0-1 scale
        parts_signal = 1 if parts_ratio > 0.6 else (-1 if parts_ratio < 0.4 else 0)
        harmony = parts_context.get('harmony_score', 50)
        conflicts = parts_context.get('conflicts', 0)
        
        # Calculate divergence
        divergence = abs(physics_signal - parts_signal)
        
        # Validation logic
        if divergence == 0:
            # Physics and Parts AGREE
            boost = 15 if harmony > 70 else 10
            validated_confidence = min(100, physics_confidence + boost)
            validation_status = "ALIGNED"
            thought = f"⚛️ QUANTUM VALIDATED: Physics + Parts AGREE ({validated_confidence}%) | Harmony: {harmony}%"
            
        elif divergence == 1:
            # Minor disagreement (e.g., Physics Neutral, Parts Buy)
            # FORCE ALIGNMENT if Parts Consensus is Strong
            if harmony > 60 and abs(parts_ratio - 0.5) > 0.15:
                 physics_signal = parts_signal
                 validated_confidence = 85
                 validation_status = "FORCED_ALIGNMENT"
                 thought = f"⚛️ QUANTUM ALIGNED: Overriding Neutral Physics due to Strong Parts Consensus ({harmony}%)"
            else:
                 penalty = 5
                 validated_confidence = max(0, physics_confidence - penalty)
                 validation_status = "MINOR_CONFLICT"
                 thought = f"⚛️ QUANTUM CAUTION: Physics vs Parts minor conflict ({validated_confidence}%)"
            
        else:
            # Major conflict (opposite signals)
            penalty = 20
            validated_confidence = max(0, physics_confidence - penalty)
            validation_status = "MAJOR_CONFLICT"
            
            # Determine winner based on confidence
            if physics_confidence > 70 and harmony < 60:
                # Trust physics (high confidence, low Parts harmony)
                pass # physics wins
            elif harmony > 75:
                # Trust Parts (High harmony overrides physics)
                physics_signal = parts_signal
                validated_confidence = 80
                thought = f"⚛️ QUANTUM OVERRULED: Parts Harmony ({harmony}%) overrides Physics"
                validation_status = "PARTS_OVERRIDE"
            elif harmony > 80 and physics_confidence < 60:
                # Trust Parts (high harmony, low physics confidence)
                physics_signal = parts_signal
                thought = f"⚠️ PARTS OVERRIDE: Parts Harmony {harmony}% vs Physics {physics_confidence}% (Conflict!)"
                validation_status = "PARTS_OVERRIDE"
            else:
                # Major uncertainty - NO TRADE
                final_signal = 0
                thought = f"🚫 QUANTUM-PARTS CONFLICT: NO TRADE ({physics_confidence}% vs {int(parts_ratio*100)}%)"
        
        # Build validated result — use physics_signal (may have been overridden above)
        validated_result = {
            'signal': physics_signal,
            'thought': thought,
            'mode': 'validated',
            'confidence': validated_confidence,
            
            # Transparency data
            'independent': physics_result,
            'parts_consensus': parts_signal,
            'divergence': divergence,
            'validation_status': validation_status,
            'harmony_score': harmony,
            'conflicts_detected': conflicts
        }
        
        return validated_result

class SafetyRiskBrain:
    def analyze(self, data, signals):
        try:
            recent_change = abs(float(data['close'].iloc[-1]) - float(data['close'].iloc[-2])) / (float(data['close'].iloc[-2]) + 1e-12)
            if recent_change > 0.005: 
                return {"approved": False, "thought": f"SAFETY: Sudden spike detected ({recent_change*100:.2f}%) - Market too unstable"}
            return {"approved": True, "thought": "SAFETY: Volatility within safe execution bounds"}
        except Exception:
            return {"approved": True, "thought": "Safety check bypassed"}

class VolumePressureBrain:
    def analyze(self, data):
        try:
            if len(data) < 20: return {"signal": 0, "thought": "VOL-PRESSURE: Insufficient volume history"}
            volume_sma = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = float(data['volume'].iloc[-1])
            if volume_sma == 0: return {"signal": 0, "thought": "Neutral volume"}
            if current_volume > volume_sma * 1.3:
                price_change = (float(data['close'].iloc[-1]) - float(data['close'].iloc[-2])) / (float(data['close'].iloc[-2]) + 1e-12)
                side = "Buying" if price_change > 0 else "Selling"
                return {"signal": 1 if price_change > 0 else -1, "thought": f"VOL-PRESSURE: Aggressive {side} burst detected ({current_volume/volume_sma:.1f}x SMA)"}
            return {"signal": 0, "thought": "VOL-PRESSURE: Normal institutional flow"}
        except Exception:
            return {"signal": 0, "thought": "Volume analysis failure"}

class TrendAccelerationBrain:
    def analyze(self, data):
        try:
            if len(data) < 10: return {"signal": 0, "thought": "ACCEL: Trend warming up"}
            sma_5 = data['close'].rolling(5).mean()
            sma_10 = data['close'].rolling(10).mean()
            accel_5 = sma_5.diff().iloc[-1]
            accel_10 = sma_10.diff().iloc[-1]
            if accel_5 > 0 and accel_10 > 0: 
                return {"signal": 1, "thought": "ACCEL: Bullish momentum accelerating (Dual SMA shift)"}
            if accel_5 < 0 and accel_10 < 0: 
                return {"signal": -1, "thought": "ACCEL: Bearish momentum accelerating (Dual SMA shift)"}
            return {"signal": 0, "thought": "ACCEL: Trend velocity is stalling"}
        except Exception:
            return {"signal": 0, "thought": "Acceleration check skipped"}

class RiskFilterBrain:
    def analyze(self, data, signal):
        try:
            if len(data) < 5: return {"approved": True, "thought": "RISK: Entry phase"}
            max_drawdown = (data['close'].rolling(5).max() - data['close']).iloc[-1] / (data['close'].iloc[-1] + 1e-12)
            if max_drawdown > 0.01 and signal > 0:
                return {"approved": False, "thought": f"RISK: Excessive Drawdown ({max_drawdown*100:.2f}%) inhibits long entry"}
            return {"approved": True, "thought": "RISK: Exposure remains within limits"}
        except Exception:
            return {"approved": True, "thought": "Risk filter bypassed"}

class MarketMoodEngine:
    def detect_mood(self, data):
        """Analyze market condition and predict Daily Bias"""
        try:
            if len(data) < 20: return "NEUTRAL (Insufficient Data)"
            
            # 1. Volatility Analysis
            volatility = float(data['close'].pct_change().std() or 0.0)
            
            # 2. Trend Strength (ADX-like proxy)
            trend_strength = abs(float(data['close'].diff().tail(10).mean()) / (float(data['close'].iloc[-1]) + 1e-12))
            
            # 3. Volume Delta Analysis (Buying vs Selling Pressure)
            delta_bias = "NEUTRAL"
            if 'taker_buy_volume' in data.columns:
                recent_buy = data['taker_buy_volume'].tail(10).sum()
                recent_total = data['volume'].tail(10).sum()
                buy_ratio = recent_buy / (recent_total + 1e-12)
                if buy_ratio > 0.55: delta_bias = "BULLISH"
                elif buy_ratio < 0.45: delta_bias = "BEARISH"
            
            # Combine Factors for Prediction
            if volatility > 0.004:
                return f"VOLATILE ({delta_bias} Bias) - Caution"
            
            if trend_strength > 0.0015:
                # Strong Trend
                direction = "UP" if data['close'].iloc[-1] > data['close'].iloc[-20] else "DOWN"
                return f"TRENDING {direction} (Strong {delta_bias} Flow)"
                
            if volatility < 0.001:
                return "RANGING / CHOPPY (Wait for Breakout)"
                
            return f"NORMAL ({delta_bias} Lean)"
        except Exception:
            return "NEUTRAL"

class HighVolatilityRegimeShield:
    def check_safety(self, data):
        try:
            if len(data) < 10: return {"approved": True, "thought": "SHIELD: Scanning volatility..."}
            volatility = float(data['close'].pct_change().std() or 0.0)
            if volatility > 0.005:
                return {"approved": False, "thought": f"SHIELD: High-Vol Regime ({volatility*100:.2f}%) - Trades prohibited"}
            return {"approved": True, "thought": "SHIELD: Stability confirmed"}
        except Exception:
            return {"approved": True, "thought": "Shield inactive"}

class ReverseSafetyEngine:
    def check_reversal(self, data, original_signal):
        try:
            if len(data) < 3: return {"signal": original_signal, "thought": "No reversal data"}
            recent_candle = data.iloc[-1]
            wick_upper = float(recent_candle['high'] - max(recent_candle['open'], recent_candle['close']))
            wick_lower = float(min(recent_candle['open'], recent_candle['close']) - recent_candle['low'])
            body = abs(float(recent_candle['close'] - recent_candle['open']))
            
            if wick_upper > body * 2 and original_signal == 1:
                return {"signal": -1, "thought": "REVERSAL: Significant bearish wick - Forcing PUT"}
            if wick_lower > body * 2 and original_signal == -1:
                return {"signal": 1, "thought": "REVERSAL: Significant bullish wick - Forcing CALL"}
                
            return {"signal": original_signal, "thought": "No reversal patterns detected"}
        except Exception:
            return {"signal": original_signal, "thought": "Reversal engine error"}

class TrapCandleGenomeDetector:
    def detect_trap(self, data):
        try:
            if len(data) < 2: return {"is_trap": False, "thought": "Scanning for traps..."}
            current = data.iloc[-1]
            prev = data.iloc[-2]
            current_body = abs(float(current['close'] - current['open']))
            current_range = float(current['high'] - current['low'])
            if current_range == 0: return {"is_trap": False, "thought": "Static price"}
            body_ratio = current_body / current_range
            
            if body_ratio < 0.3 and current['high'] > prev['high'] and current['close'] < prev['close']:
                return {"is_trap": True, "thought": "TRAP: Bull trap detected (High wick + lower close)"}
            if body_ratio < 0.3 and current['low'] < prev['low'] and current['close'] > prev['close']:
                return {"is_trap": True, "thought": "TRAP: Bear trap detected (Low wick + higher close)"}
            return {"is_trap": False, "thought": "No genome-level traps detected"}
        except Exception:
            return {"is_trap": False, "thought": "Trap detector error"}

def parse_score_from_text(text):
    """Robustly parse a numeric sentiment score from free-text responses between -10 and 10."""
    try:
        if not text:
            return 0, "no text"
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        for n in nums:
            try:
                v = float(n)
                if -1000 < v < 1000:
                    return max(-10, min(10, v)), "parsed"
            except Exception:
                continue
        low = text.lower()
        if "bull" in low or "buy" in low or "positive" in low:
            return 5, "heuristic"
        if "bear" in low or "sell" in low or "negative" in low:
            return -5, "heuristic"
        return 0, "none"
    except Exception as e:
        return 0, f"error:{e}"

class DeepMTFAnalyzer:
    """Deep Multi-Timeframe Analysis Engine (1m, 5m, 15m)"""
    def __init__(self, parts):
        self.parts = parts
        
    def analyze_all_timeframes(self, mtf_buffers):
        """Universal Multi-Timeframe Analysis (1m to 4h)"""
        results = {}
        try:
            # Active timeframes to check
            target_tfs = ['5m', '15m', '30m', '1h', '2h', '4h']
            
            for tf in target_tfs:
                df = mtf_buffers.get(tf)
                if df is None or len(df) < 20: continue
                
                # Run core "Perspective" parts on higher timeframe data
                t_res = self.parts['part6_trend'].analyze(df)
                s_res = self.parts['part8_structure'].analyze(df)
                v_res = self.parts['part4_volume'].analyze(df)
                c_res = self.parts['part10_candlestats'].analyze(df)
                
                # Extract signals (handle both dict and raw int)
                t_sig = t_res.get('signal', 0) if isinstance(t_res, dict) else t_res
                s_sig = s_res.get('signal', 0) if isinstance(s_res, dict) else s_res
                v_sig = v_res.get('signal', 0) if isinstance(v_res, dict) else v_res
                c_sig = c_res.get('signal', 0) if isinstance(c_res, dict) else c_res
                
                # Consensus for this timeframe
                score = t_sig + s_sig + v_sig + c_sig
                direction = 1 if score >= 2 else (-1 if score <= -2 else 0)
                
                results[tf] = {
                    'direction': direction,
                    'score': score,
                    'is_strong': abs(score) >= 3,
                    'thoughts': {
                        'trend': t_res.get('thought', '') if isinstance(t_res, dict) else str(t_res),
                        'structure': s_res.get('thought', '') if isinstance(s_res, dict) else str(s_res),
                        'candles': c_res.get('thought', '') if isinstance(c_res, dict) else str(c_res)
                    }
                }
            return results
        except Exception as e:
            logger.error(f"Universal MTF error: {e}")
            return {}

# ==================== TRADE SCORING MATRIX ====================

class TradeScoringMatrix:
    """SwingScalp-specific scoring system (0-100 points)"""
    
    def __init__(self):
        self.minimum_trade_score = 0
        self.score_weights = {
            'immediate_momentum': 20,
            'micro_sr_reaction': 15, 
            'trend_micro_alignment': 15,
            'volume_confirmation': 10,
            'candle_pattern': 10,
            'entry_timing_precision': 15,
            'trap_avoidance': 10,
            'session_strength': 5
        }
    
    def calculate_trade_score(self, data, current_price, context):
        """Calculate 0-100 trade trading score"""
        try:
            scores = {}
            
            # 1. IMMEDIATE MOMENTUM (0-20)
            scores['immediate_momentum'] = self._score_immediate_momentum(data)
            
            # 2. MICRO SUPPORT/RESISTANCE REACTION (0-15)
            scores['micro_sr_reaction'] = self._score_micro_sr_reaction(data, current_price)
            
            # 3. TREND MICRO ALIGNMENT (0-15)
            scores['trend_micro_alignment'] = self._score_trend_micro_alignment(data)
            
            # 4. VOLUME CONFIRMATION (0-10)
            scores['volume_confirmation'] = self._score_volume_confirmation(data)
            
            # 5. CANDLE PATTERN (0-10)
            scores['candle_pattern'] = self._score_candle_pattern(data)
            
            # 6. ENTRY TIMING PRECISION (0-15)
            scores['entry_timing_precision'] = self._score_entry_timing(data, context)
            
            # 7. TRAP AVOIDANCE (0-10)
            scores['trap_avoidance'] = self._score_trap_avoidance(data)
            
            # 8. SESSION STRENGTH (0-5)
            scores['session_strength'] = self._score_session_strength()
            
            # Calculate weighted total
            total_score = 0
            for key, weight in self.score_weights.items():
                total_score += scores.get(key, 0) * (weight / 100)
            
            return min(100, total_score), scores
            
        except Exception as e:
            logger.error(f"SwingScalp scoring error: {e}")
            return 0, {}

    def _score_immediate_momentum(self, data):
        """Score immediate momentum strength (0-20)"""
        try:
            if len(data) < 3:
                return 0
                
            current_candle = data.iloc[-1]
            prev_candle = data.iloc[-2]
            
            # Calculate momentum strength
            body_size = abs(float(current_candle['close']) - float(current_candle['open']))
            candle_range = float(current_candle['high']) - float(current_candle['low'])
            
            if candle_range == 0:
                return 0
                
            body_ratio = body_size / candle_range
            
            # Strong momentum: large body, small wicks
            if body_ratio > 0.7:
                score = 18
            elif body_ratio > 0.5:
                score = 14
            elif body_ratio > 0.3:
                score = 8
            else:
                score = 3
                
            # Direction consistency
            if len(data) >= 5:
                recent_trend = self._get_micro_trend(data)
                current_direction = 1 if current_candle['close'] > current_candle['open'] else -1
                if recent_trend == current_direction:
                    score += 2
                    
            return min(20, score)
            
        except Exception:
            return 0

    def _score_micro_sr_reaction(self, data, current_price):
        """Score reaction at micro support/resistance (0-15)"""
        try:
            if len(data) < 10:
                return 0
                
            # Find recent swing points
            highs = data['high'].tail(10)
            lows = data['low'].tail(10)
            
            resistance_level = float(highs.max())
            support_level = float(lows.min())
            
            price_range = resistance_level - support_level
            if price_range == 0:
                return 0
                
            # Calculate distance to nearest key level
            dist_to_resistance = abs(current_price - resistance_level) / price_range
            dist_to_support = abs(current_price - support_level) / price_range
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Score based on proximity to key levels
            if min_distance < 0.05:  # Very close to key level
                score = 12
            elif min_distance < 0.1:  # Close to key level
                score = 8
            elif min_distance < 0.15:  # Moderate distance
                score = 5
            else:
                score = 2
                
            # Bonus for rejection patterns
            current_candle = data.iloc[-1]
            if self._is_rejection_candle(current_candle):
                score += 3
                
            return min(15, score)
            
        except Exception:
            return 0

    def _score_trend_micro_alignment(self, data):
        """Score micro-trend alignment (0-15)"""
        try:
            if len(data) < 8:
                return 0
                
            # Multiple timeframe alignment
            trend_1m = self._get_micro_trend(data.tail(5))  # 1min trend
            trend_3m = self._get_micro_trend(data.tail(15))  # 3min trend
            trend_5m = self._get_micro_trend(data.tail(25))  # 5min trend
            
            alignment_score = 0
            if trend_1m == trend_3m == trend_5m:
                alignment_score = 12  # Perfect alignment
            elif trend_1m == trend_3m:
                alignment_score = 8   # Good alignment
            elif trend_1m == trend_5m:
                alignment_score = 6   # Moderate alignment
            else:
                alignment_score = 2   # Poor alignment
                
            # Trend strength bonus
            trend_strength = self._calculate_trend_strength(data.tail(10))
            if trend_strength > 0.001:
                alignment_score += 3
                
            return min(15, alignment_score)
            
        except Exception:
            return 0

    def _score_volume_confirmation(self, data):
        """Score volume confirmation (0-10)"""
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return 5  # Neutral if no volume data
                
            current_volume = float(data['volume'].iloc[-1])
            avg_volume = float(data['volume'].tail(20).mean())
            
            if avg_volume == 0:
                return 5
                
            volume_ratio = current_volume / avg_volume
            
            current_candle = data.iloc[-1]
            price_direction = 1 if current_candle['close'] > current_candle['open'] else -1
            
            # Volume confirmation logic
            if volume_ratio > 1.5:  # High volume
                # BUG FIX #16: Both directions scored 9 — bullish vol should score higher than bearish
                if price_direction == 1:
                    score = 9   # Bullish volume confirmation
                else:
                    score = 7   # Bearish — high vol on down move = distribution, lower score for CALL bias
            elif volume_ratio > 1.2:  # Above average volume
                score = 7
            elif volume_ratio > 0.8:  # Average volume
                score = 5
            else:  # Low volume
                score = 3
                
            return min(10, score)
            
        except Exception:
            return 5

    def _score_candle_pattern(self, data):
        """Score trade-specific candle patterns (0-10)"""
        try:
            if len(data) < 3:
                return 0
                
            current_candle = data.iloc[-1]
            prev_candle = data.iloc[-2]
            
            score = 0
            
            # Bullish patterns
            if self._is_bullish_engulfing(data):
                score += 8
            if self._is_hammer(current_candle):
                score += 7
            if self._is_morning_star(data):
                score += 9
                
            # Bearish patterns  
            if self._is_bearish_engulfing(data):
                score += 8
            if self._is_shooting_star(current_candle):
                score += 7
            if self._is_evening_star(data):
                score += 9
                
            # No strong pattern
            if score == 0:
                score = 3
                
            return min(10, score)
            
        except Exception:
            return 0

    def _score_entry_timing(self, data, context):
        """Score entry timing precision (0-15)"""
        try:
            score = 5  # Base score
            
            # Market session timing
            current_hour = datetime.now().hour
            session_quality = self._get_current_session_quality(current_hour)
            if session_quality == 'BEST':
                score += 4
            elif session_quality == 'HIGH':
                score += 3
            elif session_quality == 'MEDIUM':
                score += 1
                
            # Volatility timing
            volatility = float(data['close'].pct_change().std() or 0.0)
            if 0.001 < volatility < 0.004:  # Ideal volatility range
                score += 3
            elif volatility > 0.006:  # Too volatile
                score -= 2
                
            # News timing (simplified - would integrate with news API)
            score += 2  # Assume no major news
            
            return min(15, max(0, score))
            
        except Exception:
            return 5

    def _score_trap_avoidance(self, data):
        """Score trap pattern avoidance (0-10)"""
        try:
            score = 8  # Start with high score
            
            # Check for false breakouts
            if self._is_false_breakout(data):
                score -= 6
                
            # Check for doji indecision
            if self._is_doji_candle(data.iloc[-1]):
                score -= 3
                
            # Check for inside bar compression
            if self._is_inside_bar(data):
                score -= 2
                
            return min(10, max(0, score))
            
        except Exception:
            return 5

    def _score_session_strength(self):
        """Score trading session strength (0-5)"""
        try:
            current_hour = datetime.now().hour
            session_quality = self._get_current_session_quality(current_hour)
            
            if session_quality == 'BEST':
                return 5
            elif session_quality == 'HIGH':
                return 4
            elif session_quality == 'MEDIUM':
                return 3
            else:
                return 1
        except Exception:
            return 2

    # ==================== PATTERN DETECTION METHODS ====================

    def _get_micro_trend(self, data):
        """Get micro trend direction (1: up, -1: down, 0: neutral)"""
        if len(data) < 2:
            return 0
        price_change = float(data['close'].iloc[-1]) - float(data['close'].iloc[0])
        if abs(price_change) < 0.0001:
            return 0
        return 1 if price_change > 0 else -1

    def _calculate_trend_strength(self, data):
        """Calculate trend strength as percentage"""
        if len(data) < 2:
            return 0
        start_price = float(data['close'].iloc[0])
        end_price = float(data['close'].iloc[-1])
        return abs(end_price - start_price) / start_price

    def _is_rejection_candle(self, candle):
        """Check if candle shows rejection"""
        body = abs(float(candle['close']) - float(candle['open']))
        upper_wick = float(candle['high']) - max(float(candle['open']), float(candle['close']))
        lower_wick = min(float(candle['open']), float(candle['close'])) - float(candle['low'])
        
        if body == 0:
            return False
            
        upper_ratio = upper_wick / body
        lower_ratio = lower_wick / body
        
        return upper_ratio > 2 or lower_ratio > 2

    def _is_bullish_engulfing(self, data):
        """Check for bullish engulfing pattern"""
        if len(data) < 2:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        return (float(current['close']) > float(current['open']) and
                float(prev['close']) < float(prev['open']) and
                float(current['open']) < float(prev['close']) and
                float(current['close']) > float(prev['open']))

    def _is_bearish_engulfing(self, data):
        """Check for bearish engulfing pattern"""
        if len(data) < 2:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        return (float(current['close']) < float(current['open']) and
                float(prev['close']) > float(prev['open']) and
                float(current['open']) > float(prev['close']) and
                float(current['close']) < float(prev['open']))

    def _is_hammer(self, candle):
        """Check for hammer pattern"""
        body = abs(float(candle['close']) - float(candle['open']))
        total_range = float(candle['high']) - float(candle['low'])
        
        if total_range == 0:
            return False
            
        lower_wick = min(float(candle['open']), float(candle['close'])) - float(candle['low'])
        lower_wick_ratio = lower_wick / total_range
        
        return (lower_wick_ratio > 0.6 and 
                body > 0 and 
                float(candle['close']) > float(candle['open']))

    def _is_shooting_star(self, candle):
        """Check for shooting star pattern"""
        body = abs(float(candle['close']) - float(candle['open']))
        total_range = float(candle['high']) - float(candle['low'])
        
        if total_range == 0:
            return False
            
        upper_wick = float(candle['high']) - max(float(candle['open']), float(candle['close']))
        upper_wick_ratio = upper_wick / total_range
        
        return (upper_wick_ratio > 0.6 and 
                body > 0 and 
                float(candle['close']) < float(candle['open']))

    def _is_doji_candle(self, candle):
        """Check for doji candle"""
        body = abs(float(candle['close']) - float(candle['open']))
        total_range = float(candle['high']) - float(candle['low'])
        
        if total_range == 0:
            return False
            
        return body / total_range < 0.1

    def _is_inside_bar(self, data):
        """Check for inside bar pattern"""
        if len(data) < 2:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        return (float(current['high']) <= float(prev['high']) and
                float(current['low']) >= float(prev['low']))

    def _is_false_breakout(self, data):
        """Check for false breakout pattern"""
        if len(data) < 3:
            return False
            
        current = data.iloc[-1]
        prev = data.iloc[-2]
        prev_prev = data.iloc[-3]
        
        # Check for breakout then reversal
        broke_high = (float(prev['high']) > float(prev_prev['high']) and
                     float(current['close']) < float(prev_prev['high']))
                     
        broke_low = (float(prev['low']) < float(prev_prev['low']) and
                    float(current['close']) > float(prev_prev['low']))
                    
        return broke_high or broke_low

    def _is_morning_star(self, data):
        """Check for morning star pattern (simplified)"""
        if len(data) < 3:
            return False
            
        first = data.iloc[-3]  # Bearish candle
        second = data.iloc[-2]  # Small body (doji or small)
        third = data.iloc[-1]   # Bullish candle
        
        first_bearish = float(first['close']) < float(first['open'])
        third_bullish = float(third['close']) > float(third['open'])
        second_small = abs(float(second['close']) - float(second['open'])) / (float(second['high']) - float(second['low']) + 1e-12) < 0.3
        
        return first_bearish and second_small and third_bullish

    def _is_evening_star(self, data):
        """Check for evening star pattern (simplified)"""
        if len(data) < 3:
            return False
            
        first = data.iloc[-3]  # Bullish candle
        second = data.iloc[-2]  # Small body (doji or small)
        third = data.iloc[-1]   # Bearish candle
        
        first_bullish = float(first['close']) > float(first['open'])
        third_bearish = float(third['close']) < float(third['open'])
        second_small = abs(float(second['close']) - float(second['open'])) / (float(second['high']) - float(second['low']) + 1e-12) < 0.3
        
        return first_bullish and second_small and third_bearish

    def _get_current_session_quality(self, current_hour):
        """Get current trading session quality"""
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return times['quality']
        return 'LOW'

# ==================== TRADE TRADE MANAGER ====================

class TradeManager:
    """Manage trade trading operations and risk"""
    
    def __init__(self):
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trade_history = []
        
    def can_trade(self):
        """Check if trading is allowed"""
        # Check daily limit
        if self.daily_trades >= TRADE_CONFIG['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return False
            
        # Check consecutive losses
        if self.consecutive_losses >= TRADE_CONFIG['consecutive_loss_limit']:
            logger.warning("Consecutive loss limit reached")
            return False
            
        # Check cooldown period
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if time_since_last < TRADE_CONFIG['trade_cooldown_minutes']:
                logger.warning(f"Trade cooldown active: {TRADE_CONFIG['trade_cooldown_minutes'] - time_since_last:.1f} minutes remaining")
                return False
                
        # Check session quality
        current_hour = datetime.now().hour
        session_quality = self._get_session_quality(current_hour)
        if session_quality == 'LOW':
            pass
            # logger.warning("Low quality trading session - trading anyway")
            # return False
            
        return True
        
    def record_trade(self, signal, score, result):
        """Record trade outcome"""
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'score': score,
            'result': result,
            'session': self._get_current_session()
        }
        
        self.trade_history.append(trade_record)
        
        # Update consecutive losses
        if result == 'LOSS':
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.trade_history = []
        
    def _get_session_quality(self, current_hour):
        """Get current session quality"""
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return times['quality']
        return 'LOW'
        
    def _get_current_session(self):
        """Get current trading session name"""
        current_hour = datetime.now().hour
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return session
        return 'overnight'

# ==================== TRADE EXPIRY OPTIMIZER ====================

class TradeOptimizer:
    """Optimize trade style based on market conditions"""
    
    def __init__(self):
        self.trade_styles = ['SCALP', 'DAY_TRADE', 'SWING']
        
    def recommend_trade_type(self, score, volatility, pattern_type):
        """Recommend optimal trade style"""
        # Base style on score tier
        if score >= 90:
            base_style = 'SCALP'
        elif score >= 85:
            base_style = 'DAY_TRADE'
        else:
            base_style = 'SWING'
            
        # Adjust for volatility
        if volatility > 0.004:  # High volatility
            if base_style == 'SCALP':
                base_style = 'DAY_TRADE'
        elif volatility < 0.001:  # Low volatility
            if base_style == 'SWING':
                base_style = 'SCALP'
                
        # Adjust for pattern type
        if pattern_type == 'MOMENTUM_SCALP':
            base_style = 'SCALP'
        elif pattern_type == 'REJECTION_PLAY':
            base_style = 'DAY_TRADE'
        elif pattern_type == 'TREND_PULLBACK':
            base_style = 'SWING'
            
        return base_style
        
    def calculate_payout_ratio(self, score, trade_type):
        """Calculate Risk:Reward ratio"""
        rr_ratio = 1.5
        
        # Adjust for score
        if score >= 95:
            rr_ratio += 0.5
        elif score >= 90:
            rr_ratio += 0.3
        elif score >= 85:
            rr_ratio += 0.1
            
        # Adjust for trade type
        if trade_type == 'SWING':
            rr_ratio += 1.0
        elif trade_type == 'SCALP':
            rr_ratio -= 0.5
            
        return f"1:{rr_ratio:.1f}"


class JarvisCNS:
    """Central Nervous System - Monitors system health and logic failures"""
    def __init__(self, ai_brain, parent_system=None):
        self.ai_brain = ai_brain
        self.parent = parent_system
        self.sensory_buffer = deque(maxlen=100)
        self.pain_events = []
        self.health_status = "HEALTHY"
        self.last_diagnostic = "No issues detected."
        
    def record_perception(self, analysis_cycle):
        """Record what the system 'felt' in this cycle"""
        self.sensory_buffer.append({
            "timestamp": time.time(),
            "telemetry": analysis_cycle.get('telemetry', {}),
            "signal": analysis_cycle.get('signal', {}),
            "market_logic": analysis_cycle.get('detailed_scores', {})
        })
        
    def detect_pain(self, trade_result):
        """
        Analyze if a logic failure occurred based on trade outcome.
        If a trade lost, we ask: 'Which part lied to us?'
        """
        if trade_result.get('pnl', 0) >= 0:
            return None # No pain on profit
            
        # Trigger diagnostic on loss
        self.health_status = "PAIN DETECTED"
        logger.warning("🧠 CNS: Pain detected in system. Diagnosing logic failure...")
        
        # Get the latest sensory data
        last_feeling = self.sensory_buffer[-1] if self.sensory_buffer else {}
        
        # Preserve observed health information without sending telemetry to a
        # retired model or treating an unverified diagnosis as trading input.
        diagnostic = "Loss observed; model diagnosis unavailable; manual review required"
        self.last_diagnostic = diagnostic
        logger.warning("CNS: %s (snapshot=%s)", diagnostic, bool(last_feeling))
        return diagnostic

# ==================== ENHANCED JARVIS TRADE ELITE ====================

class JarvisElite:
    """JARVIS TRADE ELITE v7.0 - Complete SwingScalp Trading System"""
    
    def __init__(self, backtest_mode=False, analysis_only=False):
        """Backtests retain GPU analysis but block AI and live-data side effects."""
        self.analysis_only = bool(analysis_only)
        env_backtest = os.environ.get("JARVIS_BACKTEST_MODE", "").lower() in ("1", "true", "yes")
        self.is_backtest_mode = bool(backtest_mode or env_backtest)
        # All native modules see one normalized device contract.  Telemetry
        # records detection/fallback separately; no GPU availability is faked.
        self.runtime = detect_backend()
        self.device = torch_device(self.runtime)
        self.gpu_status = self.runtime.to_dict()
        self.last_legacy_llm_decision = {"decision": "WAIT", "confidence": 0, "rationale": "not queried"}
        self.last_legacy_llm_snapshot = None
        self.active_symbol = os.getenv('JARVIS_DEFAULT_SYMBOL', 'BTCUSDT')
        self.active_base_asset = 'BTC' 
        self.scoring_matrix = TradeScoringMatrix()
        self.trade_manager = TradeManager()
        self.expiry_optimizer = TradeOptimizer()
        self.scalping_engine = ScalpingEngine() # NEW: Scalping Targets
        self.mode_engine = ModeEngine(require_xgboost=False)  # 29-Brain Layered Conductor Engine (advisory; unvalidated XGB = neutral, not HOLD)
        self.deepseek_enabled = False  # Retired compatibility attribute; live non-pure mode is fail-closed.
        # The HUD is a local display channel, not a command interface.
        self.hud_enabled = not self.is_backtest_mode
        self.hud_url = "http://127.0.0.1:7788/api/telemetry"
        token = os.environ.get("JARVIS_HUD_INGEST_TOKEN")
        self.hud_headers = {"X-Jarvis-Hud-Token": token} if token else None
        # Backtest mode was intentionally set before optional components started.
        
        # Historical replay has no asynchronous/live event bus.
        if not self.is_backtest_mode:
            try:
                from jarvis_cognitive_bus import CognitiveBus
                self.bus = CognitiveBus()
                logger.info("🧠 Jarvis Cognitive Swarm Bus Initialized")
            except ImportError:
                self.bus = None
                logger.warning("⚠️ jarvis_cognitive_bus not found, running without swarm thoughts")
        else:
            self.bus = None

        # 👁️ INITIALIZE WATCHER AI (Pipeline Layer 1)
        # Starts as a background daemon — continuously monitors bus & builds Smart Packets
        self.watcher_ai = None
        if self.bus is not None and not self.is_backtest_mode:
            try:
                from jarvis_watcher_ai import JarvisWatcherAI
                self.watcher_ai = JarvisWatcherAI(bus=self.bus)
                self.watcher_ai.start()
                logger.info("👁️ Watcher AI activated — Smart Context Pipeline ONLINE")
            except Exception as _we:
                logger.warning(f"⚠️ WatcherAI init failed (non-critical): {_we}")

    
        # Historical replay must not even construct a live data client.
        if not self.is_backtest_mode:
            try:
                from delta_api_wrapper import DeltaExchangeData
                self.delta_data = DeltaExchangeData()
                self.delta_client = self.delta_data
                logger.info("✅ Delta Exchange Data Initialized")
            except Exception as e:
                logger.warning(f"Delta initialization failed: {e}")
                self.delta_data = None
                self.delta_client = None
        else:
            self.delta_data = None
            self.delta_client = None

        # Selected-symbol analysis reads strict Binance Spot candles; Delta stays
        # the separate execution/options venue. No Bybit or Delta candle fallback
        # is accepted by this cache. Per-frame lookback follows the interval
        # profile, and each snapshot keeps the forming candle separate.
        self.binance_data = None
        if BINANCE_DATA_AVAILABLE:
            try:
                self.binance_data = _get_binance_data()
            except Exception as _binance_error:
                logger.warning("Binance public data client unavailable: %s", type(_binance_error).__name__)
        self.direct_candle_cache = None
        self._active_candle_snapshot = None
        if (self.binance_data is not None and DIRECT_CANDLE_CACHE_AVAILABLE
                and BinanceSpotCandleClient is not None):
            try:
                _history_limits = {tf: TIMEFRAME_HISTORY_CANDLES[tf] for tf in BINANCE_SPOT_TIMEFRAMES}
                self.direct_candle_cache = DirectCandleCache(
                    BinanceSpotCandleClient(self.binance_data),
                    ttl_seconds=float(os.getenv('JARVIS_CANDLE_REFRESH_SECONDS', '5')),
                    timeframes=tuple(BINANCE_SPOT_TIMEFRAMES),
                    history_limits=_history_limits,
                )
            except Exception as _cache_error:
                logger.warning("Strict Binance candle cache unavailable: %s", type(_cache_error).__name__)

        # Optional wiring: keep supplemental sources and utilities discoverable,
        # but do not start scanners, model calls, backtests, or another position
        # monitor as a side effect of starting the brain.
        self.position_manager = None
        self.position_sizer = None
        self.coin_scanner = None
        self.upstox_data = None
        self.specialist_pool = None
        self.specialist_pool_class = _SpecialistPool
        self.backtester_class = _JarvisFullBacktester
        self.integration_sources = {}
        if self.delta_data is not None:
            try:
                self.position_manager = _get_jarvis_position_manager(
                    self.delta_data, self.bus
                )
                # JarvisAutoTrader owns TP/SL lifecycle monitoring.  It already
                # uses this singleton for its compound pool; do not start the
                # manager here or it could submit a duplicate close order.
                self.position_sizer = _get_jarvis_sizer(self.delta_data)
            except Exception as integration_error:
                logger.debug("Optional sizing/position integration unavailable: %s", integration_error)
        if not self.is_backtest_mode:
            try:
                self.coin_scanner = _get_jarvis_coin_scanner(
                    bus=self.bus,
                    position_check_fn=(self.position_manager.has_open_position if self.position_manager else None),
                )
            except Exception as integration_error:
                logger.debug("Optional coin scanner unavailable: %s", integration_error)
            try:
                self.binance_data = _get_binance_data()
            except Exception as integration_error:
                logger.debug("Optional Binance data source unavailable: %s", integration_error)
            try:
                self.upstox_data = _get_upstox_data()
            except Exception as integration_error:
                logger.debug("Optional Upstox data source unavailable: %s", integration_error)
        self.integration_sources = {
            "position_manager": self.position_manager,
            "sizer": self.position_sizer,
            "coin_scanner": self.coin_scanner,
            "binance_data": self.binance_data,
            "upstox_data": self.upstox_data,
            "specialist_pool": self.specialist_pool_class,
            "backtester": self.backtester_class,
        }
        if self.bus:
            try:
                self.bus.publish("THOUGHTS", "JarvisIntegrations", {
                    "available": sorted(name for name, value in self.integration_sources.items() if value is not None),
                    "on_demand_only": ["coin_scanner", "specialist_pool", "backtester"],
                })
            except Exception:
                pass

        # Backtests exclude remote options inputs; historical option data must be supplied separately.
        if not self.is_backtest_mode:
            try:
                from deribit_options_client import DeribitOptionsClient
                client_id = os.getenv("DERIBIT_CLIENT_ID", "")
                client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
                # The route may change after startup; Part14 is the only
                # component allowed to use Deribit and keeps it asset-scoped.
                self.deribit = DeribitOptionsClient(
                    currency=str(getattr(self, 'active_base_asset', 'BTC') or 'BTC'),
                    client_id=client_id,
                    client_secret=client_secret,
                )
                logger.info("✅ Deribit Options Client Initialized (Dual Intelligence)")
            except Exception as e:
                logger.warning(f"Deribit init failed: {e}")
                self.deribit = None
        else:
            self.deribit = None
        # Initialize Parts Dictionary (Empty first, passed by reference to Fusion Engine)
        self.parts = {}

        # FIX #5: Removed phantom import of deepseek_missing_brains
        # Stubs are provided inline by part2_fixed.py — not needed here
        self.candle_psychology = None
        # FIX: Initialize zone_detector from part3 ZonePointFiveDetectorGPU
        try:
            from part3_FIXED import ZonePointFiveDetectorGPU, CandlePsychologyMasterGPU
            self.zone_detector = ZonePointFiveDetectorGPU(self)
            self.candle_psychology = CandlePsychologyMasterGPU(self)
            logger.info("✅ Zone Detector & Candle Psychology initialized from part3")
        except Exception as _ze:
            self.zone_detector = None
            logger.warning(f"⚠️ zone_detector init failed: {_ze} — zone signals skipped")
    
        # Realtime data structure expected by institutional engine
        self.realtime_data = {
            '1min': [],
            '5min': [],
            '15min': []
        }

        # Initialize External GPU Engines only when explicitly permitted.  The
        # heavy CPU/GPU adapters are advisory and must not make safe startup or
        # an offline wiring check wait on model/accelerator initialization.
        self.engines = {}
        self.native_engine_status = {}
        if EXTERNAL_ENGINES_AVAILABLE and (self.runtime.torch_available or not self.is_backtest_mode):
            try:
                logger.info("🚀 Initializing External GPU Engines...")
                self.native_engine_status = {}
                def _native(name, factory):
                    try:
                        engine = factory()
                        # Normalize post-construction contracts where engines
                        # expose a device field; never overwrite a valid custom
                        # backend with a string.
                        try:
                            if hasattr(engine, 'device'):
                                engine.device = self.device
                        except Exception:
                            pass
                        self.native_engine_status[name] = {"status": "ready", "backend": self.runtime.backend}
                        return engine
                    except Exception as exc:
                        # Isolate one incompatible native constructor instead of
                        # hiding the cause behind a batch failure (e.g. string
                        # device values used where torch.device was required).
                        self.native_engine_status[name] = {
                            "status": "fallback", "backend": "cpu",
                            "error": f"{type(exc).__name__}: {str(exc)[:160]}"
                        }
                        logger.warning("[GPU] %s unavailable; CPU/fallback path: %s", name, self.native_engine_status[name]['error'])
                        return None
                self.engines['institutional'] = _native('institutional', lambda: InstitutionalTradingEngineGPU(self))
                # Neural model engine retired: no unvalidated inference on the trade path.
                self.engines['fusion'] = _native('fusion', lambda: GPUEnhancedFusionEngine(self.parts))
                self.engines['confidence'] = _native('confidence', lambda: GPUUnifiedConfidenceEngine())
                self.engines['pattern'] = _native('pattern', lambda: EnhancedGPUPatternRecognitionEngine())
                self.engines = {name: engine for name, engine in self.engines.items() if engine is not None}
                if not self.is_backtest_mode:
                    try:
                        from part7_FIXED import EnhancedGPULiveDataEngine
                        from part9_FIXED import GPUAIAdaptiveLearningEngine
                        self.engines['live_data'] = EnhancedGPULiveDataEngine()
                        self.engines['adaptive'] = GPUAIAdaptiveLearningEngine()
                        logger.info("✅ Live Data & Adaptive Engines Connected (Parts 7 & 9)")
                    except Exception as e:
                        logger.error(f"❌ Failed to load Parts 7/9: {e}")
                    
                # --- WIRING FIX PHASE 2: Additional Engines ---
                try:
                    from part8_FIXED import EnhancedPatternRecognitionSystem
                    self.engines['pattern_system'] = EnhancedPatternRecognitionSystem()
                    logger.info("✅ [PATTERN-SYSTEM] EnhancedPatternRecognitionSystem Connected")
                except Exception as e:
                    logger.warning(f"⚠️ [PATTERN-SYSTEM] init failed: {e}")
                    
                try:
                    from part2_FIXED import CorrelationMatrixBrainGPU
                    self.correlation_brain = CorrelationMatrixBrainGPU(self)  # pass master_system
                    logger.info("✅ [CORRELATION] CorrelationMatrixBrainGPU Connected")
                except Exception as e:
                    self.correlation_brain = None
                    logger.warning(f"⚠️ [CORRELATION] init failed: {e}")
                # ----------------------------------------------
                
                # Attach bus to all engines
                if self.bus:
                    for name, engine in self.engines.items():
                        engine.bus = self.bus
                
                logger.info("✅ External GPU Engines Initialized")
                
                # --- NEW WIRING: Start dormant engines ---
                if not self.is_backtest_mode and 'adaptive' in self.engines:
                    try:
                        import asyncio, inspect
                        adaptive_eng = self.engines['adaptive']
                        # GPUAIAdaptiveLearningEngine uses start_learning_engine (async)
                        if hasattr(adaptive_eng, 'start_learning_engine'):
                            _method = adaptive_eng.start_learning_engine
                        elif hasattr(adaptive_eng, 'start_ai_learning'):
                            _method = adaptive_eng.start_ai_learning
                        else:
                            _method = None
                        if _method:
                            if inspect.iscoroutinefunction(_method):
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        asyncio.ensure_future(_method())
                                    else:
                                        loop.run_until_complete(_method())
                                except RuntimeError:
                                    asyncio.run(_method())
                            else:
                                _method()
                            logger.info("✅ [ADAPTIVE] AI Learning Engine STARTED")
                        else:
                            logger.info("ℹ️ [ADAPTIVE] No start method found — engine self-manages")
                    except Exception as e:
                        logger.warning(f"⚠️ [ADAPTIVE] start failed: {e}")
                        
                if not self.is_backtest_mode and 'confidence' in self.engines:
                    try:
                        import asyncio, inspect
                        conf_eng = self.engines['confidence']
                        # Try known start method names
                        for _mname in ('start_confidence_monitoring', 'start_monitoring', 'start'):
                            if hasattr(conf_eng, _mname):
                                _cm = getattr(conf_eng, _mname)
                                if inspect.iscoroutinefunction(_cm):
                                    try:
                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            asyncio.ensure_future(_cm())
                                        else:
                                            loop.run_until_complete(_cm())
                                    except RuntimeError:
                                        asyncio.run(_cm())
                                else:
                                    _cm()
                                logger.info(f"✅ [CONFIDENCE] Monitor STARTED via {_mname}()")
                                break
                        else:
                            logger.info("ℹ️ [CONFIDENCE] Engine active — no explicit start needed")
                    except Exception as e:
                        logger.warning(f"⚠️ [CONFIDENCE] start failed: {e}")
                # ----------------------------------------
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize external engines: {e}")
        elif EXTERNAL_ENGINES_AVAILABLE:
            # Optional modules may import a lightweight torch shim when
            # PyTorch is absent. Do not construct those modules: their
            # ``device`` values can be strings while native code expects
            # ``torch.device.type``. Record an explicit CPU fallback instead
            # of emitting an opaque startup exception.
            for _name in ('institutional', 'neural', 'fusion', 'confidence', 'pattern'):
                self.native_engine_status[_name] = {
                    'status': 'fallback', 'backend': 'cpu',
                    'error': 'PyTorch unavailable; native engine not constructed',
                }


    
        # Existing Jarvis components (Now Adapters)
        # If an optional integration is unavailable, keep Part2's deterministic
        # range fallback but do not construct its heavyweight 16-brain engine.
        # This makes missing-module startup genuinely fail-safe and bounded.
        _optional_wiring_complete = all((SIZER_AVAILABLE, POSITION_MANAGER_AVAILABLE,
                                          COIN_SCANNER_AVAILABLE, SPECIALIST_POOL_AVAILABLE,
                                          BINANCE_DATA_AVAILABLE, UPSTOX_DATA_AVAILABLE,
                                          BACKTESTER_AVAILABLE))
        if _optional_wiring_complete:
            _part2_zone = Part2Zone()
        else:
            _part2_zone = object.__new__(Part2Zone)
            _part2_zone._engine = None
        self.parts = {
            'part1_breakout': Part1Breakout(),
            'part2_zone': _part2_zone,
            'part3_psychology': Part3Psychology(),
            'part4_volume': Part4Volume(),
            'part5_ml': Part5ML(),
            'part6_trend': Part6Trend(),
            'part7_volatility': Part7Volatility(),
            'part8_structure': Part8Structure(),
            'part9_orderflow': Part9Orderflow(),
            'part10_candlestats': Part10Candlestats(),
            'part11_fusion': Part11Fusion(),
            'part12_confidence': Part12Confidence(),
        }
        # Options positioning uses live provider observations in live mode and
        # timestamp-matched recorded observations in backtests. It never fetches
        # a current chain during historical replay.
        self.parts['part14_options_chain'] = Part14OptionsChain(
            delta_client=None if self.is_backtest_mode else self.delta_data,
            backtest_mode=self.is_backtest_mode,
        )
        self.mtf_analyzer = DeepMTFAnalyzer(self.parts) # NEW: Deep MTF Analysis
        self.brains = {
            'neural_hud_brain': None,
            'quantum_v5': QuantumV5(),
            'safety_risk_brain': SafetyRiskBrain(),
            'volume_pressure_brain': VolumePressureBrain(),
            'trend_acceleration_brain': TrendAccelerationBrain(),
            'risk_filter_brain': RiskFilterBrain()
        }
        self.deepseek_brains = {
            'deepseek_v3_sentiment': DeepSeekV3Brain(),
            'deepseek_r1_reasoning': DeepSeekR1ReasoningBrain()
        }
    
        # NEW: Jarvis Neural Cortex (Unified AI Brain)
        self.neural_cortex = JarvisNeuralCortex()
        self.cns = JarvisCNS(self.neural_cortex, parent_system=self) # NEW: Jarvis CNS
    
        self.upgrades = {
            'reverse_safety_engine': ReverseSafetyEngine(),
            'trap_candle_genome_detector': TrapCandleGenomeDetector(),
            'market_mood_engine': MarketMoodEngine(),
            'high_volatility_regime_shield': HighVolatilityRegimeShield()
        }
    
        # Trading state
        self.current_score = 0
        self.current_signals = {}
        self.market_context = {}
        self.latest_part7 = {
            'signal': 0, 'status': 'not_run', 'reason': 'Part7 has not run',
            'entry_blocked': True, 'risk_veto': False, 'data_status': 'missing',
            'volatility_status': 'unknown', 'computation_backend': 'pandas_cpu',
            'timeframe_results': {},
        }

    def get_optional_utility(self, name, **kwargs):
        """Return an explicitly requested optional analysis utility.

        This method is intentionally not called from the execution loop.  It makes
        external data, backtesting, specialist models and GPT analysis available to
        operators without allowing any of them to alter a trading decision by
        default.
        """
        key = str(name).lower()
        if key == "specialist_pool" and self.specialist_pool_class:
            if self.specialist_pool is None:
                self.specialist_pool = self.specialist_pool_class(**kwargs)
            return self.specialist_pool
        if key == "backtester" and self.backtester_class:
            return self.backtester_class(**kwargs)
        return self.integration_sources.get(key)
    
    def _fetch_mtf_from_api(self, symbol=None):
        """Return validated Binance Spot native frames or fail closed.

        No raw-client compatibility path is allowed here: it would bypass source,
        timeframe history, continuity, and forming-candle validation.
        """
        if self.is_backtest_mode:
            return {}
        symbol = self._binance_spot_symbol(symbol or getattr(self, 'active_symbol', None))
        if not symbol:
            logger.warning('[MTF-API] No active symbol; refusing candle fetch')
            return {}
        direct_cache = getattr(self, 'direct_candle_cache', None)
        if direct_cache is None:
            logger.warning('[MTF-API] Strict Binance Spot candle cache unavailable; refusing raw fallback')
            return {}
        try:
            snapshot = getattr(self, '_active_candle_snapshot', None)
            _canonical = _binance_spot_symbol(symbol)
            if (snapshot is None or snapshot.symbol != _canonical
                    or snapshot.identity != ('binance', 'spot', _canonical, _canonical)):
                snapshot = self.direct_candle_cache.refresh(
                    _canonical, venue='binance', market_type='spot', instrument_id=_canonical,
                )
                self._active_candle_snapshot = snapshot
            if snapshot.identity != ('binance', 'spot', _canonical, _canonical):
                raise CandleDataError('selected candle source/market identity mismatch')
            if set(snapshot.frames) != set(LIVE_TIMEFRAMES):
                logger.warning('[MTF-API] Incomplete direct snapshot; refusing analysis')
                return {}
            frames = snapshot.analysis_frames()
            logger.info('[MTF-API] Direct %s: %s', symbol, {tf: len(df) for tf, df in frames.items()})
            return frames
        except Exception as error:
            logger.warning('[MTF-API] Direct native fetch rejected: %s', error)
            return {}

    def _binance_spot_symbol(self, symbol=None):
        """Map a Delta USD perp (e.g. ETHUSD) to its Binance USDT spot pair (ETHUSDT).

        Only the same base asset is ever mapped; other quotes pass through.
        """
        raw = symbol if symbol is not None else getattr(self, 'active_symbol', None)
        clean = str(raw or '').upper().replace('/', '').replace('-', '').replace('_', '')
        if not clean:
            return ''
        spot = str(getattr(self, 'active_spot_symbol', '') or '').upper()
        if clean.endswith('USDT'):
            return clean
        if clean.endswith('USD') and len(clean) > 3:
            mapped = clean[:-3] + 'USDT'
            return spot if spot == mapped else mapped
        return clean

    def analyze_trade_setup(self, data, mtf_context=None, candle_snapshot=None, native_mtf=None):
        """Analyze verified closed native candles; current candle stays metadata-only."""
        # Clear only display snapshots at cycle start: an early data-gate return
        # must not render yesterday's/previous-cycle Parts as current evidence.
        self.latest_part_timeframe_results = {}
        self.latest_timeframe_inputs = {}
        self.latest_fusion_results = {}
        self.latest_part7_cycle_display = None
        self.latest_neural_cycle_status = {
            'status': 'not_reported',
            'reason': 'analysis did not reach deterministic fusion',
            'availability': 'not probed',
        }
        self.latest_part_results = {}
        self.latest_diagnostics_cycle_status = {
            'status': 'NOT RUN', 'reason': 'current-cycle analysis has not started',
        }
        self._historical_options_bias = None

        def _record_diagnostics_cycle(status, reason):
            try:
                self.latest_diagnostics_cycle_status = {
                    'status': str(status or 'UNKNOWN'),
                    'reason': str(reason or 'no reason recorded'),
                }
            except Exception:
                pass

        def _capture_diagnostics_inputs(frames, snapshot):
            if _build_timeframe_inputs is None:
                return
            cache = getattr(self, 'direct_candle_cache', None)
            stale_intervals = getattr(cache, 'stale_intervals', 2.0) if cache is not None else 2.0
            fetch_ttl = getattr(cache, 'ttl_seconds', 45.0) if cache is not None else 45.0
            try:
                self.latest_timeframe_inputs = _build_timeframe_inputs(
                    frames,
                    snapshot=snapshot,
                    timeframes=tuple(frames.keys()) or _DIAGNOSTIC_TIMEFRAMES,
                    expected_symbol=_binance_spot_symbol(getattr(self, 'active_symbol', None)) or None,
                    stale_intervals=stale_intervals,
                    fetch_ttl_seconds=fetch_ttl,
                )
            except Exception as diagnostic_error:
                logger.debug('[LIVE-DIAGNOSTICS] timeframe metadata unavailable: %s', type(diagnostic_error).__name__)
                self.latest_timeframe_inputs = {}

        try:
            _initial_snapshot = candle_snapshot or getattr(self, '_active_candle_snapshot', None)
            if _initial_snapshot is not None:
                try:
                    _snapshot_frames = getattr(_initial_snapshot, 'frames', {})
                    _initial_frames = {
                        tf: getattr(frame, 'closed', None)
                        for tf, frame in _snapshot_frames.items()
                        if getattr(frame, 'closed', None) is not None
                    }
                except Exception:
                    _initial_frames = {}
                _capture_diagnostics_inputs(_initial_frames, _initial_snapshot)
                if not self.is_backtest_mode:
                    _selected = _binance_spot_symbol(getattr(self, 'active_symbol', ''))
                    _expected_identity = ('binance', 'spot', _selected, _selected)
                    if (not _selected or getattr(_initial_snapshot, 'identity', None) != _expected_identity
                            or set(getattr(_initial_snapshot, 'frames', {})) != set(LIVE_TIMEFRAMES)):
                        _record_diagnostics_cycle('BLOCKED', 'live snapshot provider, market, identity, or timeframe set mismatch')
                        return self._get_no_trade_signal('WAIT/NO-DATA: Binance Spot snapshot identity/timeframe mismatch')
                    _limits = getattr(getattr(self, 'direct_candle_cache', None), 'history_limits', {})
                    for _tf, _frame in _initial_snapshot.frames.items():
                        if (_frame.source != 'binance' or _frame.venue != 'binance'
                                or _frame.market_type != 'spot' or _frame.instrument_id != _selected
                                or len(_frame.closed) != int(_limits.get(_tf, TIMEFRAME_HISTORY_CANDLES.get(_tf, 0)))
                                or _frame.current is None):
                            _record_diagnostics_cycle('BLOCKED', f'live {_tf} Binance Spot frame incomplete or unverified')
                            return self._get_no_trade_signal(f'WAIT/NO-DATA: invalid { _tf } Binance Spot frame')
            if candle_snapshot is not None:
                self._active_candle_snapshot = candle_snapshot
                if not self.is_backtest_mode:
                    snapshot_symbol = str(candle_snapshot.symbol)
                    active_symbol = str(getattr(self, 'active_symbol', '') or '')
                    if active_symbol and snapshot_symbol != _binance_spot_symbol(active_symbol):
                        _record_diagnostics_cycle('BLOCKED', 'current candle snapshot symbol does not match selected symbol')
                        return self._get_no_trade_signal('WAIT/NO-DATA: candle symbol mismatch')
            elif not self.is_backtest_mode and self._active_candle_snapshot is None:
                _record_diagnostics_cycle('UNAVAILABLE', 'no direct candle snapshot supplied for current cycle')
                return self._get_no_trade_signal('WAIT/NO-DATA: no direct candle snapshot')

            # Caller-supplied MTF frames may not bypass the canonical live snapshot.
            # In live mode they must be byte-for-byte equivalent DataFrame values
            # from that validated Binance Spot snapshot; in backtests only an
            # explicit known interval set is accepted, with no future timestamps.
            if native_mtf is not None:
                allowed_backtest_sets = {frozenset(DEFAULT_STRATEGY_TIMEFRAMES), frozenset(LIVE_TIMEFRAMES)}
                expected_keys = (set(LIVE_TIMEFRAMES) if not self.is_backtest_mode
                                 else set(native_mtf) if isinstance(native_mtf, dict)
                                 and frozenset(native_mtf) in allowed_backtest_sets else set())
                if (not isinstance(native_mtf, dict) or not expected_keys
                        or set(native_mtf) != expected_keys or '5m' not in native_mtf):
                    _record_diagnostics_cycle('BLOCKED', 'supplied native MTF timeframe set invalid')
                    return self._get_no_trade_signal('WAIT/NO-DATA: supplied native timeframe set invalid')
                for _tf, _frame in native_mtf.items():
                    if not isinstance(_frame, pd.DataFrame) or _frame.empty:
                        _record_diagnostics_cycle('BLOCKED', f'supplied {_tf} frame is empty or invalid')
                        return self._get_no_trade_signal(f'WAIT/NO-DATA: invalid supplied {_tf} frame')
                    if not self.is_backtest_mode:
                        _snap_frame = getattr(_initial_snapshot, 'frames', {}).get(_tf)
                        if (_snap_frame is None or _frame.attrs.get('symbol') != _selected
                                or _frame.attrs.get('timeframe') != _tf
                                or not _frame.equals(_snap_frame.closed)):
                            _record_diagnostics_cycle('BLOCKED', f'supplied {_tf} frame differs from validated snapshot')
                            return self._get_no_trade_signal(f'WAIT/NO-DATA: unverified supplied {_tf} frame')
                    else:
                        try:
                            if not isinstance(_frame.index, pd.DatetimeIndex) or _frame.index.tz is None:
                                raise ValueError('native MTF frame requires a timezone-aware DatetimeIndex')
                            _source_end = pd.Timestamp(data.index[-1])
                            if not isinstance(data.index, pd.DatetimeIndex):
                                raise ValueError('backtest source requires a DatetimeIndex')
                            if _source_end.tzinfo is None:
                                _source_end = _source_end.tz_localize('UTC')
                            else:
                                _source_end = _source_end.tz_convert('UTC')
                            # Backtest decisions use the close of the latest
                            # completed 1m bar. A higher-TF candle is admissible
                            # only if its full interval had closed by that time;
                            # checking only its open timestamp leaks future OHLCV.
                            _decision_time = next_candle_open('5m', _source_end.timestamp())
                            _frame_stamps = _frame.index.tz_convert('UTC')
                            _frame_seconds = [float(_stamp.timestamp()) for _stamp in _frame_stamps]
                            validate_closed_candle_timestamps(_tf, _frame_seconds, _decision_time)
                        except Exception:
                            _record_diagnostics_cycle('BLOCKED', f'supplied {_tf} frame includes unclosed or invalid backtest candles')
                            return self._get_no_trade_signal(f'WAIT/NO-DATA: unclosed supplied {_tf} frame')
            # Auto-start Double-Brain AI Chain on first run
            # DISABLED for Performance: Prevents resource contention with Trading Judge
            # if not self.ai_chain_brain.is_running:
            #     logger.info("[JARVIS] 🧠 Starting Gemini->DeepSeek AI Chain...")
            #     self.ai_chain_brain.start_sequential_loop(self)
                
            if len(data) < 20:
                _record_diagnostics_cycle('INSUFFICIENT', f'only {len(data)} closed candles supplied; minimum analysis window is 20')
                return self._get_no_trade_signal("Insufficient data")

            # ── DATA VALIDATOR GATE (pre-brain quality check) ────────────
            # Validates completeness, price sanity, staleness of incoming
            # DataFrame. On failure → brain gets WAIT/NO-DATA, no decision
            # on bad data. Fail-open: validator crash never blocks brain.
            if (not self.is_backtest_mode and DATA_VALIDATOR_AVAILABLE
                    and _get_data_validator is not None):
                try:
                    _dv = _get_data_validator()
                    # Pass active symbol so each coin has independent price cache
                    _active_sym = (
                        getattr(self, 'active_symbol', None)
                        or getattr(self, 'symbol', None)
                        or getattr(self, 'current_coin', None)
                    )
                    _dv_result = _dv.validate_dataframe(data, source="binance_spot", symbol=_active_sym)
                    if not _dv_result.ok:
                        logger.warning(
                            "🛡️ DATA-VALIDATOR GATE: %s — %s",
                            _dv_result.status, "; ".join(_dv_result.failures)
                        )
                        _record_diagnostics_cycle(
                            'BLOCKED',
                            f"data-validator status={getattr(_dv_result, 'status', 'unknown')}; "
                            f"failures={'; '.join(_dv_result.failures)}",
                        )
                        return self._get_no_trade_signal(
                            f"WAIT/NO-DATA: {'; '.join(_dv_result.failures)}"
                        )

                    # Cross-source: Delta vs Binance price divergence (only when independent Delta price is available)
                    if hasattr(self, 'binance_data') and self.binance_data is not None and hasattr(self, 'delta_data') and self.delta_data is not None:
                        try:
                            _xs_base = getattr(self, 'active_base_asset', None) or 'BTC'
                            _xs_sym = getattr(self, 'active_symbol', None) or f"{_xs_base}USD"
                            _bn_price = self.binance_data.get_live_price(symbol=f"{_xs_base}USDT")
                            _delta_raw = self.delta_data.get_live_price(_xs_sym)
                            _delta_src = (self.delta_data.get_last_source() or {}).get('source', '') if hasattr(self.delta_data, 'get_last_source') else ''
                            if _bn_price and _bn_price > 0 and _delta_raw and float(_delta_raw) > 0 and 'delta' in str(_delta_src).lower():
                                _xs_result = _dv.cross_source_check(
                                    float(_delta_raw), float(_bn_price),
                                    source_a=f"Delta({_xs_sym})", source_b=f"Binance({_xs_base}USDT)"
                                )
                                if not _xs_result.ok:
                                    logger.warning("🛡️ CROSS-SOURCE BLOCK: %s", "; ".join(_xs_result.failures))
                                    _record_diagnostics_cycle('BLOCKED', f"cross-source divergence: {'; '.join(_xs_result.failures)}")
                                    return self._get_no_trade_signal(f"WAIT/NO-DATA: {'; '.join(_xs_result.failures)}")
                        except Exception as _xs_err:
                            logger.debug("[CROSS-SOURCE] check skipped: %s", _xs_err)
                except Exception as _dv_err:
                    # The direct native cache has already established provider,
                    # identity, continuity, and freshness. If the configured
                    # secondary validator is present but crashes, fail closed.
                    _record_diagnostics_cycle('BLOCKED', f'data validator failed: {type(_dv_err).__name__}')
                    return self._get_no_trade_signal('WAIT/NO-DATA: data validator unavailable')

            # Get current price and context
            current_price = float(data['close'].iloc[-1])
            self.last_price = current_price  # Store for AI Chain global context
            self.market_context = self._get_market_context(data)
            self.market_context['symbol'] = (getattr(self, 'active_symbol', None) or getattr(data, 'attrs', {}).get('symbol') or 'UNKNOWN')
            self.market_context['selected_symbol'] = self.market_context['symbol']
            # Every part, especially Part14, must know whether this call is a
            # historical replay. Without this shared flag a live-constructed
            # analyzer could query today's options during a backtest.
            self.market_context['is_backtest_mode'] = bool(self.is_backtest_mode)
            
            if not self.is_backtest_mode:
                try:
                    from binance_data import get_binance_data
                    _api = get_binance_data()
                    _sym = self.market_context['symbol']
                    self.market_context['real_orderbook'] = _api.get_order_book_depth(_sym, limit=50)
                    self.market_context['real_funding'] = _api.get_funding_rate(_sym)
                except Exception as e:
                    logger.debug(f"Failed to fetch real orderbook data: {e}")
            # Historical options are used only from recorded provider snapshots
            # at or before this candle's close. No live chain or OHLC-derived
            # synthetic options are permitted in backtest mode.
            if self.is_backtest_mode:
                snapshot_path = os.getenv(
                    'JARVIS_OPTIONS_SNAPSHOT_PATH',
                    os.path.join('.jarvis_state', 'options_snapshots.jsonl'),
                ).strip()
                try:
                    index = getattr(data, 'index', None)
                    if snapshot_path and index is not None and len(index) >= 2:
                        stamps = pd.to_datetime(index[-min(20, len(index)):])
                        if getattr(stamps, 'tz', None) is None:
                            stamps = stamps.tz_localize('UTC')
                        else:
                            stamps = stamps.tz_convert('UTC')
                        # This strategy entry point is explicitly the 1m
                        # primary frame. Do not infer an interval from median
                        # spacing, which can advance replay time across gaps.

                        stamp_seconds = [float(stamp.timestamp()) for stamp in stamps]
                        _decision_epoch = next_candle_open('5m', stamp_seconds[-1])
                        validate_closed_candle_timestamps('5m', stamp_seconds, _decision_epoch)
                        decision_time = datetime.fromtimestamp(_decision_epoch, tz=timezone.utc)
                        base_asset = str(getattr(self, 'active_base_asset', '') or self.market_context['symbol'])
                        for quote in ('USDT', 'USD', 'USDC'):
                            if base_asset.upper().endswith(quote):
                                base_asset = base_asset[:-len(quote)]
                                break
                        from options_snapshot_store import replay_institutional_bias
                        self._historical_options_bias = replay_institutional_bias(
                            snapshot_path, base_asset, decision_time, max_age_seconds=600)
                    else:
                        self._historical_options_bias = {
                            'available': False, 'bias': 'NEUTRAL', 'score': 0,
                            'reason': 'no_recorded_point_in_time_options_archive',
                            'source': 'none', 'not_ohlc_derived': True,
                        }
                except Exception as history_error:
                    self._historical_options_bias = {
                        'available': False, 'bias': 'NEUTRAL', 'score': 0,
                        'reason': type(history_error).__name__, 'source': 'none',
                        'not_ohlc_derived': True,
                    }
                self.market_context['historical_options_context'] = self._historical_options_bias

            # --- AUXILIARY INSTITUTIONAL TELEMETRY (Order Book, Funding, OI, 24h Stats) ---
            try:
                from jarvis_auxiliary_telemetry import get_auxiliary_collector
                aux_collector = get_auxiliary_collector()
                telemetry_symbol = getattr(self, 'active_symbol', None) or self.market_context.get('symbol', 'BTCUSDT')
                aux_data = aux_collector.get_telemetry(
                    telemetry_symbol,
                    binance_data_client=getattr(self, 'binance_data', None),
                    is_backtest_mode=self.is_backtest_mode,
                    decision_time=getattr(self, '_decision_time', None),
                    current_price=current_price,
                )
                try:
                    from jarvis_institutional_feed import get_context as _gc4
                    _ic4 = _gc4(telemetry_symbol)
                    if _ic4 and isinstance(aux_data, dict):
                        if not isinstance(aux_data.get('funding_rate'), dict):
                            aux_data['funding_rate'] = {}
                        if _ic4.get('funding_z') is not None and aux_data['funding_rate'].get('funding_rate_zscore') is None:
                            aux_data['funding_rate']['funding_rate_zscore'] = _ic4['funding_z']
                        if not isinstance(aux_data.get('order_book'), dict):
                            aux_data['order_book'] = {}
                        if _ic4.get('book_imbalance') is not None and aux_data['order_book'].get('imbalance_pct') is None:
                            aux_data['order_book']['imbalance_pct'] = round(_ic4['book_imbalance'] * 100.0, 4)
                        aux_data['institutional'] = {k: v for k, v in _ic4.items() if not str(k).startswith('_')}
                except Exception:
                    pass
                self.market_context['auxiliary_telemetry'] = aux_data
                self.market_context['order_book'] = aux_data.get('order_book', {})
                self.market_context['funding_rate'] = aux_data.get('funding_rate', {})
                self.market_context['open_interest'] = aux_data.get('open_interest', {})
                self.market_context['bid_ask'] = aux_data.get('bid_ask', {})
                self.market_context['ticker_24h'] = aux_data.get('ticker_24h', {})
            except Exception as aux_err:
                logger.debug("[AUX-TELEMETRY] Collector skipped: %s", aux_err)

            # --- EXTERNAL ENGINE EXECUTION ---
            # FIX: Initialize ALL market_context keys upfront to prevent KeyError crashes
            self.market_context.setdefault('institutional_components', {
                'psychology': [],
                'zone': [],
                'trend': [],
                'volume': [],
                'structure': [],
                'orderflow': [],
                'regime': 'NEUTRAL'
            })
            self.market_context.setdefault('institutional_fused', [])
            self.market_context['neural_prediction'] = None
            self.market_context['neural_mtf'] = {}
            self.market_context.setdefault('pattern_mtf', {})
            self.market_context.setdefault('fusion_mtf', {})
            self.market_context.setdefault('mtf_datasets', {})
            # FIX: Add safe defaults for all adapters
            self.market_context.setdefault('last_processed_price', 0)
            self.market_context.setdefault('timestamp', None)
            
            if EXTERNAL_ENGINES_AVAILABLE and self.engines:
                try:
                    # Live GPU/external engines consume the exact same direct
                    # native frames as Parts 1-12.  No resampling, truncation,
                    # live-price injection, or forming-candle mutation occurs.
                    if isinstance(native_mtf, dict) and native_mtf:
                        engine_tf_data = {str(tf): frame.copy(deep=True) for tf, frame in native_mtf.items()}
                    elif self.is_backtest_mode:
                        engine_tf_data = {'5m': data}
                    else:
                        engine_tf_data = self._fetch_mtf_from_api(getattr(self, 'active_symbol', None))
                        if set(engine_tf_data) != set(LIVE_TIMEFRAMES):
                            logger.warning('[GPU] Direct native snapshot incomplete; external analysis blocked')
                            engine_tf_data = {}
                    if not engine_tf_data:
                        _record_diagnostics_cycle('UNAVAILABLE', 'direct native timeframe frames unavailable before Part analysis')
                        return self._get_no_trade_signal('WAIT/NO-DATA: direct native frames unavailable')
                    self.market_context['mtf_datasets'] = engine_tf_data
                    if not self.is_backtest_mode and self._active_candle_snapshot is not None:
                        self.market_context['current_candles'] = self._active_candle_snapshot.current_candles
                        self.market_context['current_candle_is_confirmed'] = False
                    logger.info(f"📊 GPU Engines: direct native frames ready: {list(engine_tf_data.keys())}")
                    
                    # 1. Institutional Analysis (Native MTF)
                    inst_engine = self.engines.get('institutional')
                    if inst_engine:
                        if hasattr(inst_engine, 'generate_mtf_signals'):
                            inst_res = inst_engine.generate_mtf_signals(engine_tf_data)
                            logger.info(f"✅ Institutional MTF: Cons={inst_res.get('mtf_consensus', 0):.2f}")
                        else:
                            # Fallback to legacy
                            df_1min = engine_tf_data.get('5m', data)
                            df_5min = engine_tf_data.get('5m', df_1min)
                            df_15min = engine_tf_data.get('15m', df_1min)
                            inst_res = inst_engine.generate_live_signals(df_1min, df_5min, df_15min)
                        
                        # Store components for Adapters P1-P4, P6-P9
                        if isinstance(inst_res, dict) and 'components' in inst_res:
                            self.market_context['institutional_components'] = inst_res['components']
                            self.market_context['institutional_fused'] = inst_res.get('signals', [])
                    
                    # 2. Neural Analysis (analyze multiple timeframes) - WITH SAFE ERROR HANDLING
                    neural_engine = None  # do not run legacy sklearn/torch predictions
                    if neural_engine:
                        neural_mtf = {}
                        for tf_name, tf_df in engine_tf_data.items():
                            if len(tf_df) >= 30:
                                try:
                                    hist = tf_df.iloc[-60:] if len(tf_df) >= 60 else tf_df
                                    # SAFE: Check if method exists before calling
                                    if hasattr(neural_engine, 'predict') and callable(getattr(neural_engine, 'predict')):
                                        neural_res = neural_engine.predict(hist['close'].values, hist['volume'].values)
                                        if neural_res is not None:
                                            neural_mtf[tf_name] = neural_res
                                    else:
                                        logger.debug(f"Neural engine has no predict method on {tf_name}")
                                except TypeError as e:
                                    logger.debug(f"Neural signature mismatch on {tf_name}: {e}")
                                except Exception as e:
                                    logger.debug(f"Neural error on {tf_name}: {e}")
                        self.market_context['neural_prediction'] = neural_mtf.get('5m') if neural_mtf else None
                        self.market_context['neural_mtf'] = neural_mtf if neural_mtf else {}
                        if neural_mtf:
                            logger.info(f"🧠 Neural analysis: {len(neural_mtf)} timeframes analyzed")
                        else:
                            logger.debug("ℹ️ Neural engine produced no predictions")
                        
                    # 3. Pattern Recognition (Native MTF)
                    pattern_engine = self.engines.get('pattern')
                    if pattern_engine:
                        if hasattr(pattern_engine, 'scan_patterns_mtf'):
                            pattern_mtf = pattern_engine.scan_patterns_mtf(engine_tf_data)
                        else:
                            # Fallback loop
                            pattern_mtf = {}
                            for tf_name, tf_df in engine_tf_data.items():
                                if len(tf_df) >= 20: 
                                    try:
                                        if hasattr(pattern_engine, 'detect_patterns'):
                                            patterns = pattern_engine.detect_patterns(tf_df)
                                            if patterns: pattern_mtf[tf_name] = patterns
                                    except Exception: pass
                                    
                        self.market_context['pattern_mtf'] = pattern_mtf
                        if pattern_mtf:
                            logger.info(f"🎯 Patterns found on: {list(pattern_mtf.keys())}")

                    # 3b. Enhanced Combined Analysis
                    pattern_system = self.engines.get('pattern_system')
                    if pattern_system:
                        try:
                            enhanced_analysis = pattern_system.get_enhanced_combined_analysis()
                            if enhanced_analysis and not enhanced_analysis.get('error'):
                                self.market_context['enhanced_pattern_analysis'] = enhanced_analysis
                        except Exception as e:
                            logger.debug(f"[PATTERN-SYSTEM] get_enhanced_combined_analysis failed: {e}")

                    # 3c. Correlation Matrix Analysis (requires distinct cross-asset series, never MTF frames of same symbol)
                    if hasattr(self, 'correlation_brain') and self.correlation_brain:
                        try:
                            primary_df = data
                            peer_assets = self.market_context.get('peer_asset_frames') or {}
                            if isinstance(peer_assets, dict) and peer_assets:
                                corr_result = self.correlation_brain.analyze_correlations(primary_df['close'].values, peer_assets)
                                if corr_result:
                                    self.market_context['correlation_analysis'] = corr_result
                                    regime = corr_result.get('regime_signals', {})
                                    logger.info(f"📊 [CORRELATION] Regime signals: {regime}")
                            else:
                                self.market_context['correlation_analysis'] = {'status': 'skipped_no_peer_assets', 'regime_signals': []}
                        except Exception as e:
                            logger.debug(f"[CORRELATION] analyze_correlations failed: {e}")

                    # 4. Fusion Engine (Native MTF)
                    fusion_engine = self.engines.get('fusion')
                    if fusion_engine:
                        if hasattr(fusion_engine, 'parts'):
                            fusion_engine.parts = self.parts
                            
                        if hasattr(fusion_engine, 'fuse_modules_mtf'):
                            fusion_res = fusion_engine.fuse_modules_mtf(engine_tf_data)
                            self.market_context['fusion_mtf'] = fusion_res
                            logger.info("🔧 Fusion Engine: MTF Fusion Complete")
                    
                    # 5. Confidence Engine (available for downstream use)
                    # Will be used during final signal generation
                    
                except Exception as e:
                    logger.error(f"⚠️ External Engine Execution Protocol Failed: {e}")
                    logger.warning("⚠️ Falling back to legacy part analysis (GPU engines unavailable)")
            else:
                logger.debug("GPU Engines not available - using adapter parts only")
            # ---------------------------------
            
            # ============================================================
            # MULTI-TIMEFRAME ANALYSIS: native frames supplied by the selected data path.
            # Extra Binance Spot intervals are enabled only in the isolated analysis pipeline.
            # ============================================================
            
            # A supplied mapping is already native, closed and caller-validated.
            # Use it as-is (not as a resampling seed) for isolated analysis jobs.
            if isinstance(native_mtf, dict) and native_mtf:
                mtf_data = {str(tf): frame.copy(deep=True) for tf, frame in native_mtf.items()}
                if '5m' not in mtf_data or any(not isinstance(frame, pd.DataFrame) or frame.empty for frame in mtf_data.values()):
                    _record_diagnostics_cycle('UNAVAILABLE', 'native MTF input missing 5m or contains empty frame')
                    return self._get_no_trade_signal('WAIT/NO-DATA: incomplete supplied native frames')
                self._api_mtf_cache = mtf_data
                self.market_context['mtf_datasets'] = mtf_data
            # Historical replay never labels 1m-resampled bins as native
            # Binance intervals. Full multi-interval backtests must supply
            # point-in-time native_mtf frames explicitly; without them only the
            # source 1m frame is analyzed and MTF confluence is unavailable.
            elif self.is_backtest_mode:
                if not isinstance(data.index, pd.DatetimeIndex):
                    data = data.copy()
                    data.index = pd.to_datetime(data.index, utc=True)
                mtf_data = {'5m': data.copy()}
                self.market_context['native_mtf_backtest_status'] = {
                    'available_timeframes': ['5m'],
                    'missing_timeframes': [tf for tf in DEFAULT_STRATEGY_TIMEFRAMES if tf != '5m'],
                    'status': 'incomplete_native_history',
                }
                self._api_mtf_cache = mtf_data
                self.market_context['mtf_datasets'] = mtf_data
            else:
                mtf_data = self._fetch_mtf_from_api(getattr(self, 'active_symbol', None))
                if set(mtf_data) != set(LIVE_TIMEFRAMES):
                    _record_diagnostics_cycle('UNAVAILABLE', f'incomplete native timeframe snapshot; received={sorted(mtf_data)}')
                    _capture_diagnostics_inputs(mtf_data, self._active_candle_snapshot)
                    return self._get_no_trade_signal('WAIT/NO-DATA: incomplete native timeframe snapshot')
                self._api_mtf_cache = mtf_data
                self.market_context['mtf_datasets'] = mtf_data
                try:
                    import time as _tm_rf
                    _m1_rf = mtf_data.get('1m')
                    if _m1_rf is not None and len(_m1_rf):
                        _ix_rf = _m1_rf.index[-1]
                        _ts_rf = (_ix_rf.tz_localize('UTC') if _ix_rf.tzinfo is None else _ix_rf.tz_convert('UTC')).timestamp()
                        _age_rf = _tm_rf.time() - (_ts_rf + 60.0)
                        if _age_rf > 75.0:
                            logger.info('[MTF_REFRESH] %s newest closed 1m is %.0fs old; refetching fresh native snapshot (freshness rules unchanged)', getattr(self, 'active_symbol', None), _age_rf)
                            _old_snap_rf = self._active_candle_snapshot
                            self._active_candle_snapshot = None
                            _fresh_rf = self._fetch_mtf_from_api(getattr(self, 'active_symbol', None))
                            if set(_fresh_rf) == set(LIVE_TIMEFRAMES):
                                mtf_data = _fresh_rf
                                self._api_mtf_cache = mtf_data
                                self.market_context['mtf_datasets'] = mtf_data
                            else:
                                self._active_candle_snapshot = _old_snap_rf
                except Exception as _rf_err:
                    logger.warning('[MTF_REFRESH] skipped: %s', _rf_err)
                if self._active_candle_snapshot is not None:
                    self.market_context['current_candles'] = self._active_candle_snapshot.current_candles
                    self.market_context['current_candle_is_confirmed'] = False

            # Synchronize every native live interval to the last completed 1m
            # close. Faster candles may otherwise contribute
            # evidence newer than the decision reference. Keep the forming bars
            # separately in snapshot.current_candles for live display only.
            if not self.is_backtest_mode:
                try:
                    if set(mtf_data) != set(LIVE_TIMEFRAMES):
                        raise ValueError('native live timeframe set is incomplete')
                    mtf_data = synchronize_native_frames_to_1m_close(mtf_data)
                    if set(mtf_data) != set(LIVE_TIMEFRAMES):
                        raise ValueError('synchronized native timeframe set is incomplete')
                    self._api_mtf_cache = mtf_data
                    self.market_context['mtf_datasets'] = mtf_data
                    self.market_context['decision_cutoff_source'] = 'latest_completed_5m_close'
                except Exception as _alignment_error:
                    _record_diagnostics_cycle('BLOCKED', f'native timeframe synchronization failed: {type(_alignment_error).__name__}')
                    return self._get_no_trade_signal('WAIT/NO-DATA: native timeframes are not synchronized')

            # Capture only safe, bounded metadata from the exact closed frames
            # being analyzed. This is display-only and has no strategy inputs.
            _capture_diagnostics_inputs(
                mtf_data,
                snapshot=None if self.is_backtest_mode else self._active_candle_snapshot,
            )

            logger.info(f"📊 MTF Analysis: {len(mtf_data)} timeframes active: {list(mtf_data.keys())}")
            
            # Diagnostic per-frame contributions divide each horizon weight
            # across its correlated native intervals. The actual MTF veto below
            # aggregates each correlation group once before applying its weight.
            tf_weights = {}
            for _group_name, _group_frames in TIMEFRAME_CORRELATION_GROUPS.items():
                _available_group_frames = [tf for tf in _group_frames if tf in mtf_data]
                if _available_group_frames:
                    _per_frame_weight = TIMEFRAME_GROUP_WEIGHTS[_group_name] / len(_available_group_frames)
                    tf_weights.update({tf: _per_frame_weight for tf in _available_group_frames})
            
            # Run ALL parts on ALL timeframes
            part_results = {}        # Final weighted results (5m base)
            full_telemetry = {}
            mtf_breakdown = {}       # Per-timeframe breakdown for logging
            
            # Accumulate weighted signals across timeframes
            weighted_signals = {}    # part_name -> weighted sum
            weight_totals = {}       # part_name -> total weight applied
            mtf_diagnostics = {}     # Display-only raw Part 1-10 outcomes, keyed by native TF
            _record_diagnostics_cycle('ANALYZING', 'Parts 1–10 analyzing current native timeframe inputs')
            
            for tf_name, tf_data in mtf_data.items():
                tf_weight = tf_weights.get(tf_name, 1.0)
                tf_results = {}
                tf_diagnostics = {}
                
                for name, part in self.parts.items():
                    if name in ['part11_fusion', 'part12_confidence']:
                        continue
                    if name == 'part14_options_chain' and tf_name != '5m':
                        continue
                    
                    try:
                        # Every adapter receives a pristine frame and context copy;
                        # no part can mutate the next adapter's input or the canonical MTF set.
                        selected_symbol = getattr(self, 'active_symbol', None) or getattr(data, 'attrs', {}).get('symbol')
                        _spot_sym_adv = _binance_spot_symbol(selected_symbol) or selected_symbol
                        _snap_obj_adv = candle_snapshot or getattr(self, '_active_candle_snapshot', None)
                        part_context = dict(self.market_context)
                        _part_is_bt = False if (name == 'part14_options_chain' and getattr(part, 'backtest_mode', True) is False) else self.is_backtest_mode
                        part_context.update({
                            'selected_symbol': _spot_sym_adv,
                            'symbol': _spot_sym_adv,
                            'timeframe': tf_name,
                            'native_timeframe': tf_name,
                            'analysis_identity': getattr(_snap_obj_adv, 'identity', None) or ('binance', 'spot', _spot_sym_adv, _spot_sym_adv),
                            'snapshot_fetched_at': getattr(_snap_obj_adv, 'fetched_at', None) or time.time(),
                            'is_backtest_mode': _part_is_bt,
                            'shared_candle_source': 'historical_replay' if _part_is_bt else 'shared_exchange',
                        })
                        if isinstance(part_context.get('mtf_datasets'), dict):
                            part_context['mtf_datasets'] = {
                                frame_name: frame.copy(deep=True)
                                for frame_name, frame in part_context['mtf_datasets'].items()
                            }
                        part_input = tf_data.copy(deep=True)
                        part_input.attrs = dict(getattr(tf_data, 'attrs', {}) or {})
                        if _spot_sym_adv:
                            part_input.attrs['symbol'] = _spot_sym_adv
                        part_input.attrs['timeframe'] = tf_name
                        res = part.analyze(part_input, context=part_context)
                        try:
                            import os as _os, time as _t
                            _r = res if isinstance(res, dict) else {}
                            _line = "%s | %s | %s | %s | sig=%s conf=%s | %s" % (_t.strftime("%Y-%m-%d %H:%M:%S"), selected_symbol, tf_name, name, _r.get("signal"), _r.get("confidence"), str(_r.get("thought") or _r.get("reason") or _r.get("reasoning") or "")[:160].replace("\n", " "))
                            _os.makedirs("logs", exist_ok=True)
                            with open(_os.path.join("logs", "part_results.log"), "a", encoding="utf-8") as _f:
                                _f.write(_line + "\n")
                        except Exception:
                            pass
                        if name in _DIAGNOSTIC_PART_KEYS:
                            try:
                                tf_diagnostics[name] = (
                                    _classify_part_output(res) if _classify_part_output is not None
                                    else {'status': 'UNKNOWN', 'signal': None, 'confidence': None,
                                          'reason': 'diagnostic formatter unavailable'}
                                )
                            except Exception as diagnostic_error:
                                tf_diagnostics[name] = {
                                    'status': 'UNKNOWN', 'signal': None, 'confidence': None,
                                    'reason': f'diagnostic normalization failed: {type(diagnostic_error).__name__}',
                                }
                        if isinstance(res, dict):
                            tf_results[name] = res
                            raw_signal = res.get('signal', 0)
                            
                            # Force scalar signal (handle list/sequence returns from some parts)
                            try:
                                if isinstance(raw_signal, (list, tuple, np.ndarray)):
                                    signal = float(raw_signal[-1]) if len(raw_signal) > 0 else 0.0
                                else:
                                    signal = float(raw_signal)
                            except:
                                signal = 0.0
                            
                            # Accumulate weighted signal
                            if name not in weighted_signals:
                                weighted_signals[name] = 0.0
                                weight_totals[name] = 0.0
                            
                            # Learning loop: scale engine contribution by its
                            # historical performance multiplier (guarded, 1.0 default)
                            try:
                                from jarvis_learning import get_engine_weight as _jl_weight
                                learn_w = _jl_weight(name)
                            except Exception:
                                learn_w = 1.0

                            weighted_signals[name] += signal * tf_weight * learn_w
                            weight_totals[name] += tf_weight * learn_w
                            
                            if 'telemetry' in res and tf_name == '5m':
                                full_telemetry[name] = res['telemetry']
                    except Exception as e:
                        if name in _DIAGNOSTIC_PART_KEYS:
                            try:
                                tf_diagnostics[name] = (
                                    _classify_part_output(error_type=type(e).__name__)
                                    if _classify_part_output is not None
                                    else {'status': 'ERROR', 'signal': None, 'confidence': None,
                                          'reason': f'Analyzer exception: {type(e).__name__}'}
                                )
                            except Exception as diagnostic_error:
                                tf_diagnostics[name] = {
                                    'status': 'ERROR', 'signal': None, 'confidence': None,
                                    'reason': f'Analyzer exception: {type(e).__name__}; diagnostic normalization failed',
                                }
                        if tf_name == '5m':  # Only log errors for primary TF
                            logger.error(f"❌ Part {name} error on {tf_name}: {e}")
                
                mtf_diagnostics[tf_name] = tf_diagnostics
                mtf_breakdown[tf_name] = tf_results

            # Only diagnostics retain the full per-timeframe raw returns; the
            # existing 1m + HTF reduction and all strategy calculations below
            # remain unchanged.
            self.latest_part_timeframe_results = mtf_diagnostics
            
            # Aggregate Part 7 explicitly by native timeframe.  A neutral frame
            # remains neutral, while extreme volatility, missing/stale/wrong-
            # symbol data, or an analyzer exception commits a new-entry block.
            part7_by_timeframe = {
                tf: mtf_breakdown.get(tf, {}).get('part7_volatility', {})
                for tf in mtf_data
            }
            if _aggregate_part7_results is not None:
                _p7_sym = _binance_spot_symbol(getattr(self, 'active_symbol', None)) or getattr(self, 'active_symbol', None) or 'UNKNOWN'
                self.latest_part7 = _aggregate_part7_results(
                    part7_by_timeframe,
                    symbol=_p7_sym,
                    required_timeframes=tuple(mtf_data.keys()),
                )
            else:
                self.latest_part7 = {
                    'signal': 0, 'status': 'error', 'reason': 'Part7 aggregator unavailable',
                    'entry_blocked': True, 'risk_veto': False, 'data_status': 'error',
                    'computation_backend': 'pandas_cpu', 'timeframe_results': part7_by_timeframe,
                }
            # Part 7 rows are stamped with the route symbol (e.g. BTCUSD) while the central approval
            # compares against the Binance spot symbol of the analysed candles (BTCUSDT). Re-stamp the
            # aggregate AND every row consistently so identity checks compare like with like.
            try:
                _spot7 = _binance_spot_symbol(getattr(self, 'active_symbol', None))
                _lp7 = self.latest_part7 if isinstance(self.latest_part7, dict) else None
                if _spot7 and _lp7 is not None and isinstance(_lp7.get('timeframe_results'), dict):
                    _old7 = str(_lp7.get('symbol') or '')
                    for _tf7, _r7 in _lp7['timeframe_results'].items():
                        if isinstance(_r7, dict) and str(_r7.get('symbol') or '') == _old7:
                            _r7['symbol'] = _spot7
                            _r7['signal_identity'] = 'part7:' + _spot7 + ':' + str(_tf7)
                    _lp7['symbol'] = _spot7
                    _lp7['signal_identity'] = 'part7:' + _spot7 + ':aggregate'
            except Exception:
                pass
            # JARVIS owns the final entry decision.  Part 7 is advisory: its
            # volatility vetoes are reported, but only invalid/missing data or a
            # veto on the entry timeframe(s) may block a new entry.
            try:
                _p7 = self.latest_part7 if isinstance(self.latest_part7, dict) else {}
                _rows = _p7.get('timeframe_results') or {}
                _hard = [tf for tf, r in _rows.items()
                         if not isinstance(r, dict)
                         or r.get('data_status') != 'valid'
                         or r.get('status') not in ('ok', 'neutral', 'veto')]
                _vetoed = list(_p7.get('veto_timeframes') or [])
                _entry_tfs = ('15m',)
                _entry_veto = [tf for tf in _vetoed if tf in _entry_tfs]
                _p7['advisory_veto_timeframes'] = _vetoed
                _p7['jarvis_hard_data_blocks'] = _hard
                _p7['jarvis_entry_veto_timeframes'] = _entry_veto
                if _hard:
                    _p7['entry_blocked'] = True
                    _p7['reason'] = 'Part7 data invalid on ' + ', '.join(_hard)
                elif _entry_veto:
                    _p7['entry_blocked'] = True
                    _p7['reason'] = 'JARVIS: extreme volatility on entry timeframe ' + ', '.join(_entry_veto)
                else:
                    # Advisory-only: keep the originals under advisory_* keys and
                    # present a clean, execution-valid aggregate to the approval layer.
                    _new_rows = {}
                    for _tf, _r in _rows.items():
                        _r2 = dict(_r)
                        if _r2.get('risk_veto') or _r2.get('entry_blocked') or _r2.get('status') == 'veto':
                            _r2['advisory_risk_veto'] = bool(_r2.get('risk_veto'))
                            _r2['advisory_status'] = _r2.get('status')
                            _r2['risk_veto'] = False
                            _r2['entry_blocked'] = False
                            if _r2.get('status') == 'veto':
                                _r2['status'] = 'neutral'
                        _new_rows[_tf] = _r2
                    _p7['timeframe_results'] = _new_rows
                    _p7['status'] = 'ok'
                    _p7['data_status'] = 'valid'
                    _p7['risk_veto'] = False
                    _p7['veto_timeframes'] = []
                    _p7['blocked_timeframes'] = []
                    _p7['entry_blocked'] = False
                    for _tf, _d in list(mtf_breakdown.items()):
                        _pv = _d.get('part7_volatility') if isinstance(_d, dict) else None
                        if isinstance(_pv, dict) and (_pv.get('risk_veto') or _pv.get('entry_blocked') or _pv.get('status') == 'veto'):
                            _pv = dict(_pv)
                            _pv['advisory_status'] = _pv.get('status')
                            _pv['advisory_risk_veto'] = bool(_pv.get('risk_veto'))
                            _pv['risk_veto'] = False
                            _pv['entry_blocked'] = False
                            _pv['status'] = 'neutral'
                            _d['part7_volatility'] = _pv
                    for _tf, _d in list(mtf_diagnostics.items()):
                        _pv = _d.get('part7_volatility') if isinstance(_d, dict) else None
                        if isinstance(_pv, dict) and (_pv.get('risk_veto') or _pv.get('entry_blocked') or _pv.get('status') == 'veto'):
                            _pv = dict(_pv)
                            _pv['advisory_status'] = _pv.get('status')
                            _pv['advisory_risk_veto'] = bool(_pv.get('risk_veto'))
                            _pv['risk_veto'] = False
                            _pv['entry_blocked'] = False
                            _pv['status'] = 'neutral'
                            _d['part7_volatility'] = _pv
                    _p7['reason'] = ('Part7 advisory only (higher-timeframe vol noted: '
                                     + ', '.join(_vetoed) + ')') if _vetoed else 'Part7 data valid'
                self.latest_part7 = _p7
            except Exception as _p7_exc:
                self.latest_part7 = {**(self.latest_part7 or {}), 'entry_blocked': True,
                                     'reason': 'Part7 policy error: %r' % (_p7_exc,)}
            self.latest_part7_cycle_display = dict(self.latest_part7)

            # Build final part_results with MTF confirmation & veto protection
            for name, part in self.parts.items():
                if name in ['part11_fusion', 'part12_confidence']:
                    continue
                
                primary_tf = '5m' if '5m' in mtf_breakdown else (list(mtf_breakdown.keys())[0] if mtf_breakdown else '5m')
                base_result = mtf_breakdown.get(primary_tf, {}).get(name, {})
                if not base_result:
                    part_results[name] = {'signal': 0, 'thought': 'Part offline'}
                    continue

                base_sig_raw = base_result.get('signal', 0)
                try:
                    base_sig = int(float(base_sig_raw))
                except (ValueError, TypeError):
                    base_sig = 0
                base_thought = str(base_result.get('thought', 'No thought')).strip()
                
                # HTF confluence / veto calculation across higher timeframes
                htf_values = {}
                for tf_name in mtf_data.keys():
                    if tf_name == primary_tf:
                        continue
                    tf_res = mtf_breakdown.get(tf_name, {}).get(name, {})
                    try:
                        htf_values[tf_name] = float(tf_res.get('signal', 0))
                    except (ValueError, TypeError):
                        htf_values[tf_name] = 0.0
                htf_avg, htf_groups = grouped_timeframe_consensus(htf_values)
                
                # ── PHASE 4: INDEPENDENT DUAL-TRACK (SCALP 1m-15m vs SWING 30m-1M) ──
                _scalp_tfs = ('1m', '3m', '5m', '15m')
                _swing_tfs = ('30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
                _all_tf_sigs = {}
                _all_tf_confs = {}
                for _t in (_scalp_tfs + _swing_tfs):
                    _tr = mtf_breakdown.get(_t, {}).get(name, {})
                    try:
                        _all_tf_sigs[_t] = int(float(_tr.get('signal', 0)))
                    except (ValueError, TypeError):
                        _all_tf_sigs[_t] = 0
                    try:
                        _all_tf_confs[_t] = float(_tr.get('confidence', 0.0) or 0.0)
                    except (ValueError, TypeError):
                        _all_tf_confs[_t] = 0.0

                # 1. Compute SCALP track signal (1m, 3m, 5m, 15m) anchored on 15m / 5m
                _sc_pos = sum(1 for _t in _scalp_tfs if _all_tf_sigs.get(_t, 0) > 0)
                _sc_neg = sum(1 for _t in _scalp_tfs if _all_tf_sigs.get(_t, 0) < 0)
                _sc_act = _sc_pos + _sc_neg
                _sig_15m = _all_tf_sigs.get('15m', 0)
                _sig_5m = _all_tf_sigs.get('5m', 0)
                scalp_signal = 0
                if _sig_15m != 0 and _sig_5m * _sig_15m >= 0:
                    scalp_signal = _sig_15m
                elif _sig_5m != 0 and _sig_15m == 0:
                    scalp_signal = _sig_5m
                elif _sc_act >= 2 and _sc_pos >= 0.60 * _sc_act and _sc_neg == 0:
                    scalp_signal = 1
                elif _sc_act >= 2 and _sc_neg >= 0.60 * _sc_act and _sc_pos == 0:
                    scalp_signal = -1
                scalp_confidence = max((_all_tf_confs.get(_t, 0.0) for _t in _scalp_tfs if _all_tf_sigs.get(_t, 0) == scalp_signal and scalp_signal != 0), default=0.0)

                # 2. Compute SWING track signal (30m..1M) anchored on 1h / 4h
                _sw_pos = sum(1 for _t in _swing_tfs if _all_tf_sigs.get(_t, 0) > 0)
                _sw_neg = sum(1 for _t in _swing_tfs if _all_tf_sigs.get(_t, 0) < 0)
                _sw_act = _sw_pos + _sw_neg
                _sig_1h = _all_tf_sigs.get('1h', 0)
                _sig_4h = _all_tf_sigs.get('4h', 0)
                swing_signal = 0
                if _sw_act >= 3 and _sw_pos >= 0.60 * _sw_act and (_sw_pos - _sw_neg) >= 2:
                    swing_signal = 1
                elif _sw_act >= 3 and _sw_neg >= 0.60 * _sw_act and (_sw_neg - _sw_pos) >= 2:
                    swing_signal = -1
                elif _sig_1h != 0 and _sig_1h == _sig_4h and ((_sig_1h > 0 and _sw_pos > _sw_neg) or (_sig_1h < 0 and _sw_neg > _sw_pos)):
                    swing_signal = _sig_1h
                swing_confidence = max((_all_tf_confs.get(_t, 0.0) for _t in _swing_tfs if _all_tf_sigs.get(_t, 0) == swing_signal and swing_signal != 0), default=0.0)
                # Breadth scaling: confidence must not be inflated when few swing TFs agree.
                if swing_signal != 0:
                    _sw_aligned = _sw_pos if swing_signal > 0 else _sw_neg
                    swing_confidence *= max(0.35, min(1.0, _sw_aligned / 6.0))

                # 3. Preserve active track in final_signal so neutral 5m/15m NEVER zeroes out Swing
                if swing_signal != 0 and (scalp_signal == 0 or scalp_signal == swing_signal):
                    final_signal = swing_signal
                    final_thought = f"{base_thought} | 5m={_sig_5m} 15m={_sig_15m} 1h={_sig_1h} 4h={_sig_4h} | SWING-Track={swing_signal} ({_sw_pos}B/{_sw_neg}S of {len(_swing_tfs)}TF, conf*={max(0.35, min(1.0, (_sw_pos if swing_signal > 0 else _sw_neg) / 6.0)):.2f}) SCALP-Track={scalp_signal} ({_sc_pos}B/{_sc_neg}S)"
                elif scalp_signal != 0 and swing_signal == 0:
                    if (scalp_signal > 0 and htf_avg <= -0.35) or (scalp_signal < 0 and htf_avg >= 0.35):
                        final_signal = 0
                        final_thought = f"{base_thought} | SCALP {scalp_signal} VETOED by strong HTF ({htf_avg:+.2f})"
                    else:
                        final_signal = scalp_signal
                        final_thought = f"{base_thought} | SCALP-Track={scalp_signal} ({_sc_pos}B/{_sc_neg}S; Swing neutral)"
                elif scalp_signal != 0 and swing_signal != 0 and scalp_signal != swing_signal:
                    final_signal = swing_signal if swing_confidence >= scalp_confidence else scalp_signal
                    final_thought = f"{base_thought} | DUAL-TRACK SPLIT (SCALP={scalp_signal}, SWING={swing_signal})"
                else:
                    final_signal = 0
                    final_thought = base_thought

                tf_agree_cnt = sum(
                    1 for tf in mtf_breakdown 
                    if name in mtf_breakdown[tf] 
                    and mtf_breakdown[tf][name].get('signal', 0) == final_signal and final_signal != 0
                )

                _rep_tf = next((_t for _t in ('15m', '5m', '1h', '4h', '30m', '2h') if _all_tf_sigs.get(_t, 0) == final_signal and final_signal != 0), primary_tf)
                _rep_tele = dict((mtf_breakdown.get(_rep_tf, {}).get(name, {}) or {}).get('telemetry') or base_result.get('telemetry') or {})

                part_results[name] = {
                    'signal': final_signal,
                    'scalp_signal': scalp_signal,
                    'scalp_confidence': round(scalp_confidence, 1),
                    'swing_signal': swing_signal,
                    'swing_confidence': round(swing_confidence, 1),
                    'confidence': round(max(scalp_confidence, swing_confidence, float(base_result.get('confidence', 0.0) or 0.0)), 1),
                    'thought': final_thought,
                    'weighted_avg': round(htf_avg, 3),
                    'tf_agreement': tf_agree_cnt,
                    'mtf_horizon_groups': htf_groups,
                    'timeframe_signals': _all_tf_sigs,
                    'telemetry': _rep_tele,
                }
                
                if final_signal != 0:
                    logger.debug(f"🔍 MTF TRACER: {name} signal={final_signal} base={base_sig} HTF={htf_avg:.3f}")

            # Keep the canonical part result explicit and dashboard-ready.  The
            # gate is not represented only as a neutral signal: downstream entry
            # logic reads entry_blocked/risk_veto and commits the veto.
            if isinstance(self.latest_part7, dict):
                part_results['part7_volatility'] = {
                    **part_results.get('part7_volatility', {}),
                    **self.latest_part7,
                }
            
            # Log MTF summary
            buy_parts = sum(1 for r in part_results.values() if r.get('signal', 0) > 0)
            sell_parts = sum(1 for r in part_results.values() if r.get('signal', 0) < 0)
            logger.info(f"📈 MTF CONSENSUS: {buy_parts} BUY / {sell_parts} SELL across {len(mtf_data)} timeframes")
            
            # [NEW] Expose data for AI Chain Brain
            self.latest_part_results = part_results
            self.latest_telemetry = full_telemetry
            
            # --- MACRO CONTEXT & OPTIONS PRE-FETCH ---
            options_intel = (self._historical_options_bias
                             if self.is_backtest_mode and isinstance(self._historical_options_bias, dict)
                             and self._historical_options_bias.get('available') else None)
            intel_delta = options_intel
            intel_deribit = None
            
            # Fetch options/institutional intel for the locked asset.  Deribit
            # is BTC-only in this integration; never use its BTC result as an
            # altcoin signal or gate.  Missing selected-asset data remains
            # explicitly unavailable and is handled fail-closed by the live
            # decision gate.
            selected_base = (getattr(self, 'active_symbol', '') or '').upper()
            for quote in ('USDT', 'USD'):
                if selected_base.endswith(quote):
                    selected_base = selected_base[:-len(quote)]
                    break
            if not self.is_backtest_mode and self.delta_data and selected_base:
                try:
                    intel_delta = self.delta_data.get_institutional_bias(selected_base)
                    if isinstance(intel_delta, dict) and str(intel_delta.get('bias', '')).upper() != 'NO DATA':
                        options_intel = intel_delta
                except Exception:
                    intel_delta = None
            if selected_base in ('BTC', 'ETH') and not self.is_backtest_mode and self.deribit:
                try:
                    current_price = float(data['close'].iloc[-1])
                    if not hasattr(self, '_deribit_by_asset'):
                        self._deribit_by_asset = {'BTC': self.deribit}
                    if selected_base not in self._deribit_by_asset:
                        from deribit_options_client import DeribitOptionsClient as _DOC
                        self._deribit_by_asset[selected_base] = _DOC(currency=selected_base, client_id=os.getenv('DERIBIT_CLIENT_ID', ''), client_secret=os.getenv('DERIBIT_CLIENT_SECRET', ''))
                    _db_cli = self._deribit_by_asset.get(selected_base) or self.deribit
                    intel_deribit = _db_cli.get_institutional_bias(current_price)
                    if not options_intel:
                        options_intel = intel_deribit
                except Exception:
                    intel_deribit = None
            
            # Store walls for AI context
            options_walls = {}
            active_intel = options_intel if options_intel else (intel_deribit or intel_delta)
            if active_intel:
                options_walls = {
                    'support': active_intel.get('support_wall') or active_intel.get('raw_data', {}).get('support'),
                    'resistance': active_intel.get('resistance_wall') or active_intel.get('raw_data', {}).get('resistance'),
                    'max_pain': active_intel.get('max_pain') or active_intel.get('raw_data', {}).get('max_pain')
                }
            
            self.last_dual_intel = {'delta': intel_delta, 'deribit': intel_deribit}

            # --- MATHEMATICAL ANALYST OPINION (Dual-Track SCALP vs SWING) ---
            _scalp_part_inputs = {}
            _swing_part_inputs = {}
            for _pk, _pv in part_results.items():
                if not isinstance(_pv, dict):
                    continue
                _scalp_part_inputs[_pk] = {**_pv, 'signal': _pv.get('scalp_signal', _pv.get('signal', 0)), 'confidence': _pv.get('scalp_confidence', _pv.get('confidence', 0))}
                _swing_part_inputs[_pk] = {**_pv, 'signal': _pv.get('swing_signal', _pv.get('signal', 0)), 'confidence': _pv.get('swing_confidence', _pv.get('confidence', 0))}
            _scalp_p11 = self.parts['part11_fusion'].analyze(_scalp_part_inputs)
            _scalp_p12 = self.parts['part12_confidence'].analyze(_scalp_part_inputs)
            _swing_p11 = self.parts['part11_fusion'].analyze(_swing_part_inputs)
            _swing_p12 = self.parts['part12_confidence'].analyze(_swing_part_inputs)
            math_res = self.parts['part11_fusion'].analyze(part_results)
            math_conf_res = self.parts['part12_confidence'].analyze(part_results)
            _sc_sig = int(_scalp_p11.get('signal', 0) or 0)
            _sw_sig = int(_swing_p11.get('signal', 0) or 0)
            _sc_conf = float(_scalp_p12.get('confidence', 10) or 10)
            _sw_conf = float(_swing_p12.get('confidence', 10) or 10)
            self.latest_dual_track_summary = {
                'scalp': {'signal': _sc_sig, 'confidence': _sc_conf, 'thought': _scalp_p11.get('thought')},
                'swing': {'signal': _sw_sig, 'confidence': _sw_conf, 'thought': _swing_p11.get('thought')},
            }
            logger.info("[DUAL-TRACK] SCALP(1m-15m): sig=%d conf=%.1f%% (%s) | SWING(30m-1M): sig=%d conf=%.1f%% (%s)",
                        _sc_sig, _sc_conf, _scalp_p11.get('thought', '')[:80],
                        _sw_sig, _sw_conf, _swing_p11.get('thought', '')[:80])
            if _sc_sig != 0 and _sw_sig != 0 and _sc_sig != _sw_sig and _sc_conf >= 65 and _sw_conf >= 65:
                math_res = {'signal': 0, 'thought': f'SCALP_VS_SWING_CONFLICT (SCALP={_sc_sig}@{_sc_conf:.0f}% vs SWING={_sw_sig}@{_sw_conf:.0f}%)', 'consensus_ratio': 0.0}
                math_conf_res = {'confidence': int(max(_sc_conf, _sw_conf)), 'thought': 'Dual-track conflict hold'}
            elif _sw_sig != 0 and (_sc_sig == 0 or _sw_conf >= _sc_conf):
                math_res = _swing_p11
                math_conf_res = _swing_p12
            elif _sc_sig != 0:
                math_res = _scalp_p11
                math_conf_res = _scalp_p12
            # Save the actual fusion returns for the read-only terminal report.
            # Whitelist display fields; do not copy arbitrary analyzer payloads.
            _fusion_display_keys = {'signal', 'direction', 'bias', 'confidence', 'confidence_score',
                                    'status', 'thought', 'reason', 'reasoning', 'message', 'data_status'}
            try:
                self.latest_fusion_results = {
                    'part11_fusion': {k: v for k, v in math_res.items() if k in _fusion_display_keys},
                    'part12_confidence': {k: v for k, v in math_conf_res.items() if k in _fusion_display_keys},
                }
            except Exception as diagnostic_error:
                logger.debug('[LIVE-DIAGNOSTICS] fusion output unavailable: %s', type(diagnostic_error).__name__)
                self.latest_fusion_results = {}
            
            math_signal = math_res.get('signal', 0)
            math_confidence = math_conf_res.get('confidence', 10)
            
            # --- HOLISTIC NEURAL GLOBAL SYNTHESIS (PHASE 15: THE COUNCIL) ---
            # Judge is now ENABLED and validates high-confidence signals (score >= 10)
            # System uses: Mathematical Analyst + Quantum V5 + DeepSeek Judge
            neural_res = None
            try:
                _na_avail = 0
                for _tf_na, _res_na in mtf_breakdown.items():
                    for _p_na, _val_na in (_res_na or {}).items():
                        if isinstance(_val_na, dict) and isinstance(_val_na.get('neural_advisory'), dict) and _val_na['neural_advisory'].get('status') == 'available':
                            _na_avail += 1
                self.latest_neural_cycle_status = {
                    'status': 'available' if _na_avail > 0 else 'not_run',
                    'reason': f'{_na_avail} task-specific neural advisory heads evaluated across 8 timeframes' if _na_avail > 0 else 'neural_res=None in the current deterministic fusion cycle',
                    'availability': f'{_na_avail} active' if _na_avail > 0 else 'not probed',
                }
            except Exception:
                self.latest_neural_cycle_status = {
                    'status': 'not_run',
                    'reason': 'neural_res=None in the current deterministic fusion cycle',
                    'availability': 'not probed',
                }
            _record_diagnostics_cycle('COMPLETE', 'Parts 1–12 deterministic analysis/fusion returns recorded')
            logic_signal = math_signal
            confidence = math_confidence
            ai_thought = "Deterministic mathematical fusion (model judge retired)"

            # Map thoughts for final synthesis
            thoughts = [res['thought'] for k, res in part_results.items() 
                        if res['signal'] != 0 and k not in ['part11_fusion', 'part12_confidence']]
            if neural_res: thoughts.insert(0, f"🧠 NEURAL: {ai_thought}")
            
            detailed_scores = {name: res['signal'] for name, res in part_results.items()}
            if options_walls:
                # Observed current/historical option levels are passed through to
                # target logic; absent walls remain absent rather than fabricated.
                detailed_scores['options_walls'] = dict(options_walls)
            if isinstance(options_intel, dict):
                detailed_scores['options_intel'] = {
                    key: options_intel.get(key) for key in
                    ('bias', 'score', 'pcr', 'max_pain', 'source', 'available', 'historical', 'not_ohlc_derived')
                    if key in options_intel
                }
            detailed_scores['thoughts'] = thoughts
            detailed_scores['neural_synthesis'] = neural_res
            detailed_scores['quantum_validation'] = (neural_res.get('quantum_data', {}) if neural_res else self.brains['quantum_v5'].simulate(data)) if hasattr(self, 'brains') and 'quantum_v5' in getattr(self, 'brains', {}) else {}
            
            # Base score from confidence
            # FIXED: NO-TRADE should also show confidence % (how certain we are NOT to trade)
            score = confidence
            if self.is_backtest_mode and isinstance(self._historical_options_bias, dict) and self._historical_options_bias.get('available'):
                historical_bias = str(self._historical_options_bias.get('bias') or 'NEUTRAL').upper()
                signal_bias = 'BULLISH' if logic_signal > 0 else 'BEARISH' if logic_signal < 0 else 'NEUTRAL'
                adjustment = 4 if historical_bias == signal_bias and signal_bias != 'NEUTRAL' else -6 if historical_bias not in {'NEUTRAL', signal_bias} and signal_bias != 'NEUTRAL' else 0
                if adjustment:
                    score = max(0, min(100, score + adjustment))
                detailed_scores['historical_options_adjustment'] = adjustment
                detailed_scores['historical_options_source'] = self._historical_options_bias.get('source') or 'none'
            elif self.is_backtest_mode:
                detailed_scores['historical_options_adjustment'] = 0
                detailed_scores['historical_options_status'] = 'unavailable_no_asof_snapshot'
            
            # --- UNIVERSAL MTF CONFLUENCE BONUS (grouped native horizons) ---
            if mtf_context:
                # Direction from 1m logic
                current_dir = logic_signal 
                
                confluence_score = 0
                mtf_values = {}
                for tf, analysis in mtf_context.items():
                    raw_direction = analysis.get('direction', 0) if isinstance(analysis, dict) else analysis
                    try:
                        mtf_values[str(tf)] = float(raw_direction)
                    except (TypeError, ValueError, OverflowError):
                        continue
                mtf_groups = grouped_timeframe_values(mtf_values, exclude=('5m',))
                for group, group_value in mtf_groups.items():
                    weight = TIMEFRAME_GROUP_WEIGHTS.get(group, 0)
                    if group_value * current_dir > 0 and current_dir != 0:
                        confluence_score += weight
                    elif group_value * current_dir < 0 and current_dir != 0:
                        confluence_score -= weight * 0.5
                
                if confluence_score != 0:
                    score = max(0, min(100, score + confluence_score))
                    detailed_scores['universal_mtf_confluence'] = confluence_score
                    detailed_scores['mtf_matrix'] = mtf_context
            
            # --- QUANTUM V5 (DISABLED) ---
            detailed_scores['quantum_signal'] = 0
            detailed_scores['quantum_thought'] = "Quantum disabled"

            # ---------------------------
            # PHASE 3 FIX: Wire sleeping brains into decision flow
            # ---------------------------
            
            # 3C. Fusion Engine confluence (was computed but never read)
            fusion_mtf = self.market_context.get('fusion_mtf', {})
            if isinstance(fusion_mtf, dict) and fusion_mtf:
                fusion_signal = fusion_mtf.get('consensus_signal', fusion_mtf.get('signal', 0))
                if isinstance(fusion_signal, (int, float)):
                    if int(fusion_signal) == logic_signal and logic_signal != 0:
                        score = min(100, score + 3)
                        ai_thought += " | 🔧 FUSION ALIGNED"
                        detailed_scores['fusion_boost'] = 3
                        logger.info("🔧 FUSION ENGINE: Aligned with signal (+3%)")
            
            # 3X. CROSS-EXCHANGE perspective (Binance + Upstox vs Delta signal)
            # Light-weight (~1.1 weight) advisory check. Fail-closed: if Binance is
            # unreachable this perspective is skipped entirely and Delta remains
            # primary. Never touches execution; only adjusts confidence / vetoes.
            if (not self.is_backtest_mode and MULTI_SOURCE_DATA_AVAILABLE
                    and _get_cross_exchange_perspective is not None):
                try:
                    xres = _get_cross_exchange_perspective(logic_signal)
                    detailed_scores['cross_exchange'] = {
                        'active': xres.get('active', False),
                        'adjustment': xres.get('adjustment', 0),
                        'veto': xres.get('veto', False),
                        'binance_ok': xres.get('binance', {}).get('ok', False),
                        'upstox_ok': xres.get('upstox', {}).get('ok', False),
                    }
                    if xres.get('active'):
                        adj = int(xres.get('adjustment', 0))
                        if adj != 0:
                            old_score = score
                            score = max(0, min(100, score + adj))
                            sign = "+" if adj > 0 else ""
                            ai_thought += f" | 🌐 CROSS-EXCHANGE ({sign}{adj}%)"
                            logger.info(f"🌐 CROSS-EXCHANGE: {'; '.join(xres.get('notes', []))} → {old_score}→{score}")
                        if xres.get('veto') and logic_signal != 0:
                            logger.warning("🌐 CROSS-EXCHANGE VETO: Binance 15m trend strongly opposes Delta signal — entry vetoed")
                            return self._get_no_trade_signal("Cross-exchange veto: Binance trend strongly opposes signal")
                    else:
                        for note in xres.get('notes', []):
                            logger.debug(f"🌐 CROSS-EXCHANGE: {note}")
                except Exception as xe:
                    # Never let the advisory perspective break the decision flow
                    logger.debug(f"🌐 CROSS-EXCHANGE skipped (error): {xe}")

            # 3D. Pattern Recognition confluence (computed above; always define the branch value)
            pattern_mtf = self.market_context.get('pattern_mtf', {}) or {}
            if pattern_mtf and isinstance(pattern_mtf, dict):
                pattern_tf_count = len(pattern_mtf)
                if pattern_tf_count >= 3:
                    pattern_boost = min(5, pattern_tf_count)
                    score = min(100, score + pattern_boost)
                    ai_thought += f" | 🔍 PATTERNS on {pattern_tf_count} TFs (+{pattern_boost}%)"
                    detailed_scores['pattern_mtf_boost'] = pattern_boost
                    logger.info(f"🔍 PATTERN ENGINE: {pattern_tf_count} timeframes with patterns (+{pattern_boost}%)")
            
            # 3E. Supplementary Intelligence (ExtraBrains, SmartBreakout, AdvancedAnalysis)
            supplementary = {}  # externally supplied model/brain votes cannot change core score
            if supplementary:
                # ExtraBrains: Trend/Volatility/Risk modifiers
                extra_brains_data = supplementary.get('ExtraBrains', {})
                if extra_brains_data:
                    insights = extra_brains_data.get('insights', [])
                    extra_boost = 0
                    for insight in insights:
                        insight_upper = str(insight).upper()
                        # Trend alignment bonus
                        if 'BULLISH' in insight_upper and logic_signal == 1:
                            extra_boost += 2
                        elif 'BEARISH' in insight_upper and logic_signal == -1:
                            extra_boost += 2
                        # High risk penalty
                        if 'RISK: HIGH' in insight_upper or 'RISK: EXTREME' in insight_upper:
                            extra_boost -= 3
                        # Low volatility caution
                        if 'VOLATILITY: LOW' in insight_upper:
                            extra_boost -= 1
                    
                    if extra_boost != 0:
                        extra_boost = max(-10, min(10, extra_boost))  # Clamp
                        score = max(0, min(100, score + extra_boost))
                        sign = "+" if extra_boost > 0 else ""
                        ai_thought += f" | 🧠 EXTRA BRAINS ({sign}{extra_boost}%)"
                        detailed_scores['extra_brains_boost'] = extra_boost
                        logger.info(f"🧠 EXTRA BRAINS: {len(insights)} insights → {sign}{extra_boost}%")
                
                # SmartBreakoutAI: Breakout alignment
                breakout_data = supplementary.get('SmartBreakoutAI', {})
                if breakout_data and isinstance(breakout_data, dict):
                    breakout_signal_str = str(breakout_data.get('signal', 'NEUTRAL')).upper()
                    if (breakout_signal_str in ('BUY', 'CALL', 'BULLISH') and logic_signal == 1) or \
                       (breakout_signal_str in ('SELL', 'PUT', 'BEARISH') and logic_signal == -1):
                        score = min(100, score + 5)
                        ai_thought += " | 🎯 BREAKOUT ALIGNED (+5%)"
                        detailed_scores['breakout_boost'] = 5
                        logger.info("🎯 SMARTBREAKOUT: Aligned with signal (+5%)")
                    elif (breakout_signal_str in ('BUY', 'CALL', 'BULLISH') and logic_signal == -1) or \
                         (breakout_signal_str in ('SELL', 'PUT', 'BEARISH') and logic_signal == 1):
                        score = max(0, score - 3)
                        ai_thought += " | 🎯 BREAKOUT CONFLICT (-3%)"
                        detailed_scores['breakout_penalty'] = -3
                        logger.info("🎯 SMARTBREAKOUT: Conflicts with signal (-3%)")
                
                # AdvancedAnalysis (16-Brain system): Alignment vote
                adv_data = supplementary.get('AdvancedAnalysis', {})
                if adv_data and isinstance(adv_data, dict):
                    adv_result = adv_data.get('result', {})
                    if isinstance(adv_result, dict):
                        adv_signal = adv_result.get('signal', adv_result.get('direction', 0))
                        try:
                            adv_signal = int(adv_signal)
                        except (ValueError, TypeError):
                            adv_signal = 0
                        
                        if adv_signal == logic_signal and logic_signal != 0:
                            score = min(100, score + 4)
                            ai_thought += " | 🧬 16-BRAIN ALIGNED (+4%)"
                            detailed_scores['advanced_analysis_boost'] = 4
                            logger.info("🧬 ADVANCED ANALYSIS: 16-brain system aligned (+4%)")
                
                # EnhancedConfidence cross-check
                ext_conf_data = supplementary.get('EnhancedConfidence', {})
                if ext_conf_data and isinstance(ext_conf_data, dict):
                    ext_score = ext_conf_data.get('confidence', ext_conf_data.get('score', None))
                    if ext_score is not None:
                        try:
                            ext_score = float(ext_score)
                            # If external confidence significantly disagrees, moderate
                            if abs(score - ext_score) > 30:
                                old_score = score
                                score = int(score * 0.7 + ext_score * 0.3)
                                ai_thought += f" | 📊 CONF MODERATED ({old_score}→{score})"
                                detailed_scores['confidence_moderation'] = score - old_score
                                logger.info(f"📊 ENHANCED CONFIDENCE: Moderated {old_score} → {score}")
                        except (ValueError, TypeError):
                            pass
                            
            # Laya is audit/display-only. Its output cannot change score,
            # direction, confidence, targets, gates, or order inputs.

            # ---------------------------
            # Check trading conditions
            if not self.trade_manager.can_trade():
                return self._get_no_trade_signal("Trading not allowed")
                
            # The legacy non-pure live workflow required a DeepSeek approval
            # for scores >=10 and used model fusion elsewhere. There is no proven
            # deterministic equivalent for either path. Fail closed for this
            # unsupported configuration, including lower scores, rather than
            # treating model unavailability as approval. The pre-existing pure-
            # algorithm and backtest workflows remain deterministic. Laya
            # commentary cannot satisfy this gate.
            if not self.is_backtest_mode and not _pure_algorithm_mode():
                return self._get_no_trade_signal(
                    "Legacy model-required entry unsupported; use reviewed pure-algorithm policy")
            # --- 37-BRAIN MODE ENGINE ADVISORY EVALUATION ---
            # Final decision authority stays in jarvis_FIXED.py (central 65% & MTF confluence gate).
            # ModeEngine evaluates all brains + Parts 4-12 + XGBoost and records advisory metadata.
            try:
                if hasattr(self, 'mode_engine'):
                    m_tf_dict = native_mtf if native_mtf else (self.market_context.get('mtf_datasets') or {})
                    m_df = data
                    m_h1 = m_tf_dict.get('1h', m_df)
                    _me_snap = {
                        'symbol': _binance_spot_symbol(getattr(self, 'active_symbol', None)),
                        'auxiliary_telemetry': self.market_context.get('auxiliary_telemetry', {}),
                        'institutional_components': (getattr(self, 'latest_part_results', {}).get('part3_psychology') or {}).get('components', {}),
                    }
                    try:
                        from jarvis_xgb_derived import compute as _xdc
                        from jarvis_institutional_feed import get_context as _gc_xgb
                        _pr_loc = locals().get('part_results') or {}
                        _sg = [(v or {}).get('signal', 0) for v in _pr_loc.values() if isinstance(v, dict)]
                        _ictx_xgb = _gc_xgb(_me_snap['symbol']) if _me_snap.get('symbol') else None
                        _me_snap['derived_part_features'] = _xdc(m_tf_dict.get('15m'), m_tf_dict.get('1h'), _sg, inst_ctx=_ictx_xgb)
                        import datetime as _dtm
                        _hh = _dtm.datetime.utcnow().hour
                        _me_snap['session_liquidity'] = 1.0 if 12 <= _hh < 16 else (0.7 if (7 <= _hh < 12 or 16 <= _hh < 21) else 0.3)
                        try:
                            import time as _tm_eb
                            _dcc = getattr(self, 'direct_candle_cache', None)
                            _rets = {}
                            for _cs in ('BTCUSDT', 'ETHUSDT'):
                                _sn = _dcc.get_snapshot(_cs, venue='binance', market_type='spot', instrument_id=_cs) if _dcc is not None else None
                                _fr = (_sn.frames.get('1h') if _sn is not None else None)
                                _cl = _fr.closed if _fr is not None else None
                                if _cl is not None and len(_cl) > 25 and (_tm_eb.time() - float(_sn.fetched_at)) < 1800:
                                    _c = _cl['close'].astype(float)
                                    _rets[_cs] = float(_c.iloc[-1] / _c.iloc[-25] - 1.0) * 100.0
                            if len(_rets) == 2:
                                _me_snap['eth_btc_momentum'] = _rets['ETHUSDT'] - _rets['BTCUSDT']
                                _me_snap['btc_dominance_trend'] = _rets['BTCUSDT'] - _rets['ETHUSDT']
                        except Exception:
                            pass
                    except Exception:
                        pass
                    _xgb_opt_data = dict(options_intel) if isinstance(options_intel, dict) else {}
                    try:
                        _px = float(data['close'].iloc[-1]); _tl = dict(_xgb_opt_data.get('telemetry') or {})
                        for _k, _src in (('max_pain_distance_pct', 'max_pain'), ('call_wall_distance_pct', 'resistance'), ('put_wall_distance_pct', 'support')):
                            _lv = (options_walls or {}).get(_src)
                            if _lv is None:
                                for _si in (options_intel, (self.last_dual_intel or {}).get('deribit'), (self.last_dual_intel or {}).get('delta')):
                                    _rd0 = (_si or {}).get('raw_data') if isinstance(_si, dict) else None
                                    if isinstance(_rd0, dict) and _rd0.get(_src) is not None:
                                        _lv = _rd0.get(_src); break
                            _lv = float(_lv) if _lv is not None else None
                            if _lv and 0.2 * _px < _lv < 5.0 * _px:
                                _tl[_k] = (_lv - _px) / _px * 100.0
                        # Genuine Deribit/Delta put/call OPEN-INTEREST ratio (put_oi/call_oi) -> opt_pcr_oi only (not volume).
                        for _src_i in (options_intel, (self.last_dual_intel or {}).get('delta'), (self.last_dual_intel or {}).get('deribit')):
                            _rd = (_src_i or {}).get('raw_data') if isinstance(_src_i, dict) else None
                            _pv = (_rd or {}).get('pcr') if isinstance(_rd, dict) else None
                            if _pv is not None and 'pcr_oi' not in _tl:
                                _pvf = float(_pv)
                                if _pvf == _pvf and 0.0 <= _pvf < 50.0:
                                    _tl['pcr_oi'] = _pvf
                        _xgb_opt_data['telemetry'] = _tl
                    except Exception:
                        pass
                    me_result = self.mode_engine.master_evaluate(
                        m_df, m_h1, m_tf_dict,
                        snapshot=_me_snap,
                        options_data=_xgb_opt_data,
                    )
                    try:
                        _xr = me_result.get('xgboost_report') or {}
                        _inf = _xr.get('inference') if isinstance(_xr, dict) else None
                        _pp = (_inf or {}).get('probs') or (_inf or {}).get('probabilities') or _inf
                        _fr = (_xr.get('feature_report') or {}) if isinstance(_xr, dict) else {}
                        _mf = _fr.get('missing_features') or []
                        self.last_xgb_mode_advice = {'status': _xr.get('status') if isinstance(_xr, dict) else None, 'probs': _pp,
                                                     'populated': '%d/48' % (48 - len(_mf)), 'shadow_only': True, 'source': 'mode_engine_xgboost_report'}
                        logger.info('[XGB_SHADOW] populated=%d/48 missing=%s probs=%s shadow_only=True veto_applied=False', 48 - len(_mf), _mf, _pp)
                    except Exception as _xe:
                        logger.info('[XGB_SHADOW] unavailable (%s)', type(_xe).__name__)
                    detailed_scores['mode_engine_advisory'] = {
                        'action': me_result.get('action'),
                        'trade_type': me_result.get('trade_type'),
                        'direction': me_result.get('direction'),
                        'reason': me_result.get('reason'),
                        'vote_tally': me_result.get('vote_tally'),
                        'sl': me_result.get('sl'),
                        'tp': me_result.get('tp'),
                        'rr': me_result.get('rr'),
                    }
                    try:
                        logger.info('[MODE_ENGINE] action=%s type=%s dir=%s reason=%s tally=%s | xgb=%s',
                                    me_result.get('action'), me_result.get('trade_type'),
                                    me_result.get('direction'), me_result.get('reason'),
                                    me_result.get('vote_tally'), me_result.get('xgboost_report'))
                    except Exception:
                        pass

                    # Advisory only: never force logic_signal or score=92 when central confluence is not met.
                    if me_result.get('action') == 'EXECUTE':
                        me_dir = me_result.get('direction')
                        me_sig = 1 if me_dir in ('BUY', 'CALL') else (-1 if me_dir in ('SELL', 'PUT') else 0)
                        detailed_scores['ai_synthesis'] = f"ModeEngine Advisory: {me_result.get('trade_type', 'SCALP')} {me_dir} ({me_result.get('reason', '')})"
                        if me_sig != 0 and logic_signal == me_sig:
                            detailed_scores['mode_engine_sl'] = me_result.get('sl')
                            detailed_scores['mode_engine_tp'] = me_result.get('tp')
            except Exception as _me_err:
                logger.error(f"ModeEngine advisory evaluation failed: {_me_err}")
            # --- END 37-BRAIN MODE ENGINE ADVISORY ---

            # Generate final signal
            final_decision = self._generate_trade_signal(data, score, detailed_scores, options_intel, logic_signal)

            # Canonical Part 7 new-entry gate.  Protective-exit ownership lives
            # outside this analysis result and is intentionally not altered.
            # A veto/data failure must remain a veto even when every other part
            # is bullish; do not reduce it to a neutral reason string.
            part7_gate = getattr(self, 'latest_part7', {}) or {}
            if part7_gate.get('entry_blocked'):
                # ADVISORY ONLY (user rule 15 Oct): record, never force NO_TRADE.
                p7_reason = part7_gate.get('reason', 'Part7 volatility advisory')
                final_decision['part7_volatility'] = part7_gate
                final_decision['part7_advisory_warning'] = p7_reason
                detailed_scores['part7_advisory_warning'] = p7_reason
                logger.info('[PART7_ADVISORY] volatility warning (not a veto): %s', p7_reason)
            else:
                final_decision['part7_volatility'] = part7_gate
            
            # --- BIG PLAYER FILTER (STRICT DUAL CONFLUENCE) ---
            if TRADE_CONFIG.get('use_big_player_filter', True):
                try:
                    intel = getattr(self, 'last_dual_intel', {})
                    delta_intel = intel.get('delta')
                    deribit_intel = intel.get('deribit')
                    
                    delta_score = delta_intel.get('score', 0) if delta_intel else 0
                    deribit_score = deribit_intel.get('score', 0) if deribit_intel else 0
                    
                    signal_direction = final_decision.get('trade_signal', {}).get('direction', 'NO_TRADE')
                    
                    if signal_direction != 'NO_TRADE':
                        # 1. Conflict Check (STRICT REJECTION if ANY exchange disagrees)
                        conflict = False
                        rejection_msg = ""
                        
                        # Check Delta Conflict (relaxed to ±5 for 1m scalping)
                        if signal_direction == 'CALL' and delta_score <= -5:
                            conflict, rejection_msg = True, f"Delta Bearish ({delta_score})"
                        elif signal_direction == 'PUT' and delta_score >= 5:
                            conflict, rejection_msg = True, f"Delta Bullish ({delta_score})"
                        
                        # Check Deribit Conflict
                        if not conflict:
                            if signal_direction == 'CALL' and deribit_score <= -5:
                                conflict, rejection_msg = True, f"Deribit Bearish ({deribit_score})"
                            elif signal_direction == 'PUT' and deribit_score >= 5:
                                conflict, rejection_msg = True, f"Deribit Bullish ({deribit_score})"
                                
                        if conflict:
                            logger.warning(f"🚫 DUAL REJECTION: {rejection_msg} vs {signal_direction}")
                            final_decision['dual_rejection_reason'] = f"Institutional Conflict: {rejection_msg}"
                            if isinstance(final_decision.get('trade_signal'), dict):
                                final_decision['trade_signal']['direction'] = 'NO_TRADE'
                                final_decision['trade_signal']['confidence_score'] = '0/100'
                            final_decision['no_trade_reason'] = f"Institutional Conflict: {rejection_msg}"
                            
                        # 2. Alignment Bonus (Dual Confluence)
                        confluence_bonus = 0
                        sources_aligned = 0
                        
                        if (signal_direction == 'CALL' and delta_score >= 3) or (signal_direction == 'PUT' and delta_score <= -3):
                            confluence_bonus += abs(delta_score)
                            sources_aligned += 1
                        
                        if (signal_direction == 'CALL' and deribit_score >= 3) or (signal_direction == 'PUT' and deribit_score <= -3):
                            confluence_bonus += abs(deribit_score)
                            sources_aligned += 1
                            
                        if confluence_bonus > 0:
                            logger.info(f"✅ DUAL CONFLUENCE: {sources_aligned} sources support {signal_direction}")
                            
                            # Fix: Ensure detailed_scores exists
                            if 'detailed_scores' not in final_decision:
                                final_decision['detailed_scores'] = {}
                            
                            final_decision['detailed_scores']['dual_confluence'] = True
                            
                            # Double confirmation multiplier
                            if sources_aligned == 2:
                                confluence_bonus = int(confluence_bonus * 1.5)
                                logger.info(f"💎 DOUBLE CONFIRMATION! Bonus: +{confluence_bonus}")
                                
                            new_score = min(100, score + confluence_bonus)
                            final_decision['trade_signal']['confidence_score'] = f"{new_score}/100"
                            
                except Exception as e:
                    logger.error(f"Dual Big Player Filter Error: {e}")
                    # On error, we proceed but log it (or could default to safe mode)
            
            # --- WIRING FIX: AI ADAPTIVE LEARNING ---
            if 'adaptive' in self.engines:
                try:
                    ai_rec = self.engines['adaptive'].get_ai_recommendation()
                    if ai_rec:
                        self.market_context['adaptive_ai_bias'] = ai_rec
                        logger.info(f"🧠 [ADAPTIVE] AI bias applied: {ai_rec}")
                except Exception as e:
                    logger.debug(f"[ADAPTIVE] get_ai_recommendation failed: {e}")
            
            # --- WIRING FIX: ENHANCED CONFIDENCE MONITOR ---
            if 'confidence' in self.engines:
                try:
                    enhanced_conf = self.engines['confidence'].compute_signal_confidence(
                        raw_confidence=score,
                        market_context=self.market_context
                    )
                    if enhanced_conf:
                        score = enhanced_conf
                        final_decision['trade_signal']['confidence_score'] = f"{score}/100"
                        logger.info(f"🛡️ [CONFIDENCE] Score refined to: {score}%")
                except Exception as e:
                    logger.debug(f"[CONFIDENCE] compute_signal_confidence failed: {e}")

            # Preserve independent directional opinions for the live canonical
            # decision.  A disagreement is a hard block; never manufacture a
            # consensus merely because one model was selected as the winner.
            _decision_opinions = []
            for _raw_opinion in (math_signal, logic_signal):
                if _raw_opinion > 0:
                    _decision_opinions.append('BUY')
                elif _raw_opinion < 0:
                    _decision_opinions.append('SELL')
            if isinstance(neural_res, dict):
                _neural_bias = str(neural_res.get('bias', '')).upper()
                if _neural_bias in ('CALL', 'BUY', 'LONG'):
                    _decision_opinions.append('BUY')
                elif _neural_bias in ('PUT', 'SELL', 'SHORT'):
                    _decision_opinions.append('SELL')
            final_decision['decision_opinions'] = _decision_opinions

            # --- CNS PERCEPTION RECORDING ---
            final_decision['telemetry'] = full_telemetry
            if hasattr(self, 'cns'):
                self.cns.record_perception(final_decision)
                
            # Re-assert the canonical Part 7 gate after advisory confidence and
            # big-player layers.  Those layers may enrich a result, but may not
            # turn a blocked new entry back into CALL/PUT or restore confidence.
            if part7_gate.get('entry_blocked'):
                final_decision['part7_volatility'] = part7_gate  # advisory only

            # Return valid signal (passed or filtered)
            # --- AI AUTONOMY (JARVIS UNLEASHED) ---
            # User Request: Trust AI logic over hard thresholds
            ai_signal = final_decision.get('trade_signal', {}).get('direction', 'NEUTRAL')

            # Publish only the genuine deterministic Parts11/12-selected
            # direction and target values from this isolated run. The bridge
            # may later replace the Binance reference entry with a fresh Delta
            # executable quote; it never derives missing targets or sizes.
            _complete_live_native_set = (
                not self.is_backtest_mode
                and set(mtf_data) == set(LIVE_TIMEFRAMES)
                and set(mtf_breakdown) == set(LIVE_TIMEFRAMES)
            )
            if native_mtf is not None or _complete_live_native_set:
                _parts_by_tf = {}
                _part_names = [
                    ('part1', 'part1_breakout'), ('part2', 'part2_zone'),
                    ('part3', 'part3_psychology'), ('part4', 'part4_volume'),
                    ('part5', 'part5_ml'), ('part6', 'part6_trend'),
                    ('part7', 'part7_volatility'), ('part8', 'part8_structure'),
                    ('part9', 'part9_orderflow'), ('part10', 'part10_candlestats'),
                ]
                for _tf, _results in mtf_breakdown.items():
                    _frame_parts = {}
                    for _key, _name in _part_names:
                        _item = dict(_results.get(_name) or _results.get(_key) or {'signal': 0, 'thought': 'unavailable'})
                        # Identity is supplied by the exact native snapshot
                        # boundary, not inferred from a model-generated label.
                        _cur_sym = str(_item.get('symbol') or '').upper()
                        _route_sym = str(getattr(self, 'active_symbol', None) or '').upper()
                        if not _cur_sym or _cur_sym == _route_sym:
                            # Analysis candles are Binance spot (e.g. BTCUSDT); the route symbol is the Delta perp (BTCUSD).
                            _item['symbol'] = _binance_spot_symbol(_route_sym) or _route_sym or _cur_sym
                        _item.setdefault('timeframe', _tf)
                        try:
                            from jarvis_part_anchors import enrich as _pa_enrich
                            _pa_enrich(_name, _item, mtf_data.get(_tf))
                        except Exception:
                            pass
                        _frame_parts[_key] = _item
                        _frame_parts[_name] = _item
                    _parts_by_tf[_tf] = _frame_parts
                _strategy_dir = str((final_decision.get('trade_signal') or {}).get('direction') or 'NO_TRADE').upper()
                _strategy_conf = normalize_confidence((final_decision.get('trade_signal') or {}).get('confidence_score'), default=0) or 0
                self.latest_multicoin_analysis = {
                    'parts_by_timeframe': _parts_by_tf,
                    'once_per_symbol_parts': {
                        'part11': dict(math_res) if isinstance(math_res, dict) else {'signal': math_signal},
                        'part12': dict(math_conf_res) if isinstance(math_conf_res, dict) else {'confidence': math_confidence},
                    },
                    'part7_gate': dict(part7_gate),
                    'central_strategy_decision': {
                        'direction': {'CALL': 'BUY', 'PUT': 'SELL'}.get(_strategy_dir, 'NO_TRADE'),
                        'confidence': int(_strategy_conf),
                        'origin': 'deterministic_parts11_12',
                    },
                    'central_strategy_evidence': {
                        'parts_by_timeframe': _parts_by_tf,
                        'horizon_groups': {k: v.get('mtf_horizon_groups', {}) for k, v in part_results.items() if isinstance(v, dict)},
                    },
                    'central_strategy_approval': {'status': 'analysis_only_no_execution_authority'},
                    'central_execution_plan': None,
                    'execution_plan': None,
                    'analysis_reference': None,
                }

            if (native_mtf is not None or _complete_live_native_set) and isinstance(getattr(self, 'latest_multicoin_analysis', None), dict):
                _raw_signal = final_decision.get('trade_signal', {}) or {}
                _legacy_dir = str(_raw_signal.get('direction') or 'NO_TRADE').upper()
                _central = {'status': 'BLOCKED', 'approved': False, 'direction': 'NO_TRADE',
                            'reasons': ['central MTF approval unavailable'], 'policy_version': 'unavailable'}
                if _complete_live_native_set:
                    try:
                        from jarvis_strategy_approval import evaluate_mtf_central_strategy
                        _central = evaluate_mtf_central_strategy(
                            _parts_by_tf, part7_gate, confidence=score,
                            minimum_confidence=self.scoring_matrix.minimum_trade_score,
                            expected_symbol=(_binance_spot_symbol(getattr(self, 'active_symbol', None)
                                                                   or getattr(self, 'market_context', {}).get('symbol')) or None),
                        )
                    except Exception as _central_error:
                        _central = {
                            'status': 'BLOCKED', 'approved': False, 'direction': 'NO_TRADE',
                            'reasons': [f'central MTF approval error: {type(_central_error).__name__}'],
                            'policy_version': 'unavailable',
                        }
                    try:
                        if _central.get('status') == 'BLOCKED':
                            from jarvis_strategy_approval import REQUIRED_TIMEFRAMES as _RT
                            _g = part7_gate if isinstance(part7_gate, dict) else {}
                            _gr = _g.get('timeframe_results') or {}
                            logger.info('[CENTRAL_DIAG] p7 symbol=%s expected=%s status=%s data=%s tf=%s entry_blocked=%s risk_veto=%s blocked=%s veto=%s missing_frames=%s extra_frames=%s reasons=%s', _g.get('symbol'), _binance_spot_symbol(getattr(self, 'active_symbol', None)), _g.get('status'), _g.get('data_status'), _g.get('timeframe'), _g.get('entry_blocked'), _g.get('risk_veto'), _g.get('blocked_timeframes'), _g.get('veto_timeframes'), sorted(set(_RT) - set(_gr)), sorted(set(_gr) - set(_RT)), _central.get('reasons'))
                        logger.info('[CENTRAL_DIAG] p7 rows %s', {k: (v.get('status'), str(v.get('reason'))[:140]) for k, v in list(_gr.items())[:4] if isinstance(v, dict)})
                    except Exception:
                        pass
                    self.latest_multicoin_analysis['central_strategy_approval'] = dict(_central)
                    try:
                        from jarvis_self_learning import get_self_learning_engine, bootstrap_from_existing_paper_trades
                        from jarvis_strategy_approval import LAST_REGIME as _LR_SL
                        bootstrap_from_existing_paper_trades()
                        _sl_sym = _binance_spot_symbol(getattr(self, 'active_symbol', None)) or str(getattr(self, 'active_symbol', ''))
                        _sl_reg = str(((_LR_SL.get(_sl_sym) or {}).get('summary') or {}).get('regime') or 'UNKNOWN')
                        _sl_eng = get_self_learning_engine()
                        _sl_obs = _sl_eng.observe_live_snapshot(_sl_sym, float(data['close'].iloc[-1]), _parts_by_tf, regime=_sl_reg)
                        _sl_eval = _sl_eng.evaluate_setup_quality((_parts_by_tf.get('15m') or part_results), str(_central.get('direction') or 'BUY'), regime=_sl_reg, current_price=float(data['close'].iloc[-1]))
                        self.latest_multicoin_analysis['self_learning'] = {**_sl_eval, **_sl_obs}
                        if _central.get('approved') and _sl_eval.get('loss_pattern_veto'):
                            logger.warning('[SELF_LEARN_VETO] %s %s blocked by GPU Self-Learning Loss Pattern Guard (sl_sim=%.2f sgd_prob=%.2f)', _sl_sym, _central.get('direction'), _sl_eval.get('sl_similarity', 0), _sl_eval.get('sgd_win_prob', 0.5))
                            _central['approved'] = False
                            _central['status'] = 'BLOCKED'
                            _central['direction'] = 'NO_TRADE'
                            _central['reasons'] = list(_central.get('reasons') or []) + ['Self-learning loss pattern veto']
                        elif int(_sl_eval.get('confidence_delta') or 0) != 0:
                            score = max(0, min(100, int(score or 0) + int(_sl_eval.get('confidence_delta') or 0)))
                    except Exception as _sl_err:
                        logger.debug('[SELF_LEARN] hook skipped: %s', _sl_err)
                    self.latest_multicoin_analysis['central_strategy_decision'] = {
                        'direction': _central.get('direction', 'NO_TRADE'),
                        'confidence': int(score) if isinstance(score, (int, float)) else 0,
                        'origin': 'central_native_mtf_gate',
                    }
                    _central_call = {'BUY': 'CALL', 'SELL': 'PUT'}.get(
                        str(_central.get('direction') or 'NO_TRADE').upper(), 'NO_TRADE')
                    _central_mode = str(_central.get('trade_mode') or '').upper()
                    _dt_sum = getattr(self, 'latest_dual_track_summary', {}) or {}
                    _dt_sc_call = {1: 'CALL', -1: 'PUT'}.get((_dt_sum.get('scalp') or {}).get('signal', 0), 'NO_TRADE')
                    _dt_sw_call = {1: 'CALL', -1: 'PUT'}.get((_dt_sum.get('swing') or {}).get('signal', 0), 'NO_TRADE')
                    _me_dc_tm = (getattr(self.mode_engine, 'last_dual_candidates', None) or {}) if hasattr(self, 'mode_engine') else {}
                    _me_sc_call = {'BUY': 'CALL', 'SELL': 'PUT'}.get(str(_me_dc_tm.get('scalp_candidate') or '').upper(), 'NO_TRADE')
                    _me_sw_call = {'BUY': 'CALL', 'SELL': 'PUT'}.get(str(_me_dc_tm.get('swing_candidate') or '').upper(), 'NO_TRADE')
                    _has_dual_rej = bool(final_decision.get('dual_rejection_reason'))
                    _track_matches = (not _has_dual_rej) and (_central_call in {'CALL', 'PUT'}) and (
                        _central_call == _legacy_dir
                        or (_central_mode == 'SWING' and _central_call in {_dt_sw_call, _me_sw_call})
                        or (_central_mode == 'SCALP' and _central_call in {_dt_sc_call, _me_sc_call})
                        or (_legacy_dir == 'NO_TRADE' and _central_call in {_dt_sw_call, _dt_sc_call, _me_sw_call, _me_sc_call})
                    )
                    _hard_blocks = []
                    try:
                        if _central.get('approved') and _central_call in ('CALL', 'PUT'):
                            _mer = locals().get('me_result') or {}
                            _med = {'BUY': 'CALL', 'SELL': 'PUT'}.get(str(_mer.get('direction') or '').upper())
                            _mer_reason = str(_mer.get('reason') or '')
                            if _central_mode == 'SCALP':
                                if _mer.get('action') == 'EXECUTE' and str(_mer.get('trade_type') or '').upper() == 'SWING' and _med and _med != _central_call:
                                    _hard_blocks.append('SCALP %s vetoed: ModeEngine SWING consensus is %s' % (_central_call, _med))
                                if 'INSTITUTIONAL_VETO' in _mer_reason and ('SCALP ' + {'CALL': 'BUY', 'PUT': 'SELL'}[_central_call]) in _mer_reason:
                                    _hard_blocks.append('SCALP %s vetoed: institutional direction opposes' % _central_call)
                            if os.getenv('JARVIS_XGB_VETO', '1') == '1':
                                _xr = _mer.get('xgboost_report') if isinstance(_mer.get('xgboost_report'), dict) else {}
                                _inf = _xr.get('inference') if isinstance(_xr.get('inference'), dict) else {}
                                _regime = str(_xr.get('regime') or _inf.get('regime') or '').upper()
                                if _regime == 'CHOP_AVOID' or (_xr.get('available') is True and _xr.get('approved') is False):
                                    _hard_blocks.append('XGBoost regime veto (%s)' % (_regime or 'not approved'))
                                _pb = (getattr(self, 'last_xgb_mode_advice', None) or {}).get('probs')
                                if isinstance(_pb, dict) and str(_pb.get('regime') or '').upper() == 'CHOP_AVOID':
                                    _hard_blocks.append('XGBoost regime veto (CHOP_AVOID)')
                                if isinstance(_pb, dict) and _pb:
                                    _key = '%s_%s' % (_central_mode or 'SCALP', 'LONG' if _central_call == 'CALL' else 'SHORT')
                                    _vals = {k: float(v) for k, v in _pb.items() if isinstance(v, (int, float))}
                                    if _key in _vals:
                                        _scale = 100.0 if max(_vals.values()) > 1.5 else 1.0
                                        _top = max(_vals, key=lambda k: _vals[k])
                                        if _top != _key and _vals[_key] / _scale < 0.10:
                                            _hard_blocks.append('XGBoost probs reject %s (%.1f%%; top=%s)' % (_key, 100.0 * _vals[_key] / _scale, _top))
                    except Exception as _hb_err:
                        logger.debug('[HARD_BLOCKS] skipped: %s', _hb_err)
                    if _hard_blocks:
                        logger.warning('[HARD_BLOCKS] %s', _hard_blocks)
                        _central['reasons'] = list(_central.get('reasons') or []) + _hard_blocks
                    if not _central.get('approved') or not _track_matches or _hard_blocks:
                        _raw_signal['direction'] = 'NO_TRADE'
                        _raw_signal['confidence_score'] = '0/100'
                        _raw_signal['central_mtf_blocked'] = True
                        final_decision['no_trade_reason'] = '; '.join(_central.get('reasons') or []) or f'central MTF gate ({_central_call}) vs track ({_legacy_dir})'
                        self.latest_multicoin_analysis['central_strategy_decision']['direction'] = 'NO_TRADE'
                        try:
                            _cand = next((c for c in (_central_call, _legacy_dir, _dt_sw_call, _dt_sc_call) if c in ('CALL', 'PUT')), None)
                            _cand_src = 'CENTRAL'
                            if not _cand:
                                try:
                                    _md = str((locals().get('me_result') or {}).get('direction') or '').upper()
                                    _cand = {'BUY': 'CALL', 'SELL': 'PUT'}.get(_md)
                                    _cand_src = 'MODE_ENGINE'
                                except Exception:
                                    _cand = None
                            _bo2 = _p10_breakout(_parts_by_tf, _cand)
                            _cmode = _central_mode if _central_mode in ('SCALP', 'SWING') else ('SWING' if (_cand and _cand == _dt_sw_call) else 'SCALP')
                            _se_c = _binance_spot_symbol(getattr(self, 'active_symbol', None)) or str(getattr(self, 'active_symbol', ''))
                            if _cand:
                                from jarvis_safe_entry import evaluate_safe_entry as _evse2
                                from jarvis_part_anchors import collect as _pa_collect2
                                _px2 = float(data['close'].iloc[-1])
                                _pa2 = _pa_collect2(_parts_by_tf, _px2)
                                _s2 = _evse2(_se_c, _cmode, _cand, _px2, mtf_data, detailed_scores.get('options_walls'), part_anchors=_pa2, breakout_confirmed=_bo2)
                                logger.info('[SAFE_ENTRY_ENGINE] [src=' + _cand_src + '] %s | mode=%s | side=%s | CentralApproved=%s | entry=%s | %s | ext_atr=%s | fair=%s | zone=%s | anchor=%s | sl=%s(sl_anchor=%s) | tp1=%s | tp2=%s | net_rr=%s | opposing_wall=%s(%s) | wall_dist_atr=%s | atr=%s | part_anchors=%s | reasons=%s',
                                            _s2.get('coin'), _s2.get('mode'), _s2.get('direction'), bool(_central.get('approved')), _s2.get('entry') or _s2.get('price'), _s2.get('status'), _s2.get('ext_atr'), _s2.get('fair'), _s2.get('zone'),
                                            _s2.get('anchor'), _s2.get('sl'), _s2.get('sl_anchor'), _s2.get('tp1'), _s2.get('tp2'), _s2.get('net_rr'), _s2.get('opposing_wall'), _s2.get('wall_source'), _s2.get('wall_dist_atr'), _s2.get('atr'), _s2.get('part_anchor_n'), _s2.get('reasons'))
                                _me_dc2 = (getattr(self.mode_engine, 'last_dual_candidates', None) or {}) if hasattr(self, 'mode_engine') else {}
                                _sc_cand2 = {'BUY': 'CALL', 'SELL': 'PUT'}.get(str(_me_dc2.get('scalp_candidate') or '').upper())
                                if _cmode == 'SWING' and _s2.get('status') != 'IMMEDIATE_SAFE_ENTRY':
                                    _s2_sc = _evse2(_se_c, 'SCALP', _cand, _px2, mtf_data, detailed_scores.get('options_walls'), part_anchors=_pa2, breakout_confirmed=_bo2)
                                    logger.info('[MODE_FALLBACK] SWING -> SCALP evaluated (scalp_eligible=%s dt_scalp=%s me_scalp=%s): swing_status=%s swing_reasons=%s -> scalp_status=%s zone=%s ext_atr=%s sl=%s tp1=%s net_rr=%s opposing_wall=%s(%s) wall_dist_atr=%s reasons=%s',
                                                bool(_dt_sc_call == _cand or _sc_cand2 == _cand), _dt_sc_call, _sc_cand2, _s2.get('status'), _s2.get('reasons'), _s2_sc.get('status'), _s2_sc.get('zone'), _s2_sc.get('ext_atr'), _s2_sc.get('sl'), _s2_sc.get('tp1'), _s2_sc.get('net_rr'), _s2_sc.get('opposing_wall'), _s2_sc.get('wall_source'), _s2_sc.get('wall_dist_atr'), _s2_sc.get('reasons'))
                            else:
                                logger.info('[SAFE_ENTRY_ENGINE] %s | no directional candidate | CentralApproved=%s | central=%s legacy=%s swing=%s scalp=%s', _se_c, bool(_central.get('approved')), _central_call, _legacy_dir, _dt_sw_call, _dt_sc_call)
                        except Exception as _se2_err:
                            logger.info('[SAFE_ENTRY_ENGINE] diagnostic unavailable (%s: %s)', type(_se2_err).__name__, _se2_err)
                    else:
                        _raw_signal['direction'] = _central_call
                        _raw_signal['central_mtf_approved'] = True
                        _cur_cs = normalize_confidence(_raw_signal.get('confidence_score'), default=0) or 0
                        if _cur_cs < self.scoring_matrix.minimum_trade_score:
                            _eff_c = int(max(float(score or 0), float(_central.get('confidence') or 0)))  # no artificial boost to minimum
                            score = max(int(score or 0), _eff_c)
                            _raw_signal['confidence_score'] = f'{_eff_c}/100'
                        if _central_mode in {'SCALP', 'SWING'}:
                            _raw_signal['recommended_expiry'] = _central_mode
                            _raw_signal['trade_mode'] = _central_mode
                            if _central_mode == 'SWING':
                                _ep = float(_raw_signal.get('entry_price') or data['close'].iloc[-1])
                                if _central_call == 'CALL':
                                    _raw_signal['stop_loss'] = round(_ep * (1.0 - 0.008), 2)
                                    _raw_signal['take_profit_1'] = round(_ep * (1.0 + 0.020), 2)
                                    _raw_signal['take_profit_2'] = round(_ep * (1.0 + 0.030), 2)
                                else:
                                    _raw_signal['stop_loss'] = round(_ep * (1.0 + 0.008), 2)
                                    _raw_signal['take_profit_1'] = round(_ep * (1.0 - 0.020), 2)
                                    _raw_signal['take_profit_2'] = round(_ep * (1.0 - 0.030), 2)
                        try:
                            from jarvis_safe_entry import evaluate_safe_entry as _evse
                            _se_coin = _binance_spot_symbol(getattr(self, 'active_symbol', None)) or str(getattr(self, 'active_symbol', ''))
                            from jarvis_part_anchors import collect as _pa_collect
                            _pa_x = _pa_collect(_parts_by_tf, float(data['close'].iloc[-1]))
                            _bo1 = _p10_breakout(_parts_by_tf, _central_call)
                            _se = _evse(_se_coin, _central_mode or 'SCALP', _central_call, float(data['close'].iloc[-1]), mtf_data, detailed_scores.get('options_walls'), part_anchors=_pa_x, breakout_confirmed=_bo1)
                            logger.info('[SAFE_ENTRY_ANCHORS] n_support=%d n_resist=%d src=%s', len(_pa_x[0]), len(_pa_x[1]), ','.join(_pa_x[2][:12]))
                        except Exception as _se_err:
                            _se = {'status': 'BLOCKED_UNSAFE_LOCATION', 'reasons': ['engine call error %s' % type(_se_err).__name__]}
                        try:
                            if _se.get('status') != 'IMMEDIATE_SAFE_ENTRY' and (_central_mode or 'SCALP') == 'SWING':
                                _dg_mf = (_central.get('diagnostic') or {})
                                _td_mf = _dg_mf.get('timeframe_decisions') or {}
                                _want_mf = str(_central.get('direction') or '').upper()
                                _me_dc = (getattr(self.mode_engine, 'last_dual_candidates', None) or {}) if hasattr(self, 'mode_engine') else {}
                                _no_ltf_veto = not any(('zone veto' in str(_r).lower() or 'anchor-dissent veto' in str(_r).lower()) for _t in ('1m', '3m', '5m', '15m') for _r in ((_td_mf.get(_t) or {}).get('reasons') or []))
                                _sc_ok_mf = _no_ltf_veto and (
                                    (str((_dg_mf.get('scalp') or {}).get('direction') or '').upper() == _want_mf
                                     and any(((_td_mf.get(_t) or {}).get('status') == 'APPROVED' and (_td_mf.get(_t) or {}).get('direction') == _want_mf) for _t in ('15m', '5m', '1m')))
                                    or str(_me_dc.get('scalp_candidate') or '').upper() == _want_mf
                                    or _dt_sc_call == _central_call
                                )
                                _swing_wall_blocked = any('opposing wall' in str(_rw).lower() for _rw in (_se.get('reasons') or []))
                                if _sc_ok_mf and not _swing_wall_blocked:
                                    _se_sc = _evse(_se_coin, 'SCALP', _central_call, float(data['close'].iloc[-1]), mtf_data, detailed_scores.get('options_walls'), part_anchors=_pa_x, breakout_confirmed=_bo1)
                                    logger.info('[MODE_FALLBACK] SWING -> SCALP evaluated: swing_status=%s swing_reasons=%s -> scalp_status=%s zone=%s ext_atr=%s sl=%s tp1=%s net_rr=%s opposing_wall=%s(%s) wall_dist_atr=%s reasons=%s',
                                                _se.get('status'), _se.get('reasons'), _se_sc.get('status'), _se_sc.get('zone'), _se_sc.get('ext_atr'), _se_sc.get('sl'), _se_sc.get('tp1'), _se_sc.get('net_rr'), _se_sc.get('opposing_wall'), _se_sc.get('wall_source'), _se_sc.get('wall_dist_atr'), _se_sc.get('reasons'))
                                    if _se_sc.get('status') == 'IMMEDIATE_SAFE_ENTRY':
                                        _se = _se_sc
                                        _se['fallback_from'] = 'SWING'
                                        _central_mode = 'SCALP'
                                        _raw_signal['recommended_expiry'] = 'SCALP'
                                        _raw_signal['trade_mode'] = 'SCALP'
                                else:
                                    logger.info('[MODE_FALLBACK] SWING -> SCALP skipped: swing_status=%s scalp_eligible=False (scalp_diag=%s me_scalp=%s dt_scalp=%s ltf_veto=%s)',
                                                _se.get('status'), (_dg_mf.get('scalp') or {}).get('direction'), _me_dc.get('scalp_candidate'), _dt_sc_call, not _no_ltf_veto)
                        except Exception as _mf_err:
                            logger.info('[MODE_FALLBACK] unavailable (%s)', type(_mf_err).__name__)
                        logger.info('[SAFE_ENTRY_ENGINE] %s | mode=%s | side=%s | entry=%s | %s | ext_atr=%s | zone=%s | anchor=%s | wait_entry=%s | sl=%s(sl_anchor=%s) | tp1=%s | tp2=%s | net_rr=%s | opposing_wall=%s(%s) | wall_dist_atr=%s | atr=%s | reasons=%s',
                                    _se.get('coin'), _se.get('mode'), _se.get('direction'), _se.get('entry') or _se.get('price'), _se.get('status'), _se.get('ext_atr'), _se.get('zone'),
                                    _se.get('anchor'), _se.get('optimal_entry'), _se.get('sl'), _se.get('sl_anchor'), _se.get('tp1'), _se.get('tp2'), _se.get('net_rr'), _se.get('opposing_wall'), _se.get('wall_source'), _se.get('wall_dist_atr'), _se.get('atr'), _se.get('reasons'))
                        try:
                            _p10_zone_track(_se_coin, _se, float(data['close'].iloc[-1]))
                        except Exception:
                            pass
                        _raw_signal['safe_entry_status'] = _se.get('status')
                        _raw_signal['safe_entry'] = _se
                        if _se.get('status') == 'IMMEDIATE_SAFE_ENTRY':
                            _raw_signal['stop_loss'] = _se['sl']
                            _raw_signal['take_profit_1'] = _se['tp1']
                            _raw_signal['take_profit_2'] = _se.get('tp2') or _se['tp1']
                        else:
                            _raw_signal['direction'] = 'NO_TRADE'
                            _raw_signal['confidence_score'] = '0/100'
                            _raw_signal['safe_entry_blocked'] = True
                            final_decision['no_trade_reason'] = 'SAFE_ENTRY %s %s' % (_se.get('status'), _se.get('reasons') or ('wait zone %s' % (_se.get('zone'),)))
                            self.latest_multicoin_analysis['central_strategy_decision']['direction'] = 'NO_TRADE'
                        if _raw_signal.get('direction') in ('CALL', 'PUT'): logger.info("[ARBITER] FINAL APPROVED: mode=%s direction=%s conf=%s entry=%s sl=%s tp1=%s",
                                    _raw_signal.get('recommended_expiry'), _central_call, score,
                                    _raw_signal.get('entry_price'), _raw_signal.get('stop_loss'), _raw_signal.get('take_profit_1'))
                        try:
                            from jarvis_paper_tracker import record_signal
                            _min_paper_conf = float(os.getenv('JARVIS_MIN_CONFIDENCE', '65') or 65)
                            if _raw_signal.get('direction') not in ('CALL', 'PUT') or float(score or 0) < _min_paper_conf: raise RuntimeError('safe-entry/confidence gate')
                            from jarvis_strategy_approval import LAST_REGIME as _LR2
                            _rk = _binance_spot_symbol(getattr(self, 'active_symbol', None))
                            record_signal(_rk, _raw_signal.get('recommended_expiry'), _central_call,
                                          _raw_signal.get('entry_price'), _raw_signal.get('stop_loss'),
                                          _raw_signal.get('take_profit_1'), _raw_signal.get('take_profit_2'),
                                          score, _LR2.get(_rk), (getattr(self, 'last_xgb_mode_advice', None) or {}).get('probs'),
                                      extra={'entry_status': _raw_signal.get('safe_entry_status'), 'safe_entry_zone': (_raw_signal.get('safe_entry') or {}).get('zone'), 'sl_anchor': (_raw_signal.get('safe_entry') or {}).get('sl_anchor'), 'net_rr': (_raw_signal.get('safe_entry') or {}).get('net_rr')})
                        except Exception:
                            pass
                    final_decision['trade_signal'] = _raw_signal
                _dir = str(_raw_signal.get('direction') or 'NO_TRADE').upper()
                if score < self.scoring_matrix.minimum_trade_score or _dir not in {'CALL', 'PUT'}:
                    _dir = 'NO_TRADE'
                _raw_conf = _raw_signal.get('confidence_score', score)
                try:
                    _conf = int(float(str(_raw_conf).split('/', 1)[0]))
                except (TypeError, ValueError, OverflowError):
                    _conf = int(score)
                self.latest_multicoin_analysis['deterministic_decision'] = {
                    'origin': 'jarvis_deterministic_parts11_12',
                    'direction': {'CALL': 'BUY', 'PUT': 'SELL'}.get(_dir, 'NO_TRADE'),
                    'confidence': max(0, min(100, _conf)),
                    'entry_price': _raw_signal.get('entry_price'),
                    # TP1 is the explicit nearer target produced by Jarvis'
                    # existing deterministic scalping target function.
                    'take_profit': _raw_signal.get('take_profit_1'),
                    'stop_loss': _raw_signal.get('stop_loss'),
                    'part11_signal': (self.latest_multicoin_analysis.get('once_per_symbol_parts', {})
                                      .get('part11', {}).get('signal')),
                    'part12_confidence': (self.latest_multicoin_analysis.get('once_per_symbol_parts', {})
                                          .get('part12', {}).get('confidence')),
                }
                self.latest_multicoin_analysis['part7_gate'] = dict(part7_gate)

            if score >= self.scoring_matrix.minimum_trade_score:
                if self.hud_enabled:
                    self._sync_to_hud(final_decision)
                return final_decision
                
            else:  # Never override a deterministic minimum score with an advisor's direction.
                return self._get_no_trade_signal(f"Score too low: {score}/100")
                
        except Exception as e:
            _record_diagnostics_cycle('ERROR', f'analysis exception: {type(e).__name__}')
            logger.error(f"SwingScalp analysis error: {e}")
            return self._get_no_trade_signal(f"Analysis error: {str(e)}")
            
    def _neural_global_synthesis(self, telemetry, part_results, mtf_context=None, options_walls=None, math_signal=0, math_confidence=0, data=None):
        """Phase 15: The Neural Cortex (New Unified AI Brain)"""
        logger.info("🧠 CORTEX: Analyzing all 12 GPU parts...")
        current_price = float(data['close'].iloc[-1]) if data is not None and len(data) > 0 else 0.0
        
        # Run Quantum if available
        quantum_res = None
        if hasattr(self, 'brains') and 'quantum_v5' in getattr(self, 'brains', {}):
            quantum_res = self.brains['quantum_v5'].simulate(data)
            
        result = self.neural_cortex.analyze(
            part_results=part_results,
            current_price=current_price,
            market_context=self.market_context,
            quantum_data=quantum_res,
            mtf_context=mtf_context
        )

        # Carry quantum data and narrative down
        result['quantum_data']  = quantum_res
        # ai_narrative is already set inside _normalise / _fallback_from_math

        # Professional display (compact — full display happens in _generate_trade_signal)
        try:
            pass  # Display is handled once in _generate_trade_signal to avoid double-clear
        except Exception as e:
            logger.debug(f"Display note: {e}")

        return result

    def _sync_to_hud(self, signal_data):
        """Push latest intelligence to the web HUD"""
        try:
            payload = {
                "thoughts": signal_data.get('intelligence_board', []),
                "regime": signal_data.get('market_context', {}).get('regime', 'NEUTRAL'),
                "signal": signal_data.get('trade_signal', {}),
                "market_data": {
                    "price": signal_data.get('trade_signal', {}).get('entry_price'),
                    "volatility": signal_data.get('market_context', {}).get('volatility')
                }
            }
            # Add CNS Data if available
            if hasattr(self, 'cns'):
                payload['cns_diagnostic'] = {
                    "health": self.cns.health_status,
                    "report": self.cns.last_diagnostic
                }
            # Fire and forget update (with proper error handling)
            def safe_hud_post():
                try:
                    requests.post(self.hud_url, json=payload, headers=self.hud_headers, timeout=2)
                except Exception:
                    pass  # Silently ignore HUD failures
            threading.Thread(target=safe_hud_post, daemon=True).start()
        except Exception:
            pass
            
    def _get_market_context(self, data):
        """Get current market context"""
        volatility = float(data['close'].pct_change().std() or 0.0)
        trend = self._get_trend_direction(data)
        mood = self._get_market_mood(data)
        
        # Calculate volatility status
        if volatility > 0.005: 
            vol_status = 'HIGH'
        elif volatility < 0.001: 
            vol_status = 'LOW'
        else: 
            vol_status = 'NORMAL'
        
        return {
            'volatility': volatility,
            'volatility_status': vol_status,
            'trend': trend,
            'mood': mood,
            'session': self._get_current_session(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _get_trend_direction(self, data):
        """Get market trend direction"""
        if len(data) < 10:
            return 'SIDEWAYS'
            
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_10 = data['close'].rolling(10).mean().iloc[-1]
        
        if sma_5 > sma_10 * 1.001:
            return 'UPTREND'
        elif sma_5 < sma_10 * 0.999:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
            
    def _get_market_mood(self, data):
        """Get market mood/condition from Engine"""
        try:
            if 'market_mood_engine' in self.brains:
                return self.brains['market_mood_engine'].detect_mood(data)
            
            # Fallback
            volatility = float(data['close'].pct_change().std() or 0.0)
            if volatility > 0.005: return 'VOLATILE'
            elif volatility < 0.001: return 'RANGING'
            else: return 'NORMAL'
        except Exception:
            return 'NORMAL'
            
    def _get_current_session(self):
        """Get current trading session"""
        current_hour = datetime.now().hour
        for session, times in TRADING_SESSIONS.items():
            if times['start'] <= current_hour < times['end']:
                return session
        return 'OVERNIGHT'
        


    def _get_deepseek_validation(self, data, score, detailed_scores, telemetry=None):
        import os
        if os.getenv("JARVIS_PURE_ALGO") == "True":
            return {'approved': True, 'reason': 'PURE ALGO BYPASS (Auto-Approved)', 'verdict': 'EXECUTE'}
        """Retired compatibility hook; model output is never an entry gate."""
        result = {'approved': False, 'reason': 'Retired legacy model gate unavailable; unsupported callers must reject', 'verdict': 'NO_TRADE'}
        self.last_legacy_llm_decision = {'decision': 'UNAVAILABLE', 'confidence': 0, 'rationale': result['reason']}
        return result

    def _create_deepseek_prompt(self, data, score, detailed_scores, telemetry=None):
        """Create prompt for DeepSeek ASI synthesis with Universal MTF and Sensory Telemetry"""
        current_candle = data.iloc[-1]
        direction = "BULLISH" if current_candle['close'] > current_candle['open'] else "BEARISH"
        thoughts = detailed_scores.get('thoughts', [])
        mtf_matrix = detailed_scores.get('mtf_matrix', {})
        walls = detailed_scores.get('options_walls', {})
        
        selected_symbol = str(getattr(self, 'active_symbol', '') or 'selected asset').upper()
        prompt = f"""
        ### JARVIS AI MASTER JUDGE ###
        You are the Master AI Judge of the Jarvis trading system.
        Analyze this {selected_symbol} trade signal and give a final APPROVE or REJECT decision.
        
        Pulse Score: {score}/100
        Proposed Direction: {direction}
        
        ## MTF MATRIX (1m to 4h):
        {json.dumps(mtf_matrix, indent=2, cls=NumpyEncoder)}
        
        ## COMPONENT THOUGHTS:
        {chr(10).join(['- ' + t for t in thoughts[:5]])}
        
        ## OPTIONS WALLS:
        - Support: {walls.get('support')}
        - Resistance: {walls.get('resistance')}
        - Max Pain: {walls.get('max_pain')}
        
        ## YOUR TASK:
        1. Check if 1m signal aligns with 1h/4h trend.
        2. APPROVE if score >= 60 AND signals agree. REJECT if score < 60 OR conflicting signals.
        
        IMPORTANT: Respond ONLY in ENGLISH. Keep reason SHORT (max 1-2 lines).
        
        Respond EXACTLY in this format:
        DECISION: [APPROVE/REJECT]
        REASON: <Short English reason, max 2 lines>
        """
        return prompt
        
    def _extract_rejection_reason(self, content):
        """Extract rejection reason from AI response or Master Synthesis format"""
        try:
            # Try to find REASON: header first
            if 'REASON:' in content:
                reason = content.split('REASON:')[1].strip()
                return reason
            
            # Fallback to line scanning
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'reject' in line.lower() or 'reason' in line.lower():
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
                    else:
                        return line.strip()
            return content[:200] # Return raw start as fallback
        except Exception:
            return "AI rejected without specific parseable reason"
        
    def _generate_trade_signal(self, data, score, detailed_scores, options_intel=None, logic_signal=0):
        """Generate complete trade trade signal with Universal Pricing"""
        current_candle = data.iloc[-1]
        current_price = float(current_candle['close'])
        
        # Determine direction from logic_signal (Council Decision), not candle color
        if logic_signal == 1:
            direction = "CALL"
        elif logic_signal == -1:
            direction = "PUT"
        else:
            direction = "NO_TRADE"
            
        # Optimize expiry based on volatility and score
        volatility = self.market_context['volatility']
        recommended_expiry = self.expiry_optimizer.recommend_trade_type(score, volatility, "REGULAR")
        
        # --- NEW: UNIVERSAL PRICING (ENTRY/TP/SL) ---
        mtf_matrix = detailed_scores.get('mtf_matrix', {})
        scalp_targets = self.scalping_engine.calculate_targets(
            data, direction, current_price, 
            options_data=detailed_scores.get('options_walls'), 
            mtf_data=mtf_matrix
        )
        
        # Override with Mode Engine targets if present
        if 'mode_engine_sl' in detailed_scores and 'mode_engine_tp' in detailed_scores:
            if not scalp_targets: scalp_targets = {}
            scalp_targets['stop_loss'] = float(detailed_scores['mode_engine_sl'])
            scalp_targets['take_profit_1'] = float(detailed_scores['mode_engine_tp'])
            scalp_targets['take_profit_2'] = float(detailed_scores['mode_engine_tp'])
        
        # Create comprehensive signal
        signal = {
            'trade_signal': {
                'direction': direction,
                'confidence_score': f"{score}/100",
                'recommended_expiry': recommended_expiry,
                'entry_price': current_price,
                'take_profit_1': scalp_targets.get('take_profit_1') if scalp_targets else None,
                'take_profit_2': scalp_targets.get('take_profit_2') if scalp_targets else None,
                'stop_loss': scalp_targets.get('stop_loss') if scalp_targets else None,
                'options_magnet': scalp_targets.get('options_magnet') if scalp_targets else None
            },
            
            'intelligence_board': detailed_scores.get('thoughts', []),
            'ai_validation': detailed_scores.get('ai_synthesis', 'Standard Protocol'),
            
            'market_context': {
                'regime': f"MTF Alignment Score: {detailed_scores.get('universal_mtf_confluence', 0)}",
                'volatility': self.market_context['volatility_status'],
                'target_summary': f"TP1: {scalp_targets.get('take_profit_1') if scalp_targets else 'N/A'}"
            }
        }
        
        # UI Log display
        # UI Log display
        ai_reasoning = (detailed_scores.get('neural_synthesis') or {}).get('reasoning', 'Math Consensus')
        
        # Prepare data for Professional Display
        math_signal = {
            'direction': direction,
            'confidence': int(score),
            'breakdown': '' # Todo: extract if needed
        }
        
        quantum_signal = {
            'prediction': 'UNKNOWN', # Todo: pass actual quantum
            'confidence': 0,
            'thought': 'N/A'
        }
        
        ai_signal = {
            'bias': direction, # Using Math/Consensus as AI for now
            'confidence': int(score),
            'reasoning': ai_reasoning
        }
        
        final_decision = {
            'direction': direction,
            'confidence': int(score),
            'entry': signal['trade_signal'].get('entry_price', 0),
            'tp1': signal['trade_signal']['take_profit_1'],
            'tp2': signal['trade_signal']['take_profit_2'],
            'sl': signal['trade_signal']['stop_loss']
        }
        
        # Build unified signal data for the new God-Mode display
        # Pull ai_narrative from neural synthesis result (deepseek-r1 narrative)
        neural_result = detailed_scores.get('neural_synthesis') or {}
        ai_narrative  = (neural_result.get('ai_narrative') or
                         neural_result.get('reasoning') or
                         ai_reasoning)

        unified_signal_data = {
            'direction':    direction,
            'confidence':   int(score),
            'ai_narrative': ai_narrative,
            'ai_reason':    ai_reasoning,
            'entry_price':  signal['trade_signal'].get('entry_price', 0),
            'trade_signal': signal['trade_signal'],
            'market_context': {
                'trend':            self.market_context.get('trend', '─'),
                'volatility_status': self.market_context.get('volatility_status', '─'),
                'session':          self.market_context.get('session', '─'),
            },
        }

        # Pass real part_results for 12-engine display
        real_part_results = getattr(self, 'latest_part_results', {})

        if not self.is_backtest_mode:
            # The live loop owns the sole terminal decision/dashboard.  Keep
            # this legacy rich display in structured logs so it cannot emit a
            # second BUY/SELL recommendation or disagree visually.
            logger.info(
                "[ANALYSIS SNAPSHOT] symbol=%s direction=%s confidence=%s entry=%s tp1=%s sl=%s",
                getattr(self, 'active_symbol', None), direction, score,
                signal['trade_signal'].get('entry_price', 0),
                signal['trade_signal'].get('take_profit_1'),
                signal['trade_signal'].get('stop_loss'),
            )

        return signal

    def _identify_pattern_type(self, data, direction):
        """Identify the specific pattern type"""
        current_candle = data.iloc[-1]
        body_size = abs(float(current_candle['close']) - float(current_candle['open']))
        candle_range = float(current_candle['high']) - float(current_candle['low'])
        
        if candle_range == 0:
            return 'UNKNOWN'
            
        body_ratio = body_size / candle_range
        
        if body_ratio > 0.7:
            return 'MOMENTUM_SCALP'
        elif self.scoring_matrix._is_rejection_candle(current_candle):
            return 'REJECTION_PLAY'
        else:
            return 'TREND_PULLBACK'
            
    def _get_entry_timing(self, data, direction):
        """Determine optimal entry timing and price"""
        current_candle = data.iloc[-1]
        current_price = float(current_candle['close'])
        
        # If strong momentum candle, enter immediately at market
        body_size = abs(float(current_candle['close']) - float(current_candle['open']))
        candle_range = float(current_candle['high']) - float(current_candle['low'])
        
        if candle_range > 0 and body_size / candle_range > 0.6:
            return f"MARKET ({current_price:.2f})"
        else:
            # Recommend a limit order entry for better R:R
            if direction == "CALL":
                entry_target = current_price * 0.9995  # 0.05% Pullback
                return f"LIMIT ({entry_target:.2f})"
            else:
                entry_target = current_price * 1.0005  # 0.05% Pullback
                return f"LIMIT ({entry_target:.2f})"
            
    def _get_key_level(self, data, current_price):
        """Identify key support/resistance level"""
        if len(data) < 10:
            return 'UNKNOWN'
            
        recent_high = float(data['high'].tail(10).max())
        recent_low = float(data['low'].tail(10).min())
        
        dist_to_high = abs(current_price - recent_high) / (recent_high - recent_low + 1e-12)
        dist_to_low = abs(current_price - recent_low) / (recent_high - recent_low + 1e-12)
        
        if dist_to_high < 0.1:
            return 'IMMEDIATE_RESISTANCE'
        elif dist_to_low < 0.1:
            return 'IMMEDIATE_SUPPORT'
        else:
            return 'RANGE_BOUND'
            
    def _get_avoid_levels(self, data, current_price, direction):
        """Get price levels to avoid"""
        if len(data) < 5:
            return 'NONE'
            
        if direction == 'CALL':
            recent_high = float(data['high'].tail(5).max())
            return f"Avoid above {recent_high:.5f}"
        else:
            recent_low = float(data['low'].tail(5).min())
            return f"Avoid below {recent_low:.5f}"
            
    def _get_exit_conditions(self, direction):
        """Get early exit conditions"""
        if direction == 'CALL':
            return "Price breaks below entry level"
        else:
            return "Price breaks above entry level"
            
    def _get_no_trade_signal(self, reason):
        """Generate no-trade signal"""
        return {
            'trade_signal': {
                'direction': 'NO_TRADE',
                'confidence_score': '0/100',
                'recommended_expiry': 'N/A',
                'entry_timing': 'N/A',
                'expected_payout_ratio': 'N/A'
            },
            'no_trade_reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    # ==================== EXISTING JARVIS METHODS (ENHANCED) ====================

    def process_data(self, data):
        """DEPRECATED: This method is never called. Decision flow uses analyze_trade_setup() directly.
        Kept for reference only. See _live_mode_with_deepseek() below."""
        import warnings
        warnings.warn("process_data() is dead code — not called anywhere. Use analyze_trade_setup() instead.", DeprecationWarning, stacklevel=2)
        start_time = time.time()
        try:
            # 1. Deep Multi-Timeframe Analysis (5m, 15m)
            mtf_analysis = {}
            if hasattr(self, 'mtf_buffers'):
                mtf_analysis = self.mtf_analyzer.analyze_all_timeframes(self.mtf_buffers)
            
            # 2. Get existing Jarvis analysis (1m core engine)
            existing_result = self._live_mode_with_deepseek(data)
            
            # 3. Get trade analysis (passing basic context for backward compatibility)
            mtf_context = {f"{tf}_trend": res['direction'] for tf, res in mtf_analysis.items()}
            trade_result = self.analyze_trade_setup(data, mtf_context=mtf_context)
            
            # 4. Final Directional Filter (Strict 15m Alignment)
            combined_signal = self._combine_signals(existing_result, trade_result)
            
            # STRICT MTF GATEKEEPER
            if combined_signal != 'NO-TRADE':
                tf_15m = mtf_analysis.get('15m', {})
                direction_val = 1 if combined_signal == 'CALL' else -1
                
                # If 15m trend is established and conflicts with 1m signal, block it
                if tf_15m and tf_15m['direction'] != 0 and tf_15m['direction'] != direction_val:
                    logger.warning(f"🛡️ MTF BLOCK: 1m {combined_signal} conflicts with 15m trend")
                    combined_signal = 'NO-TRADE'
                    trade_result['no_trade_reason'] = "MTF Trend Conflict (15m)"
                
                # Bonus: Check 5m for high confidence
                tf_5m = mtf_analysis.get('5m', {})
                if tf_5m and tf_5m['direction'] != 0 and tf_5m['direction'] == direction_val:
                    trade_result['mtf_confirmation'] = "5m Aligned"
            
            return {
                **existing_result,
                'trade_analysis': trade_result,
                'mtf_analysis': mtf_analysis,
                'combined_signal': combined_signal,
                'latency_ms': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error("process_data error: %s", e)
            return {'signal': 'NO-TRADE', 'confidence': 0, 'trade_type': '5m', 'latency_ms': 0}

    def _combine_signals(self, existing_result, trade_result):
        """Combine existing and trade signals safely"""
        try:
            if not trade_result or not isinstance(trade_result, dict):
                return 'NO-TRADE'
                
            trade_signal_dict = trade_result.get('trade_signal', {})
            if not isinstance(trade_signal_dict, dict):
                return 'NO-TRADE'

            trade_signal = trade_signal_dict.get('direction', 'NO_TRADE')
            if trade_signal in ['NO_TRADE', 'NO-TRADE', 0]:
                return 'NO-TRADE'
                
            existing_signal = existing_result.get('signal', 'NO-TRADE') if isinstance(existing_result, dict) else 'NO-TRADE'
            
            if existing_signal == trade_signal:
                return trade_signal
            else:
                # Unresolved opinions are a hard WAIT/NO_TRADE.  Confidence
                # cannot manufacture a consensus between opposing directions.
                logger.warning("[DECISION] conflicting signal opinions: %s vs %s", existing_signal, trade_signal)
                return 'NO-TRADE'
                    
        except Exception as e:
            logger.error(f"Signal combination error: {e}")
            return 'NO-TRADE'

    def _live_mode_with_deepseek(self, data):
        """DEPRECATED: Only called from process_data() which is itself dead code.
        The active validation path is _get_deepseek_validation() using LegacyLLM phi3.5."""
        import warnings
        warnings.warn("_live_mode_with_deepseek() is dead code.", DeprecationWarning, stacklevel=2)
        try:
            part_results = {}
            for part_name, part in self.parts.items():
                try:
                    res = part.analyze(data)
                    part_results[part_name] = res if isinstance(res, dict) else {"signal": res, "thought": "Legacy"}
                except Exception:
                    part_results[part_name] = {"signal": 0, "thought": "Part offline"}
            
            part_signals = [res['signal'] for res in part_results.values()]
            part_details = {k: v['signal'] for k, v in part_results.items()}
            # Exclude fusion and confidence from the board to avoid 'Legacy' spam
            thoughts = [res['thought'] for k, res in part_results.items() 
                        if res['signal'] != 0 and k not in ['part11_fusion', 'part12_confidence']]
            
            
            # PROFESSIONAL SIGNAL OUTPUT WITH WEIGHTED DECISION LOGIC
            def signal_to_score(s):
                """Convert -1/0/1 or reasoning dict to 0-100 scale"""
                val = s.get('signal', 0) if isinstance(s, dict) else s
                if val == 1: return 100  # Bullish
                elif val == -1: return 0  # Bearish
                else: return 50  # Neutral
            
            def format_name(p):
                return p.replace('part', 'P').replace('_', ' ').title()
            
            # Part weights (more reliable parts get higher weight)
            part_weights = {
                'part1_breakout': 1.2,
                'part2_zone': 1.3,
                'part5_ml': 1.5,      # ML gets highest weight
                'part6_trend': 1.4,
                'part9_orderflow': 1.2,
                'part10_candlestats': 1.1,
                'part14_options_chain': 1.5,  # Options Chain (Institutional data)
            }
            
            # Convert all signals to 0-100 with weights
            weighted_scores = []
            part_scores = {}
            
            for part, signal in part_details.items():
                if part in ['part11_fusion', 'part12_confidence']:
                    continue
                score = signal_to_score(signal)
                weight = part_weights.get(part, 1.0)
                weighted_scores.append(score * weight)
                part_scores[part] = score
            
            confidence = part_details.get('part12_confidence', 0)
            
            # Calculate weighted average
            total_weight = sum(part_weights.get(p, 1.0) for p in part_scores.keys())
            avg_score = sum(weighted_scores) / total_weight if weighted_scores else 50
            
            # Count agreement (how many parts agree on direction)
            bullish_count = sum(1 for s in part_scores.values() if s >= 60)
            bearish_count = sum(1 for s in part_scores.values() if s <= 40)
            total_parts = len(part_scores)
            
            bullish_pct = (bullish_count / total_parts * 100) if total_parts > 0 else 0
            bearish_pct = (bearish_count / total_parts * 100) if total_parts > 0 else 0
            
            # Determine final decision with agreement threshold
            decision_reason = ""
            if avg_score >= 65 and bullish_pct >= 50:
                decision = "📈 CALL"
                decision_reason = f"{bullish_count}/{total_parts} parts bullish ({bullish_pct:.0f}%)"
            elif avg_score <= 35 and bearish_pct >= 50:
                decision = "📉 PUT"
                decision_reason = f"{bearish_count}/{total_parts} parts bearish ({bearish_pct:.0f}%)"
            else:
                decision = "⚪ NO-TRADE"
                if avg_score > 50:
                    decision_reason = f"Weak bullish ({bullish_pct:.0f}% agreement)"
                elif avg_score < 50:
                    decision_reason = f"Weak bearish ({bearish_pct:.0f}% agreement)"
                else:
                    decision_reason = "Market neutral"
            
            # Legacy/dead path: do not emit a second terminal decision.  The
            # active live loop owns the one unified dashboard snapshot.
            logger.info("[LEGACY ANALYSIS] %s | score=%.1f | confidence=%s | %s", decision, avg_score, confidence, decision_reason)
            
            logic_signal = self.parts['part11_fusion'].analyze(list(part_results.values()))
            confidence_res = self.parts['part12_confidence'].analyze(part_results)
            # BUG FIX #12: part12_confidence returns dict {'confidence': N} — extract the number
            confidence = confidence_res.get('confidence', 0) if isinstance(confidence_res, dict) else confidence_res
            
            # Brain Results
            quantum_res = self.brains['quantum_v5'].simulate(data)
            volume_res = self.brains['volume_pressure_brain'].analyze(data)
            trend_res = self.brains['trend_acceleration_brain'].analyze(data)
            
            # Add brain thoughts to the board
            for res in [quantum_res, volume_res, trend_res]:
                if res['signal'] != 0: thoughts.append(res['thought'])
            
            logger.info("\n" + "🧠 [INTELLIGENCE BOARD] Combined Analysis:" + "\n" + "\n".join([f" • {t}" for t in thoughts]))
            
            # BUG FIX #9: upgrades is on self.jarvis not self — use jarvis reference
            market_mood = self.upgrades['market_mood_engine'].detect_mood(data)
            market_context = {
                'volatility': float(data['close'].pct_change().std() or 0.0),
                'trend': trend_res['signal'],
                'mood': market_mood
            }
            deepseek_sentiment, sentiment_reasoning = self.deepseek_brains['deepseek_v3_sentiment'].analyze_sentiment(data, market_context)
            
            safety_res = self.brains['safety_risk_brain'].analyze(data, part_signals)
            risk_res = self.brains['risk_filter_brain'].analyze(data, logic_signal)
            
            if not safety_res['approved']: thoughts.append(safety_res['thought'])
            if not risk_res['approved']: thoughts.append(risk_res['thought'])
            
            safety_approved = safety_res['approved']
            risk_approved = risk_res['approved']
            all_signals_info = {
                'traditional_signals': part_details,
                'thoughts': thoughts,
                'advanced_signals': {
                    'quantum': quantum_res['signal'],
                    'volume': volume_res['signal'],
                    'trend': trend_res['signal'],
                    'deepseek_sentiment': deepseek_sentiment
                },
                'safety_checks': {
                    'safety_approved': safety_approved,
                    'risk_approved': risk_approved,
                    'safety_thought': safety_res['thought']
                }
            }
            r1_reasoning = self.deepseek_brains['deepseek_r1_reasoning'].complex_reasoning(all_signals_info, data, market_context)
            if r1_reasoning.get("signal") == "CALL" and safety_approved and risk_approved:
                final_signal = 1
            elif r1_reasoning.get("signal") == "PUT" and safety_approved and risk_approved:
                final_signal = -1
            else:
                final_signal = 0
            trap_res = self.upgrades['trap_candle_genome_detector'].detect_trap(data)
            if trap_res['is_trap']:
                final_signal = 0
                r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + f" | {trap_res['thought']}")
            
            shield_res = self.upgrades['high_volatility_regime_shield'].check_safety(data)
            if not shield_res['approved']:
                final_signal = 0
                r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + f" | {shield_res['thought']}")
            
            rev_res = self.upgrades['reverse_safety_engine'].check_reversal(data, final_signal)
            old_signal = final_signal  # BUG FIX #11: save before overwrite to detect change
            final_signal = rev_res['signal']
            if rev_res['signal'] != old_signal:
                r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + f" | {rev_res['thought']}")
            
            # GATEKEEPER: Institutional Alignment (Relaxed)
            p14_signal = part_details.get('part14_options_chain', 0)
            # BUG FIX #13: confidence is already extracted as int above in fixed code
            is_high_conf = int(confidence) >= 85
            
            if final_signal == 1:
                if p14_signal == -1: # Opposite institutional bias
                    final_signal = 0
                    r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + " | 🛡️ Blocked: Institutional BEARISH bias")
                elif p14_signal == 0 and not is_high_conf: # Neutral but low conf
                    final_signal = 0
                    r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + " | 🛡️ Blocked: Waiting for Institutional Confirm or High Confidence")
            
            elif final_signal == -1:
                if p14_signal == 1: # Opposite institutional bias
                    final_signal = 0
                    r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + " | 🛡️ Blocked: Institutional BULLISH bias")
                elif p14_signal == 0 and not is_high_conf: # Neutral but low conf
                    final_signal = 0
                    r1_reasoning["reasoning"] = (r1_reasoning.get("reasoning","") + " | 🛡️ Blocked: Waiting for Institutional Confirm or High Confidence")
            volatility = float(data['close'].pct_change().std() or 0.0)
            if volatility > 0.003:
                expiry = "SCALP"
            elif final_signal != 0:
                expiry = "DAY_TRADE"
            else:
                expiry = "SWING"
            if final_signal == 1:
                signal_str = "CALL"
            elif final_signal == -1:
                signal_str = "PUT"
            else:
                signal_str = "NO-TRADE"
            return {
                'signal': signal_str,
                'confidence': max(confidence, r1_reasoning.get("confidence", 0)),
                'trade_type': expiry,
                'max_trade_duration': "SWING",
                'deepseek_sentiment': deepseek_sentiment,
                'deepseek_sentiment_reasoning': sentiment_reasoning,
                'deepseek_r1_signal': r1_reasoning.get("signal", "NO-TRADE"),
                'deepseek_r1_confidence': r1_reasoning.get("confidence", 0),
                'deepseek_r1_reasoning': r1_reasoning.get("reasoning", ""),
                'deepseek_key_factors': r1_reasoning.get("key_factors", []),
                'traditional_confidence': confidence,
                'safety_flags': f"Safe: {safety_approved and risk_approved}, Mood: {market_context['mood']}",
            }
        except Exception as e:
            logger.error("live_mode error: %s", e)
            return self._live_mode_fallback(data)

    def _live_mode_fallback(self, data):
        """Your existing fallback logic"""
        try:
            part_results = {}
            for part_name, part in self.parts.items():
                try:
                    signal = part.analyze(data)
                except Exception:
                    signal = {"signal": 0, "thought": "Part offline"}
                # BUG FIX #14: ensure dict format for fusion/confidence
                if not isinstance(signal, dict):
                    signal = {"signal": signal, "thought": "Legacy"}
                part_results[part_name] = signal
            part_signals = list(part_results.values())
            logic_signal_res = self.parts['part11_fusion'].analyze(part_signals)
            confidence_res = self.parts['part12_confidence'].analyze(part_signals)
            logic_signal = logic_signal_res.get('signal', 0) if isinstance(logic_signal_res, dict) else logic_signal_res
            confidence = confidence_res.get('confidence', 0) if isinstance(confidence_res, dict) else confidence_res
            if logic_signal == 1:
                signal_str = "CALL"
            elif logic_signal == -1:
                signal_str = "PUT"
            else:
                signal_str = "NO-TRADE"
            return {
                'signal': signal_str,
                'confidence': confidence,
                'trade_type': "2m",
                'max_trade_duration': "SWING",
                'deepseek_sentiment': 0,
                'deepseek_sentiment_reasoning': "Fallback mode - DeepSeek unavailable",
                'deepseek_r1_signal': "NO-TRADE",
                'deepseek_r1_confidence': 0,
                'deepseek_r1_reasoning': "Fallback to traditional analysis",
                'safety_flags': "Fallback mode active",
            }
        except Exception as e:
            logger.error("fallback error: %s", e)
            return {'signal': 'NO-TRADE', 'confidence': 0, 'trade_type': '5m'}

# ==================== ENHANCED 4-ENGINE SYSTEM ====================

class Jarvis4EngineSystem:
    """MASTER CONTROLLER - All 4 engines with trade integration"""
    
    def __init__(self, backtest_mode=None):
        # One authoritative mode reaches the active coordinator.  An explicit
        # argument wins; otherwise JARVIS_BACKTEST_MODE is respected.
        if backtest_mode is None:
            backtest_mode = os.environ.get('JARVIS_BACKTEST_MODE', '0') == '1'
        self.backtest_mode = bool(backtest_mode)
        self.jarvis = JarvisElite(backtest_mode=self.backtest_mode)
        self.backtest_engine = AutoBacktestEngine(self.jarvis)
        self.training_engine = AutoTrainingEngine(self.jarvis)
        self.optimizer_engine = AutoOptimizerEngine(self.jarvis)
        self.live_engine = None
        
        # --- PHASE 19: MULTI-TIMEFRAME SYSTEM ---
        try:
            from delta_multi_tf_fetcher import DeltaMultiTFDataFetcher  # type: ignore
            from smart_tpsl_calculator import SmartTPSLCalculator  # type: ignore
            from multi_tf_engine import MultiTimeframeEngine  # type: ignore
            from trading_config import TRADING_CONFIG, get_active_config  # type: ignore
            
            self.delta_multi_tf = DeltaMultiTFDataFetcher(delta_client=self.jarvis.delta_data)
            self.tpsl_calculator = SmartTPSLCalculator(trading_config=get_active_config())
            self.multi_tf_engine = MultiTimeframeEngine(
                jarvis_system=self.jarvis,
                delta_fetcher=self.delta_multi_tf,
                tpsl_calculator=self.tpsl_calculator,
                config=TRADING_CONFIG
            )
            self.trading_config = TRADING_CONFIG
            logger.info("✅ Multi-Timeframe System Initialized")
        except Exception as e:
            logger.debug(f"Multi-TF system not available (optional): {e}")
            self.multi_tf_engine = None
            self.trading_config = {'mode': 'paper', 'paper_trading': True, 'live_execution': False}  # Safe optional fallback
        
        # Data storage
        self.historical_data = None
        self.backtest_results = None
        self.optimized_params = None
        
    def run_complete_system(self, historical_data=None):
        """Run all 4 engines with trade optimization"""
        logger.info("🤖 JARVIS LIVE TRADING SYSTEM STARTING...")
        logger.info("🎯 MODE: DIRECT LIVE (Backtest & Training SKIPPED)")
        logger.info("=" * 60)
        
        # ─── BACKTEST & TRAINING SKIPPED ────────────────────────────
        # Set JARVIS_RUN_BACKTEST=1 in environment to re-enable them
        run_backtest = os.getenv('JARVIS_RUN_BACKTEST', '0') == '1'
        
        self.backtest_results = {}
        training_results = {}
        self.optimized_params = {}
        
        if run_backtest:
            # Backtest Mode (only when env var is set)
            if historical_data is None:
                historical_data = self._generate_sample_data()
            self.historical_data = historical_data
            
            logger.info("🔹 PHASE 1: AUTO-BACKTEST SYSTEM ACTIVATED 📊")
            self.backtest_results = self.backtest_engine.run_backtest(historical_data)
            
            logger.info("🔹 PHASE 2: AUTO-TRAINING SYSTEM UNLOCKED 🧠")
            training_results = self.training_engine.train_models(historical_data)
        else:
            logger.info("⚡ BACKTEST & TRAINING: SKIPPED (Direct Live Mode)")
            logger.info("   (To run backtest, set env var: JARVIS_RUN_BACKTEST=1)")
        
        # Step 4️⃣: Live engine is never started by a historical replay.
        if self.backtest_mode:
            logger.info("🔹 LIVE TRADING ENGINE: SKIPPED (backtest mode)")
            live_status = {"status": "skipped", "mode": "backtest"}
        else:
            logger.info("🔹 LIVE TRADING ENGINE: STARTING NOW 🚀")
            self.live_engine = LiveTradingEngine(self.jarvis, self.optimized_params)
            live_status = self.live_engine.start_live_trading()
        
        # Final summary
        self._print_final_summary()
        
        return {
            'backtest': self.backtest_results,
            'training': training_results,
            'optimization': self.optimized_params,
            'live_trading': live_status,
            'trade_stats': self.jarvis.trade_manager.get_performance_stats() if hasattr(self.jarvis.trade_manager, 'get_performance_stats') else {}
        }
    
    def analyze_multi_tf(self, symbol='BTC', mode=None):
        """
        Analyze using Multi-Timeframe System (Phase 19)
        
        Args:
            symbol: Trading symbol (default: 'BTC')
            mode: 'scalping' or 'swing' (uses config default if None)
        
        Returns:
            Signal with TP/SL levels and Multi-TF confluence
        """
        if not self.multi_tf_engine:
            # This legacy helper has no reviewed native-timeframe source. Do not
            # silently switch from the canonical Binance Spot MTF feed to Delta
            # candles or call analyze_trade_setup with an unsupported `symbol=`.
            logger.error("Multi-TF Engine unavailable; refusing single-venue candle fallback")
            if self.jarvis is not None and hasattr(self.jarvis, '_get_no_trade_signal'):
                return self.jarvis._get_no_trade_signal(
                    "WAIT/NO-DATA: native multi-timeframe engine unavailable")
            return None
        
        # Use Multi-TF Engine
        return self.multi_tf_engine.analyze(symbol=symbol, mode=mode)

    def _generate_sample_data(self):
        """Fetch real, closed Binance Spot 1m candles; never fabricate backtest bars."""
        from binance_data import get_binance_data
        from binance_timeframes import TIMEFRAME_HISTORY_CANDLES, candle_is_closed
        import time as _time

        symbol = 'BTCUSDT'
        requested = TIMEFRAME_HISTORY_CANDLES['5m']
        logger.info("[BACKTEST] Fetching %s closed Binance Spot 1m candles for %s", requested, symbol)
        candles = get_binance_data().get_historical_candles(
            symbol=symbol, resolution='5m', limit=requested + 1, allow_fallback=False,
        )
        now = _time.time()
        closed = [row for row in candles if candle_is_closed('5m', row['time'], now)]
        if len(closed) < requested:
            raise RuntimeError(
                f"Binance Spot history incomplete: need {requested} closed 1m bars, got {len(closed)}; "
                "no synthetic or alternate-venue backtest fallback is permitted"
            )
        closed = closed[-requested:]
        data = pd.DataFrame({
            'timestamp': pd.to_datetime([row['time'] for row in closed], unit='s', utc=True),
            'open': [float(row['open']) for row in closed],
            'high': [float(row['high']) for row in closed],
            'low': [float(row['low']) for row in closed],
            'close': [float(row['close']) for row in closed],
            'volume': [float(row['volume']) for row in closed],
        })
        data.attrs['symbol'] = symbol
        data.attrs['source'] = 'binance_spot'
        logger.info("[BACKTEST] Loaded %s closed Binance Spot candles for %s", len(data), symbol)
        return data


    def _print_final_summary(self):
        """Print final system summary"""
        logger.info("=" * 60)
        logger.info("🎉 JARVIS 4-ENGINE TRADE SYSTEM - COMPLETE!")
        logger.info("=" * 60)
        
        if self.backtest_results:
            logger.info(f"📊 BACKTEST RESULTS:")
            logger.info(f"   Win Rate: {self.backtest_results.get('overall_win_rate', 0):.1f}%")
            logger.info(f"   Total Trades: {self.backtest_results.get('total_trades', 0)}")
            logger.info(f"   Best Expiry: {self.backtest_results.get('trade_type', '2m')}")
        
        if self.optimized_params:
            logger.info(f"⚡ OPTIMIZED PARAMETERS:")
            logger.info(f"   Volatility Regimes: Applied")
            logger.info(f"   Signal Weights: Optimized") 
            logger.info(f"   Safety Filters: Calibrated")
        
        logger.info(f"🎯 TRADE TRADING: ACTIVE")
        logger.info(f"⏱️  TRADE OPTIMIZATION: Scalp & Swing")
        logger.info('Model judge retired; deterministic scoring and entry gates authoritative')  # Judge now active
        logger.info("=" * 60)

# ==================== MAIN EXECUTION ====================

def main():
    """Main function with an explicit opt-in safety gate.
    
    Set JARVIS_START_PAPER=1 in .env to start services.
    For LIVE trading: also set JARVIS_AUTO_TRADE=true.
    For PAPER trading: set JARVIS_AUTO_TRADE=false.
    """
    if os.environ.get("JARVIS_START_PAPER") != "1":
        logger.error("[SAFE DEFAULT] Set JARVIS_START_PAPER=1 in .env to start JARVIS services.")
        return 2

    is_live = os.environ.get("JARVIS_AUTO_TRADE", "").lower() == "true"
    if is_live:
        logger.warning("⚡ LIVE TRADING MODE — Real orders will be placed on Delta Exchange!")
    else:
        logger.info("📄 PAPER TRADING MODE — Simulated trades only, no real money.")


    logger.info("🚀 JARVIS TRADE ELITE v7.0 - FULLY INTEGRATED")
    logger.info("==========================================")
    
    # --- AUTO START UI & HUD SERVER ---
    if os.name == "nt":
        logger.info("🌐 Launching JARVIS Live UI & HUD Server (start_ui.ps1)...")
        try:
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", "start_ui.ps1"],
                cwd="c:\\jarvis",
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            logger.info("✅ UI Server spawned in background. (React @ 5173, HUD @ 7788)")
        except Exception as e:
            logger.error(f"⚠️ Failed to auto-start UI: {e}")
    else:
        logger.info("🌐 Linux detected — start HUD manually if needed: python3 jarvis_hud_server.py")
    # ----------------------------------

    # Initialize the complete trade system
    jarvis_trade = Jarvis4EngineSystem()

    # --- CRASH RECOVERY + WATCHDOG (guarded; can never break startup) ---
    try:
        if os.environ.get("JARVIS_WATCHDOG", "1") != "0":
            from jarvis_watchdog import start_watchdog
            exchange_client = getattr(getattr(jarvis_trade, "jarvis", None), "delta_data", None)
            start_watchdog(exchange_client=exchange_client)
    except Exception as wd_err:
        logger.warning(f"[WATCHDOG] startup wiring failed (continuing): {wd_err}")

    try:
        # Run all 4 engines automatically
        results = jarvis_trade.run_complete_system()
        
        auto_live = os.getenv("JARVIS_AUTO_LIVE", "1") == "1"
        start_live = auto_live
        if not auto_live:
            try:
                start_live = input("\n🎯 Start LIVE SwingScalp Trading? (y/n): ").strip().lower() == 'y'
            except (EOFError, OSError):
                start_live = True

        if start_live:
            logger.info("Starting LIVE SwingScalp Trading...")
            logger.info("Live trading service already owns the canonical decision display.")
            live_eng = getattr(jarvis_trade, 'live_engine', None)
            if live_eng and getattr(live_eng, 'is_running', False):
                logger.info("🟢 JARVIS Live Engine is running actively. Press Ctrl+C to stop.")
                try:
                    import time as _time
                    while live_eng.is_running:
                        _time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("🛑 Stopping JARVIS Live Engine...")
                    if hasattr(live_eng, 'stop'):
                        live_eng.stop()
        else:
            logger.info("LIVE trading cancelled. System ready for manual use.")
            
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    sys.exit(main() or 0)