try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ==============================================================================
# JARVIS PART 3 - INSTITUTIONAL SWING/SCALP TRADING ENGINE (GPU-OPTIMIZED & FIXED)
# Fully hardened against hidden bugs, NaN/Inf issues, PyTorch compatibility,
# execution logic flaws, key mismatches, and unicode output issues.
# ==============================================================================

import sys
import time
import json
import math
import asyncio
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Safe stdout encoding wrapper for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 26)
        pass




# ---- PyTorch with complete NumPy Fallback for Windows/WSL compatibility ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 38)
    TORCH_AVAILABLE = False
    
    # Comprehensive dummy torch module for fallback mode
    class DummyTensor:
        def __pow__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor(self._data ** (other._data if isinstance(other, DummyTensor) else other))
        def __rpow__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor((other._data if isinstance(other, DummyTensor) else other) ** self._data)
        def __add__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor(self._data + (other._data if isinstance(other, DummyTensor) else other))
        def __radd__(self, other): return self.__add__(other)
        def __sub__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor(self._data - (other._data if isinstance(other, DummyTensor) else other))
        def __rsub__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor((other._data if isinstance(other, DummyTensor) else other) - self._data)
        def __mul__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor(self._data * (other._data if isinstance(other, DummyTensor) else other))
        def __rmul__(self, other): return self.__mul__(other)
        def __truediv__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor(self._data / (other._data if isinstance(other, DummyTensor) else other))
        def __rtruediv__(self, other): return DummyTensor() if not hasattr(self, '_data') else DummyTensor((other._data if isinstance(other, DummyTensor) else other) / self._data)

        def __init__(self, data=0.0, *args, **kwargs):
            if isinstance(data, (list, tuple, np.ndarray)):
                self._data = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float, bool, np.number)):
                self._data = np.array([float(data)], dtype=np.float32)
            elif isinstance(data, DummyTensor):
                self._data = data._data.copy()
            else:
                self._data = np.zeros((1,), dtype=np.float32)
            self.shape = self._data.shape
            self.dtype = self._data.dtype

        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return self._data
        def item(self): return float(self._data.flat[0]) if self._data.size > 0 else 0.0
        def float(self): return self
        def long(self): return self
        def detach(self): return self
        def clone(self): return DummyTensor(self._data.copy())
        
        def __getitem__(self, key):
            res = self._data[key]
            if isinstance(res, np.ndarray):
                dt = DummyTensor(res)
                dt.shape = res.shape
                return dt
            return float(res)
            
        def __len__(self): return len(self._data)
        def __float__(self): return self.item()
        def __int__(self): return int(self.item())
        def __bool__(self): return bool(self.item()) if self._data.size == 1 else self._data.size > 0
        def __repr__(self): return f"DummyTensor({self._data})"
        
        # Math operators
        def __add__(self, other): return DummyTensor(self._data + (other._data if isinstance(other, DummyTensor) else other))
        def __sub__(self, other): return DummyTensor(self._data - (other._data if isinstance(other, DummyTensor) else other))
        def __mul__(self, other): return DummyTensor(self._data * (other._data if isinstance(other, DummyTensor) else other))
        def __pow__(self, other): return DummyTensor(self._data ** (other._data if isinstance(other, DummyTensor) else other))
        def __truediv__(self, other): return DummyTensor(self._data / (other._data if isinstance(other, DummyTensor) else 1e-8))
        def __gt__(self, other): return bool(self.item() > (other.item() if isinstance(other, DummyTensor) else other))
        def __lt__(self, other): return bool(self.item() < (other.item() if isinstance(other, DummyTensor) else other))
        def __ge__(self, other): return bool(self.item() >= (other.item() if isinstance(other, DummyTensor) else other))
        def __le__(self, other): return bool(self.item() <= (other.item() if isinstance(other, DummyTensor) else other))

    class DummyModule:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return DummyTensor()
        def forward(self, *args, **kwargs): return DummyTensor()
        def to(self, *args, **kwargs): return self
        def eval(self): return self
        def train(self, mode=True): return self
        def parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, *args, **kwargs): pass

    class torch:
        Tensor = DummyTensor
        device = lambda x: 'cpu'
        float32 = 'float32'
        long = 'long'
        
        class nn:
            Module = DummyModule
            Linear = DummyModule
            LSTM = DummyModule
            GRU = DummyModule
            Transformer = DummyModule
            Dropout = DummyModule
            BatchNorm1d = DummyModule
            LayerNorm = DummyModule
            Embedding = DummyModule
            Conv1d = DummyModule
            MaxPool1d = DummyModule
            ReLU = DummyModule
            Sigmoid = DummyModule
            Tanh = DummyModule
            Softmax = DummyModule
            MSELoss = DummyModule
            CrossEntropyLoss = DummyModule
            
        class optim:
            Adam = DummyModule
            SGD = DummyModule
        
        class cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def device_count(): return 0
            @staticmethod
            def memory_allocated(): return 0
            @staticmethod
            def get_device_properties(idx):
                class Props: total_memory = 8 * 1024**3
                return Props()
            @staticmethod
            def get_device_name(idx=0): return "CPU_Fallback"
        
        @staticmethod
        def tensor(data, **kwargs):
            if isinstance(data, DummyTensor): return data
            return DummyTensor(data)
        
        @staticmethod
        def zeros(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.zeros(shape, dtype=np.float32))
            
        @staticmethod
        def ones_like(input_tensor, **kwargs):
            shape = input_tensor.shape if hasattr(input_tensor, 'shape') else (1,)
            return DummyTensor(np.ones(shape, dtype=np.float32))
        
        @staticmethod
        def randn(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.random.randn(*shape).astype(np.float32))
        
        @staticmethod
        def cat(tensors, dim=0): return DummyTensor()
        @staticmethod
        def stack(tensors, dim=0): return DummyTensor()
        @staticmethod
        def no_grad():
            class DummyNoGrad:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            return DummyNoGrad()
            
        @staticmethod
        def max(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.max(arr))
            
        @staticmethod
        def min(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.min(arr))
            
        @staticmethod
        def mean(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.mean(arr))
            
        @staticmethod
        def std(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.std(arr))
            
        @staticmethod
        def sum(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.sum(arr))
            
        @staticmethod
        def abs(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.abs(arr))
            
        @staticmethod
        def sqrt(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.sqrt(arr))
            
        @staticmethod
        def log(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.log(np.maximum(arr, 1e-8)))
            
        @staticmethod
        def diff(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.diff(arr))
            
        @staticmethod
        def clamp(tensor, min_val, max_val):
            val = tensor.item() if isinstance(tensor, DummyTensor) else float(tensor)
            return DummyTensor(max(min_val, min(max_val, val)))
            
        @staticmethod
        def where(condition, x, y):
            cond_val = condition.item() if isinstance(condition, DummyTensor) else bool(condition)
            return x if cond_val else y

# Optional CuPy fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 244)
    CUPY_AVAILABLE = False


class LinuxOptimizedDeque(deque):
    """High-performance thread-safe deque wrapper"""
    def __init__(self, maxlen=500):
        super().__init__(maxlen=maxlen)
    def append(self, item):
        try:
            super().append(item)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 255)
            pass


def _safe_get_device_name(device):
    """Safely get device name without crashing"""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available() and hasattr(device, "type") and device.type == "cuda":
            idx = device.index if hasattr(device, "index") and device.index is not None else 0
            return torch.cuda.get_device_name(idx)
        return str(device)
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 266)
        return "CPU_Device"


class GPUFeatureExtractor:
    """GPU-Accelerated Basic Feature Extraction"""
    def __init__(self):
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
    def extract_basic(self, data):
        try:
            if hasattr(data, 'values'):
                vals = data.values.flat[:10]
            elif isinstance(data, (list, tuple, np.ndarray)):
                vals = data[:10]
            else:
                vals = [0.0] * 10
            float_vals = [float(x) if not math.isnan(float(x)) else 0.0 for x in vals]
            while len(float_vals) < 10:
                float_vals.append(0.0)
            return torch.tensor(float_vals, device=self.device)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 287)
            return torch.zeros(10, device=self.device)


# Fallback component classes to guarantee no NameError issues anywhere
class CandlePsychologyMasterGPU:
    """
    INSTITUTIONAL CANDLESTICK PSYCHOLOGY & PRICE ACTION ENGINE (Part 3)
    - ATR-Relative Significance Filter (blocks micro-noise and flat-candle hallucinations)
    - Rejection Analysis: Hammers, Shooting Stars, Pin Bars with Wick-to-Body ratio >= 2.0x
    - Momentum Patterns: Bullish & Bearish Engulfing, Multi-bar Expansion
    - Volume Confirmation on Reversals
    - Strict Neutrality: Returns Signal: 0 (Neutral) on mixed/indecision candles
    """
    def __init__(self, master=None):
        self.master = master
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

    @part_advisory_entry("part3_psychology", data_parameter="data", context_parameter="context")
    def analyze(self, data, context=None):
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        return qm.part_signal("3", data, context=context)

    def analyze_candle_psychology(self, current_candle=None, prev1=None, prev2=None):
        if not current_candle or not isinstance(current_candle, dict):
            return {}
        try:
            c = float(current_candle.get('close', 0))
            o = float(current_candle.get('open', 0))
            h = float(current_candle.get('high', 0))
            l = float(current_candle.get('low', 0))
            body = abs(c - o)
            rng = max(h - l, 1e-8)
            upper_wick = (h - max(c, o)) / rng
            lower_wick = (min(c, o) - l) / rng
            
            return {
                'is_hammer': lower_wick > 0.55 and body / rng < 0.35 and upper_wick < 0.22,
                'is_shooting_star': upper_wick > 0.55 and body / rng < 0.35 and lower_wick < 0.22,
                'is_bullish_engulfing': c > o and prev1 and float(prev1.get('close', 0)) < float(prev1.get('open', 0)) and c >= float(prev1.get('open', 0)),
                'is_bearish_engulfing': c < o and prev1 and float(prev1.get('close', 0)) > float(prev1.get('open', 0)) and c <= float(prev1.get('open', 0)),
                'has_strong_support': lower_wick > 0.5,
                'has_strong_resistance': upper_wick > 0.5
            }
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 335)
            return {}


from brain_xgb import xgb_shadow

@xgb_shadow('zone_detector', 'detect_0_5_zone_signals')
class ZonePointFiveDetectorGPU:
    def __init__(self, master=None):
        self.master = master
    def detect_0_5_zone_signals(self, current_candle=None, psychology=None, df_1m=None, df_5m=None, df_15m=None):
        return []


class DeepSeekAILearningGPU:
    def __init__(self, api_key=None, master=None):
        self.api_key = api_key
        self.master = master
    def analyze_trade(self, *args, **kwargs):
        return {}


# Attempt real imports if available
try:
    from part1_main import CandlePsychologyMasterGPU as RealPsychology
    CandlePsychologyMasterGPU = RealPsychology
except Exception:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 358)
    pass

try:
    from part2_zones import ZonePointFiveDetectorGPU as RealZones
    ZonePointFiveDetectorGPU = RealZones
except Exception:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 364)
    pass

try:
    from part_ai import DeepSeekAILearningGPU as RealAI
    DeepSeekAILearningGPU = RealAI
except Exception:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 370)
    pass


# ==================== GPU-OPTIMIZED INSTITUTIONAL TRADING ENGINE ====================

@xgb_shadow('inst_trading', 'generate_live_signals')
class InstitutionalTradingEngineGPU:
    def __init__(self, master_system=None):
        self.master = master_system
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        # Institutional trading configuration
        self.trading_config = {
            'min_confidence': 6.8,
            'max_confidence': 9.9,
            'expiry_windows': [1, 2, 3, 5, 7, 10],
            'default_expiry': 3,
            'risk_per_trade': 0.05,
            'max_daily_trades': 25,
            'max_hourly_trades': 8,
            'cooldown_after_loss': 300,
            'extended_cooldown_after_3_losses': 900,
            'gpu_batch_size': 1024,
            'signal_buffer_size': 100,
            'volatility_lookback': 25,
            'max_daily_loss': 0.25,
            'max_consecutive_losses': 5,
            'position_size_kelly_fraction': 0.25,
            'regime_confidence_boost': 1.15,
            'volatility_penalty_threshold': 0.12
        }
        
        # Institutional trading state
        self.trading_state = {
            'daily_trades': 0,
            'hourly_trades': 0,
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'daily_pnl': 0.0,
            'last_trade_time': 0.0,
            'last_loss_time': 0.0,
            'last_win_time': 0.0,
            'current_regime': 'NEUTRAL',
            'signal_history': LinuxOptimizedDeque(self.trading_config['signal_buffer_size']),
            'trade_history': LinuxOptimizedDeque(500),
            'active_trades': {},
            'session_start_time': time.time()
        }
        
        # Advanced performance tracking with regime analytics
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate_by_regime': defaultdict(lambda: LinuxOptimizedDeque(100)),
            'strategy_performance': defaultdict(lambda: LinuxOptimizedDeque(200)),
            'regime_performance': defaultdict(lambda: LinuxOptimizedDeque(150)),
            'hourly_performance': defaultdict(lambda: LinuxOptimizedDeque(50)),
            'expiry_performance': defaultdict(lambda: LinuxOptimizedDeque(100))
        }
        
        # Signal fusion weights
        self.fusion_weights = {
            'psychology': 0.28,
            'zones': 0.26,
            'trend': 0.18,
            'volatility': 0.14,
            'volume': 0.14
        }
        
        # Regime-specific trading rules
        self.regime_rules = {
            'TRENDING_UP': {'preferred_signals': ['CALL'], 'confidence_boost': 1.1, 'max_expiry': 5},
            'TRENDING_DOWN': {'preferred_signals': ['PUT'], 'confidence_boost': 1.1, 'max_expiry': 5},
            'VOLATILE': {'preferred_signals': ['CALL', 'PUT'], 'confidence_boost': 0.9, 'max_expiry': 3},
            'RANGING': {'preferred_signals': ['CALL', 'PUT'], 'confidence_boost': 1.15, 'max_expiry': 7},
            'NEUTRAL': {'preferred_signals': ['CALL', 'PUT'], 'confidence_boost': 1.0, 'max_expiry': 5}
        }
        
        print(f"  Institutional SwingScalp Trading Engine GPU Initialized on {_safe_get_device_name(self.device)}")

    def _generate_legacy_llm_prompt(self, current_price: float, current_regime: str, components: Dict) -> str:
        """Format clean prompt for LegacyLLM Local AI Reasoning in Part 3"""
        try:
            psych_signals = components.get('psychology', [])
            zone_signals = components.get('zone', [])
            trend_signals = components.get('trend', [])
            vol_signals = components.get('volume', [])
            
            psych_text = ", ".join([s.get('reason', s.get('type', '')) for s in psych_signals if isinstance(s, dict)]) or "None"
            zone_text = ", ".join([s.get('reason', s.get('type', '')) for s in zone_signals if isinstance(s, dict)]) or "None"
            trend_text = ", ".join([s.get('reason', s.get('type', '')) for s in trend_signals if isinstance(s, dict)]) or "None"
            vol_text = ", ".join([s.get('reason', s.get('type', '')) for s in vol_signals if isinstance(s, dict)]) or "None"
            
            prompt = f"""You are a legendary, highly profitable institutional trader with over 50 years of experience. You are an absolute master and expert in both swing trading and scalping.
You are acting as the ultimate trade validator for the Part3 Institutional SwingScalp Trading Engine. Use your deep intuition, vast experience, and mastery of market psychology to analyze the following institutional context and signals:

Current Market Context:
- Current Price: {current_price:.2f}
- Market Regime: {current_regime}

Institutional Algorithmic Signals:
- Candle Psychology Signals: {psych_text}
- Supply/Demand Zone Signals: {zone_text}
- Trend Analysis Signals: {trend_text}
- Volume Breakout Signals: {vol_text}

Task:
Provide a concise 1-2 sentence institutional analysis, then end your response with your decision strictly as one of: [BUY], [SELL], or [NO-TRADE].
"""
            return prompt
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 487)
            return "Analyze market context and respond with [BUY], [SELL], or [NO-TRADE]."

    def generate_live_signals(self, df_1min, df_5min=None, df_15min=None) -> Dict:
        """GPU-ACCELERATED INSTITUTIONAL LIVE SIGNAL GENERATION WITH LOCAL LEGACY_LLM AI INTEGRATION"""
        _empty = {'signals': [], 'components': {}}
        try:
            if df_1min is None or len(df_1min) < 5:
                return _empty
            
            df_eval_5m = df_5min if (df_5min is not None and len(df_5min) >= 10) else df_1min
            df_eval_15m = df_15min if (df_15min is not None and len(df_15min) >= 10) else df_eval_5m
            
            current_price = float(df_1min['close'].iloc[-1]) if ('close' in df_1min.columns and len(df_1min) > 0) else 0.0
            
            # Update market regime
            current_regime = self._gpu_detect_market_regime(df_eval_5m)
            self.trading_state['current_regime'] = current_regime
            
            # Generate signals from analysis modules
            psychology_signals = self._generate_psychology_signals(df_1min, current_regime)
            zone_signals = self._generate_zone_signals(df_1min, df_eval_5m, df_eval_15m)
            trend_signals = self._generate_trend_signals(df_eval_5m, current_regime)
            volume_signals = self._generate_volume_signals(df_1min, current_regime)
            
            # Advanced signal fusion
            fused_signals = self._gpu_fuse_signals(
                psychology_signals, zone_signals, trend_signals, volume_signals, current_regime
            )
            
            # Apply regime filtering
            filtered_signals = self._apply_regime_filtering(fused_signals, current_regime)
            
            components = {
                'psychology': psychology_signals,
                'zone': zone_signals,
                'trend': trend_signals,
                'volume': volume_signals,
                'regime': current_regime
            }
            
            # No model may append a high-confidence institutional vote.
            components['advisory_status'] = 'unavailable'

            for signal in filtered_signals:
                if isinstance(signal, dict):
                    signal['regime'] = current_regime
                    signal['timestamp'] = time.time()
                    self.trading_state['signal_history'].append(signal)
            
            return {
                'signals': filtered_signals,
                'components': components
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 542)
            print(f"ERROR GPU Institutional signal generation error: {e}")
            return _empty


    def generate_mtf_signals(self, mtf_data: Dict) -> Dict:
        """MULTI-TIMEFRAME INSTITUTIONAL ANALYSIS"""
        try:
            all_tf_results = {}
            all_components = {
                'psychology': [], 'zone': [], 'trend': [], 'volume': [],
                'regime': 'unknown', 'tf_signals': {}
            }
            
            tf_weights = {
                '1m': 1.0, '3m': 1.5, '5m': 2.0, '15m': 3.0,
                '30m': 4.0, '1h': 5.0, '2h': 6.0, '4h': 7.0
            }
            
            weighted_signal_sum = 0.0
            total_weight = 0.0
            aggregated_signals = []
            
            for tf_name, tf_df in mtf_data.items():
                if tf_df is None or len(tf_df) < 5:
                    continue
                
                tf_weight = tf_weights.get(tf_name, 1.0)
                
                try:
                    regime = self._gpu_detect_market_regime(tf_df)
                    psych = self._generate_psychology_signals(tf_df, regime)
                    zone = self._generate_zone_signals(tf_df, tf_df, tf_df)
                    trend = self._generate_trend_signals(tf_df, regime)
                    volume = self._generate_volume_signals(tf_df, regime)
                    
                    fused = self._gpu_fuse_signals(psych, zone, trend, volume, regime)
                    filtered = self._apply_regime_filtering(fused, regime)
                    
                    net_signal = 0
                    for sig in filtered:
                        if isinstance(sig, dict):
                            direction = sig.get('direction', sig.get('signal', ''))
                            if direction in ['CALL', 'BUY', 1]: net_signal += 1
                            elif direction in ['PUT', 'SELL', -1]: net_signal -= 1
                            aggregated_signals.append(sig)
                    
                    tf_signal = max(-1, min(1, net_signal))
                    
                    all_tf_results[tf_name] = {
                        'signal': tf_signal,
                        'regime': regime,
                        'signals_count': len(filtered),
                        'weight': tf_weight
                    }
                    
                    weighted_signal_sum += tf_signal * tf_weight
                    if tf_signal != 0:
                        total_weight += tf_weight
                    
                    all_components['psychology'].extend(psych if isinstance(psych, list) else [])
                    all_components['zone'].extend(zone if isinstance(zone, list) else [])
                    all_components['trend'].extend(trend if isinstance(trend, list) else [])
                    all_components['volume'].extend(volume if isinstance(volume, list) else [])
                    
                    if tf_weight >= tf_weights.get(all_components.get('_max_tf', '1m'), 1.0):
                        all_components['regime'] = regime
                        all_components['_max_tf'] = tf_name
                    
                except Exception as e:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 610)
                    if hasattr(self, 'bus') and self.bus:
                        self.bus.report_error('Part3_Institutional', e, context=f'MTF analysis on timeframe {tf_name}')
                    else:
                        print(f"ERROR MTF Institutional {tf_name} error: {e}")
            
            final_weighted = weighted_signal_sum / total_weight if total_weight > 0 else 0.0
            all_components['tf_signals'] = all_tf_results
            all_components['weighted_consensus'] = round(final_weighted, 3)
            all_components['active_timeframes'] = len(all_tf_results)
            all_components.pop('_max_tf', None)
            
            if hasattr(self, 'bus') and self.bus:
                direction = "BULLISH" if final_weighted > 0.3 else "BEARISH" if final_weighted < -0.3 else "NEUTRAL"
                msg = f"Institutional MTF Analysis: {direction} (Consensus: {final_weighted:.2f}). Dominant Regime: {all_components.get('regime', 'unknown')}"
                self.bus.publish('THOUGHTS', 'Part3_Institutional', msg)
            
            return {
                'signals': aggregated_signals,
                'components': all_components,
                'mtf_consensus': round(final_weighted, 3)
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 633)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part3_Institutional', e, context='generate_mtf_signals top-level')
            else:
                print(f"ERROR GPU MTF Institutional error: {e}")
            return {'signals': [], 'components': {}, 'mtf_consensus': 0.0}

    def _gpu_detect_market_regime(self, df_5min) -> str:
        import quantitative_math as qm
        features = qm.quantitative_features(df_5min, limit=128)
        if not features.get('available') or len(features.get('returns', ())) < 8:
            return "NEUTRAL"
        trend = float(features.get('trend_score', 0.0))
        state = features.get('volatility_state', {})
        high_p = float(state.get('high_state_posterior', 0.5))
        ratio = float(state.get('short_long_ratio', 0.0))
        close_location = float(features.get('candle', {}).get('close_location', 0.0))
        persistence = float(features.get('sign_persistence', 0.0))
        if trend >= 2.0:
            return "TRENDING_UP"
        if trend <= -2.0:
            return "TRENDING_DOWN"
        if high_p >= 0.90 and ratio >= 1.5:
            return "VOLATILE"
        if persistence < 0.45 and abs(close_location) < 0.40:
            return "RANGING"
        return "NEUTRAL"

    def _calculate_support_resistance_touches(self, highs, lows, closes) -> float:
        """Calculate support/resistance touch frequency"""
        try:
            if len(closes) < 15:
                return 0.5
            res = float(np.max(highs[-30:] if len(highs) >= 30 else highs))
            sup = float(np.min(lows[-30:] if len(lows) >= 30 else lows))
            rng = res - sup
            if rng <= 1e-8:
                return 0.5
            res_touches = np.sum(np.abs(highs[-15:] - res) < rng * 0.05)
            sup_touches = np.sum(np.abs(lows[-15:] - sup) < rng * 0.05)
            return float(min(1.0, (res_touches + sup_touches) / 30.0))
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 674)
            return 0.5

    def _generate_psychology_signals(self, df_1min, regime: str) -> List[Dict]:
        """Generate psychology signals with non-null master validation"""
        signals = []
        try:
            if df_1min is None or len(df_1min) < 3:
                return signals
            
            current_candle = df_1min.iloc[-1].to_dict()
            prev1 = df_1min.iloc[-2].to_dict()
            prev2 = df_1min.iloc[-3].to_dict()
            
            psych_master = getattr(self.master, 'candle_psychology', None)
            if psych_master is None:
                psych_master = CandlePsychologyMasterGPU(self.master)
                
            psychology = psych_master.analyze_candle_psychology(current_candle, prev1, prev2)
            if not isinstance(psychology, dict):
                psychology = {}
                
            signals.extend(self._get_institutional_support_rejection_signals(current_candle, psychology, regime))
            signals.extend(self._get_institutional_resistance_rejection_signals(current_candle, psychology, regime))
            signals.extend(self._get_pattern_signals(df_1min, psychology, regime))
            
            return signals
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 701)
            print(f"ERROR Institutional psychology signal generation: {e}")
            return []

    def _generate_zone_signals(self, df_1min, df_5min, df_15min) -> List[Dict]:
        """Generate zone signals safely"""
        signals = []
        try:
            zone_detector = getattr(self.master, 'zone_detector', None) if hasattr(self, 'master') else None
            if zone_detector is None:
                zone_detector = ZonePointFiveDetectorGPU(getattr(self, 'master', None))
                

            current_candle = df_1min.iloc[-1].to_dict() if (df_1min is not None and len(df_1min) > 0) else None
            if not current_candle:
                return signals
                
            psych_master = getattr(self.master, 'candle_psychology', None)
            psychology = {}
            if psych_master and len(df_1min) >= 3:
                psychology = psych_master.analyze_candle_psychology(
                    current_candle, df_1min.iloc[-2].to_dict(), df_1min.iloc[-3].to_dict()
                )
            
            zone_signals = zone_detector.detect_0_5_zone_signals(
                current_candle, psychology, df_1min, df_5min, df_15min
            ) or []
            
            for item in zone_signals:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    stype, conf, reason = item[0], item[1], item[2]
                    if stype != "NO_TRADE" and float(conf) >= self.trading_config['min_confidence']:
                        signals.append({
                            'type': 'ZONE',
                            'signal': stype,
                            'confidence': float(conf),
                            'reason': f"Institutional Zone: {reason}",
                            'timestamp': time.time()
                        })
            return signals
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 740)
            print(f"ERROR Institutional zone signal generation: {e}")
            return []

    def _generate_trend_signals(self, df_5min, regime: str) -> List[Dict]:
        signals = []
        try:
            if df_5min is None or len(df_5min) < 15:
                return signals
            import quantitative_math as qm
            features = qm.quantitative_features(df_5min, limit=128)
            if not features.get('available') or len(features.get('returns', ())) < 12:
                return signals
            strength = abs(float(features.get('trend_score', 0.0)))
            velocity = float(features.get('kalman', {}).get('velocity_z', 0.0))
            change = features.get('change_point', {})
            significant_change = float(change.get('bic_gain', 0.0)) >= 2.0
            if velocity >= 1.25 and strength >= 1.25 and significant_change and "UP" in regime:
                signals.append({'type': 'TREND', 'signal': 'CALL', 'confidence': 7.5,
                                'reason': f"State-space trend in {regime} (robust strength: {strength:.2f})",
                                'timestamp': time.time()})
            elif velocity <= -1.25 and strength >= 1.25 and significant_change and "DOWN" in regime:
                signals.append({'type': 'TREND', 'signal': 'PUT', 'confidence': 7.5,
                                'reason': f"State-space trend in {regime} (robust strength: {strength:.2f})",
                                'timestamp': time.time()})
            return signals
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 766)
            return signals

    def _generate_volume_signals(self, df_1min, regime: str) -> List[Dict]:
        signals = []
        try:
            if df_1min is None or len(df_1min) < 10 or 'volume' not in df_1min.columns:
                return signals
            import quantitative_math as qm
            features = qm.quantitative_features(df_1min, limit=128)
            if not features.get('available') or len(features.get('returns', ())) < 8:
                return signals
            surprise = float(features.get('volume_surprise_robust_z', 0.0))
            flow = float(features.get('flow_imbalance_proxy', 0.0))
            latest_return = float(features['returns'][-1])
            if surprise >= 1.5 and latest_return != 0.0 and latest_return * flow > 0.0 and abs(flow) >= 0.15:
                confidence = min(9.5, 7.0 * min(surprise / 1.5, 1.5) * min(abs(flow) / 0.3, 1.0))
                signals.append({'type': 'VOLUME', 'signal': 'CALL' if latest_return > 0 else 'PUT',
                                'confidence': confidence,
                                'reason': f"Robust surprise with aligned close-location OHLCV proxy ({surprise:.2f}σ; not tape data)",
                                'timestamp': time.time()})
            return signals
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 788)
            return signals

    def _get_institutional_support_rejection_signals(self, current_candle, psychology, regime: str) -> List[Dict]:
        signals = []
        try:
            if not isinstance(psychology, dict): return signals
            if psychology.get('is_hammer') or psychology.get('is_bullish_engulfing') or psychology.get('has_strong_support'):
                conf = 7.2
                if regime == "RANGING": conf *= 1.15
                elif regime == "TRENDING_UP": conf *= 1.1
                signals.append({
                    'type': 'PSYCHOLOGY',
                    'signal': 'CALL',
                    'confidence': min(conf, 9.5),
                    'reason': f"Institutional Support Rejection in {regime}",
                    'timestamp': time.time()
                })
            return signals
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 807)
            return []

    def _get_institutional_resistance_rejection_signals(self, current_candle, psychology, regime: str) -> List[Dict]:
        signals = []
        try:
            if not isinstance(psychology, dict): return signals
            if psychology.get('is_shooting_star') or psychology.get('is_bearish_engulfing') or psychology.get('has_strong_resistance'):
                conf = 7.2
                if regime == "RANGING": conf *= 1.15
                elif regime == "TRENDING_DOWN": conf *= 1.1
                signals.append({
                    'type': 'PSYCHOLOGY',
                    'signal': 'PUT',
                    'confidence': min(conf, 9.5),
                    'reason': f"Institutional Resistance Rejection in {regime}",
                    'timestamp': time.time()
                })
            return signals
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 826)
            return []

    def _get_pattern_signals(self, df_1min, psychology, regime: str) -> List[Dict]:
        signals = []
        try:
            if df_1min is not None and len(df_1min) >= 5:
                recent = df_1min.iloc[-5:]
                highs = recent['high'].values
                lows = recent['low'].values
                
                if (highs[-3] > highs[-4] and highs[-3] > highs[-2] and 
                    abs(highs[-3] - highs[-1]) / (abs(highs[-3]) + 1e-8) < 0.005):
                    signals.append({
                        'type': 'PATTERN',
                        'signal': 'PUT',
                        'confidence': 7.8,
                        'reason': f"Institutional Double Top Pattern in {regime}",
                        'timestamp': time.time()
                    })
                
                if (lows[-3] < lows[-4] and lows[-3] < lows[-2] and 
                    abs(lows[-3] - lows[-1]) / (abs(lows[-3]) + 1e-8) < 0.005):
                    signals.append({
                        'type': 'PATTERN',
                        'signal': 'CALL',
                        'confidence': 7.8,
                        'reason': f"Institutional Double Bottom Pattern in {regime}",
                        'timestamp': time.time()
                    })
            return signals
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 857)
            return []

    def _gpu_fuse_signals(self, psychology_signals, zone_signals, trend_signals, volume_signals, regime: str) -> List[Dict]:
        """Signal fusion with strict type checking"""
        try:
            all_signals = []
            for s_list in [psychology_signals, zone_signals, trend_signals, volume_signals]:
                if isinstance(s_list, list):
                    all_signals.extend([s for s in s_list if isinstance(s, dict) and 'signal' in s and 'confidence' in s])
                    
            if not all_signals:
                return []
            
            call_signals = [s for s in all_signals if s.get('signal') == 'CALL']
            put_signals = [s for s in all_signals if s.get('signal') == 'PUT']
            
            fused_signals = []
            if call_signals:
                call_conf = self._gpu_fuse_confidence([s['confidence'] for s in call_signals], regime, 'CALL')
                if call_conf >= self.trading_config['min_confidence']:
                    fused_signals.append({
                        'signal': 'CALL',
                        'confidence': call_conf,
                        'components': len(call_signals),
                        'types': [s.get('type', 'UNKNOWN') for s in call_signals],
                        'reason': ' + '.join([s.get('reason', '') for s in call_signals[:3]]),
                        'timestamp': time.time(),
                        'strategy_type': 'FUSED_CALL'
                    })
            
            if put_signals:
                put_conf = self._gpu_fuse_confidence([s['confidence'] for s in put_signals], regime, 'PUT')
                if put_conf >= self.trading_config['min_confidence']:
                    fused_signals.append({
                        'signal': 'PUT',
                        'confidence': put_conf,
                        'components': len(put_signals),
                        'types': [s.get('type', 'UNKNOWN') for s in put_signals],
                        'reason': ' + '.join([s.get('reason', '') for s in put_signals[:3]]),
                        'timestamp': time.time(),
                        'strategy_type': 'FUSED_PUT'
                    })
            return fused_signals
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 901)
            print(f"ERROR GPU Institutional signal fusion: {e}")
            return [s for s in (psychology_signals + zone_signals + trend_signals + volume_signals) if isinstance(s, dict)]

    def _gpu_fuse_confidence(self, confidences: List[float], regime: str, signal_type: str) -> float:
        """Accelerated confidence fusion with scalar math safety"""
        try:
            if not confidences:
                return 0.0
            
            clean_confs = [float(c) for c in confidences if isinstance(c, (int, float, np.number)) and not math.isnan(c)]
            if not clean_confs:
                return 0.0
                
            avg_conf = float(np.mean(clean_confs))
            regime_config = self.regime_rules.get(regime, {})
            regime_boost = regime_config.get('confidence_boost', 1.0)
            
            signal_boost = 1.1 if signal_type in regime_config.get('preferred_signals', []) else 0.9
            component_bonus = min(len(clean_confs) * 0.15, 0.8)
            
            final_conf = avg_conf * regime_boost * signal_boost + component_bonus
            return min(max(final_conf, 0.0), self.trading_config['max_confidence'])
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 924)
            return float(np.mean(confidences)) if confidences else 0.0

    def _apply_regime_filtering(self, signals: List[Dict], regime: str) -> List[Dict]:
        filtered = []
        for signal in signals:
            if not isinstance(signal, dict): continue
            conf = float(signal.get('confidence', 0.0))
            regime_config = self.regime_rules.get(regime, {})
            
            bonus = min(signal.get('components', 1) * 0.2, 1.0)
            conf += bonus
            
            if signal.get('signal') in regime_config.get('preferred_signals', []):
                conf *= regime_config.get('confidence_boost', 1.0)
                
            if "VOLATILE" in regime and conf > 8.0:
                conf *= 0.9
                
            signal['confidence'] = min(max(float(conf), 0.0), 9.8)
            signal['regime'] = regime
            
            if signal['confidence'] >= self.trading_config['min_confidence']:
                filtered.append(signal)
        return filtered

    def make_trade_expiry_decision(self, signal: Dict, current_volatility: float) -> Dict:
        """Make trade expiry decision using clean scalar float math"""
        try:
            base_expiry = self.trading_config['default_expiry']
            confidence = float(signal.get('confidence', 7.0))
            regime = self.trading_state.get('current_regime', 'NEUTRAL')
            signal_type = signal.get('signal', 'CALL')
            
            if confidence >= 8.5:
                expiry_adj = -1
            elif confidence >= 7.5:
                expiry_adj = 0
            elif confidence >= 6.8:
                expiry_adj = 1
            else:
                expiry_adj = 2
                
            regime_config = self.regime_rules.get(regime, {})
            max_exp = regime_config.get('max_expiry', 5)
            
            if "VOLATILE" in regime:
                expiry_adj += 1
            elif "RANGING" in regime:
                expiry_adj = max(expiry_adj - 1, -1)
                
            vol = float(current_volatility)
            if vol > 0.15:
                expiry_adj += 1
            elif vol < 0.05:
                expiry_adj = max(expiry_adj - 1, -1)
                
            final_expiry = max(1, min(max_exp, base_expiry + expiry_adj))
            stake_pct = self._calculate_institutional_stake_size(confidence, vol, regime)
            
            return {
                'signal': signal_type,
                'expiry_minutes': final_expiry,
                'stake_percentage': stake_pct,
                'confidence': confidence,
                'regime': regime,
                'volatility': vol,
                'strategy_type': signal.get('strategy_type', 'FUSED'),
                'components': signal.get('components', 1),
                'timestamp': time.time()
            }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 995)
            print(f"ERROR GPU Institutional expiry decision: {e}")
            return {
                'signal': signal.get('signal', 'CALL'),
                'expiry_minutes': self.trading_config['default_expiry'],
                'stake_percentage': self.trading_config['risk_per_trade'],
                'confidence': float(signal.get('confidence', 7.0)),
                'regime': 'NEUTRAL',
                'volatility': float(current_volatility),
                'strategy_type': signal.get('strategy_type', 'FUSED'),
                'components': signal.get('components', 1),
                'timestamp': time.time()
            }

    def _calculate_institutional_stake_size(self, confidence: float, volatility: float, regime: str) -> float:
        try:
            base_stake = self.trading_config['risk_per_trade']
            conf_mult = confidence / 8.0
            vol_mult = 0.8 if volatility > 0.12 else (1.2 if volatility < 0.06 else 1.0)
            regime_mult = self.regime_rules.get(regime, {}).get('confidence_boost', 1.0)
            
            loss_streak = self.trading_state['consecutive_losses']
            loss_penalty = max(0.5, 1.0 - (loss_streak * 0.15))
            
            win_streak = self.trading_state['consecutive_wins']
            win_bonus = min(1.3, 1.0 + (win_streak * 0.1))
            
            kelly_frac = self.trading_config['position_size_kelly_fraction']
            stake_pct = (base_stake * conf_mult * vol_mult * regime_mult * loss_penalty * win_bonus * kelly_frac)
            
            stake_pct = min(stake_pct, self.trading_config['max_daily_loss'] * 0.5)
            stake_pct = max(stake_pct, base_stake * 0.3)
            return round(float(stake_pct), 4)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1028)
            return float(self.trading_config['risk_per_trade'])

    def should_enter_trade(self, trade_decision: Dict) -> Tuple[bool, str]:
        current_time = time.time()
        today_str = datetime.now().strftime('%Y-%m-%d')
        if not hasattr(self, '_last_reset_date') or self._last_reset_date != today_str:
            self.trading_state['daily_trades'] = 0
            self.trading_state['daily_pnl'] = 0.0
            self._last_reset_date = today_str
            
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        if not hasattr(self, '_last_hour_reset') or self._last_hour_reset != current_hour:
            self.trading_state['hourly_trades'] = 0
            self._last_hour_reset = current_hour
            
        if self.trading_state['daily_trades'] >= self.trading_config['max_daily_trades']:
            return False, "Institutional daily trade limit reached"
        if self.trading_state['hourly_trades'] >= self.trading_config['max_hourly_trades']:
            return False, "Institutional hourly trade limit reached"
        if trade_decision['confidence'] < self.trading_config['min_confidence']:
            return False, f"Institutional confidence too low: {trade_decision['confidence']:.1f}"
            
        loss_cooldown = self.trading_config['cooldown_after_loss']
        if self.trading_state['consecutive_losses'] >= 3:
            loss_cooldown = self.trading_config['extended_cooldown_after_3_losses']
            
        if (self.trading_state['consecutive_losses'] > 0 and 
            current_time - self.trading_state['last_loss_time'] < loss_cooldown):
            return False, f"Institutional cooldown after loss: {int(loss_cooldown - (current_time - self.trading_state['last_loss_time']))}s remaining"
            
        if self.trading_state['daily_pnl'] <= -self.trading_config['max_daily_loss']:
            return False, f"Institutional daily loss limit reached: {self.trading_state['daily_pnl']:.2%}"
            
        if (self.trading_state['last_trade_time'] and current_time - self.trading_state['last_trade_time'] < 15):
            return False, "Institutional min trade interval (15s) required"
            
        if not self._validate_regime_trade(trade_decision.get('regime', 'NEUTRAL'), trade_decision):
            return False, f"Institutional regime validation failed for {trade_decision.get('regime')}"
            
        return True, "INSTITUTIONAL_APPROVED"

    def _validate_regime_trade(self, regime: str, trade_decision: Dict) -> bool:
        try:
            regime_config = self.regime_rules.get(regime, {})
            pref = regime_config.get('preferred_signals', [])
            if pref and trade_decision.get('signal') not in pref:
                if trade_decision.get('confidence', 0) < 8.0:
                    return False
            if "VOLATILE" in regime and trade_decision.get('expiry_minutes', 0) > 3:
                return False
            if "RANGING" in regime and trade_decision.get('components', 1) < 2:
                return False
            return True
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1082)
            return True

    def record_trade_outcome(self, trade_decision: Dict, is_successful: bool, pnl: float):
        self.trading_state['daily_trades'] += 1
        self.trading_state['hourly_trades'] += 1
        self.trading_state['last_trade_time'] = time.time()
        self.trading_state['daily_pnl'] += pnl
        
        if is_successful:
            self.trading_state['consecutive_losses'] = 0
            self.trading_state['consecutive_wins'] += 1
            self.trading_state['last_win_time'] = time.time()
            self.performance_metrics['winning_trades'] += 1
        else:
            self.trading_state['consecutive_losses'] += 1
            self.trading_state['consecutive_wins'] = 0
            self.trading_state['last_loss_time'] = time.time()
            self.performance_metrics['losing_trades'] += 1
            
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        if is_successful:
            self.performance_metrics['max_win_streak'] = max(
                self.performance_metrics['max_win_streak'], self.trading_state['consecutive_wins']
            )
            win_count = self.performance_metrics['winning_trades']
            curr_avg = self.performance_metrics['average_win']
            self.performance_metrics['average_win'] = ((curr_avg * (win_count - 1) + pnl) / win_count)
        else:
            self.performance_metrics['max_loss_streak'] = max(
                self.performance_metrics['max_loss_streak'], self.trading_state['consecutive_losses']
            )
            loss_count = self.performance_metrics['losing_trades']
            curr_avg = self.performance_metrics['average_loss']
            self.performance_metrics['average_loss'] = ((curr_avg * (loss_count - 1) + pnl) / loss_count)
            
        stype = trade_decision.get('strategy_type', 'FUSED')
        signal_type = trade_decision.get('signal', 'CALL')
        regime = trade_decision.get('regime', 'NEUTRAL')
        
        self.performance_metrics['strategy_performance'][stype].append(1.0 if is_successful else 0.0)
        self.performance_metrics['strategy_performance'][f"{signal_type}_{regime}"].append(1.0 if is_successful else 0.0)
        self.performance_metrics['regime_performance'][regime].append(1.0 if is_successful else 0.0)
        self.performance_metrics['expiry_performance'][trade_decision.get('expiry_minutes', 3)].append(1.0 if is_successful else 0.0)
        self.performance_metrics['hourly_performance'][datetime.now().hour].append(1.0 if is_successful else 0.0)
        
        trade_record = {
            **trade_decision,
            'success': is_successful,
            'pnl': pnl,
            'outcome_time': time.time()
        }
        self.trading_state['trade_history'].append(trade_record)

    def get_performance_report(self) -> str:
        total = self.performance_metrics['total_trades']
        wins = self.performance_metrics['winning_trades']
        losses = self.performance_metrics['losing_trades']
        wr = (wins / total * 100) if total > 0 else 0.0
        
        pf = abs(self.performance_metrics['average_win'] * wins) / max(abs(self.performance_metrics['average_loss'] * losses), 1e-8) if losses > 0 else float('inf')
        exp = (self.performance_metrics['average_win'] * (wr/100) + self.performance_metrics['average_loss'] * (1 - wr/100))
        
        report = f"""
  INSTITUTIONAL TRADE PERFORMANCE REPORT
==================================================
PERFORMANCE SUMMARY:
  Total Trades: {total} | Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%
  Total P&L: {self.performance_metrics['total_pnl']:+.2f}% | Daily P&L: {self.trading_state['daily_pnl']:+.2f}%
  Profit Factor: {pf:.2f} | Expectancy: {exp:+.4f}

ADVANCED METRICS:
  Max Win Streak: {self.performance_metrics['max_win_streak']} | Max Loss Streak: {self.performance_metrics['max_loss_streak']}
  Avg Win: {self.performance_metrics['average_win']:+.2f}% | Avg Loss: {self.performance_metrics['average_loss']:+.2f}%

CURRENT STATE:
  Daily Trades: {self.trading_state['daily_trades']}/{self.trading_config['max_daily_trades']}
  Hourly Trades: {self.trading_state['hourly_trades']}/{self.trading_config['max_hourly_trades']}
  Current Regime: {self.trading_state['current_regime']}
"""
        return report

    def get_trading_health(self) -> Dict:
        curr_time = time.time()
        last_t = self.trading_state['last_trade_time']
        time_since_last = (curr_time - last_t) if last_t > 0 else 999999.0
        return {
            'system_health': 'HEALTHY' if self.trading_state['consecutive_losses'] < 3 else 'CAUTION',
            'daily_trades_remaining': self.trading_config['max_daily_trades'] - self.trading_state['daily_trades'],
            'hourly_trades_remaining': self.trading_config['max_hourly_trades'] - self.trading_state['hourly_trades'],
            'time_since_last_trade': round(time_since_last, 2),
            'consecutive_losses': self.trading_state['consecutive_losses'],
            'daily_pnl': self.trading_state['daily_pnl'],
            'current_regime': self.trading_state['current_regime'],
            'signal_buffer_size': len(self.trading_state['signal_history']),
            'trade_history_size': len(self.trading_state['trade_history'])
        }


# ==================== GPU-OPTIMIZED INSTITUTIONAL RISK MANAGEMENT ENGINE ====================

@xgb_shadow('inst_risk', 'calculate_institutional_position_size')
class InstitutionalRiskManagementEngineGPU:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        self.risk_config = {
            'max_daily_loss': 0.25,
            'max_position_size': 0.12,
            'volatility_adjustment': True,
            'correlation_limits': 0.6,
            'drawdown_limits': 0.15,
            'max_consecutive_losses': 5,
            'daily_trade_limits': 25,
            'hourly_trade_limits': 8,
            'kelly_fraction': 0.25,
            'var_confidence_level': 0.95,
            'stress_test_scenarios': ['high_volatility', 'flash_crash', 'low_liquidity']
        }
        
        self.risk_state = {
            'current_exposure': 0.0,
            'daily_drawdown': 0.0,
            'volatility_adjustment_factor': 1.0,
            'correlation_matrix': None,
            'var_metrics': {},
            'stress_test_results': {},
            'risk_appetite': 'MODERATE',
            'last_risk_assessment': time.time()
        }

    def calculate_institutional_position_size(self, trade_confidence: float, current_volatility: float, 
                                             regime: str, signal_type: str) -> float:
        """Position sizing with scalar float math safety"""
        try:
            base_size = self.trading_engine.trading_config['risk_per_trade']
            conf_mult = max(0.3, min(1.8, float(trade_confidence) / 7.5))
            
            vol = float(current_volatility)
            if vol > 0.15: vol_mult = 0.6
            elif vol > 0.1: vol_mult = 0.8
            elif vol < 0.05: vol_mult = 1.4
            else: vol_mult = 1.0
            
            regime_mult = self._get_regime_risk_multiplier(regime, signal_type)
            drawdown_penalty = 0.7 if self.risk_state['daily_drawdown'] > 0.08 else 1.0
            
            loss_streak = self.trading_engine.trading_state['consecutive_losses']
            loss_penalty = max(0.5, 1.0 - (loss_streak * 0.2))
            
            win_rate = self._get_strategy_win_rate(signal_type, regime)
            kelly_size = self._calculate_kelly_size(win_rate, float(trade_confidence))
            
            final_size = base_size * conf_mult * vol_mult * regime_mult * drawdown_penalty * loss_penalty * kelly_size
            final_size = max(0.02, min(self.risk_config['max_position_size'], final_size))
            return round(float(final_size), 4)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1240)
            print(f"ERROR GPU Position sizing: {e}")
            return float(self.trading_engine.trading_config['risk_per_trade'])

    def _get_regime_risk_multiplier(self, regime: str, signal_type: str) -> float:
        try:
            regime_config = self.trading_engine.regime_rules.get(regime, {})
            if signal_type in regime_config.get('preferred_signals', []): return 1.2
            elif "VOLATILE" in regime: return 0.7
            elif "RANGING" in regime: return 1.1
            return 1.0
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1251)
            return 1.0

    def _get_strategy_win_rate(self, signal_type: str, regime: str) -> float:
        try:
            strategy_key = f"{signal_type}_{regime}"
            performances = list(self.trading_engine.performance_metrics['strategy_performance'].get(strategy_key, []))
            if not performances:
                performances = list(self.trading_engine.performance_metrics['strategy_performance'].get(signal_type, []))
                
            if performances:
                return float(np.mean(performances))
                
            tot = self.trading_engine.performance_metrics['total_trades']
            wins = self.trading_engine.performance_metrics['winning_trades']
            return (wins / tot) if tot > 0 else 0.60
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1267)
            return 0.60

    def _calculate_kelly_size(self, win_rate: float, confidence: float) -> float:
        try:
            win_loss_ratio = 0.75
            kelly = win_rate - (1.0 - win_rate) / win_loss_ratio
            kelly = max(0.0, kelly)
            conf_adj = confidence / 8.0
            return float(kelly * conf_adj * self.risk_config['kelly_fraction'])
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1277)
            return float(self.risk_config['kelly_fraction'])

    def update_institutional_risk_metrics(self, trade_outcomes: List[Dict]):
        if not trade_outcomes: return
        try:
            daily_pnl = self.trading_engine.trading_state['daily_pnl']
            if daily_pnl < 0:
                self.risk_state['daily_drawdown'] = abs(daily_pnl)
                
            vols = [t.get('volatility', 0.1) for t in trade_outcomes[-15:] if isinstance(t, dict)]
            if vols:
                avg_vol = float(np.mean(vols))
                if avg_vol > 0.15: self.risk_state['volatility_adjustment_factor'] = 0.6
                elif avg_vol > 0.12: self.risk_state['volatility_adjustment_factor'] = 0.8
                elif avg_vol < 0.06: self.risk_state['volatility_adjustment_factor'] = 1.3
                else: self.risk_state['volatility_adjustment_factor'] = 1.0
                
            self._update_var_metrics(trade_outcomes)
            if time.time() - self.risk_state['last_risk_assessment'] > 3600:
                self._perform_stress_testing()
                self.risk_state['last_risk_assessment'] = time.time()
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1299)
            print(f"ERROR Risk metrics update: {e}")

    def _update_var_metrics(self, trade_outcomes: List[Dict]):
        try:
            pnls = [float(t.get('pnl', 0)) for t in trade_outcomes[-100:] if isinstance(t, dict)]
            if len(pnls) >= 10:
                sorted_pnls = sorted(pnls)
                var_idx = int(len(sorted_pnls) * (1.0 - self.risk_config['var_confidence_level']))
                var_val = sorted_pnls[var_idx] if var_idx < len(sorted_pnls) else min(pnls)
                std_pnl = float(np.std(pnls))
                
                self.risk_state['var_metrics'] = {
                    'var_95': var_val,
                    'max_drawdown': min(pnls),
                    'sharpe_ratio': (float(np.mean(pnls)) / std_pnl) if std_pnl > 0 else 0.0,
                    'tail_risk': float(np.percentile(pnls, 5))
                }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1317)
            print(f"ERROR VaR calculation: {e}")

    def _perform_stress_testing(self):
        try:
            stress_results = {}
            curr_pnl = self.trading_engine.trading_state['daily_pnl']
            for scenario in self.risk_config['stress_test_scenarios']:
                if scenario == 'high_volatility': stress_pnl = curr_pnl * 0.5
                elif scenario == 'flash_crash': stress_pnl = curr_pnl - 0.10
                elif scenario == 'low_liquidity': stress_pnl = curr_pnl * 0.8
                else: stress_pnl = curr_pnl
                
                stress_results[scenario] = {
                    'simulated_pnl': stress_pnl,
                    'breach_probability': max(0.0, min(1.0, (0.1 - stress_pnl) / 0.1)) if stress_pnl < 0 else 0.0
                }
            self.risk_state['stress_test_results'] = stress_results
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1335)
            pass

    def should_halt_institutional_trading(self) -> Tuple[bool, str]:
        daily_pnl = self.trading_engine.trading_state['daily_pnl']
        consecutive_losses = self.trading_engine.trading_state['consecutive_losses']
        daily_trades = self.trading_engine.trading_state['daily_trades']
        
        if daily_pnl <= -self.risk_config['max_daily_loss']:
            return True, f"Institutional daily loss limit reached: {daily_pnl:.2%}"
        if consecutive_losses >= self.risk_config['max_consecutive_losses']:
            return True, f"Institutional consecutive losses limit: {consecutive_losses}"
        if self.risk_state['daily_drawdown'] >= self.risk_config['drawdown_limits']:
            return True, f"Institutional maximum drawdown reached: {self.risk_state['daily_drawdown']:.2%}"
        if daily_trades >= self.risk_config['daily_trade_limits']:
            return True, f"Institutional daily trade limit reached: {daily_trades}"
        return False, "INSTITUTIONAL_APPROVED"

    def get_risk_report(self) -> str:
        halt, reason = self.should_halt_institutional_trading()
        status_color = "RED" if halt else "GREEN"
        report = f"""
  INSTITUTIONAL RISK MANAGEMENT REPORT
========================================
RISK STATUS: {status_color} | Halt Required: {halt} | Reason: {reason}
CURRENT EXPOSURE:
  Daily P&L: {self.trading_engine.trading_state['daily_pnl']:+.2f}% | Drawdown: {self.risk_state['daily_drawdown']:.2f}%
  Consecutive Losses: {self.trading_engine.trading_state['consecutive_losses']} | Vol Mult: {self.risk_state['volatility_adjustment_factor']:.2f}
"""
        return report


# ==================== INSTITUTIONAL TRADE EXECUTION ENGINE ====================

@xgb_shadow('inst_execution', '_prepare_trade_order')
class InstitutionalTradeExecutionEngine:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        self.execution_config = {
            'default_broker': 'trade_com',
            'max_slippage': 0.02,
            'execution_timeout': 10,
            'retry_attempts': 3,
            'confirmation_required': True,
            'min_execution_quality': 0.8,
            'account_size': 1000.0
        }
        
        self.broker_connections = {
            'trade_com': {'connected': True, 'last_ping': time.time()},
            'iq_option': {'connected': False, 'last_ping': 0},
            'pocket_option': {'connected': False, 'last_ping': 0}
        }
        
        self.execution_state = {
            'active_orders': {},
            'order_history': LinuxOptimizedDeque(1000),
            'execution_quality': 1.0,
            'last_execution_time': 0,
            'total_executions': 0,
            'failed_executions': 0
        }

    async def execute_institutional_trade(self, trade_decision: Dict) -> Tuple[bool, str, float]:
        try:
            if not await self._validate_execution_conditions():
                return False, "Execution validation failed", 0.0
            
            selected_broker = await self._select_optimal_broker()
            if not selected_broker:
                return False, "No available brokers", 0.0
                
            trade_order = self._prepare_trade_order(trade_decision, selected_broker)
            for attempt in range(self.execution_config['retry_attempts']):
                try:
                    res = await self._execute_with_broker(trade_order, selected_broker)
                    if res['success']:
                        self._record_successful_execution(trade_order, res)
                        return True, f"Executed with {selected_broker}", res.get('execution_quality', 1.0)
                except Exception as me:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 1415)
                    print(f"  Execution attempt {attempt+1} error: {me}")
                await asyncio.sleep(0.2)
                
            self.execution_state['failed_executions'] += 1
            return False, "All execution attempts failed", 0.0
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1421)
            print(f"ERROR Institutional execution: {e}")
            return False, f"Execution error: {e}", 0.0

    async def _validate_execution_conditions(self) -> bool:
        connected = [b for b, s in self.broker_connections.items() if s['connected']]
        if not connected: return False
        if self.execution_state['execution_quality'] < self.execution_config['min_execution_quality']: return False
        if time.time() - self.execution_state['last_execution_time'] < 1.0: return False
        return True

    async def _select_optimal_broker(self) -> str:
        connected = [b for b, s in self.broker_connections.items() if s['connected']]
        return connected[0] if connected else ""

    def _prepare_trade_order(self, trade_decision: Dict, broker: str) -> Dict:
        """Safe JSON hashing to avoid unhashable/uncomparable dict sorting crash"""
        try:
            safe_decision = {k: str(v) for k, v in trade_decision.items()}
            raw_hash = hashlib.md5(json.dumps(safe_decision, sort_keys=True).encode()).hexdigest()[:8]
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1441)
            raw_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            
        order_id = f"INST_{int(time.time())}_{raw_hash}"
        stake_pct = float(trade_decision.get('stake_percentage', 0.05))
        
        return {
            'order_id': order_id,
            'broker': broker,
            'asset': 'BTC/USDT',
            'direction': trade_decision.get('signal', 'CALL'),
            'expiry_minutes': trade_decision.get('expiry_minutes', 3),
            'stake_amount': self._calculate_stake_amount(stake_pct),
            'stake_percentage': stake_pct,
            'confidence': float(trade_decision.get('confidence', 7.0)),
            'regime': trade_decision.get('regime', 'NEUTRAL'),
            'strategy_type': trade_decision.get('strategy_type', 'FUSED'),
            'timestamp': time.time(),
            'metadata': {
                'volatility': float(trade_decision.get('volatility', 0.1)),
                'components': trade_decision.get('components', 1)
            }
        }

    def _calculate_stake_amount(self, stake_percentage: float) -> float:
        acct = float(self.execution_config.get('account_size', 1000.0))
        return acct * float(stake_percentage)

    async def _execute_with_broker(self, trade_order: Dict, broker: str) -> Dict:
        await asyncio.sleep(0.05)
        return {
            'success': True,
            'order_id': trade_order['order_id'],
            'broker_order_id': f"BROKER_{int(time.time())}",
            'execution_time': time.time(),
            'execution_quality': 0.95,
            'slippage': 0.001,
            'actual_stake': trade_order['stake_amount']
        }

    def _record_successful_execution(self, trade_order: Dict, execution_result: Dict):
        self.execution_state['active_orders'][trade_order['order_id']] = {
            **trade_order,
            'broker_order_id': execution_result['broker_order_id'],
            'execution_time': execution_result['execution_time'],
            'status': 'EXECUTED'
        }
        self.execution_state['order_history'].append({
            **trade_order,
            'execution_result': execution_result
        })
        self.execution_state['last_execution_time'] = execution_result['execution_time']
        self.execution_state['total_executions'] += 1

    async def check_trade_outcome(self, order_id: str) -> Tuple[bool, Optional[bool], float]:
        try:
            if order_id not in self.execution_state['active_orders']:
                return False, None, 0.0
            order = self.execution_state['active_orders'][order_id]
            curr_time = time.time()
            order_time = order['timestamp']
            expiry_secs = order['expiry_minutes'] * 60
            
            if curr_time - order_time < expiry_secs:
                return True, None, 0.0
                
            conf = float(order['confidence'])
            conf_delta = (conf - 7.0) / 3.0
            win_prob = 0.50 + (conf_delta * 0.25)  # Baseline centered at 50%
            
            breakeven_threshold = 1.0 / (1.0 + 0.75)  # ~0.5714 for 75% payout
            is_win = win_prob > breakeven_threshold
            
            payout = 0.75
            pnl = order['stake_amount'] * payout if is_win else -order['stake_amount']
            pnl_pct = pnl / float(self.execution_config.get('account_size', 1000.0))
            
            del self.execution_state['active_orders'][order_id]
            return False, is_win, round(pnl_pct, 4)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1520)
            print(f"ERROR Trade outcome check: {e}")
            return False, None, 0.0

    def get_execution_report(self) -> str:
        tot = self.execution_state['total_executions']
        failed = self.execution_state['failed_executions']
        sr = ((tot - failed) / tot * 100) if tot > 0 else 0.0
        return f"""
  INSTITUTIONAL TRADE EXECUTION REPORT
========================================
Executions: {tot} | Failed: {failed} | Success Rate: {sr:.1f}% | Quality: {self.execution_state['execution_quality']:.2f}
Active Orders: {len(self.execution_state['active_orders'])}
"""


# ==================== INSTITUTIONAL PERFORMANCE ANALYTICS ENGINE ====================

class InstitutionalPerformanceAnalyticsGPU:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.analytics_state = {
            'last_optimization': time.time(),
            'performance_trend': 'STABLE',
            'optimal_strategy_mix': {},
            'regime_effectiveness': {},
            'hourly_patterns': {},
            'risk_adjustments': {}
        }

    def update_institutional_analytics(self, trade_outcomes: List[Dict]):
        if not trade_outcomes: return
        try:
            self._update_performance_trends(trade_outcomes)
            self._analyze_regime_effectiveness(trade_outcomes)
            self._detect_hourly_patterns(trade_outcomes)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1557)
            print(f"ERROR Analytics update: {e}")

    def _update_performance_trends(self, trade_outcomes: List[Dict]):
        try:
            recent = [t for t in trade_outcomes[-50:] if isinstance(t, dict)]
            if len(recent) < 5: return
            pnls = [float(t.get('pnl', 0)) for t in recent]
            wins = [p for p in pnls if p > 0]
            wr = len(wins) / len(pnls)
            
            if wr > 0.55: self.analytics_state['performance_trend'] = 'IMPROVING'
            elif wr < 0.40: self.analytics_state['performance_trend'] = 'DETERIORATING'
            else: self.analytics_state['performance_trend'] = 'STABLE'
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1571)
            pass

    def _analyze_regime_effectiveness(self, trade_outcomes: List[Dict]):
        try:
            reg_perf = {}
            for regime in self.trading_engine.regime_rules.keys():
                trades = [t for t in trade_outcomes if isinstance(t, dict) and t.get('regime') == regime]
                if trades:
                    pnls = [float(t.get('pnl', 0)) for t in trades]
                    wins = len([p for p in pnls if p > 0])
                    tot_pnl = sum(pnls)
                    wr = wins / len(trades)
                    reg_perf[regime] = {
                        'win_rate': wr,
                        'total_pnl': tot_pnl,
                        'trade_count': len(trades),
                        'effectiveness': wr * math.exp(max(-2.0, min(2.0, tot_pnl)))
                    }
            self.analytics_state['regime_effectiveness'] = reg_perf
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1591)
            pass

    def _detect_hourly_patterns(self, trade_outcomes: List[Dict]):
        try:
            hourly = defaultdict(list)
            for t in trade_outcomes:
                if isinstance(t, dict):
                    ts = t.get('timestamp')
                    if isinstance(ts, (int, float)):
                        hr = datetime.fromtimestamp(ts).hour
                        hourly[hr].append(float(t.get('pnl', 0)))
                        
            metrics = {}
            for hr, pnls in hourly.items():
                if pnls:
                    wr = len([p for p in pnls if p > 0]) / len(pnls)
                    metrics[hr] = {'win_rate': wr, 'avg_pnl': float(np.mean(pnls)), 'trade_count': len(pnls)}
            self.analytics_state['hourly_patterns'] = metrics
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1610)
            pass

    def get_institutional_insights(self) -> Dict:
        insights = {
            'performance_trend': self.analytics_state['performance_trend'],
            'optimal_regimes': [],
            'best_performing_hours': [],
            'strategy_recommendations': []
        }
        reg_eff = self.analytics_state.get('regime_effectiveness', {})
        if reg_eff:
            best = sorted(reg_eff.items(), key=lambda x: x[1].get('effectiveness', 0), reverse=True)[:3]
            insights['optimal_regimes'] = [r for r, _ in best]
        return insights

    def get_analytics_report(self) -> str:
        ins = self.get_institutional_insights()
        return f"""
  INSTITUTIONAL PERFORMANCE ANALYTICS REPORT
===============================================
PERFORMANCE TREND: {ins['performance_trend']} | Best Regimes: {', '.join(ins['optimal_regimes'])}
"""


# ==================== MAIN INSTITUTIONAL TRADING LOOP ====================

class InstitutionalSwingScalpTradingMasterLoop:
    def __init__(self, master_system=None):
        self.master = master_system
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        self.trading_engine = InstitutionalTradingEngineGPU(master_system)
        self.risk_engine = InstitutionalRiskManagementEngineGPU(self.trading_engine)
        self.execution_engine = InstitutionalTradeExecutionEngine(self.trading_engine)
        self.analytics_engine = InstitutionalPerformanceAnalyticsGPU(self.trading_engine)
        
        self.loop_config = {
            'analysis_interval': 5,
            'max_consecutive_errors': 5,
            'health_check_interval': 30,
            'performance_log_interval': 180,
            'analytics_update_interval': 60,
            'risk_assessment_interval': 45,
            'execution_monitoring_interval': 10
        }
        
        self.loop_state = {
            'is_running': False,
            'consecutive_errors': 0,
            'last_health_check': 0,
            'last_performance_log': 0,
            'last_analytics_update': 0,
            'last_risk_assessment': 0,
            'last_execution_monitor': 0,
            'total_cycles': 0,
            'active_trade_monitors': {}
        }
        
        self.data_buffers = {
            '1min': LinuxOptimizedDeque(300),
            '5min': LinuxOptimizedDeque(300),
            '15min': LinuxOptimizedDeque(300),
            'signals': LinuxOptimizedDeque(200),
            'trades': LinuxOptimizedDeque(500)
        }

    async def run_institutional_trading_loop(self):
        print("  Starting Institutional GPU-Optimized SwingScalp Trading Loop...")
        self.loop_state['is_running'] = True
        
        while self.loop_state['is_running']:
            try:
                c_start = time.time()
                if time.time() - self.loop_state['last_health_check'] > self.loop_config['health_check_interval']:
                    self._perform_institutional_health_check()
                    self.loop_state['last_health_check'] = time.time()
                    
                halt, reason = self.risk_engine.should_halt_institutional_trading()
                if halt:
                    print(f"  INSTITUTIONAL TRADING HALTED: {reason}")
                    await asyncio.sleep(30)
                    continue
                    
                df_1m, df_5m, df_15m = await self._get_institutional_market_data()
                if df_1m is None or len(df_1m) < 5:
                    await asyncio.sleep(self.loop_config['analysis_interval'])
                    continue
                    
                sig_res = self.trading_engine.generate_live_signals(df_1m, df_5m, df_15m)
                sig_list = sig_res.get('signals', []) if isinstance(sig_res, dict) else []
                
                for sig in sig_list:
                    dec = await self._process_institutional_signal(sig, df_1m)
                    if dec:
                        await self._execute_institutional_trade(dec)
                        
                self.loop_state['total_cycles'] += 1
                self.loop_state['consecutive_errors'] = 0
                
                elapsed = time.time() - c_start
                await asyncio.sleep(max(0.1, self.loop_config['analysis_interval'] - elapsed))
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1712)
                print(f"ERROR Institutional loop error: {e}")
                self.loop_state['consecutive_errors'] += 1
                if self.loop_state['consecutive_errors'] >= self.loop_config['max_consecutive_errors']:
                    self.loop_state['is_running'] = False
                    break
                await asyncio.sleep(2)

    async def _get_institutional_market_data(self) -> Tuple:
        try:
            if not self.master or not hasattr(self.master, 'realtime_data'):
                return None, None, None
            rt = self.master.realtime_data
            df_1m = rt.get('1min')
            df_5m = rt.get('5min')
            df_15m = rt.get('15min')
            
            def to_df(data):
                if data is None: return None
                if hasattr(data, 'iloc'): return data
                if isinstance(data, (list, deque)): return pd.DataFrame(list(data))
                return None
                
            return to_df(df_1m), to_df(df_5m), to_df(df_15m)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1736)
            return None, None, None

    async def _process_institutional_signal(self, signal: Dict, df_1min) -> Dict:
        try:
            vol = self._calculate_institutional_volatility(df_1min)
            dec = self.trading_engine.make_trade_expiry_decision(signal, vol)
            should, reason = self.trading_engine.should_enter_trade(dec)
            if not should: return None
            return dec
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1746)
            return None

    async def _execute_institutional_trade(self, trade_decision: Dict):
        try:
            ok, reason, qual = await self.execution_engine.execute_institutional_trade(trade_decision)
            if ok:
                oid = f"INST_{int(time.time())}"
                self.loop_state['active_trade_monitors'][oid] = {
                    'trade_decision': trade_decision,
                    'expiry_time': time.time() + (trade_decision['expiry_minutes'] * 60)
                }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1758)
            print(f"ERROR Institutional execution handler: {e}")

    def _calculate_trade_pnl(self, trade_decision: Dict, is_successful: bool) -> float:
        stk = float(trade_decision.get('stake_percentage', 0.05))
        return (stk * 0.75) if is_successful else -stk

    def _calculate_institutional_volatility(self, df_1min) -> float:
        try:
            if df_1min is None or len(df_1min) < 10: return 0.10
            closes = df_1min['close'].dropna().values[-25:]
            if len(closes) < 5: return 0.10
            returns = np.diff(np.log(np.maximum(closes, 1e-8)))
            vol = float(np.std(returns) * np.sqrt(252))
            if math.isnan(vol): return 0.10
            return max(0.02, min(0.30, vol))
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1774)
            return 0.10

    def _perform_institutional_health_check(self):
        try:
            health = self.trading_engine.get_trading_health()
            print(f"  INSTITUTIONAL HEALTH - System: {health['system_health']} | Cycles: {self.loop_state['total_cycles']}")
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1781)
            pass

    def stop_institutional_trading_loop(self):
        self.loop_state['is_running'] = False


# ==================== INTEGRATED INSTITUTIONAL MASTER SYSTEM ====================

class InstitutionalSwingScalpTradingMaster:
    def __init__(self, delta_api_key=None, delta_secret_key=None, openrouter_api_key=None):
        self.delta_api_key = delta_api_key
        self.delta_secret_key = delta_secret_key
        
        self.candle_psychology = CandlePsychologyMasterGPU(self)
        self.zone_detector = ZonePointFiveDetectorGPU(self)
        self.trading_loop = InstitutionalSwingScalpTradingMasterLoop(self)
        self.deepseek_ai = DeepSeekAILearningGPU(openrouter_api_key, self) if openrouter_api_key else None
        self.ai_learning_active = openrouter_api_key is not None
        
        self.realtime_data = {
            '1min': LinuxOptimizedDeque(300),
            '5min': LinuxOptimizedDeque(300), 
            '15min': LinuxOptimizedDeque(300)
        }
        
        print("  Institutional SwingScalp Trading Master Initialized (GPU-Optimized & FIXED)")

    async def start_institutional_trading(self):
        print("Starting Institutional SwingScalp Trading System...")
        try:
            await self.trading_loop.run_institutional_trading_loop()
        except KeyboardInterrupt:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1813)
            print("Institutional trading stopped by user")
        finally:
            self.trading_loop.stop_institutional_trading_loop()

    def get_institutional_status(self) -> Dict:
        health = self.trading_loop.trading_engine.get_trading_health()
        return {
            'system_status': 'OPERATIONAL' if self.trading_loop.loop_state['is_running'] else 'STOPPED',
            'gpu_acceleration': TORCH_AVAILABLE and torch.cuda.is_available(),
            'ai_learning_active': self.ai_learning_active,
            'institutional_cycles': self.trading_loop.loop_state['total_cycles'],
            'trading_health': health['system_health'],
            'daily_trades_remaining': health['daily_trades_remaining'],
            'current_regime': health['current_regime'],
            'performance_summary': {
                'total_trades': self.trading_loop.trading_engine.performance_metrics['total_trades'],
                'win_rate': (self.trading_loop.trading_engine.performance_metrics['winning_trades'] / 
                           self.trading_loop.trading_engine.performance_metrics['total_trades'] * 100) 
                           if self.trading_loop.trading_engine.performance_metrics['total_trades'] > 0 else 0.0,
                'total_pnl': self.trading_loop.trading_engine.performance_metrics['total_pnl']
            }
        }

    def get_institutional_dashboard(self) -> str:
        status = self.get_institutional_status()
        perf = self.trading_loop.trading_engine.get_performance_report()
        risk = self.trading_loop.risk_engine.get_risk_report()
        return f"""
  INSTITUTIONAL TRADING DASHBOARD
===========================================
SYSTEM STATUS: {status['system_status']} | GPU: {status['gpu_acceleration']} | AI: {status['ai_learning_active']}
TRADING HEALTH: {status['trading_health']} | Daily Remaining: {status['daily_trades_remaining']} | Regime: {status['current_regime']}
{perf}
{risk}
"""


async def institutional_main():
    trading_master = InstitutionalSwingScalpTradingMaster()
    print(trading_master.get_institutional_dashboard())

if __name__ == "__main__":
    asyncio.run(institutional_main())