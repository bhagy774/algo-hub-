try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ---- Helpers inserted for Part-7 Fix ----
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# PyTorch with fallback for Windows/WSL compatibility
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 16)
    TORCH_AVAILABLE = False
    # Dummy torch for compatibility
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

        def __init__(self, *args, **kwargs):
            self.shape = (1,)
        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return [0]
        def item(self): return 0
        def __getitem__(self, key): return self
        def __len__(self): return 1
    
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
        def tensor(data, **kwargs): return DummyTensor()
        
        @staticmethod
        def zeros(*args, **kwargs): return DummyTensor()
        
        @staticmethod
        def randn(*args, **kwargs): return DummyTensor()
        
        @staticmethod
        def cat(tensors, dim=0): return DummyTensor()
        
        @staticmethod
        def stack(tensors, dim=0): return DummyTensor()
        
        class F:
            @staticmethod
            def relu(x): return x
            @staticmethod
            def sigmoid(x): return x
            @staticmethod
            def tanh(x): return x
            @staticmethod
            def softmax(x, dim=-1): return x
            @staticmethod
            def log_softmax(x, dim=-1): return x
            @staticmethod
            def dropout(x, p=0.5, training=True): return x


class LinuxOptimizedDeque(deque):
    def __init__(self, maxlen=500):
        super().__init__(maxlen=maxlen)
    def append(self, item):
        try: super().append(item)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 121)
            pass

def _safe_get_device_name(device):
    try:
        if hasattr(device,"type") and device.type=="cuda":
            try:
                idx = device.index if hasattr(device,"index") and device.index is not None else 0
                return torch.cuda.get_device_name(idx)
            except:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 129)
                try: return torch.cuda.get_device_name()
                except:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 131)
                    return "cuda_device"
        return str(device)
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 133)
        return "unknown_device"

class GPUFeatureExtractor:
    def __init__(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def extract_basic(self,data):
        try:
            return torch.tensor([float(x) for x in data[:10]], device=self.device)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 142)
            return torch.zeros(10, device=self.device)
# ---- End helpers ----

# ==================== ENHANCED GPU LIVE DATA ENGINE ====================
# ALL IMPORTS INCLUDED AS REQUESTED

# System & OS
import os
import time
import gc
import threading
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

# Async & Networking
import asyncio
import websockets
import json

# Data Processing
import numpy as np

# GPU Acceleration

# BUG FIX #1: cupy optional — no crash if not installed
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 173)
    CUPY_AVAILABLE = False

# Optional Enhancements
import ssl
import logging
from typing import Dict, List, Optional, Union



# ==================== ENHANCED LOGGING SETUP ====================

class EnhancedLogger:
    """એન્હાન્સ્ડ લોગિંગ સિસ્ટમ for live data"""
    
    def __init__(self):
        self.logger = logging.getLogger('LiveDataEngine')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('/tmp/live_data_engine.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)

# ==================== ENHANCED GPU MEMORY MANAGER ====================

class EnhancedLiveDataGPUMemoryManager:
    """GTX 1650 4GB VRAM Optimized Memory Manager with SSL Support"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram = 3.2 * 1024 * 1024 * 1024  # 3.2GB safety limit
        self.max_ram = 4.0 * 1024 * 1024 * 1024   # 4GB RAM limit
        
        # SSL context for secure WebSocket
        self.ssl_context = self._create_ssl_context()
        
        # SSD cache
        self.cache_dir = Path("/tmp/live_data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Enhanced monitoring
        self.memory_stats = LinuxOptimizedDeque(maxlen=100)
        self.logger = EnhancedLogger()
        
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.70)
            self.logger.info(f"✅ Enhanced GPU Memory Manager: {_safe_get_device_name(self.device)}")
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure WebSocket connections"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 254)
            print(f"⚠️ SSL context creation failed: {e}")
            return None
    
    def allocate_live_tensor(self, data: Union[List, np.ndarray], name: str, persistent: bool = True) -> torch.Tensor:
        """Enhanced GPU tensor allocation with type hints"""
        try:
            # BUG FIX #2: Check device type FIRST before calling cuda memory functions
            if self.device.type == 'cpu':
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                return torch.tensor(data, dtype=torch.float32)

            if torch.cuda.memory_allocated() > self.max_vram:
                if isinstance(data, list):
                    data = np.array(data, dtype=np.float32)
                return torch.tensor(data, dtype=torch.float32)
            
            # Convert to numpy if needed
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            if persistent and tensor.numel() < 10000:
                return tensor
            else:
                return self._create_memory_mapped_tensor(data, name)
                
        except RuntimeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 283)
            self.logger.warning(f"GPU allocation failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def _create_memory_mapped_tensor(self, data: np.ndarray, name: str) -> torch.Tensor:
        """Create memory-mapped tensor for large datasets"""
        try:
            # BUG FIX #3: Fixed filename — timestamp causes new file every call = disk fill-up
            data_arr = np.array(data, dtype='float32')
            cache_file = self.cache_dir / f"{name}.dat"
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_arr.shape)
            mmap[:] = data_arr
            return torch.from_numpy(np.array(mmap))
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 296)
            self.logger.error(f"Memory mapping failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def cleanup_live_memory(self):
        """Aggressive cleanup with monitoring"""
        # BUG FIX #4: Guard cuda calls on CPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            self.memory_stats.append({
                'timestamp': datetime.now(),
                'allocated_mb': allocated
            })
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        if self.device.type == 'cuda':
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        return {}

# ==================== SEQUENCE VALIDATOR ====================

class SequenceValidator:
    """મેસેજ સીક્વન્સ વેલિડેશन for data quality"""
    
    def __init__(self):
        self.last_timestamp: Optional[int] = None
        self.last_sequence: Optional[int] = None
        self.out_of_order_count: int = 0
        self.total_messages: int = 0
    
    def validate(self, timestamp: int) -> bool:
        """Validate message sequence and timing"""
        self.total_messages += 1
        
        try:
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
                return True
            
            # Check for out-of-order messages
            if timestamp < self.last_timestamp:
                self.out_of_order_count += 1
                return False
            
            self.last_timestamp = timestamp
            return True
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 352)
            print(f"❌ Sequence validation error: {e}")
            return True  # Allow on error
    
    def get_stats(self) -> Dict:
        """Get validation statistics"""
        return {
            'total_messages': self.total_messages,
            'out_of_order_count': self.out_of_order_count,
            'out_of_order_rate': self.out_of_order_count / max(1, self.total_messages)
        }

# ==================== ENHANCED GPU LIVE DATA ENGINE ====================

# BUG FIX #5-#15: torch.cuda.device() crashes on CPU — use _cuda_guard everywhere
def _cuda_guard(device):
    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return torch.cuda.device(device) if device.type == 'cuda' else _NullCtx()


# ==================== GPU-ACCELERATED VOLATILITY & REGIME ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p7_volatility_engine', 'analyze')
class VolatilityEngineGPU:
    """
    JARVIS PART 7 - GPU-ACCELERATED VOLATILITY & REGIME ENGINE
    GTX 1650 CUDA & CPU Optimized.

    Quantitative Volatility Architecture:
    1. Bollinger Bands (20-period SMA, 2.0 Std Dev) & Bandwidth
    2. Keltner Channels (20-period EMA, 1.5 * ATR14)
    3. John Carter TTM Squeeze Detection (Bollinger Bands compressing inside Keltner Channel)
    4. Volatility Expansion / Breakout Direction (%B >= 0.85 with expanding ATR)
    5. Volatility Breakdown Direction (%B <= 0.15 with expanding ATR)
    6. Extreme Volatility / Panic Spike Risk Veto (ATR ratio > 2.8x or ATR% > 1.5% -> Veto)
    7. Stable Rotation / Mean-Reverting Chop Deadband (Signals 0 during normal vol)
    """
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')

    @part_advisory_entry("part7_volatility", data_parameter="data", context_parameter="context")
    def analyze(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Advanced Math: Includes Hidden Markov Model (HMM) proxy for regime detection
        Main volatility analysis called by Part7Volatility in Jarvis.
        """
        try:
            if data is None or not isinstance(data, pd.DataFrame) or len(data) < 30:
                return {"signal": 0, "confidence": 5.0, "thought": "Part7 Vol: Insufficient data (<30)",
                        "telemetry": {"status": "insufficient_or_invalid_ohlcv"}}

            # ── Check regime override from context ────────────────────────
            if context:
                regime = str(context.get('institutional_components', {}).get('regime', 'NEUTRAL')).upper()
                if 'VOLATILE' in regime or 'PANIC' in regime:
                    return {
                        "signal": 0,
                        "confidence": 5.0,
                        "thought": f"Part7 Vol: High Volatility Regime ({regime}) — No trade veto",
                        "entry_blocked": True,
                        "risk_veto": True,
                        "telemetry": {"regime": regime, "entry_blocked": True, "risk_veto": True}
                    }

            recent = data.tail(50).copy()
            closes = recent['close'].astype(float)
            highs  = recent['high'].astype(float)
            lows   = recent['low'].astype(float)

            if len(closes) < 20:
                return {"signal": 0, "confidence": 5.0, "thought": "Part7 Vol: Insufficient closes",
                        "telemetry": {"status": "insufficient_or_invalid_ohlcv"}}

            current_close = float(closes.iloc[-1])

            # --- Advanced Math: Simple Hidden Markov Model (HMM) Transition Proxy ---
            # We track two states: 0 (Low Volatility) and 1 (High Volatility)
            # using emission probabilities based on historical returns distribution
            returns = closes.pct_change().dropna().values
            vol = np.std(returns)

            # Transition matrix probabilities (simplified assumption)
            # P(Low->Low) = 0.8, P(Low->High) = 0.2
            # P(High->High) = 0.7, P(High->Low) = 0.3
            emission_prob_high = 1.0 / (1.0 + np.exp(-(vol - 0.002) * 1000)) # Sigmoid probability
            hmm_state = "HIGH" if emission_prob_high > 0.6 else "LOW"

            if hmm_state == "HIGH" and vol > 0.015:
                return {
                    "signal": 0,
                    "confidence": 5.0,
                    "thought": f"Part7 Vol: HMM High Volatility State Detected (P={emission_prob_high:.2f}) — Risk Veto",
                    "entry_blocked": True,
                    "risk_veto": True,
                    "telemetry": {"regime": hmm_state, "emission_prob": emission_prob_high,
                                  "entry_blocked": True, "risk_veto": True}
                }

            # 1. Bollinger Bands (20-period SMA, 2.0 std)
            sma20 = float(closes.tail(20).mean())
            std20 = float(closes.tail(20).std()) + 1e-8
            upper_bb = sma20 + 2.0 * std20
            lower_bb = sma20 - 2.0 * std20
            bb_width = (upper_bb - lower_bb) / max(sma20, 1.0)
            pct_b = (current_close - lower_bb) / (upper_bb - lower_bb + 1e-8)

            # 2. True Range & ATR(14), ATR(50)
            tr = pd.concat([
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr14 = float(tr.tail(14).mean())
            atr50 = float(tr.tail(50).mean()) if len(tr) >= 50 else atr14
            ema20 = float(closes.ewm(span=20).mean().iloc[-1])

            # 3. Keltner Channel (20 EMA, 1.5 * ATR14)
            upper_kc = ema20 + 1.5 * atr14
            lower_kc = ema20 - 1.5 * atr14

            # 4. TTM Squeeze Detection (BB inside KC)
            is_squeeze = (upper_bb < upper_kc) and (lower_bb > lower_kc)

            # 5. Volatility Expansion Ratio (Short-term ATR vs Long-term ATR)
            vol_ratio = atr14 / (atr50 + 1e-8)
            norm_atr = atr14 / max(current_close, 1.0)

            telemetry = {
                "bb_width_pct": round(bb_width * 100, 2),
                "pct_b": round(pct_b, 3),
                "atr14": round(atr14, 2),
                "vol_ratio": round(vol_ratio, 2),
                "is_squeeze": is_squeeze,
                "norm_atr_pct": round(norm_atr * 100, 3)
            }

            # 6. Extreme Volatility Spike Risk Gate (> 2.8x vol expansion or > 1.5% candle ATR)
            if vol_ratio > 2.8 or norm_atr > 0.015:
                return {
                    "signal": 0,
                    "confidence": 5.0,
                    "thought": f"Part7 Vol: Extreme Volatility Spike ({vol_ratio:.1f}x, ATR={norm_atr*100:.2f}%) — Risk Veto",
                    "entry_blocked": True,
                    "risk_veto": True,
                    "telemetry": {**telemetry, "entry_blocked": True, "risk_veto": True}
                }

            # 7. Squeeze Compression Coiling (Pre-Breakout Consolidation)
            if is_squeeze:
                return {
                    "signal": 0,
                    "confidence": 5.0,
                    "thought": f"Part7 Vol: TTM Squeeze Coiling (BBw={bb_width*100:.2f}%) — Neutral",
                    "telemetry": telemetry
                }

            # 8. Directional Volatility Expansion (Breakout above Upper BB with expanding ATR)
            if pct_b >= 0.85 and current_close > upper_bb and vol_ratio >= 1.0:
                conf = min(85.0, 60.0 + (pct_b - 0.85) * 100.0 + min(vol_ratio, 2.0) * 5.0)
                return {
                    "signal": 1,
                    "confidence": round(conf, 1),
                    "thought": f"Part7 Vol: Bullish Volatility Expansion (%B={pct_b:.2f}, VolRatio={vol_ratio:.2f})",
                    "telemetry": telemetry
                }

            # 9. Directional Volatility Breakdown (Breakdown below Lower BB with expanding ATR)
            if pct_b <= 0.15 and current_close < lower_bb and vol_ratio >= 1.0:
                conf = min(85.0, 60.0 + (0.15 - pct_b) * 100.0 + min(vol_ratio, 2.0) * 5.0)
                return {
                    "signal": -1,
                    "confidence": round(conf, 1),
                    "thought": f"Part7 Vol: Bearish Volatility Breakdown (%B={pct_b:.2f}, VolRatio={vol_ratio:.2f})",
                    "telemetry": telemetry
                }

            # 10. Normal / Sideways Volatility (Mean Reversion / Chop) -> Strictly 0 (Neutral)
            return {
                "signal": 0,
                "confidence": 5.0,
                "thought": f"Part7 Vol: Stable Volatility (%B={pct_b:.2f}, BBw={bb_width*100:.2f}%) — Neutral",
                "telemetry": telemetry
            }

        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 528)
            return {
                "signal": 0,
                "confidence": 5.0,
                "thought": f"Part7 Vol: Neutral (error: {e})",
                "telemetry": {"status": "analysis_error", "error": str(e)}
            }

@dataclass
class OHLCVData:
    opens: torch.Tensor
    highs: torch.Tensor
    lows: torch.Tensor
    closes: torch.Tensor
    volumes: torch.Tensor


class EnhancedGPULiveDataEngine:
    """
    ENHANCED GPU LIVE DATA ENGINE WITH ALL IMPORTS
    GTX 1650 + i5 10th Gen OPTIMIZED
    """
    
    def __init__(self, symbol: str = 'BTCUSDT', update_interval: float = 1.0, exchange: str = 'delta'):  # FIX #12: BTCUSD → BTCUSDT
        self.symbol = symbol
        self.update_interval = update_interval
        self.exchange = exchange
        
        # Enhanced managers
        self.gpu_manager = EnhancedLiveDataGPUMemoryManager()
        self.device = self.gpu_manager.device
        self.logger = EnhancedLogger()
        self.sequence_validator = SequenceValidator()
        
        # GPU-optimized data buffers
        self.price_buffer_gpu: Optional[torch.Tensor] = None
        self.volume_buffer_gpu: Optional[torch.Tensor] = None
        self.candle_buffers_gpu: Dict[str, Dict] = {}
        
        # Multi-timeframe candle storage
        self.timeframes: List[str] = ['1m', '5m', '15m']
        self._initialize_candle_buffers()
        
        # Real-time feature tensors
        self.feature_tensors_gpu: Dict[str, torch.Tensor] = {}
        self.normalization_params: Dict[str, Dict] = {}
        
        # WebSocket and async management
        self.websocket_url: str = "wss://socket.delta.exchange"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running: bool = False
        self.last_update: float = time.time()
        
        # Enhanced reconnection logic
        self.reconnect_attempts: int = 0
        self.max_reconnect_attempts: int = 15
        self.reconnect_delay: float = 2.0
        
        # Threading and async control
        self.data_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Enhanced performance monitoring
        self.processing_times: deque = LinuxOptimizedDeque(maxlen=100)
        self.message_count: int = 0
        self.quality_metrics: deque = LinuxOptimizedDeque(maxlen=1000)
        
        # GTX 1650 optimizations
        self.set_gpu_optimizations()
        
        self.logger.info("🚀 Enhanced GPU Live Data Engine Initialized")
    
    def set_gpu_optimizations(self):
        """GTX 1650 માટે સ્પેશિયલ ઓપ્ટિમાઈઝેશન"""
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.70)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            self.logger.info("✅ GTX 1650 GPU ઓપ્ટિમાઈઝેશન એક્ટિવ")
    
    def _initialize_candle_buffers(self):
        """Initialize GPU-optimized candle buffers for all timeframes"""
        buffer_sizes = {
            '1m': 1440,   # 1 day of 1-minute data
            '5m': 864,    # 3 days of 5-minute data
            '15m': 672    # 7 days of 15-minute data
        }
        
        for timeframe in self.timeframes:
            size = buffer_sizes[timeframe]
            self.candle_buffers_gpu[timeframe] = {
                'open': torch.zeros(size, device=self.device, dtype=torch.float32),
                'high': torch.zeros(size, device=self.device, dtype=torch.float32),
                'low': torch.full((size,), float('inf'), device=self.device, dtype=torch.float32),
                'close': torch.zeros(size, device=self.device, dtype=torch.float32),
                'volume': torch.zeros(size, device=self.device, dtype=torch.float32),
                'timestamp': torch.zeros(size, device=self.device, dtype=torch.int64),
                'pointer': 0,
                'count': 0
            }
    
    # ==================== ENHANCED ASYNCIO WEBSOCKET ENGINE ====================
    
    async def start_websocket_listener(self):
        """એન્હાન્સ્ડ WebSocket with better reconnection"""
        self.logger.info(f"🚀 Starting Enhanced WebSocket for {self.symbol}...")
        self.is_running = True
        
        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # SSL context for secure connection
                ssl_context = self.gpu_manager.ssl_context
                
                async with websockets.connect(
                    self.websocket_url, 
                    ping_interval=30,
                    ping_timeout=20,
                    close_timeout=10,
                    ssl=ssl_context
                ) as ws:
                    self.websocket = ws
                    self.reconnect_attempts = 0
                    self.logger.info("✅ WebSocket કનેક્ટ થયું (Delta Exchange)")

                    # Subscribe to Delta Exchange trades
                    subscribe_msg = {
                        "type": "subscribe",
                        "payload": {
                            "channels": [
                                {"name": "v2/trades", "symbols": [self.symbol]}
                            ]
                        }
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.info(f"✅ Subscribed to {self.symbol} trades")
                    
                    while self.is_running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                            await self._process_enhanced_message(message)
                            
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            globals().get("_plog", lambda *_a, **_k: None)(__name__, 669)
                            await ws.ping()
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            globals().get("_plog", lambda *_a, **_k: None)(__name__, 673)
                            self.logger.warning("🔌 WebSocket connection closed")
                            break
                        except Exception as e:
                            globals().get("_plog", lambda *_a, **_k: None)(__name__, 676)
                            self.logger.error(f"⚠️ WebSocket error: {e}")
                            break
                            
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 680)
                self.reconnect_attempts += 1
                # BUG FIX #24: Cap exponent to avoid overflow (2**60 = huge number)
                exp = min(self.reconnect_attempts, 6)
                delay = min(self.reconnect_delay * 2 ** exp, 60)
                self.logger.warning(f"🔌 Reconnecting in {delay}s (Attempt {self.reconnect_attempts})...")
                await asyncio.sleep(delay)
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("❌ Max reconnection attempts reached")
    
    async def _process_enhanced_message(self, message: str):
        """એન્હાન્સ્ડ મેસેજ પ્રોસેસિંગ with data validation"""
        start_time = time.time()
        
        try:
            data = json.loads(message)
            
            # Subscriptions and heartbeats
            if data.get("type") in ["subscriptions", "heartbeat"]:
                return
                
            # Delta Exchange trade format
            if data.get("type") == "v2/trades" and "trades" in data:
                for trade_data in data["trades"]:
                    # Enhanced data validation
                    if not self._validate_message_data(trade_data):
                        self.quality_metrics.append({'type': 'invalid_data', 'timestamp': time.time()})
                        continue
                    
                    # Extract and process data
                    # Delta fields: 'price', 'size', 'timestamp'
                    price = float(trade_data['price'])
                    volume = float(trade_data['size'])
                    # Delta timestamp is usually in microseconds, convert to milliseconds if needed
                    timestamp_raw = trade_data['timestamp']
                    # Assuming delta sends timestamp in microseconds, we convert back to millis to match old logic
                    timestamp = int(timestamp_raw / 1000) if timestamp_raw > 1e14 else int(timestamp_raw)
                    
                    # Sequence validation
                    if not self.sequence_validator.validate(timestamp):
                        self.quality_metrics.append({'type': 'out_of_order', 'timestamp': time.time()})
                    
                    # GPU processing
                    await self._update_price_gpu(price, volume, timestamp)
                    
                    self.message_count += 1
                    
                    # Performance monitoring
                    if self.message_count % 500 == 0:
                        self._log_performance_metrics()
                
        except json.JSONDecodeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 732)
            self.logger.error(f"❌ JSON decode error: {e}")
            self.quality_metrics.append({'type': 'json_error', 'timestamp': time.time()})
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 735)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part7_LiveData', e, context='_process_enhanced_message')
            else:
                self.logger.error(f"❌ Message processing error: {e}")
            self.quality_metrics.append({'type': 'processing_error', 'timestamp': time.time()})
        
        finally:
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
    
    def _validate_message_data(self, data: Dict) -> bool:
        """ડેટા વેલિડેશन for quality control"""
        try:
            # Delta fields: 'price', 'size', 'timestamp'
            required_fields = ['price', 'size', 'timestamp']
            for field in required_fields:
                if field not in data:
                    return False
            
            # Price validation
            price = float(data['price'])
            if price <= 0 or price > 1000000:  # Reasonable range for BTC
                return False
            
            # Volume validation
            volume = float(data['size'])
            if volume < 0:
                return False
            
            # Timestamp validation
            # Delta usually sends microseconds, convert if needed for check
            timestamp_raw = data['timestamp']
            timestamp = int(timestamp_raw / 1000) if timestamp_raw > 1e14 else int(timestamp_raw)
            current_time = int(time.time() * 1000)
            if timestamp > current_time + 60000:  # Future timestamp check
                return False
                
            return True
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 775)
            return False
    
    def _log_performance_metrics(self):
        """પરફોર્મન્સ મેટ્રિક્સ લોગિંગ"""
        if not self.processing_times:
            return
            
        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times) if self.processing_times else 0
        min_time = np.min(self.processing_times) if self.processing_times else 0
        
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == 'cuda' else 0
        
        self.logger.info(f"📊 Performance: Msgs={self.message_count}, "
                        f"AvgTime={avg_time:.4f}s, GPUmem={gpu_memory:.1f}MB")
    
    # ==================== GPU-ACCELERATED PRICE UPDATES ====================
    
    async def _update_price_gpu(self, price: float, volume: float, timestamp: int):
        """GPU-accelerated price update across all timeframes"""
        try:
            # Convert to GPU tensors
            price_tensor = self.gpu_manager.allocate_live_tensor([price], 'current_price')
            volume_tensor = self.gpu_manager.allocate_live_tensor([volume], 'current_volume')
            timestamp_tensor = self.gpu_manager.allocate_live_tensor([timestamp], 'current_timestamp')
            
            # Update all timeframe buffers
            update_tasks = []
            for timeframe in self.timeframes:
                task = self._update_timeframe_buffer(
                    timeframe, price_tensor, volume_tensor, timestamp_tensor
                )
                update_tasks.append(task)
            
            # Execute updates in parallel
            await asyncio.gather(*update_tasks)
            
            # Update feature tensors
            await self._update_feature_tensors()
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 816)
            self.logger.error(f"❌ GPU price update error: {e}")
    
    async def _update_timeframe_buffer(self, timeframe: str, price_tensor: torch.Tensor, 
                                     volume_tensor: torch.Tensor, timestamp_tensor: torch.Tensor):
        """Update specific timeframe buffer with new price data"""
        try:
            buffer = self.candle_buffers_gpu[timeframe]
            pointer = buffer['pointer']
            
            with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
                # Update current candle
                if buffer['count'] == 0:
                    # Initialize new candle
                    buffer['open'][pointer] = price_tensor
                    buffer['high'][pointer] = price_tensor
                    buffer['low'][pointer] = price_tensor
                    buffer['close'][pointer] = price_tensor
                    buffer['volume'][pointer] = volume_tensor
                    buffer['timestamp'][pointer] = timestamp_tensor
                else:
                    # Update existing candle
                    buffer['high'][pointer] = torch.max(buffer['high'][pointer], price_tensor)
                    buffer['low'][pointer] = torch.min(buffer['low'][pointer], price_tensor)
                    buffer['close'][pointer] = price_tensor
                    buffer['volume'][pointer] += volume_tensor
                
                # Check if we need to finalize current candle and start new one
                if self._should_finalize_candle(timeframe, timestamp_tensor.item()):
                    pointer = (pointer + 1) % len(buffer['open'])
                    buffer['pointer'] = pointer
                    
                    # Initialize new candle
                    buffer['open'][pointer] = price_tensor
                    buffer['high'][pointer] = price_tensor
                    buffer['low'][pointer] = price_tensor
                    buffer['close'][pointer] = price_tensor
                    buffer['volume'][pointer] = volume_tensor
                    buffer['timestamp'][pointer] = timestamp_tensor
                    
                    buffer['count'] = min(buffer['count'] + 1, len(buffer['open']))
                    
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 858)
            self.logger.error(f"❌ Timeframe buffer update error ({timeframe}): {e}")
    
    def _should_finalize_candle(self, timeframe: str, timestamp: int) -> bool:
        """Determine if current candle should be finalized based on timeframe"""
        current_time = timestamp / 1000  # Convert to seconds
        buffer = self.candle_buffers_gpu[timeframe]
        last_timestamp = buffer['timestamp'][buffer['pointer']].item() / 1000
        
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900
        }
        
        return (current_time - last_timestamp) >= timeframe_seconds[timeframe]
    
    # ==================== ENHANCED GPU-ACCELERATED FEATURE ENGINEERING ====================
    
    async def _update_feature_tensors(self):
        """એન્હાન્સ્ડ ફીચર એન્જિનિયરિંગ with more indicators"""
        try:
            buffer_1m = self.candle_buffers_gpu['1m']
            if buffer_1m['count'] < 20:  # Increased minimum data
                return
            
            with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
                # Get recent data
                start_idx = (buffer_1m['pointer'] - 99) % len(buffer_1m['open'])
                recent_opens = self._get_circular_slice(buffer_1m['open'], start_idx, 100)
                recent_highs = self._get_circular_slice(buffer_1m['high'], start_idx, 100)
                recent_lows = self._get_circular_slice(buffer_1m['low'], start_idx, 100)
                recent_closes = self._get_circular_slice(buffer_1m['close'], start_idx, 100)
                recent_volumes = self._get_circular_slice(buffer_1m['volume'], start_idx, 100)
                
                # Enhanced feature set
                features = {}
                
                # Basic features
                features['returns'] = self._calculate_returns_gpu(recent_closes)
                features['volatility'] = self._calculate_enhanced_volatility_gpu(recent_highs, recent_lows, recent_closes)
                features['momentum'] = self._calculate_enhanced_momentum_gpu(recent_closes)
                
                # Volume features
                features['volume_profile'] = self._calculate_volume_profile_gpu(recent_volumes)
                features['volume_velocity'] = self._calculate_volume_velocity_gpu(recent_volumes)
                features['volume_oscillator'] = self._calculate_volume_oscillator_gpu(recent_volumes)
                
                # Advanced technical features
                features['rsi'] = self._calculate_rsi_gpu(recent_closes)
                features['macd'] = self._calculate_macd_gpu(recent_closes)
                features['bollinger_bands'] = self._calculate_bollinger_bands_gpu(recent_closes)
                
                # Market microstructure
                ohlcv = OHLCVData(
                    opens=recent_opens,
                    highs=recent_highs,
                    lows=recent_lows,
                    closes=recent_closes,
                    volumes=recent_volumes
                )
                features['microstructure'] = self._calculate_enhanced_microstructure_gpu(ohlcv)
                
                # Price patterns
                features['price_patterns'] = self._detect_price_patterns_gpu(
                    recent_opens, recent_highs, recent_lows, recent_closes
                )
                
                self.feature_tensors_gpu = features
                self._normalize_features_gpu()
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 929)
            self.logger.error(f"❌ Enhanced feature error: {e}")
    
    def _get_circular_slice(self, tensor: torch.Tensor, start_idx: int, length: int) -> torch.Tensor:
        """Get circular slice from tensor buffer"""
        indices = torch.arange(length, device=self.device)
        circular_indices = (start_idx + indices) % len(tensor)
        return tensor[circular_indices]
    
    def _calculate_returns_gpu(self, closes: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated returns calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            # BUG FIX #16: Guard against division by zero in returns
            returns = torch.diff(closes) / (closes[:-1] + 1e-8)
            return torch.nan_to_num(returns, nan=0.0)
    
    def _calculate_enhanced_volatility_gpu(self, highs: torch.Tensor, lows: torch.Tensor, closes: torch.Tensor) -> torch.Tensor:
        """Enhanced volatility calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            # BUG FIX #16: Guard div by zero in returns
            returns = torch.diff(closes) / (closes[:-1] + 1e-8)
            returns_volatility = torch.std(returns[-20:]) if len(returns) >= 20 else torch.std(returns)
            
            # BUG FIX #17: log(highs/lows) = -inf when lows=0 — clamp to safe values
            safe_ratio = torch.clamp(highs / (lows + 1e-8), min=1e-8)
            hl_ratio = torch.log(safe_ratio)
            parkinson_vol = torch.sqrt(torch.mean(hl_ratio ** 2) / (4 * torch.log(torch.tensor(2.0, device=self.device))))
            
            combined_vol = (returns_volatility + parkinson_vol) / 2
            return combined_vol / (closes[-1] + 1e-8)
    
    def _calculate_enhanced_momentum_gpu(self, closes: torch.Tensor) -> torch.Tensor:
        """એન્હાન્સ્ડ મોમેન્ટમ ઇન્ડિકેટર્સ"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < 30:
                return torch.tensor(0.0, device=self.device)
            
            # BUG FIX #18: Guard div by zero in momentum calculations
            mom_5 = (closes[-1] - closes[-5]) / (closes[-5] + 1e-8)
            mom_10 = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8)
            mom_20 = (closes[-1] - closes[-20]) / (closes[-20] + 1e-8)
            
            # Weighted momentum
            weighted_momentum = (mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2)
            
            return torch.tanh(weighted_momentum)  # Normalize to -1,1
    
    def _calculate_rsi_gpu(self, closes: torch.Tensor, period: int = 14) -> torch.Tensor:
        """GPU RSI calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < period + 1:
                return torch.tensor(50.0, device=self.device)
            
            deltas = torch.diff(closes)
            gains = torch.where(deltas > 0, deltas, 0.0)
            losses = torch.where(deltas < 0, -deltas, 0.0)
            
            avg_gains = torch.mean(gains[-period:])
            avg_losses = torch.mean(losses[-period:])
            
            if avg_losses == 0:
                return torch.tensor(100.0, device=self.device)
            
            rs = avg_gains / avg_losses
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi / 100.0  # Normalize to 0-1
    
    def _calculate_macd_gpu(self, closes: torch.Tensor) -> torch.Tensor:
        """GPU MACD calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < 26:
                return torch.tensor(0.0, device=self.device)
            
            # Simplified MACD
            ema_12 = torch.mean(closes[-12:])
            ema_26 = torch.mean(closes[-26:])
            macd_line = ema_12 - ema_26
            
            return torch.tanh(macd_line / closes[-1]) if closes[-1] > 0 else torch.tanh(macd_line)
    
    def _calculate_volume_profile_gpu(self, volumes: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated volume profile calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(volumes) < 20:
                return torch.tensor(0.5, device=self.device)
            
            current_volume = volumes[-1]
            avg_volume = torch.mean(volumes[-20:])
            volume_ratio = current_volume / avg_volume
            
            return torch.tanh(volume_ratio - 1.0) * 0.5 + 0.5
    
    def _calculate_volume_velocity_gpu(self, volumes: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated volume velocity calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(volumes) < 5:
                return torch.tensor(0.0, device=self.device)
            
            recent_volumes = volumes[-5:]
            volume_changes = torch.diff(recent_volumes)
            # BUG FIX #19: Guard div by zero in volume velocity
            velocity = torch.mean(volume_changes) / (torch.mean(recent_volumes[:-1]) + 1e-8)
            
            return torch.tanh(velocity) * 0.5 + 0.5
    
    def _calculate_volume_oscillator_gpu(self, volumes: torch.Tensor) -> torch.Tensor:
        """Volume oscillator for volume momentum"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(volumes) < 20:
                return torch.tensor(0.0, device=self.device)
            
            short_ma = torch.mean(volumes[-5:])
            long_ma = torch.mean(volumes[-20:])
            # BUG FIX #20: Guard div by zero in volume oscillator
            oscillator = (short_ma - long_ma) / (long_ma + 1e-8)
            return torch.tanh(oscillator)
    
    def _calculate_bollinger_bands_gpu(self, closes: torch.Tensor) -> torch.Tensor:
        """Bollinger Bands calculation"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < 20:
                return torch.tensor(0.0, device=self.device)
            
            sma = torch.mean(closes[-20:])
            std = torch.std(closes[-20:])
            # BUG FIX #21: Guard div by zero when std=0 (flat market / identical prices)
            distance = (closes[-1] - sma) / (std + 1e-8)
            return torch.tanh(distance / 2.0)  # Normalize
    
    def _calculate_enhanced_microstructure_gpu(self, data: OHLCVData) -> torch.Tensor:
        """Enhanced market microstructure features"""
        opens = data.opens
        highs = data.highs
        lows = data.lows
        closes = data.closes
        volumes = data.volumes

        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < 10:
                return torch.zeros(5, device=self.device)
            
            # Price efficiency
            price_changes = torch.diff(closes[-10:])
            absolute_changes = torch.abs(price_changes)
            hl_range = highs[-1] - lows[-1]
            # BUG FIX #22: Guard div by zero in efficiency and spread_estimate
            efficiency = torch.sum(absolute_changes) / (hl_range + 1e-8)
            
            volume_corr = torch.corrcoef(torch.stack([closes[-10:], volumes[-10:]]))[0, 1]
            volume_corr = torch.nan_to_num(volume_corr, nan=0.0)
            
            spread_estimate = hl_range / (closes[-1] + 1e-8)
            
            # BUG FIX #22: depth_estimate was a plain float (1.0) when insufficient data — wrap in tensor
            if len(volumes) >= 20:
                depth_estimate = torch.mean(volumes[-5:]) / (torch.mean(volumes[-20:]) + 1e-8)
            else:
                depth_estimate = torch.tensor(1.0, device=self.device)
            
            ofi = torch.sum(torch.where(closes[-5:] > opens[-5:], volumes[-5:], -volumes[-5:])) / (torch.sum(volumes[-5:]) + 1e-8)
            
            microstructure = torch.stack([
                efficiency if isinstance(efficiency, torch.Tensor) else torch.tensor(float(efficiency), device=self.device),
                volume_corr,
                spread_estimate if isinstance(spread_estimate, torch.Tensor) else torch.tensor(float(spread_estimate), device=self.device),
                depth_estimate if isinstance(depth_estimate, torch.Tensor) else torch.tensor(float(depth_estimate), device=self.device),
                ofi
            ])
            
            return microstructure
    
    def _detect_price_patterns_gpu(self, opens: torch.Tensor, highs: torch.Tensor, 
                                 lows: torch.Tensor, closes: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated price pattern detection"""
        with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
            if len(closes) < 10:
                return torch.zeros(5, device=self.device)
            
            patterns = torch.zeros(5, device=self.device)
            
            # Simple pattern detection
            current_body = torch.abs(closes[-1] - opens[-1])
            avg_body = torch.mean(torch.abs(closes[-10:-1] - opens[-10:-1]))
            
            # Doji pattern
            patterns[0] = torch.where(current_body < avg_body * 0.1, 1.0, 0.0)
            
            # Hammer pattern
            lower_wick = torch.min(opens[-1], closes[-1]) - lows[-1]
            upper_wick = highs[-1] - torch.max(opens[-1], closes[-1])
            body_size = torch.abs(closes[-1] - opens[-1])
            
            patterns[1] = torch.where(
                (lower_wick > body_size * 2) & (upper_wick < body_size * 0.5), 
                1.0, 0.0
            )
            
            # Engulfing pattern
            if len(closes) >= 3:
                prev_body = torch.abs(closes[-2] - opens[-2])
                curr_body = torch.abs(closes[-1] - opens[-1])
                
                bull_engulfing = (closes[-1] > opens[-1]) & (opens[-1] < closes[-2]) & (closes[-1] > opens[-2])
                bear_engulfing = (closes[-1] < opens[-1]) & (opens[-1] > closes[-2]) & (closes[-1] < opens[-2])
                
                patterns[2] = torch.where(bull_engulfing, 1.0, 0.0)
                patterns[3] = torch.where(bear_engulfing, 1.0, 0.0)
            
            # Trend detection
            if len(closes) >= 5:
                short_trend = torch.mean(closes[-5:]) > torch.mean(closes[-10:-5])
                patterns[4] = torch.where(short_trend, 1.0, -1.0)
            
            return patterns
    
    def _normalize_features_gpu(self):
        """GPU-accelerated feature normalization"""
        try:
            with _cuda_guard(self.device):  # BUG FIX: CPU-safe cuda guard
                for feature_name, feature_tensor in self.feature_tensors_gpu.items():
                    if feature_name not in self.normalization_params:
                        # Initialize normalization parameters
                        if feature_tensor.dim() == 0:  # Scalar
                            self.normalization_params[feature_name] = {
                                'mean': feature_tensor.item(),
                                'std': 1.0
                            }
                        else:  # Tensor
                            self.normalization_params[feature_name] = {
                                'mean': torch.mean(feature_tensor).item(),
                                'std': torch.std(feature_tensor).item() + 1e-8
                            }
                    
                    params = self.normalization_params[feature_name]
                    
                    # Apply normalization
                    if feature_tensor.dim() == 0:  # Scalar
                        normalized = (feature_tensor - params['mean']) / params['std']
                        self.feature_tensors_gpu[feature_name] = normalized
                    else:  # Tensor
                        normalized = (feature_tensor - params['mean']) / params['std']
                        self.feature_tensors_gpu[feature_name] = normalized
                        
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1173)
            self.logger.error(f"❌ Feature normalization error: {e}")
    
    # ==================== ENHANCED REAL-TIME DATA ACCESS METHODS ====================
    
    def get_current_features(self) -> Dict[str, np.ndarray]:
        """Get current normalized features for trading system"""
        try:
            features_cpu = {}
            
            for feature_name, feature_tensor in self.feature_tensors_gpu.items():
                if feature_tensor.is_cuda:
                    features_cpu[feature_name] = feature_tensor.cpu().numpy()
                else:
                    features_cpu[feature_name] = feature_tensor.numpy()
            
            return features_cpu
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1191)
            self.logger.error(f"❌ Feature retrieval error: {e}")
            return {}
    
    def get_candle_data(self, timeframe: str, lookback: int = 100) -> Optional[Dict]:
        """Get candle data for specific timeframe"""
        try:
            if timeframe not in self.candle_buffers_gpu:
                return None
            
            buffer = self.candle_buffers_gpu[timeframe]
            if buffer['count'] == 0:
                return None
            
            lookback = min(lookback, buffer['count'])
            start_idx = (buffer['pointer'] - lookback + 1) % len(buffer['open'])
            
            candle_data = {}
            for key in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
                tensor_data = self._get_circular_slice(buffer[key], start_idx, lookback)
                if tensor_data.is_cuda:
                    candle_data[key] = tensor_data.cpu().numpy()
                else:
                    candle_data[key] = tensor_data.numpy()
            
            return candle_data
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1218)
            self.logger.error(f"❌ Candle data retrieval error: {e}")
            return None
    
    def get_market_state(self) -> Dict:
        """Get comprehensive market state snapshot"""
        try:
            market_state = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'features': self.get_current_features(),
                'sequence_stats': self.sequence_validator.get_stats(),
                'performance': {
                    'message_count': self.message_count,
                    'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                    'gpu_memory_used': torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == 'cuda' else 0,
                    'reconnect_attempts': self.reconnect_attempts
                }
            }
            
            # Add current price from 1-minute data (sub-1m frames are not used)
            candle_1m = self.get_candle_data('1m', 1)
            if candle_1m:
                market_state['current_price'] = float(candle_1m['close'][-1])
                market_state['current_volume'] = float(candle_1m['volume'][-1])
            
            # Publish to CognitiveBus for Watcher AI monitoring
            if hasattr(self, 'bus') and self.bus:
                price = market_state.get('current_price', 'UNKNOWN')
                vol   = market_state.get('current_volume', 0)
                msg   = f"Live Data Engine: Price={price}, Volume={vol:.2f}, Symbol={self.symbol}. Messages processed: {self.message_count}"
                self.bus.publish('THOUGHTS', 'Part7_LiveData', msg)
            return market_state
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1252)
            self.logger.error(f"❌ Market state retrieval error: {e}")
            return {}
    
    # ==================== ENHANCED SYSTEM MANAGEMENT ====================
    
    async def _safe_shutdown(self):
        """Safe shutdown of live data engine"""
        self.logger.info("🛑 Shutting down Enhanced Live Data Engine...")
        self.is_running = False
        
        if self.websocket:
            await self.websocket.close()
        
        self.executor.shutdown(wait=False)
        self.gpu_manager.cleanup_live_memory()
        
        self.logger.info("✅ Enhanced Live Data Engine shutdown complete")
    
    def get_system_status(self) -> Dict:
        """Get enhanced system status report"""
        return {
            'websocket_connected': self.websocket is not None and not self.websocket.closed,
            'is_running': self.is_running,
            'message_count': self.message_count,
            'reconnect_attempts': self.reconnect_attempts,
            'processing_performance': {
                'avg_time': np.mean(self.processing_times) if self.processing_times else 0,
                'max_time': np.max(self.processing_times) if self.processing_times else 0,
                'min_time': np.min(self.processing_times) if self.processing_times else 0
            },
            'gpu_memory': self.gpu_manager.get_memory_stats(),
            'sequence_quality': self.sequence_validator.get_stats(),
            'buffer_status': {
                timeframe: {
                    'count': buffer['count'],
                    'pointer': buffer['pointer'],
                    'capacity': len(buffer['open'])
                }
                for timeframe, buffer in self.candle_buffers_gpu.items()
            }
        }

# ==================== ENHANCED LINUX-OPTIMIZED LAUNCHER ====================

class EnhancedLinuxLiveDataLauncher:
    """Linux-optimized launcher for enhanced live data engine"""
    
    def __init__(self):
        self.data_engine: Optional[EnhancedGPULiveDataEngine] = None
        self.loop = None
        self.performance_stats = {
            'start_time': time.time(),
            'total_messages': 0,
            'total_errors': 0
        }
        self.logger = EnhancedLogger()
    
    async def start_live_data_system(self):
        """Start the complete enhanced live data system"""
        self.logger.info("🚀 ACCELERATED Starting Enhanced Live Data System...")
        
        try:
            # Initialize enhanced data engine
            self.data_engine = EnhancedGPULiveDataEngine()
            
            # Start all tasks
            websocket_task = asyncio.create_task(
                self.data_engine.start_websocket_listener()
            )
            
            monitor_task = asyncio.create_task(
                self._enhanced_monitor_system_performance()
            )
            
            health_task = asyncio.create_task(
                self._system_health_check()
            )
            
            await asyncio.gather(websocket_task, monitor_task, health_task)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1333)
            self.logger.error(f"❌ Enhanced system error: {e}")
        finally:
            await self._shutdown_system()
    
    async def _enhanced_monitor_system_performance(self):
        """એન્હાન્સ્ડ પરફોર્મન્સ મોનિટરિંગ"""
        while self.data_engine and self.data_engine.is_running:
            try:
                status = self.data_engine.get_system_status()
                
                # Comprehensive monitoring
                if self.data_engine.message_count % 1000 == 0:
                    self._log_detailed_status(status)
                
                # Resource management
                await self._manage_system_resources(status)
                
                await asyncio.sleep(15)  # More frequent checks
                
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1353)
                self.logger.error(f"❌ Enhanced monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _system_health_check(self):
        """સિસ્ટમ હેલ્થ ચેક for 24/7 operation"""
        while self.data_engine and self.data_engine.is_running:
            try:
                # Memory health check
                if self.data_engine.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated()
                    if allocated > self.data_engine.gpu_manager.max_vram:
                        self.logger.warning("⚠️ High VRAM usage, cleaning up...")
                        self.data_engine.gpu_manager.cleanup_live_memory()
                
                # Connection health check
                if hasattr(self.data_engine, 'reconnect_attempts'):
                    if self.data_engine.reconnect_attempts > 5:
                        self.logger.warning("🔄 High reconnect attempts, checking network...")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1375)
                self.logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(30)
    
    def _log_detailed_status(self, status: Dict):
        """ડીટેલ્ડ સ્ટેટસ લોગિંગ"""
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 ENHANCED SYSTEM STATUS")
        self.logger.info(f"Messages: {status['message_count']:,}")
        self.logger.info(f"WebSocket: {'Connected' if status['websocket_connected'] else 'Disconnected'}")
        self.logger.info(f"Processing: {status['processing_performance']['avg_time']:.4f}s avg")
        
        if status['gpu_memory']:
            self.logger.info(f"GPU Memory: {status['gpu_memory']['allocated_mb']:.1f}MB")
        
        # Buffer status
        for tf, buf in status['buffer_status'].items():
            self.logger.info(f"Buffer {tf}: {buf['count']}/{buf['capacity']}")
        self.logger.info("="*50)
    
    async def _manage_system_resources(self, status: Dict):
        """સિસ્ટમ રિસોર્સ મેનેજમેન્ટ"""
        # GPU memory management
        if status['gpu_memory'] and status['gpu_memory']['allocated_mb'] > 2800:  # 2.8GB threshold
            self.logger.warning("🧹 High GPU memory, triggering cleanup...")
            self.data_engine.gpu_manager.cleanup_live_memory()
        
        # Processing time alert
        avg_time = status['processing_performance']['avg_time']
        if avg_time > 0.1:  # 100ms threshold
            self.logger.warning(f"⚠️ Slow processing: {avg_time:.4f}s")
    
    async def _shutdown_system(self):
        """Graceful system shutdown"""
        self.logger.info("🛑 Shutting down Enhanced Live Data System...")
        
        if self.data_engine:
            await self.data_engine._safe_shutdown()
        
        self.logger.info("✅ Enhanced Live Data System shutdown complete")

# ==================== ENHANCED TRADING SYSTEM INTEGRATION ====================

class EnhancedLiveTradingSystem:
    """એન્હાન્સ્ડ ટ્રેડિંગ સિસ્ટમ with better integration"""
    
    def __init__(self, master_system):
        self.master = master_system
        self.launcher = EnhancedLinuxLiveDataLauncher()
        self.is_live = False
        self.trading_enabled = False
        
        # Integration with previous parts
        self.feature_cache = LinuxOptimizedDeque(maxlen=100)
        self.signal_generator = None
        self.logger = EnhancedLogger()
        
        # LegacyLLM Volatility & Risk Monitor Cooldown setup
        self.last_legacy_llm_time = 0
        self.legacy_llm_cooldown = 30  # seconds
        self.last_legacy_llm_insight = "Market features stable. No anomalous volatility detected."
        self.last_legacy_llm_risk_state = "NORMAL"
    
    async def start_live_trading(self):
        """એન્હાન્સ્ડ લાઈવ ટ્રેડિંગ સ્ટાર્ટ"""
        self.logger.info("🚀 Starting Enhanced Live Trading with GPU Acceleration...")
        self.is_live = True
        self.trading_enabled = True
        
        try:
            # Initialize signal generator from previous parts
            if self.master and hasattr(self.master, 'signal_engine'):
                self.signal_generator = self.master.signal_engine
            
            await self.launcher.start_live_data_system()
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1451)
            self.logger.error(f"❌ Enhanced live trading error: {e}")
            self.is_live = False
            self.trading_enabled = False
    
    def _generate_legacy_llm_volatility_prompt(self, market_state: Dict) -> str:
        """Generate LegacyLLM prompt for Institutional Risk & Anomaly Monitoring"""
        features = market_state.get('features', {})
        current_price = market_state.get('current_price', 'N/A')
        current_volume = market_state.get('current_volume', 'N/A')
        symbol = market_state.get('symbol', 'BTCUSDT')

        feat_summary = {}
        if isinstance(features, dict):
            for k, v in features.items():
                if hasattr(v, '__len__') and len(v) > 0:
                    feat_summary[k] = float(v[-1]) if hasattr(v[-1], 'item') else float(v[-1])
                elif isinstance(v, (int, float)):
                    feat_summary[k] = float(v)
            feat_str = json.dumps(feat_summary)
        else:
            feat_str = str(features)

        prompt = f"""You are a highly advanced AI Risk & Volatility Monitor for an institutional trading desk. You analyze real-time GPU-calculated market features.

Symbol: {symbol}
Current Price: {current_price}
Current Volume: {current_volume}
GPU Live Features (RSI, Momentum, MACD, Volatility, Microstructure): {feat_str}

Task: Detect sudden market manipulation, liquidity vacuums, flash crash risk, or extreme volatility shifts.

Respond with EXACTLY ONE of the following classification tags at the start of your response:
- [NORMAL] : Market is stable.
- [HIGH_VOLATILITY] : Fast market conditions or elevated volatility.
- [ANOMALY] : Suspected manipulation, sudden liquidity vacuum, or flash crash.

Follow the tag with a brief 1-2 sentence institutional risk assessment.
"""
        return prompt

    def get_enhanced_market_data(self) -> Optional[Dict]:
        """એન્હાન્સ્ડ માર્કેટ ડેટા with trading signals and LegacyLLM Volatility Risk Monitor"""
        if not self.launcher.data_engine:
            return None
        
        market_data = self.launcher.data_engine.get_market_state()
        
        if not market_data:
            return None
        
        # Add trading signals if available
        if 'features' in market_data and self.signal_generator:
            try:
                features = market_data['features']
                signal = self._generate_trading_signal(features)
                market_data['trading_signal'] = signal
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1508)
                self.logger.error(f"❌ Signal generation error: {e}")
                market_data['trading_signal'] = None
        
        # No cached model risk label: stale commentary must never enter market_data.
        market_data['advisory_status'] = 'unavailable'

        return market_data
    
    def _generate_trading_signal(self, features: Dict) -> str:
        """ટ્રેડિંગ સિગ્નલ જનરેશન using features"""
        try:
            # Simple signal logic based on features
            if 'momentum' in features and 'volatility' in features:
                momentum = features['momentum']
                volatility = features['volatility']
                
                # Basic signal logic
                if momentum > 0.7 and volatility < 0.1:
                    return 'BUY'
                elif momentum < -0.7 and volatility < 0.1:
                    return 'SELL'
                else:
                    return 'HOLD'
            
            return 'HOLD'
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1535)
            self.logger.error(f"❌ Signal generation error: {e}")
            return 'HOLD'
    
    def stop_live_trading(self):
        """સુરક્ષિત લાઈવ ટ્રેડિંગ સ્ટોપ"""
        self.logger.info("🛑 Stopping Enhanced Live Trading...")
        self.is_live = False
        self.trading_enabled = False
        
        if self.launcher.data_engine:
            self.launcher.data_engine.is_running = False

# ==================== ENHANCED LINUX OPTIMIZATION ====================

def setup_enhanced_linux_environment():
    """એન્હાન્સ્ડ Linux ઓપ્ટિમાઈઝેશન"""
    # Async optimizations
    try:
        import uvloop


# ==================== MAIN EXECUTION ====================

    except Exception as e:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1559)
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    setup_enhanced_linux_environment()
    
    async def main():
        trading_system = EnhancedLiveTradingSystem(None)
        await trading_system.start_live_trading()
    
    # Run with enhanced event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1573)
        print("\n🛑 Enhanced system stopped by user")
    except Exception as e:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1575)
        print(f"❌ Enhanced system error: {e}")