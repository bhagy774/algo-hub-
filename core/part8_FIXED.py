try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ---- Helpers inserted for Part-8 Fix ----
from collections import deque


# PyTorch with fallback for Windows/WSL compatibility
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 13)
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
        try:
            super().append(item)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 119)
            pass

def _safe_get_device_name(device):
    try:
        if hasattr(device, "type") and device.type == "cuda":
            try:
                idx = device.index if hasattr(device, "index") and device.index is not None else 0
                return torch.cuda.get_device_name(idx)
            except:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 128)
                try:
                    return torch.cuda.get_device_name()
                except:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 131)
                    return "cuda_device"
        return str(device)
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 134)
        return "unknown_device"

class GPUFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def extract_basic(self, data):
        try:
            return torch.tensor([float(x) for x in data[:10]], device=self.device)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 143)
            return torch.zeros(10, device=self.device)
# ---- End helpers ----

# ==================== GPU-OPTIMIZED PART-8: PATTERN RECOGNITION ENGINE ====================
# DEEPSEEK AI-POWERED REWRITE - FULL HARDWARE OPTIMIZATION
# LINUX UBUNTU + GTX 1650 CUDA + i5 10th Gen OPTIMIZED

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# FIX #7: CuPy fallback — if CUDA/CuPy not installed, fall back to NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 160)
    import numpy as cp  # type: ignore
    CUPY_AVAILABLE = False
import numpy as np
import asyncio
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import pandas as pd
from pathlib import Path
import gc
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from contextlib import nullcontext  # FIX BUG #7: needed for CPU-safe cuda.device() context




# ==================== ENHANCED GPU PATTERN MEMORY MANAGER ====================

class EnhancedPatternGPUMemoryManager:
    """GTX 1650 4GB VRAM Optimized Memory Manager for Advanced Pattern Recognition"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram = 3.2 * 1024 * 1024 * 1024  # 3.2GB safety limit
        self.max_ram = 4.0 * 1024 * 1024 * 1024   # 4GB RAM limit
        
        # Enhanced SSD cache for pattern libraries
        self.cache_dir = Path("/tmp/pattern_cache_v2")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Pattern memory optimization
        self.pattern_cache = {}
        self.template_buffers = {}
        
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.70)  # 70% of 4GB for GTX 1650
            print(f"🚀 Enhanced Pattern GPU Memory: {_safe_get_device_name(self.device)}")
    
    def allocate_pattern_tensor(self, data: np.ndarray, name: str, persistent: bool = True) -> torch.Tensor:
        """Enhanced GPU tensor allocation optimized for pattern data"""
        try:
            # Check memory constraints — FIX BUG #1: CPU fallback needs explicit device='cpu'
            if self.device.type == 'cpu' or torch.cuda.memory_allocated() > self.max_vram:
                return torch.tensor(data, dtype=torch.float32, device='cpu')
            
            # Convert to tensor with GPU optimization
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            # Smart persistence decision
            if persistent and tensor.numel() < 10000:  # Medium tensors in GPU
                self.pattern_cache[name] = tensor
                return tensor
            else:
                # Large tensors use advanced memory mapping
                return self._create_enhanced_memory_mapped_tensor(data, name)
                
        except RuntimeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 220)
            print(f"⚠️ Enhanced Pattern GPU allocation failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def _create_enhanced_memory_mapped_tensor(self, data: np.ndarray, name: str) -> torch.Tensor:
        """Create advanced memory-mapped tensor for large pattern datasets"""
        try:
            timestamp = int(time.time() * 1000)
            cache_file = self.cache_dir / f"{name}_{timestamp}.dat"
            
            # Create memory map with optimization
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data.shape)
            mmap[:] = data
            mmap.flush()

            # FIX BUG #2: torch.from_numpy(mmap) shares memory with the disk-backed file.
            # When cache is cleaned up, accessing the tensor causes a SIGSEGV (hard crash).
            # Use torch.tensor() to create an independent COPY of the data.
            result = torch.tensor(np.array(mmap), dtype=torch.float32)
            del mmap  # Release the mmap handle explicitly
            return result
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 241)
            print(f"⚠️ Enhanced Pattern memory mapping failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def cleanup_pattern_memory(self):
        """Aggressive cleanup with pattern-specific optimization"""
        # FIX BUG #3: torch.cuda.empty_cache() raises RuntimeError on CPU-only machines
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Clear pattern cache
        self.pattern_cache.clear()


# ==================== GPU-ACCELERATED MARKET STRUCTURE & SMC ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p8_structure_engine', 'analyze')
class MarketStructureEngineGPU:
    """
    JARVIS PART 8 - GPU-ACCELERATED MARKET STRUCTURE & SMART MONEY CONCEPTS (SMC) ENGINE
    GTX 1650 CUDA & CPU Optimized.

    Quantitative Market Structure Architecture:
    1. Multi-Bar Fractal Pivots (order k=2): Validates true Swing Highs and Swing Lows.
    2. Break of Structure (BOS): Decisive close beyond swing pivot with ATR threshold buffer.
    3. Change of Character (CHoCH): Trend reversal shift (Higher Low broken or Lower High overtaken).
    4. EMA20 Alignment & Volume Filter: Eliminates low-liquidity wick traps and false breakouts.
    5. Range Consolidation / Equilibrium Deadband: Strictly signals 0 (Neutral) while price is oscillating inside the range.
    """
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')

    @part_advisory_entry("part8_structure", data_parameter="data", context_parameter="context")
    def analyze(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        return qm.part_signal("8", data, context=context)


# ==================== ENHANCED GPU-ACCELERATED PATTERN RECOGNITION ENGINE ====================

class EnhancedGPUPatternRecognitionEngine:
    """
    INSTITUTIONAL-GRADE ENHANCED PATTERN RECOGNITION ENGINE
    GTX 1650 CUDA Optimized for Real-Time Multi-Pattern Detection
    """
    
    def __init__(self, symbol: str = 'BTCUSDT', pattern_lookback: int = 200):
        self.symbol = symbol
        self.pattern_lookback = pattern_lookback
        self.gpu_manager = EnhancedPatternGPUMemoryManager()
        self.device = self.gpu_manager.device
        
        # Enhanced GPU-optimized pattern buffers
        self.candle_sequences_gpu: Dict[str, torch.Tensor] = {}
        self.pattern_templates_gpu: Dict[str, Dict[str, torch.Tensor]] = {}
        self.detected_patterns_gpu: Dict[str, Dict] = {}
        
        # Advanced pattern memory systems
        self.short_term_memory = LinuxOptimizedDeque(maxlen=1000)
        self.long_term_memory = LinuxOptimizedDeque(maxlen=5000)
        self.pattern_performance = defaultdict(lambda: LinuxOptimizedDeque(maxlen=500))
        self.pattern_statistics = defaultdict(dict)
        
        # Enhanced pattern configuration
        self.pattern_config = {
            'reversal_lookback': 12,
            'continuation_lookback': 10,
            'liquidity_sweep_threshold': 0.0025,  # 0.25% move
            'institutional_sequence_length': 6,
            'volatility_normalization': True,
            'min_pattern_confidence': 0.65,
            'microstructure_lookback': 25,
            'adaptive_confidence': True,
            'multi_timeframe_analysis': True,
            'pattern_fusion_enabled': True
        }
        
        # Real-time analysis enhancements
        self.active_patterns = LinuxOptimizedDeque(maxlen=100)
        self.pattern_signals = LinuxOptimizedDeque(maxlen=200)
        self.pattern_clusters = defaultdict(list)
        
        # Performance monitoring
        self.analysis_times = LinuxOptimizedDeque(maxlen=100)
        self.pattern_counts = defaultdict(int)
        
        # Threading and async control
        self.data_lock = threading.RLock()
        self.is_running = True
        self.executor = ThreadPoolExecutor(max_workers=4)  # i5 4-core optimized
        
        # LegacyLLM Pattern Technical Analysis Validation Cooldown setup
        self.last_legacy_llm_time = 0
        self.legacy_llm_cooldown = 30  # seconds
        self.last_legacy_llm_validation = "CONFIRM_PATTERN"
        self.last_legacy_llm_insight = "Geometric patterns validated against volume context."

        # Initialize enhanced GPU pattern templates
        self._initialize_enhanced_pattern_templates_gpu()
        
        print("🚀 ENHANCED GPU Pattern Recognition Engine Initialized - Linux Optimized")
    
    def _initialize_enhanced_pattern_templates_gpu(self):
        """Initialize advanced GPU-optimized pattern templates"""
        try:
            # Enhanced Reversal pattern templates
            self.pattern_templates_gpu['reversal'] = {
                'double_top': self._create_enhanced_double_top_template(),
                'double_bottom': self._create_enhanced_double_bottom_template(),
                'head_shoulders': self._create_enhanced_head_shoulders_template(),
                'inverse_head_shoulders': self._create_enhanced_inverse_head_shoulders_template(),
                'triple_top': self._create_triple_top_template(),
                'triple_bottom': self._create_triple_bottom_template(),
                'rounding_top': self._create_rounding_top_template(),
                'rounding_bottom': self._create_rounding_bottom_template()
            }
            
            # Enhanced Continuation pattern templates
            self.pattern_templates_gpu['continuation'] = {
                'flag_bullish': self._create_enhanced_flag_bullish_template(),
                'flag_bearish': self._create_enhanced_flag_bearish_template(),
                'pennant_bullish': self._create_enhanced_pennant_bullish_template(),
                'pennant_bearish': self._create_enhanced_pennant_bearish_template(),
                'triangle_ascending': self._create_triangle_ascending_template(),
                'triangle_descending': self._create_triangle_descending_template(),
                'wedge_rising': self._create_wedge_rising_template(),
                'wedge_falling': self._create_wedge_falling_template()
            }
            
            # Advanced Institutional pattern templates
            self.pattern_templates_gpu['institutional'] = {
                'accumulation_phase1': self._create_accumulation_phase1_template(),
                'accumulation_phase2': self._create_accumulation_phase2_template(),
                'distribution_phase1': self._create_distribution_phase1_template(),
                'distribution_phase2': self._create_distribution_phase2_template(),
                're_accumulation': self._create_re_accumulation_template(),
                're_distribution': self._create_re_distribution_template(),
                'smart_money_accumulation': self._create_smart_money_accumulation_template(),
                'smart_money_distribution': self._create_smart_money_distribution_template()
            }
            
            # Liquidity and Microstructure patterns
            self.pattern_templates_gpu['liquidity'] = {
                'liquidity_sweep_bullish': self._create_liquidity_sweep_bullish_template(),
                'liquidity_sweep_bearish': self._create_liquidity_sweep_bearish_template(),
                'liquidity_grab_bullish': self._create_liquidity_grab_bullish_template(),
                'liquidity_grab_bearish': self._create_liquidity_grab_bearish_template(),
                'stop_hunt_bullish': self._create_stop_hunt_bullish_template(),
                'stop_hunt_bearish': self._create_stop_hunt_bearish_template()
            }
            
            print(f"✅ Loaded {sum(len(templates) for templates in self.pattern_templates_gpu.values())} pattern templates")
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 395)
            print(f"⚠️ Enhanced pattern template initialization warning: {e}")
    
    # ==================== ENHANCED PATTERN TEMPLATES ====================
    
    def _create_enhanced_double_top_template(self):
        """Enhanced GPU template for double top reversal pattern"""
        return torch.tensor([1.0, 0.92, 1.0, 0.88, 0.94, 0.96], device=self.device, dtype=torch.float32)
    
    def _create_enhanced_double_bottom_template(self):
        """Enhanced GPU template for double bottom reversal pattern"""
        return torch.tensor([0.92, 1.0, 0.92, 1.0, 0.94, 0.96], device=self.device, dtype=torch.float32)
    
    def _create_enhanced_head_shoulders_template(self):
        """Enhanced GPU template for head and shoulders pattern"""
        return torch.tensor([0.92, 1.0, 1.15, 1.0, 0.92, 0.88], device=self.device, dtype=torch.float32)
    
    def _create_enhanced_inverse_head_shoulders_template(self):
        """Enhanced GPU template for inverse head and shoulders pattern"""
        return torch.tensor([1.08, 1.0, 0.85, 1.0, 1.08, 1.12], device=self.device, dtype=torch.float32)
    
    def _create_triple_top_template(self):
        """GPU template for triple top pattern"""
        return torch.tensor([1.0, 0.95, 1.0, 0.93, 1.0, 0.90], device=self.device, dtype=torch.float32)
    
    def _create_triple_bottom_template(self):
        """GPU template for triple bottom pattern"""
        return torch.tensor([0.95, 1.0, 0.95, 1.0, 0.95, 1.0], device=self.device, dtype=torch.float32)
    
    def _create_enhanced_flag_bullish_template(self):
        """Enhanced GPU template for bullish flag pattern"""
        return torch.tensor([1.0, 0.75, 0.80, 0.78, 0.82, 0.85], device=self.device, dtype=torch.float32)
    
    def _create_enhanced_flag_bearish_template(self):
        """Enhanced GPU template for bearish flag pattern"""
        return torch.tensor([0.85, 0.82, 0.78, 0.80, 0.75, 1.0], device=self.device, dtype=torch.float32)
    
    def _create_accumulation_phase1_template(self):
        """GPU template for institutional accumulation phase 1"""
        return torch.tensor([0.85, 0.88, 0.92, 0.95, 0.98, 1.0], device=self.device, dtype=torch.float32)
    
    def _create_smart_money_accumulation_template(self):
        """GPU template for smart money accumulation"""
        return torch.tensor([0.90, 0.87, 0.92, 0.89, 0.94, 0.96], device=self.device, dtype=torch.float32)

    def _create_rounding_top_template(self):
        return torch.tensor([0.80, 0.90, 0.98, 1.0, 0.98, 0.90, 0.80], device=self.device, dtype=torch.float32)

    def _create_rounding_bottom_template(self):
        return torch.tensor([1.0, 0.90, 0.82, 0.80, 0.82, 0.90, 1.0], device=self.device, dtype=torch.float32)

    def _create_enhanced_pennant_bullish_template(self):
        return torch.tensor([0.80, 1.0, 0.90, 0.95, 0.92, 0.98], device=self.device, dtype=torch.float32)

    def _create_enhanced_pennant_bearish_template(self):
        return torch.tensor([1.0, 0.80, 0.90, 0.85, 0.88, 0.82], device=self.device, dtype=torch.float32)

    def _create_triangle_ascending_template(self):
        return torch.tensor([0.80, 1.0, 0.85, 1.0, 0.90, 1.0], device=self.device, dtype=torch.float32)

    def _create_triangle_descending_template(self):
        return torch.tensor([1.0, 0.80, 0.95, 0.80, 0.90, 0.80], device=self.device, dtype=torch.float32)

    def _create_wedge_rising_template(self):
        return torch.tensor([0.80, 0.90, 0.87, 0.95, 0.93, 1.0], device=self.device, dtype=torch.float32)

    def _create_wedge_falling_template(self):
        return torch.tensor([1.0, 0.90, 0.93, 0.85, 0.87, 0.80], device=self.device, dtype=torch.float32)

    def _create_accumulation_phase2_template(self):
        return torch.tensor([0.88, 0.85, 0.90, 0.87, 0.95, 1.0], device=self.device, dtype=torch.float32)

    def _create_distribution_phase1_template(self):
        return torch.tensor([1.0, 0.98, 0.95, 0.92, 0.88, 0.85], device=self.device, dtype=torch.float32)

    def _create_distribution_phase2_template(self):
        return torch.tensor([0.98, 1.0, 0.95, 0.92, 0.87, 0.82], device=self.device, dtype=torch.float32)

    def _create_re_accumulation_template(self):
        return torch.tensor([0.90, 0.95, 0.92, 0.97, 0.94, 1.0], device=self.device, dtype=torch.float32)

    def _create_re_distribution_template(self):
        return torch.tensor([0.95, 0.90, 0.93, 0.87, 0.89, 0.82], device=self.device, dtype=torch.float32)

    def _create_smart_money_distribution_template(self):
        return torch.tensor([0.95, 1.0, 0.94, 0.98, 0.90, 0.85], device=self.device, dtype=torch.float32)

    def _create_liquidity_sweep_bullish_template(self):
        return torch.tensor([0.95, 0.92, 0.85, 0.96, 1.0, 1.02], device=self.device, dtype=torch.float32)

    def _create_liquidity_sweep_bearish_template(self):
        return torch.tensor([0.95, 0.98, 1.08, 0.94, 0.90, 0.88], device=self.device, dtype=torch.float32)

    def _create_liquidity_grab_bullish_template(self):
        return torch.tensor([0.92, 0.88, 0.82, 0.95, 1.0, 1.01], device=self.device, dtype=torch.float32)

    def _create_liquidity_grab_bearish_template(self):
        return torch.tensor([0.96, 1.0, 1.06, 0.92, 0.87, 0.85], device=self.device, dtype=torch.float32)

    def _create_stop_hunt_bullish_template(self):
        return torch.tensor([0.94, 0.90, 0.80, 0.98, 1.01, 1.03], device=self.device, dtype=torch.float32)

    def _create_stop_hunt_bearish_template(self):
        return torch.tensor([0.96, 1.01, 1.09, 0.91, 0.86, 0.84], device=self.device, dtype=torch.float32)
    
    # ==================== MULTI-TIMEFRAME PATTERN SCANNING ====================
    
    def scan_patterns_mtf(self, mtf_data):
        """
        Scan for patterns across ALL timeframes
        Accepts dict: {'1m': df, '3m': df, '5m': df, ...}
        Returns patterns found on each timeframe
        """
        try:
            all_patterns = {}
            
            for tf_name, tf_df in mtf_data.items():
                if tf_df is None or len(tf_df) < 20:
                    continue
                
                try:
                    tf_patterns = []
                    
                    # Extract OHLCV as numpy arrays
                    closes = tf_df['close'].values.astype(float)
                    highs = tf_df['high'].values.astype(float)
                    lows = tf_df['low'].values.astype(float)
                    volumes = tf_df['volume'].values.astype(float) if 'volume' in tf_df.columns else np.ones(len(closes))
                    
                    candle_data = {
                        'close': closes, 'high': highs, 'low': lows,
                        'open': tf_df['open'].values.astype(float),
                        'volume': volumes
                    }
                    
                    # FIX BUG #4: attribute is 'pattern_templates_gpu', not 'pattern_templates'
                    # FIX BUG #5: must iterate nested dict (category -> templates), not top-level
                    if hasattr(self, 'pattern_templates_gpu') and self.pattern_templates_gpu:
                        for category, category_templates in self.pattern_templates_gpu.items():
                            for pattern_name, template in category_templates.items():
                                try:
                                    if hasattr(template, 'shape') and len(template.shape) > 0:
                                        template_len = template.shape[0]
                                        if len(closes) >= template_len:
                                            # Normalize and correlate
                                            segment = closes[-template_len:]
                                            seg_norm = (segment - segment.min()) / (segment.max() - segment.min() + 1e-8)

                                            if hasattr(template, 'cpu'):
                                                tmpl_np = template.detach().cpu().numpy().flatten()
                                            else:
                                                tmpl_np = np.array(template).flatten()

                                            if len(tmpl_np) == len(seg_norm):
                                                corr = np.corrcoef(seg_norm, tmpl_np)[0, 1]
                                                if abs(corr) > 0.7:
                                                    tf_patterns.append({
                                                        'pattern': pattern_name,
                                                        'correlation': round(float(corr), 3),
                                                        'timeframe': tf_name,
                                                        'signal': 1 if 'bullish' in pattern_name.lower() or 'bottom' in pattern_name.lower() else (-1 if 'bearish' in pattern_name.lower() or 'top' in pattern_name.lower() else 0)
                                                    })
                                except Exception:
                                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 557)
                                    pass

                    if tf_patterns:
                        all_patterns[tf_name] = tf_patterns
                        
                except Exception as e:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 563)
                    pass  # Skip failed TFs silently
            
            if all_patterns:
                total = sum(len(p) for p in all_patterns.values())
                msg = f"Pattern MTF: Detected {total} patterns across {list(all_patterns.keys())}"
                # print(f"  🎯 {msg}")
                if hasattr(self, 'bus') and self.bus:
                    self.bus.publish('THOUGHTS', 'Part8_Pattern', msg)
            
            return all_patterns
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 575)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part8_Pattern', e, context='scan_patterns_mtf top-level')
            else:
                print(f"ERROR MTF Pattern scan error: {e}")
            return {}
    
    # ==================== ENHANCED GPU-ACCELERATED PATTERN DETECTION ====================
    
    async def analyze_enhanced_candle_sequences(self, candle_data: Dict[str, np.ndarray]) -> Dict:
        """
        Enhanced GPU-accelerated pattern detection with advanced features
        """
        start_time = time.time()
        
        try:
            # Convert candle data to enhanced GPU tensors
            candle_tensors = self._prepare_enhanced_candle_tensors_gpu(candle_data)
            
            if candle_tensors is None:
                return {'error': 'Invalid candle data'}
            
            # Enhanced GPU-accelerated pattern detection
            detection_results = await self._detect_enhanced_patterns_gpu(candle_tensors)
            
            # Update advanced pattern memory
            await self._update_enhanced_pattern_memory(detection_results)
            
            # Generate sophisticated trading signals
            await self._generate_enhanced_pattern_signals(detection_results)
            
            # Update performance metrics
            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)
            
            return {
                'success': True,
                'patterns_detected': len(detection_results),
                'analysis_time': analysis_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 617)
            print(f"❌ Enhanced pattern analysis error: {e}")
            return {'error': str(e)}
    
    def _prepare_enhanced_candle_tensors_gpu(self, candle_data: Dict[str, np.ndarray]) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare enhanced candle data for GPU processing"""
        try:
            required_keys = ['open', 'high', 'low', 'close', 'volume']
            if not all(key in candle_data for key in required_keys):
                return None
            
            # Convert to enhanced GPU tensors
            tensors = {}
            for key in required_keys:
                data = candle_data[key]
                if len(data) < self.pattern_lookback:
                    # Enhanced padding with interpolation
                    padded_data = self._enhanced_data_padding(data, self.pattern_lookback)
                    tensors[key] = self.gpu_manager.allocate_pattern_tensor(padded_data, f'{key}_enhanced')
                else:
                    tensors[key] = self.gpu_manager.allocate_pattern_tensor(
                        data[-self.pattern_lookback:], f'{key}_recent'
                    )
            
            # Calculate advanced features
            tensors['body_ratio'] = self._calculate_enhanced_body_ratios_gpu(tensors)
            tensors['wick_ratios'] = self._calculate_enhanced_wick_ratios_gpu(tensors)
            tensors['price_momentum'] = self._calculate_enhanced_price_momentum_gpu(tensors)
            tensors['volume_profile'] = self._calculate_volume_profile_gpu(tensors)
            tensors['volatility_measure'] = self._calculate_volatility_measure_gpu(tensors)
            
            return tensors
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 650)
            print(f"❌ Enhanced candle tensor preparation error: {e}")
            return None
    
    def _enhanced_data_padding(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Enhanced data padding with intelligent interpolation"""
        if len(data) == 0:
            return np.zeros(target_length, dtype=np.float32)
        
        current_length = len(data)
        if current_length >= target_length:
            return data[-target_length:]
        
        # Intelligent padding using last known value with slight noise
        padding_needed = target_length - current_length
        last_value = data[-1] if len(data) > 0 else 0

        # FIX BUG #6: last_value=0 or negative causes std≤0 → ValueError in np.random.normal
        # Use abs() + epsilon to ensure std is always positive
        padding = np.random.normal(last_value, abs(last_value) * 0.001 + 1e-8, padding_needed)

        # FIX BUG #14: padding must go BEFORE data, not after.
        # Original code appended padding to the END making the newest candles synthetic noise.
        # Correct: padding fills historical gaps at the START, real data is at the END.
        padded_data = np.concatenate([padding, data])
        
        return padded_data[-target_length:]
    
    def _calculate_enhanced_body_ratios_gpu(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced GPU-accelerated body ratio calculation"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()  # FIX BUG #7
            with _ctx:
                bodies = torch.abs(tensors['close'] - tensors['open'])
                ranges = tensors['high'] - tensors['low']

                # FIX BUG #8: torch.div() has no 'where' parameter — raises TypeError
                # Use torch.where() to safely divide only where ranges != 0
                body_ratios = torch.where(ranges != 0, bodies / (ranges + 1e-8), torch.zeros_like(bodies))
                
                # Apply smoothing filter
                if len(body_ratios) > 3:
                    weights = torch.tensor([0.25, 0.5, 0.25], device=self.device)
                    # FIX BUG #9: .squeeze() removes ALL size-1 dims including length dim when N=1.
                    # Use .squeeze(0).squeeze(0) to only remove batch and channel dims safely.
                    smoothed = torch.conv1d(body_ratios.unsqueeze(0).unsqueeze(0),
                                          weights.unsqueeze(0).unsqueeze(0),
                                          padding=1).squeeze(0).squeeze(0)
                    return smoothed
                
                return body_ratios
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 701)
            print(f"⚠️ Enhanced body ratio calculation warning: {e}")
            return torch.zeros_like(tensors['open'])
    
    def _calculate_volume_profile_gpu(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate volume profile features"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()  # FIX BUG #7
            with _ctx:
                volume = tensors['volume']
                price_range = tensors['high'] - tensors['low']
                
                # Volume intensity relative to price range
                volume_intensity = volume / (price_range + 1e-8)
                normalized_volume = volume_intensity / (torch.mean(volume_intensity) + 1e-8)
                
                return normalized_volume
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 718)
            print(f"⚠️ Volume profile calculation warning: {e}")
            return torch.ones_like(tensors['volume'])

    def _calculate_enhanced_wick_ratios_gpu(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced GPU-accelerated wick ratio calculation"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                upper_wicks = tensors['high'] - torch.maximum(tensors['open'], tensors['close'])
                lower_wicks = torch.minimum(tensors['open'], tensors['close']) - tensors['low']
                ranges = tensors['high'] - tensors['low']
                total_wicks = upper_wicks + lower_wicks
                return torch.where(ranges > 0, total_wicks / (ranges + 1e-8), torch.zeros_like(total_wicks))
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 732)
            print(f"⚠️ Enhanced wick ratio calculation warning: {e}")
            return torch.zeros_like(tensors['open'])

    def _calculate_enhanced_price_momentum_gpu(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced GPU-accelerated price momentum calculation"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                closes = tensors['close']
                if len(closes) > 1:
                    diffs = torch.zeros_like(closes)
                    diffs[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-8)
                    return diffs
                return torch.zeros_like(closes)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 747)
            print(f"⚠️ Enhanced price momentum calculation warning: {e}")
            return torch.zeros_like(tensors['open'])

    def _calculate_volatility_measure_gpu(self, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate volatility measure feature"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                ranges = tensors['high'] - tensors['low']
                closes = tensors['close']
                return ranges / (closes + 1e-8)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 759)
            print(f"⚠️ Volatility measure calculation warning: {e}")
            return torch.zeros_like(tensors['open'])
    
    # ==================== ADVANCED PATTERN DETECTION ALGORITHMS ====================
    
    async def _detect_enhanced_patterns_gpu(self, candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Enhanced GPU-accelerated pattern detection with multiple algorithms"""
        try:
            detected_patterns = {}

            # FIX BUG #11: asyncio.gather() must be OUTSIDE cuda.device() context.
            # The CUDA device context is thread-local and does NOT propagate through async awaits.
            # Each detection method manages its own device context internally.
            normalized_prices = self._enhanced_price_normalization_gpu(candle_tensors['close'])

            detection_tasks = [
                self._detect_enhanced_reversal_patterns_gpu(normalized_prices, candle_tensors),
                self._detect_enhanced_continuation_patterns_gpu(normalized_prices, candle_tensors),
                self._detect_advanced_institutional_patterns_gpu(candle_tensors),
                self._detect_liquidity_microstructure_patterns_gpu(candle_tensors),
                self._detect_complex_pattern_combinations_gpu(normalized_prices, candle_tensors)
            ]

            # Execute all detection algorithms
            results = await asyncio.gather(*detection_tasks)

            # Merge results with confidence weighting
            for result in results:
                detected_patterns.update(result)

            # Apply pattern fusion and conflict resolution
            fused_patterns = self._apply_pattern_fusion(detected_patterns)

            self.detected_patterns_gpu = fused_patterns
            return fused_patterns

        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 796)
            print(f"❌ Enhanced pattern detection error: {e}")
            return {}
    
    def _enhanced_price_normalization_gpu(self, prices: torch.Tensor) -> torch.Tensor:
        """Advanced price normalization with volatility adjustment"""
        try:
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()  # FIX BUG #7
            with _ctx:
                # Z-score normalization with volatility scaling
                price_mean = torch.mean(prices)
                price_std = torch.std(prices)
                
                if price_std > 0:
                    normalized = (prices - price_mean) / price_std
                    # Scale to 0-1 range — FIX BUG #10: add epsilon to prevent division by zero when min==max
                    scaled = (normalized - torch.min(normalized)) / (torch.max(normalized) - torch.min(normalized) + 1e-8)
                    return scaled
                else:
                    return torch.zeros_like(prices)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 816)
            print(f"⚠️ Enhanced price normalization warning: {e}")
            return torch.zeros_like(prices)
    
    async def _detect_enhanced_reversal_patterns_gpu(self, normalized_prices: torch.Tensor, 
                                                   candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Enhanced reversal pattern detection with volume confirmation"""
        try:
            reversal_patterns = {}
            
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()  # FIX BUG #7
            with _ctx:
                for pattern_name, template in self.pattern_templates_gpu['reversal'].items():
                    pattern_scores = []
                    volume_confidences = []
                    
                    for i in range(len(normalized_prices) - len(template) + 1):
                        price_slice = normalized_prices[i:i+len(template)]
                        
                        # Advanced correlation with multiple metrics
                        correlation = torch.corrcoef(torch.stack([price_slice, template]))[0, 1]
                        correlation = torch.nan_to_num(correlation, nan=0.0)
                        
                        # Dynamic time warping distance
                        dtw_distance = torch.norm(price_slice - template)
                        dtw_similarity = torch.exp(-dtw_distance)
                        
                        # Volume confirmation for reversal patterns
                        volume_slice = candle_tensors['volume'][i:i+len(template)]
                        # FIX BUG #12: torch.diff on length-1 returns empty → torch.mean(empty)=nan
                        volume_trend = torch.mean(torch.diff(volume_slice)) if len(volume_slice) > 1 else torch.tensor(0.0, device=self.device)
                        volume_confidence = torch.sigmoid(volume_trend * 10)  # Scale for sensitivity
                        
                        # Combined score with volume weighting
                        combined_score = (correlation * 0.4 + dtw_similarity * 0.4 + volume_confidence * 0.2)
                        pattern_scores.append(combined_score.item())
                        volume_confidences.append(volume_confidence.item())
                    
                    # Find best match with volume confirmation
                    if pattern_scores:
                        best_score = max(pattern_scores)
                        best_index = pattern_scores.index(best_score)
                        volume_confidence = volume_confidences[best_index]
                        
                        # Enhanced confidence calculation
                        final_confidence = self._calculate_enhanced_confidence(
                            best_score, volume_confidence, 'reversal', pattern_name
                        )
                        
                        if final_confidence > self.pattern_config['min_pattern_confidence']:
                            reversal_patterns[pattern_name] = {
                                'score': final_confidence,
                                'position': best_index,
                                'type': 'reversal',
                                'timestamp': time.time(),
                                'volume_confidence': volume_confidence,
                                'pattern_strength': best_score
                            }
            
            return reversal_patterns
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 877)
            print(f"⚠️ Enhanced reversal pattern detection warning: {e}")
            return {}

    async def _detect_enhanced_continuation_patterns_gpu(self, normalized_prices: torch.Tensor,
                                                         candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Enhanced continuation pattern detection"""
        try:
            continuation_patterns = {}
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                for pattern_name, template in self.pattern_templates_gpu.get('continuation', {}).items():
                    pattern_scores = []
                    for i in range(len(normalized_prices) - len(template) + 1):
                        price_slice = normalized_prices[i:i+len(template)]
                        correlation = torch.corrcoef(torch.stack([price_slice, template]))[0, 1]
                        correlation = torch.nan_to_num(correlation, nan=0.0)
                        dtw_distance = torch.norm(price_slice - template)
                        dtw_similarity = torch.exp(-dtw_distance)
                        score = (correlation * 0.5 + dtw_similarity * 0.5).item()
                        pattern_scores.append(score)
                    
                    if pattern_scores:
                        best_score = max(pattern_scores)
                        best_index = pattern_scores.index(best_score)
                        final_conf = self._calculate_enhanced_confidence(best_score, 0.7, 'continuation', pattern_name)
                        if final_conf > self.pattern_config['min_pattern_confidence']:
                            continuation_patterns[pattern_name] = {
                                'score': final_conf,
                                'position': best_index,
                                'type': 'continuation',
                                'timestamp': time.time(),
                                'volume_confidence': 0.7,
                                'pattern_strength': best_score
                            }
            return continuation_patterns
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 913)
            print(f"⚠️ Continuation pattern detection warning: {e}")
            return {}

    async def _detect_advanced_institutional_patterns_gpu(self, candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Advanced Math: Dynamic Time Warping (DTW) inspired similarity distance logic for pattern recognition"""
        try:
            institutional_patterns = {}
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                volume = candle_tensors['volume']
                closes = candle_tensors['close']

                # DTW simplified Euclidean distance heuristic for pattern matching
                if len(closes) >= 10:
                    recent_prices = closes[-10:]
                    recent_vol = volume[-10:]

                    # Target Accumulation Template (Down, flat, up)
                    acc_template = torch.tensor([1.0, 0.8, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0], device=self.device)
                    # Normalize recent prices 0 to 1
                    min_p = torch.min(recent_prices)
                    max_p = torch.max(recent_prices)
                    if max_p > min_p:
                        norm_prices = (recent_prices - min_p) / (max_p - min_p)

                        # Calculate Euclidean distance (simplified DTW path)
                        dtw_dist = torch.sqrt(torch.sum((norm_prices - acc_template) ** 2))

                        # The closer to 0, the stronger the match
                        pattern_strength = float(max(0.0, 1.0 - (dtw_dist.item() / 3.16))) # 3.16 is sqrt(10)

                        vol_avg = torch.mean(volume[-10:])
                        price_change = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8)

                        if volume[-1] > vol_avg * 1.5 and price_change > 0.005 and pattern_strength > 0.6:
                            institutional_patterns['smart_money_accumulation'] = {
                                'score': pattern_strength,
                                'position': len(closes) - 1,
                                'type': 'institutional',
                                'timestamp': time.time(),
                                'volume_confidence': 0.85,
                                'pattern_strength': pattern_strength
                            }
                        elif volume[-1] > vol_avg * 1.5 and price_change < -0.005:
                            institutional_patterns['smart_money_distribution'] = {
                                'score': 0.82,
                                'position': len(closes) - 1,
                                'type': 'institutional',
                                'timestamp': time.time(),
                                'volume_confidence': 0.85,
                                'pattern_strength': 0.82
                            }
            return institutional_patterns
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 967)
            print(f"⚠️ Institutional pattern detection warning: {e}")
            return {}

    async def _detect_liquidity_microstructure_patterns_gpu(self, candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Liquidity and microstructure pattern detection"""
        try:
            liquidity_patterns = {}
            _ctx = torch.cuda.device(self.device) if self.device.type == 'cuda' else nullcontext()
            with _ctx:
                highs = candle_tensors['high']
                lows = candle_tensors['low']
                closes = candle_tensors['close']
                if len(highs) >= 20:
                    prior_high = torch.max(highs[-20:-5])
                    prior_low = torch.min(lows[-20:-5])
                    curr_high = highs[-1]
                    curr_low = lows[-1]
                    curr_close = closes[-1]

                    if curr_high > prior_high and curr_close < prior_high:
                        liquidity_patterns['liquidity_sweep_bearish'] = {
                            'score': 0.85,
                            'position': len(highs) - 1,
                            'type': 'liquidity',
                            'timestamp': time.time(),
                            'volume_confidence': 0.80,
                            'pattern_strength': 0.85
                        }
                    elif curr_low < prior_low and curr_close > prior_low:
                        liquidity_patterns['liquidity_sweep_bullish'] = {
                            'score': 0.85,
                            'position': len(lows) - 1,
                            'type': 'liquidity',
                            'timestamp': time.time(),
                            'volume_confidence': 0.80,
                            'pattern_strength': 0.85
                        }
            return liquidity_patterns
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1006)
            print(f"⚠️ Liquidity pattern detection warning: {e}")
            return {}

    async def _detect_complex_pattern_combinations_gpu(self, normalized_prices: torch.Tensor,
                                                        candle_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Complex pattern combinations detection"""
        return {}

    def _apply_pattern_fusion(self, detected_patterns: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply pattern fusion and resolve conflicting patterns"""
        if not detected_patterns:
            return {}
        fused = {}
        sorted_pats = sorted(detected_patterns.items(), key=lambda item: item[1].get('score', 0), reverse=True)
        for name, data in sorted_pats:
            if len(fused) < 10:
                fused[name] = data
        return fused

    def _generate_legacy_llm_pattern_prompt(self, market_context: Dict, detected_patterns: Dict) -> str:
        """Generate LegacyLLM prompt for chart pattern technical analysis validation"""
        pats_str = json.dumps(detected_patterns, default=str)
        ctx_str = json.dumps(market_context, default=str)

        prompt = f"""You are an Elite Technical Analyst AI for an institutional trading desk. You specialize in validating chart patterns detected by GPU algorithms.

Market Context: {ctx_str}
GPU-Detected Geometric Patterns: {pats_str}

Task: Review the GPU-detected chart patterns (e.g. Bull Flag, Head & Shoulders, Liquidity Sweep). Evaluate if the volume profile and market context support these patterns.

Respond with EXACTLY ONE of the following validation tags at the beginning:
- [CONFIRM_PATTERN] : The detected patterns are valid and supported by technical context.
- [REJECT_PATTERN] : The patterns appear invalid, conflicted, or false breakouts.
- [WAIT] : Patterns are ambiguous or insufficient confirmation.

Follow the tag with a 1-2 sentence institutional technical analysis validation.
"""
        return prompt

    def validate_patterns_with_legacy_llm(self, market_context: Dict, detected_patterns: Dict) -> Tuple[str, str]:
        """Legacy compatibility: model pattern verdicts cannot change a signal."""
        return "UNAVAILABLE", "Pattern advisory unavailable; deterministic patterns unchanged"
    
    def _calculate_enhanced_confidence(self, base_score: float, volume_confidence: float, 
                                     pattern_type: str, pattern_name: str) -> float:
        """Calculate enhanced pattern confidence with multiple factors"""
        try:
            # Base confidence from pattern matching
            confidence = base_score
            
            # Volume confirmation boost
            if volume_confidence > 0.7:
                confidence *= 1.2
            elif volume_confidence < 0.3:
                confidence *= 0.8
            
            # Pattern-type specific adjustments
            if pattern_type == 'reversal':
                if 'head_shoulders' in pattern_name:
                    confidence *= 1.1  # Higher reliability for H&S patterns
            
            # Historical performance adjustment
            # FIX BUG #13: success_rate is initialized to 0.5 and NEVER updated in
            # _update_pattern_statistics() — this multiplier always evaluates to 1.0 (0.5+0.5).
            # To activate: wire trade outcome feedback into _update_pattern_statistics().
            # Kept here as a hook for future implementation.
            if pattern_name in self.pattern_statistics:
                historical_success = self.pattern_statistics[pattern_name].get('success_rate', 0.5)
                confidence *= (0.5 + historical_success)
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1080)
            print(f"⚠️ Enhanced confidence calculation warning: {e}")
            return base_score
    
    # ==================== ENHANCED PATTERN MEMORY AND SIGNAL GENERATION ====================
    
    async def _update_enhanced_pattern_memory(self, detected_patterns: Dict[str, Dict]):
        """Update enhanced pattern memory systems with performance tracking"""
        try:
            current_time = time.time()
            
            for pattern_name, pattern_data in detected_patterns.items():
                # Create enhanced memory entry
                memory_entry = {
                    'pattern': pattern_name,
                    'data': pattern_data,
                    'timestamp': current_time,
                    'performance': None,
                    'market_context': self._get_current_market_context()
                }
                
                # Add to short-term memory
                self.short_term_memory.append(memory_entry)
                
                # Update pattern statistics
                self._update_pattern_statistics(pattern_name, pattern_data)
            
            # Advanced archiving logic
            await self._advanced_pattern_archiving()
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1110)
            print(f"❌ Enhanced pattern memory update error: {e}")
    
    def _get_current_market_context(self) -> Dict:
        """Get current market context for pattern analysis"""
        # FIX BUG #15: analysis_times measures system processing time (seconds), NOT market volatility.
        # Using processing time as volatility proxy gives wrong labels (slow hardware = 'high volatility').
        # Renamed to 'system_load' to reflect what it actually measures.
        recent_avg_time = np.mean(list(self.analysis_times)[-10:]) if len(self.analysis_times) > 10 else 0
        return {
            'system_load': 'high' if recent_avg_time > 0.01 else 'normal',  # >10ms = high load
            'pattern_density': len(self.detected_patterns_gpu),
            'time_of_day': datetime.now().hour
        }
    
    def _update_pattern_statistics(self, pattern_name: str, pattern_data: Dict):
        """Update pattern performance statistics"""
        try:
            if pattern_name not in self.pattern_statistics:
                self.pattern_statistics[pattern_name] = {
                    'detection_count': 0,
                    'success_count': 0,
                    'success_rate': 0.5,
                    'avg_confidence': 0.0,
                    'last_detected': time.time()
                }
            
            stats = self.pattern_statistics[pattern_name]
            stats['detection_count'] += 1
            stats['avg_confidence'] = (
                (stats['avg_confidence'] * (stats['detection_count'] - 1) + pattern_data['score']) 
                / stats['detection_count']
            )
            stats['last_detected'] = time.time()
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1145)
            print(f"⚠️ Pattern statistics update warning: {e}")
    
    async def _advanced_pattern_archiving(self):
        """Advanced pattern archiving with intelligent selection"""
        try:
            # Archive based on pattern significance and recent frequency
            if len(self.short_term_memory) >= 50:
                recent_patterns = list(self.short_term_memory)[-20:]
                
                # Filter significant patterns for archiving
                significant_patterns = [
                    p for p in recent_patterns 
                    if p['data']['score'] > 0.7  # High confidence
                    or 'head_shoulders' in p['pattern']  # Important patterns
                    or 'liquidity_sweep' in p['pattern']
                ]
                
                self.long_term_memory.extend(significant_patterns)
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1165)
            print(f"⚠️ Advanced pattern archiving warning: {e}")
    
    async def _generate_enhanced_pattern_signals(self, detected_patterns: Dict[str, Dict]):
        """Generate enhanced trading signals with sophisticated logic"""
        try:
            current_signals = []
            
            for pattern_name, pattern_data in detected_patterns.items():
                signal = self._enhanced_pattern_to_signal(pattern_name, pattern_data)
                if signal:
                    current_signals.append(signal)
            
            # Apply signal filtering and prioritization
            filtered_signals = self._filter_and_prioritize_signals(current_signals)
            
            # Update signal history
            self.pattern_signals.extend(filtered_signals)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1184)
            print(f"❌ Enhanced pattern signal generation error: {e}")
    
    def _enhanced_pattern_to_signal(self, pattern_name: str, pattern_data: Dict) -> Optional[Dict]:
        """Convert pattern to enhanced trading signal with sophisticated logic"""
        try:
            base_confidence = pattern_data.get('score', 0.5)
            pattern_type = pattern_data.get('type', '')
            
            # Enhanced signal logic with multiple factors
            signal_info = self._get_pattern_signal_mapping(pattern_name, pattern_type)
            
            if signal_info:
                # Calculate enhanced confidence
                enhanced_confidence = self._calculate_signal_confidence(base_confidence, pattern_data, pattern_name)
                
                return {
                    'signal': signal_info['action'],
                    'confidence': enhanced_confidence,
                    'pattern': pattern_name,
                    'pattern_type': pattern_type,
                    'reason': signal_info['reason'],
                    'timestamp': datetime.now().isoformat(),
                    'pattern_strength': pattern_data.get('pattern_strength', 0.0),
                    'volume_confidence': pattern_data.get('volume_confidence', 0.5)
                }
            
            return None
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1213)
            print(f"⚠️ Enhanced pattern to signal conversion warning: {e}")
            return None
    
    def _get_pattern_signal_mapping(self, pattern_name: str, pattern_type: str) -> Optional[Dict]:
        """Get pattern to signal mapping with enhanced logic"""
        pattern_mappings = {
            # Reversal patterns
            'double_top': {'action': 'SELL', 'reason': 'Double top reversal pattern detected'},
            'double_bottom': {'action': 'BUY', 'reason': 'Double bottom reversal pattern detected'},
            'head_shoulders': {'action': 'SELL', 'reason': 'Head and shoulders reversal detected'},
            'inverse_head_shoulders': {'action': 'BUY', 'reason': 'Inverse head and shoulders reversal detected'},
            
            # Continuation patterns
            'flag_bullish': {'action': 'BUY', 'reason': 'Bullish flag continuation pattern'},
            'flag_bearish': {'action': 'SELL', 'reason': 'Bearish flag continuation pattern'},
            
            # Institutional patterns
            'accumulation_phase1': {'action': 'BUY', 'reason': 'Institutional accumulation detected'},
            'smart_money_accumulation': {'action': 'BUY', 'reason': 'Smart money accumulation pattern'},
            
            # Liquidity patterns
            'liquidity_sweep_bullish': {'action': 'BUY', 'reason': 'Bullish liquidity sweep detected'},
            'liquidity_sweep_bearish': {'action': 'SELL', 'reason': 'Bearish liquidity sweep detected'},
        }
        
        return pattern_mappings.get(pattern_name)
    
    def _calculate_signal_confidence(self, base_confidence: float, pattern_data: Dict, pattern_name: str) -> float:
        """Calculate enhanced signal confidence"""
        try:
            confidence = base_confidence
            
            # Volume confirmation boost
            volume_confidence = pattern_data.get('volume_confidence', 0.5)
            if volume_confidence > 0.7:
                confidence *= 1.15
            
            # Pattern strength adjustment
            pattern_strength = pattern_data.get('pattern_strength', 0.5)
            confidence *= (0.8 + pattern_strength * 0.4)
            
            # Historical performance adjustment
            if pattern_name in self.pattern_statistics:
                success_rate = self.pattern_statistics[pattern_name].get('success_rate', 0.5)
                confidence *= (0.7 + success_rate * 0.6)
            
            return min(confidence * 10, 10.0)  # Scale to 0-10
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1262)
            print(f"⚠️ Signal confidence calculation warning: {e}")
            return base_confidence * 8
    
    def _filter_and_prioritize_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter and prioritize trading signals"""
        try:
            if not signals:
                return []
            
            # Filter by minimum confidence
            min_confidence = 6.0  # Minimum confidence threshold
            filtered_signals = [s for s in signals if s['confidence'] >= min_confidence]
            
            # Prioritize by confidence and pattern type
            filtered_signals.sort(key=lambda x: (
                -x['confidence'],  # Higher confidence first
                0 if x['pattern_type'] == 'reversal' else 1,  # Reversal patterns prioritized
                0 if 'liquidity' in x['pattern'] else 1  # Liquidity patterns prioritized
            ))
            
            return filtered_signals[:5]  # Return top 5 signals
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1285)
            print(f"⚠️ Signal filtering warning: {e}")
            return signals
    
    # ==================== REAL-TIME ANALYSIS METHODS ====================
    
    def get_enhanced_pattern_analysis(self) -> Dict:
        """Get comprehensive enhanced pattern analysis results"""
        try:
            # FIX BUG #16: pattern_signals deque never purges stale signals.
            # Old signals (hours/days old) remain 'active'. Filter to last 5 minutes only.
            now = datetime.now()
            recent_signals = [
                s for s in self.pattern_signals
                if (now - datetime.fromisoformat(s['timestamp'])).total_seconds() < 300
            ]
            analysis = {
                'timestamp': now.isoformat(),
                'symbol': self.symbol,
                'detected_patterns': {},
                'active_signals': recent_signals,
                'pattern_statistics': dict(self.pattern_statistics),
                'performance_metrics': {
                    'avg_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0,
                    'total_patterns_detected': sum(self.pattern_counts.values()),
                    'short_term_memory_usage': len(self.short_term_memory),
                    'long_term_memory_usage': len(self.long_term_memory)
                },
                'system_health': {
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == 'cuda' else 0,
                    'active_pattern_templates': sum(len(templates) for templates in self.pattern_templates_gpu.values())
                }
            }
            
            # Convert GPU patterns to CPU for output
            for pattern_name, pattern_data in self.detected_patterns_gpu.items():
                analysis['detected_patterns'][pattern_name] = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in pattern_data.items()
                }

            context = self._get_current_market_context()
            vote, insight = self.validate_patterns_with_legacy_llm(context, analysis.get('detected_patterns', {}))
            analysis['legacy_llm_pattern_validation'] = vote
            analysis['legacy_llm_insight'] = insight
            
            return analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1333)
            print(f"❌ Enhanced pattern analysis retrieval error: {e}")
            return {'error': str(e)}
    
    def get_enhanced_trading_signals(self) -> List[Dict]:
        """Get current enhanced trading signals from patterns"""
        try:
            signals = list(self.pattern_signals)
            context = self._get_current_market_context()
            vote, insight = self.validate_patterns_with_legacy_llm(context, self.detected_patterns_gpu)

            annotated_signals = []
            for s in signals:
                s_copy = dict(s)
                s_copy['legacy_llm_pattern_validation'] = vote
                s_copy['legacy_llm_insight'] = insight
                # Model commentary cannot boost or penalize mathematical confidence.
                annotated_signals.append(s_copy)

            return annotated_signals
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1353)
            print(f"❌ Enhanced trading signal retrieval error: {e}")
            return []
    
    # ==================== SYSTEM MANAGEMENT ====================
    
    async def enhanced_shutdown(self):
        """Safe shutdown of enhanced pattern recognition engine"""
        print("🛑 Shutting down Enhanced GPU Pattern Recognition Engine...")
        self.is_running = False
        
        self.executor.shutdown(wait=True)
        self.gpu_manager.cleanup_pattern_memory()
        
        # Save pattern statistics
        self._save_pattern_statistics()
        
        print("✅ Enhanced Pattern Recognition Engine shutdown complete")
    
    def _save_pattern_statistics(self):
        """Save pattern statistics to file"""
        try:
            stats_file = Path("/tmp/pattern_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(dict(self.pattern_statistics), f, indent=2)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1378)
            print(f"⚠️ Pattern statistics save warning: {e}")
    
    def get_enhanced_system_status(self) -> Dict:
        """Get enhanced system status report"""
        return {
            'is_running': self.is_running,
            'pattern_detection': {
                'active_patterns': len(self.detected_patterns_gpu),
                'recent_signals': len(self.pattern_signals),
                'templates_loaded': sum(len(templates) for templates in self.pattern_templates_gpu.values()),
                'pattern_categories': list(self.pattern_templates_gpu.keys())
            },
            'memory_systems': {
                'short_term_capacity': self.short_term_memory.maxlen,
                'long_term_capacity': self.long_term_memory.maxlen,
                'short_term_usage': len(self.short_term_memory),
                'long_term_usage': len(self.long_term_memory),
                'pattern_statistics_count': len(self.pattern_statistics)
            },
            'performance_metrics': {
                'avg_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0,
                'recent_analysis_times': list(self.analysis_times)[-5:],
                'total_analyses_performed': len(self.analysis_times)
            },
            'gpu_memory': {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == 'cuda' else 0,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024 if self.device.type == 'cuda' else 0
            }
        }

# ==================== ENHANCED INTEGRATION SYSTEM ====================

class EnhancedPatternRecognitionSystem:
    """Enhanced pattern recognition system with advanced GPU acceleration"""
    
    def __init__(self, live_data_engine=None, orderflow_engine=None):
        self.live_data_engine = live_data_engine
        self.orderflow_engine = orderflow_engine
        self.pattern_engine = EnhancedGPUPatternRecognitionEngine()
        self.is_active = False
        self.analysis_interval = 0.5  # 500ms analysis cycle
    
    async def start_enhanced_pattern_analysis(self):
        """Start enhanced real-time pattern analysis"""
        print("🚀 Starting Enhanced GPU Pattern Analysis System...")
        self.is_active = True
        
        try:
            while self.is_active:
                start_time = time.time()
                
                # Get live data from Part 7 integration
                market_data = await self._get_live_market_data()
                
                if market_data and 'candle_data' in market_data:
                    # Perform enhanced pattern analysis
                    analysis_result = await self.pattern_engine.analyze_enhanced_candle_sequences(
                        market_data['candle_data']
                    )
                    
                    # Log performance
                    if analysis_result.get('success'):
                        print(f"✅ Pattern Analysis: {analysis_result['patterns_detected']} patterns in {analysis_result['analysis_time']:.3f}s")
                
                # Maintain analysis interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.analysis_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1448)
            print(f"❌ Enhanced pattern analysis error: {e}")
            self.is_active = False
    
    async def _get_live_market_data(self) -> Optional[Dict]:
        """Get live market data from Part 7 integration"""
        try:
            if self.live_data_engine and hasattr(self.live_data_engine, 'get_candle_data'):
                # Get 1-minute candle data for pattern analysis
                candle_data = self.live_data_engine.get_candle_data('1m', 200)
                return {'candle_data': candle_data} if candle_data else None
            else:
                # Fallback: Generate sample data for testing
                return self._generate_sample_market_data()
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1462)
            print(f"⚠️ Live market data fetch warning: {e}")
            return None
    
    def _generate_sample_market_data(self) -> Dict:
        """Generate sample market data for testing"""
        length = 200
        return {
            'candle_data': {
                'open': np.random.normal(50000, 1000, length).astype(np.float32),
                'high': np.random.normal(50200, 1000, length).astype(np.float32),
                'low': np.random.normal(49800, 1000, length).astype(np.float32),
                'close': np.random.normal(50100, 1000, length).astype(np.float32),
                'volume': np.random.normal(1000, 200, length).astype(np.float32)
            }
        }
    
    def get_enhanced_combined_analysis(self) -> Dict:
        """Get enhanced combined pattern and market analysis"""
        try:
            pattern_analysis = self.pattern_engine.get_enhanced_pattern_analysis()
            
            combined_analysis = {
                'timestamp': datetime.now().isoformat(),
                'pattern_analysis': pattern_analysis,
                'trading_signals': self.pattern_engine.get_enhanced_trading_signals(),
                'system_status': self.pattern_engine.get_enhanced_system_status()
            }
            
            # Add orderflow analysis if available
            if self.orderflow_engine:
                orderflow_analysis = self.orderflow_engine.get_orderflow_analysis()
                combined_analysis['orderflow_analysis'] = orderflow_analysis
            
            return combined_analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1498)
            print(f"❌ Enhanced combined analysis error: {e}")
            return {'error': str(e)}
    
    async def stop_enhanced_pattern_analysis(self):
        """Stop enhanced pattern analysis"""
        print("🛑 Stopping Enhanced Pattern Analysis...")
        self.is_active = False
        await self.pattern_engine.enhanced_shutdown()

# ==================== ENHANCED LINUX OPTIMIZATION ====================

def setup_enhanced_linux_pattern_environment():
    """Setup enhanced Linux-optimized environment for pattern recognition"""
    # CPU optimization for i5 4-core
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    
    # Enhanced GPU memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    
    # Async performance optimizations
    os.environ['PYTHONASYNCIODEBUG'] = '0'
    
    # Pattern recognition specific optimizations
    os.environ['PATTERN_ANALYSIS_BATCH_SIZE'] = '50'
    
    print("✅ Enhanced Linux environment optimized for advanced pattern recognition")

# ==================== MAIN EXECUTION ====================

async def main():
    """Main execution function for enhanced pattern recognition"""
    setup_enhanced_linux_pattern_environment()
    
    # Initialize enhanced pattern recognition system
    pattern_system = EnhancedPatternRecognitionSystem()
    
    try:
        # Start enhanced pattern analysis
        await pattern_system.start_enhanced_pattern_analysis()
        
    except KeyboardInterrupt:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1541)
        print("\n🛑 Enhanced pattern analysis stopped by user")
    except Exception as e:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1543)
        print(f"❌ Enhanced pattern system error: {e}")
    finally:
        await pattern_system.stop_enhanced_pattern_analysis()

if __name__ == "__main__":
    # Run the enhanced pattern recognition system
    asyncio.run(main())