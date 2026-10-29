try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ---- Helpers inserted for Part-11 Fix ----
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

# ==================== GPU-OPTIMIZED PART-11: UNIFIED CONFIDENCE ENGINE ====================
# DEEPSEEK AI-POWERED REWRITE - FULL HARDWARE OPTIMIZATION
# LINUX UBUNTU + GTX 1650 CUDA + i5 10th Gen OPTIMIZED

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# BUG FIX #1: cupy optional — no crash if not installed
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 160)
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
import re
from typing import Dict, List, Tuple, Any, Optional, Union




# ==================== GPU CONFIDENCE MEMORY MANAGER ====================

class ConfidenceGPUMemoryManager:
    """GTX 1650 4GB VRAM Optimized Memory Manager for Confidence Engine"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram = 3.0 * 1024 * 1024 * 1024  # 3.0GB safety limit
        self.max_ram = 4.0 * 1024 * 1024 * 1024   # 4GB RAM limit
        
        # SSD cache for confidence history
        self.cache_dir = Path("/tmp/confidence_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.75)  # 75% of 4GB
            print(f"OK Confidence GPU Memory: {_safe_get_device_name(self.device)}")
    
    def allocate_confidence_tensor(self, data, name, persistent=True):
        """GPU tensor allocation optimized for confidence data"""
        try:
            # BUG FIX #2: Check device type FIRST before calling cuda memory functions
            if self.device.type == 'cpu':
                return torch.tensor(data, dtype=torch.float32)
            
            if torch.cuda.memory_allocated() > self.max_vram:
                return torch.tensor(data, dtype=torch.float32)
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            if persistent and tensor.numel() < 5000:
                return tensor
            else:
                return self._create_memory_mapped_tensor(data, name)
                
        except RuntimeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 214)
            print(f"WARNING Confidence GPU allocation failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def _create_memory_mapped_tensor(self, data, name):
        """Create memory-mapped tensor for large confidence datasets"""
        try:
            # BUG FIX #3: data could be list — convert to numpy array first
            data_array = np.array(data, dtype='float32')
            # BUG FIX #15: Use fixed filename per 'name' to avoid disk fill-up
            cache_file = self.cache_dir / f"{name}.dat"
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_array.shape)
            mmap[:] = data_array
            return torch.from_numpy(np.array(mmap))  # copy to avoid mmap lifecycle issues
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 228)
            print(f"WARNING Confidence memory mapping failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def cleanup_confidence_memory(self):
        """Aggressive cleanup for confidence operations"""
        # BUG FIX #4: Only call CUDA functions when actually on CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

# BUG FIX #5 #6 #7: torch.cuda.device(cpu_device) crashes — use this guard everywhere
def _cuda_guard(device):
    class _NullContext:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    if device.type == 'cuda':
        return torch.cuda.device(device)
    return _NullContext()

# ==================== GPU-ACCELERATED UNIFIED CONFIDENCE ENGINE ====================

class GPUUnifiedConfidenceEngine:
    """
    INSTITUTIONAL-GRADE UNIFIED CONFIDENCE ENGINE
    GTX 1650 CUDA Optimized for Real-Time Signal Validation
    """
    
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.gpu_manager = ConfidenceGPUMemoryManager()
        self.device = self.gpu_manager.device
        
        # GPU-optimized confidence matrices
        self.confidence_matrix_gpu = None
        self.signal_scores_gpu = None
        self.final_confidence_gpu = None
        
        # Multi-source scoring configuration
        self.scoring_weights = {
            'zone_reaction': 1.4,
            'candle_psychology': 1.2,
            'trend_direction': 1.3,
            'volume_spike': 1.1,
            'breakout_failed': 1.2,
            'pattern_recognition': 1.3,
            'ai_learning': 1.5,
            'trend_reasoning': 1.4
        }
        
        # Confidence processing parameters
        self.confidence_config = {
            'smoothing_window': 10,
            'noise_threshold': 0.15,
            'min_confidence': 6.0,
            'max_confidence': 9.8,
            'volatility_adjustment': True,
            'expiry_adjustment': True,
            'denoising_enabled': True
        }
        
        # Performance tracking
        self.confidence_history = deque(maxlen=1000)
        self.signal_validation_log = deque(maxlen=500)
        self.expiry_performance = defaultdict(lambda: deque(maxlen=200))
        
        # Threading control
        self.data_lock = threading.RLock()
        self.is_active = True
        self.executor = ThreadPoolExecutor(max_workers=4)  # i5 4-core optimized
        
        # LegacyLLM Confidence Analyst Cooldown setup
        self.last_legacy_llm_time = 0
        self.legacy_llm_cooldown = 30  # seconds
        self.last_legacy_llm_adjustment = 0
        self.last_legacy_llm_insight = "Mathematical confidence verified by AI Risk Analyst."

        # Initialize GPU tensors
        self._initialize_confidence_tensors()
        
        print("ACCELERATED GPU Unified Confidence Engine Initialized - Linux Optimized")
    
    def _initialize_confidence_tensors(self):
        """Initialize GPU-optimized confidence tensors"""
        try:
            # Confidence matrix: [signals x features]
            self.confidence_matrix_gpu = torch.zeros((8, 10), device=self.device, dtype=torch.float32)
            
            # Signal scores vector
            self.signal_scores_gpu = torch.zeros(8, device=self.device, dtype=torch.float32)
            
            # Final confidence tensor
            self.final_confidence_gpu = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 322)
            print(f"WARNING Confidence tensor initialization warning: {e}")
    
    # ==================== GPU-ACCELERATED CONFIDENCE PROCESSING ====================
    
    async def compute_unified_confidence(self, signal_data: dict):
        """
        Compute unified confidence score with GPU acceleration
        """
        try:
            # Convert signal data to GPU tensors
            signal_tensors = await self._prepare_signal_tensors_gpu(signal_data)
            
            if signal_tensors is None:
                return 0.0, "Signal data preparation failed"
            
            # GPU-accelerated multi-source scoring
            confidence_matrix = await self._build_confidence_matrix_gpu(signal_tensors)
            
            # Apply confidence smoothing and noise reduction
            smoothed_matrix = await self._apply_confidence_smoothing_gpu(confidence_matrix)
            
            # Fuse multi-source scores
            fused_scores = await self._fuse_confidence_scores_gpu(smoothed_matrix)
            
            # Apply final adjustments
            final_confidence = await self._apply_final_adjustments_gpu(fused_scores, signal_data)
            
            # Apply LegacyLLM AI Chief Risk Analyst Confidence Adjustment
            final_confidence, legacy_llm_adj, legacy_llm_insight = self.adjust_confidence_with_legacy_llm(final_confidence, signal_data)

            # Validate signal
            validation_result = await self._validate_signal_gpu(final_confidence, signal_data)
            
            if hasattr(self, 'bus') and self.bus:
                msg = f"Confidence Engine: Final Score = {final_confidence:.1f}%. {validation_result} | Model adj unavailable: {legacy_llm_adj:+d}"
                self.bus.publish('THOUGHTS', 'Part11_Confidence', msg)
                
            return final_confidence, f"{validation_result} | Model adj unavailable: {legacy_llm_adj:+d}"
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 362)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part11_Confidence', e, context='compute_unified_confidence')
            else:
                print(f"ERROR Unified confidence computation error: {e}")
            return 0.0, f"Confidence computation error: {str(e)}"
    
    async def _prepare_signal_tensors_gpu(self, signal_data):
        """Prepare signal data for GPU processing"""
        try:
            tensors = {}
            
            # Zone reaction scoring
            if 'zone_reaction' in signal_data:
                zone_data = signal_data['zone_reaction']
                zone_features = self._extract_zone_features_gpu(zone_data)
                tensors['zone_reaction'] = zone_features
            
            # Candle psychology scoring
            if 'candle_psychology' in signal_data:
                psych_data = signal_data['candle_psychology']
                psych_features = self._extract_psychology_features_gpu(psych_data)
                tensors['candle_psychology'] = psych_features
            
            # Trend direction scoring
            if 'trend_direction' in signal_data:
                trend_data = signal_data['trend_direction']
                trend_features = self._extract_trend_features_gpu(trend_data)
                tensors['trend_direction'] = trend_features
            
            # Volume spike scoring
            if 'volume_spike' in signal_data:
                volume_data = signal_data['volume_spike']
                volume_features = self._extract_volume_features_gpu(volume_data)
                tensors['volume_spike'] = volume_features
            
            # Breakout/failed breakout scoring
            if 'breakout_analysis' in signal_data:
                breakout_data = signal_data['breakout_analysis']
                breakout_features = self._extract_breakout_features_gpu(breakout_data)
                tensors['breakout_failed'] = breakout_features
            
            # Pattern recognition scoring
            if 'pattern_signals' in signal_data:
                pattern_data = signal_data['pattern_signals']
                pattern_features = self._extract_pattern_features_gpu(pattern_data)
                tensors['pattern_recognition'] = pattern_features
            
            # AI learning scoring
            if 'ai_learning' in signal_data:
                ai_data = signal_data['ai_learning']
                ai_features = self._extract_ai_features_gpu(ai_data)
                tensors['ai_learning'] = ai_features
            
            # Trend reasoning scoring
            if 'trend_reasoning' in signal_data:
                reasoning_data = signal_data['trend_reasoning']
                reasoning_features = self._extract_reasoning_features_gpu(reasoning_data)
                tensors['trend_reasoning'] = reasoning_features
            
            return tensors
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 424)
            print(f"ERROR Signal tensor preparation error: {e}")
            return None
    
    def _extract_zone_features_gpu(self, zone_data):
        """GPU-accelerated zone reaction feature extraction"""
        try:
            features = [
                zone_data.get('zone_strength', 0),
                zone_data.get('reaction_intensity', 0),
                zone_data.get('multiple_timeframes', 0),
                zone_data.get('volume_confirmation', 0),
                zone_data.get('price_alignment', 0)
            ]
            # Pad to 10 features
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'zone_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 441)
            print(f"WARNING Zone feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_psychology_features_gpu(self, psych_data):
        """GPU-accelerated candle psychology feature extraction"""
        try:
            features = [
                psych_data.get('body_strength', 0),
                psych_data.get('wick_balance', 0),
                psych_data.get('close_position', 0),
                psych_data.get('sentiment_score', 0),
                psych_data.get('momentum_alignment', 0),
                psych_data.get('rejection_presence', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'psych_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 458)
            print(f"WARNING Psychology feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_trend_features_gpu(self, trend_data):
        """GPU-accelerated trend direction feature extraction"""
        try:
            features = [
                trend_data.get('trend_strength', 0),
                trend_data.get('multi_timeframe_alignment', 0),
                trend_data.get('momentum_consistency', 0),
                trend_data.get('swing_structure', 0),
                trend_data.get('trend_duration', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'trend_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 474)
            print(f"WARNING Trend feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_volume_features_gpu(self, volume_data):
        """GPU-accelerated volume spike feature extraction"""
        try:
            features = [
                volume_data.get('spike_intensity', 0),
                volume_data.get('volume_trend', 0),
                volume_data.get('buy_sell_ratio', 0),
                volume_data.get('volume_clusters', 0),
                volume_data.get('institutional_presence', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'volume_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 490)
            print(f"WARNING Volume feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_breakout_features_gpu(self, breakout_data):
        """GPU-accelerated breakout feature extraction"""
        try:
            features = [
                breakout_data.get('breakout_strength', 0),
                breakout_data.get('volume_confirmation', 0),
                breakout_data.get('failed_breakout_risk', 0),
                breakout_data.get('support_resistance_quality', 0),
                breakout_data.get('momentum_persistence', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'breakout_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 506)
            print(f"WARNING Breakout feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_pattern_features_gpu(self, pattern_data):
        """GPU-accelerated pattern recognition feature extraction"""
        try:
            features = [
                pattern_data.get('pattern_strength', 0),
                pattern_data.get('pattern_complexity', 0),
                pattern_data.get('historical_accuracy', 0),
                pattern_data.get('timeframe_confirmation', 0),
                pattern_data.get('volume_alignment', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'pattern_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 522)
            print(f"WARNING Pattern feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_ai_features_gpu(self, ai_data):
        """GPU-accelerated AI learning feature extraction"""
        try:
            features = [
                ai_data.get('learning_confidence', 0),
                ai_data.get('historical_performance', 0),
                ai_data.get('pattern_recognition', 0),
                ai_data.get('market_regime_adaptation', 0),
                ai_data.get('risk_adjustment', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'ai_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 538)
            print(f"WARNING AI feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    def _extract_reasoning_features_gpu(self, reasoning_data):
        """GPU-accelerated trend reasoning feature extraction"""
        try:
            features = [
                reasoning_data.get('reasoning_depth', 0),
                reasoning_data.get('multi_factor_alignment', 0),
                reasoning_data.get('probabilistic_confidence', 0),
                reasoning_data.get('market_context', 0),
                reasoning_data.get('risk_assessment', 0)
            ]
            features.extend([0] * (10 - len(features)))
            return self.gpu_manager.allocate_confidence_tensor(features, 'reasoning_features')
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 554)
            print(f"WARNING Reasoning feature extraction warning: {e}")
            return torch.zeros(10, device=self.device)
    
    async def _build_confidence_matrix_gpu(self, signal_tensors):
        """GPU-accelerated confidence matrix assembly"""
        try:
            # BUG FIX #5: Use _cuda_guard instead of torch.cuda.device()
            with _cuda_guard(self.device):
                matrix = torch.zeros((8, 10), device=self.device, dtype=torch.float32)
                
                signal_mapping = {
                    'zone_reaction': 0,
                    'candle_psychology': 1,
                    'trend_direction': 2,
                    'volume_spike': 3,
                    'breakout_failed': 4,
                    'pattern_recognition': 5,
                    'ai_learning': 6,
                    'trend_reasoning': 7
                }
                
                for signal_type, tensor in signal_tensors.items():
                    if signal_type in signal_mapping:
                        row_idx = signal_mapping[signal_type]
                        # BUG FIX #12 #13: Move tensor to correct device before assignment
                        tensor = tensor.to(self.device)
                        if len(tensor) == 10:
                            matrix[row_idx] = tensor
                        else:
                            padded_tensor = torch.zeros(10, device=self.device)
                            min_len = min(len(tensor), 10)
                            padded_tensor[:min_len] = tensor[:min_len]
                            matrix[row_idx] = padded_tensor
                
                return matrix
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 591)
            print(f"ERROR Confidence matrix build error: {e}")
            return torch.zeros((8, 10), device=self.device)
    
    async def _apply_confidence_smoothing_gpu(self, confidence_matrix):
        """GPU-accelerated confidence smoothing and noise reduction"""
        try:
            # BUG FIX #6: Use _cuda_guard
            with _cuda_guard(self.device):
                if self.confidence_config['denoising_enabled']:
                    smoothed_matrix = torch.zeros_like(confidence_matrix)
                    win = self.confidence_config['smoothing_window']
                    
                    for i in range(confidence_matrix.shape[0]):
                        signal_row = confidence_matrix[i]
                        row_len = len(signal_row)
                        
                        if row_len >= win:
                            windows = signal_row.unfold(0, win, 1)
                            smoothed_values = torch.mean(windows, dim=1)  # shape: (row_len - win + 1,)
                            
                            # BUG FIX #8: Robust padding — replicate edges, handle odd pad_size
                            pad_total = row_len - len(smoothed_values)
                            pad_left  = pad_total // 2
                            pad_right = pad_total - pad_left
                            
                            parts = []
                            if pad_left > 0:
                                parts.append(smoothed_values[:1].expand(pad_left))
                            parts.append(smoothed_values)
                            if pad_right > 0:
                                parts.append(smoothed_values[-1:].expand(pad_right))
                            
                            padded = torch.cat(parts)
                            smoothed_matrix[i] = padded[:row_len]
                        else:
                            smoothed_matrix[i] = signal_row
                    
                    noise_mask = torch.abs(smoothed_matrix) < self.confidence_config['noise_threshold']
                    smoothed_matrix[noise_mask] = 0.0
                    
                    return smoothed_matrix
                else:
                    return confidence_matrix
                    
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 636)
            print(f"ERROR Confidence smoothing error: {e}")
            return confidence_matrix
    
    async def _fuse_confidence_scores_gpu(self, smoothed_matrix):
        """Advanced Math: Bayesian Belief Network / Dempster-Shafer theory for multi-source fusion"""
        try:
            # BUG FIX #7: Use _cuda_guard
            with _cuda_guard(self.device):
                signal_scores = torch.zeros(8, device=self.device, dtype=torch.float32)
                
                # BUG FIX #11: Store raw means first, apply weights in fusion step
                # Original code multiplied weights INTO signal_scores then divided by total_weight
                # = double application. Now: raw_scores * weights / total_weight
                weight_keys = ['zone_reaction', 'candle_psychology', 'trend_direction', 
                               'volume_spike', 'breakout_failed', 'pattern_recognition',
                               'ai_learning', 'trend_reasoning']
                
                for i, key in enumerate(weight_keys):
                    raw_mean = torch.mean(smoothed_matrix[i])
                    signal_scores[i] = raw_mean  # raw score, no weight yet
                
                # Apply weights and normalize correctly
                weight_vals = torch.tensor(
                    [self.scoring_weights[k] for k in weight_keys],
                    device=self.device, dtype=torch.float32
                )
                
                # Advanced Math: Dempster-Shafer Evidence Combination Proxy
                # Instead of simple weighted average, we calculate joint belief mass
                # Convert raw scores to probabilities [0, 1]
                prob_scores = torch.clamp(signal_scores, 0.0, 1.0)

                # Calculate joint belief (product of probabilities, weighted)
                # We use a log-sum-exp trick proxy to combine independent evidence
                weighted_log_probs = weight_vals * torch.log(prob_scores + 1e-8)
                joint_evidence = torch.exp(torch.sum(weighted_log_probs) / torch.sum(weight_vals))

                # Combine with standard weighted average for stability
                total_weight = weight_vals.sum()
                fused_score = torch.sum(signal_scores * weight_vals) / (total_weight + 1e-8)
                
                # Blended final score (Dempster-Shafer influence + standard average)
                final_fused = (0.7 * fused_score) + (0.3 * joint_evidence)

                # Scale to 0-10 range
                final_fused = torch.clamp(final_fused * 10, 0.0, 10.0)
                
                return final_fused.item()
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 686)
            print(f"ERROR Score fusion error: {e}")
            return 0.0
    
    async def _apply_final_adjustments_gpu(self, fused_score, signal_data):
        """GPU-accelerated final confidence adjustments"""
        try:
            adjusted_score = fused_score
            
            # Volatility adjustment
            if self.confidence_config['volatility_adjustment']:
                volatility_factor = await self._calculate_volatility_adjustment_gpu(signal_data)
                adjusted_score *= volatility_factor
            
            # Expiry adjustment
            if self.confidence_config['expiry_adjustment']:
                expiry_factor = await self._calculate_expiry_adjustment_gpu(signal_data)
                adjusted_score *= expiry_factor
            
            # BUG FIX #9: Don't force floor to min_confidence — weak signals should stay weak
            # Only clamp to valid range [0, max_confidence], let caller decide minimum
            adjusted_score = max(0.0, min(adjusted_score, self.confidence_config['max_confidence']))
            
            return adjusted_score
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 711)
            print(f"ERROR Final adjustments error: {e}")
            return fused_score
    
    async def _calculate_volatility_adjustment_gpu(self, signal_data):
        """GPU-accelerated volatility adjustment calculation"""
        try:
            if 'market_analysis' not in signal_data:
                return 1.0
            
            market_data = signal_data['market_analysis']
            volatility = market_data.get('volatility', 0.01)
            
            # Higher volatility reduces confidence
            volatility_factor = 1.0 / (1.0 + volatility * 5)
            
            return max(0.5, min(volatility_factor, 1.2))
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 729)
            print(f"WARNING Volatility adjustment warning: {e}")
            return 1.0
    
    async def _calculate_expiry_adjustment_gpu(self, signal_data):
        """GPU-accelerated expiry adjustment calculation"""
        try:
            if 'expiry_analysis' not in signal_data:
                return 1.0
            
            expiry_data = signal_data['expiry_analysis']
            suggested_expiry = expiry_data.get('suggested_expiry_minutes', 5)
            
            # Shorter expiries get confidence boost, longer expiries get reduction
            if suggested_expiry <= 3:
                return 1.1  # +10% for very short expiries
            elif suggested_expiry <= 5:
                return 1.05  # +5% for short expiries
            elif suggested_expiry <= 10:
                return 1.0  # No adjustment for medium expiries
            else:
                return 0.9  # -10% for longer expiries
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 752)
            print(f"WARNING Expiry adjustment warning: {e}")
            return 1.0
    
    async def _validate_signal_gpu(self, final_confidence, signal_data):
        """GPU-accelerated final signal validation"""
        try:
            validation_checks = []
            
            # Minimum confidence check
            if final_confidence >= self.confidence_config['min_confidence']:
                validation_checks.append("  Meets minimum confidence")
            else:
                validation_checks.append("  Below minimum confidence threshold")
            
            # Multi-source confirmation check
            signal_sources = len([k for k in signal_data.keys() if k in self.scoring_weights])
            if signal_sources >= 3:
                validation_checks.append("  Multiple signal sources confirmed")
            else:
                validation_checks.append("  Insufficient signal sources")
            
            # Market regime compatibility
            if 'market_regime' in signal_data:
                regime = signal_data['market_regime']
                if regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING']:
                    validation_checks.append("  Compatible market regime")
                else:
                    validation_checks.append("  Unfavorable market regime")
            
            # Volume confirmation
            if 'volume_analysis' in signal_data:
                volume_data = signal_data['volume_analysis']
                if volume_data.get('volume_spike', False) or volume_data.get('above_average_volume', False):
                    validation_checks.append("  Volume confirmation present")
                else:
                    validation_checks.append("  Volume confirmation weak")
            
            return " | ".join(validation_checks)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 792)
            print(f"ERROR Signal validation error: {e}")
            return "Signal validation failed"

    def _generate_legacy_llm_confidence_prompt(self, raw_confidence: float, signal_data: Dict) -> str:
        """Generate LegacyLLM prompt for Chief Risk & Confidence Analyst score adjustment"""
        breakdown = {k: signal_data[k] for k in self.scoring_weights.keys() if k in signal_data}
        b_str = json.dumps(breakdown, default=str)

        prompt = f"""You are the Chief Risk & Confidence Analyst for a quantitative trading fund. You evaluate the mathematical confidence score of a trading signal.

Mathematical Confidence Score: {raw_confidence:.1f}
Signal Components Breakdown: {b_str}

Task: Review the individual confidence components (Zone, Trend, Volatility, Volume, Patterns). 
- If all components align perfectly, apply a positive adjustment (+1 to +20).
- If there are conflicting signals (e.g. Trend Bullish but Volume Bearish), apply a negative penalty (-1 to -20).
- If signals are neutral/balanced, apply 0 adjustment.

Respond with EXACTLY ONE adjustment tag in format [ADJUST: X] (where X is an integer between -20 and +20) at the start of your response, e.g. [ADJUST: +10] or [ADJUST: -15].

Follow the tag with a 1-sentence risk analyst justification.
"""
        return prompt

    def adjust_confidence_with_legacy_llm(self, raw_confidence: float, signal_data: Dict) -> Tuple[float, int, str]:
        """Legacy hook: no stale adjustment and no model-derived confidence."""
        return raw_confidence, 0, "Model adjustment unavailable; mathematical score unchanged"
    
    # ==================== REAL-TIME CONFIDENCE MONITORING ====================
    
    async def update_confidence_history(self, confidence_score, signal_type, outcome=None, expiry_minutes=None):
        """Update confidence history with performance tracking"""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence_score,
                'signal_type': signal_type,
                'outcome': outcome,
                'market_regime': 'UNKNOWN',
                # BUG FIX #10: Actually add expiry_minutes to history_entry
                'expiry_minutes': expiry_minutes
            }
            
            self.confidence_history.append(history_entry)
            
            # BUG FIX #10: Now expiry_minutes IS in history_entry — this check works correctly
            if outcome is not None and history_entry.get('expiry_minutes') is not None:
                expiry = history_entry['expiry_minutes']
                self.expiry_performance[expiry].append({
                    'confidence': confidence_score,
                    'outcome': outcome,
                    'timestamp': history_entry['timestamp']
                })
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 847)
            print(f"ERROR Confidence history update error: {e}")
    
    def get_confidence_analysis(self):
        """Get comprehensive confidence analysis"""
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {
                    'average_confidence': 0.0,
                    'win_rate': 0.0,
                    'recent_trend': 'STABLE'
                },
                'performance_by_signal_type': {},
                'expiry_performance': {}
            }
            
            # Calculate average confidence
            if self.confidence_history:
                recent_confidences = [entry['confidence'] for entry in list(self.confidence_history)[-100:]]
                analysis['current_metrics']['average_confidence'] = np.mean(recent_confidences)
                
                # Calculate win rate for trades with outcomes
                trades_with_outcomes = [entry for entry in self.confidence_history if entry['outcome'] is not None]
                if trades_with_outcomes:
                    wins = sum(1 for entry in trades_with_outcomes if entry['outcome'] == 'WIN')
                    analysis['current_metrics']['win_rate'] = wins / len(trades_with_outcomes)
            
            # BUG FIX #14: Use .get() to avoid KeyError if signal_type missing
            signal_types = set(entry.get('signal_type') for entry in self.confidence_history if entry.get('signal_type'))
            for signal_type in signal_types:
                type_entries = [entry for entry in self.confidence_history if entry['signal_type'] == signal_type]
                if type_entries:
                    avg_confidence = np.mean([entry['confidence'] for entry in type_entries])
                    analysis['performance_by_signal_type'][signal_type] = {
                        'average_confidence': avg_confidence,
                        'trade_count': len(type_entries)
                    }
            
            # Expiry performance
            for expiry, performances in self.expiry_performance.items():
                if performances:
                    expiry_wins = sum(1 for p in performances if p['outcome'] == 'WIN')
                    win_rate = expiry_wins / len(performances)
                    analysis['expiry_performance'][expiry] = {
                        'win_rate': win_rate,
                        'total_trades': len(performances),
                        'average_confidence': np.mean([p['confidence'] for p in performances])
                    }
            
            analysis['legacy_llm_confidence_adjustment'] = self.last_legacy_llm_adjustment
            analysis['legacy_llm_insight'] = self.last_legacy_llm_insight

            return analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 902)
            print(f"ERROR Confidence analysis error: {e}")
            return {'error': str(e)}
    
    # ==================== SYSTEM MANAGEMENT ====================
    
    async def shutdown(self):
        """Safe shutdown of confidence engine"""
        print("  Shutting down GPU Confidence Engine...")
        self.is_active = False
        
        self.executor.shutdown(wait=False)
        self.gpu_manager.cleanup_confidence_memory()
        
        print("OK Confidence Engine shutdown complete")
    
    def get_system_status(self):
        """Get system status report"""
        return {
            'is_active': self.is_active,
            'confidence_metrics': {
                'history_size': len(self.confidence_history),
                'validation_log_size': len(self.signal_validation_log),
                'expiry_tracking': len(self.expiry_performance)
            },
            'gpu_memory': {
                'allocated': torch.cuda.memory_allocated() / 1024 / 1024 if self.device.type == 'cuda' else 0,
                'cached': torch.cuda.memory_reserved() / 1024 / 1024 if self.device.type == 'cuda' else 0
            },
            'scoring_weights': self.scoring_weights,
            'configuration': self.confidence_config
        }

# ==================== INTEGRATION WITH TRADING SYSTEM ====================

class EnhancedConfidenceSystem:
    """Enhanced confidence system with GPU acceleration"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.confidence_engine = GPUUnifiedConfidenceEngine()
        self.is_active = False
    
    async def start_confidence_monitoring(self):
        """Start confidence monitoring system"""
        print("  Starting GPU Confidence Monitoring...")
        self.is_active = True
        
        try:
            # Continuous confidence monitoring loop
            while self.is_active:
                await asyncio.sleep(0.5)  # 500ms monitoring cycle
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 955)
            print(f"ERROR Confidence monitoring error: {e}")
            self.is_active = False
    
    async def compute_signal_confidence(self, signal_data: dict):
        """Compute comprehensive signal confidence"""
        try:
            confidence, validation = await self.confidence_engine.compute_unified_confidence(signal_data)
            
            # Log the validation result
            validation_entry = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'validation': validation,
                'signal_type': signal_data.get('signal_type', 'UNKNOWN')
            }
            self.confidence_engine.signal_validation_log.append(validation_entry)
            
            return {
                'confidence': confidence,
                'validation': validation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 979)
            print(f"ERROR Signal confidence computation error: {e}")
            return {'confidence': 0.0, 'validation': f"Error: {str(e)}", 'timestamp': datetime.now().isoformat()}
    
    def get_confidence_insights(self):
        """Get confidence system insights"""
        try:
            return self.confidence_engine.get_confidence_analysis()
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 987)
            print(f"ERROR Confidence insights error: {e}")
            return {'error': str(e)}
    
    async def stop_confidence_monitoring(self):
        """Stop confidence monitoring"""
        print("  Stopping Confidence Monitoring...")
        self.is_active = False
        await self.confidence_engine.shutdown()


# ==================== GPU-OPTIMIZED PART-11: SIGNAL FUSION ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p11_signal_fusion_engine', 'analyze')
class SignalFusionEngineGPU:
    """
    INSTITUTIONAL QUANTITATIVE SIGNAL FUSION ENGINE (PART 11)
    GTX 1650 CUDA & Vectorized Consensus Architecture
    - Dynamic Part Weighting Matrix based on engine statistical edge
    - Multi-Engine Quorum Requirement (Minimum 3 active agreeing engines)
    - Minimum Weighted Support Threshold (>= 3.5 total active weight)
    - Strong 65%+ Consensus Ratio
    - Strict S/R Zone Veto Enforcement (Blocks long resistance / short support)
    - Contradictory Anchor Dissent Veto (Blocks trades when Trend/Structure opposes)
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Engine reliability weights based on quant performance & statistical edge
        self.weights = {
            'part1_breakout': 1.1,      # Volatility breakout
            'part2_zone': 1.3,          # Supply & Demand Zone reaction
            'part3_psychology': 1.0,    # Candlestick psychology & patterns
            'part4_volume': 1.3,        # Volume profile (POC / Value Area)
            'part5_ml': 1.2,            # Sharpe drift & regression t-statistic
            'part6_trend': 1.5,         # Trend Engine (ADX + EMA spread) - Anchor
            'part7_volatility': 1.1,    # Squeeze & Bollinger Expansion
            'part8_structure': 1.4,     # Fractal Market Structure (BOS/CHoCH) - Anchor
            'part9_orderflow': 1.4,     # Order Flow CVD Delta Imbalance - Anchor
            'part10_candlestats': 1.0,  # Candle Body Acceleration & Dominance
        }
        
        self.min_active_engines = 3
        self.min_weighted_score = 3.5
        self.min_consensus_pct = 0.65
        
        self._weight_tensor = None
        if TORCH_AVAILABLE:
            try:
                weight_list = [self.weights[k] for k in sorted(self.weights.keys())]
                self._weight_tensor = torch.tensor(weight_list, dtype=torch.float32, device=self.device)
                self._sorted_keys = sorted(self.weights.keys())
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1038)
                self._weight_tensor = None

    def analyze(self, part_results: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """
        Fuses outputs of all active engines into an institutional consensus signal.
        Returns {"signal": int (-1, 0, 1), "thought": str, "consensus_ratio": float, "active_count": int, ...}
        """
        if isinstance(part_results, dict):
            results_dict = part_results
            p2 = part_results.get('part2_zone', {})
        elif isinstance(part_results, list):
            results_dict = {f"part_{i}": r for i, r in enumerate(part_results)}
            p2 = next((r for r in part_results if isinstance(r, dict) and (
                'support zone' in r.get('thought', '').lower() or 'resistance zone' in r.get('thought', '').lower()
            )), {})
        else:
            return {"signal": 0, "thought": "Invalid input to SignalFusionEngine"}

        p2_thought = str(p2.get('thought', '')).lower()
        p2_sig = p2.get('signal', 0)

        weighted_buy = 0.0
        weighted_sell = 0.0
        weighted_neutral = 0.0
        total_weight_den = 0.0
        active_buy_engines = []
        active_sell_engines = []
        active_neutral_engines = []

        for name, r in results_dict.items():
            if not isinstance(r, dict):
                continue
            if name == 'part7_volatility':
                continue  # PHASE4: Part 7 is volatility regime/advisory only, never directional denominator
            sig = r.get('signal', 0)
            thought = str(r.get('thought', '')).lower()
            try:
                sig = float(sig)
            except (ValueError, TypeError):
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1075)
                continue
                
            # Skip offline or errored engines so they don't dilute confidence
            if any(k in thought for k in ["error", "offline", "fallback", "missing"]):
                continue

            # MTF annotation accepts the live 8-frame denominator and older /4 telemetry.
            import re
            mtf_match = re.search(r'mtf: (\d+)/(\d+)', thought)
            if mtf_match and sig != 0:
                mtf_agree, mtf_total = int(mtf_match.group(1)), int(mtf_match.group(2))
                if mtf_total > 0 and mtf_agree / mtf_total < 0.5:
                    sig = 0  # Low cross-frame support remains neutral.

            w = self.weights.get(name, 1.0)
            total_weight_den += w  # FULL weight goes to denominator regardless of signal strength
            
            if sig > 0:
                weighted_buy += w * sig
                active_buy_engines.append(name)
            elif sig < 0:
                weighted_sell += w * abs(sig)
                active_sell_engines.append(name)
            else:
                weighted_neutral += w
                active_neutral_engines.append(name)

        total_weight = total_weight_den
        total_active_engines = len(active_buy_engines) + len(active_sell_engines) + len(active_neutral_engines)

        # 1. Quorum & Confluence Threshold: Require at least 3 active engines & 3.5 total weight
        if total_active_engines < self.min_active_engines or total_weight < self.min_weighted_score:
            return {
                "signal": 0,
                "thought": f"Fusion Neutral: Insufficient confluence ({total_active_engines}/{self.min_active_engines} engines, weight {total_weight:.1f}/{self.min_weighted_score:.1f})",
                "consensus_ratio": 0.0,
                "active_count": total_active_engines
            }

        buy_ratio = weighted_buy / total_weight
        sell_ratio = weighted_sell / total_weight
        # PHASE4: neutral engines must not dilute a real directional majority once quorum (>=3 active engines & >=3.2 weight) is met.
        _n_dir = len(active_buy_engines) + len(active_sell_engines)
        _active_w = weighted_buy + weighted_sell
        if _n_dir >= 3 and _active_w >= 3.2:
            buy_ratio = weighted_buy / _active_w
            sell_ratio = weighted_sell / _active_w

        # 2. Bullish Confluence
        if buy_ratio >= self.min_consensus_pct:
            # Zone Veto: Block BUY if at Resistance Zone
            if (p2_sig == -1 or 'resistance zone' in p2_thought) and not (p2_sig != -1 and 'part6_trend' in active_buy_engines and any(e in active_buy_engines for e in ('part8_structure', 'part9_orderflow'))):
                return {
                    "signal": 0,
                    "thought": f"Zone Veto: BUY blocked at Resistance Zone ({p2.get('thought')})",
                    "consensus_ratio": buy_ratio,
                    "active_count": total_active_engines
                }
            
            # Major Anchor Dissent Check
            strong_opponents = [eng for eng in active_sell_engines if eng in ('part6_trend', 'part8_structure', 'part9_orderflow')]
            if len(strong_opponents) >= 2:
                return {
                    "signal": 0,
                    "thought": f"Fusion Dissent Veto: BUY opposed by key anchors {strong_opponents}",
                    "consensus_ratio": buy_ratio,
                    "active_count": total_active_engines
                }
            
            return {
                "signal": 1,
                "thought": f"Fusion Bullish: {len(active_buy_engines)} engines ({buy_ratio*100:.0f}% weighted consensus, score={weighted_buy:.1f})",
                "consensus_ratio": buy_ratio,
                "active_count": total_active_engines,
                "weighted_score": weighted_buy
            }

        # 3. Bearish Confluence
        elif sell_ratio >= self.min_consensus_pct:
            # Zone Veto: Block SELL if at Support Zone
            if (p2_sig == 1 or 'support zone' in p2_thought) and not (p2_sig != 1 and 'part6_trend' in active_sell_engines and any(e in active_sell_engines for e in ('part8_structure', 'part9_orderflow'))):
                return {
                    "signal": 0,
                    "thought": f"Zone Veto: SELL blocked at Support Zone ({p2.get('thought')})",
                    "consensus_ratio": sell_ratio,
                    "active_count": total_active_engines
                }
            
            # Major Anchor Dissent Check
            strong_opponents = [eng for eng in active_buy_engines if eng in ('part6_trend', 'part8_structure', 'part9_orderflow')]
            if len(strong_opponents) >= 2:
                return {
                    "signal": 0,
                    "thought": f"Fusion Dissent Veto: SELL opposed by key anchors {strong_opponents}",
                    "consensus_ratio": sell_ratio,
                    "active_count": total_active_engines
                }

            return {
                "signal": -1,
                "thought": f"Fusion Bearish: {len(active_sell_engines)} engines ({sell_ratio*100:.0f}% weighted consensus, score={weighted_sell:.1f})",
                "consensus_ratio": sell_ratio,
                "active_count": total_active_engines,
                "weighted_score": weighted_sell
            }

        # 4. Split / Conflicted Consensus
        return {
            "signal": 0,
            "thought": f"Fusion Split: Conflict between Bull ({weighted_buy:.1f}) and Bear ({weighted_sell:.1f})",
            "consensus_ratio": max(buy_ratio, sell_ratio),
            "active_count": total_active_engines
        }

    @part_advisory_entry("part11_fusion", data_parameter="results_by_timeframe", context_parameter="context", evidence_context_key="neural_advisory_evidence")
    def analyze_multi_timeframe(self, results_by_timeframe: Dict[str, Any], context=None) -> Dict[str, Any]:
        """Fuse each native frame independently, then summarize all native frames.

        This remains Parts 1–10 evidence, not execution authority. Every row is
        passed through the existing Part 11 engine with its canonical part names.
        """
        # Same native frame set as the central MTF strategy. Each correlation
        # group shares one horizon weight so correlated frames do not multiply it.
        from binance_timeframes import (
            BINANCE_SPOT_TIMEFRAMES as timeframes,
            TIMEFRAME_CORRELATION_GROUPS, TIMEFRAME_GROUP_WEIGHTS,
        )
        tf_weights = {tf: TIMEFRAME_GROUP_WEIGHTS[g] / len(members)
                      for g, members in TIMEFRAME_CORRELATION_GROUPS.items() for tf in members}
        if not isinstance(results_by_timeframe, dict) or set(results_by_timeframe) != set(timeframes):

            return {"signal": 0, "thought": "Part11 MTF evidence incomplete",
                    "data_status": "error", "timeframe_coverage": []}
        required_parts = set(self.weights)
        per_timeframe = {}
        buy_weight = sell_weight = 0.0
        total_weight = sum(tf_weights.values())
        for timeframe in timeframes:
            parts = results_by_timeframe.get(timeframe)
            if not isinstance(parts, dict) or not required_parts.issubset(parts):
                return {"signal": 0, "thought": f"Part11 {timeframe} evidence incomplete",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            result = self.analyze(parts)
            try:
                signal = float(result.get("signal", 0))
            except (TypeError, ValueError, OverflowError):
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1214)
                return {"signal": 0, "thought": f"Part11 {timeframe} signal malformed",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            if not np.isfinite(signal):
                return {"signal": 0, "thought": f"Part11 {timeframe} signal non-finite",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            per_timeframe[timeframe] = result
            if signal > 0:
                buy_weight += tf_weights[timeframe] * min(1.0, signal)
            elif signal < 0:
                sell_weight += tf_weights[timeframe] * min(1.0, abs(signal))
        buy_ratio, sell_ratio = buy_weight / total_weight, sell_weight / total_weight
        signal = 1 if buy_ratio >= 0.65 else -1 if sell_ratio >= 0.65 else 0
        direction = "bullish" if signal > 0 else "bearish" if signal < 0 else "neutral"
        agreed = sum(1 for result in per_timeframe.values() if int(result.get("signal", 0) or 0) == signal) if signal else 0
        return {"signal": signal, "thought": f"Part11 native MTF {direction}: {agreed}/8 frames; diagnostic only",
                "data_status": "valid", "timeframe": "aggregate",
                "timeframe_coverage": list(timeframes), "timeframe_results": per_timeframe,
                "weighted_buy_ratio": round(buy_ratio, 6), "weighted_sell_ratio": round(sell_ratio, 6)}


# ==================== LINUX OPTIMIZATION ====================

def setup_linux_confidence_environment():
    """Setup Linux-optimized environment for confidence processing"""
    # Set thread affinity for i5 4-core CPU
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Enable GPU memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Set async performance parameters
    os.environ['PYTHONASYNCIODEBUG'] = '0'
    
    print("OK Linux environment optimized for confidence processing")

if __name__ == "__main__":
    setup_linux_confidence_environment()
    print("ACCELERATED Part11 Unified Signal Fusion & Confidence Engine Ready")