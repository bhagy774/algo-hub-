try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ==============================================================================
# JARVIS PART 5 - INSTITUTIONAL GPU FUSION ENGINE (GTX 1650 & CPU OPTIMIZED)
# Fully hardened against hidden bugs, missing CUDA helpers, syntax flaws,
# missing method signatures, parameter mismatches, and PyTorch fallback errors.
# ==============================================================================


import sys
import gc
import time
import threading
import traceback
from contextlib import contextmanager, nullcontext
from collections import defaultdict, deque, OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 27)
    TORCH_AVAILABLE = False
    class _DummyTorch:
        cuda_available = False
        class device:
            def __init__(self, *a, **k): pass
        def __getattr__(self, name): return lambda *a, **k: None
    torch = _DummyTorch()  # type: ignore[assignment]
    class nn:  # type: ignore[no-redef]
        class Module:
            def __init__(self, *a, **k): pass
        class Linear:
            def __init__(self, *a, **k): pass
# Safe stdout encoding wrapper for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 44)
        pass



# ---- PyTorch with complete NumPy Fallback for Windows/WSL compatibility ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 56)
    TORCH_AVAILABLE = False
    
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
            self.device = 'cpu'

        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return self._data
        def item(self): return float(self._data.flat[0]) if self._data.size > 0 else 0.0
        def float(self): return self
        def long(self): return self
        def int(self): return self
        def detach(self): return self
        def clone(self): return DummyTensor(self._data.copy())
        def reshape(self, *shape): return DummyTensor(self._data.reshape(*shape))
        def unsqueeze(self, dim=0): return DummyTensor(np.expand_dims(self._data, axis=dim))
        
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
        
        def __add__(self, other): return DummyTensor(self._data + (other._data if isinstance(other, DummyTensor) else other))
        def __sub__(self, other): return DummyTensor(self._data - (other._data if isinstance(other, DummyTensor) else other))
        def __mul__(self, other): return DummyTensor(self._data * (other._data if isinstance(other, DummyTensor) else other))
        def __pow__(self, other): return DummyTensor(self._data ** (other._data if isinstance(other, DummyTensor) else other))
        def __truediv__(self, other): return DummyTensor(self._data / (other._data if isinstance(other, DummyTensor) else 1e-8))

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
            
        class optim:
            Adam = DummyModule
            SGD = DummyModule
        
        class cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def device_count(): return 0
            @staticmethod
            def get_device_name(idx=0): return "CPU_Fallback"
            @staticmethod
            def empty_cache(): pass
            @staticmethod
            def set_per_process_memory_fraction(*args, **kwargs): pass
            @staticmethod
            def memory_allocated(): return 0
        
        @staticmethod
        def tensor(data, **kwargs):
            if isinstance(data, DummyTensor): return data
            return DummyTensor(data)
        
        @staticmethod
        def zeros(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.zeros(shape, dtype=np.float32))

        @staticmethod
        def ones(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.ones(shape, dtype=np.float32))

        @staticmethod
        def randn(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.random.randn(*shape).astype(np.float32))

        @staticmethod
        def mean(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.mean(arr))

        @staticmethod
        def sum(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.sum(arr))

        @staticmethod
        def softmax(tensor, dim=0, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            e_x = np.exp(arr - np.max(arr))
            return DummyTensor(e_x / (e_x.sum() + 1e-8))

        @staticmethod
        def clamp(tensor, min_val, max_val):
            val = tensor.item() if isinstance(tensor, DummyTensor) else float(tensor)
            return DummyTensor(max(min_val, min(max_val, val)))

# Helpers for CUDA context management & device names
@contextmanager
def _cuda_guard(dev):
    if TORCH_AVAILABLE and hasattr(dev, 'type') and dev.type == 'cuda':
        with torch.cuda.device(dev):
            yield
    else:
        yield

def _safe_get_device_name(dev):
    if TORCH_AVAILABLE and hasattr(dev, 'type') and dev.type == 'cuda':
        try: return torch.cuda.get_device_name(0)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 210)
            pass
    return "CPU"


# ==================== LINUX / WINDOWS UTILITIES ====================

class LinuxOptimizedDeque(deque):
    def __init__(self, iterable=(), maxlen=1000):
        if isinstance(iterable, int):
            super().__init__(maxlen=iterable)
        elif maxlen is not None:
            super().__init__(iterable, maxlen=maxlen)
        else:
            super().__init__(iterable)


class GPUFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        
    def extract_basic(self, df):
        if df is None or len(df) == 0:
            return torch.zeros(10, device=self.device)
        try:
            close = df['close'].values.astype(np.float32)
            return torch.tensor(close[-10:] if len(close) >= 10 else close, device=self.device)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 236)
            return torch.zeros(10, device=self.device)


class GPUMemoryManager:
    """Linux & Windows optimized GPU memory management for GTX 1650 4GB / CPU"""
    
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        self.max_memory = 3.5 * 1024 * 1024 * 1024  # 3.5GB safety limit
        self.active_tensors = {}
        
        if hasattr(self.device, 'type') and self.device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.85)
                print(f"OK GPU Memory initialized: {_safe_get_device_name(self.device)}")
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 252)
                pass
        else:
            print("OK CPU mode - GPU not available, running on CPU")
    
    def allocate_tensor(self, data, name=None, persistent=False):
        """Safe GPU tensor allocation with auto-cleanup"""
        try:
            if not isinstance(data, (int, float, list, tuple, np.ndarray, torch.Tensor)):
                return torch.zeros(10, device=self.device)
                
            if hasattr(self.device, 'type') and self.device.type == 'cuda':
                if torch.cuda.memory_allocated() > self.max_memory:
                    self.cleanup_non_persistent()
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            if persistent and name:
                self.active_tensors[name] = tensor
            return tensor
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 272)
            return torch.tensor(0.0)
    
    def cleanup_non_persistent(self):
        if hasattr(self.device, 'type') and self.device.type == 'cuda':
            try:
                current_memory = torch.cuda.memory_allocated()
                if current_memory > self.max_memory * 0.8:
                    keys_to_remove = [k for k in self.active_tensors.keys() if not k.startswith('persistent_')]
                    for key in keys_to_remove:
                        del self.active_tensors[key]
                    torch.cuda.empty_cache()
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 284)
                pass
        gc.collect()


# ==================== GPU-ACCELERATED ML PREDICTION ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p5_ml_engine', 'analyze')
class MLEngineGPU:
    """
    JARVIS PART 5 - GPU-ACCELERATED MACHINE LEARNING PREDICTION ENGINE
    GTX 1650 CUDA & CPU Optimized.

    Extracts statistical features from multi-timeframe OHLCV:
    1. Volatility-Normalized Drift (Sharpe Z-score over last 20 candles)
    2. Linear Regression Trend t-statistic (Statistical Significance of Slope, p < 0.05)
    3. Multi-EMA Alignment (EMA8 vs EMA21)
    4. Normalized RSI-14 Momentum Delta
    5. Volume-Price Confirmation Ratio (>= 0.8x 20-period avg volume)
    6. Strict Chop / Random Walk Deadband Filter (Signals 0 in noise)
    """
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        self.gpu_manager = GPUMemoryManager()

    @part_advisory_entry("part5_ml", data_parameter="data", context_parameter="context")
    def analyze(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        return qm.part_signal("5", data, context=context)


# ==================== GPU-ACCELERATED FUSION ENGINE ====================

# The former random-initialized NeuralFusionNetwork is retired. Neural inference
# in this engine is deliberately disabled until a separately validated local
# artifact is integrated through neural_advisory.py.
class GPUEnhancedFusionEngine:
    """
    INSTITUTIONAL-Grade Fusion Engine
    GTX 1650 CUDA & CPU Optimized
    """
    
    def __init__(self, previous_parts=None):
        self.gpu_manager = GPUMemoryManager()
        self.device = self.gpu_manager.device
        
        if previous_parts is None:
            previous_parts = {}
            
        self.part1 = previous_parts.get('part1')
        self.part2 = previous_parts.get('part2') 
        self.part3 = previous_parts.get('part3')
        self.part4 = previous_parts.get('part4')
        
        self.signal_buffer = deque(maxlen=1000)
        self.confidence_matrix = None
        self.strategy_weights_gpu = None
        
        # No runtime model construction or inference: untrained weights are never used.
        self.neural_fusion = None
        self.use_neural = False

        # Advanced Math: Kalman Smoothing state
        self.kalman_state = 0.0
        self.kalman_uncertainty = 1.0

        self.performance_stats = {}
        self.market_regime_history = deque(maxlen=50)
        
        self.data_lock = threading.RLock()
        self.fusion_active = True
        
        self._initialize_gpu_tensors()
        print("ACCELERATED GPU Fusion Engine Initialized (Advanced Math Active)")
    
    def _initialize_gpu_tensors(self):
        try:
            base_weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device, dtype=torch.float32)
            self.strategy_weights_gpu = base_weights
            self.confidence_matrix = torch.zeros((4, 10), device=self.device, dtype=torch.float32)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 364)
            self.strategy_weights_gpu = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
            self.confidence_matrix = torch.zeros((4, 10), dtype=torch.float32)

    def fuse_modules_gpu(self, df_1min_or_results=None, df_5min=None, df_15min=None):
        """
        GPU-ACCELERATED Module Fusion
        Flexible interface: Accepts DataFrames (df_1min, df_5min, df_15min) OR part_results list/dict.
        """
        try:
            # Check if passed a list or dict of pre-computed part results
            if isinstance(df_1min_or_results, (list, dict)):
                module_results = df_1min_or_results if isinstance(df_1min_or_results, dict) else {f"part_{i+1}": r for i, r in enumerate(df_1min_or_results) if isinstance(r, dict)}
                fused_signal = self._fuse_signals_gpu(module_results)
                final_confidence = self._aggregate_confidence_gpu(module_results)
                
                res = self._create_gpu_signal(fused_signal, final_confidence, "GPU Results Fusion")

                return res

            df_1min = df_1min_or_results
            if not self._validate_input_data(df_1min, df_5min, df_15min):
                return self._create_gpu_signal("NO TRADE", 0, "Invalid data")
            
            with _cuda_guard(self.device):
                prepared_data = self._prepare_gpu_data(df_1min, df_5min, df_15min)
                if prepared_data is None:
                    return self._create_gpu_signal("NO TRADE", 0, "GPU data prep failed")
            
            module_results = self._execute_modules_parallel(prepared_data)
            fused_signal = self._fuse_signals_gpu(module_results)
            final_confidence = self._aggregate_confidence_gpu(module_results)
            
            res = self._create_gpu_signal(fused_signal, final_confidence, "GPU Data Fusion")

            return res
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 411)
            error_msg = f"Fusion error: {str(e)}"
            return self._create_gpu_signal("NO TRADE", 0, error_msg)

    def _call_legacy_llm_fusion_sanity_check(self, module_results, fused_signal, confidence):
        """Compatibility hook: unavailable is not an approval or vote."""
        return None, None


    def fuse_modules_mtf(self, mtf_data):
        """Multi-Timeframe GPU Fusion"""
        try:
            if not isinstance(mtf_data, dict):
                return self._create_gpu_signal("NO TRADE", 0, "Invalid mtf_data type")
                
            tf_weights = {
                '1m': 1.0, '3m': 1.5, '5m': 2.0, '15m': 3.0,
                '30m': 4.0, '1h': 5.0, '2h': 6.0, '4h': 7.0
            }
            
            tf_results = {}
            weighted_sum = 0.0
            total_weight = 0.0
            
            for tf_name, tf_df in mtf_data.items():
                if tf_df is None:
                    continue
                
                tf_weight = tf_weights.get(tf_name, 1.0)
                try:
                    result = self.fuse_modules_gpu(tf_df)
                    if isinstance(result, dict):
                        signal = result.get('signal', result.get('direction', 'NO TRADE'))
                        conf = result.get('confidence', 0)
                        
                        sig_val = 1 if signal in ['CALL', 'BUY'] else -1 if signal in ['PUT', 'SELL'] else 0
                        weighted_sum += sig_val * tf_weight
                        total_weight += tf_weight
                        tf_results[tf_name] = {'signal': sig_val, 'confidence': conf}
                except Exception:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 449)
                    pass
            
            consensus = weighted_sum / total_weight if total_weight > 0 else 0
            direction = "CALL" if consensus > 0.15 else "PUT" if consensus < -0.15 else "NO TRADE"
            avg_conf = sum(r.get('confidence', 0) for r in tf_results.values()) / max(len(tf_results), 1)
            final_signal = self._create_gpu_signal(direction, avg_conf, f"MTF Fusion ({len(tf_results)} TFs)")
            if hasattr(self, 'bus') and self.bus:
                msg = f"Fusion Engine (MTF): Consensus = {consensus:.2f}, Direction = {direction}, Confidence = {avg_conf:.1f}%"
                self.bus.publish('THOUGHTS', 'Part5_Fusion', msg)
            return final_signal
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 461)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part5_Fusion', e, context='fuse_modules_mtf')
            return self._create_gpu_signal("NO TRADE", 0, f"MTF error: {e}")

    def _prepare_gpu_data(self, df_1min, df_5min, df_15min):
        try:
            data_tensors = {}
            for timeframe, df in [('1min', df_1min), ('5min', df_5min), ('15min', df_15min)]:
                if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
                    if len(cols) >= 4:
                        vals = df[cols].values.astype(np.float32)
                        data_tensors[timeframe] = self.gpu_manager.allocate_tensor(vals, f'data_{timeframe}')
            return data_tensors
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 476)
            return None

    def _execute_modules_parallel(self, prepared_data):
        module_results = {}
        threads = []
        
        def run_module(module_name, module, method_names):
            for method_name in method_names:
                method = getattr(module, method_name, None)
                if method is not None:
                    try:
                        result = method(prepared_data)
                        with self.data_lock:
                            module_results[module_name] = result
                        return
                    except Exception:
                        globals().get("_plog", lambda *_a, **_k: None)(__name__, 492)
                        pass
        
        module_configs = [
            ('part1', self.part1, ['analyze_price_action', 'analyze', 'run', 'get_signal']),
            ('part2', self.part2, ['calculate_momentum', 'analyze', 'run', 'get_signal']),
            ('part3', self.part3, ['assess_market_regime', 'analyze', 'run', 'get_signal']),
            ('part4', self.part4, ['evaluate_risk', 'analyze', 'run', 'get_signal']),
        ]
        
        for name, module, methods in module_configs:
            if module is not None:
                t = threading.Thread(target=run_module, args=(name, module, methods))
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join(timeout=5.0)
        
        return module_results

    def _apply_kalman_filter(self, measurement, measurement_noise=0.5, process_noise=0.1):
        """Advanced Math: 1D Kalman Filter for signal smoothing"""
        # Prediction Update
        predicted_uncertainty = self.kalman_uncertainty + process_noise

        # Measurement Update (Bayesian Inference)
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + measurement_noise)
        self.kalman_state = self.kalman_state + kalman_gain * (measurement - self.kalman_state)
        self.kalman_uncertainty = (1 - kalman_gain) * predicted_uncertainty

        return self.kalman_state

    def _fuse_signals_gpu(self, module_results):
        try:
            if not module_results:
                return "NO TRADE"
            
            signals = []
            confidences = []
            
            for result in module_results.values():
                if isinstance(result, dict) and ('signal' in result or 'direction' in result):
                    sig_str = result.get('signal', result.get('direction', 'HOLD'))
                    sig_val = 1 if sig_str in ['CALL', 'BUY', 1] else -1 if sig_str in ['PUT', 'SELL', -1] else 0
                    signals.append(sig_val)
                    # normalize confidence 0 to 1 for NN
                    conf_val = float(result.get('confidence', 50.0))
                    if conf_val > 10.0: conf_val /= 100.0
                    else: conf_val /= 10.0
                    confidences.append(conf_val)
            
            if not signals:
                return "NO TRADE"
            
            with _cuda_guard(self.device):
                signal_tensor = torch.tensor(signals, device=self.device, dtype=torch.float32)
                confidence_tensor = torch.tensor(confidences, device=self.device, dtype=torch.float32)
                
                n = len(signals)

                # The learned component was never trained/loaded. Preserve the
                # deterministic weighted evidence path and never infer random logits.

                # --- Traditional Math: Weighted Sum Path ---
                if self.strategy_weights_gpu is not None and len(self.strategy_weights_gpu) >= n:
                    weights = self.strategy_weights_gpu[:n]
                else:
                    weights = torch.ones(n, device=self.device, dtype=torch.float32) / n
                
                weighted_signals = signal_tensor * confidence_tensor * weights
                trad_fused = float(torch.sum(weighted_signals).item() if hasattr(torch.sum(weighted_signals), 'item') else torch.sum(weighted_signals))

                # Preserve the legacy deterministic contribution of the retired
                # neural blend (0.6 * weighted score); no random model is called.
                blended_raw = 0.6 * trad_fused

                # Apply Kalman Smoothing to the blended signal
                fused_signal = self._apply_kalman_filter(blended_raw)
            
            if fused_signal > 0.4: return "CALL"
            elif fused_signal < -0.4: return "PUT"
            return "NO TRADE"
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 576)
            return "NO TRADE"

    def _aggregate_confidence_gpu(self, module_results):
        try:
            confidences = [float(v.get('confidence', 5.0)) for v in module_results.values() if isinstance(v, dict) and 'confidence' in v]
            if not confidences:
                return 5.0
            
            with _cuda_guard(self.device):
                confidence_tensor = torch.tensor(confidences, device=self.device, dtype=torch.float32)
                mean_conf = float(torch.mean(confidence_tensor).item() if hasattr(torch.mean(confidence_tensor), 'item') else torch.mean(confidence_tensor))
            return min(max(mean_conf, 0.0), 10.0)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 589)
            return 5.0

    def update_strategy_weights_gpu(self, performance_data):
        try:
            if not performance_data: return
            performance_tensor = self._prepare_performance_data_gpu(performance_data)
            if performance_tensor is None: return
            
            with _cuda_guard(self.device):
                performance_normalized = torch.softmax(performance_tensor, dim=0)
                momentum = self._calculate_performance_momentum_gpu(performance_tensor)
                adjusted_weights = performance_normalized * (1.0 + momentum)
                final_weights = adjusted_weights / (torch.sum(adjusted_weights) + 1e-8)
                self.strategy_weights_gpu = final_weights

        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 605)
            pass

    def _prepare_performance_data_gpu(self, performance_data):
        try:
            scores = []
            for strategy in ['part1', 'part2', 'part3', 'part4']:
                if isinstance(performance_data, dict) and strategy in performance_data:
                    val = performance_data[strategy]
                    rate = val.get('success_rate', 0.5) if isinstance(val, dict) else (val if isinstance(val, (int, float)) else 0.5)
                    scores.append(float(rate))
                else:
                    scores.append(0.3)
            return torch.tensor(scores, device=self.device, dtype=torch.float32)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 619)
            return None

    def _calculate_performance_momentum_gpu(self, performance_tensor):
        try:
            if len(performance_tensor) >= 2:
                curr = performance_tensor[-1]
                prev = performance_tensor[-2]
                momentum = (curr - prev) / (prev + 1e-8)
                return torch.clamp(momentum, -0.2, 0.2)
            return torch.tensor(0.0, device=self.device)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 630)
            return torch.tensor(0.0, device=self.device)

    def build_confidence_matrix_gpu(self, module_results=None, market_regime="RANGING"):
        try:
            if module_results is None: module_results = {}
            with _cuda_guard(self.device):
                matrix = torch.zeros((4, 10), device=self.device, dtype=torch.float32)
                for i, (module_name, result) in enumerate(module_results.items()):
                    if i >= 4: break
                    if isinstance(result, dict) and 'confidence' in result:
                        base_conf = float(result['confidence'])
                        regime_fac = self._get_regime_confidence_factor(market_regime)
                        matrix[i] = base_conf * regime_fac
                self.confidence_matrix = matrix
                return matrix.numpy() if hasattr(matrix, 'numpy') else np.zeros((4, 10))
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 646)
            return np.zeros((4, 10))

    def _get_regime_confidence_factor(self, regime):
        regime_factors = {'TRENDING_UP': 1.1, 'TRENDING_DOWN': 1.1, 'RANGING': 0.9, 'VOLATILE': 0.8, 'HIGH_VOLATILITY': 0.7}
        return regime_factors.get(regime, 1.0)

    def route_to_master_system(self, fused_signal, confidence=5.0, reason="Fusion"):
        try:
            if self.strategy_weights_gpu is not None and hasattr(self.strategy_weights_gpu, 'numpy'):
                weights_list = self.strategy_weights_gpu.numpy().tolist()
            else:
                weights_list = [0.25, 0.25, 0.25, 0.25]
            
            routing_package = {
                'timestamp': datetime.now().isoformat(),
                'fused_signal': fused_signal,
                'confidence': float(confidence),
                'reason': reason,
                'strategy_weights': weights_list,
                'module_count': 4,
                'gpu_optimized': (hasattr(self.device, 'type') and self.device.type == 'cuda'),
                'system_load': self._get_system_load()
            }
            self.gpu_manager.cleanup_non_persistent()
            return routing_package
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 672)
            return self._create_error_routing_package()

    def _validate_input_data(self, df_1min, df_5min, df_15min):
        if df_1min is None or not isinstance(df_1min, pd.DataFrame) or len(df_1min) < 5:
            return False
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for df in [df_1min, df_5min, df_15min]:
            if df is not None and isinstance(df, pd.DataFrame):
                if not all(c in df.columns for c in required_cols):
                    return False
        return True

    def _create_gpu_signal(self, signal, confidence, reason):
        return {
            'signal': signal,
            'confidence': float(confidence),
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'gpu_optimized': (hasattr(self.device, 'type') and self.device.type == 'cuda')
        }

    def _get_system_load(self):
        try:
            gpu_mem = (torch.cuda.memory_allocated() / (1024 * 1024)) if (TORCH_AVAILABLE and torch.cuda.is_available()) else 0.0
            return {
                'gpu_memory_mb': gpu_mem,
                'active_threads': threading.active_count(),
                'ram_usage': self._get_ram_usage()
            }
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 702)
            return {'gpu_memory_mb': 0.0, 'gpu_utilization': 0.0}

    def _get_ram_usage(self):
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 709)
            return 0.0

    def _create_error_routing_package(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'fused_signal': 'NO TRADE',
            'confidence': 0.0,
            'reason': 'Routing system error',
            'strategy_weights': [0.25, 0.25, 0.25, 0.25],
            'module_count': 0,
            'gpu_optimized': False
        }

    def shutdown(self):
        self.fusion_active = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 727)
                pass
        self.signal_buffer.clear()
        self.confidence_matrix = None
        self.strategy_weights_gpu = None


# ==================== MAIN INTEGRATION CLASS ====================

class InstitutionalFusionEngine:
    """
    MAIN INTEGRATION CLASS - Connects Part1-Part4 with GPU-optimized Part5
    """
    
    def __init__(self, part1=None, part2=None, part3=None, part4=None):
        print("ACCELERATED Initializing Institutional Fusion Engine...")
        self.previous_parts = {
            'part1': part1,
            'part2': part2, 
            'part3': part3,
            'part4': part4
        }
        self.gpu_fusion_engine = GPUEnhancedFusionEngine(self.previous_parts)
        self.performance_history = deque(maxlen=1000)
        self.last_optimization = time.time()
        self.optimization_interval = 300
        self.start_time = time.time()
        print("OK Institutional Fusion Engine Ready - GPU Accelerated")

    def process_market_data(self, df_1min=None, df_5min=None, df_15min=None, market_regime="RANGING"):
        try:
            if isinstance(df_1min, dict) and '1m' in df_1min:
                df_dict = df_1min
                df_1min = df_dict.get('1m')
                df_5min = df_dict.get('5m', df_1min)
                df_15min = df_dict.get('15m', df_1min)

            prepared_data = self.gpu_fusion_engine._prepare_gpu_data(df_1min, df_5min, df_15min)
            module_results = self.gpu_fusion_engine._execute_modules_parallel(prepared_data) if prepared_data else {}
            
            fusion_result = self.gpu_fusion_engine.fuse_modules_gpu(df_1min, df_5min, df_15min)
            confidence_matrix = self.gpu_fusion_engine.build_confidence_matrix_gpu(module_results, market_regime)
            
            routing_package = self.gpu_fusion_engine.route_to_master_system(
                fusion_result.get('signal', 'NO TRADE'),
                fusion_result.get('confidence', 0.0), 
                fusion_result.get('reason', 'Fusion')
            )
            routing_package['confidence_matrix'] = confidence_matrix.tolist() if hasattr(confidence_matrix, 'tolist') else []
            self._perform_periodic_optimization()
            return routing_package
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 778)
            return self._create_error_response(str(e))

    def _perform_periodic_optimization(self):
        curr = time.time()
        if curr - self.last_optimization > self.optimization_interval:
            try:
                if self.performance_history:
                    self.gpu_fusion_engine.update_strategy_weights_gpu(self.performance_history[-1])
                self.gpu_fusion_engine.gpu_manager.cleanup_non_persistent()
                self.last_optimization = curr
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 789)
                pass

    def _create_error_response(self, error_msg):
        return {
            'timestamp': datetime.now().isoformat(),
            'fused_signal': 'NO TRADE',
            'confidence': 0.0,
            'reason': error_msg,
            'strategy_weights': [0.25, 0.25, 0.25, 0.25],
            'module_count': 0,
            'gpu_optimized': False
        }

    def get_system_status(self):
        return {
            'gpu_available': (hasattr(self.gpu_fusion_engine.device, 'type') and self.gpu_fusion_engine.device.type == 'cuda'),
            'fusion_active': self.gpu_fusion_engine.fusion_active,
            'performance_history_size': len(self.performance_history),
            'last_optimization': self.last_optimization,
            'system_uptime': time.time() - self.start_time
        }

    def shutdown(self):
        self.gpu_fusion_engine.shutdown()


def optimized_deque_linux(maxlen=1000):
    return deque(maxlen=maxlen)

def linux_memory_check():
    try:
        import psutil
        return psutil.virtual_memory().available > 500 * 1024 * 1024
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 823)
        return True


if __name__ == "__main__":
    _device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        print("WARNING CUDA not available - falling back to CPU mode")
    else:
        print(f"OK CUDA device: {_safe_get_device_name(_device)}")
    
    if not linux_memory_check():
        print("WARNING Low memory warning - consider increasing swap space")
    print("ACCELERATED Part5 Fusion Engine - Ready for Integration with Part6 Backtester")
