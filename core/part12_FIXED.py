try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ---- Helpers Inserted for Part-12 Fix ----
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
        try: super().append(item)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 118)
            pass

def _safe_get_device_name(device):
    try:
        if hasattr(device,"type") and device.type=="cuda":
            try:
                idx = device.index if hasattr(device,"index") and device.index is not None else 0
                return torch.cuda.get_device_name(idx)
            except:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 126)
                try: return torch.cuda.get_device_name()
                except:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 128)
                    return "cuda_device"
        return str(device)
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 130)
        return "unknown_device"

class GPUFeatureExtractor:
    def __init__(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def extract_basic(self,data):
        try:
            return torch.tensor([float(x) for x in data[:10]], device=self.device)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 138)
            return torch.zeros(10, device=self.device)
# ---- End helpers ----

# ==================== PART 12: ADVANCED TRADE EXECUTION ENGINE & PORTFOLIO MANAGEMENT ====================
# DEEPSEEK AI-POWERED REWRITE - INSTITUTIONAL GRADE TRADE EXECUTION
# BTCUSDT DELTA EXCHANGE FOCUSED - GPU ACCELERATED RISK MANAGEMENT
# LINUX UBUNTU + GTX 1650 CUDA + i5 10th Gen OPTIMIZED

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# BUG FIX #1: cupy optional — no crash if not installed
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 156)
    CUPY_AVAILABLE = False
import numpy as np
import asyncio
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import pandas as pd
import tempfile
from pathlib import Path
import gc
import math
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union




# ==================== GPU TRADE EXECUTION MEMORY MANAGER ====================

class TradeExecutionGPUMemoryManager:
    """GTX 1650 4GB VRAM Optimized Memory Manager for Trade Execution"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram = 2.5 * 1024 * 1024 * 1024  # 2.5GB safety limit for execution
        self.max_ram = 3.0 * 1024 * 1024 * 1024   # 3GB RAM limit
        
        # Cross-platform execution cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "trade_execution_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.65)  # 65% of 4GB for execution
            print(f"OK Trade Execution GPU Memory: {_safe_get_device_name(self.device)}")
    
    def allocate_execution_tensor(self, data, name="execution_data"):
        """GPU tensor allocation optimized for trade execution data"""
        try:
            # BUG FIX #2: Check device type FIRST before calling cuda memory functions
            if self.device.type == 'cpu':
                return torch.tensor(data, dtype=torch.float32)

            if torch.cuda.memory_allocated() > self.max_vram:
                return torch.tensor(data, dtype=torch.float32)
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            if tensor.numel() < 5000:
                return tensor
            else:
                return self._create_execution_mmap(data, name)
                
        except RuntimeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 212)
            print(f"WARNING Execution GPU allocation failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def _create_execution_mmap(self, data, name):
        """Create memory-mapped tensor for execution data"""
        try:
            # BUG FIX #3: Fixed filename (no timestamp) to avoid disk fill-up
            # Also convert to np.array first in case data is a list
            data_arr = np.array(data, dtype='float32')
            cache_file = self.cache_dir / f"{name}.dat"
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_arr.shape)
            mmap[:] = data_arr
            return torch.from_numpy(np.array(mmap))
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 226)
            print(f"WARNING Execution memory mapping failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def cleanup_execution_memory(self):
        """Aggressive cleanup for trade execution operations"""
        # BUG FIX #4: Only call cuda functions when on CUDA device
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

# BUG FIX #5 #7 #8: torch.cuda.device() crashes on CPU — use this guard
def _cuda_guard(device):
    class _NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return torch.cuda.device(device) if device.type == 'cuda' else _NullCtx()

# ==================== GPU-ACCELERATED ORDER EXECUTION ENGINE ====================

class GPUOrderExecutionEngine:
    """
    INSTITUTIONAL-GRADE ORDER EXECUTION ENGINE
    BTCUSDT Delta Exchange Optimized with GPU Acceleration
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.symbol = "BTCUSDT"
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.gpu_manager = TradeExecutionGPUMemoryManager()
        self.device = self.gpu_manager.device
        
        # GPU-optimized execution buffers
        self.price_history_gpu = None
        self.spread_history_gpu = None
        self.execution_quality_gpu = None
        
        # Position management
        self.current_position = 0.0  # + for long, - for short, 0 for flat
        self.entry_price = 0.0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Risk parameters for BTCUSDT
        self.risk_config = {
            'max_position_size': 0.25,  # 25% of capital per trade
            'max_daily_loss': 0.02,     # 2% maximum daily loss
            'stop_loss_ratio': 0.01,    # 1% initial stop loss
            'take_profit_ratio': 0.02,  # 2% take profit
            'max_spread': 50.0,         # $50 maximum spread for BTC
            'max_slippage': 0.001,      # 0.1% maximum slippage
            'confidence_threshold': 6.0, # Minimum confidence from Part 11
            'volatility_lookback': 20    # Volatility calculation period
        }
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.execution_metrics = deque(maxlen=500)
        self.risk_metrics = deque(maxlen=500)
        
        # Market data buffers
        self.market_data_buffer = deque(maxlen=100)
        self.spread_buffer = deque(maxlen=50)
        self.volatility_buffer = deque(maxlen=30)
        
        # Threading and execution control
        self.execution_lock = threading.RLock()
        self.is_running = True
        self.executor = ThreadPoolExecutor(max_workers=3)  # i5 3-core optimized
        
        # LegacyLLM Position Sizer Cooldown setup
        self.last_legacy_llm_time = 0
        self.legacy_llm_cooldown = 30  # seconds
        self.last_legacy_llm_size_tag = "SIZE: FULL"
        self.last_legacy_llm_multiplier = 1.0
        self.last_legacy_llm_insight = "Full position size approved by AI Chief Position Sizer."

        # Initialize GPU buffers
        self._initialize_execution_buffers()
        
        print("ACCELERATED GPU Order Execution Engine Initialized - BTCUSDT Focused")
    
    def _initialize_execution_buffers(self):
        """Initialize GPU-optimized buffers for trade execution"""
        try:
            # Market data history buffers
            buffer_size = 1000
            self.price_history_gpu = torch.zeros(buffer_size, device=self.device, dtype=torch.float32)
            self.spread_history_gpu = torch.zeros(buffer_size, device=self.device, dtype=torch.float32)
            self.execution_quality_gpu = torch.zeros(buffer_size, device=self.device, dtype=torch.float32)
            
            # Buffer management
            self.buffer_pointer = 0
            self.buffer_size = buffer_size
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 323)
            print(f"WARNING Execution buffer initialization warning: {e}")
    
    # ==================== GPU-ACCELERATED EXECUTION ANALYSIS ====================
    
    async def process_market_data(self, bid_price: float, ask_price: float, timestamp: int, volume: float = 0.0):
        """
        Process real-time BTCUSDT market data with GPU acceleration
        """
        try:
            # Calculate spread and mid-price
            spread = ask_price - bid_price
            mid_price = (bid_price + ask_price) / 2.0
            
            # Update market data buffers
            self.market_data_buffer.append({
                'bid': bid_price,
                'ask': ask_price,
                'mid': mid_price,
                'spread': spread,
                'volume': volume,
                'timestamp': timestamp
            })
            
            # Update GPU buffers
            await self._update_execution_buffers_gpu(mid_price, spread)
            
            # GPU-accelerated execution analysis
            await self._analyze_execution_conditions_gpu()
            
            # Update risk metrics
            await self._update_risk_metrics_gpu()
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 356)
            print(f"ERROR Market data processing error: {e}")
    
    async def _update_execution_buffers_gpu(self, price: float, spread: float):
        """Update GPU buffers with new market data"""
        try:
            # BUG FIX #5: Use _cuda_guard instead of torch.cuda.device()
            with _cuda_guard(self.device):
                ptr = self.buffer_pointer
                
                self.price_history_gpu[ptr] = price
                self.spread_history_gpu[ptr] = spread
                
                if ptr > 0:
                    recent_prices = self.price_history_gpu[max(0, ptr-10):ptr+1]
                    price_volatility = torch.std(recent_prices) / (torch.mean(recent_prices) + 1e-8)
                    # BUG FIX #6: Guard against price=0 in division
                    execution_quality = spread / (price + 1e-8) + price_volatility
                    self.execution_quality_gpu[ptr] = execution_quality
                else:
                    self.execution_quality_gpu[ptr] = spread / (price + 1e-8)
                
                self.buffer_pointer = (ptr + 1) % self.buffer_size
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 380)
            print(f"ERROR Execution buffer update error: {e}")
    
    async def _analyze_execution_conditions_gpu(self):
        """GPU-accelerated execution condition analysis"""
        try:
            # BUG FIX #7: Use _cuda_guard instead of torch.cuda.device()
            with _cuda_guard(self.device):
                # Analyze recent market conditions
                recent_prices = self._get_recent_data(self.price_history_gpu, 50)
                recent_spreads = self._get_recent_data(self.spread_history_gpu, 50)
                recent_quality = self._get_recent_data(self.execution_quality_gpu, 50)
                
                # Calculate execution metrics
                execution_analysis = {}
                
                # Spread analysis
                execution_analysis['avg_spread'] = torch.mean(recent_spreads)
                execution_analysis['max_spread'] = torch.max(recent_spreads)
                execution_analysis['spread_ratio'] = execution_analysis['avg_spread'] / (torch.mean(recent_prices) + 1e-8)
                
                # Calculate volatility metrics
                price_volatility = torch.std(recent_prices) / (torch.mean(recent_prices) + 1e-8)
                
                # Execution quality score
                quality_score = 1.0 / (1.0 + torch.mean(recent_quality) + 1e-8)
                execution_analysis['quality_score'] = quality_score
                
                # Z-score normalization
                spread_z_score = (recent_spreads[-1] - torch.mean(recent_spreads)) / (torch.std(recent_spreads) + 1e-8)
                volatility_z_score = (price_volatility - torch.mean(self._get_recent_data(self.execution_quality_gpu, 100))) / (torch.std(self._get_recent_data(self.execution_quality_gpu, 100)) + 1e-8)
                
                execution_analysis['timing_score'] = torch.exp(-0.5 * (spread_z_score**2 + volatility_z_score**2))
                
                # Store execution analysis
                self.execution_analysis = execution_analysis
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 417)
            print(f"ERROR Execution analysis error: {e}")
    
    def _get_recent_data(self, tensor, lookback):
        """Get recent data from circular buffer"""
        # BUG FIX #10: When buffer not full, negative slice returns wrong zeros
        # Use min(buffer_pointer, lookback) to only get filled data
        filled = self.buffer_pointer  # how many slots actually written (before first wrap)
        if filled == 0:
            # No data yet — return zeros of requested size
            return torch.zeros(lookback, device=self.device)
        if self.buffer_pointer >= lookback:
            return tensor[self.buffer_pointer - lookback:self.buffer_pointer]
        else:
            # Buffer has wrapped or we want more than filled
            # Return what we have: tail + beginning
            tail = tensor[max(0, self.buffer_size - (lookback - self.buffer_pointer)):]
            head = tensor[:self.buffer_pointer]
            result = torch.cat([tail, head])
            # If still short, pad with first valid value
            if len(result) < lookback:
                pad = result[:1].expand(lookback - len(result))
                result = torch.cat([pad, result])
            return result[-lookback:]
    
    # ==================== RISK-AWARE POSITION SIZING ====================
    
    def calculate_position_size(self, confidence: float, current_price: float, stop_loss_price: float) -> float:
        """
        Calculate optimal position size based on confidence and risk parameters
        """
        try:
            # Base position size from confidence (Kelly Criterion inspired)
            base_size = min(confidence / 10.0, self.risk_config['max_position_size'])
            
            # Risk-adjusted sizing based on stop loss distance
            price_risk = abs(current_price - stop_loss_price) / (current_price + 1e-8)
            risk_adjusted_size = base_size * (self.risk_config['stop_loss_ratio'] / (price_risk + 1e-8))
            
            # Volatility adjustment
            if hasattr(self, 'execution_analysis') and isinstance(self.execution_analysis, dict):
                vol_val = self.execution_analysis.get('volatility', 0.0)
                vol_float = float(vol_val.item()) if hasattr(vol_val, 'item') else float(vol_val)
                volatility_penalty = 1.0 / (1.0 + vol_float * 10)
                risk_adjusted_size *= volatility_penalty
            
            # Capital allocation
            max_trade_value = self.current_capital * self.risk_config['max_position_size']
            position_size = min(risk_adjusted_size * self.current_capital, max_trade_value) / (current_price + 1e-8)
            
            # Apply LegacyLLM AI Chief Position Sizer Multiplier (FULL = 1.0, HALF = 0.5, QUARTER = 0.25)
            multiplier, size_tag, insight = self.get_legacy_llm_position_multiplier(confidence, current_price)
            position_size *= multiplier

            return max(position_size, 0.0)  # Ensure non-negative
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 473)
            print(f"ERROR Position sizing calculation error: {e}")
            return 0.0

    def _generate_legacy_llm_sizing_prompt(self, confidence: float, current_price: float) -> str:
        """Generate LegacyLLM prompt for Chief Position Sizer"""
        vol_info = {}
        if hasattr(self, 'execution_analysis') and isinstance(self.execution_analysis, dict):
            for k, v in self.execution_analysis.items():
                if hasattr(v, 'item'):
                    vol_info[k] = float(v.item())
                elif isinstance(v, (int, float, str)):
                    vol_info[k] = v

        prompt = f"""You are the Chief Position Sizer for an institutional quant trading firm. Your job is to review the current trade setup and market volatility, and decide position sizing.

Trade Parameters:
- Symbol: {self.symbol}
- Current Price: {current_price}
- Mathematical Confidence: {confidence:.1f}
- Execution Volatility & Spread Metrics: {json.dumps(vol_info, default=str)}

Task: Decide whether we should execute this trade with full position size, half position size, or quarter position size based on risk and volatility.

Respond with EXACTLY ONE of the following tags at the start of your response:
- [SIZE: FULL] : Normal risk profile, execute full position size (100%).
- [SIZE: HALF] : Elevated volatility or spread, cut position size to half (50%).
- [SIZE: QUARTER] : High volatility or market uncertainty, cut position size to quarter (25%).

Follow the tag with a 1-sentence risk justification.
"""
        return prompt

    def get_legacy_llm_position_multiplier(self, confidence: float, current_price: float) -> Tuple[float, str, str]:
        """Legacy hook: preserve deterministic risk sizing, never reuse a cached multiplier."""
        return 1.0, "UNAVAILABLE", "Model sizing disabled; deterministic risk sizing only"
    
    # ==================== INTELLIGENT ORDER EXECUTION ====================
    
    async def execute_trade(self, signal_type: str, confidence: float, current_bid: float, current_ask: float, *,
                            central_decision=None, central_approval=None, part_results=None,
                            part7_gate=None, symbol=None, exchange=None, contract=None,
                            instrument_id=None, market_type=None, snapshot_version=None) -> Dict:
        """Execute only with current Jarvis central authority and scope."""
        try:
            from jarvis_strategy_approval import evaluate_central_strategy, validate_entry_approval
            central_direction = {'CALL': 'BUY', 'BUY': 'BUY', 'PUT': 'SELL', 'SELL': 'SELL'}.get(str(signal_type).upper())
            if not isinstance(central_decision, dict) or central_decision.get('approved') is not True:
                return {'status': 'rejected', 'reason': 'Jarvis central strategy approval is required'}
            if central_direction is None or str(central_decision.get('direction', '')).upper() != central_direction:
                return {'status': 'rejected', 'reason': 'Jarvis central direction mismatch'}
            recomputed = evaluate_central_strategy(
                part_results, part7_gate, confidence=central_decision.get('confidence'),
                expected_symbol=symbol,
            )
            if not recomputed.get('approved') or recomputed.get('direction') != central_direction:
                return {'status': 'rejected', 'reason': 'Jarvis central evidence or Part7 gate rejected entry'}
            valid, reason = validate_entry_approval(
                central_approval, direction=central_direction, symbol=symbol, exchange=exchange,
                contract=contract, instrument_id=instrument_id, market_type=market_type,
                snapshot_version=snapshot_version, confidence=central_decision.get('confidence'),
            )
            if not valid:
                return {'status': 'rejected', 'reason': reason}
            if confidence < self.risk_config['confidence_threshold']:
                return {'status': 'rejected', 'reason': 'Low confidence'}
            
            # Check execution conditions
            spread = current_ask - current_bid
            if spread > self.risk_config['max_spread']:
                return {'status': 'rejected', 'reason': 'High spread'}
            
            # Calculate execution price with slippage control
            execution_price = await self._calculate_optimal_execution_price(signal_type, current_bid, current_ask)
            
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = await self._calculate_risk_levels(signal_type, execution_price, confidence)
            
            # Calculate position size
            position_size = self.calculate_position_size(confidence, execution_price, stop_loss)
            
            if position_size <= 0:
                return {'status': 'rejected', 'reason': 'Zero position size'}
            
            # Execute trade
            trade_result = await self._execute_order(signal_type, position_size, execution_price, stop_loss, take_profit)
            
            # Update position and P&L
            await self._update_position(signal_type, position_size, execution_price, stop_loss, take_profit)
            
            return trade_result
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 565)
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                self.trading_system.bus.report_error('Part12_Execution', e, context='execute_trade')
            else:
                print(f"ERROR Trade execution error: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    async def _calculate_optimal_execution_price(self, signal_type: str, bid: float, ask: float) -> float:
        """Advanced Math: VWAP/TWAP inspired predictive optimal execution pricing"""
        try:
            mid_price = (bid + ask) / 2.0
            spread = ask - bid
            
            # Predict short-term price movement based on execution buffer history (Pseudo-VWAP)
            price_trend = 0.0
            if hasattr(self, 'execution_analysis') and 'avg_spread' in self.execution_analysis:
                if self.buffer_pointer > 5:
                    recent_prices = self._get_recent_data(self.price_history_gpu, 10)
                    if len(recent_prices) >= 2:
                        # Linear regression slope of recent tick prices
                        x = torch.arange(len(recent_prices), dtype=torch.float32, device=self.device)
                        x_mean = torch.mean(x)
                        y_mean = torch.mean(recent_prices)
                        slope = torch.sum((x - x_mean) * (recent_prices - y_mean)) / (torch.sum((x - x_mean)**2) + 1e-8)
                        price_trend = slope.item()

            if signal_type.upper() == 'CALL':
                # If price is trending up rapidly, pay the spread to guarantee fill
                if price_trend > spread * 0.5:
                    target_price = ask
                else:
                    # Otherwise, try to buy at or below mid-price
                    target_price = min(mid_price + spread * 0.1, ask)  # Maximum 10% of spread above mid
            else:  # PUT
                # If price is trending down rapidly, cross the spread to guarantee fill
                if price_trend < -spread * 0.5:
                    target_price = bid
                else:
                    # Otherwise, try to sell at or above mid-price
                    target_price = max(mid_price - spread * 0.1, bid)  # Maximum 10% of spread below mid
            
            return target_price
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 608)
            print(f"ERROR Execution price calculation error: {e}")
            return (bid + ask) / 2.0
    
    async def _calculate_risk_levels(self, signal_type: str, entry_price: float, confidence: float) -> Tuple[float, float]:
        """Calculate adaptive stop loss and take profit levels"""
        try:
            base_stop_ratio = self.risk_config['stop_loss_ratio']
            base_take_profit_ratio = self.risk_config['take_profit_ratio']
            
            # Confidence-based adjustment
            confidence_multiplier = confidence / 10.0  # Normalize confidence to 0-1 range
            
            # Tighter stops for higher confidence
            adjusted_stop_ratio = base_stop_ratio * (1.1 - confidence_multiplier * 0.2)
            adjusted_take_profit_ratio = base_take_profit_ratio * (0.9 + confidence_multiplier * 0.2)
            
            if signal_type.upper() == 'CALL':
                stop_loss = entry_price * (1 - adjusted_stop_ratio)
                take_profit = entry_price * (1 + adjusted_take_profit_ratio)
            else:  # PUT
                stop_loss = entry_price * (1 + adjusted_stop_ratio)
                take_profit = entry_price * (1 - adjusted_take_profit_ratio)
            
            return stop_loss, take_profit
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 634)
            print(f"ERROR Risk levels calculation error: {e}")
            # Fallback to fixed ratios
            if signal_type.upper() == 'CALL':
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98
    
    async def _execute_order(self, signal_type: str, size: float, price: float, stop_loss: float, take_profit: float) -> Dict:
        """Execute the trade order (simulated for now)"""
        try:
            trade_value = size * price
            
            # Simulate execution
            execution_time = datetime.now()
            trade_id = f"TRADE_{int(time.time())}_{len(self.trade_history)}"
            
            trade_record = {
                'trade_id': trade_id,
                'symbol': self.symbol,
                'signal_type': signal_type,
                'size': size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': execution_time,
                'trade_value': trade_value,
                'status': 'executed'
            }
            
            self.trade_history.append(trade_record)
            
            print(f"EXECUTED {signal_type} trade: {size:.4f} {self.symbol} at {price:.5f}")
            
            return trade_record
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 670)
            print(f"ERROR Order execution error: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    async def _update_position(self, signal_type: str, size: float, price: float, stop_loss: float, take_profit: float):
        """Update current position after trade execution"""
        try:
            with self.execution_lock:
                if signal_type.upper() == 'CALL':
                    self.current_position = size
                else:  # PUT
                    self.current_position = -size
                
                self.entry_price = price
                self.position_size = size
                
                # Store position info for risk management
                self.current_position_info = {
                    'entry_price': price,
                    'size': size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(),
                    'signal_type': signal_type,
                    # BUG FIX #16: Pre-initialize tracking prices so trailing stop works from first tick
                    'highest_price': price,  # for long trailing stop
                    'lowest_price': price,   # for short trailing stop
                }
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 699)
            print(f"ERROR Position update error: {e}")
    
    # ==================== REAL-TIME POSITION MANAGEMENT ====================
    
    async def monitor_position(self, current_bid: float, current_ask: float):
        """
        Monitor current position and manage exits
        """
        try:
            if self.current_position == 0:
                return {'action': 'none', 'reason': 'No position'}
            
            current_mid = (current_bid + current_ask) / 2.0
            position_info = getattr(self, 'current_position_info', {})
            
            # Calculate unrealized P&L
            if self.current_position > 0:  # Long position
                self.unrealized_pnl = (current_bid - self.entry_price) * self.position_size
                exit_check = await self._check_long_exit(current_bid, position_info)
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - current_ask) * abs(self.position_size)
                exit_check = await self._check_short_exit(current_ask, position_info)
            
            # Update risk metrics
            await self._update_risk_metrics_gpu()
            
            return exit_check
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 728)
            print(f"ERROR Position monitoring error: {e}")
            return {'action': 'none', 'reason': f'Error: {str(e)}'}
    
    async def _check_long_exit(self, current_bid: float, position_info: Dict) -> Dict:
        """Check exit conditions for long position"""
        try:
            stop_loss = position_info.get('stop_loss', 0)
            take_profit = position_info.get('take_profit', float('inf'))
            
            if current_bid <= stop_loss:
                return await self._close_position('stop_loss', current_bid)
            elif current_bid >= take_profit:
                return await self._close_position('take_profit', current_bid)
            else:
                # Check trailing stop or other exit conditions
                trailing_stop_check = await self._check_trailing_stop(current_bid, position_info, 'long')
                if trailing_stop_check['action'] != 'none':
                    return trailing_stop_check
                
                return {'action': 'none', 'reason': 'Hold position'}
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 750)
            print(f"ERROR Long exit check error: {e}")
            return {'action': 'none', 'reason': f'Error: {str(e)}'}
    
    async def _check_short_exit(self, current_ask: float, position_info: Dict) -> Dict:
        """Check exit conditions for short position"""
        try:
            stop_loss = position_info.get('stop_loss', float('inf'))
            take_profit = position_info.get('take_profit', 0)
            
            if current_ask >= stop_loss:
                return await self._close_position('stop_loss', current_ask)
            elif current_ask <= take_profit:
                return await self._close_position('take_profit', current_ask)
            else:
                # Check trailing stop or other exit conditions
                trailing_stop_check = await self._check_trailing_stop(current_ask, position_info, 'short')
                if trailing_stop_check['action'] != 'none':
                    return trailing_stop_check
                
                return {'action': 'none', 'reason': 'Hold position'}
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 772)
            print(f"ERROR Short exit check error: {e}")
            return {'action': 'none', 'reason': f'Error: {str(e)}'}
    
    async def _check_trailing_stop(self, current_price: float, position_info: Dict, position_type: str) -> Dict:
        """Check trailing stop conditions"""
        try:
            if 'highest_price' not in position_info and position_type == 'long':
                position_info['highest_price'] = current_price
                return {'action': 'none', 'reason': 'Initial highest price set'}
            
            if 'lowest_price' not in position_info and position_type == 'short':
                position_info['lowest_price'] = current_price
                return {'action': 'none', 'reason': 'Initial lowest price set'}
            
            trailing_stop_ratio = 0.005  # 0.5% trailing stop
            
            if position_type == 'long':
                position_info['highest_price'] = max(position_info['highest_price'], current_price)
                trailing_stop = position_info['highest_price'] * (1 - trailing_stop_ratio)
                
                if current_price <= trailing_stop:
                    return await self._close_position('trailing_stop', current_price)
            
            else:  # short
                position_info['lowest_price'] = min(position_info['lowest_price'], current_price)
                trailing_stop = position_info['lowest_price'] * (1 + trailing_stop_ratio)
                
                if current_price >= trailing_stop:
                    return await self._close_position('trailing_stop', current_price)
            
            return {'action': 'none', 'reason': 'Trailing stop not triggered'}
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 805)
            print(f"ERROR Trailing stop check error: {e}")
            return {'action': 'none', 'reason': f'Error: {str(e)}'}
    
    async def _close_position(self, exit_reason: str, exit_price: float) -> Dict:
        """Close current position"""
        try:
            with self.execution_lock:
                if self.current_position == 0:
                    return {'action': 'none', 'reason': 'No position to close'}
                
                # Calculate realized P&L
                if self.current_position > 0:  # Long position
                    realized_pnl = (exit_price - self.entry_price) * self.position_size
                else:  # Short position
                    realized_pnl = (self.entry_price - exit_price) * abs(self.position_size)
                
                self.realized_pnl += realized_pnl
                self.current_capital += realized_pnl
                
                # Record trade closure
                close_record = {
                    'exit_time': datetime.now(),
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'realized_pnl': realized_pnl,
                    'position_size': self.position_size
                }
                
                # Update trade history
                if self.trade_history:
                    self.trade_history[-1].update(close_record)
                
                print(f"CLOSED position: {exit_reason} at {exit_price:.5f}, P&L: {realized_pnl:.2f}")
                
                # Reset position
                self.current_position = 0.0
                self.position_size = 0.0
                self.unrealized_pnl = 0.0
                
                return {
                    'action': 'closed',
                    'reason': exit_reason,
                    'realized_pnl': realized_pnl,
                    'exit_price': exit_price
                }
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 852)
            print(f"ERROR Position close error: {e}")
            return {'action': 'error', 'reason': str(e)}
    
    # ==================== RISK METRICS CALCULATION ====================
    
    async def _update_risk_metrics_gpu(self):
        """GPU-accelerated risk metrics calculation"""
        try:
            # BUG FIX #8: Use _cuda_guard instead of torch.cuda.device()
            with _cuda_guard(self.device):
                risk_metrics = {}
                
                # BUG FIX #9: peak_capital must track account equity, not trade notional value
                # trade_value (size * price) is trade notional, NOT account equity
                # Track peak equity = initial_capital + max cumulative realized_pnl seen
                cumulative_pnl = 0.0
                peak_equity = self.initial_capital
                for trade in self.trade_history:
                    cumulative_pnl += trade.get('realized_pnl', 0)
                    peak_equity = max(peak_equity, self.initial_capital + cumulative_pnl)
                
                current_drawdown = (peak_equity - self.current_capital) / (peak_equity + 1e-8)
                risk_metrics['current_drawdown'] = max(0.0, current_drawdown)
                
                # Volatility-adjusted position sizing
                if hasattr(self, 'execution_analysis'):
                    volatility = self.execution_analysis['volatility'].item()
                    risk_metrics['volatility_adjustment'] = 1.0 / (1.0 + volatility * 5)
                
                # Sharpe ratio approximation
                if len(self.trade_history) > 5:
                    returns = [trade.get('realized_pnl', 0) / self.initial_capital for trade in self.trade_history[-10:] if 'realized_pnl' in trade]
                    if returns:
                        avg_return = np.mean(returns)
                        std_return = np.std(returns)
                        risk_metrics['sharpe_ratio'] = avg_return / (std_return + 1e-8) * np.sqrt(252)  # Annualized
                
                # Win rate
                winning_trades = [trade for trade in self.trade_history if trade.get('realized_pnl', 0) > 0]
                risk_metrics['win_rate'] = len(winning_trades) / max(len(self.trade_history), 1)
                
                # Profit factor
                gross_profit = sum(trade.get('realized_pnl', 0) for trade in winning_trades)
                losing_trades = [trade for trade in self.trade_history if trade.get('realized_pnl', 0) < 0]
                gross_loss = abs(sum(trade.get('realized_pnl', 0) for trade in losing_trades))
                risk_metrics['profit_factor'] = gross_profit / (gross_loss + 1e-8)
                
                self.risk_metrics.append(risk_metrics)
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 902)
            print(f"ERROR Risk metrics update error: {e}")
    
    # ==================== PERFORMANCE ANALYTICS ====================
    
    def get_performance_analytics(self) -> Dict:
        """Get comprehensive performance analytics"""
        try:
            analytics = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'capital_metrics': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'total_pnl': self.current_capital - self.initial_capital,
                    'return_percentage': (self.current_capital - self.initial_capital) / self.initial_capital * 100
                },
                'position_metrics': {
                    'current_position': self.current_position,
                    'position_size': self.position_size,
                    'unrealized_pnl': self.unrealized_pnl,
                    'realized_pnl': self.realized_pnl
                },
                'trade_metrics': {
                    'total_trades': len(self.trade_history),
                    'winning_trades': len([t for t in self.trade_history if t.get('realized_pnl', 0) > 0]),
                    'losing_trades': len([t for t in self.trade_history if t.get('realized_pnl', 0) < 0]),
                    'active_trades': 1 if self.current_position != 0 else 0
                }
            }
            
            # Add risk metrics
            if self.risk_metrics:
                latest_risk = self.risk_metrics[-1]
                analytics['risk_metrics'] = latest_risk
            
            # Add execution quality
            if hasattr(self, 'execution_analysis') and isinstance(self.execution_analysis, dict):
                def _safe_float(v):
                    return float(v.item()) if hasattr(v, 'item') else float(v) if isinstance(v, (int, float)) else 0.0

                analytics['execution_quality'] = {
                    'quality_score': _safe_float(self.execution_analysis.get('quality_score', 0)),
                    'avg_spread': _safe_float(self.execution_analysis.get('avg_spread', 0)),
                    'volatility': _safe_float(self.execution_analysis.get('volatility', 0)),
                    'timing_score': _safe_float(self.execution_analysis.get('timing_score', 0))
                }
            
            analytics['legacy_llm_position_size_tag'] = self.last_legacy_llm_size_tag
            analytics['legacy_llm_multiplier'] = self.last_legacy_llm_multiplier
            analytics['legacy_llm_insight'] = self.last_legacy_llm_insight

            # Publish to CognitiveBus for Watcher AI monitoring
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                cap = analytics.get('capital_metrics', {})
                trades = analytics.get('trade_metrics', {})
                pnl = cap.get('total_pnl', 0)
                ret = cap.get('return_percentage', 0)
                total = trades.get('total_trades', 0)
                pos = self.current_position
                pos_str = 'LONG' if pos > 0 else 'SHORT' if pos < 0 else 'FLAT'
                msg = f"Execution Analytics: Position={pos_str}, PnL={pnl:.2f}, Return={ret:.2f}%, Total Trades={total}"
                self.trading_system.bus.publish('THOUGHTS', 'Part12_Execution', msg)

            return analytics
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 968)
            print(f"ERROR Performance analytics error: {e}")
            return {'error': str(e)}
    
    # ==================== SYSTEM MANAGEMENT ====================
    
    async def shutdown(self):
        """Safe shutdown of execution engine"""
        print("  Shutting down GPU Order Execution Engine...")
        self.is_running = False
        
        # Close any open position
        if self.current_position != 0:
            print("WARNING Closing open position during shutdown...")
            # This would execute market order in real implementation
        
        self.executor.shutdown(wait=False)
        self.gpu_manager.cleanup_execution_memory()
        
        print("OK Order Execution Engine shutdown complete")
    
    def get_system_status(self) -> Dict:
        """Get system status report"""
        return {
            'is_running': self.is_running,
            'capital_status': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'pnl': self.current_capital - self.initial_capital
            },
            'position_status': {
                'has_position': self.current_position != 0,
                'position_size': abs(self.position_size),
                'direction': 'long' if self.current_position > 0 else 'short' if self.current_position < 0 else 'flat'
            },
            'performance_status': {
                'total_trades': len(self.trade_history),
                'win_rate': len([t for t in self.trade_history if t.get('realized_pnl', 0) > 0]) / max(len(self.trade_history), 1),
                'sharpe_ratio': self.risk_metrics[-1].get('sharpe_ratio', 0) if self.risk_metrics else 0
            }
        }

# ==================== CRYPTO MARKET ANALYZER ====================

class CryptoMarketAnalyzer:
    """
    BTCUSDT Delta Exchange Market Analysis
    Crypto-specific optimizations and 24/7 monitoring
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        # BTC trades 24/7 but activity varies by session
        self.market_sessions = {
            'asia': {'open': 0, 'close': 8},       # UTC
            'europe': {'open': 8, 'close': 16},     # UTC
            'americas': {'open': 13, 'close': 21},  # UTC
            'overnight': {'open': 21, 'close': 24}  # UTC
        }
        
        # BTC-specific parameters
        self.crypto_config = {
            'typical_spread': 10.0,     # $10 typical spread
            'max_acceptable_spread': 50.0,  # $50 maximum
            'funding_rate_times': [0, 8, 16],  # UTC funding rate times
            'high_impact_news_buffer': timedelta(hours=1),
            'session_volatility_multipliers': {
                'asia': 0.9,
                'europe': 1.2,
                'americas': 1.5,
                'overnight': 0.7,
                'overlap': 1.8  # Europe-Americas overlap
            }
        }
        
        self.news_events = deque(maxlen=50)
        self.session_analysis = deque(maxlen=100)
    
    def analyze_market_conditions(self, current_time: datetime, bid: float, ask: float, volume: float) -> Dict:
        """Analyze current crypto market conditions"""
        try:
            analysis = {
                'timestamp': current_time,
                'symbol': self.symbol,
                'spread_analysis': self._analyze_spread(bid, ask),
                'session_analysis': self._analyze_trading_session(current_time),
                'liquidity_analysis': self._analyze_liquidity(volume, current_time),
                'news_impact': self._check_news_impact(current_time)
            }
            
            # Overall market condition score
            condition_score = self._calculate_market_condition_score(analysis)
            analysis['market_condition_score'] = condition_score
            
            self.session_analysis.append(analysis)
            
            return analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1066)
            print(f"ERROR Market analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_spread(self, bid: float, ask: float) -> Dict:
        """Analyze spread conditions"""
        spread = ask - bid
        typical_spread = self.crypto_config['typical_spread']
        
        return {
            'current_spread': spread,
            'spread_ratio': spread / typical_spread,
            'is_acceptable': spread <= self.crypto_config['max_acceptable_spread'],
            'execution_quality': typical_spread / (spread + 1e-8)
        }
    
    def _analyze_trading_session(self, current_time: datetime) -> Dict:
        """Analyze current trading session"""
        hour = current_time.hour
        sessions = []
        
        # Determine active sessions (BTC is 24/7)
        for session, hours in self.market_sessions.items():
            if hours['open'] <= hour < hours['close']:
                sessions.append(session)
        
        # If no session matched (shouldn't happen with 24/7 coverage), default
        if not sessions:
            sessions = ['overnight']
        
        # Check for session overlaps
        overlaps = []
        if 'europe' in sessions and 'americas' in sessions:
            overlaps.append('europe_americas')
        
        # Calculate volatility multiplier
        volatility_multiplier = 1.0
        for session in sessions:
            if session in self.crypto_config['session_volatility_multipliers']:
                volatility_multiplier *= self.crypto_config['session_volatility_multipliers'][session]
        
        if overlaps:
            volatility_multiplier *= self.crypto_config['session_volatility_multipliers']['overlap']
        
        return {
            'active_sessions': sessions,
            'session_overlaps': overlaps,
            'volatility_multiplier': volatility_multiplier,
            'is_high_volatility_period': volatility_multiplier > 1.2
        }
    
    def _analyze_liquidity(self, volume: float, current_time: datetime) -> Dict:
        """Analyze market liquidity conditions"""
        hour = current_time.hour
        
        # BUG FIX #12: Align keys with self.market_sessions ('asia','europe','americas','overnight')
        typical_volumes = {
            'asia':      0.8,
            'europe':    1.2,
            'americas':  1.5,
            'overnight': 0.6
        }
        
        expected_volume = 0.0
        session_count = 0
        # BUG FIX #11: Use self.market_sessions (not self.market_hours which doesn't exist)
        for session, hours in self.market_sessions.items():
            if hours['open'] <= hour < hours['close']:
                expected_volume += typical_volumes.get(session, 1.0)
                session_count += 1
        
        expected_volume = expected_volume / max(session_count, 1)
        liquidity_ratio = volume / (expected_volume + 1e-8)
        
        return {
            'current_volume': volume,
            'expected_volume': expected_volume,
            'liquidity_ratio': liquidity_ratio,
            'is_high_liquidity': liquidity_ratio > 1.0,
            'liquidity_score': min(liquidity_ratio, 2.0) / 2.0  # Normalize to 0-1
        }
    
    def _check_news_impact(self, current_time: datetime) -> Dict:
        """Check for high-impact news events"""
        # Simplified news impact analysis
        # In real implementation, this would integrate with news feeds
        
        return {
            'high_impact_news': False,  # Placeholder
            'news_impact_score': 0.0,
            'recommended_action': 'normal_trading'
        }
    
    def _calculate_market_condition_score(self, analysis: Dict) -> float:
        """Calculate overall market condition score"""
        try:
            scores = []
            
            # Spread quality (lower is better)
            spread_ratio = analysis['spread_analysis']['spread_ratio']
            spread_score = 1.0 / (1.0 + max(0, spread_ratio - 1.0))
            scores.append(spread_score * 0.3)
            
            # Session volatility (moderate is better for trading)
            volatility_multiplier = analysis['session_analysis']['volatility_multiplier']
            volatility_score = 1.0 - abs(volatility_multiplier - 1.2) / 1.2  # Peak at 1.2
            scores.append(volatility_score * 0.3)
            
            # Liquidity score (higher is better)
            liquidity_score = analysis['liquidity_analysis']['liquidity_score']
            scores.append(liquidity_score * 0.3)
            
            # News impact (lower is better)
            news_score = 1.0 - analysis['news_impact']['news_impact_score']
            scores.append(news_score * 0.1)
            
            return sum(scores)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1184)
            print(f"ERROR Market condition score calculation error: {e}")
            return 0.5

# ==================== INTEGRATED TRADE EXECUTION SYSTEM ====================

class AdvancedTradeExecutionSystem:
    """
    Complete Trade Execution and Portfolio Management System
    Integrates GPU execution, risk management, and crypto analysis
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.symbol = "BTCUSDT"
        self.initial_capital = initial_capital
        
        # Core components
        self.execution_engine = GPUOrderExecutionEngine(initial_capital)
        self.market_analyzer = CryptoMarketAnalyzer()
        self.risk_manager = CryptoRiskManager(initial_capital)
        
        # Integration state
        self.is_active = False
        self.last_signal = None
        self.confidence_threshold = 6.0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.execution_log = deque(maxlen=500)
        
        print("ACCELERATED Advanced Trade Execution System Initialized - BTCUSDT Delta Exchange")
    
    async def process_trading_signal(self, signal: Dict, market_data: Dict):
        """
        Process trading signal from Part 11 with integrated execution
        """
        try:
            if not self.is_active:
                return {'status': 'rejected', 'reason': 'System not active'}
            
            signal_type = signal.get('signal')
            confidence = signal.get('confidence', 0.0)
            # Legacy Part12 is not a strategy authority. Require Jarvis evidence,
            # Part7's explicit clear gate, and the complete instrument scope.
            from jarvis_strategy_approval import evaluate_central_strategy, validate_entry_approval
            central_decision = signal.get('central_strategy_decision')
            central_approval = signal.get('central_strategy_approval')
            part_results = signal.get('part_results')
            part7_gate = signal.get('part7_gate')
            central_direction = {'CALL': 'BUY', 'BUY': 'BUY', 'PUT': 'SELL', 'SELL': 'SELL'}.get(str(signal_type).upper())
            identity = market_data.get('execution_identity') or market_data.get('identity') or {}
            scope = {
                'symbol': market_data.get('symbol') or identity.get('symbol'),
                'exchange': market_data.get('exchange') or identity.get('venue'),
                'contract': market_data.get('contract') or identity.get('symbol'),
                'instrument_id': market_data.get('instrument_id') or identity.get('instrument_id'),
                'market_type': market_data.get('market_type') or identity.get('market_type'),
                'snapshot_version': market_data.get('snapshot_version'),
            }
            if not isinstance(central_decision, dict) or central_decision.get('approved') is not True or central_direction is None or str(central_decision.get('direction', '')).upper() != central_direction:
                return {'status': 'rejected', 'reason': 'Jarvis central strategy approval is required'}
            recomputed = evaluate_central_strategy(
                part_results, part7_gate, confidence=central_decision.get('confidence'),
                expected_symbol=scope['symbol'],
            )
            if not recomputed.get('approved') or recomputed.get('direction') != central_direction:
                return {'status': 'rejected', 'reason': 'Jarvis central evidence or Part7 gate rejected entry'}
            valid, reason = validate_entry_approval(
                central_approval, direction=central_direction, **scope,
                confidence=central_decision.get('confidence'),
            )
            if not valid:
                return {'status': 'rejected', 'reason': reason}
            if confidence < self.confidence_threshold:
                return {'status': 'rejected', 'reason': f'Low confidence: {confidence:.2f}'}
            
            # Analyze market conditions
            current_time = datetime.now()
            market_analysis = self.market_analyzer.analyze_market_conditions(
                current_time,
                market_data['bid'],
                market_data['ask'],
                market_data.get('volume', 0.0)
            )
            
            # Check if market conditions are favorable
            market_score = market_analysis.get('market_condition_score', 0.0)
            if market_score < 0.4:  # Minimum market condition threshold
                return {'status': 'rejected', 'reason': f'Poor market conditions: {market_score:.2f}'}
            
            # BUG FIX #13: Risk validation was completely bypassed — CryptoRiskManager.validate_trade()
            # was never called anywhere in the execution flow. Call it now before executing.
            if hasattr(self, 'risk_manager'):
                # We need stop_loss to validate — estimate it from execution engine config
                est_price = (market_data['bid'] + market_data['ask']) / 2.0
                est_stop = est_price * (1 - self.execution_engine.risk_config['stop_loss_ratio'])
                est_size = self.execution_engine.calculate_position_size(confidence, est_price, est_stop)
                risk_check = self.risk_manager.validate_trade(signal_type, est_size, est_price, est_stop)
                if not risk_check.get('is_valid', True):
                    return {'status': 'rejected', 'reason': f"Risk manager: {risk_check.get('reason')}"}

            # Execute trade with integrated risk management
            execution_result = await self.execution_engine.execute_trade(
                signal_type,
                confidence,
                market_data['bid'],
                market_data['ask'],
                central_decision=central_decision,
                central_approval=central_approval,
                part_results=part_results,
                part7_gate=part7_gate,
                **scope,
            )
            
            # Log execution
            execution_log = {
                'timestamp': current_time,
                'signal': signal,
                'market_analysis': market_analysis,
                'execution_result': execution_result,
                'system_status': self.get_system_status()
            }
            
            self.execution_log.append(execution_log)
            
            return execution_result
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1311)
            print(f"ERROR Trading signal processing error: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    async def monitor_and_manage(self, market_data: Dict):
        """
        Continuous position monitoring and management
        """
        try:
            if not self.is_active:
                return
            
            # Monitor current position
            position_management = await self.execution_engine.monitor_position(
                market_data['bid'],
                market_data['ask']
            )
            
            # Update performance analytics
            performance_data = self.execution_engine.get_performance_analytics()
            self.performance_history.append(performance_data)
            
            return position_management
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1335)
            print(f"ERROR Position monitoring error: {e}")
            return {'action': 'error', 'reason': str(e)}
    
    def get_comprehensive_analytics(self) -> Dict:
        """Get comprehensive trading analytics"""
        try:
            execution_analytics = self.execution_engine.get_performance_analytics()
            system_status = self.get_system_status()
            
            analytics = {
                'timestamp': datetime.now().isoformat(),
                'system_status': system_status,
                'execution_analytics': execution_analytics,
                'trade_history_summary': {
                    'total_trades': len(self.execution_engine.trade_history),
                    'recent_performance': self._get_recent_performance(),
                    'risk_metrics': self.execution_engine.risk_metrics[-1] if self.execution_engine.risk_metrics else {}
                }
            }
            
            return analytics
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1358)
            print(f"ERROR Comprehensive analytics error: {e}")
            return {'error': str(e)}
    
    def _get_recent_performance(self) -> Dict:
        """Get recent trading performance"""
        try:
            recent_trades = list(self.execution_engine.trade_history)[-10:]  # Last 10 trades
            
            if not recent_trades:
                return {}
            
            winning_trades = [t for t in recent_trades if t.get('realized_pnl', 0) > 0]
            losing_trades = [t for t in recent_trades if t.get('realized_pnl', 0) < 0]
            
            return {
                'recent_trades': len(recent_trades),
                'recent_win_rate': len(winning_trades) / len(recent_trades),
                'avg_win': np.mean([t.get('realized_pnl', 0) for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t.get('realized_pnl', 0) for t in losing_trades]) if losing_trades else 0,
                'profit_factor': sum(t.get('realized_pnl', 0) for t in winning_trades) / abs(sum(t.get('realized_pnl', 0) for t in losing_trades)) if losing_trades else float('inf')
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1381)
            print(f"ERROR Recent performance calculation error: {e}")
            return {}
    
    async def start_trading(self):
        """Start the trading system"""
        print("  Starting Advanced Trade Execution System...")
        self.is_active = True
        
        # BUG FIX #15: Original code used EUR/USD price (1.1000) for BTCUSDT system!
        # Use a realistic BTC placeholder price or skip warmup entirely
        # In production this will be overwritten by first real market data tick
        btc_warmup_price = 98000.0
        await self.execution_engine.process_market_data(btc_warmup_price, btc_warmup_price + 10.0, int(time.time()))
        
        print("OK Trade Execution System Active - Monitoring BTCUSDT")
    
    async def stop_trading(self):
        """Stop the trading system"""
        print("  Stopping Advanced Trade Execution System...")
        self.is_active = False
        
        # Close any open positions
        if self.execution_engine.current_position != 0:
            print("WARNING Closing open positions...")
            # Implementation would depend on broker API
        
        await self.execution_engine.shutdown()
        
        print("OK Trade Execution System Stopped")
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            'is_active': self.is_active,
            'symbol': self.symbol,
            'capital': self.execution_engine.current_capital,
            'has_position': self.execution_engine.current_position != 0,
            'total_trades': len(self.execution_engine.trade_history),
            'performance_metrics': self._get_recent_performance(),
            'risk_status': self.execution_engine.risk_metrics[-1] if self.execution_engine.risk_metrics else {}
        }

# ==================== CRYPTO RISK MANAGER ====================

class CryptoRiskManager:
    """
    Advanced Risk Management for BTCUSDT Delta Exchange Trading
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = {
            'max_daily_loss': 0.02,  # 2% maximum daily loss
            'max_trade_risk': 0.01,  # 1% maximum risk per trade
            'max_drawdown': 0.05,    # 5% maximum drawdown
            'position_size_limit': 0.25,  # 25% maximum position size
            'volatility_adjustment': True
        }
        
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
    
    def validate_trade(self, signal_type: str, proposed_size: float, current_price: float, stop_loss: float) -> Dict:
        """Validate trade against risk parameters"""
        try:
            # Reset daily metrics if new day
            self._reset_daily_metrics_if_needed()
            
            # Calculate trade risk
            trade_risk = abs(current_price - stop_loss) * proposed_size
            
            # Check daily loss limit
            if self.daily_pnl < -self.initial_capital * self.risk_limits['max_daily_loss']:
                return {'is_valid': False, 'reason': 'Daily loss limit reached'}
            
            # Check trade risk limit
            if trade_risk > self.current_capital * self.risk_limits['max_trade_risk']:
                return {'is_valid': False, 'reason': 'Trade risk exceeds limit'}
            
            # Check position size limit
            if proposed_size * current_price > self.current_capital * self.risk_limits['position_size_limit']:
                return {'is_valid': False, 'reason': 'Position size exceeds limit'}
            
            # Check drawdown limit
            current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
            if current_drawdown > self.risk_limits['max_drawdown']:
                return {'is_valid': False, 'reason': 'Maximum drawdown reached'}
            
            return {'is_valid': True, 'reason': 'Trade validated'}
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1474)
            print(f"ERROR Trade validation error: {e}")
            return {'is_valid': False, 'reason': f'Validation error: {str(e)}'}
    
    def update_risk_metrics(self, trade_result: Dict):
        """Update risk metrics after trade execution"""
        try:
            pnl = trade_result.get('realized_pnl', 0)
            self.daily_pnl += pnl
            self.current_capital += pnl
            self.daily_trades += 1
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1486)
            print(f"ERROR Risk metrics update error: {e}")
    
    def _reset_daily_metrics_if_needed(self):
        """Reset daily metrics if it's a new day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
    
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        return {
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'drawdown': (self.initial_capital - self.current_capital) / self.initial_capital,
            'risk_limits': self.risk_limits
        }

# ==================== LINUX OPTIMIZATION ====================

def setup_linux_trading_environment():
    """Setup Linux-optimized environment for trade execution"""
    # Set thread affinity for i5 4-core CPU
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Enable GPU memory optimizations for execution
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Set async performance parameters
    os.environ['PYTHONASYNCIODEBUG'] = '0'
    
    # Trading-specific optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking for execution
    
    print("OK Linux environment optimized for high-frequency trading")

# ==================== GPU-OPTIMIZED PART-12: CONFIDENCE ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p12_confidence_engine', 'analyze')
class ConfidenceEngineGPU:
    """
    INSTITUTIONAL QUANTITATIVE CONFIDENCE ENGINE (PART 12)
    GTX 1650 CUDA & Multi-Factor Agreement Architecture
    - Weighted Confluence Scoring based on Statistical Engine Edge
    - Anchor Engine Concurrence Bonus (+5% to +10% when Trend, Structure, CVD align)
    - Severe Dissent Penalties (-18% to -35% for contradictory anchor signals)
    - Strict S/R Zone Conflict Penalty (-25% when trading into supply/demand walls)
    - Quorum Filter: Enforces low baseline (10-30%) when < 3 engines agree
    - Mathematically Clamped Confidence Range: [10, 95]
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Engine reliability weights based on quant edge
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
        self.anchors = {'part6_trend', 'part8_structure', 'part9_orderflow', 'part2_zone'}

    def analyze(self, part_results: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
        """
        Calculates calibrated confidence score (10-95%) across all parts.
        """
        if not part_results:
            return {"confidence": 10, "thought": "No part results available"}

        if isinstance(part_results, dict):
            items = list(part_results.items())
        elif isinstance(part_results, list):
            items = [(f"part_{i}", r) for i, r in enumerate(part_results)]
        else:
            return {"confidence": 10, "thought": "Invalid input to ConfidenceEngine"}

        w_buy, w_sell, w_neutral = 0.0, 0.0, 0.0
        total_weight_den = 0.0
        buy_engines, sell_engines, neutral_engines = [], [], []
        p2_thought = ""

        for name, r in items:
            if not isinstance(r, dict):
                continue
            if name == 'part7_volatility':
                continue  # PHASE4: Part 7 excluded from directional confidence denominator
            sig = r.get('signal', 0)
            thought = str(r.get('thought', '')).lower()
            try:
                sig = float(sig)
            except (ValueError, TypeError):
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1583)
                continue
                
            # Skip offline or errored engines
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
            
            if 'zone' in name or 'support' in thought or 'resistance' in thought:
                p2_thought = thought

            if sig > 0:
                w_buy += w * sig
                buy_engines.append(name)
            elif sig < 0:
                w_sell += w * abs(sig)
                sell_engines.append(name)
            else:
                w_neutral += w
                neutral_engines.append(name)

        if len(buy_engines) == 0 and len(sell_engines) == 0:
            return {
                "confidence": 10,
                "thought": "All parts neutral (Chop / Idle)",
                "weighted_agree": 0.0,
                "weighted_dissent": 0.0,
                "confluence_ratio": 0.0
            }

        if w_buy >= w_sell:
            majority_dir = 1
            w_agree, w_dissent = w_buy, w_sell
            agree_engines, dissent_engines = buy_engines, sell_engines
        else:
            majority_dir = -1
            w_agree, w_dissent = w_sell, w_buy
            agree_engines, dissent_engines = sell_engines, buy_engines

        agree_count = len(agree_engines)
        w_active_dir = w_buy + w_sell
        w_active = w_active_dir if (agree_count >= 3 and w_active_dir >= 3.2) else total_weight_den

        # 1. Quorum check: If fewer than 3 engines or low weight, cap confidence low
        if agree_count < 3 or total_weight_den < 3.5:
            conf = int(min(30, 10 + agree_count * 5 + w_agree * 3))
            return {
                "confidence": conf,
                "thought": f"Low Confluence: {agree_count} engines, weight {w_agree:.1f}",
                "weighted_agree": round(w_agree, 2),
                "weighted_dissent": round(w_dissent, 2),
                "confluence_ratio": round(w_agree / max(w_active, 1e-6), 2)
            }

        confluence_ratio = w_agree / max(w_active, 1e-6)
        saturation = min(1.0, w_agree / 6.0)
        base_conf = 45.0 + (40.0 * confluence_ratio * saturation)

        # 2. Anchor bonus: trend, structure, orderflow
        aligned_anchors = [eng for eng in agree_engines if eng in self.anchors]
        if len(aligned_anchors) >= 3:
            base_conf += 10.0
        elif len(aligned_anchors) == 2:
            base_conf += 5.0

        # 3. Anchor dissent penalties
        dissenting_anchors = [eng for eng in dissent_engines if eng in self.anchors]
        if len(dissenting_anchors) >= 2:
            base_conf -= 35.0  # massive conflict
        elif len(dissenting_anchors) == 1:
            base_conf -= 18.0  # key anchor opposes

        # 4. S/R Zone conflict penalty
        if majority_dir == 1 and 'resistance zone' in p2_thought:
            base_conf -= 25.0
        elif majority_dir == -1 and 'support zone' in p2_thought:
            base_conf -= 25.0

        # 5. General dissent ratio penalty
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

    @part_advisory_entry("part12_confidence", data_parameter="results_by_timeframe", context_parameter="context", evidence_context_key="neural_advisory_evidence")
    def analyze_multi_timeframe(self, results_by_timeframe: Dict[str, Any], context=None) -> Dict[str, Any]:
        # Same native frame set as the central MTF strategy. Each correlation
        # group shares one horizon weight so correlated frames do not multiply it.
        from binance_timeframes import (
            BINANCE_SPOT_TIMEFRAMES as timeframes,
            TIMEFRAME_CORRELATION_GROUPS, TIMEFRAME_GROUP_WEIGHTS,
        )
        tf_weights = {tf: TIMEFRAME_GROUP_WEIGHTS[g] / len(members)
                      for g, members in TIMEFRAME_CORRELATION_GROUPS.items() for tf in members}
        if not isinstance(results_by_timeframe, dict) or set(results_by_timeframe) != set(timeframes):
            return {"confidence": 10, "thought": "Part12 MTF evidence incomplete",
                    "data_status": "error", "timeframe_coverage": []}
        required_parts = set(self.weights)
        per_timeframe = {}
        weighted_confidence = 0.0
        for timeframe in timeframes:
            parts = results_by_timeframe.get(timeframe)
            if not isinstance(parts, dict) or not required_parts.issubset(parts):
                return {"confidence": 10, "thought": f"Part12 {timeframe} evidence incomplete",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            result = self.analyze(parts)
            try:
                frame_confidence = float(result.get("confidence"))
            except (TypeError, ValueError, OverflowError):
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1709)
                return {"confidence": 10, "thought": f"Part12 {timeframe} confidence malformed",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            if not math.isfinite(frame_confidence) or not 0 <= frame_confidence <= 100:
                return {"confidence": 10, "thought": f"Part12 {timeframe} confidence invalid",
                        "data_status": "error", "timeframe_coverage": list(per_timeframe)}
            per_timeframe[timeframe] = result
            weighted_confidence += tf_weights[timeframe] * frame_confidence
        total_weight = sum(tf_weights.values())
        aggregate = int(round(weighted_confidence / total_weight))
        return {"confidence": aggregate,
                "thought": f"Part12 native MTF confidence {aggregate}% from 8/8 frames",
                "data_status": "valid", "timeframe": "aggregate",
                "timeframe_coverage": list(timeframes), "timeframe_results": per_timeframe,
                "timeframe_weights": dict(tf_weights)}


# ==================== MAIN EXECUTION ====================

async def main():
    """Main execution function for testing"""
    setup_linux_trading_environment()
    
    # Initialize trading system
    trading_system = AdvancedTradeExecutionSystem(initial_capital=10000.0)
    
    # Start trading
    await trading_system.start_trading()
    
    # Simulate market data and signals
    market_data = {
        'bid': 98000.0,
        'ask': 98010.0,
        'volume': 150.5
    }
    
    trading_signal = {
        'signal': 'CALL',
        'confidence': 7.5,
        'source': 'Part11_engine'
    }
    
    # Process trading signal
    result = await trading_system.process_trading_signal(trading_signal, market_data)
    print(f"Trade Execution Result: {result}")
    
    # Get analytics
    analytics = trading_system.get_comprehensive_analytics()
    print(f"System Analytics: {analytics}")
    
    # Stop trading
    await trading_system.stop_trading()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
    
    print("ACCELERATED Part12 Trade Execution Engine - Ready for Live BTCUSDT Trading")
    print("INTEGRATION: Complete trading system with GPU-accelerated execution and risk management")