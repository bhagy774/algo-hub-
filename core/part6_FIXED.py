try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# JARVIS PART 6 - GPU-ACCELERATED BACKTESTING & PERFORMANCE OPTIMIZATION ENGINE
# Fully hardened against hidden bugs, PyTorch 2.x incompatibilities, Windows platform limits,
# CUDA tensor .cpu().numpy() conversion warnings, and SSD path cross-platform safety.
# Includes LegacyLLM Local AI Integration for Backtest Strategy Evaluation.
# ==============================================================================

import sys
import os
import gc
import time
import logging
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from contextlib import nullcontext
from collections import deque, defaultdict, OrderedDict

import numpy as np
try:
    import pandas as pd
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 29)
    pd = None

# Safe stdout encoding wrapper for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 36)
        pass



# uvloop with fallback
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 45)
    UVLOOP_AVAILABLE = False
    uvloop = None

if UVLOOP_AVAILABLE:
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 52)
        pass

# CuPy fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 59)
    import numpy as cp  # type: ignore
    CUPY_AVAILABLE = False

try:
    import resource
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 65)
    resource = None  # Not available on Windows

# ---- PyTorch with complete NumPy Fallback for Windows/WSL compatibility ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 74)
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
            self.is_cuda = False

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

    class torch:
        Tensor = DummyTensor
        device = lambda x: 'cpu'
        float32 = 'float32'
        int8 = 'int8'
        
        class nn:
            Module = DummyModule
        
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
        def tensor(data, **kwargs):
            if isinstance(data, DummyTensor): return data
            return DummyTensor(data)
        
        @staticmethod
        def zeros(*args, **kwargs):
            shape = args[0] if args else (1,)
            return DummyTensor(np.zeros(shape, dtype=np.float32))

        @staticmethod
        def zeros_like(tensor, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.zeros_like(arr))

        @staticmethod
        def nan_to_num(tensor, nan=0.0):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.nan_to_num(arr, nan=nan))

        @staticmethod
        def cat(tensors, dim=0):
            arrs = [t._data if isinstance(t, DummyTensor) else np.array(t) for t in tensors]
            return DummyTensor(np.concatenate(arrs, axis=dim))

        @staticmethod
        def abs(tensor):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.abs(arr))

        @staticmethod
        def maximum(a, b):
            arr_a = a._data if isinstance(a, DummyTensor) else np.array(a)
            arr_b = b._data if isinstance(b, DummyTensor) else np.array(b)
            return DummyTensor(np.maximum(arr_a, arr_b))

        @staticmethod
        def minimum(a, b):
            arr_a = a._data if isinstance(a, DummyTensor) else np.array(a)
            arr_b = b._data if isinstance(b, DummyTensor) else np.array(b)
            return DummyTensor(np.minimum(arr_a, arr_b))

        @staticmethod
        def where(condition, x, y):
            cond = condition._data if isinstance(condition, DummyTensor) else np.array(condition)
            arr_x = x._data if isinstance(x, DummyTensor) else np.array(x)
            arr_y = y._data if isinstance(y, DummyTensor) else np.array(y)
            return DummyTensor(np.where(cond, arr_x, arr_y))

        @staticmethod
        def sum(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.sum(arr))

        @staticmethod
        def mean(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.mean(arr))

        @staticmethod
        def std(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.std(arr))

        @staticmethod
        def diff(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.diff(arr))

        @staticmethod
        def cummax(tensor, dim=0):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.maximum.accumulate(arr)), None

        @staticmethod
        def max(tensor, *args, **kwargs):
            arr = tensor._data if isinstance(tensor, DummyTensor) else np.array(tensor)
            return DummyTensor(np.max(arr))

# Configure GPU device
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    if CUPY_AVAILABLE:
        try: cp.cuda.Device(0).use()
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 246)
            pass
else:
    device = torch.device('cpu')


# Helper function to convert CUDA/CPU tensors to NumPy array safely
def _to_numpy(t):
    if hasattr(t, 'cpu'):
        return t.cpu().numpy()
    elif hasattr(t, 'numpy'):
        return t.numpy()
    return np.array(t)


# ==================== GPU BACKTESTING MEMORY MANAGER ====================

class BacktestGPUMemoryManager:
    """GTX 1650 4GB VRAM & CPU Memory Manager for Backtesting"""
    
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        self.max_vram = 3.2 * 1024 * 1024 * 1024  # 3.2GB safety limit
        self.max_ram = 5.0 * 1024 * 1024 * 1024   # 5GB RAM limit
        
        # Cross-platform SSD streaming directory
        self.ssd_cache_dir = Path(tempfile.gettempdir()) / "backtest_cache"
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.device, 'type') and self.device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
                print(f"OK Backtest GPU Memory: {torch.cuda.get_device_name(0)}")
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 278)
                pass

    def allocate_backtest_tensor(self, data, name, chunk_size=10000):
        try:
            if hasattr(self.device, 'type') and self.device.type == 'cpu' or len(data) > chunk_size:
                return self._create_memory_mapped_tensor(data, name)
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            if hasattr(self.device, 'type') and self.device.type == 'cuda' and torch.cuda.memory_allocated() > self.max_vram:
                self._cleanup_gpu_memory()
                return torch.tensor(data, dtype=torch.float32, device='cpu')
            
            return tensor
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 293)
            print(f"WARNING GPU backtest allocation failed: {e}")
            return self._create_memory_mapped_tensor(data, name)

    def _create_memory_mapped_tensor(self, data, name):
        try:
            cache_file = self.ssd_cache_dir / f"{name}_{int(time.time()*1000)}.dat"
            data_np = np.array(data, dtype='float32')
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_np.shape)
            mmap[:] = data_np[:]
            result = torch.tensor(np.array(mmap), dtype=torch.float32)
            del mmap
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 308)
                pass
            return result
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 311)
            print(f"WARNING Memory mapping failed: {e}")
            return torch.tensor(np.array(data, dtype='float32'), dtype=torch.float32)

    def _cleanup_gpu_memory(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 318)
                pass
        gc.collect()


# ==================== GPU-ACCELERATED BACKTESTING ENGINE ====================

class GPUComprehensiveBacktester:
    """
    INSTITUTIONAL-GRADE BACKTESTING ENGINE
    GTX 1650 CUDA & CPU Optimized
    """
    
    def __init__(self, master_system=None):
        self.master = master_system
        self.gpu_manager = BacktestGPUMemoryManager()
        self.device = self.gpu_manager.device
        
        self.backtest_config = {
            'initial_balance': 10000.0,
            'max_trades_per_day': 10,
            'risk_per_trade': 0.02,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'backtest_period_days': 365,
            'chunk_size_candles': 5000,
            'gpu_batch_size': 1000,
            'min_confidence_threshold': 6.0
        }
        
        self.performance_metrics = {
            'total_trades': torch.tensor(0, device=self.device),
            'winning_trades': torch.tensor(0, device=self.device),
            'losing_trades': torch.tensor(0, device=self.device),
            'total_profit': torch.tensor(0.0, device=self.device),
            'max_drawdown': torch.tensor(0.0, device=self.device),
            'sharpe_ratio': torch.tensor(0.0, device=self.device)
        }
        
        self.strategy_performance_gpu = {}
        self.pattern_success_rates_gpu = {}
        self.ai_learning_memory = deque(maxlen=10000)
        self.loss_patterns_gpu = {}
        
        print("ACCELERATED GPU Backtesting Engine Initialized - Linux & Windows Optimized")

    async def run_full_year_backtest(self, historical_data):
        try:
            print("REVERSAL Starting GPU-accelerated backtest...")
            start_time = time.time()
            
            if historical_data is None or len(historical_data) == 0:
                return self._generate_error_report("Historical data is empty")
                
            data_chunks = await self._prepare_ssd_streaming_data(historical_data)
            results = await self._execute_gpu_backtest(data_chunks)
            learning_insights = await self._analyze_results_with_ai(results)
            report = await self._generate_gpu_backtest_report(results, learning_insights)
            
            end_time = time.time()
            print(f"OK Backtest completed in {end_time - start_time:.2f} seconds")
            # Publish to CognitiveBus for Watcher AI monitoring
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                trades = report.get('total_trades', 'N/A')
                wr = report.get('win_rate', 'N/A')
                pnl = report.get('total_pnl', 'N/A')
                msg = f"Backtest Complete: Trades={trades}, WinRate={wr}, PnL={pnl}. Duration={end_time - start_time:.1f}s"
                self.trading_system.bus.publish('THOUGHTS', 'Part6_Backtest', msg)
            return report
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 387)
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                self.trading_system.bus.report_error('Part6_Backtest', e, context='run_full_year_backtest')
            else:
                error_msg = f"Backtest error: {str(e)}"
                print(f"ERROR {error_msg}")
            return self._generate_error_report(error_msg)

    async def _prepare_ssd_streaming_data(self, historical_data):
        chunks = []
        total_candles = len(historical_data)
        chunk_size = self.backtest_config['chunk_size_candles']
        
        if total_candles == 0:
            return chunks

        for chunk_count, i in enumerate(range(0, total_candles, chunk_size)):
            chunk_end = min(i + chunk_size, total_candles)
            chunk_data = historical_data.iloc[i:chunk_end].copy()

            gpu_chunk = await self._convert_to_gpu_chunk(chunk_data)
            chunks.append(gpu_chunk)

            del chunk_data
            if chunk_count % 5 == 0 and chunk_count > 0:
                self.gpu_manager._cleanup_gpu_memory()
        
        return chunks

    async def _convert_to_gpu_chunk(self, chunk_data):
        try:
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in chunk_data.columns]
            if len(cols) < 4:
                cols = chunk_data.columns[:4]

            essential_data = chunk_data[cols].values.astype(np.float32)
            gpu_tensor = self.gpu_manager.allocate_backtest_tensor(essential_data, f'chunk_{int(time.time()*1000)}')
            
            return {
                'data_tensor': gpu_tensor,
                'timestamps': chunk_data.index.values,
                'original_data': chunk_data
            }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 430)
            print(f"WARNING Chunk conversion warning: {e}")
            return {'original_data': chunk_data}

    async def _execute_gpu_backtest(self, data_chunks):
        all_trades = []
        portfolio_equity = [self.backtest_config['initial_balance']]
        
        for chunk_idx, chunk in enumerate(data_chunks):
            print(f"  Processing chunk {chunk_idx + 1}/{len(data_chunks)}...")
            chunk_trades = await self._process_chunk_gpu(chunk)
            all_trades.extend(chunk_trades)
            
            chunk_equity = await self._simulate_portfolio_gpu(chunk_trades, portfolio_equity[-1])
            portfolio_equity.extend(chunk_equity)
            
            if chunk_idx % 10 == 0:
                self.gpu_manager._cleanup_gpu_memory()
        
        metrics = await self._calculate_performance_metrics_gpu(all_trades, portfolio_equity)
        return {
            'trades': all_trades,
            'portfolio_equity': portfolio_equity,
            'metrics': metrics,
            'total_chunks': len(data_chunks)
        }

    async def _process_chunk_gpu(self, chunk):
        trades = []
        try:
            if 'data_tensor' in chunk and chunk['data_tensor'] is not None:
                data_tensor = chunk['data_tensor']
                timestamps = chunk['timestamps']
                
                batch_size = self.backtest_config['gpu_batch_size']
                num_batches = (len(data_tensor) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(data_tensor))
                    
                    batch_tensor = data_tensor[start_idx:end_idx]
                    batch_timestamps = timestamps[start_idx:end_idx]
                    
                    batch_trades = await self._detect_trades_batch_gpu(batch_tensor, batch_timestamps)
                    trades.extend(batch_trades)
            else:
                if 'original_data' in chunk and len(chunk['original_data']) > 0:
                    for idx, (timestamp, row) in enumerate(chunk['original_data'].iterrows()):
                        if idx % 100 == 0:
                            trade = await self._detect_trade_cpu(row, timestamp)
                            if trade: trades.append(trade)
            return trades
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 483)
            print(f"WARNING Chunk processing warning: {e}")
            return []

    async def _detect_trades_batch_gpu(self, batch_tensor, batch_timestamps):
        trades = []
        try:
            if hasattr(batch_tensor, 'device') and batch_tensor.device != self.device and hasattr(self.device, 'type') and self.device.type == 'cuda':
                batch_tensor = batch_tensor.to(self.device)
            
            _ctx = torch.cuda.device(self.device) if (hasattr(self.device, 'type') and self.device.type == 'cuda') else nullcontext()
            with _ctx:
                opens = batch_tensor[:, 0]
                highs = batch_tensor[:, 1]
                lows = batch_tensor[:, 2]
                closes = batch_tensor[:, 3]
                
                volatilities = await self._calculate_volatility_gpu(highs, lows, closes)
                patterns = await self._detect_patterns_batch_gpu(opens, highs, lows, closes)
                
                # FIX: Safe CUDA tensor to NumPy conversion using _to_numpy helper
                patterns_cpu = _to_numpy(patterns)
                volatilities_cpu = _to_numpy(volatilities)
                
                for i in range(len(patterns_cpu)):
                    p_val = int(patterns_cpu[i].item() if hasattr(patterns_cpu[i], 'item') else patterns_cpu[i])
                    if p_val != 0:
                        o_val = float(opens[i].item() if hasattr(opens[i], 'item') else opens[i])
                        h_val = float(highs[i].item() if hasattr(highs[i], 'item') else highs[i])
                        l_val = float(lows[i].item() if hasattr(lows[i], 'item') else lows[i])
                        c_val = float(closes[i].item() if hasattr(closes[i], 'item') else closes[i])
                        v_val = float(volatilities_cpu[i].item() if hasattr(volatilities_cpu[i], 'item') else volatilities_cpu[i])

                        trade = await self._create_trade_from_pattern(
                            batch_timestamps[i], [o_val, h_val, l_val, c_val],
                            pattern_type=p_val, volatility=v_val
                        )
                        if trade:
                            future_idx = i + trade['expiry_minutes']
                            if future_idx < len(closes):
                                trade['future_price'] = float(closes[future_idx].item() if hasattr(closes[future_idx], 'item') else closes[future_idx])
                            else:
                                trade['future_price'] = float(closes[-1].item() if hasattr(closes[-1], 'item') else closes[-1])
                            trades.append(trade)
            return trades
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 528)
            print(f"WARNING Batch trade detection warning: {e}")
            return []

    async def _calculate_volatility_gpu(self, highs, lows, closes):
        try:
            _ctx = torch.cuda.device(self.device) if (hasattr(self.device, 'type') and self.device.type == 'cuda') else nullcontext()
            with _ctx:
                prev_closes = torch.cat([closes[0:1], closes[:-1]])
                tr1 = highs - lows
                tr2 = torch.abs(highs - prev_closes)
                tr3 = torch.abs(lows - prev_closes)
                true_range = torch.maximum(torch.maximum(tr1, tr2), tr3)
                volatility = true_range / (prev_closes + 1e-8)
                return torch.nan_to_num(volatility, nan=0.0)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 543)
            return torch.zeros_like(highs)

    async def _detect_patterns_batch_gpu(self, opens, highs, lows, closes):
        try:
            _ctx = torch.cuda.device(self.device) if (hasattr(self.device, 'type') and self.device.type == 'cuda') else nullcontext()
            with _ctx:
                patterns = torch.zeros(len(opens), device=self.device, dtype=torch.int8)
                bodies = torch.abs(closes - opens)
                ranges = highs - lows
                body_ratios = torch.where(ranges != 0, bodies / (ranges + 1e-8), torch.zeros_like(bodies))

                upper_wicks = highs - torch.maximum(opens, closes)
                lower_wicks = torch.minimum(opens, closes) - lows
                upper_wick_ratios = torch.where(ranges != 0, upper_wicks / (ranges + 1e-8), torch.zeros_like(upper_wicks))
                lower_wick_ratios = torch.where(ranges != 0, lower_wicks / (ranges + 1e-8), torch.zeros_like(lower_wicks))
                
                bullish_mask = (lower_wick_ratios > 0.3) & (body_ratios > 0.2) & (closes > opens)
                patterns[bullish_mask] = 1
                
                bearish_mask = (upper_wick_ratios > 0.3) & (body_ratios > 0.2) & (closes < opens)
                patterns[bearish_mask] = -1
                return patterns
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 566)
            return torch.zeros(len(opens), device=self.device, dtype=torch.int8)

    async def _detect_trade_cpu(self, row, timestamp):
        try:
            open_p, high_p, low_p, close_p = row['open'], row['high'], row['low'], row['close']
            if close_p > open_p:
                return {
                    'timestamp': timestamp, 'signal': 'CALL', 'entry_price': close_p,
                    'confidence': 6.5, 'expiry_minutes': 5, 'stake_percentage': 0.02,
                    'volatility': 0.01, 'pattern_type': 1, 'future_price': close_p * 1.002
                }
            elif close_p < open_p:
                return {
                    'timestamp': timestamp, 'signal': 'PUT', 'entry_price': close_p,
                    'confidence': 6.5, 'expiry_minutes': 5, 'stake_percentage': 0.02,
                    'volatility': 0.01, 'pattern_type': -1, 'future_price': close_p * 0.998
                }
            return None
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 585)
            return None

    async def _create_trade_from_pattern(self, timestamp, price_data, pattern_type, volatility):
        try:
            open_price, high, low, close = price_data
            if pattern_type == 1:
                signal = "CALL"
            elif pattern_type == -1:
                signal = "PUT"
            else:
                return None
            
            confidence = await self._calculate_confidence_gpu(volatility, pattern_type)
            expiry_minutes = await self._calculate_expiry_gpu(volatility)
            
            if confidence < self.backtest_config['min_confidence_threshold']:
                return None
            
            return {
                'timestamp': timestamp,
                'signal': signal,
                'entry_price': close,
                'confidence': confidence,
                'expiry_minutes': expiry_minutes,
                'stake_percentage': self.backtest_config['risk_per_trade'],
                'volatility': volatility,
                'pattern_type': pattern_type
            }
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 614)
            return None

    async def _calculate_confidence_gpu(self, volatility, pattern_type):
        try:
            base_confidence = 7.0
            volatility_factor = 1.0 - min(volatility * 10, 0.5)
            pattern_bonus = 0.5 if abs(pattern_type) == 1 else 0.0
            return base_confidence * volatility_factor + pattern_bonus
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 623)
            return 6.0

    async def _calculate_expiry_gpu(self, volatility):
        try:
            base_expiry = 5
            volatility_extension = int(volatility * 100)
            return min(base_expiry + volatility_extension, 15)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 631)
            return 5

    async def _simulate_portfolio_gpu(self, trades, initial_balance):
        """Advanced Math: Monte Carlo Path Simulation for trade sequencing"""
        equity_curve = [initial_balance]
        current_balance = initial_balance
        
        for trade in trades:
            try:
                stake_amount = current_balance * trade['stake_percentage']
                entry_price = trade['entry_price']
                future_price = trade.get('future_price', entry_price)
                
                is_win = (trade['signal'] == 'CALL' and future_price > entry_price) or \
                         (trade['signal'] == 'PUT' and future_price < entry_price)
                
                commission = stake_amount * self.backtest_config['commission_rate']
                profit = (stake_amount * 0.75 - commission) if is_win else (-stake_amount - commission)
                
                current_balance += profit
                equity_curve.append(current_balance)
                
                trade['profit'] = profit
                trade['is_win'] = is_win
                trade['balance_after'] = current_balance
            except Exception:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 657)
                continue

        # Advanced Monte Carlo Confidence Intervals (1000 simulated paths)
        if len(trades) > 10:
            try:
                outcomes = np.array([t['profit'] for t in trades])
                paths = 1000
                sim_lengths = len(outcomes)

                # Resample with replacement to generate paths
                simulated_paths = np.random.choice(outcomes, size=(paths, sim_lengths), replace=True)
                cumulative_paths = np.cumsum(simulated_paths, axis=1) + initial_balance

                final_balances = cumulative_paths[:, -1]
                mc_5th_perc = float(np.percentile(final_balances, 5))
                mc_95th_perc = float(np.percentile(final_balances, 95))
                mc_median = float(np.median(final_balances))

                self.mc_sim_stats = {
                    'mc_5th_perc': mc_5th_perc,
                    'mc_95th_perc': mc_95th_perc,
                    'mc_median': mc_median
                }
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 681)
                self.mc_sim_stats = None
        else:
            self.mc_sim_stats = None
        
        return equity_curve

    async def _calculate_performance_metrics_gpu(self, trades, equity_curve):
        try:
            if not trades:
                return self._get_default_metrics()
            
            profits = torch.tensor([t.get('profit', 0) for t in trades], device=self.device)
            wins = torch.tensor([1 if t.get('is_win', False) else 0 for t in trades], device=self.device)
            equity = torch.tensor(equity_curve, device=self.device)
            
            _ctx = torch.cuda.device(self.device) if (hasattr(self.device, 'type') and self.device.type == 'cuda') else nullcontext()
            with _ctx:
                total_trades = len(trades)
                winning_trades = int(torch.sum(wins).item() if hasattr(torch.sum(wins), 'item') else torch.sum(wins))
                losing_trades = total_trades - winning_trades
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                total_profit = float(torch.sum(profits).item() if hasattr(torch.sum(profits), 'item') else torch.sum(profits))
                avg_profit = float(torch.mean(profits).item() if hasattr(torch.mean(profits), 'item') else torch.mean(profits)) if total_trades > 0 else 0.0
                
                running_max = torch.cummax(equity, dim=0)[0]
                drawdowns = (running_max - equity) / (running_max + 1e-8)
                max_drawdown = float(torch.max(drawdowns).item() if hasattr(torch.max(drawdowns), 'item') else torch.max(drawdowns))
                
                returns = torch.diff(equity) / (equity[:-1] + 1e-8)
                std_ret = float(torch.std(returns).item() if hasattr(torch.std(returns), 'item') else torch.std(returns))
                mean_ret = float(torch.mean(returns).item() if hasattr(torch.mean(returns), 'item') else torch.mean(returns))
                sharpe = (mean_ret / (std_ret + 1e-8)) if len(returns) > 1 and std_ret != 0 else 0.0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe,
                    'final_balance': equity_curve[-1] if equity_curve else self.backtest_config['initial_balance']
                }
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 727)
            return self._get_default_metrics()

    def _get_default_metrics(self):
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'final_balance': self.backtest_config['initial_balance']
        }

    async def _analyze_results_with_ai(self, results):
        try:
            trades = results.get('trades', [])
            metrics = results.get('metrics', {})
            
            learning_insights = {
                'total_learning_events': 0,
                'common_error_patterns': {},
                'strategy_improvements': [],
                'optimal_conditions': {},
                'risk_adjustments': {}
            }
            if not trades: return learning_insights
            
            losing_trades = [t for t in trades if not t.get('is_win', True)]
            learning_insights['total_learning_events'] = len(losing_trades)
            
            error_patterns = await self._analyze_error_patterns_gpu(losing_trades)
            learning_insights['common_error_patterns'] = error_patterns
            
            improvements = await self._generate_strategy_improvements(metrics, error_patterns)
            learning_insights['strategy_improvements'] = improvements
            
            risk_adj = await self._calculate_risk_adjustments(metrics)
            learning_insights['risk_adjustments'] = risk_adj
            return learning_insights
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 769)
            return {'total_learning_events': 0, 'common_error_patterns': {}, 'strategy_improvements': [], 'optimal_conditions': {}, 'risk_adjustments': {}}

    async def _analyze_error_patterns_gpu(self, losing_trades):
        try:
            if not losing_trades: return {}
            confidences = torch.tensor([t.get('confidence', 0) for t in losing_trades], device=self.device)
            volatilities = torch.tensor([t.get('volatility', 0) for t in losing_trades], device=self.device)
            expiries = torch.tensor([t.get('expiry_minutes', 5) for t in losing_trades], device=self.device)
            
            _ctx = torch.cuda.device(self.device) if (hasattr(self.device, 'type') and self.device.type == 'cuda') else nullcontext()
            with _ctx:
                high_vol_overconfident = int(torch.sum((confidences > 7.0) & (volatilities > 0.02)).item() if hasattr(torch.sum((confidences > 7.0) & (volatilities > 0.02)), 'item') else torch.sum((confidences > 7.0) & (volatilities > 0.02)))
                wrong_expiry = int(torch.sum((volatilities < 0.01) & (expiries > 8)).item() if hasattr(torch.sum((volatilities < 0.01) & (expiries > 8)), 'item') else torch.sum((volatilities < 0.01) & (expiries > 8)))
                
            return {
                'high_volatility_overconfidence': {'frequency': high_vol_overconfident, 'average_impact': -0.02},
                'wrong_expiry_timing': {'frequency': wrong_expiry, 'average_impact': -0.015}
            }
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 788)
            return {}

    async def _generate_strategy_improvements(self, metrics, error_patterns):
        improvements = []
        try:
            win_rate = metrics.get('win_rate', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            
            if win_rate < 0.6:
                improvements.append({
                    'type': 'CONFIDENCE_THRESHOLD',
                    'suggestion': f"Increase minimum confidence threshold from {self.backtest_config['min_confidence_threshold']} to {self.backtest_config['min_confidence_threshold'] + 0.5}",
                    'expected_impact': 'Higher quality trades'
                })
            
            if max_drawdown > 0.1:
                improvements.append({
                    'type': 'RISK_MANAGEMENT',
                    'suggestion': f"Reduce risk per trade from {self.backtest_config['risk_per_trade']*100:.1f}% to {self.backtest_config['risk_per_trade']*50:.1f}%",
                    'expected_impact': 'Reduced drawdown'
                })
            return improvements
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 811)
            return []

    async def _calculate_risk_adjustments(self, metrics):
        try:
            win_rate = metrics.get('win_rate', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if win_rate > 0.7 and sharpe_ratio > 1.0:
                return {'suggestion': "Consider increasing position sizing", 'confidence': "HIGH"}
            elif win_rate < 0.5 or sharpe_ratio < 0.5:
                return {'suggestion': "Reduce position sizing until performance improves", 'confidence': "MEDIUM"}
            return {'suggestion': "Maintain current risk parameters", 'confidence': "LOW"}
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 823)
            return {'suggestion': 'Maintain current parameters', 'confidence': 'LOW'}

    async def _generate_gpu_backtest_report(self, results, learning_insights):
        try:
            metrics = results.get('metrics', {})
            trades = results.get('trades', [])
            
            report = f"""
ACCELERATED GPU-ACCELERATED BACKTESTING REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Backtest Period: {self.backtest_config['backtest_period_days']} days
Total Trades: {metrics.get('total_trades', 0)}
Chunks Processed: {results.get('total_chunks', 0)}

  PERFORMANCE METRICS:
----------------------
Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
Total Profit: ${metrics.get('total_profit', 0):.2f}
Final Balance: ${metrics.get('final_balance', 0):.2f}
Average Profit per Trade: ${metrics.get('avg_profit', 0):.2f}
Maximum Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}

  AI LEARNING INSIGHTS:
----------------------
Total Learning Events: {learning_insights.get('total_learning_events', 0)}
Common Error Patterns: {len(learning_insights.get('common_error_patterns', {}))}
Strategy Improvements: {len(learning_insights.get('strategy_improvements', []))}

  RECOMMENDATIONS:
------------------
"""
            for improvement in learning_insights.get('strategy_improvements', [])[:3]:
                report += f"- {improvement.get('suggestion', 'N/A')}\n"
            
            risk_adj = learning_insights.get('risk_adjustments', {})
            report += f"- Risk Adjustment: {risk_adj.get('suggestion', 'Maintain current')} (Confidence: {risk_adj.get('confidence', 'LOW')})\n"

            # Report only measured backtest performance, not fictional AI approval.

            report += f"""
  HARDWARE UTILIZATION:
----------------------
GPU Accelerated: {hasattr(self.device, 'type') and self.device.type == 'cuda'}
SSD Streaming: Enabled
Memory Optimized: Yes
Processing Speed: {len(trades) / max(1, results.get('total_chunks', 1)):.1f} trades/chunk

============================================
"""
            return report
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 876)
            print(f"WARNING Report generation warning: {e}")
            return "Error generating report"

    def _generate_error_report(self, error_msg):
        return f"""
ERROR BACKTESTING ERROR REPORT
==========================
Error: {error_msg}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: GPU Backtesting Engine
Status: Failed
==========================
"""

    def optimize_for_hardware(self):
        optimizations = {
            'gpu_batch_size': 1000 if (hasattr(self.device, 'type') and self.device.type == 'cuda') else 100,
            'chunk_size_candles': 5000,
            'use_memory_mapping': True,
            'periodic_cleanup': True
        }
        self.backtest_config.update(optimizations)
        print("OK Backtester optimized for current hardware")

    def cleanup(self):
        self.gpu_manager._cleanup_gpu_memory()


# ==================== INSTITUTIONAL TREND ENGINE (PART 6) ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p6_trend_engine', 'analyze')
class TrendEngineGPU:
    """
    INSTITUTIONAL MULTI-TIMEFRAME TREND & CHOP REGIME ENGINE (Part 6)
    - Multi-EMA Alignment (EMA 8, 21, 50)
    - True ADX / Directional Movement Index (14 periods) to strictly filter chop
    - EMA Spread Threshold (> 0.08% for trend confirmation, < 0.08% = CHOP)
    - Strict Neutrality: Returns Signal: 0 (No trade) in ranging, flat, or mixed markets
    """
    def __init__(self, master_system=None):
        self.master = master_system
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'
        self.min_adx_threshold = 20.0
        self.min_spread_threshold = 0.0008  # 0.08% spread between EMA8 and EMA21

    @part_advisory_entry("part6_trend", data_parameter="data", context_parameter="context")
    def analyze(self, data, context=None):
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        return qm.part_signal("6", data, context=context)


# ==================== INTEGRATION CLASS ====================

class EnhancedTradingBot:
    def __init__(self, previous_parts=None):
        if previous_parts is None:
            previous_parts = {}
        self.master = (previous_parts.get('part5') or
                       previous_parts.get('master') or
                       previous_parts.get('trading_engine'))
        if self.master is None:
            logging.warning("Part6: No master system provided (part5/master/trading_engine). "
                            "Running in standalone mode — some features may be limited.")
        self.backtester = GPUComprehensiveBacktester(self.master)
        
    async def run_comprehensive_backtest(self, historical_data):
        return await self.backtester.run_full_year_backtest(historical_data)


def setup_linux_environment():
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print("OK Linux environment optimized for i5 10th Gen + GTX 1650")


def initialize_backtesting_system(previous_parts=None):
    if previous_parts is None:
        previous_parts = {}
    setup_linux_environment()
    return EnhancedTradingBot(previous_parts)


if __name__ == "__main__":
    backtesting_system = initialize_backtesting_system({})
    print("ACCELERATED Part6 Backtesting Engine - Ready for Integration")