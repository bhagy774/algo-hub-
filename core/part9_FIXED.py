try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ==================== GPU-ACCELERATED AI LEARNING & ADAPTIVE STRATEGY OPTIMIZATION ENGINE ====================
# DEEPSEEK AI-POWERED REWRITE - INSTITUTIONAL GRADE REINFORCEMENT LEARNING
# LINUX UBUNTU + GTX 1650 CUDA + i5 10th Gen OPTIMIZED

import os

# PyTorch with fallback for Windows/WSL compatibility
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 15)
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


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'



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
import json
import random
from typing import Dict, List, Tuple, Any, Optional, Union




# ==================== GPU-ACCELERATED ORDERFLOW & CVD DELTA ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p9_orderflow_engine', 'analyze')
class OrderflowEngineGPU:
    """
    JARVIS PART 9 - GPU-ACCELERATED ORDERFLOW & CUMULATIVE VOLUME DELTA (CVD) ENGINE
    GTX 1650 CUDA & CPU Optimized.

    Quantitative Orderflow Architecture:
    1. Intra-Candle Bid-Ask Volume Delta Proxy:
       Uses wick-body ratio and price progression to allocate buy vs sell aggressive flow.
    2. Cumulative Volume Delta (CVD-20):
       Aggregates directional orderflow momentum across the rolling 20-bar auction window.
    3. Short-Term Delta Acceleration (CVD-5):
       Measures real-time orderflow surge over the most recent 5 candles.
    4. Orderflow Absorption Divergence:
       - Bullish Absorption: Price falling but CVD surging positive (aggressive institutional accumulation).
       - Bearish Absorption: Price rising but CVD surging negative (aggressive institutional distribution).
    5. Trend Orderflow Imbalance:
       Requires normalized CVD ratio > +0.30 (Bullish) or < -0.30 (Bearish) with EMA20 alignment.
    6. Auction Equilibrium / Balanced Deadband:
       When buying and selling volumes are balanced (-0.30 <= CVD <= +0.30), strictly returns 0 (Neutral).
    """
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')

    @part_advisory_entry("part9_orderflow", data_parameter="data", context_parameter="context")
    def analyze(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        _res = qm.part_signal("9", data, context=context)
        try:  # PHASE3 diagnostic: log live metrics + gate reason for every NEUTRAL cycle
            import json as _j, logging as _lg, os as _os
            _lgr = _lg.getLogger("part9_diag")
            if not _lgr.handlers:
                _d = _os.path.join(_os.getcwd(), "logs", "files"); _os.makedirs(_d, exist_ok=True)
                _h = _lg.FileHandler(_os.path.join(_d, "part9_fixed.log"), encoding="utf-8")
                _h.setFormatter(_lg.Formatter("%(asctime)s %(message)s")); _lgr.addHandler(_h); _lgr.setLevel(_lg.INFO); _lgr.propagate = False
            if isinstance(_res, dict) and not _res.get("signal"):
                _lgr.info("[PART9_DIAG] NEUTRAL len=%s thought=%s telemetry=%s" % (len(data) if hasattr(data, "__len__") else "?", str(_res.get("thought"))[:200], _j.dumps(_res.get("telemetry", {}), default=str)[:600]))
        except Exception:
            pass
        return _res


# ==================== GPU AI MEMORY MANAGER ====================

class AIMemoryGPUManager:
    """GTX 1650 4GB VRAM Optimized Memory Manager for AI Learning"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_vram = 3.0 * 1024 * 1024 * 1024  # 3.0GB safety limit
        self.max_ram = 4.0 * 1024 * 1024 * 1024   # 4GB RAM limit
        
        # Cross-platform temp cache directory for long-term memory
        self.cache_dir = Path(tempfile.gettempdir()) / "ai_memory_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.75)  # 75% of 4GB
            print(f"OK AI Memory GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    def allocate_ai_tensor(self, data, name, persistent=True):
        """GPU tensor allocation optimized for AI learning data"""
        try:
            # BUG FIX #1: Check device type FIRST before calling cuda memory functions
            if self.device.type == 'cpu':
                return torch.tensor(data, dtype=torch.float32)

            if torch.cuda.memory_allocated() > self.max_vram:
                return torch.tensor(data, dtype=torch.float32)
            
            tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            
            if persistent and tensor.numel() < 10000:
                return tensor
            else:
                return self._create_memory_mapped_tensor(data, name)
                
        except RuntimeError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 205)
            print(f"WARNING AI GPU allocation failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def _create_memory_mapped_tensor(self, data, name):
        """Create memory-mapped tensor for large AI datasets"""
        try:
            # BUG FIX #2: Fixed filename per 'name' to avoid disk fill-up
            cache_file = self.cache_dir / f"{name}.dat"
            data_arr = np.array(data, dtype='float32')
            mmap = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_arr.shape)
            mmap[:] = data_arr
            return torch.from_numpy(np.array(mmap))  # copy to avoid mmap lifecycle issues
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 218)
            print(f"WARNING AI memory mapping failed: {e}")
            return torch.tensor(data, dtype=torch.float32)
    
    def cleanup_ai_memory(self):
        """Aggressive cleanup for AI operations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==================== STRATEGY WEIGHT OPTIMIZER ====================

class StrategyWeightOptimizer:
    """Dynamic strategy parameter adjustment based on AI learning"""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
        
        # Strategy weights storage
        self.pattern_weights = defaultdict(lambda: 1.0)  # Default weight 1.0
        self.market_regime_weights = defaultdict(lambda: 1.0)
        self.risk_parameters = {
            'max_position_size': 0.1,  # 10% of capital
            'stop_loss_pct': 0.02,     # 2% stop loss
            'risk_reward_ratio': 2.0,  # 1:2 risk reward
            'volatility_multiplier': 1.0
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.weight_adjustments = deque(maxlen=500)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        
        print("OK Strategy Weight Optimizer Initialized")
    
    async def update_weights_based_on_performance(self, trade_outcome: dict):
        """Update strategy weights based on trade performance"""
        try:
            pattern_type = trade_outcome.get('pattern_type', 'unknown')
            market_regime = trade_outcome.get('market_regime', 'unknown')
            is_win = bool(trade_outcome.get('is_win', False))
            
            raw_profit = trade_outcome.get('profit_pct', 0.0)
            profit_pct = float(raw_profit.item()) if hasattr(raw_profit, 'item') else float(raw_profit)
            
            raw_confidence = trade_outcome.get('confidence', 0.0)
            conf_val = float(raw_confidence.item()) if hasattr(raw_confidence, 'item') else float(raw_confidence)
            confidence = max(0.0, min(conf_val, 10.0)) / 10.0  # Normalize 0-10 → 0-1
            
            # Calculate weight adjustment
            if is_win:
                # Positive reinforcement for winning patterns
                weight_change = self.learning_rate * (1.0 + profit_pct * 10 + confidence * 0.5)
            else:
                # Negative reinforcement for losing patterns
                weight_change = -self.learning_rate * (1.0 + abs(profit_pct) * 10 + confidence * 0.5)
            
            # Apply momentum
            if len(self.weight_adjustments) > 0:
                last_adjustment = self.weight_adjustments[-1].get('weight_change', 0)
                weight_change = self.momentum * last_adjustment + (1 - self.momentum) * weight_change
            
            # Update pattern weight
            current_weight = self.pattern_weights[pattern_type]
            new_weight = max(0.1, min(3.0, current_weight + weight_change))  # Clamp between 0.1 and 3.0
            self.pattern_weights[pattern_type] = new_weight
            
            # Update market regime weight
            regime_weight = self.market_regime_weights[market_regime]
            new_regime_weight = max(0.1, min(3.0, regime_weight + weight_change * 0.5))
            self.market_regime_weights[market_regime] = new_regime_weight
            
            # Record adjustment
            adjustment_record = {
                'timestamp': time.time(),
                'pattern_type': pattern_type,
                'market_regime': market_regime,
                'weight_change': weight_change,
                'new_pattern_weight': new_weight,
                'new_regime_weight': new_regime_weight,
                'profit_pct': profit_pct,
                'is_win': is_win
            }
            self.weight_adjustments.append(adjustment_record)
            
            # Update risk parameters based on performance
            await self._update_risk_parameters(trade_outcome)
            
            return adjustment_record
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 312)
            print(f"ERROR Strategy weight update error: {e}")
            return None
    
    async def _update_risk_parameters(self, trade_outcome: dict):
        """Update risk parameters based on trading performance"""
        try:
            # BUG FIX #11: Append current trade FIRST so it's included in analysis
            self.performance_history.append(trade_outcome)
            
            recent_performance = list(self.performance_history)[-100:]
            if len(recent_performance) < 20:
                return
            
            win_rate = sum(1 for p in recent_performance if p.get('is_win', False)) / len(recent_performance)
            avg_profit = np.mean([p.get('profit_pct', 0) for p in recent_performance])
            max_drawdown = min([p.get('profit_pct', 0) for p in recent_performance])
            
            # Adjust position size based on performance
            if win_rate > 0.6 and avg_profit > 0.01:
                # Increase position size for good performance
                self.risk_parameters['max_position_size'] = min(0.2, self.risk_parameters['max_position_size'] * 1.05)
            elif win_rate < 0.4 or avg_profit < -0.01:
                # Decrease position size for poor performance
                self.risk_parameters['max_position_size'] = max(0.05, self.risk_parameters['max_position_size'] * 0.95)
            
            # Adjust stop loss based on volatility
            if 'volatility' in trade_outcome.get('market_analysis', {}):
                volatility = trade_outcome['market_analysis']['volatility']
                self.risk_parameters['stop_loss_pct'] = max(0.01, min(0.05, volatility * 0.5))
                self.risk_parameters['volatility_multiplier'] = max(0.5, min(2.0, 1.0 / (volatility + 0.1)))
            
            # Adjust risk-reward ratio
            if avg_profit > 0.02:
                self.risk_parameters['risk_reward_ratio'] = min(3.0, self.risk_parameters['risk_reward_ratio'] * 1.02)
            elif avg_profit < -0.01:
                self.risk_parameters['risk_reward_ratio'] = max(1.5, self.risk_parameters['risk_reward_ratio'] * 0.98)
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 350)
            print(f"WARNING Risk parameter update warning: {e}")
    
    def get_strategy_weights(self, pattern_type: str, market_regime: str):
        """Get combined strategy weight for pattern and market regime"""
        pattern_weight = self.pattern_weights[pattern_type]
        regime_weight = self.market_regime_weights[market_regime]
        
        # Combined weight (geometric mean)
        combined_weight = (pattern_weight * regime_weight) ** 0.5
        
        return {
            'pattern_weight': pattern_weight,
            'regime_weight': regime_weight,
            'combined_weight': combined_weight,
            'risk_parameters': self.risk_parameters.copy()
        }
    
    def get_performance_analysis(self):
        """Get comprehensive performance analysis"""
        try:
            if len(self.performance_history) == 0:
                return {'error': 'No performance data available'}
            
            recent_trades = list(self.performance_history)[-100:]
            win_rate = sum(1 for t in recent_trades if t.get('is_win', False)) / len(recent_trades)
            avg_profit = np.mean([t.get('profit_pct', 0) for t in recent_trades])
            total_profit = sum(t.get('profit_pct', 0) for t in recent_trades)
            
            # Pattern performance
            pattern_performance = defaultdict(list)
            for trade in recent_trades:
                pattern = trade.get('pattern_type', 'unknown')
                pattern_performance[pattern].append(trade.get('profit_pct', 0))
            
            pattern_analysis = {}
            for pattern, profits in pattern_performance.items():
                if len(profits) >= 5:  # Minimum samples
                    pattern_analysis[pattern] = {
                        'count': len(profits),
                        'win_rate': sum(1 for p in profits if p > 0) / len(profits),
                        'avg_profit': np.mean(profits),
                        'current_weight': self.pattern_weights[pattern]
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'total_profit': total_profit,
                    'trade_count': len(recent_trades)
                },
                'pattern_analysis': pattern_analysis,
                'risk_parameters': self.risk_parameters,
                'active_patterns': len(self.pattern_weights),
                'active_regimes': len(self.market_regime_weights)
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 409)
            print(f"ERROR Performance analysis error: {e}")
            return {'error': str(e)}

# ==================== PATTERN PERFORMANCE TRACKER ====================

class PatternPerformanceTracker:
    """Comprehensive pattern success monitoring and analytics"""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
        
        # Pattern performance storage
        self.pattern_stats = defaultdict(lambda: {
            'total_occurrences': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'recent_performance': deque(maxlen=100),
            'market_condition_performance': defaultdict(lambda: deque(maxlen=50)),
            'confidence_scores': deque(maxlen=200)
        })
        
        # Time-based performance tracking
        self.hourly_performance = defaultdict(lambda: deque(maxlen=24))
        self.daily_performance = deque(maxlen=30)
        
        # Market regime performance
        self.regime_performance = defaultdict(lambda: deque(maxlen=100))
        
        print("OK Pattern Performance Tracker Initialized")
    
    async def record_pattern_performance(self, trade_data: dict):
        """Record pattern performance for learning"""
        try:
            pattern_type = trade_data.get('pattern_type', 'unknown')
            market_regime = trade_data.get('market_regime', 'unknown')
            is_win = trade_data.get('is_win', False)
            profit_pct = trade_data.get('profit_pct', 0.0)
            confidence = trade_data.get('confidence', 0.0)
            timestamp = trade_data.get('timestamp', time.time())
            
            # Update pattern statistics
            stats = self.pattern_stats[pattern_type]
            stats['total_occurrences'] += 1
            stats['total_profit'] += profit_pct
            
            if is_win:
                stats['successful_trades'] += 1
            
            # Record recent performance
            performance_entry = {
                'timestamp': timestamp,
                'profit_pct': profit_pct,
                'is_win': is_win,
                'confidence': confidence,
                'market_regime': market_regime
            }
            stats['recent_performance'].append(performance_entry)
            
            # Record market condition performance
            stats['market_condition_performance'][market_regime].append(performance_entry)
            
            # Record confidence scores
            stats['confidence_scores'].append({
                'timestamp': timestamp,
                'confidence': confidence,
                'was_correct': is_win
            })
            
            # Update time-based performance
            hour = datetime.fromtimestamp(timestamp).hour
            self.hourly_performance[hour].append(performance_entry)
            
            # Update regime performance
            self.regime_performance[market_regime].append(performance_entry)
            
            # BUG FIX #4: Safe .get() to avoid KeyError if 'timestamp' missing
            if len(self.daily_performance) == 0 or \
               datetime.fromtimestamp(timestamp).date() != datetime.fromtimestamp(
                   self.daily_performance[-1].get('timestamp', 0)).date():
                self.daily_performance.append({
                    'timestamp': timestamp,
                    'total_trades': 1,
                    'successful_trades': 1 if is_win else 0,
                    'total_profit': profit_pct,
                    'patterns_used': [pattern_type]
                })
            else:
                last_day = self.daily_performance[-1]
                last_day['total_trades'] += 1
                last_day['total_profit'] += profit_pct
                if is_win:
                    last_day['successful_trades'] += 1
                if pattern_type not in last_day['patterns_used']:
                    last_day['patterns_used'].append(pattern_type)
            
            return True
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 508)
            print(f"ERROR Pattern performance recording error: {e}")
            return False
    
    def get_pattern_effectiveness(self, pattern_type: str, market_regime: str = None):
        """Get pattern effectiveness analysis"""
        try:
            if pattern_type not in self.pattern_stats:
                return {'error': f'Pattern {pattern_type} not found'}
            
            stats = self.pattern_stats[pattern_type]
            total_trades = stats['total_occurrences']
            
            if total_trades == 0:
                return {'error': 'No trades recorded for this pattern'}
            
            # Overall effectiveness
            success_rate = stats['successful_trades'] / total_trades
            avg_profit = stats['total_profit'] / total_trades
            
            # Recent performance (last 50 trades)
            recent_trades = list(stats['recent_performance'])[-50:]
            recent_success_rate = sum(1 for t in recent_trades if t['is_win']) / len(recent_trades) if recent_trades else 0
            recent_avg_profit = np.mean([t['profit_pct'] for t in recent_trades]) if recent_trades else 0
            
            # Market regime specific performance
            regime_specific = {}
            if market_regime and market_regime in stats['market_condition_performance']:
                regime_trades = list(stats['market_condition_performance'][market_regime])
                if regime_trades:
                    regime_specific = {
                        'success_rate': sum(1 for t in regime_trades if t['is_win']) / len(regime_trades),
                        'avg_profit': np.mean([t['profit_pct'] for t in regime_trades]),
                        'trade_count': len(regime_trades)
                    }
            
            # Confidence analysis
            confidence_scores = list(stats['confidence_scores'])
            if confidence_scores:
                avg_confidence = np.mean([c['confidence'] for c in confidence_scores])
                confidence_accuracy = sum(1 for c in confidence_scores if c['was_correct']) / len(confidence_scores)
            else:
                avg_confidence = 0
                confidence_accuracy = 0
            
            return {
                'pattern_type': pattern_type,
                'overall_metrics': {
                    'total_trades': total_trades,
                    'success_rate': success_rate,
                    'avg_profit': avg_profit,
                    'total_profit': stats['total_profit']
                },
                'recent_metrics': {
                    'success_rate': recent_success_rate,
                    'avg_profit': recent_avg_profit,
                    'sample_size': len(recent_trades)
                },
                'regime_specific': regime_specific,
                'confidence_analysis': {
                    'average_confidence': avg_confidence,
                    'confidence_accuracy': confidence_accuracy,
                    'sample_size': len(confidence_scores)
                },
                'market_regime_performance': {
                    regime: {
                        'success_rate': sum(1 for t in trades if t['is_win']) / len(trades),
                        'avg_profit': np.mean([t['profit_pct'] for t in trades]),
                        'trade_count': len(trades)
                    }
                    for regime, trades in stats['market_condition_performance'].items()
                    if len(trades) >= 3  # Minimum samples
                }
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 583)
            print(f"ERROR Pattern effectiveness analysis error: {e}")
            return {'error': str(e)}
    
    def get_market_regime_analysis(self):
        """Get comprehensive market regime performance analysis"""
        try:
            regime_analysis = {}
            
            for regime, trades in self.regime_performance.items():
                if len(trades) >= 10:  # Minimum samples
                    regime_analysis[regime] = {
                        'total_trades': len(trades),
                        'success_rate': sum(1 for t in trades if t['is_win']) / len(trades),
                        'avg_profit': np.mean([t['profit_pct'] for t in trades]),
                        'best_patterns': self._get_best_patterns_for_regime(regime)
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'regime_analysis': regime_analysis,
                'hourly_performance': {
                    hour: {
                        'total_trades': len(trades),
                        'success_rate': sum(1 for t in trades if t['is_win']) / len(trades) if trades else 0,
                        'avg_profit': np.mean([t['profit_pct'] for t in trades]) if trades else 0
                    }
                    for hour, trades in self.hourly_performance.items()
                    if len(trades) >= 5  # Minimum samples per hour
                }
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 615)
            print(f"ERROR Market regime analysis error: {e}")
            return {'error': str(e)}
    
    def _get_best_patterns_for_regime(self, regime: str):
        """Get best performing patterns for specific market regime"""
        try:
            pattern_scores = []
            
            for pattern_type, stats in self.pattern_stats.items():
                regime_trades = list(stats['market_condition_performance'][regime])
                if len(regime_trades) >= 5:  # Minimum samples
                    success_rate = sum(1 for t in regime_trades if t['is_win']) / len(regime_trades)
                    avg_profit = np.mean([t['profit_pct'] for t in regime_trades])
                    # Score combines success rate and profitability
                    score = success_rate * (1 + avg_profit * 10)
                    pattern_scores.append((pattern_type, score, len(regime_trades)))
            
            # Sort by score descending
            pattern_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [
                {
                    'pattern_type': pattern,
                    'score': score,
                    'trade_count': count
                }
                for pattern, score, count in pattern_scores[:10]  # Top 10 patterns
            ]
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 645)
            print(f"WARNING Best patterns analysis warning: {e}")
            return []

# ==================== REAL-TIME LEARNING SCHEDULER ====================

class RealTimeLearningScheduler:
    """Learning session management with market hours optimization"""
    
    def __init__(self):
        self.is_market_hours = False
        self.learning_active = False
        self.last_training_time = 0
        self.training_interval = 300  # 5 minutes between training sessions
        
        # Market hours configuration (Forex - 24/5 but optimized for active sessions)
        self.active_sessions = {
            'asian': (20, 4),    # 8 PM - 4 AM UTC
            'london': (6, 12),   # 6 AM - 12 PM UTC  
            'new_york': (12, 18) # 12 PM - 6 PM UTC
        }
        
        # Performance tracking
        self.training_sessions = deque(maxlen=100)
        self.learning_metrics = deque(maxlen=500)
        
        print("OK Real-Time Learning Scheduler Initialized")
    
    def _is_in_session(self, current_hour, start_hour, end_hour):
        """BUG FIX #3 #12: Correctly handle sessions that cross midnight (e.g. 20:00-04:00)"""
        if start_hour <= end_hour:
            # Normal session: e.g. 06:00-12:00
            return start_hour <= current_hour < end_hour
        else:
            # Midnight-crossing session: e.g. 20:00-04:00
            return current_hour >= start_hour or current_hour < end_hour

    def is_learning_active(self):
        """Check if learning should be active based on market conditions"""
        current_hour = datetime.utcnow().hour
        current_weekday = datetime.utcnow().weekday()
        
        # Only learn during weekdays
        if current_weekday >= 5:
            return False
        
        # BUG FIX #3: Use _is_in_session for correct midnight-crossing handling
        for session, (start_hour, end_hour) in self.active_sessions.items():
            if self._is_in_session(current_hour, start_hour, end_hour):
                return True
        
        return False
    
    async def schedule_training(self, rl_learner, strategy_optimizer, pattern_tracker):
        """Schedule and execute training sessions"""
        try:
            current_time = time.time()
            
            # Check if it's time for training
            if current_time - self.last_training_time < self.training_interval:
                return False
            
            # Check if learning should be active
            if not self.is_learning_active():
                return False
            
            print("  Starting scheduled AI training session...")
            
            # Execute training
            training_result = await self._execute_training_session(
                rl_learner, strategy_optimizer, pattern_tracker
            )
            
            # Record training session
            self.training_sessions.append({
                'timestamp': current_time,
                'result': training_result,
                'experiences_processed': len(rl_learner.memory) if (hasattr(rl_learner, 'memory') and hasattr(rl_learner.memory, '__len__')) else 0
            })
            
            self.last_training_time = current_time
            self.learning_active = True
            
            print(f"OK Training session completed: {training_result}")
            return True
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 731)
            print(f"ERROR Training scheduling error: {e}")
            self.learning_active = False
            return False
    
    async def _execute_training_session(self, rl_learner, strategy_optimizer, pattern_tracker):
        """Execute comprehensive training session"""
        try:
            training_metrics = {}
            
            # 1. Reinforcement learning training
            # BUG FIX #8: hasattr check before calling private method
            if hasattr(rl_learner, 'memory') and hasattr(rl_learner.memory, '__len__') and len(rl_learner.memory) >= getattr(rl_learner, 'batch_size', 32):
                if hasattr(rl_learner, '_train_network'):
                    await rl_learner._train_network()
                    training_metrics['rl_training'] = 'completed'
                else:
                    training_metrics['rl_training'] = 'skipped_no_method'
            
            # 2. Strategy weight optimization
            if hasattr(strategy_optimizer, 'performance_history'):
                recent_trades = list(strategy_optimizer.performance_history)[-50:]
                for trade in recent_trades:
                    await strategy_optimizer.update_weights_based_on_performance(trade)
                training_metrics['strategy_optimization'] = 'completed'
            
            # 3. Pattern performance analysis
            pattern_analysis = pattern_tracker.get_market_regime_analysis()
            training_metrics['pattern_analysis'] = 'completed'
            
            # Record learning metrics
            self.learning_metrics.append({
                'timestamp': time.time(),
                'metrics': training_metrics,
                'rl_memory_size': len(rl_learner.memory) if (hasattr(rl_learner, 'memory') and hasattr(rl_learner.memory, '__len__')) else 0,
                'pattern_count': len(pattern_tracker.pattern_stats) if hasattr(pattern_tracker, 'pattern_stats') else 0
            })
            
            return training_metrics
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 771)
            print(f"ERROR Training execution error: {e}")
            return {'error': str(e)}
    
    def get_learning_schedule_analysis(self):
        """Get learning schedule and performance analysis"""
        try:
            current_status = {
                'is_learning_active': self.is_learning_active(),
                'learning_scheduler_active': self.learning_active,
                'current_market_hours': self.is_market_hours,
                'time_until_next_training': max(0, self.training_interval - (time.time() - self.last_training_time)),
                'utc_time': datetime.utcnow().strftime("%H:%M"),
                'utc_hour': datetime.utcnow().hour
            }
            
            # Training session statistics
            if self.training_sessions:
                recent_sessions = list(self.training_sessions)[-10:]
                current_status['recent_sessions'] = len(recent_sessions)
                current_status['avg_training_interval'] = np.mean([
                    recent_sessions[i+1]['timestamp'] - recent_sessions[i]['timestamp'] 
                    for i in range(len(recent_sessions)-1)
                ]) if len(recent_sessions) > 1 else 0
            else:
                current_status['recent_sessions'] = 0
                current_status['avg_training_interval'] = 0
            
            # Active sessions analysis
            active_sessions = []
            current_hour = datetime.utcnow().hour
            for session, (start, end) in self.active_sessions.items():
                # BUG FIX #12: Use _is_in_session for correct midnight-crossing
                is_active = self._is_in_session(current_hour, start, end)
                active_sessions.append({
                    'session': session,
                    'hours': f"{start:02d}:00-{end:02d}:00 UTC",
                    'is_active': is_active
                })
            
            current_status['trading_sessions'] = active_sessions
            
            return current_status
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 815)
            print(f"ERROR Learning schedule analysis error: {e}")
            return {'error': str(e)}

# ==================== MAIN GPU AI LEARNING ENGINE ====================

# BUG FIX #5: GPUReinforcementLearner is used but NOT defined in this file
# Try importing from wherever it lives, otherwise provide a safe placeholder
try:
    from gpu_rl_learner import GPUReinforcementLearner  # adjust path as needed
except Exception:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 825)
    class GPUReinforcementLearner:  # safe fallback when gpu_rl_learner is absent
        """Minimal no-op RL learner: keeps Part 7/9 loading; learning degrades
        gracefully to a neutral policy instead of crashing the engine."""
        def __init__(self, gpu_manager=None, *a, **kw):
            self.gpu_manager = gpu_manager
        def __getattr__(self, name):
            def _noop(*a, **kw):
                return None
            return _noop

class GPUAIAdaptiveLearningEngine:
    """
    INSTITUTIONAL-GRADE AI LEARNING ENGINE
    GPU-Accelerated Reinforcement Learning + Adaptive Strategy Optimization
    """
    
    def __init__(self, trading_system=None):
        self.trading_system = trading_system
        self.gpu_manager = AIMemoryGPUManager()
        self.device = self.gpu_manager.device
        
        # Core components
        self.rl_learner = GPUReinforcementLearner(gpu_manager=self.gpu_manager)
        self.strategy_optimizer = StrategyWeightOptimizer(self.gpu_manager)
        self.pattern_tracker = PatternPerformanceTracker(self.gpu_manager)
        self.learning_scheduler = RealTimeLearningScheduler()
        
        # System state
        self.is_running = False
        self.learning_loop_task = None
        self.performance_metrics = deque(maxlen=1000)
        self._start_time = time.time()       # BUG FIX #7: was never set → uptime always 0
        self._last_metrics_time = time.time()  # BUG FIX #6: track metrics time properly
        
        # Integration with trading system
        self.trade_callbacks = []
        
        # LegacyLLM Strategy Optimization Cooldown setup
        self.last_legacy_llm_time = 0
        self.legacy_llm_cooldown = 300  # 5 minutes
        self.last_legacy_llm_recommendation = "MAINTAIN_WEIGHTS"
        self.last_legacy_llm_insight = "Strategy weights optimal for current market conditions."
        
        print("ACCELERATED GPU AI Learning Engine Initialized - Linux Optimized")
    
    async def start_learning_engine(self):
        """Start the AI learning engine"""
        print("  Starting GPU AI Adaptive Learning Engine...")
        self.is_running = True
        
        try:
            # Start continuous learning loop
            self.learning_loop_task = asyncio.create_task(self._continuous_learning_loop())
            
            print("OK AI Learning Engine Started Successfully")
            return True
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 883)
            print(f"ERROR Failed to start AI learning engine: {e}")
            self.is_running = False
            return False
    
    async def _continuous_learning_loop(self):
        """Main continuous learning loop"""
        while self.is_running:
            try:
                # Schedule and execute training
                await self.learning_scheduler.schedule_training(
                    self.rl_learner, 
                    self.strategy_optimizer, 
                    self.pattern_tracker
                )
                
                # Process any pending trade outcomes
                await asyncio.sleep(1)  # 1-second cycle
                
                # BUG FIX #6: Use time-delta instead of modulo — modulo misses cycles often
                if time.time() - self._last_metrics_time >= 30:
                    await self._record_performance_metrics()
                    self._last_metrics_time = time.time()
                
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 907)
                if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                    self.trading_system.bus.report_error('Part9_Adaptive', e, context='_continuous_learning_loop')
                else:
                    print(f"ERROR Learning loop error: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
    
    async def process_trade_outcome(self, trade_result: dict):
        """
        Process trade outcome through all AI learning components
        """
        try:
            # 1. Record in pattern performance tracker
            await self.pattern_tracker.record_pattern_performance(trade_result)
            
            # 2. Update strategy weights
            weight_update = await self.strategy_optimizer.update_weights_based_on_performance(trade_result)
            
            # 3. Process in reinforcement learner
            await self.rl_learner.process_trade_outcome(trade_result)
            
            # 4. Record in performance history
            self.strategy_optimizer.performance_history.append(trade_result)
            
            # 5. Call any registered callbacks
            for callback in self.trade_callbacks:
                try:
                    await callback(trade_result)
                except Exception as e:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 935)
                    print(f"WARNING Trade callback error: {e}")
            
            return {
                'pattern_tracking': True,
                'weight_updated': weight_update is not None,
                'rl_learning': True
            }
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 944)
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                self.trading_system.bus.report_error('Part9_Adaptive', e, context='process_trade_outcome')
            else:
                print(f"ERROR Trade outcome processing error: {e}")
            return {'error': str(e)}
    
    async def get_ai_trading_recommendation(self, market_state: dict, pattern_signals: dict):
        """
        Get AI-powered trading recommendation with adaptive strategy weights
        """
        try:
            # 1. Get RL recommendation
            rl_recommendation = await self.rl_learner.get_trading_recommendation(market_state)
            
            # 2. Get strategy weights for patterns
            pattern_type = pattern_signals.get('primary_pattern', 'unknown')
            market_regime = market_state.get('market_regime', 'unknown')
            
            strategy_weights = self.strategy_optimizer.get_strategy_weights(pattern_type, market_regime)
            
            # 3. Get pattern effectiveness
            pattern_effectiveness = self.pattern_tracker.get_pattern_effectiveness(pattern_type, market_regime)
            
            # 4. Combine recommendations
            base_confidence = rl_recommendation.get('confidence', 0.0)
            weight_multiplier = strategy_weights.get('combined_weight', 1.0)
            
            # Adjust confidence based on pattern effectiveness and strategy weights
            adjusted_confidence = base_confidence * weight_multiplier
            
            # Apply risk parameters
            risk_params = strategy_weights.get('risk_parameters', {})
            
            recommendation = {
                'action': rl_recommendation.get('action', 'HOLD'),
                'confidence': min(10.0, adjusted_confidence),  # Cap at 10
                'base_confidence': base_confidence,
                'weight_multiplier': weight_multiplier,
                'pattern_type': pattern_type,
                'market_regime': market_regime,
                'strategy_weights': strategy_weights,
                'pattern_effectiveness': pattern_effectiveness,
                'risk_parameters': risk_params,
                'timestamp': datetime.now().isoformat(),
                'rl_epsilon': rl_recommendation.get('epsilon', 0.0),
                'q_values': rl_recommendation.get('q_values', [])
            }
            
            return recommendation
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 995)
            print(f"ERROR AI trading recommendation error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def _record_performance_metrics(self):
        """Record comprehensive performance metrics"""
        try:
            metrics = {
                'timestamp': time.time(),
                'rl_analysis': self.rl_learner.get_learning_analysis(),
                'strategy_analysis': self.strategy_optimizer.get_performance_analysis(),
                'pattern_analysis': self.pattern_tracker.get_market_regime_analysis(),
                'schedule_analysis': self.learning_scheduler.get_learning_schedule_analysis(),
                'memory_usage': self.rl_learner.get_memory_usage(),
                'gpu_memory': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            }
            
            self.performance_metrics.append(metrics)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1018)
            print(f"WARNING Performance metrics recording warning: {e}")
    
    def _generate_legacy_llm_learning_prompt(self, learning_summary: Dict) -> str:
        """Generate LegacyLLM prompt for Chief Strategy Officer Strategy Weight Optimization"""
        summary_str = json.dumps(learning_summary, default=str)

        prompt = f"""You are a Chief Strategy Officer AI for an elite quantitative trading firm. Your job is to review the recent performance of our algorithmic trading strategies and recommend real-time strategy weight adjustments.

Performance Metrics & Learning Summary: {summary_str}

Task: Analyze strategy win rates, profit metrics, and market regime adaptations. Determine if we should adjust strategy weights.

Respond with EXACTLY ONE of the following classification tags at the start of your response:
- [BOOST_TREND_STRATEGY] : Trend-following strategies are outperforming; increase trend weights.
- [PENALIZE_BREAKOUTS] : Breakout strategies are suffering false breakouts; decrease breakout weights.
- [MAINTAIN_WEIGHTS] : Performance is balanced across strategies; maintain current weights.

Follow the tag with a 1-2 sentence Chief Strategy Officer executive recommendation.
"""
        return prompt

    def validate_strategy_weights_with_legacy_llm(self, learning_summary: Dict) -> Tuple[str, str]:
        """No model-generated adaptive weight recommendation or cached vote."""
        return "UNAVAILABLE", "Model strategy recommendation unavailable"

    def get_comprehensive_analysis(self):
        """Get comprehensive AI learning analysis"""
        try:
            current_metrics = self.performance_metrics[-1] if self.performance_metrics else {}
            summary = self._get_learning_summary()
            rec, insight = self.validate_strategy_weights_with_legacy_llm(summary)

            analysis = {
                'system_status': {
                    'is_running': self.is_running,
                    'learning_active': self.learning_scheduler.learning_active,
                    'device': str(self.device),
                    'uptime': self._get_uptime()
                },
                'performance_metrics': current_metrics,
                'summary': summary,
                'legacy_llm_strategy_recommendation': rec,
                'legacy_llm_insight': insight
            }
            
            # Publish to CognitiveBus for Watcher AI monitoring
            if hasattr(self, 'trading_system') and hasattr(self.trading_system, 'bus') and self.trading_system.bus:
                running = analysis.get('system_status', {}).get('is_running', False)
                rec = str(analysis.get('legacy_llm_strategy_recommendation', 'N/A'))[:60]
                msg = f"Adaptive Learning: Status={'RUNNING' if running else 'IDLE'}. Strategy Recommendation: {rec}"
                self.trading_system.bus.publish('THOUGHTS', 'Part9_Adaptive', msg)
            return analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1072)
            print(f"ERROR Comprehensive analysis error: {e}")
            return {'error': str(e)}
    
    def _get_uptime(self):
        """Get system uptime"""
        if hasattr(self, '_start_time'):
            return time.time() - self._start_time
        return 0
    
    def _get_learning_summary(self):
        """Get learning summary"""
        try:
            total_experiences = len(self.rl_learner.memory) if (hasattr(self.rl_learner, 'memory') and hasattr(self.rl_learner.memory, '__len__')) else 0
            total_patterns = len(self.pattern_tracker.pattern_stats) if hasattr(self.pattern_tracker, 'pattern_stats') else 0
            learning_steps = getattr(self.rl_learner, 'learning_steps', 0)
            
            return {
                'total_learning_experiences': total_experiences,
                'patterns_tracked': total_patterns,
                'learning_steps_completed': learning_steps,
                'strategy_weights_optimized': len(self.strategy_optimizer.pattern_weights),
                'training_sessions_completed': len(self.learning_scheduler.training_sessions)
            }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1096)
            print(f"WARNING Learning summary error: {e}")
            return {}
    
    def register_trade_callback(self, callback):
        """Register callback for trade outcomes"""
        self.trade_callbacks.append(callback)
    
    async def shutdown(self):
        """Safe shutdown of AI learning engine"""
        print("  Shutting down GPU AI Adaptive Learning Engine...")
        self.is_running = False
        
        if self.learning_loop_task:
            self.learning_loop_task.cancel()
            try:
                await self.learning_loop_task
            except asyncio.CancelledError:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1113)
                pass
        
        # Shutdown components
        # BUG FIX #9: shutdown may not be async — safe check
        if hasattr(self.rl_learner, 'shutdown'):
            try:
                result = self.rl_learner.shutdown()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1123)
                print(f"WARNING RL learner shutdown warning: {e}")
        
        # Cleanup GPU memory
        self.gpu_manager.cleanup_ai_memory()
        
        print("OK AI Adaptive Learning Engine shutdown complete")

# ==================== INTEGRATION WITH EXISTING SYSTEM ====================

class EnhancedAILearningSystem:
    """Enhanced AI learning system with GPU acceleration"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.ai_engine = GPUAIAdaptiveLearningEngine(trading_system)
        self.is_active = False
    
    async def start_ai_learning(self):
        """Start AI learning system"""
        print("  Starting Enhanced GPU AI Learning System...")
        self.is_active = True
        
        try:
            # Register with trading system callbacks
            if hasattr(self.trading_system, 'register_trade_callback'):
                self.trading_system.register_trade_callback(self.process_trade_outcome)
            
            # Start AI engine
            success = await self.ai_engine.start_learning_engine()
            
            if success:
                print("OK Enhanced AI Learning System Started")
                return True
            else:
                print("ERROR Failed to start Enhanced AI Learning System")
                self.is_active = False
                return False
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1162)
            print(f"ERROR AI learning system start error: {e}")
            self.is_active = False
            return False
    
    async def process_trade_outcome(self, trade_result: dict):
        """Process trade outcome through AI learning"""
        try:
            return await self.ai_engine.process_trade_outcome(trade_result)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1171)
            print(f"ERROR Trade outcome processing error: {e}")
            return {'error': str(e)}
    
    async def get_ai_recommendation(self, market_state: dict, pattern_signals: dict):
        """Get AI trading recommendation"""
        try:
            return await self.ai_engine.get_ai_trading_recommendation(market_state, pattern_signals)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1179)
            print(f"ERROR AI recommendation error: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
    def get_learning_insights(self):
        """Get AI learning insights"""
        try:
            return self.ai_engine.get_comprehensive_analysis()
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1187)
            print(f"ERROR Learning insights error: {e}")
            return {'error': str(e)}
    
    async def stop_ai_learning(self):
        """Stop AI learning system"""
        print("  Stopping Enhanced AI Learning System...")
        self.is_active = False
        await self.ai_engine.shutdown()

# ==================== LINUX OPTIMIZATION ====================

def setup_linux_ai_environment():
    """Setup Linux-optimized environment for AI learning"""
    # Set thread affinity for i5 4-core CPU
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Enable GPU memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Set async performance parameters
    os.environ['PYTHONASYNCIODEBUG'] = '0'
    
    print("OK Linux environment optimized for AI learning")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    setup_linux_ai_environment()
    
    # Initialize AI learning system
    ai_system = EnhancedAILearningSystem(trading_system=None)
    
    print("ACCELERATED Part9 AI Learning Engine - Ready for Production Trading")
    print("FEATURES:")
    print("  • GPU-Accelerated Reinforcement Learning")
    print("  • Adaptive Strategy Weight Optimization") 
    print("  • Pattern Performance Tracking & Analytics")
    print("  • Real-Time Learning Scheduler")
    print("  • Market Regime Adaptation")
    print("  • Risk Parameter Calibration")
    print("  • Linux + GTX 1650 Optimized")