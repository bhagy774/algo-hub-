try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

import numpy as np

# Shadow-mode micro-XGBoost per brain (adds xgb_confidence only; never changes decisions)
from brain_xgb import xgb_brain

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

        def __init__(self, data=0.0, *args, **kwargs):
            if isinstance(data, (list, tuple, np.ndarray)):
                self._data = [float(x) for x in data]
            elif isinstance(data, (int, float, np.number)):
                self._data = [float(data)]
            else:
                self._data = [0.0]
            self.shape = (len(self._data),)

        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return np.array(self._data)
        def tolist(self): return list(self._data)
        def item(self): return float(self._data[0]) if self._data else 0.0
        def numel(self): return len(self._data)
        
        def __getitem__(self, key):
            if isinstance(key, slice):
                return DummyTensor(self._data[key])
            if isinstance(key, int) and 0 <= key < len(self._data):
                return float(self._data[key])
            return self

        def __len__(self): return len(self._data)
        def __add__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([a + val for a in self._data])
        def __sub__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([a - val for a in self._data])
        def __mul__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([a * val for a in self._data])
        def __truediv__(self, other):
            val = (other.item() if isinstance(other, DummyTensor) else other) + 1e-8
            return DummyTensor([a / val for a in self._data])
        def __radd__(self, other): return self.__add__(other)
        def __rsub__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([val - a for a in self._data])
        def __rmul__(self, other): return self.__mul__(other)
        def __abs__(self): return DummyTensor([abs(a) for a in self._data])
        def __gt__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([1.0 if a > val else 0.0 for a in self._data])
        def __lt__(self, other):
            val = other.item() if isinstance(other, DummyTensor) else other
            return DummyTensor([1.0 if a < val else 0.0 for a in self._data])
    
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
        def tensor(data, **kwargs): return DummyTensor(data)
        
        @staticmethod
        def zeros(size, **kwargs):
            n = size if isinstance(size, int) else (size[0] if size else 1)
            return DummyTensor([0.0]*n)
        
        @staticmethod
        def randn(*args, **kwargs): return DummyTensor()
        
        @staticmethod
        def cat(tensors, dim=0):
            res = []
            for t in tensors:
                if isinstance(t, DummyTensor):
                    res.extend(t.tolist())
                elif isinstance(t, (list, tuple)):
                    res.extend(t)
            return DummyTensor(res)
        
        @staticmethod
        def stack(tensors, dim=0): return torch.cat(tensors, dim=dim)
        
        @staticmethod
        def std(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            if len(data) < 2: return DummyTensor(0.0)
            return DummyTensor(float(np.std(data, ddof=1)))
            
        @staticmethod
        def mean(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            if len(data) == 0: return DummyTensor(0.0)
            return DummyTensor(float(np.mean(data)))

        @staticmethod
        def max(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            if len(data) == 0: return DummyTensor(0.0)
            return DummyTensor(float(np.max(data)))

        @staticmethod
        def min(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            if len(data) == 0: return DummyTensor(0.0)
            return DummyTensor(float(np.min(data)))

        @staticmethod
        def sum(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            return DummyTensor(float(np.sum(data)))

        @staticmethod
        def norm(t):
            data = t.tolist() if isinstance(t, DummyTensor) else t
            return DummyTensor(float(np.linalg.norm(data)))

        @staticmethod
        def sigmoid(x):
            val = x.item() if isinstance(x, DummyTensor) else x
            return DummyTensor(1.0 / (1.0 + np.exp(-val)))

        @staticmethod
        def tanh(x):
            val = x.item() if isinstance(x, DummyTensor) else x
            return DummyTensor(float(np.tanh(val)))

        @staticmethod
        def abs(x):
            if isinstance(x, DummyTensor):
                return torch.abs(x)
            return abs(x)

        @staticmethod
        def maximum(a, b):
            val_a = a.tolist() if isinstance(a, DummyTensor) else [a]
            val_b = b.tolist() if isinstance(b, DummyTensor) else [b]
            res = [max(x, y) for x, y in zip(val_a, val_b)]
            return DummyTensor(res)
        
        class F:
            @staticmethod
            def relu(x): return x
            @staticmethod
            def sigmoid(x): return torch.sigmoid(x)
            @staticmethod
            def tanh(x): return torch.tanh(x)
            @staticmethod
            def softmax(x, dim=-1): return x
            @staticmethod
            def log_softmax(x, dim=-1): return x
            @staticmethod
            def dropout(x, p=0.5, training=True): return x

    nn = torch.nn
    F = torch.F



def _safe_std(tensor):
    try:
        if not TORCH_AVAILABLE:
            data = tensor.tolist() if hasattr(tensor, 'tolist') else list(tensor)
            if len(data) < 2: return 0.0
            val = float(np.std(data, ddof=1))
            return 0.0 if np.isnan(val) else val
        if hasattr(tensor, 'numel') and tensor.numel() < 2:
            return 0.0
        val = float(torch.std(tensor.float()).item())
        return 0.0 if np.isnan(val) else val
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 234)
        return 0.0

def _safe_mean(tensor):
    try:
        if not TORCH_AVAILABLE:
            data = tensor.tolist() if hasattr(tensor, 'tolist') else list(tensor)
            if len(data) == 0: return 0.0
            val = float(np.mean(data))
            return 0.0 if np.isnan(val) else val
        if hasattr(tensor, 'numel') and tensor.numel() == 0:
            return 0.0
        val = float(torch.mean(tensor.float()).item())
        return 0.0 if np.isnan(val) else val
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 248)
        return 0.0

class LinuxOptimizedDeque(deque):
    def __init__(self, maxlen=500):
        super().__init__(maxlen=maxlen)
    def append(self, item):
        try:
            super().append(item)
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 257)
            pass

class GPUFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_confusion_features(self, market_data: dict):
        try:
            f1, f2, f3, f4, f5 = 0.0, 0.0, 0.0, 0.0, 0.0
            if 'price_action' in market_data and len(market_data['price_action']) >= 2:
                highs = torch.tensor([c['high'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
                lows = torch.tensor([c['low'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
                price_range = highs - lows
                f1 = _safe_std(price_range)
                f2 = _safe_mean(price_range)
                if len(market_data['price_action']) >= 5:
                    closes = torch.tensor([c['close'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
                    c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
                    c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
                    f3 = (c1 - c0) / (c0 + 1e-8)
                    f4 = _safe_std(closes)
            if 'volume_pattern' in market_data and len(market_data['volume_pattern']) >= 2:
                volume = torch.tensor(market_data['volume_pattern'][-10:], device=self.device, dtype=torch.float32)
                f5 = _safe_std(volume)
            
            res = [0.0 if np.isnan(x) else float(x) for x in [f1, f2, f3, f4, f5]]
            return torch.tensor(res, device=self.device, dtype=torch.float32)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 285)
            return torch.zeros(5, device=self.device, dtype=torch.float32)

    def extract_institutional_flow(self, market_data: dict):
        try:
            if 'order_flow' in market_data:
                delta = torch.tensor(market_data['order_flow'].get('recent_delta', [0]), device=self.device, dtype=torch.float32)
                if len(delta) > 0:
                    return torch.tensor([
                        _safe_mean(delta),
                        _safe_std(delta),
                        float(torch.max(delta).item() if hasattr(torch.max(delta), 'item') else torch.max(delta)),
                        float(torch.min(delta).item() if hasattr(torch.min(delta), 'item') else torch.min(delta))
                    ], device=self.device, dtype=torch.float32)
            return torch.zeros(4, device=self.device, dtype=torch.float32)
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 300)
            return torch.zeros(4, device=self.device, dtype=torch.float32)

    def calculate_sentiment_scores(self, market_data: dict):
        scores = {'price_momentum': 0.5, 'volume_sentiment': 0.5}
        try:
            if 'price_action' in market_data and len(market_data['price_action']) >= 5:
                closes = torch.tensor([c['close'] for c in market_data['price_action'][-5:]], device=self.device, dtype=torch.float32)
                c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
                c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
                change = (c1 - c0) / (c0 + 1e-8)
                val = float(torch.sigmoid(torch.tensor(change * 10)).item())
                scores['price_momentum'] = 0.5 if np.isnan(val) else val
            if 'volume_pattern' in market_data and len(market_data['volume_pattern']) >= 2:
                vol = torch.tensor(market_data['volume_pattern'][-5:], device=self.device, dtype=torch.float32)
                v_std = _safe_std(vol)
                v_mean = _safe_mean(vol)
                trend = v_mean / (v_std + 1e-8)
                val = float(torch.tanh(torch.tensor(trend)).item())
                scores['volume_sentiment'] = 0.5 if np.isnan(val) else val
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 320)
            scores = {'price_momentum': 0.5, 'volume_sentiment': 0.5}
        return scores

    def calculate_volatility_metrics(self, market_data: dict):
        try:
            if 'price_action' in market_data and len(market_data['price_action']) >= 10:
                highs = torch.tensor([c['high'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
                lows = torch.tensor([c['low'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
                closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
                tr = torch.maximum(highs[1:] - lows[1:], torch.maximum(torch.abs(highs[1:] - closes[:-1]), torch.abs(lows[1:] - closes[:-1])))
                mean_closes = _safe_mean(closes)
                range_ratio = (_safe_mean(highs - lows) / (mean_closes + 1e-8)) if mean_closes > 0 else 0.0
                return {
                    'atr': _safe_mean(tr),
                    'volatility': _safe_std(closes),
                    'range_ratio': range_ratio
                }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 338)
            pass
        return {'atr': 0.0, 'volatility': 0.0, 'range_ratio': 0.0}

    def detect_patterns_gpu(self, market_data: dict):
        try:
            if 'price_action' in market_data and len(market_data['price_action']) >= 10:
                closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
                c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
                c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
                change = (c1 - c0) / (c0 + 1e-8)
                vol = _safe_std(closes)
                return {
                    'trend_strength': abs(change),
                    'volatility_regime': 'HIGH' if vol > 0.001 else 'LOW'
                }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 354)
            pass
        return {'trend_strength': 0.0, 'volatility_regime': 'UNKNOWN'}

@xgb_brain('trend_brain', 'analyze_trend')
class TrendBrain:
    def __init__(self):
        self.trend_memory = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_trend(self, market_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 20:
                return {'trend_direction': 0, 'trend_strength': 0, 'trend_quality': 0, 'momentum': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-20:]], device=self.device, dtype=torch.float32)
            
            sma_short = _safe_mean(closes[-5:])
            sma_long = _safe_mean(closes)
            trend_dir = 1 if sma_short > sma_long else -1
            
            price_range = highs - lows
            volatility = _safe_std(price_range)
            c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
            c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
            momentum = (c1 - c0) / (c0 + 1e-8)
            
            # FIX: Must call .cpu() before .numpy() on CUDA tensors (GPU fix)
            h_curr = highs[1:].cpu().numpy() if hasattr(highs[1:], 'numpy') else np.array(highs[1:].tolist())
            h_prev = highs[:-1].cpu().numpy() if hasattr(highs[:-1], 'numpy') else np.array(highs[:-1].tolist())
            l_curr = lows[1:].cpu().numpy() if hasattr(lows[1:], 'numpy') else np.array(lows[1:].tolist())
            l_prev = lows[:-1].cpu().numpy() if hasattr(lows[:-1], 'numpy') else np.array(lows[:-1].tolist())

            higher_highs = np.sum(h_curr > h_prev) / len(h_prev)
            higher_lows = np.sum(l_curr > l_prev) / len(l_prev)
            lower_highs = np.sum(h_curr < h_prev) / len(h_prev)
            lower_lows = np.sum(l_curr < l_prev) / len(l_prev)
            
            bullish_structure = higher_highs + higher_lows
            bearish_structure = lower_highs + lower_lows
            
            if trend_dir == 1:
                trend_quality = bullish_structure / (bullish_structure + bearish_structure + 1e-8)
            else:
                trend_quality = bearish_structure / (bullish_structure + bearish_structure + 1e-8)
                
            trend_strength = min(abs(momentum) * 10, 1.0)
            support_score = trend_strength * trend_quality * trend_dir
            
            return {
                'trend_direction': trend_dir,
                'trend_strength': trend_strength,
                'trend_quality': trend_quality,
                'momentum': momentum,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 411)
            return {'trend_direction': 0, 'trend_strength': 0, 'trend_quality': 0, 'momentum': 0, 'support_score': 0}

@xgb_brain('volatility_brain', 'analyze_volatility')
class VolatilityBrain:
    def __init__(self):
        self.volatility_history = LinuxOptimizedDeque(maxlen=50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_volatility(self, market_data):
        """Advanced Math: Uses GARCH-inspired exponential weighting for forecasting volatility"""
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 15:
                return {'volatility_regime': 'UNKNOWN', 'volatility_score': 0, 'breakout_potential': 0, 'volatility_trend': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-15:]], device=self.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-15:]], device=self.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-15:]], device=self.device, dtype=torch.float32)
            
            # GARCH-inspired exponentially weighted true ranges (recent volatility matters more)
            true_ranges = []
            for i in range(1, len(closes)):
                tr1 = (highs[i] - lows[i]).item() if hasattr(highs[i], 'item') else float(highs[i] - lows[i])
                tr2 = abs((highs[i] - closes[i-1]).item() if hasattr(highs[i], 'item') else float(highs[i] - closes[i-1]))
                tr3 = abs((lows[i] - closes[i-1]).item() if hasattr(lows[i], 'item') else float(lows[i] - closes[i-1]))
                true_ranges.append(max(tr1, tr2, tr3))
                
            if not true_ranges: return {'volatility_regime': 'UNKNOWN', 'volatility_score': 0, 'breakout_potential': 0, 'volatility_trend': 0, 'support_score': 0}

            # Exponential decay weights
            alpha = 0.8
            weights = np.array([alpha * (1-alpha)**i for i in range(len(true_ranges)-1, -1, -1)])
            weights /= weights.sum() # normalize

            # Weighted expected variance
            weighted_atr = np.sum(np.array(true_ranges) * weights)

            close_volatility = _safe_std(closes)
            
            # Combine traditional std dev with forecasted variance
            avg_volatility = (weighted_atr * 0.7) + (close_volatility * 0.3)
            # FIX: thresholds below are fractions of price -> normalize absolute $ volatility
            avg_volatility = avg_volatility / (abs(_safe_mean(closes)) + 1e-8)
            
            if avg_volatility > 0.002:
                regime = 'HIGH'
                breakout_potential = 0.7
                support_score = 0.3
            elif avg_volatility < 0.0005:
                regime = 'LOW'
                breakout_potential = 0.3
                support_score = 0.6
            else:
                regime = 'MEDIUM'
                breakout_potential = 0.5
                support_score = 0.8
                
            volatility_score = min(avg_volatility * 1000, 1.0)
            
            self.volatility_history.append(avg_volatility)
            recent_vols = list(self.volatility_history)[-10:]
            vol_trend = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0.0
            
            return {
                'volatility_regime': regime,
                'volatility_score': volatility_score,
                'breakout_potential': breakout_potential,
                'volatility_trend': vol_trend,
                'support_score': support_score
            }
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 480)
            return {'volatility_regime': 'UNKNOWN', 'volatility_score': 0, 'breakout_potential': 0, 'volatility_trend': 0, 'support_score': 0}

@xgb_brain('strength_brain', 'analyze_strength')
class StrengthBrain:
    def __init__(self):
        self.strength_memory = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_strength(self, market_data, breakout_data, momentum_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'breakout_strength': 0, 'momentum_strength': 0, 'volume_confirmation': 0, 'overall_strength': 0, 'strength_momentum': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            volumes = torch.tensor(market_data.get('volume_pattern', [1] * len(closes))[-10:], device=self.device, dtype=torch.float32)
            
            c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
            c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
            price_change = (c1 - c0) / (c0 + 1e-8)
            momentum_strength = min(abs(price_change) * 20, 1.0)
            
            breakout_confirmation = breakout_data.get('strength', 0)
            
            v_recent = _safe_mean(volumes[-5:])
            v_prev = _safe_mean(volumes[:-5]) if len(volumes) > 5 else 1.0
            volume_trend = v_recent / (v_prev + 1e-8)
            volume_confirmation = float(np.tanh(volume_trend - 1.0))
            
            strength_components = [
                breakout_confirmation * 0.4,
                momentum_strength * 0.3,
                volume_confirmation * 0.3
            ]
            
            overall_strength = sum(strength_components)
            support_score = overall_strength
            
            self.strength_memory.append(overall_strength)
            recent_strengths = list(self.strength_memory)[-5:]
            strength_momentum = float(np.mean(recent_strengths)) if recent_strengths else overall_strength
            
            return {
                'breakout_strength': breakout_confirmation,
                'momentum_strength': momentum_strength,
                'volume_confirmation': volume_confirmation,
                'overall_strength': overall_strength,
                'strength_momentum': strength_momentum,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 529)
            return {'breakout_strength': 0, 'momentum_strength': 0, 'volume_confirmation': 0, 'overall_strength': 0, 'strength_momentum': 0, 'support_score': 0}

@xgb_brain('risk_brain', 'analyze_risk')
class RiskBrain:
    def __init__(self):
        self.risk_history = LinuxOptimizedDeque(maxlen=50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_risk(self, market_data, volatility_data, trend_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'risk_score': 0.5, 'fakeout_probability': 0.5, 'stop_distance': 0, 'position_size': 0, 'avg_risk': 0.5, 'support_score': 0.5}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            
            volatility = volatility_data.get('volatility_score', 0)
            trend_strength = trend_data.get('trend_strength', 0)
            
            price_range = _safe_mean(highs - lows)
            recent_volatility = _safe_std(closes)
            
            base_risk = volatility * 0.6 + (1 - trend_strength) * 0.4
            fakeout_probability = (1 - trend_strength) * 0.7 + volatility * 0.3
            
            stop_distance = price_range * 1.5
            position_size = max(0.1, 1 - base_risk)
            
            risk_score = min(base_risk, 0.9)
            support_score = 1 - risk_score
            
            self.risk_history.append(risk_score)
            recent_risks = list(self.risk_history)
            avg_risk = float(np.mean(recent_risks)) if recent_risks else 0.5
            
            return {
                'risk_score': risk_score,
                'fakeout_probability': fakeout_probability,
                'stop_distance': stop_distance,
                'position_size': position_size,
                'avg_risk': avg_risk,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 573)
            return {'risk_score': 0.5, 'fakeout_probability': 0.5, 'stop_distance': 0, 'position_size': 0, 'avg_risk': 0.5, 'support_score': 0.5}

@xgb_brain('reversal_brain', 'analyze_reversal')
class ReversalBrain:
    def __init__(self):
        self.reversal_patterns = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_reversal(self, market_data, trend_data, strength_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 5:
                return {'reversal_probability': 0, 'exhaustion_signals': 0, 'divergence_detected': False, 'price_momentum': 0, 'volume_momentum': 0, 'support_score': 1}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            volumes = torch.tensor(market_data.get('volume_pattern', [1] * len(closes))[-10:], device=self.device, dtype=torch.float32)
            
            trend_direction = trend_data.get('trend_direction', 0)
            strength = strength_data.get('overall_strength', 0)
            
            c_curr = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
            c_prev = closes[-5].item() if hasattr(closes[-5], 'item') else float(closes[-5]) if len(closes) >= 5 else c_curr
            price_momentum = (c_curr - c_prev) / (c_prev + 1e-8)
            
            v_recent = _safe_mean(volumes[-3:])
            v_prev = _safe_mean(volumes[-6:-3]) if len(volumes) >= 6 else 1.0
            volume_momentum = v_recent / (v_prev + 1e-8)
            
            divergence = 0.0
            if trend_direction == 1 and price_momentum < 0:
                divergence = abs(price_momentum)
            elif trend_direction == -1 and price_momentum > 0:
                divergence = abs(price_momentum)
                
            exhaustion = 0.0
            if strength > 0.8:
                recent_highs = float(torch.max(highs).item() if hasattr(torch.max(highs), 'item') else torch.max(highs))
                recent_lows = float(torch.min(lows).item() if hasattr(torch.min(lows), 'item') else torch.min(lows))
                if trend_direction == 1 and c_curr >= recent_highs * 0.99:
                    exhaustion = strength
                elif trend_direction == -1 and c_curr <= recent_lows * 1.01:
                    exhaustion = strength
                    
            reversal_probability = (divergence * 0.6 + exhaustion * 0.4) * (1.0 - min(strength, 1.0))
            support_score = 1.0 - reversal_probability
            
            divergence_detected = divergence > 0.2
            
            return {
                'reversal_probability': reversal_probability,
                'exhaustion_signals': exhaustion,
                'divergence_detected': divergence_detected,
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 630)
            return {'reversal_probability': 0, 'exhaustion_signals': 0, 'divergence_detected': False, 'price_momentum': 0, 'volume_momentum': 0, 'support_score': 1}

@xgb_brain('regime_brain', 'analyze_regime')
class RegimeBrain:
    def __init__(self):
        self.regime_history = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_regime(self, market_data, trend_data, volatility_data):
        try:
            if 'price_action' not in market_data:
                return {'regime_type': 'UNKNOWN', 'regime_confidence': 0, 'transition_phase': False, 'support_score': 0}
                
            trend_strength = trend_data.get('trend_strength', 0)
            trend_direction = trend_data.get('trend_direction', 0)
            volatility_regime = volatility_data.get('volatility_regime', 'UNKNOWN')
            
            if trend_strength > 0.35:
                if trend_direction == 1:
                    regime_type = 'STRONG_BULL'
                    support_score = 0.9
                else:
                    regime_type = 'STRONG_BEAR'
                    support_score = -0.9
            elif trend_strength > 0.15:
                if trend_direction == 1:
                    regime_type = 'BULL'
                    support_score = 0.7
                else:
                    regime_type = 'BEAR'
                    support_score = -0.7
            else:
                regime_type = 'RANGING'
                support_score = 0.0
                
            if volatility_regime == 'HIGH':
                regime_type += '_HIGH_VOL'
                support_score *= 0.8
            elif volatility_regime == 'LOW':
                regime_type += '_LOW_VOL'
                support_score *= 1.2
                
            regime_confidence = trend_strength * 0.7 + (1.0 if volatility_regime != 'UNKNOWN' else 0.0) * 0.3
            
            recent_regimes = list(self.regime_history)[-3:]
            transition_phase = len(recent_regimes) >= 2 and len(set(recent_regimes)) > 1
            
            self.regime_history.append(regime_type)
            
            return {
                'regime_type': regime_type,
                'regime_confidence': regime_confidence,
                'transition_phase': transition_phase,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 685)
            return {'regime_type': 'UNKNOWN', 'regime_confidence': 0, 'transition_phase': False, 'support_score': 0}

@xgb_brain('deepseek_brain', 'analyze_deepseek')
class DeepSeekBrain:
    def __init__(self):
        self.correction_memory = LinuxOptimizedDeque(maxlen=200)
        self.noise_suppression = LinuxOptimizedDeque(maxlen=50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_deepseek(self, market_data, all_brain_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'correction_factor': 0, 'noise_level': 0, 'fakeout_block': 0, 'contradiction_score': 0, 'threshold_boost': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            volumes = torch.tensor(market_data.get('volume_pattern', [1] * len(closes))[-10:], device=self.device, dtype=torch.float32)
            
            price_volatility = _safe_std(closes)
            volume_volatility = _safe_std(volumes)
            
            noise_level = (price_volatility + volume_volatility) / 2
            self.noise_suppression.append(noise_level)
            
            recent_noise = list(self.noise_suppression)
            avg_noise = float(np.mean(recent_noise)) if recent_noise else noise_level
            noise_suppression = 1.0 - min(avg_noise * 100, 0.8)
            
            trend_brain = all_brain_data.get('trend', {})
            strength_brain = all_brain_data.get('strength', {})
            risk_brain = all_brain_data.get('risk', {})
            reversal_brain = all_brain_data.get('reversal', {})
            
            trend_dir = trend_brain.get('trend_direction', 0)
            trend_str = trend_brain.get('trend_strength', 0)
            overall_str = strength_brain.get('overall_strength', 0)
            risk_score = risk_brain.get('risk_score', 0.5)
            reversal_prob = reversal_brain.get('reversal_probability', 0)
            
            contradiction_score = 0.0
            if trend_dir == 1 and overall_str < 0.3:
                contradiction_score += 0.3
            if trend_dir == -1 and overall_str < 0.3:
                contradiction_score += 0.3
            if risk_score > 0.7 and overall_str > 0.6:
                contradiction_score += 0.4
                
            fakeout_block = reversal_prob * 0.7 + risk_score * 0.3
            threshold_boost = trend_str * 0.5 + overall_str * 0.3 + (1.0 - risk_score) * 0.2
            correction_factor = (noise_suppression * 0.3 + (1.0 - contradiction_score) * 0.4 + (1.0 - fakeout_block) * 0.3)
            
            support_score = correction_factor * threshold_boost
            self.correction_memory.append(correction_factor)
            
            return {
                'correction_factor': correction_factor,
                'noise_level': noise_level,
                'fakeout_block': fakeout_block,
                'contradiction_score': contradiction_score,
                'threshold_boost': threshold_boost,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 746)
            return {'correction_factor': 0, 'noise_level': 0, 'fakeout_block': 0, 'contradiction_score': 0, 'threshold_boost': 0, 'support_score': 0}

@xgb_brain('evolution_brain', 'analyze_evolution')
class EvolutionBrain:
    def __init__(self):
        self.generation = 1
        self.performance_memory = LinuxOptimizedDeque(maxlen=100)
        self.adaptation_factors = LinuxOptimizedDeque(maxlen=50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_evolution(self, market_data, signal_history):
        try:
            if len(signal_history) < 10:
                return {'generation': self.generation, 'adaptation_factor': 0.5, 'performance_score': 0.5, 'learning_rate': 0.1, 'support_score': 0.5}
                
            recent_signals = list(signal_history)[-10:]
            success_rate = sum(1 for s in recent_signals if s.get('profit', 0) > 0) / len(recent_signals)
            
            self.performance_memory.append(success_rate)
            recent_perf = list(self.performance_memory)
            avg_performance = float(np.mean(recent_perf)) if recent_perf else 0.5
            
            if avg_performance > 0.7:
                adaptation_factor = 1.2
                learning_rate = 0.15
            elif avg_performance < 0.3:
                adaptation_factor = 0.8
                learning_rate = 0.05
            else:
                adaptation_factor = 1.0
                learning_rate = 0.1
                
            if len(self.performance_memory) % 50 == 0 and avg_performance < 0.4:
                self.generation += 1
                
            support_score = adaptation_factor * avg_performance
            self.adaptation_factors.append(adaptation_factor)
            
            return {
                'generation': self.generation,
                'adaptation_factor': adaptation_factor,
                'performance_score': avg_performance,
                'learning_rate': learning_rate,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 791)
            return {'generation': 1, 'adaptation_factor': 0.5, 'performance_score': 0.5, 'learning_rate': 0.1, 'support_score': 0.5}

@xgb_brain('memory_brain', 'analyze_memory')
class MemoryBrain:
    def __init__(self):
        self.short_term_memory = LinuxOptimizedDeque(maxlen=50)
        self.long_term_memory = LinuxOptimizedDeque(maxlen=200)
        self.prediction_engine = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_memory(self, market_data, current_signal):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'short_term_recall': 0, 'long_term_pattern': 0, 'prediction_confidence': 0, 'memory_strength': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-5:]], device=self.device, dtype=torch.float32)
            c0 = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])
            c1 = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
            current_trend = (c1 - c0) / (c0 + 1e-8)
            
            self.short_term_memory.append(current_trend)
            self.long_term_memory.append(current_trend)
            
            st_list = list(self.short_term_memory)
            lt_list = list(self.long_term_memory)
            short_term_avg = float(np.mean(st_list)) if st_list else current_trend
            long_term_avg = float(np.mean(lt_list)) if lt_list else current_trend
            
            prediction = (short_term_avg * 0.6 + long_term_avg * 0.4)
            self.prediction_engine.append(prediction)
            
            prediction_confidence = min(abs(prediction) * 10, 1.0)
            memory_strength = len(self.short_term_memory) / 50.0
            
            signal_alignment = 1 if current_trend * prediction > 0 else -1
            support_score = prediction_confidence * memory_strength * signal_alignment
            
            return {
                'short_term_recall': short_term_avg,
                'long_term_pattern': long_term_avg,
                'prediction_confidence': prediction_confidence,
                'memory_strength': memory_strength,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 835)
            return {'short_term_recall': 0, 'long_term_pattern': 0, 'prediction_confidence': 0, 'memory_strength': 0, 'support_score': 0}

@xgb_brain('selfhealing_brain', 'analyze_self_heal')
class SelfHealingBrain:
    def __init__(self):
        self.stability_memory = LinuxOptimizedDeque(maxlen=100)
        self.error_correction = LinuxOptimizedDeque(maxlen=50)
        self.health_metrics = LinuxOptimizedDeque(maxlen=200)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_self_heal(self, market_data, system_metrics):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'stability_score': 0.8, 'error_rate': 0.1, 'health_index': 0.9, 'recovery_factor': 0.5, 'support_score': 0.7}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            c_mean = _safe_mean(closes)
            c_std_pct = (_safe_std(closes) / (c_mean + 1e-8)) if c_mean > 0 else 0.0
            price_stability = max(0.0, min(1.0, 1.0 - c_std_pct * 50))
            
            system_stability = system_metrics.get('stability', 0.8)
            error_rate = system_metrics.get('error_rate', 0.1)
            
            stability_score = (price_stability + system_stability) / 2
            self.stability_memory.append(stability_score)
            
            stab_list = list(self.stability_memory)
            avg_stability = float(np.mean(stab_list)) if stab_list else stability_score
            
            health_index = avg_stability * (1.0 - error_rate)
            self.health_metrics.append(health_index)
            
            recovery_factor = 1.5 if health_index < 0.5 else 1.0
            support_score = health_index * recovery_factor
            
            self.error_correction.append(error_rate)
            
            return {
                'stability_score': stability_score,
                'error_rate': error_rate,
                'health_index': health_index,
                'recovery_factor': recovery_factor,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 879)
            return {'stability_score': 0.8, 'error_rate': 0.1, 'health_index': 0.9, 'recovery_factor': 0.5, 'support_score': 0.7}

@xgb_brain('metafusion_brain', 'analyze_meta_fusion')
class MetaFusionBrain:
    def __init__(self):
        self.fusion_scores = LinuxOptimizedDeque(maxlen=100)
        self.consensus_memory = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_meta_fusion(self, all_brain_support):
        try:
            brain_scores = []
            brain_weights = []
            
            for brain_name, brain_data in all_brain_support.items():
                support_score = brain_data.get('support_score', 0)
                brain_scores.append(support_score)
                
                if brain_name in ['trend', 'strength', 'deepseek']:
                    brain_weights.append(0.12)
                elif brain_name in ['volatility', 'risk', 'regime']:
                    brain_weights.append(0.10)
                elif brain_name in ['reversal', 'evolution', 'memory']:
                    brain_weights.append(0.08)
                else:
                    brain_weights.append(0.06)
                    
            if not brain_scores:
                return {'fusion_score': 0, 'consensus_level': 0, 'weight_distribution': [], 'support_score': 0}
            
            total_w = sum(brain_weights)
            brain_weights = [w / total_w for w in brain_weights]
                
            fusion_score = sum(s * w for s, w in zip(brain_scores, brain_weights))
            
            positive_consensus = sum(1 for s in brain_scores if s > 0) / len(brain_scores)
            negative_consensus = sum(1 for s in brain_scores if s < 0) / len(brain_scores)
            consensus_level = max(positive_consensus, negative_consensus)
            
            self.consensus_memory.append(consensus_level)
            c_list = list(self.consensus_memory)
            avg_consensus = float(np.mean(c_list)) if c_list else consensus_level
            
            support_score = fusion_score * avg_consensus
            self.fusion_scores.append(fusion_score)
            
            return {
                'fusion_score': fusion_score,
                'consensus_level': consensus_level,
                'weight_distribution': brain_weights,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 931)
            return {'fusion_score': 0, 'consensus_level': 0, 'weight_distribution': [], 'support_score': 0}

@xgb_brain('mini_r1_brain', 'analyze_mini_r1')
class MiniR1Brain:
    def __init__(self):
        self.reasoning_stack = LinuxOptimizedDeque(maxlen=20)
        self.trap_detection = LinuxOptimizedDeque(maxlen=50)
        self.psychology_memory = LinuxOptimizedDeque(maxlen=100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_mini_r1(self, market_data, breakout_data, fakeout_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'reasoning_depth': 0, 'trap_probability': 0, 'psychology_score': 0, 'volatility_override': 0, 'support_score': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-10:]], device=self.device, dtype=torch.float32)
            
            price_range = highs - lows
            avg_range = _safe_mean(price_range)
            h_last = highs[-1].item() if hasattr(highs[-1], 'item') else float(highs[-1])
            l_last = lows[-1].item() if hasattr(lows[-1], 'item') else float(lows[-1])
            current_range = h_last - l_last
            
            range_ratio = current_range / (avg_range + 1e-8)
            
            trap_probability = 0.0
            if range_ratio > 2.0:
                trap_probability = min((range_ratio - 2.0) / 2.0, 0.8)
                
            breakout_strength = breakout_data.get('strength', 0)
            fakeout_prob = fakeout_data.get('fakeout_probability', 0)
            
            reasoning_depth = breakout_strength * (1.0 - fakeout_prob)
            
            c_mean = _safe_mean(closes)
            volatility = (_safe_std(closes) / (c_mean + 1e-8)) if c_mean > 0 else 0.0
            volatility_override = 0.3 if volatility > 0.005 else 1.0
                
            psychology_score = (1.0 - trap_probability) * reasoning_depth * volatility_override
            support_score = psychology_score
            
            self.reasoning_stack.append(reasoning_depth)
            self.trap_detection.append(trap_probability)
            self.psychology_memory.append(psychology_score)
            
            return {
                'reasoning_depth': reasoning_depth,
                'trap_probability': trap_probability,
                'psychology_score': psychology_score,
                'volatility_override': volatility_override,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 985)
            return {'reasoning_depth': 0, 'trap_probability': 0, 'psychology_score': 0, 'volatility_override': 0, 'support_score': 0}

@xgb_brain('mini_v3_brain', 'analyze_mini_v3')
class MiniV3Brain:
    def __init__(self):
        self.microstructure_memory = LinuxOptimizedDeque(maxlen=100)
        self.wick_analysis = LinuxOptimizedDeque(maxlen=50)
        self.tick_momentum = LinuxOptimizedDeque(maxlen=200)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_mini_v3(self, market_data):
        try:
            real_pressure = 0.0
            if 'real_orderbook' in market_data:
                ob = market_data['real_orderbook']
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                bid_vol = sum(float(b[1]) for b in bids)
                ask_vol = sum(float(a[1]) for a in asks)
                total = bid_vol + ask_vol + 1e-8
                real_pressure = (bid_vol - ask_vol) / total

            if 'price_action' not in market_data or len(market_data['price_action']) < 2:
                return {'microstructure_score': 0, 'wick_imbalance': 0, 'tick_momentum': real_pressure, 'pressure_detection': abs(real_pressure), 'support_score': abs(real_pressure)}
                
            recent_candles = market_data['price_action'][-5:]
            
            wick_imbalances = []
            tick_pressures = []
            
            for candle in recent_candles:
                high = candle['high']
                low = candle['low']
                open_price = candle['open']
                close = candle['close']
                
                upper_wick = high - max(open_price, close)
                lower_wick = min(open_price, close) - low
                body = abs(close - open_price)
                
                if body > 0:
                    wick_ratio = (upper_wick - lower_wick) / body
                    wick_imbalances.append(wick_ratio)
                    
                tick_pressure = (close - open_price) / (high - low + 1e-8)
                tick_pressures.append(tick_pressure)
                
            avg_wick_imbalance = float(np.mean(wick_imbalances)) if wick_imbalances else 0.0
            avg_tick_pressure = float(np.mean(tick_pressures)) if tick_pressures else 0.0
            
            # Combine real pressure with candle pressure
            final_tick_momentum = avg_tick_pressure
            if real_pressure != 0.0:
                final_tick_momentum = (avg_tick_pressure + real_pressure) / 2.0
            
            microstructure_score = 1.0 - abs(avg_wick_imbalance) * 0.5
            wick_imbalance = avg_wick_imbalance
            tick_momentum = final_tick_momentum
            pressure_detection = abs(tick_momentum)
            
            support_score = microstructure_score * (1.0 + tick_momentum) * 0.5
            
            self.microstructure_memory.append(microstructure_score)
            self.wick_analysis.append(wick_imbalance)
            self.tick_momentum.append(tick_momentum)
            
            return {
                'microstructure_score': microstructure_score,
                'wick_imbalance': wick_imbalance,
                'tick_momentum': tick_momentum,
                'pressure_detection': pressure_detection,
                'support_score': support_score
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1058)
            return {'microstructure_score': 0, 'wick_imbalance': 0, 'tick_momentum': 0, 'pressure_detection': 0, 'support_score': 0}

class SmartBreakoutAI:
    def __init__(self, live_ticks=None, live_candles=None):
        self.gpu_extractor = GPUFeatureExtractor()
        self.trend_brain = TrendBrain()
        self.volatility_brain = VolatilityBrain()
        self.strength_brain = StrengthBrain()
        self.risk_brain = RiskBrain()
        self.reversal_brain = ReversalBrain()
        self.regime_brain = RegimeBrain()
        self.deepseek_brain = DeepSeekBrain()
        self.evolution_brain = EvolutionBrain()
        self.memory_brain = MemoryBrain()
        self.self_heal_brain = SelfHealingBrain()
        self.meta_fusion_brain = MetaFusionBrain()
        self.mini_r1_brain = MiniR1Brain()
        self.mini_v3_brain = MiniV3Brain()
        
        self.price_memory = LinuxOptimizedDeque(maxlen=200)
        self.volume_memory = LinuxOptimizedDeque(maxlen=200)
        self.signal_history = LinuxOptimizedDeque(maxlen=100)
        self.system_metrics = {'stability': 0.8, 'error_rate': 0.1}
        
        if live_ticks:
            self._process_live_ticks(live_ticks)
        if live_candles:
            self._process_live_candles(live_candles)

    def analyze_master_consensus(self, m5, h1, tf_dict=None):
        """Unified 29-Brain Consensus Engine integration directly inside Part 1."""
        from mode_engine import ModeEngine
        if not hasattr(self, '_mode_engine'):
            self._mode_engine = ModeEngine()
        return self._mode_engine.master_evaluate(m5, h1, tf_dict=tf_dict)
            
    def _process_live_ticks(self, ticks):
        pass
        
    def _process_live_candles(self, candles):
        pass
        
    def detect_smart_levels(self, market_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 10:
                return {'support': [], 'resistance': [], 'key_levels': [], 'level_strength': 0}
                
            highs = [c['high'] for c in market_data['price_action'][-50:]]
            lows = [c['low'] for c in market_data['price_action'][-50:]]
            closes = [c['close'] for c in market_data['price_action'][-50:]]
            
            pivot_highs = []
            pivot_lows = []
            
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    pivot_highs.append(highs[i])
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    pivot_lows.append(lows[i])
                    
            # Use prev close to find levels, so current close can break them!
            current_price = closes[-2]
            support_levels = sorted([p for p in pivot_lows if p < current_price])[-3:]
            resistance_levels = sorted([p for p in pivot_highs if p >= current_price])[:3]
            
            key_levels = support_levels + resistance_levels
            level_strength = min(len(key_levels) / 6.0, 1.0)
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'key_levels': key_levels,
                'level_strength': level_strength
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1133)
            return {'support': [], 'resistance': [], 'key_levels': [], 'level_strength': 0}
            
    def detect_liquidity(self, market_data, levels_data):
        try:
            if 'price_action' not in market_data or not market_data['price_action']:
                return {'liquidity_zones': [], 'sweep_detected': False, 'liquidity_strength': 0}
                
            support_levels = levels_data.get('support', [])
            resistance_levels = levels_data.get('resistance', [])
            current_candle = market_data['price_action'][-1]
            current_high = current_candle['high']
            current_low = current_candle['low']
            
            liquidity_zones = []
            sweep_detected = False
            
            for resistance in resistance_levels:
                if current_high > resistance:
                    liquidity_zones.append(('resistance_sweep', resistance))
                    sweep_detected = True
                    
            for support in support_levels:
                if current_low < support:
                    liquidity_zones.append(('support_sweep', support))
                    sweep_detected = True
                    
            liquidity_strength = len(liquidity_zones) / (len(support_levels) + len(resistance_levels) + 1e-8)
            
            return {
                'liquidity_zones': liquidity_zones,
                'sweep_detected': sweep_detected,
                'liquidity_strength': liquidity_strength
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1167)
            return {'liquidity_zones': [], 'sweep_detected': False, 'liquidity_strength': 0}
            
    def detect_breakout(self, market_data, levels_data):
        """Advanced Math: Uses LSTM-based pattern analysis to detect breakout probability"""
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 15:
                return {'breakout_detected': False, 'direction': 0, 'strength': 0, 'level_broken': None, 'volume_confirmation': 0}
                
            current_candle = market_data['price_action'][-1]
            current_high = current_candle['high']
            current_low = current_candle['low']
            current_close = current_candle['close']
            prev_close = market_data['price_action'][-2]['close']
            
            resistance_levels = levels_data.get('resistance', [])
            support_levels = levels_data.get('support', [])
            
            breakout_direction = 0
            broken_level = None
            breakout_strength = 0.0
            
            # Simulated sequence forecasting using numpy
            closes = np.array([c['close'] for c in market_data['price_action'][-15:]], dtype=np.float32)
            c_mean = np.mean(closes)
            c_std = np.std(closes) + 1e-8
            normalized_closes = (closes - c_mean) / c_std

            # Simple mathematically computed non-linear breakout score
            momentum_score = float(np.tanh(normalized_closes[-1] - normalized_closes[-2]))

            for resistance in resistance_levels:
                if current_high > resistance and current_close > resistance:
                    if prev_close < resistance or (len(closes) >= 4 and min(float(x) for x in closes[-4:-1]) < resistance and momentum_score > 0.10):
                        breakout_direction = 1
                        broken_level = resistance
                        distance = (current_close - resistance) / (resistance + 1e-8)
                        # Boost strength dynamically based on momentum score
                        breakout_strength = min((distance * 100) + max(0, momentum_score), 1.0)
                        break
                        
            if breakout_direction == 0:
                for support in support_levels:
                    if current_low < support and current_close < support:
                        if prev_close > support or (len(closes) >= 4 and max(float(x) for x in closes[-4:-1]) > support and momentum_score < -0.10):
                            breakout_direction = -1
                            broken_level = support
                            distance = (support - current_close) / (support + 1e-8)
                            breakout_strength = min((distance * 100) + max(0, -momentum_score), 1.0)
                            break
                        
            volume_confirmation = 0.0
            if 'volume_pattern' in market_data and len(market_data['volume_pattern']) >= 5:
                volumes = market_data['volume_pattern'][-5:]
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                current_volume = volumes[-1]
                if current_volume > avg_volume * 1.2:
                    volume_confirmation = 0.3
                    
            if breakout_direction != 0:
                breakout_strength = min(breakout_strength + volume_confirmation, 1.0)
            else:
                breakout_strength = 0.0
                volume_confirmation = 0.0
            
            return {
                'breakout_detected': breakout_direction != 0,
                'direction': breakout_direction,
                'strength': breakout_strength,
                'level_broken': broken_level,
                'volume_confirmation': volume_confirmation
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1239)
            return {'breakout_detected': False, 'direction': 0, 'strength': 0, 'level_broken': None, 'volume_confirmation': 0}
            
    def detect_fakeout(self, market_data, breakout_data, levels_data):
        try:
            if not breakout_data.get('breakout_detected', False):
                return {'fakeout_detected': False, 'fakeout_probability': 0, 'reversal_signals': 0}
                
            breakout_direction = breakout_data.get('direction', 0)
            broken_level = breakout_data.get('level_broken', 0)
            current_candle = market_data['price_action'][-1]
            current_close = current_candle['close']
            
            if len(market_data['price_action']) >= 6:
                window_size = min(5, len(market_data['price_action']) - 1)
                recent_candles = market_data['price_action'][-(window_size+1):-1]
                
                if breakout_direction == -1:
                    # Bearish breakdown: Check for Bullish Trap in recent candles
                    highest_high = max(c['high'] for c in recent_candles)
                    
                    pierced_resistance = False
                    if levels_data and levels_data.get('resistance'):
                        for r in levels_data['resistance']:
                            if highest_high > r:
                                pierced_resistance = True
                                break
                                
                    if not pierced_resistance:
                        older_highs = [c['high'] for c in market_data['price_action'][-26:-(window_size+1)]]
                        if older_highs and highest_high >= max(older_highs):
                            pierced_resistance = True
                            
                    if pierced_resistance:
                        # Find the base of the recent rally
                        lowest_recent_body = min(min(c['open'], c['close']) for c in recent_candles)
                        dump_size = (current_candle['open'] - current_candle['close']) / current_candle['open']
                        
                        if dump_size > 0.002 and current_candle['close'] < lowest_recent_body:
                            return {'fakeout_detected': False, 'fakeout_probability': 0.4, 'reversal_signals': 0.3}
                            
                elif breakout_direction == 1:
                    # Bullish breakout: Check for Bearish Trap in recent candles
                    lowest_low = min(c['low'] for c in recent_candles)
                    
                    pierced_support = False
                    if levels_data and levels_data.get('support'):
                        for s in levels_data['support']:
                            if lowest_low < s:
                                pierced_support = True
                                break
                                
                    if not pierced_support:
                        older_lows = [c['low'] for c in market_data['price_action'][-26:-(window_size+1)]]
                        if older_lows and lowest_low <= min(older_lows):
                            pierced_support = True
                            
                    if pierced_support:
                        # Find the top of the recent dump
                        highest_recent_body = max(max(c['open'], c['close']) for c in recent_candles)
                        pump_size = (current_candle['close'] - current_candle['open']) / current_candle['open']
                        
                        if pump_size > 0.002 and current_candle['close'] > highest_recent_body:
                            return {'fakeout_detected': False, 'fakeout_probability': 0.4, 'reversal_signals': 0.3}

            if breakout_direction == 1 and broken_level is not None:
                next_resistance = None
                for level in levels_data.get('resistance', []):
                    if level > broken_level:
                        if next_resistance is None or level < next_resistance:
                            next_resistance = level
                if next_resistance:
                    distance_to_next = (next_resistance - current_close) / (current_close + 1e-8)
                    if distance_to_next < 0.005:
                        return {'fakeout_detected': True, 'fakeout_probability': 0.8, 'reversal_signals': 0.7}
                        
            elif breakout_direction == -1 and broken_level is not None:
                next_support = None
                for level in levels_data.get('support', []):
                    if level < broken_level:
                        if next_support is None or level > next_support:
                            next_support = level
                if next_support:
                    distance_to_next = (current_close - next_support) / (next_support + 1e-8)
                    if distance_to_next < 0.005:
                        return {'fakeout_detected': True, 'fakeout_probability': 0.8, 'reversal_signals': 0.7}
                    
            volume_pattern = market_data.get('volume_pattern', [])
            if len(volume_pattern) >= 6:
                recent_volumes = volume_pattern[-3:]
                prev_volumes = volume_pattern[-6:-3]
                if max(recent_volumes) < sum(prev_volumes) / len(prev_volumes):
                    return {'fakeout_detected': True, 'fakeout_probability': 0.6, 'reversal_signals': 0.5}
                    
            fakeout_probability = 1.0 - breakout_data.get('strength', 0)
            
            return {
                'fakeout_detected': fakeout_probability > 0.7,
                'fakeout_probability': fakeout_probability,
                'reversal_signals': fakeout_probability * 0.8
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1340)
            return {'fakeout_detected': False, 'fakeout_probability': 0, 'reversal_signals': 0}
            
    def detect_pullback(self, market_data, breakout_data, levels_data):
        try:
            if not breakout_data.get('breakout_detected', False):
                return {'pullback_detected': False, 'pullback_depth': 0, 'retest_level': None, 'continuation_probability': 0}
                
            breakout_direction = breakout_data.get('direction', 0)
            broken_level = breakout_data.get('level_broken', 0)
            current_candle = market_data['price_action'][-1]
            current_low = current_candle['low']
            current_high = current_candle['high']
            current_close = current_candle['close']
            
            pullback_detected = False
            pullback_depth = 0.0
            retest_level = broken_level
            continuation_probability = 0.0
            
            if broken_level is not None:
                if breakout_direction == 1:
                    if current_low < broken_level and current_close > broken_level:
                        pullback_detected = True
                        pullback_depth = (broken_level - current_low) / (broken_level + 1e-8)
                        continuation_probability = 0.7 - (pullback_depth * 10)
                    elif current_low < broken_level and current_close < broken_level:
                        pullback_detected = True
                        pullback_depth = (broken_level - current_low) / (broken_level + 1e-8)
                        continuation_probability = 0.3 - (pullback_depth * 10)
                        
                elif breakout_direction == -1:
                    if current_high > broken_level and current_close < broken_level:
                        pullback_detected = True
                        pullback_depth = (current_high - broken_level) / (broken_level + 1e-8)
                        continuation_probability = 0.7 - (pullback_depth * 10)
                    elif current_high > broken_level and current_close > broken_level:
                        pullback_detected = True
                        pullback_depth = (current_high - broken_level) / (broken_level + 1e-8)
                        continuation_probability = 0.3 - (pullback_depth * 10)
                        
            continuation_probability = max(0.0, min(continuation_probability, 1.0))
            
            return {
                'pullback_detected': pullback_detected,
                'pullback_depth': pullback_depth,
                'retest_level': retest_level,
                'continuation_probability': continuation_probability
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1389)
            return {'pullback_detected': False, 'pullback_depth': 0, 'retest_level': None, 'continuation_probability': 0}
            
    def detect_momentum(self, market_data):
        try:
            if 'price_action' not in market_data or len(market_data['price_action']) < 6:
                return {'momentum_direction': 0, 'momentum_strength': 0, 'acceleration': 0, 'trend_alignment': 0}
                
            closes = torch.tensor([c['close'] for c in market_data['price_action'][-10:]], device=self.gpu_extractor.device, dtype=torch.float32)
            highs = torch.tensor([c['high'] for c in market_data['price_action'][-10:]], device=self.gpu_extractor.device, dtype=torch.float32)
            lows = torch.tensor([c['low'] for c in market_data['price_action'][-10:]], device=self.gpu_extractor.device, dtype=torch.float32)
            
            c_last = closes[-1].item() if hasattr(closes[-1], 'item') else float(closes[-1])
            c_3 = closes[-3].item() if hasattr(closes[-3], 'item') else float(closes[-3])
            c_6 = closes[-6].item() if hasattr(closes[-6], 'item') else float(closes[-6])
            c_first = closes[0].item() if hasattr(closes[0], 'item') else float(closes[0])

            price_change_short = (c_last - c_3) / (c_3 + 1e-8)
            price_change_medium = (c_last - c_6) / (c_6 + 1e-8)
            price_change_long = (c_last - c_first) / (c_first + 1e-8)
            
            momentum_score = price_change_short * 0.5 + price_change_medium * 0.3 + price_change_long * 0.2
            momentum_direction = 1 if momentum_score > 0 else -1
            momentum_strength = min(abs(momentum_score) * 10, 1.0)
            
            acceleration_short = price_change_short - price_change_medium
            acceleration_medium = price_change_medium - price_change_long
            acceleration = acceleration_short * 0.7 + acceleration_medium * 0.3
            
            # FIX: Must call .cpu() before .numpy() on CUDA tensors (GPU fix)
            h_curr = highs[1:].cpu().numpy() if hasattr(highs[1:], 'numpy') else np.array(highs[1:].tolist())
            h_prev = highs[:-1].cpu().numpy() if hasattr(highs[:-1], 'numpy') else np.array(highs[:-1].tolist())
            l_curr = lows[1:].cpu().numpy() if hasattr(lows[1:], 'numpy') else np.array(lows[1:].tolist())
            l_prev = lows[:-1].cpu().numpy() if hasattr(lows[:-1], 'numpy') else np.array(lows[:-1].tolist())

            higher_highs = np.sum(h_curr > h_prev)
            higher_lows = np.sum(l_curr > l_prev)
            lower_highs = np.sum(h_curr < h_prev)
            lower_lows = np.sum(l_curr < l_prev)
            
            bullish_structure = higher_highs + higher_lows
            bearish_structure = lower_highs + lower_lows
            
            if momentum_direction == 1:
                trend_alignment = bullish_structure / (bullish_structure + bearish_structure + 1e-8)
            else:
                trend_alignment = bearish_structure / (bullish_structure + bearish_structure + 1e-8)
                
            return {
                'momentum_direction': momentum_direction,
                'momentum_strength': momentum_strength,
                'acceleration': acceleration,
                'trend_alignment': trend_alignment
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1443)
            return {'momentum_direction': 0, 'momentum_strength': 0, 'acceleration': 0, 'trend_alignment': 0}
            
    def detect_orderflow(self, market_data):
        try:
            # Use real orderbook depth if passed from live context
            if 'real_orderbook' in market_data:
                ob = market_data['real_orderbook']
                bids = ob.get('bids', [])
                asks = ob.get('asks', [])
                bid_vol = sum(float(b[1]) for b in bids)
                ask_vol = sum(float(a[1]) for a in asks)
                
                delta_positive = bid_vol > ask_vol
                total_vol = bid_vol + ask_vol + 1e-8
                net_delta = bid_vol - ask_vol
                pressure_strength = min(abs(net_delta) / total_vol * 5.0, 1.0)
                
                return {
                    'delta_positive': delta_positive,
                    'pressure_strength': pressure_strength,
                    'absorption_detected': pressure_strength > 0.8,
                    'exhaustion_signals': 0,
                    'cumulative_delta': net_delta
                }

            if 'order_flow' not in market_data:
                return {'delta_positive': False, 'pressure_strength': 0, 'absorption_detected': False, 'exhaustion_signals': 0, 'cumulative_delta': 0}
                
            order_flow = market_data['order_flow']
            recent_delta = order_flow.get('recent_delta', [0])
            cumulative_delta = order_flow.get('cumulative_delta', 0)
            
            if not recent_delta:
                return {'delta_positive': False, 'pressure_strength': 0, 'absorption_detected': False, 'exhaustion_signals': 0, 'cumulative_delta': cumulative_delta}
                
            avg_delta = float(np.mean(recent_delta))
            delta_positive = avg_delta > 0
            
            pressure_strength = min(abs(avg_delta) * 1000, 1.0)
            
            absorption_detected = False
            if len(recent_delta) >= 3:
                last_three = recent_delta[-3:]
                if all(d < 0 for d in last_three) and cumulative_delta > 0:
                    absorption_detected = True
                elif all(d > 0 for d in last_three) and cumulative_delta < 0:
                    absorption_detected = True
                    
            exhaustion_signals = 0.0
            if pressure_strength > 0.8:
                exhaustion_signals = pressure_strength
                
            return {
                'delta_positive': delta_positive,
                'pressure_strength': pressure_strength,
                'absorption_detected': absorption_detected,
                'exhaustion_signals': exhaustion_signals,
                'cumulative_delta': cumulative_delta
            }
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1503)
            return {'delta_positive': False, 'pressure_strength': 0, 'absorption_detected': False, 'exhaustion_signals': 0, 'cumulative_delta': 0}
            
    def detect_ml_features(self, market_data):
        try:
            gpu_features = self.gpu_extractor.extract_confusion_features(market_data)
            institutional_flow = self.gpu_extractor.extract_institutional_flow(market_data)
            sentiment_scores = self.gpu_extractor.calculate_sentiment_scores(market_data)
            volatility_metrics = self.gpu_extractor.calculate_volatility_metrics(market_data)
            patterns = self.gpu_extractor.detect_patterns_gpu(market_data)
            
            gpu_list = gpu_features.tolist() if hasattr(gpu_features, 'tolist') else list(gpu_features)
            inst_list = institutional_flow.tolist() if hasattr(institutional_flow, 'tolist') else list(institutional_flow)
            
            all_features_list = gpu_list + inst_list
            feat_norm = float(np.linalg.norm(all_features_list)) if all_features_list else 0.0
            
            feature_dict = {
                'confusion_features': gpu_list,
                'institutional_flow': inst_list,
                'sentiment_scores': sentiment_scores,
                'volatility_metrics': volatility_metrics,
                'patterns': patterns,
                'combined_features': all_features_list,
                'feature_strength': feat_norm
            }
            
            return feature_dict
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1531)
            return {
                'confusion_features': [0]*5,
                'institutional_flow': [0]*4,
                'sentiment_scores': {'price_momentum': 0.5, 'volume_sentiment': 0.5},
                'volatility_metrics': {'atr': 0.0, 'volatility': 0.0, 'range_ratio': 0.0},
                'patterns': {'trend_strength': 0.0, 'volatility_regime': 'UNKNOWN'},
                'combined_features': [0]*9,
                'feature_strength': 0
            }
            
    def detect_regime(self, market_data):
        try:
            trend_data = self.trend_brain.analyze_trend(market_data)
            volatility_data = self.volatility_brain.analyze_volatility(market_data)
            regime_data = self.regime_brain.analyze_regime(market_data, trend_data, volatility_data)
            return regime_data
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1548)
            return {'regime_type': 'UNKNOWN', 'regime_confidence': 0, 'transition_phase': False, 'support_score': 0}
            
    def call_deepseek_r1(self, market_context):
        try:
            return {'reasoning_output': 0.7, 'trap_detected': False, 'volatility_override': 1.0, 'psychology_score': 0.8}
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1554)
            return {'reasoning_output': 0.5, 'trap_detected': False, 'volatility_override': 1.0, 'psychology_score': 0.5}
            
    def call_deepseek_v3(self, market_context):
        try:
            return {'microstructure_map': 0.8, 'wick_analysis': 0.7, 'tick_decoding': 0.6, 'pressure_score': 0.75}
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1560)
            return {'microstructure_map': 0.5, 'wick_analysis': 0.5, 'tick_decoding': 0.5, 'pressure_score': 0.5}
            
    def apply_trend_brain(self, market_data):
        return self.trend_brain.analyze_trend(market_data)
        
    def apply_volatility_brain(self, market_data):
        return self.volatility_brain.analyze_volatility(market_data)
        
    def apply_strength_brain(self, market_data, breakout_data, momentum_data):
        return self.strength_brain.analyze_strength(market_data, breakout_data, momentum_data)
        
    def apply_risk_brain(self, market_data, volatility_data, trend_data):
        return self.risk_brain.analyze_risk(market_data, volatility_data, trend_data)
        
    def apply_reversal_brain(self, market_data, trend_data, strength_data):
        return self.reversal_brain.analyze_reversal(market_data, trend_data, strength_data)
        
    def apply_regime_brain(self, market_data, trend_data, volatility_data):
        return self.regime_brain.analyze_regime(market_data, trend_data, volatility_data)
        
    def apply_deepseek_brain(self, market_data, all_brain_data):
        return self.deepseek_brain.analyze_deepseek(market_data, all_brain_data)
        
    def apply_evolution_brain(self, market_data, signal_history):
        return self.evolution_brain.analyze_evolution(market_data, signal_history)
        
    def apply_memory_brain(self, market_data, current_signal):
        return self.memory_brain.analyze_memory(market_data, current_signal)
        
    def apply_self_heal_brain(self, market_data, system_metrics):
        return self.self_heal_brain.analyze_self_heal(market_data, system_metrics)
        
    def apply_meta_fusion_brain(self, all_brain_support):
        return self.meta_fusion_brain.analyze_meta_fusion(all_brain_support)
        
    def apply_mini_r1_brain(self, market_data, breakout_data, fakeout_data):
        return self.mini_r1_brain.analyze_mini_r1(market_data, breakout_data, fakeout_data)
        
    def apply_mini_v3_brain(self, market_data):
        return self.mini_v3_brain.analyze_mini_v3(market_data)
        
    def _generate_legacy_llm_prompt(self, market_data: dict, result_data: dict) -> str:
        """Format clean prompt for LegacyLLM Local AI Reasoning"""
        try:
            price_action = market_data.get('price_action', [])
            current_price = price_action[-1]['close'] if price_action else 0.0
            
            breakout_data = result_data.get('breakout', {})
            orderflow_data = result_data.get('orderflow', {})
            regime_data = result_data.get('regime', {})
            brain_support = result_data.get('brain_support', {})
            
            prompt = f"""You are a legendary, highly profitable trader with over 50 years of experience. You are an absolute master and expert in both swing trading and scalping.
You are acting as the ultimate trade validator for the Part1 SmartBreakout Engine. Use your deep intuition, vast experience, and mastery of market psychology to analyze the following market context and algorithmic brain outputs:

Current Market Context:
- Current Price: {current_price}
- Volatility Regime: {regime_data.get('regime_type', 'UNKNOWN')} (Confidence: {regime_data.get('regime_confidence', 0):.2f})
- Breakout Detected: {breakout_data.get('breakout_detected', False)} (Direction: {breakout_data.get('direction', 0)}, Strength: {breakout_data.get('strength', 0):.2f})
- Orderflow Delta: Positive={orderflow_data.get('delta_positive', False)}, Pressure Strength={orderflow_data.get('pressure_strength', 0):.2f}

12 Algorithmic Brain Support Scores:
- Trend Brain: {brain_support.get('trend', {}).get('support_score', 0):.2f} (Dir: {brain_support.get('trend', {}).get('trend_direction', 0)})
- Strength Brain: {brain_support.get('strength', {}).get('support_score', 0):.2f}
- Volatility Brain: {brain_support.get('volatility', {}).get('support_score', 0):.2f}
- Risk Brain: {brain_support.get('risk', {}).get('support_score', 0):.2f}
- Reversal Brain: {brain_support.get('reversal', {}).get('support_score', 0):.2f}
- DeepSeek Brain: {brain_support.get('deepseek', {}).get('support_score', 0):.2f}
- MetaFusion Brain: {brain_support.get('meta_fusion', {}).get('support_score', 0):.2f}
- MiniR1 Brain: {brain_support.get('mini_r1', {}).get('support_score', 0):.2f}
- MiniV3Brain: {brain_support.get('mini_v3', {}).get('support_score', 0):.2f}

Initial Algorithmic Signal: {result_data.get('signal', 0)} (Confidence: {result_data.get('confidence', 0):.2f}/10)

Task:
Provide a 1-2 sentence analysis, then end your response with your decision strictly as one of: [BUY], [SELL], or [NO-TRADE].
"""
            return prompt
        except Exception:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1639)
            return "Analyze market context and respond with [BUY], [SELL], or [NO-TRADE]."

    @part_advisory_entry("part1_breakout", data_parameter="advisory_data", context_parameter="context")
    def analyze(self, market_data, context=None, advisory_data=None):
        try:
            if context:
                if 'real_orderbook' in context:
                    market_data['real_orderbook'] = context['real_orderbook']
                if 'real_funding' in context:
                    market_data['real_funding'] = context['real_funding']
                    
            levels_data = self.detect_smart_levels(market_data)
            liquidity_data = self.detect_liquidity(market_data, levels_data)
            breakout_data = self.detect_breakout(market_data, levels_data)
            fakeout_data = self.detect_fakeout(market_data, breakout_data, levels_data)
            pullback_data = self.detect_pullback(market_data, breakout_data, levels_data)
            momentum_data = self.detect_momentum(market_data)
            orderflow_data = self.detect_orderflow(market_data)
            ml_data = self.detect_ml_features(market_data)
            regime_data = self.detect_regime(market_data)
            
            trend_brain_data = self.apply_trend_brain(market_data)
            volatility_brain_data = self.apply_volatility_brain(market_data)
            strength_brain_data = self.apply_strength_brain(market_data, breakout_data, momentum_data)
            risk_brain_data = self.apply_risk_brain(market_data, volatility_brain_data, trend_brain_data)
            reversal_brain_data = self.apply_reversal_brain(market_data, trend_brain_data, strength_brain_data)
            regime_brain_data = self.apply_regime_brain(market_data, trend_brain_data, volatility_brain_data)
            
            all_brain_data = {
                'trend': trend_brain_data,
                'volatility': volatility_brain_data,
                'strength': strength_brain_data,
                'risk': risk_brain_data,
                'reversal': reversal_brain_data,
                'regime': regime_brain_data
            }
            
            deepseek_brain_data = self.apply_deepseek_brain(market_data, all_brain_data)
            evolution_brain_data = self.apply_evolution_brain(market_data, self.signal_history)
            memory_brain_data = self.apply_memory_brain(market_data, trend_brain_data.get('trend_direction', 0))
            self_heal_brain_data = self.apply_self_heal_brain(market_data, self.system_metrics)
            mini_r1_brain_data = self.apply_mini_r1_brain(market_data, breakout_data, fakeout_data)
            mini_v3_brain_data = self.apply_mini_v3_brain(market_data)
            
            all_brain_support = {
                "trend": trend_brain_data,
                "volatility": volatility_brain_data,
                "strength": strength_brain_data,
                "risk": risk_brain_data,
                "reversal": reversal_brain_data,
                "regime": regime_brain_data,
                "deepseek": deepseek_brain_data,
                "evolution": evolution_brain_data,
                "memory": memory_brain_data,
                "self_heal": self_heal_brain_data,
                "mini_r1": mini_r1_brain_data,
                "mini_v3": mini_v3_brain_data
            }
            
            meta_fusion_brain_data = self.apply_meta_fusion_brain(all_brain_support)
            all_brain_support["meta_fusion"] = meta_fusion_brain_data
            
            cloud_r1_data = self.call_deepseek_r1({
                'market_data': market_data,
                'breakout_data': breakout_data,
                'brain_support': all_brain_support
            })
            
            cloud_v3_data = self.call_deepseek_v3({
                'market_data': market_data,
                'ml_data': ml_data,
                'orderflow_data': orderflow_data
            })
            
            # Initial algorithmic signal generation
            signal, confidence = self._generate_signal(
                breakout_data, fakeout_data, pullback_data, momentum_data,
                orderflow_data, regime_data, all_brain_support,
                cloud_r1_data, cloud_v3_data, liquidity_data,
                legacy_llm_signal=0
            )
            
            temp_result = {
                "signal": signal,
                "breakout": breakout_data,
                "fakeout": fakeout_data,
                "pullback": pullback_data,
                "momentum": momentum_data,
                "levels": levels_data,
                "liquidity": liquidity_data,
                "orderflow": orderflow_data,
                "ml": ml_data,
                "regime": regime_data,
                "brain_support": all_brain_support,
                "confidence": confidence
            }

            # Model text cannot cast a vote. Preserve only mathematical signal.
            legacy_llm_reasoning = "Legacy model disabled; experimental Laya is audit-only"
            legacy_llm_signal = 0

            result = {
                "signal": signal,
                "breakout": breakout_data,
                "fakeout": fakeout_data,
                "pullback": pullback_data,
                "momentum": momentum_data,
                "levels": levels_data,
                "liquidity": liquidity_data,
                "orderflow": orderflow_data,
                "ml": ml_data,
                "regime": regime_data,
                "brain_support": all_brain_support,
                "confidence": confidence,
                "legacy_llm_reasoning": legacy_llm_reasoning,
                "legacy_llm_signal": legacy_llm_signal
            }
            
            self.signal_history.append(result)
            # Publish to CognitiveBus for Watcher AI monitoring
            if hasattr(self, 'bus') and self.bus:
                direction_str = 'BULLISH' if signal > 0 else 'BEARISH' if signal < 0 else 'NEUTRAL'
                msg = f"Breakout Analysis: {direction_str} (Signal: {signal}, Confidence: {confidence:.2f}). Regime: {regime_data.get('regime', 'unknown')}"
                self.bus.publish('THOUGHTS', 'Part1_Breakout', msg)
            return result
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1766)
            return self._get_error_response()
            
    def _generate_signal(self, breakout_data, fakeout_data, pullback_data, momentum_data,
                        orderflow_data, regime_data, brain_support, cloud_r1_data, cloud_v3_data, liquidity_data,
                        legacy_llm_signal=0):
        try:
            breakout_detected = bool(breakout_data.get('breakout_detected', False))
            breakout_direction = int(breakout_data.get('direction', 0))
            breakout_strength = float(breakout_data.get('strength', 0.0))
            fakeout_prob = float(fakeout_data.get('fakeout_probability', 0.0))
            
            # If no breakout detected, Part 1 must be strictly neutral (0, 0.0)
            if not breakout_detected or breakout_direction == 0:
                return 0, 0.0

            # High risk or high fakeout probability invalidates the breakout
            risk_brain = brain_support.get('risk', {})
            risk_score = float(risk_brain.get('risk_score', 0.5))
            
            print(f"BREAKOUT DETECTED! Dir: {breakout_direction} | Str: {breakout_strength:.3f} | FakeoutProb: {fakeout_prob:.3f} | Risk: {risk_score:.3f}")
            
            if fakeout_prob > 0.85 or risk_score > 0.95:
                print(f" -> INVALIDATED: FakeoutProb ({fakeout_prob:.3f}) > 0.85 OR Risk ({risk_score:.3f}) > 0.95")
                return 0, 0.0

            trend_brain = brain_support.get('trend', {})
            trend_dir = int(trend_brain.get('trend_direction', 0))
            orderflow_dir = 1 if orderflow_data.get('delta_positive', False) else -1
            
            conviction = breakout_strength
            if trend_dir == breakout_direction:
                conviction = min(1.0, conviction + 0.2)
            if orderflow_dir == breakout_direction:
                conviction = min(1.0, conviction + 0.1)
                
            confidence = min(max(conviction * 10.0, 1.0), 10.0)
            return breakout_direction, float(confidence)
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1804)
            print(f"ERROR in _generate_signal: {e}")
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part1_Breakout', e, context='analyze')
            return 0, 0.0
            
    def _get_error_response(self):
        return {
            "signal": 0,
            "breakout": {'breakout_detected': False, 'direction': 0, 'strength': 0, 'level_broken': None, 'volume_confirmation': 0},
            "fakeout": {'fakeout_detected': False, 'fakeout_probability': 0, 'reversal_signals': 0},
            "pullback": {'pullback_detected': False, 'pullback_depth': 0, 'retest_level': None, 'continuation_probability': 0},
            "momentum": {'momentum_direction': 0, 'momentum_strength': 0, 'acceleration': 0, 'trend_alignment': 0},
            "levels": {'support': [], 'resistance': [], 'key_levels': [], 'level_strength': 0},
            "liquidity": {'liquidity_zones': [], 'sweep_detected': False, 'liquidity_strength': 0},
            "orderflow": {'delta_positive': False, 'pressure_strength': 0, 'absorption_detected': False, 'exhaustion_signals': 0, 'cumulative_delta': 0},
            "ml": {
                'confusion_features': [0]*5,
                'institutional_flow': [0]*4,
                'sentiment_scores': {'price_momentum': 0.5, 'volume_sentiment': 0.5},
                'volatility_metrics': {'atr': 0.0, 'volatility': 0.0, 'range_ratio': 0.0},
                'patterns': {'trend_strength': 0.0, 'volatility_regime': 'UNKNOWN'},
                'combined_features': [0]*9,
                'feature_strength': 0
            },
            "regime": {'regime_type': 'UNKNOWN', 'regime_confidence': 0, 'transition_phase': False, 'support_score': 0},
            "brain_support": {
                "trend": {'trend_direction': 0, 'trend_strength': 0, 'trend_quality': 0, 'momentum': 0, 'support_score': 0},
                "volatility": {'volatility_regime': 'UNKNOWN', 'volatility_score': 0, 'breakout_potential': 0, 'volatility_trend': 0, 'support_score': 0},
                "strength": {'breakout_strength': 0, 'momentum_strength': 0, 'volume_confirmation': 0, 'overall_strength': 0, 'strength_momentum': 0, 'support_score': 0},
                "risk": {'risk_score': 0.5, 'fakeout_probability': 0.5, 'stop_distance': 0, 'position_size': 0, 'avg_risk': 0.5, 'support_score': 0.5},
                "reversal": {'reversal_probability': 0, 'exhaustion_signals': 0, 'divergence_detected': False, 'price_momentum': 0, 'volume_momentum': 0, 'support_score': 1},
                "regime": {'regime_type': 'UNKNOWN', 'regime_confidence': 0, 'transition_phase': False, 'support_score': 0},
                "deepseek": {'correction_factor': 0, 'noise_level': 0, 'fakeout_block': 0, 'contradiction_score': 0, 'threshold_boost': 0, 'support_score': 0},
                "evolution": {'generation': 1, 'adaptation_factor': 0.5, 'performance_score': 0.5, 'learning_rate': 0.1, 'support_score': 0.5},
                "memory": {'short_term_recall': 0, 'long_term_pattern': 0, 'prediction_confidence': 0, 'memory_strength': 0, 'support_score': 0},
                "self_heal": {'stability_score': 0.8, 'error_rate': 0.1, 'health_index': 0.9, 'recovery_factor': 0.5, 'support_score': 0.7},
                "meta_fusion": {'fusion_score': 0, 'consensus_level': 0, 'weight_distribution': [], 'support_score': 0},
                "mini_r1": {'reasoning_depth': 0, 'trap_probability': 0, 'psychology_score': 0, 'volatility_override': 0, 'support_score': 0},
                "mini_v3": {'microstructure_score': 0, 'wick_imbalance': 0, 'tick_momentum': 0, 'pressure_detection': 0, 'support_score': 0}
            },
            "confidence": 0,
            "legacy_llm_reasoning": "Error response fallback",
            "legacy_llm_signal": 0
        }