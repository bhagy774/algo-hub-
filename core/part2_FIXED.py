try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# PyTorch with robust fallback for Windows/Linux/CPU compatibility
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 10)
    TORCH_AVAILABLE = False
    import numpy as np
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

        def __init__(self, data=0, *args, **kwargs):
            if isinstance(data, (list, tuple, np.ndarray)):
                self.arr = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float)):
                self.arr = np.array([data], dtype=np.float32)
            elif isinstance(data, DummyTensor):
                self.arr = data.arr.copy()
            else:
                self.arr = np.array([0.0], dtype=np.float32)
            self.shape = self.arr.shape
            self.dtype = 'float32'
            self.device = 'cpu'

        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def numpy(self): return self.arr
        def item(self): return float(self.arr.flat[0]) if self.arr.size > 0 else 0.0
        def __getitem__(self, key):
            res = self.arr[key]
            return DummyTensor(res) if isinstance(res, np.ndarray) else res
        def __len__(self): return len(self.arr)
        def dim(self): return self.arr.ndim
        def size(self, dim=None): return self.arr.shape[dim] if dim is not None else self.arr.shape
        def unsqueeze(self, dim): return DummyTensor(np.expand_axis(self.arr, dim) if hasattr(np, 'expand_axis') else np.expand_dims(self.arr, dim))
        def expand(self, *sizes): return self
        def mean(self, *args, **kwargs): return float(np.mean(self.arr)) if self.arr.size > 0 else 0.0
        def std(self, *args, **kwargs): return float(np.std(self.arr)) if self.arr.size > 1 else 0.0
        def sum(self, *args, **kwargs): return float(np.sum(self.arr))
        def max(self, *args, **kwargs): return float(np.max(self.arr)) if self.arr.size > 0 else 0.0
        def min(self, *args, **kwargs): return float(np.min(self.arr)) if self.arr.size > 0 else 0.0
        def abs(self): return DummyTensor(np.abs(self.arr))
        def __add__(self, other): return DummyTensor(self.arr + (other.arr if isinstance(other, DummyTensor) else other))
        def __radd__(self, other): return self.__add__(other)
        def __sub__(self, other): return DummyTensor(self.arr - (other.arr if isinstance(other, DummyTensor) else other))
        def __rsub__(self, other): return DummyTensor((other.arr if isinstance(other, DummyTensor) else other) - self.arr)
        def __mul__(self, other): return DummyTensor(self.arr * (other.arr if isinstance(other, DummyTensor) else other))
        def __rmul__(self, other): return self.__mul__(other)
        def __truediv__(self, other):
            denom = other.arr if isinstance(other, DummyTensor) else other
            return DummyTensor(self.arr / (denom + 1e-8))
        def __rtruediv__(self, other):
            num = other.arr if isinstance(other, DummyTensor) else other
            return DummyTensor(num / (self.arr + 1e-8))
        def __gt__(self, other): return self.arr > (other.arr if isinstance(other, DummyTensor) else other)
        def __lt__(self, other): return self.arr < (other.arr if isinstance(other, DummyTensor) else other)
        def __ge__(self, other): return self.arr >= (other.arr if isinstance(other, DummyTensor) else other)
        def __le__(self, other): return self.arr <= (other.arr if isinstance(other, DummyTensor) else other)

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
        FloatTensor = DummyTensor
        device = lambda x: 'cpu'
        float32 = 'float32'
        long = 'long'

        class nn:
            Module = DummyModule
            Linear = DummyModule
            LSTM = DummyModule
            GRU = DummyModule
            Transformer = DummyModule
            TransformerEncoderLayer = DummyModule
            TransformerEncoder = DummyModule
            Dropout = DummyModule
            BatchNorm1d = DummyModule
            LayerNorm = DummyModule
            Embedding = DummyModule
            Conv1d = DummyModule
            MaxPool1d = DummyModule
            AdaptiveAvgPool1d = DummyModule
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
            def get_device_name(idx=0): return 'CPU'

        class no_grad:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        @staticmethod
        def tensor(data, **kwargs): return DummyTensor(data)
        @staticmethod
        def zeros(*shape, **kwargs):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
            return DummyTensor(np.zeros(shape, dtype=np.float32)) if shape else DummyTensor(0)
        @staticmethod
        def randn(*shape, **kwargs):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
            return DummyTensor(np.random.randn(*shape)) if shape else DummyTensor(0)
        @staticmethod
        def cat(tensors, dim=0): return DummyTensor()
        @staticmethod
        def abs(input): return input.abs() if hasattr(input, 'abs') else DummyTensor(np.abs(input))
        @staticmethod
        def arange(*args, **kwargs): return DummyTensor(np.arange(*args))
        @staticmethod
        def max(input, *args, **kwargs): return DummyTensor(np.max(input.arr)) if hasattr(input, 'arr') and input.arr.size > 0 else DummyTensor(0)
        @staticmethod
        def min(input, *args, **kwargs): return DummyTensor(np.min(input.arr)) if hasattr(input, 'arr') and input.arr.size > 0 else DummyTensor(0)
        @staticmethod
        def all(*args, **kwargs): return DummyTensor([True])
        @staticmethod
        def sum(input, *args, **kwargs): return DummyTensor(np.sum(input.arr)) if hasattr(input, 'arr') else DummyTensor(0)
        @staticmethod
        def stack(tensors, dim=0): return DummyTensor()
        @staticmethod
        def is_tensor(x): return isinstance(x, DummyTensor)
        @staticmethod
        def mean(x, *args, **kwargs): return np.mean(x.arr) if isinstance(x, DummyTensor) else np.mean(x)
        @staticmethod
        def std(x, *args, **kwargs): return np.std(x.arr) if isinstance(x, DummyTensor) else np.std(x)
        @staticmethod
        def diff(x, *args, **kwargs): return DummyTensor(np.diff(x.arr)) if isinstance(x, DummyTensor) else DummyTensor(np.diff(x))

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

    nn = torch.nn
    F = torch.F
    optim = torch.optim

# CuPy fallback — if CUDA/CuPy not installed, fall back to NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 182)
    import numpy as cp  # type: ignore
    CUPY_AVAILABLE = False
import numpy as np
from collections import defaultdict, deque
import time
import pickle
import hashlib
import requests
import os
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings


warnings.filterwarnings('ignore')




# [FIX] Removed broken import from non-existent 'deepseek_missing_brains' module.
# All brain classes are defined in this file below.
# ZonePointFiveDetectorGPU and CandlePsychologyMasterGPU come from Part 1 (SmartBreakoutAI).
# Stub classes provided here so Part 2 can run standalone without crashing.

class ZonePointFiveDetectorGPU:
    """Stub: Implemented in Part 1 (SmartBreakoutAI). Detects 0.5 zone signals."""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def detect_0_5_zone_signals(self, current_candle, psychology, df_1min, df_5min, df_15min):
        return []

class CandlePsychologyMasterGPU:
    """Stub: Implemented in Part 1 (SmartBreakoutAI). Analyzes candle psychology."""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def analyze_candle_psychology(self, candle):
        high = candle.get('high', 0)
        low = candle.get('low', 0)
        open_p = candle.get('open', 0)
        close = candle.get('close', 0)
        total_range = high - low if high != low else 1e-8
        body = abs(close - open_p)
        upper_wick = high - max(close, open_p)
        lower_wick = min(close, open_p) - low
        return {
            'is_bullish': close > open_p,
            'is_bearish': close < open_p,
            'body_ratio': body / total_range,
            'upper_wick_ratio': upper_wick / total_range,
            'lower_wick_ratio': lower_wick / total_range,
            'has_strong_rejection': max(upper_wick, lower_wick) > total_range * 0.5,
            'has_strong_momentum': body > total_range * 0.6,
        }

# ==================== LINUX OPTIMIZATION UTILS ====================

def _safe_get_device_name(device):
    """Safely get GPU device name — defined early so all classes can use it"""
    try:
        if hasattr(device, 'type') and device.type == 'cuda':
            try:
                idx = device.index if hasattr(device, 'index') and device.index is not None else 0
                return torch.cuda.get_device_name(idx)
            except:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 251)
                try:
                    return torch.cuda.get_device_name()
                except:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 254)
                    return "cuda_device"
        return "CPU"
    except:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 257)
        return "UNKNOWN_DEVICE"

class LinuxOptimizedDeque(deque):
    """Linux-optimized deque for high-performance memory management"""
    def __init__(self, maxlen=None):
        super().__init__(maxlen=maxlen)
        
    def push_tensor(self, tensor):
        """Push tensor to deque with memory check"""
        if torch.is_tensor(tensor) and tensor.is_cuda:
            tensor = tensor.cpu()
        self.append(tensor)

# ==================== ADVANCED AI/ML MODELS ====================

class LSTMPredictor(nn.Module):
    """LSTM Neural Network for Time Series Prediction"""
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class TransformerPredictor(nn.Module):
    """Transformer Model for Market Pattern Recognition"""
    def __init__(self, d_model=64, nhead=8, num_layers=4, dim_feedforward=256, output_size=3):
        super(TransformerPredictor, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)
        self.d_model = d_model
        
    def forward(self, x):
        x = x * np.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc_out(x)

class CNNFeatureExtractor(nn.Module):
    """CNN for Pattern Feature Extraction"""
    def __init__(self, input_channels=5, output_size=32):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ==================== NEURAL NETWORK MANAGER ====================

class NeuralNetworkManager:
    """Manages AI model training, retraining, and inference"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models = {
            'lstm_predictor': LSTMPredictor().to(self.device),
            'transformer_predictor': TransformerPredictor().to(self.device),
            'cnn_extractor': CNNFeatureExtractor().to(self.device)
        }
        
        self.optimizers = {
            name: optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            for name, model in self.models.items()
        }
        
        self.criterion = nn.MSELoss()
        
        # Training data storage
        self.training_data = {
            'price_sequences': LinuxOptimizedDeque(10000),
            'volume_sequences': LinuxOptimizedDeque(10000),
            'targets': LinuxOptimizedDeque(10000)
        }
        
        # Traditional ML models
        self.ml_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # Auto-retraining configuration
        self.retraining_config = {
            'retraining_interval': 1000,  # candles
            'minimum_samples': 500,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'performance_threshold': 0.6
        }
        
        self.training_stats = defaultdict(lambda: LinuxOptimizedDeque(100))
        
        print(f"OK Neural Network Manager Initialized (GPU: {_safe_get_device_name(self.device)})")

    def add_training_data(self, price_sequence, volume_sequence, target):
        """Add new training data with automatic retraining triggers"""
        self.training_data['price_sequences'].append(price_sequence)
        self.training_data['volume_sequences'].append(volume_sequence)
        self.training_data['targets'].append(target)
        
        # Trigger retraining if enough data accumulated
        if len(self.training_data['price_sequences']) % self.retraining_config['retraining_interval'] == 0:
            if len(self.training_data['price_sequences']) >= self.retraining_config['minimum_samples']:
                self.retrain_all_models()

    def retrain_all_models(self):
        """Retrain all AI/ML models with accumulated data"""
        try:
            if len(self.training_data['price_sequences']) < self.retraining_config['minimum_samples']:
                return
            
            print("STARTING AI MODEL RETRAINING...")
            
            # Prepare data
            price_data = np.array([seq for seq in self.training_data['price_sequences']])
            volume_data = np.array([seq for seq in self.training_data['volume_sequences']])
            targets = np.array([t for t in self.training_data['targets']])
            
            # Train traditional ML models
            self._train_ml_models(price_data, volume_data, targets)
            
            # Train neural networks
            self._train_neural_networks(price_data, volume_data, targets)
            
            print("AI MODEL RETRAINING COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 415)
            print(f"ERROR AI model retraining failed: {e}")

    def _train_ml_models(self, price_data, volume_data, targets):
        """Train traditional machine learning models"""
        try:
            # Feature engineering
            features = self._create_ml_features(price_data, volume_data)
            
            if not self.is_scaler_fitted:
                features_scaled = self.scaler.fit_transform(features)
                self.is_scaler_fitted = True
            else:
                features_scaled = self.scaler.transform(features)
            
            # Train each ML model
            for name, model in self.ml_models.items():
                model.fit(features_scaled, targets)
                print(f"OK {name} training completed")
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 435)
            print(f"ERROR ML model training failed: {e}")

    def _train_neural_networks(self, price_data, volume_data, targets):
        """Train neural network models"""
        try:
            # Convert to PyTorch tensors
            price_tensor = torch.FloatTensor(price_data).to(self.device)
            # FIX: Expand targets from [N] to [N, 3] to match model output_size=3
            target_tensor = torch.FloatTensor(targets).to(self.device)
            if target_tensor.dim() == 1:
                target_tensor = target_tensor.unsqueeze(1).expand(-1, 3)
            
            # Train LSTM
            self._train_single_model('lstm_predictor', price_tensor, target_tensor)
            
            # Train Transformer
            self._train_single_model('transformer_predictor', price_tensor, target_tensor)
            
            # Train CNN
            cnn_input = price_tensor.unsqueeze(1)  # FIX: [N, seq_len] -> [N, 1, seq_len] for CNN
            self._train_single_model('cnn_extractor', cnn_input, target_tensor)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 458)
            print(f"ERROR Neural network training failed: {e}")

    def _train_single_model(self, model_name, inputs, targets):
        """Train a single neural network model"""
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        self.training_stats[model_name].append({
            'timestamp': time.time(),
            'loss': loss.item(),
            'samples_used': len(inputs)
        })

    def _create_ml_features(self, price_data, volume_data):
        """Create features for ML models"""
        features = []
        
        for i in range(len(price_data)):
            price_seq = price_data[i]
            volume_seq = volume_data[i]
            
            # Technical features
            returns = np.diff(price_seq) / price_seq[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            momentum = price_seq[-1] - price_seq[0] if len(price_seq) > 0 else 0
            
            # Volume features
            volume_mean = np.mean(volume_seq) if len(volume_seq) > 0 else 0
            volume_std = np.std(volume_seq) if len(volume_seq) > 0 else 0
            
            # Statistical features
            price_mean = np.mean(price_seq)
            price_std = np.std(price_seq)
            price_skew = self._calculate_skewness(price_seq)
            price_kurtosis = self._calculate_kurtosis(price_seq)
            
            feature_vector = [
                volatility, momentum, volume_mean, volume_std,
                price_mean, price_std, price_skew, price_kurtosis
            ]
            
            features.append(feature_vector)
        
        return np.array(features)

    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean((data - mean) ** 3) / (std ** 3)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean((data - mean) ** 4) / (std ** 4) - 3

    def predict(self, price_sequence, volume_sequence):
        """Retired compatibility hook: no unvalidated model outputs in trading."""
        return {}
        try:
            predictions = {}
            
            # ML model predictions
            features = self._create_ml_features(
                np.array([price_sequence]), 
                np.array([volume_sequence])
            )
            
            if self.is_scaler_fitted:
                features_scaled = self.scaler.transform(features)
                
                for name, model in self.ml_models.items():
                    try:
                        pred = model.predict(features_scaled)[0]
                        predictions[name] = float(pred)
                    except:
                        globals().get("_plog", lambda *_a, **_k: None)(__name__, 552)
                        predictions[name] = 0.0
            
            # Neural network predictions
            price_tensor = torch.FloatTensor([price_sequence]).to(self.device)
            
            for name, model in self.models.items():
                try:
                    model.eval()
                    with torch.no_grad():
                        if name == 'cnn_extractor':
                            input_tensor = price_tensor.unsqueeze(1)  # FIX: [1, seq_len] -> [1, 1, seq_len]
                        else:
                            input_tensor = price_tensor
                        
                        pred = model(input_tensor).cpu().numpy()[0]
                        predictions[name] = float(pred.mean())
                except:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 569)
                    predictions[name] = 0.0
            if hasattr(self, 'bus') and self.bus:
                avg_pred = sum(predictions.values()) / max(1, len(predictions))
                direction = "BULLISH" if avg_pred > 0.05 else "BEARISH" if avg_pred < -0.05 else "NEUTRAL"
                msg = f"Neural Network Predictions: {direction} (Avg score: {avg_pred:.3f})"
                self.bus.publish('THOUGHTS', 'Part2_Neural', msg)
                
            return predictions
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 579)
            if hasattr(self, 'bus') and self.bus:
                self.bus.report_error('Part2_Neural', e, context='predict - neural/ml inference')
            else:
                print(f"ERROR AI prediction failed: {e}")
            return {'default': 0.0}

# ==================== AUTO UPDATE SYSTEM ====================

class AutoUpdateSystem:
    """
    FIX #15: DISABLED — Remote Code Execution Risk
    This class previously downloaded code from GitHub (wrong repo) and overwrote local files.
    Kept as stub to avoid breaking any references.
    """
    def __init__(self, master_system):
        self.master = master_system
        self.enabled = False  # DISABLED for security
        print("[AutoUpdateSystem] [WARNING] Disabled for security (remote code execution risk)")

    def check_for_updates(self):
        return False  # Always disabled

    def _download_update(self, repo_data):
        return False  # Always disabled


# ==================== SPECIALIZED ANALYSIS BRAINS ====================

# 1. ZonePointFiveDetectorGPU (Already implemented in Part 1)
# 2. CandlePsychologyMasterGPU (Already implemented in Part 1)

# 3. VolumeProfileBrainGPU
from brain_xgb import xgb_shadow

@xgb_shadow('volume_profile_brain', 'analyze_volume_profile')
class VolumeProfileBrainGPU:
    """Advanced Volume Analysis and Profiling"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.volume_config = {
            'profile_periods': [50, 100, 200],
            'volume_cluster_size': 0.001,
            'high_volume_threshold': 2.0,
            'volume_acceleration_periods': [5, 10, 20],
            'gpu_batch_size': 512
        }
        
        self.volume_profiles = {
            'short_term': LinuxOptimizedDeque(100),
            'medium_term': LinuxOptimizedDeque(200),
            'long_term': LinuxOptimizedDeque(500)
        }
        
        self.volume_zones = {
            'high_volume_nodes': LinuxOptimizedDeque(50),
            'low_volume_gaps': LinuxOptimizedDeque(50),
            'volume_clusters': LinuxOptimizedDeque(100)
        }
        
        self.gpu_buffers = {}
        self._init_volume_gpu_buffers()
        
        print(f"OK Volume Profile Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_volume_gpu_buffers(self):
        """Initialize GPU buffers for volume calculations"""
        self.gpu_buffers['volume_analysis'] = torch.zeros(
            self.volume_config['gpu_batch_size'], device=self.device
        )
        self.gpu_buffers['price_volume'] = torch.zeros(
            self.volume_config['gpu_batch_size'], device=self.device
        )

    def analyze_volume_profile(self, df, timeframe):
        """GPU-accelerated volume profile analysis"""
        try:
            if len(df) < 20:
                return {}
            
            # Move data to GPU
            with torch.no_grad():
                closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
                volumes = torch.tensor(df['volume'].values, device=self.device, dtype=torch.float32)
                highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
                lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
                
                # Calculate volume profile
                price_range = torch.max(highs) - torch.min(lows)
                bin_size = price_range * self.volume_config['volume_cluster_size']
                
                if bin_size <= 0:
                    return {}
                
                # GPU-accelerated volume clustering
                num_bins = int((price_range / bin_size).item()) + 1
                volume_profile = torch.zeros(num_bins, device=self.device)
                
                for i in range(len(closes)):
                    price = closes[i]
                    volume = volumes[i]
                    
                    bin_index = int(((price - torch.min(lows)) / bin_size).item())
                    bin_index = max(0, min(num_bins-1, bin_index))
                    volume_profile[bin_index] += volume
                
                # Find high volume nodes
                avg_volume = torch.mean(volume_profile)
                high_volume_threshold = avg_volume * self.volume_config['high_volume_threshold']
                high_volume_bins = volume_profile > high_volume_threshold
                
                high_volume_nodes = []
                for i in range(num_bins):
                    if high_volume_bins[i]:
                        price_level = torch.min(lows) + (i * bin_size) + (bin_size / 2)
                        high_volume_nodes.append({
                            'price': price_level.item(),
                            'volume': volume_profile[i].item(),
                            'strength': (volume_profile[i] / avg_volume).item(),
                            'timeframe': timeframe
                        })
                
                # Update storage
                self._update_volume_zones(high_volume_nodes, timeframe)
                
                return {
                    'volume_profile': volume_profile.cpu().numpy(),
                    'high_volume_nodes': high_volume_nodes,
                    'price_bins': [torch.min(lows).item() + i * bin_size.item() for i in range(num_bins)],
                    'total_volume': torch.sum(volumes).item(),
                    'avg_volume': torch.mean(volumes).item()
                }
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 711)
            print(f"ERROR Volume profile analysis failed: {e}")
            return {}

    def _update_volume_zones(self, new_nodes, timeframe):
        """Update volume zone storage"""
        current_nodes = list(self.volume_zones['high_volume_nodes'])
        
        for new_node in new_nodes:
            similar_node = None
            for existing_node in current_nodes:
                tolerance = existing_node['price'] * 0.001
                if abs(existing_node['price'] - new_node['price']) <= tolerance:
                    similar_node = existing_node
                    break
            
            if similar_node:
                # Update existing node
                similar_node['volume'] = max(similar_node['volume'], new_node['volume'])
                similar_node['strength'] = max(similar_node['strength'], new_node['strength'])
            else:
                current_nodes.append(new_node)
        
        # Keep only strongest nodes
        current_nodes.sort(key=lambda x: x['strength'], reverse=True)
        self.volume_zones['high_volume_nodes'] = LinuxOptimizedDeque(maxlen=50)
        self.volume_zones['high_volume_nodes'].extend(current_nodes[:50])

    def detect_volume_signals(self, current_candle, psychology, df):
        """Detect trading signals based on volume analysis"""
        try:
            signals = []
            current_volume = current_candle.get('volume', 0)
            current_price = current_candle['close']
            
            # Volume spike detection
            volume_spike = self.detect_volume_spike(df)
            
            # High volume node rejection & momentum
            for node in list(self.volume_zones['high_volume_nodes'])[:10]:
                tolerance = node['price'] * 0.005
                
                if abs(current_price - node['price']) <= tolerance:
                    if psychology.get('is_bullish', False) or psychology.get('lower_wick_ratio', 0) > 0.3:
                        confidence = 7.0 + (node.get('strength', 0.5) * 1.5)
                        signals.append(("CALL", min(confidence, 9.5), 
                                     f"BULLISH REJECTION AT HIGH VOLUME NODE | Strength: {node.get('strength', 0.5):.2f}"))
                    elif psychology.get('is_bearish', False) or psychology.get('upper_wick_ratio', 0) > 0.3:
                        confidence = 7.0 + (node.get('strength', 0.5) * 1.5)
                        signals.append(("PUT", min(confidence, 9.5),
                                     f"BEARISH REJECTION AT HIGH VOLUME NODE | Strength: {node.get('strength', 0.5):.2f}"))
            
            # General volume momentum signal if volume spike present
            if volume_spike and not signals:
                c_dir = "CALL" if current_candle['close'] >= current_candle['open'] else "PUT"
                signals.append((c_dir, 7.5, f"VOLUME SPIKE MOMENTUM | Direction: {c_dir}"))
                
            return signals
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 768)
            print(f"ERROR Volume signal detection failed: {e}")
            return []

    def detect_volume_spike(self, df, lookback=20):
        """Detect volume spikes compared to recent average"""
        try:
            if len(df) < lookback + 1:
                return False
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(lookback).mean()
            
            return current_volume > (avg_volume * 1.25)
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 783)
            return False

# 4. MarketStructureBrainGPU
@xgb_shadow('market_structure_brain', 'analyze_market_structure')
class MarketStructureBrainGPU:
    """Market Structure Analysis and Identification"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.structure_config = {
            'swing_periods': [5, 10, 20],
            'trend_confirmation_periods': [50, 100],
            'structure_break_threshold': 0.002,
            'key_level_confidence': 0.7,
            'gpu_batch_size': 512
        }
        
        self.market_structure = {
            'higher_highs': LinuxOptimizedDeque(100),
            'higher_lows': LinuxOptimizedDeque(100),
            'lower_highs': LinuxOptimizedDeque(100),
            'lower_lows': LinuxOptimizedDeque(100),
            'key_support_levels': LinuxOptimizedDeque(50),
            'key_resistance_levels': LinuxOptimizedDeque(50)
        }
        
        self.current_trend = {
            'primary_trend': 'SIDEWAYS',
            'secondary_trend': 'SIDEWAYS',
            'trend_strength': 0.5,
            'last_swing_high': None,
            'last_swing_low': None
        }
        
        print(f"OK Market Structure Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def analyze_market_structure(self, df, timeframe):
        """GPU-accelerated market structure analysis"""
        try:
            if len(df) < 50:
                return self.current_trend
            
            # Move data to GPU
            with torch.no_grad():
                highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
                lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
                closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
                
                # Find swing points
                swing_highs, swing_lows = self._find_swing_points(highs, lows)
                
                # Analyze market structure
                structure_analysis = self._analyze_structure_changes(swing_highs, swing_lows)
                
                # Update key levels
                self._update_key_levels(swing_highs, swing_lows, timeframe)
                
                # Determine trend
                self._determine_trend(structure_analysis, closes)
                
                return self.current_trend
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 846)
            print(f"ERROR Market structure analysis failed: {e}")
            return self.current_trend

    def _find_swing_points(self, highs, lows):
        """Find swing highs and lows using GPU acceleration"""
        swing_highs = []
        swing_lows = []
        
        for period in self.structure_config['swing_periods']:
            if len(highs) < period * 2:
                continue
                
            # GPU-accelerated rolling window operations
            for i in range(period, len(highs) - period):
                window_highs = highs[i-period:i+period+1]
                window_lows = lows[i-period:i+period+1]
                
                center_high = highs[i]
                center_low = lows[i]
                
                # Check for swing high
                if torch.all(center_high >= window_highs):
                    swing_highs.append({
                        'price': center_high.item(),
                        'index': i,
                        'period': period
                    })
                
                # Check for swing low
                if torch.all(center_low <= window_lows):
                    swing_lows.append({
                        'price': center_low.item(),
                        'index': i,
                        'period': period
                    })
        
        return swing_highs, swing_lows

    def _analyze_structure_changes(self, swing_highs, swing_lows):
        """Analyze changes in market structure"""
        analysis = {
            'hh': len(self.market_structure['higher_highs']) > 0,
            'hl': len(self.market_structure['higher_lows']) > 0,
            'lh': len(self.market_structure['lower_highs']) > 0,
            'll': len(self.market_structure['lower_lows']) > 0,
            'structure_breaks': 0
        }
        
        # Analyze recent swings for structure changes
        recent_swing_highs = sorted(swing_highs, key=lambda x: x['index'])[-5:]
        recent_swing_lows = sorted(swing_lows, key=lambda x: x['index'])[-5:]
        
        if len(recent_swing_highs) >= 2:
            last_high = recent_swing_highs[-1]['price']
            prev_high = recent_swing_highs[-2]['price']
            
            if last_high > prev_high:
                analysis['hh'] = True
                self.market_structure['higher_highs'].append({
                    'price': last_high,
                    'timestamp': time.time()
                })
            else:
                analysis['lh'] = True
                self.market_structure['lower_highs'].append({
                    'price': last_high,
                    'timestamp': time.time()
                })
        
        if len(recent_swing_lows) >= 2:
            last_low = recent_swing_lows[-1]['price']
            prev_low = recent_swing_lows[-2]['price']
            
            if last_low > prev_low:
                analysis['hl'] = True
                self.market_structure['higher_lows'].append({
                    'price': last_low,
                    'timestamp': time.time()
                })
            else:
                analysis['ll'] = True
                self.market_structure['lower_lows'].append({
                    'price': last_low,
                    'timestamp': time.time()
                })
        
        return analysis

    def _update_key_levels(self, swing_highs, swing_lows, timeframe):
        """Update key support and resistance levels"""
        # Process swing highs as resistance
        for swing_high in swing_highs[-10:]:
            level = swing_high['price']
            self._add_key_level(level, 'resistance', timeframe)
        
        # Process swing lows as support
        for swing_low in swing_lows[-10:]:
            level = swing_low['price']
            self._add_key_level(level, 'support', timeframe)

    def _add_key_level(self, level, level_type, timeframe):
        """Add key level with confidence scoring"""
        levels_storage = (self.market_structure['key_resistance_levels'] 
                         if level_type == 'resistance' 
                         else self.market_structure['key_support_levels'])
        
        current_levels = list(levels_storage)
        tolerance = level * 0.001
        
        similar_level = None
        for existing_level in current_levels:
            if abs(existing_level['price'] - level) <= tolerance:
                similar_level = existing_level
                break
        
        if similar_level:
            similar_level['confidence'] = min(similar_level['confidence'] + 0.1, 1.0)
            similar_level['timeframes'].add(timeframe)
            similar_level['last_touch'] = time.time()
        else:
            current_levels.append({
                'price': level,
                'type': level_type,
                'confidence': 0.5,
                'timeframes': {timeframe},
                'last_touch': time.time(),
                'first_seen': time.time()
            })
        
        # Sort by confidence and keep top levels
        current_levels.sort(key=lambda x: x['confidence'], reverse=True)
        levels_storage.clear()
        levels_storage.extend(current_levels[:20])

    def _determine_trend(self, structure_analysis, closes):
        """Determine current market trend"""
        hh = structure_analysis['hh']
        hl = structure_analysis['hl']
        lh = structure_analysis['lh']
        ll = structure_analysis['ll']
        
        # Trend determination logic
        if hh and hl:
            trend = "BULLISH"
            strength = 0.8
        elif lh and ll:
            trend = "BEARISH"
            strength = 0.8
        elif hh and not ll:
            trend = "BULLISH"
            strength = 0.6
        elif ll and not hh:
            trend = "BEARISH"
            strength = 0.6
        else:
            trend = "SIDEWAYS"
            strength = 0.5
        
        # Additional confirmation using price position
        if len(closes) >= 50:
            short_ma = torch.mean(closes[-20:])
            long_ma = torch.mean(closes[-50:])
            
            if short_ma > long_ma and trend == "BULLISH":
                strength = min(strength + 0.2, 1.0)
            elif short_ma < long_ma and trend == "BEARISH":
                strength = min(strength + 0.2, 1.0)
            elif (short_ma > long_ma and trend == "BEARISH") or (short_ma < long_ma and trend == "BULLISH"):
                strength = max(strength - 0.3, 0.1)
        
        self.current_trend.update({
            'primary_trend': trend,
            'trend_strength': strength,
            'last_update': time.time()
        })

    def get_structure_signals(self, current_price, psychology):
        """Get trading signals based on market structure"""
        signals = []
        
        # Trend-following signals
        if self.current_trend['primary_trend'] == "BULLISH" and self.current_trend['trend_strength'] > 0.7:
            if psychology['is_bullish'] and psychology['body_ratio'] > 0.6:
                signals.append(("CALL", 7.0 + self.current_trend['trend_strength'] * 2.0,
                              f"TREND FOLLOWING BULLISH | Strength: {self.current_trend['trend_strength']:.2f}"))
        
        elif self.current_trend['primary_trend'] == "BEARISH" and self.current_trend['trend_strength'] > 0.7:
            if psychology['is_bearish'] and psychology['body_ratio'] > 0.6:
                signals.append(("PUT", 7.0 + self.current_trend['trend_strength'] * 2.0,
                              f"TREND FOLLOWING BEARISH | Strength: {self.current_trend['trend_strength']:.2f}"))
        
        # Key level signals
        for support in list(self.market_structure['key_support_levels'])[:5]:
            if abs(current_price - support['price']) / support['price'] < 0.001:
                if psychology['has_strong_rejection'] and psychology['lower_wick_ratio'] > 0.3:
                    confidence = 7.5 + (support['confidence'] * 2.0)
                    signals.append(("CALL", min(confidence, 9.5),
                                  f"SUPPORT BOUNCE | Confidence: {support['confidence']:.2f}"))
        
        for resistance in list(self.market_structure['key_resistance_levels'])[:5]:
            if abs(current_price - resistance['price']) / resistance['price'] < 0.001:
                if psychology['has_strong_rejection'] and psychology['upper_wick_ratio'] > 0.3:
                    confidence = 7.5 + (resistance['confidence'] * 2.0)
                    signals.append(("PUT", min(confidence, 9.5),
                                  f"RESISTANCE REJECTION | Confidence: {resistance['confidence']:.2f}"))
        
        return signals

# 5. OrderFlowBrainGPU
@xgb_shadow('order_flow_brain', 'analyze_order_flow')
class OrderFlowBrainGPU:
    """Order Flow and Liquidity Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.orderflow_config = {
            'imbalance_threshold': 1.5,
            'absorption_periods': [5, 10],
            'liquidity_zones_depth': 0.005,
            'gpu_batch_size': 256
        }
        
        self.order_flow_data = {
            'buy_imbalances': LinuxOptimizedDeque(200),
            'sell_imbalances': LinuxOptimizedDeque(200),
            'absorption_zones': LinuxOptimizedDeque(100),
            'liquidity_pools': LinuxOptimizedDeque(100)
        }
        
        self.gpu_buffers = {}
        self._init_orderflow_gpu_buffers()
        
        print(f"OK Order Flow Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_orderflow_gpu_buffers(self):
        """Initialize GPU buffers for order flow calculations"""
        self.gpu_buffers['order_imbalance'] = torch.zeros(
            self.orderflow_config['gpu_batch_size'], device=self.device
        )

    def analyze_order_flow(self, df, bid_ask_data=None):
        """Analyze order flow and liquidity conditions"""
        try:
            if len(df) < 20:
                return {}
            
            analysis = {}
            
            # Calculate order imbalances
            analysis['imbalances'] = self._calculate_order_imbalances(df)
            
            # Detect absorption
            analysis['absorption'] = self._detect_absorption(df)
            
            # Identify liquidity zones
            analysis['liquidity_zones'] = self._identify_liquidity_zones(df)
            
            # Update storage
            self._update_order_flow_storage(analysis)
            
            return analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1109)
            print(f"ERROR Order flow analysis failed: {e}")
            return {}

    def _calculate_order_imbalances(self, df):
        """Calculate buy/sell order imbalances"""
        imbalances = []
        
        if len(df) < 10:
            return imbalances
        
        # Use price and volume to infer order flow
        closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
        volumes = torch.tensor(df['volume'].values, device=self.device, dtype=torch.float32)
        highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
        lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
        
        # Calculate price-based imbalance
        price_changes = torch.diff(closes)
        volume_weighted = volumes[1:] * torch.abs(price_changes)
        
        buy_volume = torch.sum(volume_weighted[price_changes > 0])
        sell_volume = torch.sum(volume_weighted[price_changes < 0])
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            buy_ratio = (buy_volume / total_volume).item()
            sell_ratio = (sell_volume / total_volume).item()
            
            imbalance_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else float('inf')
            
            if imbalance_ratio > self.orderflow_config['imbalance_threshold']:
                imbalances.append({
                    'type': 'BUY_IMBALANCE',
                    'strength': min(imbalance_ratio / 3.0, 1.0),
                    'timestamp': time.time()
                })
            elif imbalance_ratio < 1.0 / self.orderflow_config['imbalance_threshold']:
                imbalances.append({
                    'type': 'SELL_IMBALANCE',
                    'strength': min((1.0 / imbalance_ratio) / 3.0, 1.0),
                    'timestamp': time.time()
                })
        
        return imbalances

    def _detect_absorption(self, df):
        """Detect absorption patterns in order flow"""
        absorption_signals = []
        
        if len(df) < 20:
            return absorption_signals
        
        # Look for high volume with small price movement (absorption)
        for i in range(10, len(df)):
            recent_volume = df['volume'].iloc[i-5:i].mean()
            current_volume = df['volume'].iloc[i]
            volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            
            price_range = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
            
            # Absorption: high volume with small price range
            if volume_ratio > 2.0 and price_range < 0.001:
                absorption_signals.append({
                    'price': df['close'].iloc[i],
                    'strength': min(volume_ratio / 3.0, 1.0),
                    'timestamp': time.time(),
                    'volume_ratio': volume_ratio
                })
        
        return absorption_signals

    def _identify_liquidity_zones(self, df):
        """Identify liquidity zones based on volume clustering"""
        liquidity_zones = []
        
        if len(df) < 50:
            return liquidity_zones
        
        # Find price levels with high volume concentration
        price_levels = np.linspace(df['low'].min(), df['high'].max(), 100)
        volume_at_price = np.zeros_like(price_levels)
        
        for i in range(len(df)):
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            volume = df['volume'].iloc[i]
            
            for j, price in enumerate(price_levels):
                if low <= price <= high:
                    volume_at_price[j] += volume
        
        # Identify high volume zones
        avg_volume = np.mean(volume_at_price)
        high_volume_threshold = avg_volume * 1.5
        
        for i in range(1, len(price_levels)-1):
            if (volume_at_price[i] > high_volume_threshold and
                volume_at_price[i] > volume_at_price[i-1] and
                volume_at_price[i] > volume_at_price[i+1]):
                
                liquidity_zones.append({
                    'price': price_levels[i],
                    'volume_strength': volume_at_price[i] / avg_volume,
                    'type': 'LIQUIDITY_POOL'
                })
        
        return liquidity_zones

    def _update_order_flow_storage(self, analysis):
        """Update order flow data storage"""
        # Update imbalances
        for imbalance in analysis.get('imbalances', []):
            if imbalance['type'] == 'BUY_IMBALANCE':
                self.order_flow_data['buy_imbalances'].append(imbalance)
            else:
                self.order_flow_data['sell_imbalances'].append(imbalance)
        
        # Update absorption zones
        for absorption in analysis.get('absorption', []):
            self.order_flow_data['absorption_zones'].append(absorption)
        
        # Update liquidity pools
        for zone in analysis.get('liquidity_zones', []):
            self.order_flow_data['liquidity_pools'].append(zone)

    def get_orderflow_signals(self, current_price, psychology, volume_spike):
        """Get trading signals based on order flow analysis"""
        signals = []
        
        # Recent buy imbalances
        recent_buy_imbalances = [im for im in list(self.order_flow_data['buy_imbalances'])[-5:] 
                               if time.time() - im['timestamp'] < 3600]
        
        if recent_buy_imbalances and psychology['is_bullish']:
            avg_strength = np.mean([im['strength'] for im in recent_buy_imbalances])
            confidence = 7.0 + (avg_strength * 2.0)
            signals.append(("CALL", min(confidence, 9.0),
                          f"ORDER FLOW BULLISH | Imbalance Strength: {avg_strength:.2f}"))
        
        # Recent sell imbalances
        recent_sell_imbalances = [im for im in list(self.order_flow_data['sell_imbalances'])[-5:] 
                                if time.time() - im['timestamp'] < 3600]
        
        if recent_sell_imbalances and psychology['is_bearish']:
            avg_strength = np.mean([im['strength'] for im in recent_sell_imbalances])
            confidence = 7.0 + (avg_strength * 2.0)
            signals.append(("PUT", min(confidence, 9.0),
                          f"ORDER FLOW BEARISH | Imbalance Strength: {avg_strength:.2f}"))
        
        # Absorption signals
        recent_absorption = [absorp for absorp in list(self.order_flow_data['absorption_zones'])[-5:]
                           if time.time() - absorp['timestamp'] < 3600]
        
        for absorption in recent_absorption:
            if abs(current_price - absorption['price']) / absorption['price'] < 0.001:
                if volume_spike:
                    # Absorption breaking = strong move
                    if psychology['is_bullish']:
                        signals.append(("CALL", 8.0,
                                      f"ABSORPTION BREAKOUT BULLISH | Strength: {absorption['strength']:.2f}"))
                    elif psychology['is_bearish']:
                        signals.append(("PUT", 8.0,
                                      f"ABSORPTION BREAKOUT BEARISH | Strength: {absorption['strength']:.2f}"))
        
        return signals

# 6. MomentumOscillatorBrainGPU
@xgb_shadow('momentum_osc_brain', 'calculate_all_oscillators')
class MomentumOscillatorBrainGPU:
    """Advanced Momentum and Oscillator Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.momentum_config = {
            'rsi_periods': [6, 14, 21],
            'stoch_periods': [14, 21],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'momentum_periods': [10, 20, 50],
            'gpu_batch_size': 512
        }
        
        self.momentum_data = {
            'rsi_signals': LinuxOptimizedDeque(200),
            'stoch_signals': LinuxOptimizedDeque(200),
            'macd_signals': LinuxOptimizedDeque(200),
            'momentum_divergences': LinuxOptimizedDeque(100)
        }
        
        self.gpu_buffers = {}
        self._init_momentum_gpu_buffers()
        
        print(f"OK Momentum Oscillator Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_momentum_gpu_buffers(self):
        """Initialize GPU buffers for momentum calculations"""
        self.gpu_buffers['price_calc'] = torch.zeros(
            self.momentum_config['gpu_batch_size'], device=self.device
        )
        self.gpu_buffers['indicator_calc'] = torch.zeros(
            self.momentum_config['gpu_batch_size'], device=self.device
        )

    def calculate_all_oscillators(self, df):
        """Calculate all momentum oscillators using GPU acceleration"""
        try:
            if len(df) < 50:
                return {}
            
            # Move data to GPU
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            
            oscillators = {}
            
            # RSI Calculation
            oscillators['rsi'] = {}
            for period in self.momentum_config['rsi_periods']:
                if len(closes) >= period:
                    rsi = self._calculate_rsi_gpu(closes, period)
                    oscillators['rsi'][period] = rsi
            
            # Stochastic Calculation
            oscillators['stochastic'] = {}
            for period in self.momentum_config['stoch_periods']:
                if len(closes) >= period:
                    stoch = self._calculate_stochastic_gpu(highs, lows, closes, period)
                    oscillators['stochastic'][period] = stoch
            
            # MACD Calculation
            if len(closes) >= self.momentum_config['macd_slow']:
                macd, signal, histogram = self._calculate_macd_gpu(closes)
                oscillators['macd'] = {
                    'macd': macd,
                    'signal': signal,
                    'histogram': histogram
                }
            
            # Momentum Calculation
            oscillators['momentum'] = {}
            for period in self.momentum_config['momentum_periods']:
                if len(closes) >= period:
                    momentum = self._calculate_momentum_gpu(closes, period)
                    oscillators['momentum'][period] = momentum
            
            # Detect divergences
            oscillators['divergences'] = self._detect_divergences(closes, oscillators)
            
            # Update signals
            self._update_momentum_signals(oscillators)
            
            return oscillators
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1366)
            print(f"ERROR Momentum oscillator calculation failed: {e}")
            return {}

    def _calculate_rsi_gpu(self, closes, period):
        """GPU-accelerated RSI calculation"""
        try:
            if len(closes) < period + 1:
                return 50.0
            
            deltas = torch.diff(closes)
            gains = torch.where(deltas > 0, deltas, torch.tensor(0.0, device=self.device))
            losses = torch.where(deltas < 0, -deltas, torch.tensor(0.0, device=self.device))
            
            # Calculate rolling averages
            avg_gains = torch.zeros_like(closes)
            avg_losses = torch.zeros_like(closes)
            
            for i in range(period, len(closes)):
                avg_gains[i] = torch.mean(gains[i-period:i])
                avg_losses[i] = torch.mean(losses[i-period:i])
            
            rs = avg_gains / (avg_losses + 1e-10)  # FIX: epsilon prevents division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi[-1].item() if not torch.isnan(rsi[-1]) else 50.0
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1393)
            return 50.0

    def _calculate_stochastic_gpu(self, highs, lows, closes, period):
        """GPU-accelerated Stochastic calculation"""
        try:
            if len(closes) < period:
                return 50.0
            
            current_close = closes[-1]
            period_low = torch.min(lows[-period:])
            period_high = torch.max(highs[-period:])
            
            if period_high - period_low == 0:
                return 50.0
            
            stoch = ((current_close - period_low) / (period_high - period_low)) * 100
            return stoch.item()
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1412)
            return 50.0

    def _calculate_macd_gpu(self, closes):
        """GPU-accelerated MACD calculation"""
        try:
            if len(closes) < self.momentum_config['macd_slow']:
                return 0.0, 0.0, 0.0
            
            # Calculate EMAs
            ema_fast = self._calculate_ema_gpu(closes, self.momentum_config['macd_fast'])
            ema_slow = self._calculate_ema_gpu(closes, self.momentum_config['macd_slow'])
            
            macd_line = ema_fast - ema_slow
            signal_line = self._calculate_ema_gpu(macd_line, self.momentum_config['macd_signal'])
            histogram = macd_line - signal_line
            
            return macd_line[-1].item(), signal_line[-1].item(), histogram[-1].item()
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1431)
            return 0.0, 0.0, 0.0

    def _calculate_ema_gpu(self, data, period):
        """Calculate EMA on GPU"""
        if len(data) < period:
            return data
        
        alpha = 2.0 / (period + 1.0)
        ema = torch.zeros_like(data)
        ema[period-1] = torch.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def _calculate_momentum_gpu(self, closes, period):
        """Calculate price momentum on GPU"""
        if len(closes) < period:
            return 0.0
        
        momentum = ((closes[-1] - closes[-period]) / closes[-period]) * 100
        return momentum.item()

    def _detect_divergences(self, closes, oscillators):
        """Detect momentum divergences"""
        divergences = []
        
        # Simple divergence detection
        if len(closes) >= 20:
            price_trend = closes[-1] > closes[-10]  # True if rising
            rsi_trend = False
            
            if 'rsi' in oscillators and 14 in oscillators['rsi']:
                current_rsi = oscillators['rsi'][14]
                rsi_trend = current_rsi > 50  # Simplified trend detection
                
                # Regular divergence
                if price_trend and not rsi_trend and current_rsi < 70:
                    divergences.append({
                        'type': 'BEARISH_DIVERGENCE',
                        'strength': (70 - current_rsi) / 20.0,
                        'indicator': 'RSI'
                    })
                elif not price_trend and rsi_trend and current_rsi > 30:
                    divergences.append({
                        'type': 'BULLISH_DIVERGENCE', 
                        'strength': (current_rsi - 30) / 20.0,
                        'indicator': 'RSI'
                    })
        
        return divergences

    def _update_momentum_signals(self, oscillators):
        """Update momentum signal storage"""
        current_time = time.time()
        
        # RSI signals
        if 'rsi' in oscillators:
            for period, value in oscillators['rsi'].items():
                if value < 30:
                    self.momentum_data['rsi_signals'].append({
                        'type': 'RSI_OVERSOLD',
                        'period': period,
                        'value': value,
                        'timestamp': current_time
                    })
                elif value > 70:
                    self.momentum_data['rsi_signals'].append({
                        'type': 'RSI_OVERBOUGHT',
                        'period': period,
                        'value': value,
                        'timestamp': current_time
                    })
        
        # Stochastic signals
        if 'stochastic' in oscillators:
            for period, value in oscillators['stochastic'].items():
                if value < 20:
                    self.momentum_data['stoch_signals'].append({
                        'type': 'STOCH_OVERSOLD',
                        'period': period,
                        'value': value,
                        'timestamp': current_time
                    })
                elif value > 80:
                    self.momentum_data['stoch_signals'].append({
                        'type': 'STOCH_OVERBOUGHT',
                        'period': period,
                        'value': value,
                        'timestamp': current_time
                    })
        
        # MACD signals
        if 'macd' in oscillators:
            macd_data = oscillators['macd']
            if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
                self.momentum_data['macd_signals'].append({
                    'type': 'MACD_BULLISH',
                    'histogram': macd_data['histogram'],
                    'timestamp': current_time
                })
            elif macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
                self.momentum_data['macd_signals'].append({
                    'type': 'MACD_BEARISH',
                    'histogram': macd_data['histogram'],
                    'timestamp': current_time
                })
        
        # Divergence signals
        for divergence in oscillators.get('divergences', []):
            self.momentum_data['momentum_divergences'].append({
                **divergence,
                'timestamp': current_time
            })

    def get_momentum_signals(self, current_price, psychology):
        """Get trading signals based on momentum analysis"""
        signals = []
        current_time = time.time()
        
        # Recent RSI oversold with bullish psychology
        recent_oversold = [sig for sig in list(self.momentum_data['rsi_signals'])[-5:]
                         if sig['type'] == 'RSI_OVERSOLD' and current_time - sig['timestamp'] < 3600]
        
        if recent_oversold and psychology['is_bullish']:
            avg_strength = np.mean([(30 - sig['value']) / 10.0 for sig in recent_oversold])
            confidence = 7.0 + (avg_strength * 2.0)
            signals.append(("CALL", min(confidence, 9.0),
                          f"MOMENTUM REVERSAL BULLISH | RSI Oversold Strength: {avg_strength:.2f}"))
        
        # Recent RSI overbought with bearish psychology
        recent_overbought = [sig for sig in list(self.momentum_data['rsi_signals'])[-5:]
                           if sig['type'] == 'RSI_OVERBOUGHT' and current_time - sig['timestamp'] < 3600]
        
        if recent_overbought and psychology['is_bearish']:
            avg_strength = np.mean([(sig['value'] - 70) / 10.0 for sig in recent_overbought])
            confidence = 7.0 + (avg_strength * 2.0)
            signals.append(("PUT", min(confidence, 9.0),
                          f"MOMENTUM REVERSAL BEARISH | RSI Overbought Strength: {avg_strength:.2f}"))
        
        # MACD bullish crossover confirmation
        recent_macd_bullish = [sig for sig in list(self.momentum_data['macd_signals'])[-3:]
                             if sig['type'] == 'MACD_BULLISH' and current_time - sig['timestamp'] < 3600]
        
        if len(recent_macd_bullish) >= 2 and psychology['is_bullish']:
            avg_histogram = np.mean([sig['histogram'] for sig in recent_macd_bullish])
            confidence = 7.5 + (abs(avg_histogram) * 10.0)
            signals.append(("CALL", min(confidence, 9.0),
                          f"MACD BULLISH CONFIRMATION | Histogram Strength: {avg_histogram:.4f}"))
        
        # MACD bearish crossover confirmation
        recent_macd_bearish = [sig for sig in list(self.momentum_data['macd_signals'])[-3:]
                             if sig['type'] == 'MACD_BEARISH' and current_time - sig['timestamp'] < 3600]
        
        if len(recent_macd_bearish) >= 2 and psychology['is_bearish']:
            avg_histogram = np.mean([sig['histogram'] for sig in recent_macd_bearish])
            confidence = 7.5 + (abs(avg_histogram) * 10.0)
            signals.append(("PUT", min(confidence, 9.0),
                          f"MACD BEARISH CONFIRMATION | Histogram Strength: {avg_histogram:.4f}"))
        
        # Divergence signals
        recent_divergences = [div for div in list(self.momentum_data['momentum_divergences'])[-3:]
                            if current_time - div['timestamp'] < 3600]
        
        for div in recent_divergences:
            if div['type'] == 'BULLISH_DIVERGENCE' and psychology['is_bullish']:
                confidence = 7.0 + (div['strength'] * 2.0)
                signals.append(("CALL", min(confidence, 9.0),
                              f"BULLISH DIVERGENCE | Strength: {div['strength']:.2f}"))
            elif div['type'] == 'BEARISH_DIVERGENCE' and psychology['is_bearish']:
                confidence = 7.0 + (div['strength'] * 2.0)
                signals.append(("PUT", min(confidence, 9.0),
                              f"BEARISH DIVERGENCE | Strength: {div['strength']:.2f}"))
        
        return signals

# 7. VolatilityRegimeBrainGPU
@xgb_shadow('vol_regime_brain', 'analyze_volatility_regime')
class VolatilityRegimeBrainGPU:
    """Volatility Regime Detection and Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.volatility_config = {
            'regime_periods': [20, 50, 100],
            'volatility_breakout_threshold': 2.0,
            'squeeze_threshold': 0.5,
            'gpu_batch_size': 512
        }
        
        self.volatility_data = {
            'current_regime': 'NORMAL',
            'regime_history': LinuxOptimizedDeque(500),
            'volatility_expansion': LinuxOptimizedDeque(200),
            'volatility_contraction': LinuxOptimizedDeque(200),
            'squeeze_signals': LinuxOptimizedDeque(100)
        }
        
        self.gpu_buffers = {}
        self._init_volatility_gpu_buffers()
        
        print(f"OK Volatility Regime Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_volatility_gpu_buffers(self):
        """Initialize GPU buffers for volatility calculations"""
        self.gpu_buffers['volatility_calc'] = torch.zeros(
            self.volatility_config['gpu_batch_size'], device=self.device
        )

    def analyze_volatility_regime(self, df):
        """Analyze current volatility regime using GPU acceleration"""
        try:
            if len(df) < 50:
                return self.volatility_data['current_regime']
            
            # Move data to GPU
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            
            volatility_analysis = {}
            
            # Calculate volatility for different periods
            for period in self.volatility_config['regime_periods']:
                if len(closes) >= period:
                    volatility = self._calculate_volatility_gpu(closes, period)
                    volatility_analysis[period] = volatility
            
            # Determine current regime
            regime = self._determine_volatility_regime(volatility_analysis)
            
            # Detect volatility breakouts and squeezes
            breakout_signals = self._detect_volatility_breakouts(highs, lows, closes)
            squeeze_signals = self._detect_volatility_squeeze(highs, lows, closes)
            
            # Update storage
            self._update_volatility_data(regime, breakout_signals, squeeze_signals)
            
            return regime
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1673)
            print(f"ERROR Volatility regime analysis failed: {e}")
            return self.volatility_data['current_regime']

    def _calculate_volatility_gpu(self, closes, period):
        """GPU-accelerated volatility calculation"""
        try:
            if len(closes) < period:
                return 0.0
            
            returns = torch.diff(closes) / closes[:-1]
            period_returns = returns[-period:]
            
            volatility = torch.std(period_returns) * np.sqrt(252)  # Annualized
            return volatility.item()
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1689)
            return 0.0

    def _determine_volatility_regime(self, volatility_analysis):
        """Determine current volatility regime"""
        if not volatility_analysis:
            return 'NORMAL'
        
        # Use medium-term volatility as reference
        ref_period = 50
        if ref_period not in volatility_analysis:
            ref_period = list(volatility_analysis.keys())[0]
        
        current_vol = volatility_analysis[ref_period]
        
        # Simple regime classification
        if current_vol < 0.1:  # 10% annualized volatility
            return 'LOW_VOLATILITY'
        elif current_vol < 0.25:  # 25% annualized volatility
            return 'NORMAL'
        elif current_vol < 0.5:  # 50% annualized volatility
            return 'HIGH_VOLATILITY'
        else:
            return 'EXTREME_VOLATILITY'

    def _detect_volatility_breakouts(self, highs, lows, closes):
        """Detect volatility breakout signals"""
        breakouts = []
        
        if len(closes) < 20:
            return breakouts
        
        # Calculate recent volatility
        recent_vol = self._calculate_volatility_gpu(closes, 20)
        
        # Calculate average true range
        if len(highs) >= 2:
            # FIX: convert torch scalars to float before max()
            tr1 = (highs[-1] - lows[-1]).item()
            tr2 = abs((highs[-1] - closes[-2]).item())
            tr3 = abs((lows[-1] - closes[-2]).item())
            true_range = max(tr1, tr2, tr3)

            avg_true_range = torch.mean(highs[-20:] - lows[-20:]).item()  # FIX: .item() to float

            if true_range > avg_true_range * self.volatility_config['volatility_breakout_threshold']:
                breakouts.append({
                    'type': 'VOLATILITY_EXPANSION',
                    'strength': true_range / (avg_true_range + 1e-8),
                    'timestamp': time.time()
                })
        
        return breakouts

    def _detect_volatility_squeeze(self, highs, lows, closes):
        """Detect volatility squeeze signals"""
        squeezes = []
        
        if len(closes) < 20:
            return squeezes
        
        # Calculate Bollinger Band width and Keltner Channel width
        bb_width = self._calculate_bb_width(closes, 20)
        kc_width = self._calculate_kc_width(highs, lows, closes, 20)
        
        if bb_width > 0 and kc_width > 0:
            squeeze_ratio = bb_width / kc_width
            
            if squeeze_ratio < self.volatility_config['squeeze_threshold']:
                squeezes.append({
                    'type': 'VOLATILITY_SQUEEZE',
                    'strength': (1.0 - squeeze_ratio) * 2.0,
                    'timestamp': time.time()
                })
        
        return squeezes

    def _calculate_bb_width(self, closes, period):
        """Calculate Bollinger Band width"""
        if len(closes) < period:
            return 0.0
        
        sma = torch.mean(closes[-period:])
        std = torch.std(closes[-period:])
        
        return (2 * std / sma).item() if sma > 0 else 0.0

    def _calculate_kc_width(self, highs, lows, closes, period):
        """Calculate Keltner Channel width"""
        if len(closes) < period:
            return 0.0
        
        # Typical price
        typical_price = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        atr = torch.mean(highs[-period:] - lows[-period:])
        
        kc_upper = torch.mean(typical_price) + 2 * atr
        kc_lower = torch.mean(typical_price) - 2 * atr
        
        return ((kc_upper - kc_lower) / torch.mean(typical_price)).item()

    def _update_volatility_data(self, regime, breakouts, squeezes):
        """Update volatility data storage"""
        self.volatility_data['current_regime'] = regime
        self.volatility_data['regime_history'].append({
            'regime': regime,
            'timestamp': time.time()
        })
        
        for breakout in breakouts:
            self.volatility_data['volatility_expansion'].append(breakout)
        
        for squeeze in squeezes:
            self.volatility_data['squeeze_signals'].append(squeeze)

    def get_volatility_signals(self, current_price, psychology):
        """Get trading signals based on volatility regime"""
        signals = []
        current_time = time.time()
        
        regime = self.volatility_data['current_regime']
        
        # Volatility expansion signals
        recent_expansions = [exp for exp in list(self.volatility_data['volatility_expansion'])[-3:]
                           if current_time - exp['timestamp'] < 3600]
        
        if recent_expansions and regime in ['HIGH_VOLATILITY', 'EXTREME_VOLATILITY']:
            # In high volatility, look for momentum continuation
            if psychology['has_strong_momentum']:
                if psychology['is_bullish']:
                    signals.append(("CALL", 8.0,
                                  f"HIGH VOL MOMENTUM BULLISH | Regime: {regime}"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", 8.0,
                                  f"HIGH VOL MOMENTUM BEARISH | Regime: {regime}"))
        
        # Volatility squeeze signals
        recent_squeezes = [sq for sq in list(self.volatility_data['squeeze_signals'])[-3:]
                         if current_time - sq['timestamp'] < 3600]
        
        if recent_squeezes and regime in ['LOW_VOLATILITY', 'NORMAL']:
            # In low volatility, look for breakout setups
            if psychology['body_ratio'] > 0.7:  # Strong momentum candle
                if psychology['is_bullish']:
                    signals.append(("CALL", 8.5,
                                  f"VOLATILITY SQUEEZE BREAKOUT BULLISH | Strength: {recent_squeezes[0]['strength']:.2f}"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", 8.5,
                                  f"VOLATILITY SQUEEZE BREAKOUT BEARISH | Strength: {recent_squeezes[0]['strength']:.2f}"))
        
        # Regime-specific signals
        if regime == 'LOW_VOLATILITY':
            # In low volatility, favor range-bound strategies
            if psychology['has_strong_rejection']:
                if psychology['lower_wick_ratio'] > psychology['upper_wick_ratio']:
                    signals.append(("CALL", 7.5, "LOW VOL RANGE SUPPORT BOUNCE"))
                else:
                    signals.append(("PUT", 7.5, "LOW VOL RANGE RESISTANCE REJECTION"))
        
        elif regime == 'EXTREME_VOLATILITY':
            # In extreme volatility, be cautious or use smaller positions
            if psychology['has_strong_momentum']:
                confidence = 6.0  # Lower confidence in extreme volatility
                if psychology['is_bullish']:
                    signals.append(("CALL", confidence, "EXTREME VOL MOMENTUM BULLISH (CAUTION)"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", confidence, "EXTREME VOL MOMENTUM BEARISH (CAUTION)"))
        
        return signals

# 8. CycleAnalysisBrainGPU  
@xgb_shadow('cycle_brain', 'analyze_market_cycles')
class CycleAnalysisBrainGPU:
    """Market Cycle Analysis and Identification"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cycle_config = {
            'cycle_periods': [10, 20, 40, 80, 160],
            'dominant_cycle_threshold': 0.7,
            'cycle_phase_tolerance': 0.1,
            'gpu_batch_size': 512
        }
        
        self.cycle_data = {
            'dominant_cycle': None,
            'cycle_phases': LinuxOptimizedDeque(500),
            'cycle_turns': LinuxOptimizedDeque(200),
            'seasonal_patterns': LinuxOptimizedDeque(1000)
        }
        
        self.gpu_buffers = {}
        self._init_cycle_gpu_buffers()
        
        print(f"OK Cycle Analysis Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_cycle_gpu_buffers(self):
        """Initialize GPU buffers for cycle analysis"""
        self.gpu_buffers['cycle_calc'] = torch.zeros(
            self.cycle_config['gpu_batch_size'], device=self.device
        )

    def analyze_market_cycles(self, df):
        """Analyze market cycles using GPU-accelerated methods"""
        try:
            if len(df) < 100:
                return {}
            
            # Move data to GPU
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            
            cycle_analysis = {}
            
            # Detect dominant cycle using FFT
            dominant_cycle = self._detect_dominant_cycle_fft(closes)
            cycle_analysis['dominant_cycle'] = dominant_cycle
            
            # Calculate current cycle phase
            current_phase = self._calculate_cycle_phase(closes, dominant_cycle)
            cycle_analysis['current_phase'] = current_phase
            
            # Identify cycle turns
            cycle_turns = self._identify_cycle_turns(closes, dominant_cycle)
            cycle_analysis['cycle_turns'] = cycle_turns
            
            # Update storage
            self._update_cycle_data(cycle_analysis)
            
            return cycle_analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1919)
            print(f"ERROR Market cycle analysis failed: {e}")
            return {}

    def _detect_dominant_cycle_fft(self, closes):
        """Detect dominant cycle using Fast Fourier Transform"""
        try:
            # Use CPU for FFT (more efficient for this operation)
            closes_cpu = closes.cpu().numpy()
            
            # Remove trend
            detrended = closes_cpu - np.mean(closes_cpu)
            
            # Apply FFT
            fft_result = np.fft.fft(detrended)
            frequencies = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequency (ignore DC component and negative frequencies)
            magnitudes = np.abs(fft_result)
            positive_freq_idx = frequencies > 0
            dominant_idx = np.argmax(magnitudes[positive_freq_idx])
            dominant_frequency = frequencies[positive_freq_idx][dominant_idx]
            
            if dominant_frequency > 0:
                cycle_period = int(1.0 / dominant_frequency)
                return max(10, min(cycle_period, 200))  # Reasonable bounds
            else:
                return 20  # Default fallback
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1948)
            return 20

    def _calculate_cycle_phase(self, closes, cycle_period):
        """Calculate current cycle phase"""
        if cycle_period is None or len(closes) < cycle_period:
            return 0.0
        
        # Simple phase calculation based on position in cycle
        current_position = len(closes) % cycle_period
        phase = (current_position / cycle_period) * 2 * np.pi
        
        return phase

    def _identify_cycle_turns(self, closes, cycle_period):
        """Identify potential cycle turning points"""
        turns = []
        
        if cycle_period is None or len(closes) < cycle_period * 2:
            return turns
        
        # Look for local extremes at cycle intervals
        for i in range(cycle_period, len(closes) - cycle_period):
            window = closes[i-cycle_period:i+cycle_period]
            center = closes[i]
            
            # Check for local maximum
            if torch.all(center >= window):
                turns.append({
                    'type': 'CYCLE_HIGH',
                    'price': center.item(),
                    'index': i,
                    'timestamp': time.time()
                })
            # Check for local minimum
            elif torch.all(center <= window):
                turns.append({
                    'type': 'CYCLE_LOW', 
                    'price': center.item(),
                    'index': i,
                    'timestamp': time.time()
                })
        
        return turns[-5:]  # Return only recent turns

    def _update_cycle_data(self, cycle_analysis):
        """Update cycle data storage"""
        if cycle_analysis['dominant_cycle']:
            self.cycle_data['dominant_cycle'] = cycle_analysis['dominant_cycle']
        
        self.cycle_data['cycle_phases'].append({
            'phase': cycle_analysis.get('current_phase', 0.0),
            'timestamp': time.time(),
            'dominant_cycle': cycle_analysis.get('dominant_cycle')
        })
        
        for turn in cycle_analysis.get('cycle_turns', []):
            self.cycle_data['cycle_turns'].append(turn)

    def get_cycle_signals(self, current_price, psychology):
        """Get trading signals based on cycle analysis"""
        signals = []
        
        if not self.cycle_data['dominant_cycle']:
            return signals
        
        current_phase = 0.0
        if len(self.cycle_data['cycle_phases']) > 0:
            current_phase = self.cycle_data['cycle_phases'][-1]['phase']
        
        # Convert phase to 0-1 range for easier interpretation
        normalized_phase = (current_phase % (2 * np.pi)) / (2 * np.pi)
        
        # Cycle-based signals
        if 0.75 <= normalized_phase <= 1.0 or 0.0 <= normalized_phase <= 0.25:
            # Cycle bottoming zone - look for bullish setups
            if psychology['is_bullish'] and psychology['has_strong_rejection']:
                cycle_strength = 1.0 - abs(normalized_phase - 0.0) if normalized_phase <= 0.25 else abs(normalized_phase - 1.0)
                confidence = 7.0 + (cycle_strength * 2.0)
                signals.append(("CALL", min(confidence, 9.0),
                              f"CYCLE BOTTOM BULLISH | Phase: {normalized_phase:.2f}"))
        
        elif 0.25 <= normalized_phase <= 0.75:
            # Cycle topping zone - look for bearish setups
            if psychology['is_bearish'] and psychology['has_strong_rejection']:
                cycle_strength = 1.0 - abs(normalized_phase - 0.5)
                confidence = 7.0 + (cycle_strength * 2.0)
                signals.append(("PUT", min(confidence, 9.0),
                              f"CYCLE TOP BEARISH | Phase: {normalized_phase:.2f}"))
        
        # Recent cycle turn signals
        recent_turns = list(self.cycle_data['cycle_turns'])[-5:]
        for turn in recent_turns:
            if time.time() - turn['timestamp'] < 3600:  # Last hour
                if turn['type'] == 'CYCLE_LOW' and psychology['is_bullish']:
                    signals.append(("CALL", 8.0, "RECENT CYCLE LOW BOUNCE"))
                elif turn['type'] == 'CYCLE_HIGH' and psychology['is_bearish']:
                    signals.append(("PUT", 8.0, "RECENT CYCLE HIGH REJECTION"))
        
        return signals

# 9. CorrelationMatrixBrainGPU
@xgb_shadow('correlation_brain', 'analyze_correlations')
class CorrelationMatrixBrainGPU:
    """Inter-market Correlation Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.correlation_config = {
            'correlation_periods': [20, 50, 100],
            'strong_correlation_threshold': 0.7,
            'gpu_batch_size': 512
        }
        
        self.correlation_data = {
            'asset_correlations': defaultdict(lambda: LinuxOptimizedDeque(200)),
            'sector_relationships': LinuxOptimizedDeque(500),
            'risk_on_off_signals': LinuxOptimizedDeque(200)
        }
        
        # Example correlated assets (would be populated with real data)
        self.correlated_assets = ['SPX', 'NDX', 'RUT', 'VIX', 'DXY', 'TLT', 'GLD']
        
        self.gpu_buffers = {}
        self._init_correlation_gpu_buffers()
        
        print(f"OK Correlation Matrix Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_correlation_gpu_buffers(self):
        """Initialize GPU buffers for correlation calculations"""
        self.gpu_buffers['correlation_calc'] = torch.zeros(
            self.correlation_config['gpu_batch_size'], device=self.device
        )

    def analyze_correlations(self, primary_data, correlated_data_dict):
        """Advanced Math: PCA-based Eigen-decomposition for inter-market correlation"""
        try:
            if len(primary_data) < 50:
                return {}
            
            correlation_analysis = {}
            
            # Move primary data to GPU
            primary_returns = self._calculate_returns_gpu(primary_data)
            
            # Gather all asset returns into a single matrix for PCA
            valid_assets = []
            return_matrix_cols = [primary_returns[-50:]]

            _tf_tokens = {'1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M','1min','5min','15min','60m','240m'}
            for asset_name, asset_data in correlated_data_dict.items():
                if str(asset_name) in _tf_tokens or len(asset_data) < 51:
                    continue
                asset_returns = self._calculate_returns_gpu(asset_data)
                if len(asset_returns) < 20:
                    continue
                
                # Calculate basic correlation on aligned returns (never raw prices)
                asset_correlations = {}
                for period in self.correlation_config['correlation_periods']:
                    if len(primary_returns) >= period and len(asset_returns) >= period:
                        correlation = self._calculate_correlation_gpu(
                            primary_returns[-period:], 
                            asset_returns[-period:]
                        )
                        asset_correlations[period] = correlation
                
                correlation_analysis[asset_name] = asset_correlations

                # Prep for PCA
                if len(asset_returns) >= 50 and len(primary_returns) >= 50:
                    valid_assets.append(asset_name)
                    return_matrix_cols.append(asset_returns[-50:])
            
            # Perform PCA (Eigen-decomposition) to find the principal market driver
            if len(valid_assets) >= 2:
                try:
                    # Stack into (Time x Assets)
                    matrix = torch.stack(return_matrix_cols, dim=1)

                    # Mean centering
                    mean_vec = torch.mean(matrix, dim=0)
                    centered = matrix - mean_vec

                    # Covariance matrix (Assets x Assets)
                    cov_matrix = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)

                    # Eigen-decomposition
                    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

                    # Sort eigenvalues and eigenvectors in descending order
                    sorted_indices = torch.argsort(eigenvalues, descending=True)
                    eigenvalues = eigenvalues[sorted_indices]
                    eigenvectors = eigenvectors[:, sorted_indices]

                    # Extract the principal component vector
                    pc1_vector = eigenvectors[:, 0]
                    variance_explained = (eigenvalues[0] / torch.sum(eigenvalues)).item()

                    pca_insights = {
                        'primary_pc1_weight': pc1_vector[0].item(),
                        'variance_explained': variance_explained,
                        'asset_weights': {name: pc1_vector[i+1].item() for i, name in enumerate(valid_assets)}
                    }
                    correlation_analysis['pca_insights'] = pca_insights
                except Exception as e:
                    globals().get("_plog", lambda *_a, **_k: None)(__name__, 2151)
                    correlation_analysis['pca_insights'] = {'error': str(e)}

            # Detect regime changes
            regime_signals = self._detect_correlation_regime(correlation_analysis)
            correlation_analysis['regime_signals'] = regime_signals
            
            # Update storage
            self._update_correlation_data(correlation_analysis)
            
            return correlation_analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 2163)
            print(f"ERROR Correlation analysis failed: {e}")
            return {}

    def _calculate_returns_gpu(self, price_data):
        """Calculate returns on GPU"""
        if len(price_data) < 2:
            return torch.tensor([], device=self.device)
        
        if hasattr(price_data, 'columns'):
            _col = 'close' if 'close' in price_data.columns else ('Close' if 'Close' in price_data.columns else price_data.columns[-1])
            price_data = price_data[_col].astype(float).tolist()
        elif hasattr(price_data, 'tolist'):
            price_data = price_data.tolist()
        prices = torch.tensor(price_data, device=self.device, dtype=torch.float32)
        returns = torch.diff(prices) / prices[:-1]
        
        return returns

    def _calculate_correlation_gpu(self, returns1, returns2):
        """Calculate correlation coefficient on GPU"""
        try:
            if len(returns1) != len(returns2) or len(returns1) < 10:
                return 0.0
            
            # Ensure both are tensors on GPU
            if isinstance(returns1, list):
                returns1 = torch.tensor(returns1, device=self.device, dtype=torch.float32)
            if isinstance(returns2, list):
                returns2 = torch.tensor(returns2, device=self.device, dtype=torch.float32)
            
            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([returns1, returns2]))[0, 1]
            
            return correlation.item() if not torch.isnan(correlation) else 0.0
            
        except:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 2194)
            return 0.0

    def _detect_correlation_regime(self, correlation_analysis):
        """Detect correlation regime changes"""
        regime_signals = []
        
        # Simple regime detection based on VIX correlation
        if 'VIX' in correlation_analysis:
            vix_correlations = correlation_analysis['VIX']
            
            for period, correlation in vix_correlations.items():
                if abs(correlation) > self.correlation_config['strong_correlation_threshold']:
                    if correlation < 0:  # Strong negative correlation with VIX
                        regime_signals.append({
                            'type': 'RISK_ON',
                            'strength': abs(correlation),
                            'period': period,
                            'timestamp': time.time()
                        })
                    else:  # Strong positive correlation with VIX
                        regime_signals.append({
                            'type': 'RISK_OFF', 
                            'strength': abs(correlation),
                            'period': period,
                            'timestamp': time.time()
                        })
        
        return regime_signals

    def _update_correlation_data(self, correlation_analysis):
        """Update correlation data storage"""
        current_time = time.time()
        
        # Update asset correlations
        for asset_name, correlations in correlation_analysis.items():
            if asset_name == 'regime_signals':
                continue
                
            for period, correlation in correlations.items():
                self.correlation_data['asset_correlations'][asset_name].append({
                    'period': period,
                    'correlation': correlation,
                    'timestamp': current_time
                })
        
        # Update regime signals
        for signal in correlation_analysis.get('regime_signals', []):
            self.correlation_data['risk_on_off_signals'].append(signal)

    def get_correlation_signals(self, current_price, psychology):
        """Get trading signals based on correlation analysis"""
        signals = []
        current_time = time.time()
        
        # Recent risk-on signals
        recent_risk_on = [sig for sig in list(self.correlation_data['risk_on_off_signals'])[-5:]
                        if sig['type'] == 'RISK_ON' and current_time - sig['timestamp'] < 3600]
        
        if recent_risk_on and psychology['is_bullish']:
            avg_strength = np.mean([sig['strength'] for sig in recent_risk_on])
            confidence = 7.5 + (avg_strength * 1.5)
            signals.append(("CALL", min(confidence, 9.0),
                          f"RISK-ON REGIME BULLISH | Correlation Strength: {avg_strength:.2f}"))
        
        # Recent risk-off signals
        recent_risk_off = [sig for sig in list(self.correlation_data['risk_on_off_signals'])[-5:]
                         if sig['type'] == 'RISK_OFF' and current_time - sig['timestamp'] < 3600]
        
        if recent_risk_off and psychology['is_bearish']:
            avg_strength = np.mean([sig['strength'] for sig in recent_risk_off])
            confidence = 7.5 + (avg_strength * 1.5)
            signals.append(("PUT", min(confidence, 9.0),
                          f"RISK-OFF REGIME BEARISH | Correlation Strength: {avg_strength:.2f}"))
        
        return signals

# 10. PatternRecognitionBrainGPU
@xgb_shadow('pattern_brain', 'recognize_chart_patterns')
class PatternRecognitionBrainGPU:
    """Advanced Chart Pattern Recognition"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pattern_config = {
            'pattern_lengths': [5, 10, 20],
            'similarity_threshold': 0.8,
            'confirmation_candles': 2,
            'gpu_batch_size': 512
        }
        
        self.pattern_library = {
            'bullish_patterns': LinuxOptimizedDeque(500),
            'bearish_patterns': LinuxOptimizedDeque(500),
            'reversal_patterns': LinuxOptimizedDeque(500),
            'continuation_patterns': LinuxOptimizedDeque(500)
        }
        
        self.gpu_buffers = {}
        self._init_pattern_gpu_buffers()
        
        print(f"OK Pattern Recognition Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_pattern_gpu_buffers(self):
        """Initialize GPU buffers for pattern recognition"""
        self.gpu_buffers['pattern_match'] = torch.zeros(
            self.pattern_config['gpu_batch_size'], device=self.device
        )

    def recognize_chart_patterns(self, df):
        """Recognize chart patterns in price data"""
        try:
            if len(df) < 20:
                return []
            
            patterns_found = []
            
            # Move data to GPU
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            opens = torch.tensor(df['open'].values, device=self.device, dtype=torch.float32)
            
            # Look for common patterns
            patterns_found.extend(self._detect_double_top_bottom(highs, lows))
            patterns_found.extend(self._detect_head_shoulders(highs, lows))
            patterns_found.extend(self._detect_triangle_patterns(highs, lows))
            patterns_found.extend(self._detect_flag_pennants(highs, lows))
            
            # Update pattern library
            self._update_pattern_library(patterns_found)
            
            return patterns_found
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 2328)
            print(f"ERROR Pattern recognition failed: {e}")
            return []

    def _detect_double_top_bottom(self, highs, lows):
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(highs) < 20:
            return patterns
        
        # Look for double top (bearish reversal)
        for i in range(10, len(highs) - 10):
            # Check for two similar highs with trough in between
            left_high = highs[i-5]
            right_high = highs[i+5]
            trough = lows[i]

            # FIX: .item() so high_similarity is a Python float, not a tensor
            high_similarity = (abs(left_high - right_high) / ((left_high + right_high) / 2)).item()

            if high_similarity < 0.01:  # Within 1%
                patterns.append({
                    'type': 'DOUBLE_TOP',
                    'price_level': (left_high + right_high).item() / 2,
                    'neckline': trough.item(),
                    'confidence': float(1.0 - high_similarity),
                    'timestamp': time.time()
                })

        # Look for double bottom (bullish reversal)
        for i in range(10, len(lows) - 10):
            # Check for two similar lows with peak in between
            left_low = lows[i-5]
            right_low = lows[i+5]
            peak = highs[i]

            # FIX: .item() so low_similarity is a Python float, not a tensor
            low_similarity = (abs(left_low - right_low) / ((left_low + right_low) / 2)).item()

            if low_similarity < 0.01:  # Within 1%
                patterns.append({
                    'type': 'DOUBLE_BOTTOM',
                    'price_level': (left_low + right_low).item() / 2,
                    'neckline': peak.item(),
                    'confidence': float(1.0 - low_similarity),
                    'timestamp': time.time()
                })
        
        return patterns

    def _detect_head_shoulders(self, highs, lows):
        """Detect head and shoulders patterns"""
        patterns = []
        
        if len(highs) < 15:
            return patterns
        
        # Simplified head and shoulders detection
        for i in range(7, len(highs) - 7):
            left_shoulder = highs[i-6]
            head = highs[i-3]
            right_shoulder = highs[i]
            neckline = (lows[i-6] + lows[i]).item() / 2
            
            # Basic pattern criteria — FIX: .item() on tensor comparison
            if (head > left_shoulder and head > right_shoulder and
                (abs(left_shoulder - right_shoulder) / ((left_shoulder + right_shoulder) / 2)).item() < 0.02):
                
                patterns.append({
                    'type': 'HEAD_SHOULDERS',
                    'head_price': head.item(),
                    'shoulder_price': (left_shoulder + right_shoulder).item() / 2,
                    'neckline': neckline,
                    'confidence': 0.7,
                    'timestamp': time.time()
                })
        
        return patterns

    def _detect_triangle_patterns(self, highs, lows):
        """Detect triangle patterns (symmetrical, ascending, descending)"""
        patterns = []
        
        if len(highs) < 10:
            return patterns
        
        # Look for converging highs and lows
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        high_slope = self._calculate_slope_gpu(recent_highs)
        low_slope = self._calculate_slope_gpu(recent_lows)
        
        # Symmetrical triangle (both slopes converging)
        if high_slope < 0 and low_slope > 0:
            patterns.append({
                'type': 'SYMMETRICAL_TRIANGLE',
                'direction': 'CONSOLIDATION',
                'confidence': 0.6,
                'timestamp': time.time()
            })
        # Ascending triangle (flat highs, rising lows)
        elif abs(high_slope) < 0.001 and low_slope > 0:
            patterns.append({
                'type': 'ASCENDING_TRIANGLE',
                'direction': 'BULLISH',
                'confidence': 0.7,
                'timestamp': time.time()
            })
        # Descending triangle (falling highs, flat lows)
        elif high_slope < 0 and abs(low_slope) < 0.001:
            patterns.append({
                'type': 'DESCENDING_TRIANGLE', 
                'direction': 'BEARISH',
                'confidence': 0.7,
                'timestamp': time.time()
            })
        
        return patterns

    def _calculate_slope_gpu(self, data):
        """Calculate slope of data using linear regression on GPU"""
        if len(data) < 2:
            return 0.0
        
        x = torch.arange(len(data), device=self.device, dtype=torch.float32)
        x_mean = torch.mean(x)
        y_mean = torch.mean(data)
        
        numerator = torch.sum((x - x_mean) * (data - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        
        return (numerator / denominator).item() if denominator > 0 else 0.0

    def _detect_flag_pennants(self, highs, lows):
        """Detect flag and pennant patterns"""
        patterns = []
        
        if len(highs) < 15:
            return patterns
        
        # Look for small consolidation after strong move
        recent_range = torch.mean(highs[-5:] - lows[-5:])
        earlier_range = torch.mean(highs[-15:-10] - lows[-15:-10])
        
        if recent_range < earlier_range * 0.5:  # Significant range contraction
            # Check if preceded by strong move
            price_change = abs(highs[-15] - lows[-1]) / ((highs[-15] + lows[-1]) / 2)
            
            if price_change > 0.02:  # At least 2% move
                patterns.append({
                    'type': 'FLAG_PENNANT',
                    'direction': 'CONTINUATION',
                    'confidence': 0.6,
                    'timestamp': time.time()
                })
        
        return patterns

    def _update_pattern_library(self, new_patterns):
        """Update pattern library with new patterns"""
        for pattern in new_patterns:
            pattern_type = pattern['type']
            
            if 'TOP' in pattern_type or 'BEARISH' in pattern_type:
                self.pattern_library['bearish_patterns'].append(pattern)
            elif 'BOTTOM' in pattern_type or 'BULLISH' in pattern_type:
                self.pattern_library['bullish_patterns'].append(pattern)
            elif 'REVERSAL' in pattern_type:
                self.pattern_library['reversal_patterns'].append(pattern)
            elif 'CONTINUATION' in pattern_type:
                self.pattern_library['continuation_patterns'].append(pattern)

    def get_pattern_signals(self, current_price, psychology):
        """Get trading signals based on pattern recognition"""
        signals = []
        current_time = time.time()
        
        # Recent bullish patterns
        recent_bullish = [pat for pat in list(self.pattern_library['bullish_patterns'])[-5:]
                        if current_time - pat['timestamp'] < 3600]
        
        for pattern in recent_bullish:
            if psychology['is_bullish']:
                confidence = 7.0 + (pattern['confidence'] * 2.0)
                signals.append(("CALL", min(confidence, 9.0),
                              f"BULLISH PATTERN: {pattern['type']} | Confidence: {pattern['confidence']:.2f}"))
        
        # Recent bearish patterns
        recent_bearish = [pat for pat in list(self.pattern_library['bearish_patterns'])[-5:]
                        if current_time - pat['timestamp'] < 3600]
        
        for pattern in recent_bearish:
            if psychology['is_bearish']:
                confidence = 7.0 + (pattern['confidence'] * 2.0)
                signals.append(("PUT", min(confidence, 9.0),
                              f"BEARISH PATTERN: {pattern['type']} | Confidence: {pattern['confidence']:.2f}"))
        
        # Recent reversal patterns
        recent_reversals = [pat for pat in list(self.pattern_library['reversal_patterns'])[-5:]
                          if current_time - pat['timestamp'] < 3600]
        
        for pattern in recent_reversals:
            # Reversal patterns work best with confirmation
            if pattern['type'] in ['DOUBLE_TOP', 'HEAD_SHOULDERS'] and psychology['is_bearish']:
                confidence = 7.5 + (pattern['confidence'] * 1.5)
                signals.append(("PUT", min(confidence, 9.0),
                              f"REVERSAL PATTERN: {pattern['type']} | Confidence: {pattern['confidence']:.2f}"))
            elif pattern['type'] in ['DOUBLE_BOTTOM'] and psychology['is_bullish']:
                confidence = 7.5 + (pattern['confidence'] * 1.5)
                signals.append(("CALL", min(confidence, 9.0),
                              f"REVERSAL PATTERN: {pattern['type']} | Confidence: {pattern['confidence']:.2f}"))
        
        return signals

# 11. SupportResistanceBrainGPU
@xgb_shadow('sr_brain', 'calculate_support_resistance')
class SupportResistanceBrainGPU:
    """Dynamic Support and Resistance Level Calculation"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.sr_config = {
            'pivot_periods': [5, 10, 20],
            'cluster_tolerance': 0.002,
            'level_strength_threshold': 0.6,
            'gpu_batch_size': 512
        }
        
        self.sr_levels = {
            'support_levels': LinuxOptimizedDeque(100),
            'resistance_levels': LinuxOptimizedDeque(100),
            'strong_levels': LinuxOptimizedDeque(50),
            'tested_levels': LinuxOptimizedDeque(200)
        }
        
        self.gpu_buffers = {}
        self._init_sr_gpu_buffers()
        
        print(f"OK Support Resistance Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_sr_gpu_buffers(self):
        """Initialize GPU buffers for support/resistance calculations"""
        self.gpu_buffers['sr_calc'] = torch.zeros(
            self.sr_config['gpu_batch_size'], device=self.device
        )

    def calculate_support_resistance(self, df):
        """Calculate dynamic support and resistance levels"""
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}
            
            # Move data to GPU
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            
            levels = {
                'support': [],
                'resistance': []
            }
            
            # Method 1: Pivot Points
            pivot_levels = self._calculate_pivot_points(highs, lows, closes)
            levels['support'].extend(pivot_levels['support'])
            levels['resistance'].extend(pivot_levels['resistance'])
            
            # Method 2: Recent Swing Points
            swing_levels = self._calculate_swing_points(highs, lows)
            levels['support'].extend(swing_levels['support'])
            levels['resistance'].extend(swing_levels['resistance'])
            
            # Method 3: Volume-Weighted Price Levels
            volume_levels = self._calculate_volume_weighted_levels(df)
            levels['support'].extend(volume_levels['support'])
            levels['resistance'].extend(volume_levels['resistance'])
            
            # Method 4: Moving Average Levels
            ma_levels = self._calculate_ma_levels(closes)
            levels['support'].extend(ma_levels['support'])
            levels['resistance'].extend(ma_levels['resistance'])
            
            # Cluster and strengthen levels
            final_levels = self._cluster_and_strengthen_levels(levels)
            
            # Update storage
            self._update_sr_storage(final_levels)
            
            return final_levels
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 2620)
            print(f"ERROR Support/resistance calculation failed: {e}")
            return {'support': [], 'resistance': []}

    def _calculate_pivot_points(self, highs, lows, closes):
        """Calculate pivot point levels"""
        pivot_levels = {'support': [], 'resistance': []}
        
        for period in self.sr_config['pivot_periods']:
            if len(highs) < period:
                continue
            
            # Classic pivot point calculation
            pivot = (torch.max(highs[-period:]) + torch.min(lows[-period:]) + closes[-1]) / 3
            
            r1 = 2 * pivot - torch.min(lows[-period:])
            s1 = 2 * pivot - torch.max(highs[-period:])
            r2 = pivot + (torch.max(highs[-period:]) - torch.min(lows[-period:]))
            s2 = pivot - (torch.max(highs[-period:]) - torch.min(lows[-period:]))
            
            pivot_levels['resistance'].extend([r1.item(), r2.item()])
            pivot_levels['support'].extend([s1.item(), s2.item()])
        
        return pivot_levels

    def _calculate_swing_points(self, highs, lows):
        """Calculate support/resistance from swing points"""
        swing_levels = {'support': [], 'resistance': []}
        
        # Find recent swing highs (resistance)
        for i in range(5, len(highs) - 5):
            if (torch.all(highs[i] >= highs[i-5:i]) and 
                torch.all(highs[i] >= highs[i+1:i+6])):
                swing_levels['resistance'].append(highs[i].item())
        
        # Find recent swing lows (support)
        for i in range(5, len(lows) - 5):
            if (torch.all(lows[i] <= lows[i-5:i]) and 
                torch.all(lows[i] <= lows[i+1:i+6])):
                swing_levels['support'].append(lows[i].item())
        
        return swing_levels

    def _calculate_volume_weighted_levels(self, df):
        """Calculate volume-weighted support/resistance levels"""
        volume_levels = {'support': [], 'resistance': []}
        
        if len(df) < 20:
            return volume_levels
        
        # Use price levels with high volume as support/resistance
        price_bins = np.linspace(df['low'].min(), df['high'].max(), 50)
        volume_at_price = np.zeros_like(price_bins)
        
        for i in range(len(df)):
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            volume = df['volume'].iloc[i]
            
            for j, price in enumerate(price_bins):
                if low <= price <= high:
                    volume_at_price[j] += volume
        
        # Identify high volume nodes
        avg_volume = np.mean(volume_at_price)
        high_volume_threshold = avg_volume * 1.5
        
        for i in range(1, len(price_bins)-1):
            if (volume_at_price[i] > high_volume_threshold and
                volume_at_price[i] > volume_at_price[i-1] and
                volume_at_price[i] > volume_at_price[i+1]):
                
                # Classify as support or resistance based on position relative to current price
                current_price = df['close'].iloc[-1]
                level_type = 'support' if price_bins[i] < current_price else 'resistance'
                volume_levels[level_type].append(price_bins[i])
        
        return volume_levels

    def _calculate_ma_levels(self, closes):
        """Calculate moving average based support/resistance"""
        ma_levels = {'support': [], 'resistance': []}
        
        ma_periods = [20, 50, 100, 200]
        
        for period in ma_periods:
            if len(closes) >= period:
                ma = torch.mean(closes[-period:])
                ma_levels['support'].append(ma.item())
                ma_levels['resistance'].append(ma.item())
        
        return ma_levels

    def _cluster_and_strengthen_levels(self, levels):
        """Cluster similar levels and calculate strengths"""
        clustered_levels = {'support': [], 'resistance': []}
        
        for level_type in ['support', 'resistance']:
            level_values = levels[level_type]
            
            if not level_values:
                continue
            
            # Cluster similar levels
            clusters = []
            for value in level_values:
                found_cluster = False
                for cluster in clusters:
                    if abs(cluster['price'] - value) / cluster['price'] < self.sr_config['cluster_tolerance']:
                        cluster['values'].append(value)
                        cluster['count'] += 1
                        found_cluster = True
                        break
                
                if not found_cluster:
                    clusters.append({
                        'price': value,
                        'values': [value],
                        'count': 1,
                        'sources': 1
                    })
            
            # Calculate cluster strength
            for cluster in clusters:
                strength = min(cluster['count'] / 5.0, 1.0)  # Normalize strength
                
                if strength >= self.sr_config['level_strength_threshold']:
                    clustered_levels[level_type].append({
                        'price': cluster['price'],
                        'strength': strength,
                        'touch_count': cluster['count']
                    })
            
            # Sort by strength
            clustered_levels[level_type].sort(key=lambda x: x['strength'], reverse=True)
        
        return clustered_levels

    def _update_sr_storage(self, levels):
        """Update support/resistance level storage"""
        # Update support levels
        current_support = list(self.sr_levels['support_levels'])
        for new_level in levels['support']:
            self._add_or_update_level(current_support, new_level, 'support')
        
        self.sr_levels['support_levels'] = LinuxOptimizedDeque(maxlen=100)
        self.sr_levels['support_levels'].extend(current_support[:100])
        
        # Update resistance levels
        current_resistance = list(self.sr_levels['resistance_levels'])
        for new_level in levels['resistance']:
            self._add_or_update_level(current_resistance, new_level, 'resistance')
        
        self.sr_levels['resistance_levels'] = LinuxOptimizedDeque(maxlen=100)
        self.sr_levels['resistance_levels'].extend(current_resistance[:100])
        
        # Update strong levels (both support and resistance with high strength)
        strong_levels = []
        for level in current_support + current_resistance:
            if level['strength'] > 0.8:
                strong_levels.append(level)
        
        strong_levels.sort(key=lambda x: x['strength'], reverse=True)
        self.sr_levels['strong_levels'] = LinuxOptimizedDeque(maxlen=50)
        self.sr_levels['strong_levels'].extend(strong_levels[:50])

    def _add_or_update_level(self, current_levels, new_level, level_type):
        """Add or update support/resistance level"""
        tolerance = new_level['price'] * self.sr_config['cluster_tolerance']
        
        similar_level = None
        for level in current_levels:
            if abs(level['price'] - new_level['price']) <= tolerance:
                similar_level = level
                break
        
        if similar_level:
            # Update existing level
            similar_level['strength'] = max(similar_level['strength'], new_level['strength'])
            similar_level['touch_count'] += new_level['touch_count']
            similar_level['last_updated'] = time.time()
        else:
            # Add new level
            new_level['type'] = level_type
            new_level['first_seen'] = time.time()
            new_level['last_updated'] = time.time()
            current_levels.append(new_level)
        
        # Sort by strength
        current_levels.sort(key=lambda x: x['strength'], reverse=True)

    def get_sr_signals(self, current_price, psychology):
        """Get trading signals based on support/resistance levels"""
        signals = []
        
        # Support level bounce signals
        for support in list(self.sr_levels['support_levels'])[:10]:
            tolerance = support['price'] * 0.001
            if abs(current_price - support['price']) <= tolerance:
                if psychology['is_bullish'] and psychology['has_strong_rejection']:
                    confidence = 7.0 + (support['strength'] * 2.0)
                    signals.append(("CALL", min(confidence, 9.0),
                                  f"SUPPORT BOUNCE | Strength: {support['strength']:.2f} | Touches: {support['touch_count']}"))
        
        # Resistance level rejection signals
        for resistance in list(self.sr_levels['resistance_levels'])[:10]:
            tolerance = resistance['price'] * 0.001
            if abs(current_price - resistance['price']) <= tolerance:
                if psychology['is_bearish'] and psychology['has_strong_rejection']:
                    confidence = 7.0 + (resistance['strength'] * 2.0)
                    signals.append(("PUT", min(confidence, 9.0),
                                  f"RESISTANCE REJECTION | Strength: {resistance['strength']:.2f} | Touches: {resistance['touch_count']}"))
        
        # Strong level breakout signals
        for strong_level in list(self.sr_levels['strong_levels'])[:5]:
            tolerance = strong_level['price'] * 0.002  # Wider tolerance for breakouts
            
            # Bullish breakout above resistance
            if (strong_level['type'] == 'resistance' and 
                current_price > strong_level['price'] + tolerance and
                psychology['is_bullish'] and psychology['has_strong_momentum']):
                
                confidence = 8.0 + (strong_level['strength'] * 1.0)
                signals.append(("CALL", min(confidence, 9.0),
                              f"RESISTANCE BREAKOUT | Strength: {strong_level['strength']:.2f}"))
            
            # Bearish breakout below support
            elif (strong_level['type'] == 'support' and
                  current_price < strong_level['price'] - tolerance and
                  psychology['is_bearish'] and psychology['has_strong_momentum']):
                
                confidence = 8.0 + (strong_level['strength'] * 1.0)
                signals.append(("PUT", min(confidence, 9.0),
                              f"SUPPORT BREAKDOWN | Strength: {strong_level['strength']:.2f}"))
        
        return signals

    def get_nearest_levels(self, current_price):
        """Find the nearest dynamic support (below) and resistance (above) level."""
        sup_prices = [s['price'] for s in self.sr_levels.get('support_levels', []) if s['price'] < current_price]
        res_prices = [r['price'] for r in self.sr_levels.get('resistance_levels', []) if r['price'] > current_price]
        
        nearest_sup = max(sup_prices) if sup_prices else current_price * 0.99
        nearest_res = min(res_prices) if res_prices else current_price * 1.01
        
        return {'support': nearest_sup, 'resistance': nearest_res}

# 12. TrendAnalysisBrainGPU
@xgb_shadow('trend_analysis_brain', 'analyze_multi_timeframe_trends')
class TrendAnalysisBrainGPU:
    """Multi-timeframe Trend Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.trend_config = {
            'trend_periods': [20, 50, 100, 200],
            'timeframes': ['1min', '5min', '15min', '1h', '4h', '1d'],
            'trend_strength_threshold': 0.6,
            'gpu_batch_size': 512
        }
        
        self.trend_data = {
            'current_trends': {},
            'trend_strengths': LinuxOptimizedDeque(500),
            'trend_changes': LinuxOptimizedDeque(200),
            'multi_tf_alignment': LinuxOptimizedDeque(100)
        }
        
        # Initialize trend data for each timeframe
        for tf in self.trend_config['timeframes']:
            self.trend_data['current_trends'][tf] = {
                'direction': 'SIDEWAYS',
                'strength': 0.5,
                'slope': 0.0,
                'last_update': 0
            }
        
        self.gpu_buffers = {}
        self._init_trend_gpu_buffers()
        
        print(f"OK Trend Analysis Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_trend_gpu_buffers(self):
        """Initialize GPU buffers for trend analysis"""
        self.gpu_buffers['trend_calc'] = torch.zeros(
            self.trend_config['gpu_batch_size'], device=self.device
        )

    def analyze_multi_timeframe_trends(self, df_dict):
        """Analyze trends across multiple timeframes"""
        try:
            trend_analysis = {}
            
            for timeframe, df in df_dict.items():
                if timeframe not in self.trend_config['timeframes']:
                    continue
                
                if len(df) < 50:
                    trend_analysis[timeframe] = self.trend_data['current_trends'][timeframe]
                    continue
                
                # Analyze trend for this timeframe
                trend_info = self._analyze_single_timeframe_trend(df, timeframe)
                trend_analysis[timeframe] = trend_info
                
                # Update storage
                self.trend_data['current_trends'][timeframe] = trend_info
            
            # Analyze multi-timeframe alignment
            alignment = self._analyze_multi_tf_alignment(trend_analysis)
            trend_analysis['alignment'] = alignment
            
            # Update alignment storage
            self.trend_data['multi_tf_alignment'].append({
                'alignment': alignment,
                'timestamp': time.time()
            })
            
            return trend_analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 2930)
            print(f"ERROR Multi-timeframe trend analysis failed: {e}")
            return {}

    def _analyze_single_timeframe_trend(self, df, timeframe):
        import quantitative_math as qm
        closes = df['close'] if hasattr(df, 'columns') and 'close' in df.columns else []
        evidence = qm.close_return_trend(closes)
        now = time.time()
        if not evidence.get('available'):
            return {'direction': 'SIDEWAYS', 'strength': 0.5, 'slope': 0.0, 'last_update': now,
                    'evidence_status': 'insufficient_or_invalid_ohlcv'}
        score = float(evidence.get('trend_score', 0.0))
        direction = 'BULLISH' if score >= 1.0 else 'BEARISH' if score <= -1.0 else 'SIDEWAYS'
        strength = min(1.0, abs(score) / 5.0) if direction != 'SIDEWAYS' else 0.5
        return {'direction': direction, 'strength': strength,
                'slope': float(evidence.get('slope', 0.0)), 'last_update': now,
                'evidence_status': 'robust_log_return_trend'}

    def _analyze_multi_tf_alignment(self, trend_analysis):
        """Analyze alignment of trends across timeframes"""
        if not trend_analysis:
            return 'NEUTRAL'
        
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        
        for timeframe, trend in trend_analysis.items():
            if timeframe == 'alignment':
                continue
                
            if trend['direction'] == 'BULLISH':
                bullish_count += 1
                total_strength += trend['strength']
            elif trend['direction'] == 'BEARISH':
                bearish_count += 1
                total_strength += trend['strength']
        
        total_tfs = len(trend_analysis) - 1  # Exclude alignment key
        
        if total_tfs == 0:
            return 'NEUTRAL'
        
        if bullish_count > bearish_count * 1.5:  # Significant majority
            return 'STRONG_BULLISH'
        elif bearish_count > bullish_count * 1.5:
            return 'STRONG_BEARISH'
        elif bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def get_trend_signals(self, current_price, psychology, multi_tf_alignment):
        """Get trading signals based on trend analysis"""
        signals = []
        
        # Trend-following signals
        if multi_tf_alignment in ['BULLISH', 'STRONG_BULLISH']:
            if psychology['is_bullish'] and psychology['has_strong_momentum']:
                strength_bonus = 2.0 if multi_tf_alignment == 'STRONG_BULLISH' else 1.0
                signals.append(("CALL", 7.5 + strength_bonus,
                              f"TREND FOLLOWING BULLISH | Alignment: {multi_tf_alignment}"))
        
        elif multi_tf_alignment in ['BEARISH', 'STRONG_BEARISH']:
            if psychology['is_bearish'] and psychology['has_strong_momentum']:
                strength_bonus = 2.0 if multi_tf_alignment == 'STRONG_BEARISH' else 1.0
                signals.append(("PUT", 7.5 + strength_bonus,
                              f"TREND FOLLOWING BEARISH | Alignment: {multi_tf_alignment}"))
        
        # Counter-trend signals (only in strong alignment scenarios)
        if multi_tf_alignment == 'STRONG_BULLISH':
            # Look for bearish reversals at extreme levels
            if psychology['is_bearish'] and psychology['has_strong_rejection']:
                signals.append(("PUT", 6.5, "COUNTER-TREND BEARISH IN STRONG BULLISH TREND"))
        
        elif multi_tf_alignment == 'STRONG_BEARISH':
            # Look for bullish reversals at extreme levels
            if psychology['is_bullish'] and psychology['has_strong_rejection']:
                signals.append(("CALL", 6.5, "COUNTER-TREND BULLISH IN STRONG BEARISH TREND"))
        
        return signals

# 13. MarketRegimeBrainGPU
@xgb_shadow('market_regime_brain', 'detect_market_regime')
class MarketRegimeBrainGPU:
    """Market Condition and Regime Detection"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.regime_config = {
            'regime_periods': [50, 100, 200],
            'volatility_thresholds': {'LOW': 0.1, 'NORMAL': 0.25, 'HIGH': 0.5},
            'trend_thresholds': {'SIDEWAYS': 0.3, 'TRENDING': 0.7},
            'volume_regime_threshold': 1.5,
            'gpu_batch_size': 512
        }
        
        self.regime_data = {
            'current_regime': 'NORMAL_VOL_SIDEWAYS',
            'regime_history': LinuxOptimizedDeque(500),
            'regime_changes': LinuxOptimizedDeque(100),
            'regime_statistics': defaultdict(lambda: LinuxOptimizedDeque(100))
        }
        
        self.gpu_buffers = {}
        self._init_regime_gpu_buffers()
        
        print(f"OK Market Regime Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_regime_gpu_buffers(self):
        """Initialize GPU buffers for regime detection"""
        self.gpu_buffers['regime_calc'] = torch.zeros(
            self.regime_config['gpu_batch_size'], device=self.device
        )

    def detect_market_regime(self, df, volume_data=None):
        """Detect current market regime"""
        try:
            if len(df) < 100:
                return self.regime_data['current_regime']
            
            # Move data to GPU
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            
            regime_analysis = {}
            
            # 1. Volatility Regime
            regime_analysis['volatility'] = self._detect_volatility_regime(closes)
            
            # 2. Trend Regime
            regime_analysis['trend'] = self._detect_trend_regime(closes)
            
            # 3. Volume Regime
            if volume_data is not None and len(volume_data) >= 50:
                regime_analysis['volume'] = self._detect_volume_regime(volume_data)
            else:
                regime_analysis['volume'] = 'NORMAL_VOLUME'
            
            # 4. Combine into comprehensive regime
            comprehensive_regime = self._combine_regimes(regime_analysis)
            regime_analysis['comprehensive'] = comprehensive_regime
            
            # Update storage
            self._update_regime_data(regime_analysis)
            
            return comprehensive_regime
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 3082)
            print(f"ERROR Market regime detection failed: {e}")
            return self.regime_data['current_regime']

    def _detect_volatility_regime(self, closes):
        """Detect volatility-based regime"""
        if len(closes) < 50:
            return 'NORMAL_VOLATILITY'
        
        # Calculate annualized volatility
        returns = torch.diff(closes) / closes[:-1]
        volatility = torch.std(returns) * np.sqrt(252)
        
        vol_thresholds = self.regime_config['volatility_thresholds']
        
        if volatility < vol_thresholds['LOW']:
            return 'LOW_VOLATILITY'
        elif volatility < vol_thresholds['NORMAL']:
            return 'NORMAL_VOLATILITY'
        elif volatility < vol_thresholds['HIGH']:
            return 'HIGH_VOLATILITY'
        else:
            return 'EXTREME_VOLATILITY'

    def _detect_trend_regime(self, closes):
        """Detect trend-based regime"""
        if len(closes) < 100:
            return 'SIDEWAYS'
        
        # Calculate trend strength using multiple methods
        ma_20 = torch.mean(closes[-20:])
        ma_50 = torch.mean(closes[-50:])
        ma_100 = torch.mean(closes[-100:])
        
        # Method 1: Moving average alignment
        ma_alignment = (ma_20 > ma_50 > ma_100) or (ma_20 < ma_50 < ma_100)
        
        # Method 2: ADX-like trend strength (simplified)
        high_low_range = torch.mean(torch.abs(closes[-20:] - torch.mean(closes[-20:])))
        trend_strength = high_low_range / torch.mean(closes[-20:]) if torch.mean(closes[-20:]) > 0 else 0
        
        trend_thresholds = self.regime_config['trend_thresholds']
        
        if ma_alignment and trend_strength > trend_thresholds['TRENDING']:
            if ma_20 > ma_50:
                return 'STRONG_UPTREND'
            else:
                return 'STRONG_DOWNTREND'
        elif trend_strength > trend_thresholds['TRENDING']:
            if ma_20 > ma_50:
                return 'UPTREND'
            else:
                return 'DOWNTREND'
        elif trend_strength < trend_thresholds['SIDEWAYS']:
            return 'SIDEWAYS'
        else:
            return 'RANGING'

    def _detect_volume_regime(self, volume_data):
        """Detect volume-based regime"""
        if len(volume_data) < 50:
            return 'NORMAL_VOLUME'
        
        volumes = torch.tensor(volume_data, device=self.device, dtype=torch.float32)
        
        current_volume = volumes[-1]
        avg_volume = torch.mean(volumes[-50:])
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > self.regime_config['volume_regime_threshold']:
            return 'HIGH_VOLUME'
        elif volume_ratio < 1.0 / self.regime_config['volume_regime_threshold']:
            return 'LOW_VOLUME'
        else:
            return 'NORMAL_VOLUME'

    def _combine_regimes(self, regime_analysis):
        """Combine individual regimes into comprehensive market regime"""
        volatility = regime_analysis['volatility']
        trend = regime_analysis['trend']
        volume = regime_analysis['volume']
        
        # Simple combination logic
        if 'STRONG' in trend and 'HIGH' in volatility:
            return 'TRENDING_HIGH_VOL'
        elif 'STRONG' in trend and 'NORMAL' in volatility:
            return 'TRENDING_NORMAL_VOL'
        elif 'STRONG' in trend and 'LOW' in volatility:
            return 'TRENDING_LOW_VOL'
        elif 'SIDEWAYS' in trend and 'HIGH' in volatility:
            return 'VOLATILE_SIDEWAYS'
        elif 'SIDEWAYS' in trend and 'NORMAL' in volatility:
            return 'NORMAL_SIDEWAYS'
        elif 'SIDEWAYS' in trend and 'LOW' in volatility:
            return 'LOW_VOL_SIDEWAYS'
        else:
            return 'MIXED_REGIME'

    def _update_regime_data(self, regime_analysis):
        """Update regime data storage"""
        current_regime = regime_analysis['comprehensive']
        previous_regime = self.regime_data['current_regime']
        
        self.regime_data['current_regime'] = current_regime
        
        # Record regime history
        self.regime_data['regime_history'].append({
            'regime': current_regime,
            'timestamp': time.time(),
            'volatility': regime_analysis['volatility'],
            'trend': regime_analysis['trend'],
            'volume': regime_analysis.get('volume', 'UNKNOWN')
        })
        
        # Record regime changes
        if current_regime != previous_regime:
            self.regime_data['regime_changes'].append({
                'from': previous_regime,
                'to': current_regime,
                'timestamp': time.time()
            })

    def get_regime_signals(self, current_price, psychology, current_regime):
        """Get trading signals based on market regime"""
        signals = []
        
        # Regime-specific trading strategies
        if 'TRENDING' in current_regime:
            # Trend-following strategies
            if 'UPTREND' in current_regime and psychology['is_bullish']:
                confidence = 8.0 if 'STRONG' in current_regime else 7.0
                signals.append(("CALL", confidence, f"TREND-FOLLOWING IN {current_regime}"))
            
            elif 'DOWNTREND' in current_regime and psychology['is_bearish']:
                confidence = 8.0 if 'STRONG' in current_regime else 7.0
                signals.append(("PUT", confidence, f"TREND-FOLLOWING IN {current_regime}"))
        
        elif 'SIDEWAYS' in current_regime:
            # Range-bound strategies
            if psychology['has_strong_rejection']:
                if psychology['lower_wick_ratio'] > psychology['upper_wick_ratio']:
                    confidence = 7.5 if 'LOW_VOL' in current_regime else 7.0
                    signals.append(("CALL", confidence, f"RANGE SUPPORT IN {current_regime}"))
                else:
                    confidence = 7.5 if 'LOW_VOL' in current_regime else 7.0
                    signals.append(("PUT", confidence, f"RANGE RESISTANCE IN {current_regime}"))
        
        elif 'VOLATILE' in current_regime:
            # High volatility strategies - more cautious
            if psychology['has_strong_momentum'] and psychology['body_ratio'] > 0.7:
                confidence = 6.5  # Lower confidence in high volatility
                if psychology['is_bullish']:
                    signals.append(("CALL", confidence, f"MOMENTUM IN {current_regime} (CAUTION)"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", confidence, f"MOMENTUM IN {current_regime} (CAUTION)"))
        
        # Regime change anticipation
        recent_changes = list(self.regime_data['regime_changes'])[-3:]
        if len(recent_changes) >= 2:
            # Multiple recent changes indicate unstable regime
            if psychology['has_strong_momentum']:
                confidence = 7.0
                if psychology['is_bullish']:
                    signals.append(("CALL", confidence, "REGIME INSTABILITY BULLISH BREAKOUT"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", confidence, "REGIME INSTABILITY BEARISH BREAKDOWN"))
        
        return signals

# 14. PriceActionBrainGPU
@xgb_shadow('price_action_brain', 'analyze_price_action')
class PriceActionBrainGPU:
    """Pure Price Action Analysis"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pa_config = {
            'pin_bar_threshold': 0.6,
            'inside_bar_requirement': 0.8,
            'outside_bar_multiplier': 1.2,
            'engulfing_ratio': 1.5,
            'gpu_batch_size': 512
        }
        
        self.pa_patterns = {
            'pin_bars': LinuxOptimizedDeque(200),
            'inside_bars': LinuxOptimizedDeque(200),
            'outside_bars': LinuxOptimizedDeque(200),
            'engulfing_bars': LinuxOptimizedDeque(200),
            'rejection_clusters': LinuxOptimizedDeque(100)
        }
        
        self.gpu_buffers = {}
        self._init_pa_gpu_buffers()
        
        print(f"OK Price Action Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_pa_gpu_buffers(self):
        """Initialize GPU buffers for price action analysis"""
        self.gpu_buffers['pa_calc'] = torch.zeros(
            self.pa_config['gpu_batch_size'], device=self.device
        )

    def analyze_price_action(self, df):
        """Analyze pure price action patterns"""
        try:
            if len(df) < 10:
                return []
            
            patterns_found = []
            
            # Move data to GPU
            highs = torch.tensor(df['high'].values, device=self.device, dtype=torch.float32)
            lows = torch.tensor(df['low'].values, device=self.device, dtype=torch.float32)
            closes = torch.tensor(df['close'].values, device=self.device, dtype=torch.float32)
            opens = torch.tensor(df['open'].values, device=self.device, dtype=torch.float32)
            
            # Analyze recent candles for patterns
            for i in range(max(1, len(df)-10), len(df)):
                current_candle = {
                    'high': highs[i].item(),
                    'low': lows[i].item(), 
                    'close': closes[i].item(),
                    'open': opens[i].item()
                }
                
                if i > 0:
                    previous_candle = {
                        'high': highs[i-1].item(),
                        'low': lows[i-1].item(),
                        'close': closes[i-1].item(),
                        'open': opens[i-1].item()
                    }
                else:
                    previous_candle = None
                
                # Detect various price action patterns
                patterns_found.extend(self._detect_pin_bars(current_candle))
                patterns_found.extend(self._detect_inside_bars(current_candle, previous_candle))
                patterns_found.extend(self._detect_outside_bars(current_candle, previous_candle))
                patterns_found.extend(self._detect_engulfing_bars(current_candle, previous_candle))
            
            # Update pattern storage
            self._update_pa_patterns(patterns_found)
            
            return patterns_found
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 3329)
            print(f"ERROR Price action analysis failed: {e}")
            return []

    def _detect_pin_bars(self, candle):
        """Detect pin bar (rejection) patterns"""
        patterns = []
        
        high = candle['high']
        low = candle['low']
        close = candle['close']
        open_price = candle['open']
        
        total_range = high - low
        if total_range <= 0:
            return patterns
        
        body_size = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low
        
        # Bullish pin bar (hammer)
        if (lower_wick >= total_range * self.pa_config['pin_bar_threshold'] and
            body_size <= total_range * 0.3 and
            upper_wick <= total_range * 0.1):
            
            patterns.append({
                'type': 'BULLISH_PIN_BAR',
                'price': close,
                'wick_ratio': lower_wick / total_range,
                'timestamp': time.time()
            })
        
        # Bearish pin bar (shooting star)
        elif (upper_wick >= total_range * self.pa_config['pin_bar_threshold'] and
              body_size <= total_range * 0.3 and
              lower_wick <= total_range * 0.1):
            
            patterns.append({
                'type': 'BEARISH_PIN_BAR',
                'price': close,
                'wick_ratio': upper_wick / total_range,
                'timestamp': time.time()
            })
        
        return patterns

    def _detect_inside_bars(self, current_candle, previous_candle):
        """Detect inside bar patterns"""
        patterns = []
        
        if previous_candle is None:
            return patterns
        
        # Current candle is inside previous candle's range
        if (current_candle['high'] <= previous_candle['high'] and
            current_candle['low'] >= previous_candle['low'] and
            (current_candle['high'] - current_candle['low']) <= 
            (previous_candle['high'] - previous_candle['low']) * self.pa_config['inside_bar_requirement']):
            
            patterns.append({
                'type': 'INSIDE_BAR',
                'price': current_candle['close'],
                'mother_high': previous_candle['high'],
                'mother_low': previous_candle['low'],
                'timestamp': time.time()
            })
        
        return patterns

    def _detect_outside_bars(self, current_candle, previous_candle):
        """Detect outside bar patterns"""
        patterns = []
        
        if previous_candle is None:
            return patterns
        
        # Current candle engulfs previous candle's range
        if (current_candle['high'] > previous_candle['high'] and
            current_candle['low'] < previous_candle['low'] and
            (current_candle['high'] - current_candle['low']) >= 
            (previous_candle['high'] - previous_candle['low']) * self.pa_config['outside_bar_multiplier']):
            
            patterns.append({
                'type': 'OUTSIDE_BAR',
                'price': current_candle['close'],
                'engulfing_ratio': (current_candle['high'] - current_candle['low']) / 
                                  max(previous_candle['high'] - previous_candle['low'], 1e-8),
                'timestamp': time.time()
            })
        
        return patterns

    def _detect_engulfing_bars(self, current_candle, previous_candle):
        """Detect engulfing bar patterns"""
        patterns = []
        
        if previous_candle is None:
            return patterns
        
        current_body = abs(current_candle['close'] - current_candle['open'])
        previous_body = abs(previous_candle['close'] - previous_candle['open'])
        
        if current_body <= previous_body * 0.1:  # Avoid very small bodies
            return patterns
        
        # Bullish engulfing
        if (current_candle['close'] > current_candle['open'] and  # Current bullish
            previous_candle['close'] < previous_candle['open'] and  # Previous bearish
            current_candle['open'] < previous_candle['close'] and
            current_candle['close'] > previous_candle['open'] and
            current_body > previous_body * self.pa_config['engulfing_ratio']):
            
            patterns.append({
                'type': 'BULLISH_ENGULFING',
                'price': current_candle['close'],
                'engulfing_strength': current_body / max(previous_body, 1e-8),
                'timestamp': time.time()
            })
        
        # Bearish engulfing
        elif (current_candle['close'] < current_candle['open'] and  # Current bearish
              previous_candle['close'] > previous_candle['open'] and  # Previous bullish
              current_candle['open'] > previous_candle['close'] and
              current_candle['close'] < previous_candle['open'] and
              current_body > previous_body * self.pa_config['engulfing_ratio']):
            
            patterns.append({
                'type': 'BEARISH_ENGULFING',
                'price': current_candle['close'],
                'engulfing_strength': current_body / max(previous_body, 1e-8),
                'timestamp': time.time()
            })
        
        return patterns

    def _update_pa_patterns(self, new_patterns):
        """Update price action pattern storage"""
        for pattern in new_patterns:
            pattern_type = pattern['type']
            
            if 'BULLISH' in pattern_type:
                if 'PIN_BAR' in pattern_type:
                    self.pa_patterns['pin_bars'].append(pattern)
                elif 'ENGULFING' in pattern_type:
                    self.pa_patterns['engulfing_bars'].append(pattern)
            
            elif 'BEARISH' in pattern_type:
                if 'PIN_BAR' in pattern_type:
                    self.pa_patterns['pin_bars'].append(pattern)
                elif 'ENGULFING' in pattern_type:
                    self.pa_patterns['engulfing_bars'].append(pattern)
            
            elif 'INSIDE_BAR' in pattern_type:
                self.pa_patterns['inside_bars'].append(pattern)
            
            elif 'OUTSIDE_BAR' in pattern_type:
                self.pa_patterns['outside_bars'].append(pattern)

    def get_pa_signals(self, current_price, psychology):
        """Get trading signals based on price action patterns"""
        signals = []
        current_time = time.time()
        
        # Recent bullish pin bars
        recent_bullish_pins = [pat for pat in list(self.pa_patterns['pin_bars'])[-5:]
                             if pat['type'] == 'BULLISH_PIN_BAR' and current_time - pat['timestamp'] < 3600]
        
        for pin in recent_bullish_pins:
            if (psychology['is_bullish'] and 
                psychology['lower_wick_ratio'] > 0.3 and
                abs(current_price - pin['price']) / pin['price'] < 0.002):
                
                confidence = 7.0 + (pin['wick_ratio'] * 2.0)
                signals.append(("CALL", min(confidence, 9.0),
                              f"BULLISH PIN BAR CONFIRMATION | Wick Ratio: {pin['wick_ratio']:.2f}"))
        
        # Recent bearish pin bars
        recent_bearish_pins = [pat for pat in list(self.pa_patterns['pin_bars'])[-5:]
                             if pat['type'] == 'BEARISH_PIN_BAR' and current_time - pat['timestamp'] < 3600]
        
        for pin in recent_bearish_pins:
            if (psychology['is_bearish'] and
                psychology['upper_wick_ratio'] > 0.3 and
                abs(current_price - pin['price']) / pin['price'] < 0.002):
                
                confidence = 7.0 + (pin['wick_ratio'] * 2.0)
                signals.append(("PUT", min(confidence, 9.0),
                              f"BEARISH PIN BAR CONFIRMATION | Wick Ratio: {pin['wick_ratio']:.2f}"))
        
        # Recent engulfing patterns
        recent_bullish_engulfing = [pat for pat in list(self.pa_patterns['engulfing_bars'])[-5:]
                                  if pat['type'] == 'BULLISH_ENGULFING' and current_time - pat['timestamp'] < 3600]
        
        for engulf in recent_bullish_engulfing:
            if psychology['is_bullish'] and psychology['body_ratio'] > 0.6:
                confidence = 7.5 + min(engulf['engulfing_strength'] * 0.5, 1.5)
                signals.append(("CALL", min(confidence, 9.0),
                              f"BULLISH ENGULFING FOLLOW-THROUGH | Strength: {engulf['engulfing_strength']:.2f}"))
        
        recent_bearish_engulfing = [pat for pat in list(self.pa_patterns['engulfing_bars'])[-5:]
                                  if pat['type'] == 'BEARISH_ENGULFING' and current_time - pat['timestamp'] < 3600]
        
        for engulf in recent_bearish_engulfing:
            if psychology['is_bearish'] and psychology['body_ratio'] > 0.6:
                confidence = 7.5 + min(engulf['engulfing_strength'] * 0.5, 1.5)
                signals.append(("PUT", min(confidence, 9.0),
                              f"BEARISH ENGULFING FOLLOW-THROUGH | Strength: {engulf['engulfing_strength']:.2f}"))
        
        # Inside bar breakouts
        recent_inside_bars = [pat for pat in list(self.pa_patterns['inside_bars'])[-5:]
                            if current_time - pat['timestamp'] < 3600]
        
        for inside in recent_inside_bars:
            # Breakout above mother high
            if current_price > inside['mother_high'] and psychology['is_bullish']:
                signals.append(("CALL", 8.0, "INSIDE BAR BULLISH BREAKOUT"))
            # Breakout below mother low
            elif current_price < inside['mother_low'] and psychology['is_bearish']:
                signals.append(("PUT", 8.0, "INSIDE BAR BEARISH BREAKOUT"))
        
        return signals

# 15. InstitutionalFlowBrainGPU
@xgb_shadow('inst_flow_brain', 'analyze_institutional_flow')
class InstitutionalFlowBrainGPU:
    """Smart Money and Institutional Flow Tracking"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.flow_config = {
            'large_trade_threshold': 100000,  # $100k minimum for institutional
            'accumulation_periods': [20, 50],
            'distribution_detection': 0.7,
            'gpu_batch_size': 256
        }
        
        self.institutional_data = {
            'large_trades': LinuxOptimizedDeque(1000),
            'accumulation_zones': LinuxOptimizedDeque(100),
            'distribution_zones': LinuxOptimizedDeque(100),
            'smart_money_signals': LinuxOptimizedDeque(200)
        }
        
        self.gpu_buffers = {}
        self._init_flow_gpu_buffers()
        
        print(f"OK Institutional Flow Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_flow_gpu_buffers(self):
        """Initialize GPU buffers for institutional flow analysis"""
        self.gpu_buffers['flow_calc'] = torch.zeros(
            self.flow_config['gpu_batch_size'], device=self.device
        )

    def analyze_institutional_flow(self, df, trade_data=None):
        """Analyze institutional order flow and smart money activity"""
        try:
            flow_analysis = {}
            
            # Method 1: Volume analysis for large trades
            if trade_data is not None:
                large_trades = self._identify_large_trades(trade_data)
                flow_analysis['large_trades'] = large_trades
            
            # Method 2: Price-volume divergence for accumulation/distribution
            flow_analysis['accumulation'] = self._detect_accumulation(df)
            flow_analysis['distribution'] = self._detect_distribution(df)
            
            # Method 3: Unusual options activity (simulated)
            flow_analysis['unusual_activity'] = self._detect_unusual_activity(df)
            
            # Update storage
            self._update_institutional_data(flow_analysis)
            
            return flow_analysis
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 3606)
            print(f"ERROR Institutional flow analysis failed: {e}")
            return {}

    def _identify_large_trades(self, trade_data):
        """Identify large trades potentially from institutions"""
        large_trades = []
        
        for trade in trade_data[-100:]:  # Recent trades
            if trade.get('size', 0) * trade.get('price', 0) >= self.flow_config['large_trade_threshold']:
                large_trades.append({
                    'price': trade['price'],
                    'size': trade['size'],
                    'value': trade['size'] * trade['price'],
                    'timestamp': trade.get('timestamp', time.time()),
                    'side': trade.get('side', 'UNKNOWN')
                })
        
        return large_trades

    def _detect_accumulation(self, df):
        """Detect accumulation patterns (smart money buying)"""
        accumulation_signals = []
        
        if len(df) < 50:
            return accumulation_signals
        
        # Look for price stability or slight decline with increasing volume
        recent_prices = df['close'].tail(20)
        recent_volumes = df['volume'].tail(20)
        
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Accumulation: stable/slightly down price with increasing volume
        if abs(price_trend) < np.std(recent_prices) * 0.5 and volume_trend > 0:
            confidence = min(volume_trend / np.mean(recent_volumes) * 10, 1.0)
            
            accumulation_signals.append({
                'type': 'ACCUMULATION',
                'price_level': np.mean(recent_prices),
                'confidence': confidence,
                'timestamp': time.time()
            })
        
        return accumulation_signals

    def _detect_distribution(self, df):
        """Detect distribution patterns (smart money selling)"""
        distribution_signals = []
        
        if len(df) < 50:
            return distribution_signals
        
        # Look for price stability or slight rise with decreasing volume
        recent_prices = df['close'].tail(20)
        recent_volumes = df['volume'].tail(20)
        
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Distribution: stable/slightly up price with decreasing volume
        if price_trend > 0 and volume_trend < 0:
            confidence = min(abs(volume_trend) / np.mean(recent_volumes) * 10, 1.0)
            
            distribution_signals.append({
                'type': 'DISTRIBUTION',
                'price_level': np.mean(recent_prices),
                'confidence': confidence,
                'timestamp': time.time()
            })
        
        return distribution_signals

    def _detect_unusual_activity(self, df):
        """Detect unusual trading activity"""
        unusual_signals = []
        
        if len(df) < 20:
            return unusual_signals
        
        # Look for abnormal volume spikes
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        if current_volume > avg_volume * 3:  # 3x average volume
            unusual_signals.append({
                'type': 'HIGH_VOLUME_SPIKE',
                'volume_ratio': current_volume / avg_volume,
                'timestamp': time.time()
            })
        
        # Look for large range expansion
        current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        avg_range = (df['high'].tail(20) - df['low'].tail(20)).mean()
        
        if current_range > avg_range * 2:  # 2x average range
            unusual_signals.append({
                'type': 'LARGE_RANGE_EXPANSION',
                'range_ratio': current_range / avg_range,
                'timestamp': time.time()
            })
        
        return unusual_signals

    def _update_institutional_data(self, flow_analysis):
        """Update institutional data storage"""
        current_time = time.time()
        
        # Update large trades
        for trade in flow_analysis.get('large_trades', []):
            self.institutional_data['large_trades'].append(trade)
        
        # Update accumulation zones
        for accumulation in flow_analysis.get('accumulation', []):
            self.institutional_data['accumulation_zones'].append(accumulation)
        
        # Update distribution zones
        for distribution in flow_analysis.get('distribution', []):
            self.institutional_data['distribution_zones'].append(distribution)
        
        # Create smart money signals
        smart_signals = []
        
        # Accumulation signals
        recent_accumulation = [acc for acc in flow_analysis.get('accumulation', [])
                             if current_time - acc['timestamp'] < 3600]
        
        for acc in recent_accumulation:
            smart_signals.append({
                'type': 'SMART_MONEY_BUYING',
                'price_level': acc['price_level'],
                'confidence': acc['confidence'],
                'timestamp': current_time
            })
        
        # Distribution signals
        recent_distribution = [dist for dist in flow_analysis.get('distribution', [])
                             if current_time - dist['timestamp'] < 3600]
        
        for dist in recent_distribution:
            smart_signals.append({
                'type': 'SMART_MONEY_SELLING',
                'price_level': dist['price_level'],
                'confidence': dist['confidence'],
                'timestamp': current_time
            })
        
        # Unusual activity signals
        for unusual in flow_analysis.get('unusual_activity', []):
            smart_signals.append({
                'type': unusual['type'],
                'strength': unusual.get('volume_ratio', unusual.get('range_ratio', 1.0)),
                'timestamp': current_time
            })
        
        # Update smart money signals
        for signal in smart_signals:
            self.institutional_data['smart_money_signals'].append(signal)

    def get_institutional_signals(self, current_price, psychology):
        """Get trading signals based on institutional flow"""
        signals = []
        current_time = time.time()
        
        # Recent smart money buying
        recent_buying = [sig for sig in list(self.institutional_data['smart_money_signals'])[-5:]
                       if sig['type'] == 'SMART_MONEY_BUYING' and current_time - sig['timestamp'] < 3600]
        
        for buying in recent_buying:
            if (psychology['is_bullish'] and 
                abs(current_price - buying['price_level']) / buying['price_level'] < 0.002):
                
                confidence = 7.5 + (buying['confidence'] * 1.5)
                signals.append(("CALL", min(confidence, 9.0),
                              f"SMART MONEY ACCUMULATION | Confidence: {buying['confidence']:.2f}"))
        
        # Recent smart money selling
        recent_selling = [sig for sig in list(self.institutional_data['smart_money_signals'])[-5:]
                        if sig['type'] == 'SMART_MONEY_SELLING' and current_time - sig['timestamp'] < 3600]
        
        for selling in recent_selling:
            if (psychology['is_bearish'] and
                abs(current_price - selling['price_level']) / selling['price_level'] < 0.002):
                
                confidence = 7.5 + (selling['confidence'] * 1.5)
                signals.append(("PUT", min(confidence, 9.0),
                              f"SMART MONEY DISTRIBUTION | Confidence: {selling['confidence']:.2f}"))
        
        # Unusual activity signals
        recent_unusual = [sig for sig in list(self.institutional_data['smart_money_signals'])[-5:]
                        if ('SPIKE' in sig['type'] or 'EXPANSION' in sig['type']) and current_time - sig['timestamp'] < 3600]
        
        for unusual in recent_unusual:
            if unusual['type'] == 'HIGH_VOLUME_SPIKE':
                if psychology['is_bullish']:
                    signals.append(("CALL", 8.0, "HIGH VOLUME BREAKOUT BULLISH"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", 8.0, "HIGH VOLUME BREAKDOWN BEARISH"))
            
            elif unusual['type'] == 'LARGE_RANGE_EXPANSION':
                # Large range often precedes sustained moves
                if psychology['is_bullish']:
                    signals.append(("CALL", 7.5, "RANGE EXPANSION BULLISH FOLLOW-THROUGH"))
                elif psychology['is_bearish']:
                    signals.append(("PUT", 7.5, "RANGE EXPANSION BEARISH FOLLOW-THROUGH"))
        
        return signals

# 16. SignalFusionBrainGPU
@xgb_shadow('signal_fusion_brain', 'fuse_signals')
class SignalFusionBrainGPU:
    """Multi-Brain Signal Fusion and Confidence Weighting"""
    def __init__(self, master_system):
        self.master = master_system
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.fusion_config = {
            'minimum_confidence': 6.0,
            'corroboration_bonus': 0.5,
            'contradiction_penalty': 1.0,
            'recent_signal_window': 300,  # 5 minutes
            'gpu_batch_size': 512
        }
        
        self.fusion_data = {
            'recent_signals': LinuxOptimizedDeque(1000),
            'signal_performance': defaultdict(lambda: LinuxOptimizedDeque(500)),
            'brain_weights': defaultdict(lambda: 1.0),  # Dynamic brain weighting
            'fused_decisions': LinuxOptimizedDeque(500)
        }
        
        # Initialize brain weights
        brains = [
            'ZonePointFiveDetectorGPU', 'CandlePsychologyMasterGPU', 'VolumeProfileBrainGPU',
            'MarketStructureBrainGPU', 'OrderFlowBrainGPU', 'MomentumOscillatorBrainGPU',
            'VolatilityRegimeBrainGPU', 'CycleAnalysisBrainGPU', 'CorrelationMatrixBrainGPU',
            'PatternRecognitionBrainGPU', 'SupportResistanceBrainGPU', 'TrendAnalysisBrainGPU',
            'MarketRegimeBrainGPU', 'PriceActionBrainGPU', 'InstitutionalFlowBrainGPU'
        ]
        
        for brain in brains:
            self.fusion_data['brain_weights'][brain] = 1.0
        
        self.gpu_buffers = {}
        self._init_fusion_gpu_buffers()
        
        print(f"OK Signal Fusion Brain Initialized (GPU: {_safe_get_device_name(self.device)})")

    def _init_fusion_gpu_buffers(self):
        """Initialize GPU buffers for signal fusion"""
        self.gpu_buffers['fusion_calc'] = torch.zeros(
            self.fusion_config['gpu_batch_size'], device=self.device
        )

    def fuse_signals(self, all_signals, current_price, market_context):
        """Fuse signals from all brains with intelligent weighting"""
        try:
            fused_signals = []
            
            # Group signals by type (CALL/PUT)
            call_signals = [sig for sig in all_signals if sig[0] == "CALL"]
            put_signals = [sig for sig in all_signals if sig[0] == "PUT"]
            
            # Fuse CALL signals
            if call_signals:
                fused_call = self._fuse_single_direction(call_signals, "CALL", current_price, market_context)
                if fused_call:
                    fused_signals.append(fused_call)
            
            # Fuse PUT signals
            if put_signals:
                fused_put = self._fuse_single_direction(put_signals, "PUT", current_price, market_context)
                if fused_put:
                    fused_signals.append(fused_put)
            
            # Update performance tracking
            self._update_signal_performance(all_signals)
            
            # Update brain weights based on recent performance
            self._update_brain_weights(all_signals)
            
            return fused_signals
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 3889)
            print(f"ERROR Signal fusion failed: {e}")
            return []

    def _fuse_single_direction(self, signals, direction, current_price, market_context):
        """Fuse signals for a single direction (CALL or PUT)"""
        if not signals:
            return None
        
        # Calculate weighted confidence
        total_weighted_confidence = 0.0
        total_weight = 0.0
        all_reasons = []
        
        for signal in signals:
            signal_type, confidence, reason = signal
            brain_name = self._extract_brain_name(reason)
            brain_weight = self.fusion_data['brain_weights'].get(brain_name, 1.0)
            
            weighted_confidence = confidence * brain_weight
            total_weighted_confidence += weighted_confidence
            total_weight += brain_weight
            all_reasons.append(f"{brain_name}: {reason}")
        
        if total_weight == 0:
            return None
        
        effective_weight = max(total_weight, 3.0)
        base_confidence = total_weighted_confidence / effective_weight
        
        unique_brains = len(set(brain.split(':')[0] for brain in all_reasons))
        corroboration_count = unique_brains
        
        # SCALP CONSENSUS: At least 5 brains must agree
        if corroboration_count < 5:
            return None
            
        base_confidence += self.fusion_config['corroboration_bonus'] * (corroboration_count - 5)
        
        # Apply market context adjustments
        context_adjusted_confidence = self._apply_context_adjustment(
            base_confidence, direction, market_context
        )
        
        final_confidence = min(max(context_adjusted_confidence, 0.0), 9.9)
        
        if final_confidence < self.fusion_config['minimum_confidence']:
            return None
        
        # Create fused reason
        zone_confirmed = any(str(x).startswith("SupportResistanceBrainGPU") for x in all_reasons)
        fused_reason = f"FUSED {direction} SIGNAL | "
        fused_reason += "ZONE_CONFIRMED | " if zone_confirmed else "ZONE_NOT_CONFIRMED | "
        fused_reason += f"Confidence: {final_confidence:.2f} | "
        fused_reason += f"Contributing Brains: {corroboration_count} | "
        fused_reason += " | ".join(all_reasons[:3])  # Top 3 reasons
        
        fused_signal = (direction, final_confidence, fused_reason)
        
        # Record fused decision
        self.fusion_data['fused_decisions'].append({
            'signal': fused_signal,
            'timestamp': time.time(),
            'price': current_price,
            'contributing_signals': len(signals)
        })
        
        return fused_signal

    def _extract_brain_name(self, reason):
        """Extract brain name from signal reason"""
        if reason.startswith('['):
            end_idx = reason.find(']')
            if end_idx != -1:
                return reason[1:end_idx]
        return 'UnknownBrain'

    def _apply_context_adjustment(self, confidence, direction, market_context):
        """Apply market context adjustments to confidence"""
        adjusted_confidence = confidence
        
        # Trend alignment adjustment
        if 'trend' in market_context:
            trend = market_context['trend']
            if ((direction == "CALL" and trend in ["BULLISH", "STRONG_BULLISH"]) or
                (direction == "PUT" and trend in ["BEARISH", "STRONG_BEARISH"])):
                adjusted_confidence += 0.5
            elif ((direction == "CALL" and trend in ["BEARISH", "STRONG_BEARISH"]) or
                  (direction == "PUT" and trend in ["BULLISH", "STRONG_BULLISH"])):
                adjusted_confidence -= 0.5
            elif trend in ["SIDEWAYS", "NEUTRAL"]:
                adjusted_confidence -= 3.0  # Heavy penalty for directional bets in sideways markets
        
        # Volatility regime adjustment
        if 'volatility_regime' in market_context:
            vol_regime = market_context['volatility_regime']
            if vol_regime in ["HIGH_VOLATILITY", "EXTREME_VOLATILITY"]:
                adjusted_confidence -= 0.3  # Reduce confidence in high volatility
            elif vol_regime == "LOW_VOLATILITY":
                adjusted_confidence += 0.2  # Increase confidence in low volatility
        
        # Time of day adjustment (simplified)
        current_hour = time.localtime().tm_hour
        if current_hour in [9, 10, 14, 15]:  # Market open/close hours
            adjusted_confidence += 0.2  # Higher confidence during active hours
        elif current_hour in [0, 1, 2, 3, 4]:  # Overnight hours
            adjusted_confidence -= 0.3  # Lower confidence overnight
        
        return max(adjusted_confidence, 0.0)

    def _update_signal_performance(self, current_signals):
        """Update signal performance tracking"""
        current_time = time.time()
        
        # Add current signals to recent signals
        for signal in current_signals:
            self.fusion_data['recent_signals'].append({
                'signal': signal,
                'timestamp': current_time,
                'price_at_signal': self.master.current_context.get('price', 0.0)  # FIX: was get_current_price() which doesn't exist
            })
        
        # Clean old signals — FIX: preserve maxlen=1000, don't lose memory management
        recent_cutoff = current_time - self.fusion_config['recent_signal_window']
        fresh_signals = [sig for sig in self.fusion_data['recent_signals'] if sig['timestamp'] >= recent_cutoff]
        self.fusion_data['recent_signals'] = LinuxOptimizedDeque(maxlen=1000)
        self.fusion_data['recent_signals'].extend(fresh_signals)

    def _update_brain_weights(self, current_signals):
        """Update brain weights based on recent performance"""
        # This would require actual trade outcomes to properly implement
        # For now, use a simple reinforcement based on signal frequency and corroboration
        
        current_time = time.time()
        performance_window = 3600  # 1 hour window
        
        # Count signals by brain in recent period
        brain_signal_counts = defaultdict(int)
        brain_corroboration_scores = defaultdict(int)
        
        for signal in current_signals:
            _, _, reason = signal
            brain_name = self._extract_brain_name(reason)
            brain_signal_counts[brain_name] += 1
        
        # Simple weight adjustment based on recent activity
        for brain_name, count in brain_signal_counts.items():
            if count >= 2:  # Brains with multiple signals get slight boost
                self.fusion_data['brain_weights'][brain_name] = min(
                    self.fusion_data['brain_weights'][brain_name] * 1.05, 2.0
                )
            else:
                # Slight decay for less active brains
                self.fusion_data['brain_weights'][brain_name] = max(
                    self.fusion_data['brain_weights'][brain_name] * 0.99, 0.5
                )

    def get_fusion_report(self):
        """Generate fusion performance report"""
        total_signals = len(self.fusion_data['recent_signals'])
        recent_fused = list(self.fusion_data['fused_decisions'])[-10:]
        
        report = f"""
SIGNAL FUSION BRAIN REPORT
==========================
Recent Signals: {total_signals}
Recent Fused Decisions: {len(recent_fused)}

Brain Weights:
"""
        
        for brain, weight in sorted(self.fusion_data['brain_weights'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {brain}: {weight:.2f}\n"
        
        report += "\nRecent Fused Signals:\n"
        for i, decision in enumerate(recent_fused[-5:]):
            signal = decision['signal']
            report += f"{i+1}. {signal[0]} | Confidence: {signal[1]:.2f} | Brains: {decision['contributing_signals']}\n"
        
        return report

# ==================== ADVANCED ANALYSIS SYSTEM MASTER ====================

class AdvancedAnalysisSystem:
    """Master system coordinating all 16 analysis brains"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize all brains
        self.brains = {
            'zone_detector': ZonePointFiveDetectorGPU(self),
            'candle_psychology': CandlePsychologyMasterGPU(self),
            'volume_profile': VolumeProfileBrainGPU(self),
            'market_structure': MarketStructureBrainGPU(self),
            'order_flow': OrderFlowBrainGPU(self),
            'momentum_oscillator': MomentumOscillatorBrainGPU(self),
            'volatility_regime': VolatilityRegimeBrainGPU(self),
            'cycle_analysis': CycleAnalysisBrainGPU(self),
            'correlation_matrix': CorrelationMatrixBrainGPU(self),
            'pattern_recognition': PatternRecognitionBrainGPU(self),
            'support_resistance': SupportResistanceBrainGPU(self),
            'trend_analysis': TrendAnalysisBrainGPU(self),
            'market_regime': MarketRegimeBrainGPU(self),
            'price_action': PriceActionBrainGPU(self),
            'institutional_flow': InstitutionalFlowBrainGPU(self),
            'signal_fusion': SignalFusionBrainGPU(self)
        }
        
        # AI/ML System
        self.neural_network_manager = None  # retired model inference; mathematical brains remain active
        
        # Auto Update System
        self.auto_update_system = AutoUpdateSystem(self)

        # Market data storage
        self.market_data = {
            '1min': LinuxOptimizedDeque(1000),
            '5min': LinuxOptimizedDeque(500),
            '15min': LinuxOptimizedDeque(200)
        }
        
        self.current_context = {
            'price': 0.0,
            'trend': 'UNKNOWN',
            'volatility': 'UNKNOWN',
            'regime': 'UNKNOWN',
            'psychology': {},
            'timestamp': 0
        }
        
        print("ADVANCED ANALYSIS SYSTEM INITIALIZED WITH 16 BRAINS")
        print("Model inference: retired (deterministic signal fusion only)")
        print(f"Auto Update System: ACTIVE")

    def process_master_consensus(self, m5, h1, tf_dict=None):
        """Unified 29-Brain Consensus Engine integration directly inside Part 2."""
        from mode_engine import ModeEngine
        if not hasattr(self, '_mode_engine'):
            self._mode_engine = ModeEngine()
        return self._mode_engine.master_evaluate(m5, h1, tf_dict=tf_dict)


    def process_market_data(self, df_1min, df_5min, df_15min, current_candle):
        """Process new market data through all brains"""
        try:
            # FIX: Do NOT append full DataFrames to memory history on every tick (Causes Massive Memory Leak in LIVE)
            # self.market_data['1min'].append(df_1min)
            # self.market_data['5min'].append(df_5min) 
            # self.market_data['15min'].append(df_15min)
            
            # Update current context
            self.current_context.update({
                'price': current_candle['close'],
                'timestamp': time.time()
            })
            
            # Get candle psychology
            psychology = self.brains['candle_psychology'].analyze_candle_psychology(current_candle)
            self.current_context['psychology'] = psychology
            
            # Run all analysis brains
            all_signals = []
            
            # 1. Zone Detector Signals
            zone_signals = self.brains['zone_detector'].detect_0_5_zone_signals(
                current_candle, psychology, df_1min, df_5min, df_15min
            )
            all_signals.extend([(s[0], s[1], f'[ZonePointFiveDetectorGPU] {s[2]}') for s in zone_signals])
            
            # 2. Volume Profile Signals
            volume_profile = self.brains['volume_profile'].analyze_volume_profile(df_1min, '1min')
            volume_signals = self.brains['volume_profile'].detect_volume_signals(
                current_candle, psychology, df_1min
            )
            all_signals.extend([(s[0], s[1], f'[VolumeProfileBrainGPU] {s[2]}') for s in volume_signals])
            
            # 3. Market Structure Signals
            market_structure = self.brains['market_structure'].analyze_market_structure(df_1min, '1min')
            structure_signals = self.brains['market_structure'].get_structure_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[MarketStructureBrainGPU] {s[2]}') for s in structure_signals])
            
            # 4. Order Flow Signals
            order_flow = self.brains['order_flow'].analyze_order_flow(df_1min)
            volume_spike = self.brains['volume_profile'].detect_volume_spike(df_1min)
            orderflow_signals = self.brains['order_flow'].get_orderflow_signals(
                current_candle['close'], psychology, volume_spike
            )
            all_signals.extend([(s[0], s[1], f'[OrderFlowBrainGPU] {s[2]}') for s in orderflow_signals])
            
            # 5. Momentum Signals
            momentum = self.brains['momentum_oscillator'].calculate_all_oscillators(df_1min)
            momentum_signals = self.brains['momentum_oscillator'].get_momentum_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[MomentumOscillatorBrainGPU] {s[2]}') for s in momentum_signals])
            
            # 6. Volatility Signals
            volatility_regime = self.brains['volatility_regime'].analyze_volatility_regime(df_1min)
            volatility_signals = self.brains['volatility_regime'].get_volatility_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[VolatilityRegimeBrainGPU] {s[2]}') for s in volatility_signals])
            
            # 7. Cycle Signals
            market_cycles = self.brains['cycle_analysis'].analyze_market_cycles(df_1min)
            cycle_signals = self.brains['cycle_analysis'].get_cycle_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[CycleAnalysisBrainGPU] {s[2]}') for s in cycle_signals])
            
            # 8. Correlation Signals (simplified - would need actual correlated data)
            correlation_signals = self.brains['correlation_matrix'].get_correlation_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[CorrelationMatrixBrainGPU] {s[2]}') for s in correlation_signals])
            
            # 9. Pattern Recognition Signals
            patterns = self.brains['pattern_recognition'].recognize_chart_patterns(df_1min)
            pattern_signals = self.brains['pattern_recognition'].get_pattern_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[PatternRecognitionBrainGPU] {s[2]}') for s in pattern_signals])
            
            # 10. Support/Resistance Signals
            sr_levels = self.brains['support_resistance'].calculate_support_resistance(df_1min)
            sr_signals = self.brains['support_resistance'].get_sr_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[SupportResistanceBrainGPU] {s[2]}') for s in sr_signals])
            
            # 11. Trend Analysis Signals
            multi_tf_trends = self.brains['trend_analysis'].analyze_multi_timeframe_trends({
                '1min': df_1min, '5min': df_5min, '15min': df_15min
            })
            trend_signals = self.brains['trend_analysis'].get_trend_signals(
                current_candle['close'], psychology, multi_tf_trends.get('alignment', 'NEUTRAL')
            )
            all_signals.extend([(s[0], s[1], f'[TrendAnalysisBrainGPU] {s[2]}') for s in trend_signals])
            
            # 12. Market Regime Signals
            market_regime = self.brains['market_regime'].detect_market_regime(df_1min)
            regime_signals = self.brains['market_regime'].get_regime_signals(
                current_candle['close'], psychology, market_regime
            )
            all_signals.extend([(s[0], s[1], f'[MarketRegimeBrainGPU] {s[2]}') for s in regime_signals])
            
            # 13. Price Action Signals
            pa_patterns = self.brains['price_action'].analyze_price_action(df_1min)
            pa_signals = self.brains['price_action'].get_pa_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[PriceActionBrainGPU] {s[2]}') for s in pa_signals])
            
            # 14. Institutional Flow Signals
            institutional_flow = self.brains['institutional_flow'].analyze_institutional_flow(df_1min)
            institutional_signals = self.brains['institutional_flow'].get_institutional_signals(
                current_candle['close'], psychology
            )
            all_signals.extend([(s[0], s[1], f'[InstitutionalFlowBrainGPU] {s[2]}') for s in institutional_signals])
            
            # Update market context with new information
            self.current_context.update({
                'trend': multi_tf_trends.get('alignment', 'NEUTRAL'),
                'volatility': volatility_regime,
                'regime': market_regime
            })
            
            # 15. Fuse all signals
            fused_signals = self.brains['signal_fusion'].fuse_signals(
                all_signals, current_candle['close'], self.current_context
            )
            
            # 16. AI/ML Enhancement
            ai_enhanced_signals = fused_signals  # deterministic signals only; no model score boost
            
            # No model may append a synthetic trade signal to the fused math.
            # Check for system updates
            self.auto_update_system.check_for_updates()
            
            return ai_enhanced_signals
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 4290)
            print(f"ERROR Advanced analysis system processing failed: {e}")
            return []

    def _enhance_with_ai(self, signals, df):
        """Compatibility hook: model boosts cannot influence signal confidence."""
        return signals
        try:
            if not signals or len(df) < 50:
                return signals
            
            # Prepare data for AI prediction
            recent_prices = df['close'].tail(50).values
            recent_volumes = df['volume'].tail(50).values
            
            # Get AI predictions
            ai_predictions = self.neural_network_manager.predict(recent_prices, recent_volumes)
            
            # Enhance signals based on AI predictions
            enhanced_signals = []
            for signal in signals:
                signal_type, confidence, reason = signal
                
                # Apply AI confidence adjustment
                ai_confidence_boost = 0.0
                for model_name, prediction in ai_predictions.items():
                    if abs(prediction) > 0.1:  # Significant prediction
                        if (signal_type == "CALL" and prediction > 0) or (signal_type == "PUT" and prediction < 0):
                            ai_confidence_boost += abs(prediction) * 0.5
                
                enhanced_confidence = min(confidence + ai_confidence_boost, 9.9)
                
                enhanced_reason = f"{reason} | AI Boost: +{ai_confidence_boost:.2f}"
                enhanced_signals.append((signal_type, enhanced_confidence, enhanced_reason))
            
            return enhanced_signals
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 4327)
            print(f"ERROR AI enhancement failed: {e}")
            return signals

    def get_system_report(self):
        """Generate comprehensive system report"""
        report = "ADVANCED ANALYSIS SYSTEM - 16 BRAINS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Brain status
        report += "BRAIN STATUS:\n"
        for brain_name, brain in self.brains.items():
            report += f"- {brain_name}: ACTIVE\n"
        
        # Market context
        report += f"\nMARKET CONTEXT:\n"
        report += f"Price: {self.current_context.get('price', 'N/A')}\n"
        report += f"Trend: {self.current_context.get('trend', 'UNKNOWN')}\n"
        report += f"Volatility: {self.current_context.get('volatility', 'UNKNOWN')}\n"
        report += f"Regime: {self.current_context.get('regime', 'UNKNOWN')}\n"
        
        # AI/ML Status
        report += f"\nAI/ML SYSTEM:\n"
        report += "Model inference: retired; deterministic signals only\n"
        
        # Fusion Report
        fusion_report = self.brains['signal_fusion'].get_fusion_report()
        report += f"\n{fusion_report}"
        
        return report

    @part_advisory_entry("part2_zone", data_parameter="data", context_parameter="context")
    def analyze_native_zone(self, data, timeframe="1m", context=None):
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        return qm.part_signal("2", data, timeframe=timeframe, context=context)

# ==================== SYSTEM INITIALIZATION ====================

if __name__ == "__main__":
    # Initialize the complete advanced analysis system
    advanced_system = AdvancedAnalysisSystem()
    
    print("\n" + "="*60)
    print("ADVANCED TRADING SYSTEM INITIALIZATION COMPLETE")
    print("="*60)
    print("FEATURES:")
    print("- 16 Specialized Analysis Brains")
    print("- GPU Acceleration for All Calculations") 
    print("- AI/ML Model Integration (LSTM, Transformer, CNN)")
    print("- Neural Network Retraining System")
    print("- Automatic Code Updates")
    print("- Real-time Signal Fusion")
    print("="*60)
    
    # Display system report
    print(advanced_system.get_system_report())