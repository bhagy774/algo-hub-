try:
    from jarvis_partlog import log_swallowed as _plog
except Exception:  # logging must never break a Part
    def _plog(*_a, **_k):
        return None
from neural_advisory import part_advisory_entry

# ==================== GPU-OPTIMIZED PART-10: FINAL EXECUTION ENGINE ====================
# DEEPSEEK AI-POWERED REWRITE - COMPLETE SYSTEM INTEGRATION
# LINUX UBUNTU + GTX 1650 CUDA + i5 10th Gen OPTIMIZED

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Mapping
import os
# PyTorch with fallback for Windows/WSL compatibility
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 17)
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
        def numpy(self): return np.array([0.0])
        def item(self): return 0.0
        def __getitem__(self, key): return self
        def __len__(self): return 1
        def detach(self): return self
    
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
# BUG FIX #1: cupy optional — no crash if not installed
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    globals().get("_plog", lambda *_a, **_k: None)(__name__, 128)
    CUPY_AVAILABLE = False
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import time
import json
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from collections import deque, defaultdict

from typing import Dict, List, Tuple, Any, Optional, Union




# ==================== GPU-ACCELERATED CANDLESTICK STATS ENGINE ====================

try:
    from brain_xgb import xgb_shadow
except Exception:  # xgboost helper missing -> no-op, engine unchanged
    def xgb_shadow(name, method):
        return lambda cls: cls

@xgb_shadow('p10_candlestats_engine', 'analyze')
class CandleStatsEngineGPU:
    """
    JARVIS PART 10 - GPU-ACCELERATED CANDLESTICK STATISTICAL & PRICE ACTION ENGINE
    GTX 1650 CUDA & CPU Optimized.

    Quantitative Candlestick Analytics Architecture:
    1. Volatility Baseline (ATR14): Anchors all body and expansion measurements to prevent micro-noise triggers.
    2. Body-to-Range Efficiency Ratio: Requires real bodies to exceed 50% of total candle range (eliminates wick indecision/dojis).
    3. Multi-Bar Directional Runs (Streak): Detects consecutive higher closes (Bullish Run) or lower closes (Bearish Run).
    4. Body Acceleration vs Baseline: Compares current 3-bar body size to 10-bar baseline, strictly scaled to ATR.
    5. Run Statistics (10-bar Count): 6+ or 8+ green/red dominance with expansion confirmation.
    6. Indecision & Mixed Deadband: Strictly returns 0 (Neutral) during spinning tops, micro-noise, or mixed bodies.
    """
    def __init__(self):
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')

    @part_advisory_entry("part10_candlestats", data_parameter="data", context_parameter="context")
    def analyze(self, data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Task-specific OHLCV evidence. The analyzer decorator still adds only its own advisory."""
        import quantitative_math as qm
        _res = qm.part_signal("10", data, context=context)
        try:  # PHASE3 diagnostic: log live metrics + gate reason for every NEUTRAL cycle
            import json as _j, logging as _lg, os as _os
            _lgr = _lg.getLogger("part10_diag")
            if not _lgr.handlers:
                _d = _os.path.join(_os.getcwd(), "logs", "files"); _os.makedirs(_d, exist_ok=True)
                _h = _lg.FileHandler(_os.path.join(_d, "part10_fixed.log"), encoding="utf-8")
                _h.setFormatter(_lg.Formatter("%(asctime)s %(message)s")); _lgr.addHandler(_h); _lgr.setLevel(_lg.INFO); _lgr.propagate = False
            if isinstance(_res, dict) and not _res.get("signal"):
                _lgr.info("[PART10_DIAG] NEUTRAL len=%s thought=%s telemetry=%s" % (len(data) if hasattr(data, "__len__") else "?", str(_res.get("thought"))[:200], _j.dumps(_res.get("telemetry", {}), default=str)[:600]))
        except Exception:
            pass
        return _res


# ==================== SYSTEM CONFIGURATION ====================
import threading
from dataclasses import dataclass
import logging
from pathlib import Path
import gc


# ==================== UNIFIED SYSTEM CONFIGURATION ====================

@dataclass
class SystemConfig:
    """Complete system configuration for Parts 1-11 integration"""
    
    # API Configuration - Delta Exchange (India)
    delta_base_url: str = "https://api.delta.exchange"
    delta_testnet_url: str = "https://cdn-ind.testnet.deltaex.org"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    telegram_base_url: str = "https://api.telegram.org"
    
    # Rate Limits
    delta_rate_limit: float = 10.0
    openrouter_rate_limit: float = 2.0
    telegram_rate_limit: float = 1.0
    
    # Trading Parameters
    trading_symbol: str = "BTCUSDT"
    expiry_seconds: int = 60
    min_confidence: float = 6.0
    cooldown_seconds: int = 30
    
    # System Optimization
    request_timeout: float = 5.0
    execution_interval: float = 0.1
    max_workers: int = 4
    max_queue_size: int = 100

# ==================== ASYNC HTTP CLIENT WITH RATE LIMITING ====================

class AsyncHTTPClient:
    """High-performance async HTTP client with intelligent rate limiting"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters = defaultdict(lambda: {'last_call': 0, 'call_count': 0})
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'AdvancedTradingBot/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _enforce_rate_limit(self, endpoint: str):
        """Intelligent rate limiting with adaptive backoff"""
        current_time = time.time()
        limiter = self.rate_limiters[endpoint]
        
        # Get rate limit for endpoint
        rate_limit = getattr(self.config, f"{endpoint}_rate_limit", 1.0)
        min_interval = 1.0 / rate_limit
        
        # Calculate time since last call
        time_since_last = current_time - limiter['last_call']
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
            
        # Update rate limiter
        limiter['last_call'] = time.time()
        limiter['call_count'] += 1
        
    async def get_json(self, url: str, endpoint: str = "delta") -> Optional[Dict]:
        """Async JSON GET with comprehensive error handling"""
        try:
            await self._enforce_rate_limit(endpoint)
            
            if not self.session:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logging.warning(f"HTTP {response.status} for GET {url}")
            else:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logging.warning(f"HTTP {response.status} for GET {url}")
                        
        except asyncio.TimeoutError:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 277)
            logging.error(f"Timeout fetching {url}")
        except aiohttp.ClientError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 279)
            logging.error(f"Client error for {url}: {e}")
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 281)
            logging.error(f"Unexpected error in GET {url}: {e}")
            
        return None
        
    async def post_json(self, url: str, data: Dict, endpoint: str = "openrouter") -> Optional[Dict]:
        """Async JSON POST with comprehensive error handling"""
        try:
            await self._enforce_rate_limit(endpoint)
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "AdvancedTradingBot/1.0"
            }
            
            if not self.session:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.post(url, json=data, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
            else:
                async with self.session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                        
        except asyncio.TimeoutError:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 306)
            logging.error(f"Timeout posting to {url}")
        except aiohttp.ClientError as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 308)
            logging.error(f"Client error for POST {url}: {e}")
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 310)
            logging.error(f"Unexpected error in POST {url}: {e}")
            
        return None

# ==================== HIGH-PRECISION DELTA EXCHANGE PRICE FEED ====================

class DeltaPriceFeed:
    """Ultra-fast Delta Exchange price feed with GPU-accelerated caching"""
    
    def __init__(self, http_client: AsyncHTTPClient, symbol: str = "BTCUSDT"):
        self.http_client = http_client
        self.symbol = symbol
        self.last_price: Optional[float] = None
        self.price_cache = deque(maxlen=500)  # Fixed-size price history
        self.cache_lock = threading.RLock()
        
    async def get_current_price(self) -> Optional[float]:
        """Get current price with multi-layer caching"""
        try:
            url = f"{self.http_client.config.delta_base_url}/v2/tickers/{self.symbol}"
            response = await self.http_client.get_json(url, "delta")
            
            if response and "result" in response:
                result = response["result"]
                price = float(result.get("close", result.get("mark_price", 0)))
            elif response and "close" in response:
                price = float(response["close"])
            elif response and "mark_price" in response:
                price = float(response["mark_price"])
            else:
                return self.last_price
            
            if price > 0:
                with self.cache_lock:
                    self.last_price = price
                    timestamp = datetime.now()
                    # BUG FIX #2: deque(maxlen=500) auto-handles overflow — manual popleft causes double-drop
                    self.price_cache.append((timestamp, price))
                
                return price
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 352)
            logging.error(f"Delta price feed error: {e}")
            
        # Return cached price as fallback
        return self.last_price
        
    async def get_price_history(self, lookback_seconds: int = 300) -> List[float]:
        """Get recent price history for analysis"""
        cutoff_time = datetime.now() - timedelta(seconds=lookback_seconds)
        
        with self.cache_lock:
            recent_prices = [
                price for timestamp, price in self.price_cache
                if timestamp >= cutoff_time
            ]
            
        return recent_prices if recent_prices else [self.last_price] if self.last_price else [0.0]

# ==================== DEEPSEEK AI VALIDATION ENGINE ====================

class DeepSeekValidator:
    """GPU-accelerated AI validation using DeepSeek Reasoner"""
    
    def __init__(self, http_client: AsyncHTTPClient, api_key: str = ""):
        self.http_client = http_client
        self.api_key = api_key
        self.enabled = bool(api_key)
        self.validation_cache = deque(maxlen=100)  # Cache recent validations
        
    async def validate_signal(self, signal_data: Dict) -> Dict[str, Any]:
        """Retired DeepSeekValidator model judge; unavailable never means approved."""
        return {"status": "UNAVAILABLE", "verdict": "NO_TRADE", "reasoning": "DeepSeekValidator retired"}
            
    def _build_validation_prompt(self, signal_data: Dict) -> str:
        """Build comprehensive validation prompt"""
        direction = signal_data.get("direction", "UNKNOWN")
        confidence = signal_data.get("confidence", 0)
        analysis = signal_data.get("analysis", "")
        patterns = signal_data.get("pattern_analysis", {})
        
        prompt = f"""
        TRADE OPTIONS TRADING SIGNAL VALIDATION
        
        SIGNAL DETAILS:
        - Direction: {direction}
        - Confidence: {confidence:.1f}/10
        - Analysis: {analysis}
        
        PATTERN ANALYSIS:
        {json.dumps(patterns, indent=2)}
        
        VALIDATION REQUEST:
        1. Assess signal validity based on technical patterns
        2. Evaluate risk-reward ratio
        3. Check market condition alignment
        4. Provide confidence adjustment recommendation
        5. Identify potential pitfalls
        
        RESPONSE FORMAT:
        VALID/CAUTION/INVALID|Reasoning|Confidence Adjustment
        
        Keep reasoning under 250 characters.
        """
        
        return prompt
        
    def _parse_validation_result(self, ai_response: str, signal_data: Dict) -> Dict[str, Any]:
        """Parse AI response into structured validation result"""
        try:
            lines = ai_response.split('|')
            if len(lines) >= 3:
                status = lines[0].strip().upper()
                reasoning = lines[1].strip()
                confidence_adj = lines[2].strip()
                
                return {
                    "status": status,
                    "reasoning": reasoning,
                    "confidence_adjustment": confidence_adj,
                    "original_confidence": signal_data.get("confidence", 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "UNKNOWN",
                    "reasoning": ai_response[:200],
                    "confidence_adjustment": "0",
                    "original_confidence": signal_data.get("confidence", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 443)
            logging.error(f"Validation parsing error: {e}")
            return {
                "status": "ERROR",
                "reasoning": "Parse error",
                "confidence_adjustment": "0",
                "original_confidence": signal_data.get("confidence", 0),
                "timestamp": datetime.now().isoformat()
            }

# ==================== LEGACY_LLM LOCAL AI TRADE VALIDATOR (THE JUDGE) ====================

# Model judges are retired; validator stays disabled (fail closed).
LEGACY_LLM_INTEGRATION_AVAILABLE = False


class LegacyLLMLocalValidator:
    """100% Offline Local AI Trade Validator (The Judge) using LegacyLLM"""
    
    def __init__(self):
        self.enabled = LEGACY_LLM_INTEGRATION_AVAILABLE
        self.validation_cache = deque(maxlen=100)
        self.log_file = Path(tempfile.gettempdir()) / "judge_verdict.log"
    
    def _build_validation_prompt(self, signal_data: Dict) -> str:
        direction = signal_data.get("direction", signal_data.get("action", "UNKNOWN"))
        confidence = signal_data.get("confidence", 0)
        entry_price = signal_data.get("entry_price", 0)
        score = signal_data.get("score", 0)
        stop_loss = signal_data.get("stop_loss", 0)
        take_profit = signal_data.get("take_profit_1", 0)
        ai_reason = signal_data.get("ai_reason", "")
        
        prompt = f"""You are The Judge, the final institutional risk-management AI for a high-frequency quant trading desk.

Trade Signal Candidate:
- Direction: {direction}
- Confidence: {confidence}/100
- Entry Price: {entry_price}
- Score: {score}
- Stop Loss: {stop_loss}
- Take Profit: {take_profit}
- Strategy Reason: {ai_reason}

Task: You hold ultimate VETO power over this trade. Evaluate if this trade should be executed or cancelled based on risk and confidence.

Respond with EXACTLY ONE of the following tags at the beginning:
- [EXECUTE] : Approve trade for order execution.
- [VETO] : Cancel trade due to high risk, low confidence, or adverse conditions.

Follow the tag with a 1-sentence institutional risk justification.
"""
        return prompt

    async def validate_signal(self, signal_data: Dict) -> Dict[str, Any]:
        """Retired LegacyLLMLocalValidator model judge; unavailable never means approved."""
        return {"status": "UNAVAILABLE", "verdict": "NO_TRADE", "reasoning": "LegacyLLMLocalValidator retired"}

    def _append_verdict_log(self, result: Dict):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{result['timestamp']}] Verdict: {result['verdict']} | {result['reasoning']}\n")
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 505)
            logging.warning(f"Failed writing judge verdict log: {e}")

# ==================== HIGH-RELIABILITY TELEGRAM NOTIFIER ====================

class TelegramNotifier:
    """Non-blocking Telegram notification system with queue management"""
    
    def __init__(self, http_client: AsyncHTTPClient, token: str = "", chat_id: str = ""):
        self.http_client = http_client
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        # BUG FIX #4: Don't create asyncio.Queue in __init__ — must be inside event loop
        # Lazy initialization in start() instead
        self.message_queue = None
        self.worker_task = None
        self.stats = {
            "sent": 0,
            "failed": 0,
            "queued": 0
        }
        
    async def start(self):
        """Start notification worker"""
        if not self.enabled:
            return
        # BUG FIX #4: Create Queue here inside the running event loop
        self.message_queue = asyncio.Queue(maxsize=self.http_client.config.max_queue_size)
        self.worker_task = asyncio.create_task(self._notification_worker())
        logging.info("Telegram notifier started")
        
    async def stop(self):
        """Stop notification worker"""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 543)
                pass
                
    async def send_message(self, message: str, priority: bool = False):
        """Queue message for sending (non-blocking)"""
        if not self.enabled:
            return False
        # BUG FIX #4: Queue may not be initialized yet if start() not called
        if self.message_queue is None:
            logging.warning("Telegram queue not initialized — call start() first")
            return False
            
        try:
            if priority and not self.message_queue.empty():
                # For priority messages, clear queue and send immediately
                while not self.message_queue.empty():
                    try:
                        self.message_queue.get_nowait()
                        self.message_queue.task_done()
                    except asyncio.QueueEmpty:
                        globals().get("_plog", lambda *_a, **_k: None)(__name__, 562)
                        break
                        
            await asyncio.wait_for(
                self.message_queue.put(message),
                timeout=0.5
            )
            self.stats["queued"] += 1
            return True
            
        except asyncio.TimeoutError:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 572)
            logging.warning("Telegram queue full, message dropped")
            self.stats["failed"] += 1
            return False
            
    async def _notification_worker(self):
        """Background worker for sending notifications"""
        while True:
            try:
                message = await self.message_queue.get()
                
                success = await self._send_message_internal(message)
                
                if success:
                    self.stats["sent"] += 1
                else:
                    self.stats["failed"] += 1
                    
                self.message_queue.task_done()
                
                # Rate limiting
                await asyncio.sleep(1.0 / self.http_client.config.telegram_rate_limit)
                
            except asyncio.CancelledError:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 595)
                break
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 597)
                logging.error(f"Notification worker error: {e}")
                await asyncio.sleep(1)
                
    async def _send_message_internal(self, message: str) -> bool:
        """Internal message sending implementation"""
        try:
            url = f"{self.http_client.config.telegram_base_url}/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message[:4000],  # Telegram limit
                "parse_mode": "HTML"
            }
            
            response = await self.http_client.post_json(url, payload, "telegram")
            return response is not None
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 614)
            logging.error(f"Telegram send error: {e}")
            return False

# ==================== INSTITUTIONAL TRADE EXECUTION ENGINE ====================

class SwingScalpTradeExecutor:
    """High-precision trade execution with GPU acceleration - Delta Exchange"""
    
    def __init__(self, price_feed: DeltaPriceFeed, symbol: str = "BTCUSDT"):
        self.price_feed = price_feed
        self.symbol = symbol
        self.active_trades = {}
        self.trade_history = deque(maxlen=1000)
        self.trade_counter = 0
        self.performance_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_profit": 0.0
        }
        
    async def execute_trade(self, direction: str, expiry_seconds: int = 60, *,
                            central_decision=None, central_approval=None, part_results=None,
                            part7_gate=None, symbol=None, exchange=None, contract=None,
                            instrument_id=None, market_type=None, snapshot_version=None) -> Dict[str, Any]:
        """Run this legacy simulator only after Jarvis central entry approval."""
        try:
            from jarvis_strategy_approval import evaluate_central_strategy, validate_entry_approval
            central_direction = {'CALL': 'BUY', 'BUY': 'BUY', 'PUT': 'SELL', 'SELL': 'SELL'}.get(str(direction).upper())
            if not isinstance(central_decision, Mapping) or central_decision.get('approved') is not True:
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
        except Exception as exc:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 662)
            return {'status': 'rejected', 'reason': f'Jarvis central approval validation failed: {exc}'}
        trade_id = self._generate_trade_id()
        start_time = datetime.now()
        expiry_time = start_time + timedelta(seconds=expiry_seconds)
        
        try:
            # Get entry price
            entry_price = await self.price_feed.get_current_price()
            if not entry_price:
                return self._create_trade_error("Price feed unavailable")
                
            # Register trade
            self.active_trades[trade_id] = {
                "direction": direction,
                "entry_price": entry_price,
                "start_time": start_time,
                "expiry_time": expiry_time,
                "status": "ACTIVE"
            }
            
            # Wait for expiry with high precision
            await self._precision_wait_for_expiry(trade_id, expiry_seconds)
            
            # Calculate result
            result = await self._calculate_trade_result(trade_id)
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 694)
            logging.error(f"Trade execution error: {e}")
            return self._create_trade_error(str(e))
            
    async def _precision_wait_for_expiry(self, trade_id: str, expiry_seconds: int):
        """High-precision expiry waiting with cancellation support"""
        # BUG FIX #5: Replaced 6000-iteration 10ms busy-poll with single sleep
        # Old: for _ in range(int(expiry_seconds/0.01)): await asyncio.sleep(0.01)
        # That was 6000 event loop iterations for 60s — starves other tasks
        await asyncio.sleep(expiry_seconds)
        # Check if trade was cancelled during sleep
        if trade_id not in self.active_trades:
            return
                
    async def _calculate_trade_result(self, trade_id: str) -> Dict[str, Any]:
        """Calculate trade result with comprehensive analytics"""
        if trade_id not in self.active_trades:
            return self._create_trade_error("Trade not found")
            
        trade = self.active_trades[trade_id]
        
        # Get exit price at exact expiry time
        exit_price = await self.price_feed.get_current_price()
        if not exit_price:
            # BUG FIX #6: Don't use entry_price as fallback — it causes guaranteed LOSS recording
            # Instead, mark trade as error/skip
            del self.active_trades[trade_id]
            return self._create_trade_error("Price feed unavailable at expiry — trade skipped")
            
        # Determine outcome
        if trade["direction"] == "CALL":
            is_win = exit_price > trade["entry_price"]
        else:  # PUT
            is_win = exit_price < trade["entry_price"]
            
        # BUG FIX #7: Make profit_pct signed — positive for WIN, negative for LOSS
        price_diff = abs(exit_price - trade["entry_price"])
        raw_pct = (price_diff / trade["entry_price"]) * 100
        profit_pct = raw_pct if is_win else -raw_pct
        
        result = {
            "trade_id": trade_id,
            "direction": trade["direction"],
            "entry_price": trade["entry_price"],
            "exit_price": exit_price,
            "start_time": trade["start_time"],
            "expiry_time": trade["expiry_time"],
            "result": "WIN" if is_win else "LOSS",
            "profit_pct": profit_pct,
            "price_movement": exit_price - trade["entry_price"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Cleanup and archive
        del self.active_trades[trade_id]
        self.trade_history.append(result)
        
        return result
        
    def _generate_trade_id(self) -> str:
        """Generate unique trade identifier"""
        self.trade_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"TRADE_{timestamp}_{self.trade_counter}"
        
    def _create_trade_error(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "trade_id": "ERROR",
            "direction": "UNKNOWN",
            "entry_price": 0.0,
            "exit_price": 0.0,
            "result": "ERROR",
            "profit_pct": 0.0,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        
    def _update_performance_stats(self, trade_result: Dict):
        """Update comprehensive performance statistics"""
        if trade_result["result"] == "ERROR":
            return
            
        self.performance_stats["total_trades"] += 1
        
        if trade_result["result"] == "WIN":
            self.performance_stats["winning_trades"] += 1
            # BUG FIX #7: profit_pct is now signed — just add directly
            self.performance_stats["total_profit"] += trade_result["profit_pct"]
        else:
            # profit_pct is negative for losses — adding negative = subtracting
            self.performance_stats["total_profit"] += trade_result["profit_pct"]
            
        # Calculate derivatives
        self.performance_stats["win_rate"] = (
            self.performance_stats["winning_trades"] / self.performance_stats["total_trades"]
        )
        self.performance_stats["avg_profit"] = (
            self.performance_stats["total_profit"] / self.performance_stats["total_trades"]
        )

# ==================== FINAL EXECUTION ENGINE - COMPLETE SYSTEM INTEGRATION ====================

class FinalExecutionEngine:
    """
    COMPLETE SYSTEM INTEGRATION ENGINE
    Orchestrates ALL Parts (1-11) with institutional-grade execution
    """
    
    def __init__(self, 
                 # Core AI Engines from Previous Parts
                 data_collector,	   # Part 1
                 preprocessor,		   # Part 2  
                 feature_engine,	   # Part 3
                 volume_analyzer,	   # Part 4
                 market_analyzer,	   # Part 5
                 signal_generator,	   # Part 6
                 risk_manager,		   # Part 7
                 pattern_engine,	   # Part 8
                 ai_learner,		   # Part 9
                 confidence_engine,    # Part 11
                 
                 # System Configuration
                 symbol: str = "BTCUSDT",
                 openrouter_key: str = "",
                 telegram_token: str = "",
                 telegram_chat_id: str = ""):
        
        # Store all AI engines
        self.data_collector = data_collector
        self.preprocessor = preprocessor
        self.feature_engine = feature_engine
        self.volume_analyzer = volume_analyzer
        self.market_analyzer = market_analyzer
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.pattern_engine = pattern_engine
        self.ai_learner = ai_learner
        self.confidence_engine = confidence_engine
        
        # System configuration - Delta Exchange
        self.config = SystemConfig(trading_symbol=symbol)
        
        # Async infrastructure - Delta Exchange
        self.http_client = AsyncHTTPClient(self.config)
        self.price_feed = DeltaPriceFeed(self.http_client, symbol)
        self.trade_executor = SwingScalpTradeExecutor(self.price_feed, symbol)
        self.deepseek_validator = DeepSeekValidator(self.http_client, openrouter_key)
        self.legacy_llm_validator = LegacyLLMLocalValidator()
        self.telegram_notifier = TelegramNotifier(self.http_client, telegram_token, telegram_chat_id)
        
        # Execution state
        self.is_running = False
        self.last_signal_time = 0
        self.system_stats = {
            "start_time": datetime.now(),
            "signals_processed": 0,
            "trades_executed": 0,
            "total_runtime": 0.0
        }
        
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_optimized = self.device.type == 'cuda'
        
        logging.info(f"Final Execution Engine initialized - GPU: {self.gpu_optimized}")
        
    def execute_strategy(self, unified_signal: dict) -> dict:
        """Synchronous bridge for the unified live path execution.
        Maps unified_signal into the async trading pipeline."""
        import asyncio
        
        direction = unified_signal.get('direction', 'NO-TRADE')
        if direction not in ('BUY', 'SELL'):
            logging.info(f"[EXEC] Skipping — direction is {direction}")
            return {"status": "skipped", "reason": f"Direction is {direction}"}
        
        confidence = unified_signal.get('confidence', 0)
        if confidence < 60:
            logging.info(f"[EXEC] Skipping — confidence {confidence} below threshold")
            return {"status": "skipped", "reason": f"Low confidence: {confidence}"}
        
        # Map to internal signal format
        internal_signal = {
            'direction': 'CALL' if direction == 'BUY' else 'PUT',
            'confidence': confidence,
            'entry_price': unified_signal.get('entry_price', 0),
            'score': unified_signal.get('score', 0),
            'stop_loss': unified_signal.get('stop_loss', 0),
            'take_profit_1': unified_signal.get('take_profit_1', 0),
            'ai_reason': unified_signal.get('ai_reason', ''),
        }
        
        logging.info(f"🚀 [EXEC] Executing {internal_signal['direction']} | Confidence: {confidence}%")
        # Publish to CognitiveBus for Watcher AI monitoring
        if hasattr(self, 'bus') and self.bus:
            msg = f"Execution Engine: Direction={direction}, Confidence={confidence}%. Entry price={unified_signal.get('entry_price', 0)}"
            self.bus.publish('THOUGHTS', 'Part10_Execution', msg)
        
        try:
            # Try to use existing event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context — schedule as task
                future = asyncio.ensure_future(self._process_trading_signal(internal_signal))
                return {"status": "submitted_async", "direction": internal_signal['direction']}
            except RuntimeError:
                # No running loop — create one
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 900)
                return asyncio.run(self._process_trading_signal(internal_signal))
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 903)
            logging.error(f"❌ [EXEC] execute_strategy error: {e}")
            return {"status": "error", "reason": str(e)}
        
    async def start_engine(self):
        """Start the complete trading system"""
        logging.info("ACCELERATED Starting Complete Trading System")
        
        self.is_running = True
        self.system_stats["start_time"] = datetime.now()
        
        # Start background services
        await self.telegram_notifier.start()
        
        # System startup notification
        await self.telegram_notifier.send_message(
            "🚀 ADVANCED TRADING SYSTEM STARTED\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💱 Symbol: {self.config.trading_symbol}\n"
            f"🖥️ GPU Acceleration: {self.gpu_optimized}\n"
            f"🔧 AI Engines: 11/11 Active\n"
            f"📊 Confidence Threshold: {self.config.min_confidence}/10"
        )
        
        try:
            # Main execution loop with async context
            async with self.http_client:
                await self._main_execution_loop()
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 932)
            logging.critical(f"Fatal system error: {e}")
            await self.telegram_notifier.send_message(
                f"❌ CRITICAL SYSTEM ERROR\n{str(e)}",
                priority=True
            )
        finally:
            await self._shutdown_system()
            
    async def _main_execution_loop(self):
        """High-performance main execution loop"""
        loop_start_time = time.time()
        
        while self.is_running:
            try:
                iteration_start = time.time()
                
                # Get final trading signal from confidence engine
                final_signal = await self._get_final_signal()
                
                if final_signal and self._validate_signal(final_signal):
                    # BUG FIX #12: Count only actual valid signals, not every loop iteration
                    self.system_stats["signals_processed"] += 1
                    await self._process_trading_signal(final_signal)
                    
                # Update system runtime
                self.system_stats["total_runtime"] = time.time() - loop_start_time
                
                # Adaptive sleep for optimal CPU usage
                elapsed = time.time() - iteration_start
                sleep_time = max(0.01, self.config.execution_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 965)
                logging.error(f"Main loop iteration error: {e}")
                await asyncio.sleep(1)  # Error recovery delay
                
    async def _get_final_signal(self) -> Optional[Dict]:
        """Retrieve final trading signal from confidence engine"""
        try:
            # BUG FIX #8: run_in_executor can't run async (coroutine) functions
            # Check if method is a coroutine function and await it directly
            method = getattr(self.confidence_engine, 'get_final_signal', None)
            if method is None:
                return None
            if asyncio.iscoroutinefunction(method):
                return await method()
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.http_client.executor, method)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 983)
            logging.error(f"Signal retrieval error: {e}")
            return None
            
    def _validate_signal(self, signal: Dict) -> bool:
        """Comprehensive signal validation"""
        if not signal or signal.get("direction") not in ["CALL", "PUT"]:
            return False
            
        # Confidence threshold
        confidence = signal.get("confidence", 0)
        if confidence < self.config.min_confidence:
            return False
            
        # Cooldown period
        current_time = time.time()
        if current_time - self.last_signal_time < self.config.cooldown_seconds:
            return False
            
        # Risk management approval
        if hasattr(self.risk_manager, 'approve_trade'):
            try:
                if not self.risk_manager.approve_trade(signal):
                    return False
            except Exception as e:
                globals().get("_plog", lambda *_a, **_k: None)(__name__, 1007)
                logging.warning(f"Risk approval error: {e}")
                
        return True
        
    async def _process_trading_signal(self, signal: Dict):
        """Complete signal processing pipeline"""
        self.last_signal_time = time.time()
        self.system_stats["trades_executed"] += 1
        
        try:
            # This legacy independent execution pipeline had no audited deterministic
            # replacement for its two model judges. Fail CLOSED for new entries.
            # Do not send a pre-trade alert or place an order; protective exit
            # handling lives elsewhere and is unaffected.
            logging.warning("Part 10 legacy trade entry disabled: model judges retired")
            return {"status": "no_trade", "reason": "legacy entry lacks deterministic validation"}

            # Step 2: Pre-trade notification
            await self._send_pre_trade_alert(signal, validation_result)
            
            # Step 3: Execute trade
            trade_result = await self.trade_executor.execute_trade(
                direction=signal["direction"],
                expiry_seconds=self.config.expiry_seconds
            )
            
            # Step 4: Performance tracking
            self._update_system_performance(trade_result)
            
            # Step 5: Result notification
            await self._send_trade_result(trade_result, signal)
            
            # Step 6: AI Learning feedback
            await self._update_ai_systems(trade_result, signal, validation_result)
            
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1043)
            logging.error(f"Signal processing pipeline error: {e}")
            await self.telegram_notifier.send_message(
                f"⚠️ TRADE PROCESSING ERROR\n{str(e)}"
            )
            
    async def _send_pre_trade_alert(self, signal: Dict, validation: Dict):
        """Send comprehensive pre-trade alert"""
        direction_emoji = "🟢" if signal["direction"] == "CALL" else "🔴"
        status_emoji = "✅" if validation.get("status") == "VALID" else "⚠️"
        
        message = (
            f"{direction_emoji} NEW TRADE SIGNAL\n"
            f"Pair: {self.config.trading_symbol}\n"
            f"Direction: {signal['direction']}\n"
            f"Confidence: {signal.get('confidence', 0):.1f}/10\n"
            f"AI Validation: {status_emoji} {validation.get('status', 'UNKNOWN')}\n"
            f"Reasoning: {validation.get('reasoning', 'N/A')[:100]}...\n"
            f"Expiry: {self.config.expiry_seconds}s"
        )
        
        await self.telegram_notifier.send_message(message)
        
    async def _send_trade_result(self, trade_result: Dict, original_signal: Dict):
        """Send detailed trade result"""
        result_emoji = "💰" if trade_result["result"] == "WIN" else "💸"
        # BUG FIX #11: Base color on result field, profit_pct sign was unreliable before fix
        profit_color = "🟢" if trade_result["result"] == "WIN" else "🔴"
        
        message = (
            f"{result_emoji} TRADE COMPLETED\n"
            f"Result: {trade_result['result']}\n"
            f"Direction: {trade_result['direction']}\n"
            f"Entry: {trade_result['entry_price']:.5f}\n"
            f"Exit: {trade_result['exit_price']:.5f}\n"
            f"P&L: {profit_color} {trade_result['profit_pct']:.2f}%\n"
            f"System Win Rate: {self.trade_executor.performance_stats['win_rate']:.1%}\n"
            f"Total Trades: {self.trade_executor.performance_stats['total_trades']}"
        )
        
        await self.telegram_notifier.send_message(message)
        
    def _update_system_performance(self, trade_result: Dict):
        """Update comprehensive system performance metrics"""
        # Performance stats are automatically updated in trade_executor
        pass
        
    async def _update_ai_systems(self, trade_result: Dict, original_signal: Dict, validation: Dict):
        """Update all AI systems with trade outcome"""
        try:
            learning_package = {
                "trade_result": trade_result,
                "original_signal": original_signal,
                "validation": validation,
                "market_data": await self.price_feed.get_price_history(300),
                "timestamp": datetime.now().isoformat()
            }
            
            # BUG FIX #9: process_trade_outcome is async in Part 9 — await directly
            # run_in_executor cannot run coroutines — silently returns coroutine object
            if hasattr(self.ai_learner, 'process_trade_outcome'):
                method = self.ai_learner.process_trade_outcome
                if asyncio.iscoroutinefunction(method):
                    await method(learning_package)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self.http_client.executor, method, learning_package)
                
            # BUG FIX #9: Same fix for confidence engine update
            if hasattr(self.confidence_engine, 'update_confidence'):
                method = self.confidence_engine.update_confidence
                if asyncio.iscoroutinefunction(method):
                    await method(learning_package)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self.http_client.executor, method, learning_package)
                
        except Exception as e:
            globals().get("_plog", lambda *_a, **_k: None)(__name__, 1120)
            logging.error(f"AI system update error: {e}")
            
    async def _shutdown_system(self):
        """Graceful system shutdown"""
        logging.info("Shutting down Final Execution Engine")
        self.is_running = False
        
        # Send shutdown notification with performance summary
        stats = self.trade_executor.performance_stats
        runtime = self.system_stats["total_runtime"]
        
        await self.telegram_notifier.send_message(
            "🛑 TRADING SYSTEM STOPPED\n"
            f"Total Runtime: {runtime:.1f}s\n"
            f"Signals Processed: {self.system_stats['signals_processed']}\n"
            f"Trades Executed: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate']:.1%}\n"
            f"Avg Profit: {stats['avg_profit']:.2f}%\n"
            f"Total Profit: {stats['total_profit']:.2f}%"
        )
        
        # Stop background services
        await self.telegram_notifier.stop()
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system": {
                "is_running": self.is_running,
                "gpu_accelerated": self.gpu_optimized,
                "start_time": self.system_stats["start_time"].isoformat(),
                "total_runtime": self.system_stats["total_runtime"],
                "signals_processed": self.system_stats["signals_processed"]
            },
            "trading": self.trade_executor.performance_stats,
            "notifications": self.telegram_notifier.stats,
            "timestamp": datetime.now().isoformat()
        }

    async def stop_engine(self):
        """BUG FIX #10: stop_engine() was called in main() but never defined — added"""
        logging.info("Stop requested — shutting down Final Execution Engine")
        self.is_running = False
        await self._shutdown_system()

# ==================== SYSTEM INITIALIZATION & OPTIMIZATION ====================

def setup_linux_trading_environment():
    """Optimize Linux environment for high-frequency trading"""
    # CPU Optimization
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # GPU Memory Optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Async Performance
    os.environ['PYTHONASYNCIODEBUG'] = '0'
    
    # Network Optimization
    os.environ['AIOHTTP_NO_EXTENSIONS'] = '0'
    
    logging.info("Linux trading environment optimized")

async def main():
    """Main execution function"""
    setup_linux_trading_environment()
    
    # Initialize your AI engines from Parts 1-11 here
    # data_collector = Part1DataCollector()
    # preprocessor = Part2Preprocessor()
    # ... etc for all 11 parts
    
    # Create final execution engine
    engine = FinalExecutionEngine(
        data_collector=None,      # Replace with actual instances
        preprocessor=None,
        feature_engine=None,
        volume_analyzer=None,
        market_analyzer=None,
        signal_generator=None,
        risk_manager=None,
        pattern_engine=None,
        ai_learner=None,
        confidence_engine=None,
        openrouter_key="your_key_here",      # Optional
        telegram_token="your_token_here",    # Optional
        telegram_chat_id="your_chat_id"      # Optional
    )
    
    try:
        await engine.start_engine()
    except KeyboardInterrupt:
        globals().get("_plog", lambda *_a, **_k: None)(__name__, 1213)
        logging.info("Received shutdown signal")
        await engine.stop_engine()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run the complete system
    asyncio.run(main())