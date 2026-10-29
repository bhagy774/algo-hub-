import logging
import json
import os
import traceback
import threading
from datetime import datetime
from queue import Queue, Empty
from collections import defaultdict, deque


class CognitiveLogger:
    """Asynchronously logs AI thoughts and health events to a central file."""
    
    def __init__(self, log_dir="logs", filename="jarvis_thoughts.log"):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(self.log_dir, self.filename)
        self.queue = Queue()
        self.is_running = True
        
        # Ensure log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Start background logging thread
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()
        
    def _log_worker(self):
        """Background thread that writes messages to disk sequentially."""
        while self.is_running:
            try:
                message = self.queue.get(timeout=1.0)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                log_entry = f"[{timestamp}] [{message['topic']}] [{message['sender']}] {message['payload']}\n"
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                self.queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in CognitiveLogger: {e}")
                
    def log(self, topic, sender, payload):
        """Add a message to the logging queue (non-blocking)."""
        if isinstance(payload, dict):
            try:
                payload = json.dumps(payload)
            except:
                payload = str(payload)
        self.queue.put({'topic': topic, 'sender': sender, 'payload': payload})
        
    def stop(self):
        self.is_running = False
        self.thread.join(timeout=2.0)


class PartHealthMonitor:
    """
    Tracks error history per part.
    If a part fails repeatedly, it raises a CRITICAL alert on the bus.
    """
    CRITICAL_THRESHOLD = 3   # After 3 consecutive errors -> CRITICAL
    WARN_THRESHOLD = 1       # First error -> WARNING
    
    def __init__(self):
        self._error_counts = defaultdict(int)
        self._last_error = {}
        self._ok_parts = set()
        
    def record_error(self, part_name: str, error: Exception, context: str = "") -> str:
        """Record an error and return severity level: WARNING or CRITICAL"""
        self._error_counts[part_name] += 1
        self._last_error[part_name] = str(error)
        self._ok_parts.discard(part_name)
        count = self._error_counts[part_name]
        
        if count >= self.CRITICAL_THRESHOLD:
            return "CRITICAL"
        return "WARNING"
    
    def record_ok(self, part_name: str):
        """Part recovered successfully - reset error count."""
        if part_name in self._error_counts and self._error_counts[part_name] > 0:
            self._error_counts[part_name] = 0
        self._ok_parts.add(part_name)
    
    def get_status(self) -> dict:
        """Get a snapshot of all part statuses."""
        status = {}
        for part, count in self._error_counts.items():
            if count == 0:
                status[part] = "OK"
            elif count >= self.CRITICAL_THRESHOLD:
                status[part] = f"CRITICAL ({count} errors) | Last: {self._last_error.get(part,'?')}"
            else:
                status[part] = f"WARNING ({count} errors) | Last: {self._last_error.get(part,'?')}"
        return status
    

class CognitiveBus:
    """
    Central nervous system for Jarvis.
    - Pub/Sub routing between AI agents.
    - Self-Diagnostic HEALTH monitoring.
    - AI-powered error analysis when available.
    """
    
    def __init__(self):
        self.subscribers = {
            'THOUGHTS': [],
            'HEALTH': [],
            'SIGNALS': [],
            'ERROR': [],
            'GEMINI_INSIGHT': [],   # Gemini Supreme Advisor -- 10-min strategic insights
            'ORACLE_FORECAST': [],  # JARVIS Market Oracle -- 5-min multi-timeframe market map
            'HEALTH_REPORT': [],    # JARVIS Doctor -- self-healing fix reports
            'ALL': []
        }
        self.logger = CognitiveLogger()
        self.health_monitor = PartHealthMonitor()

        # In-memory ring buffer -- last 100 THOUGHTS messages
        # Used by GeminiSupremeAdvisor to read parts' opinions
        self._thoughts_buffer: deque = deque(maxlen=100)
        # Latest message per sender (for quick per-part lookup)
        self._latest_by_sender: dict = {}
        # Latest Oracle forecast
        self._latest_oracle_forecast: dict = {}
        # Doctor health reports ring-buffer (last 50)
        self._doctor_reports: deque = deque(maxlen=50)

        self.logger.log('SYSTEM', 'CognitiveBus', 'Jarvis Cognitive Swarm Bus Initialized.')
        
    def subscribe(self, topic, callback):
        """Subscribe a function to a specific topic channel."""
        topic = topic.upper()
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        
    def publish(self, topic, sender, payload):
        """Publish a message to all subscribers of a topic, and log it."""
        topic = topic.upper()
        self.logger.log(topic, sender, payload)

        message = {
            'topic': topic,
            'sender': sender,
            'payload': payload,
            'timestamp': datetime.now().isoformat()
        }

        # Store THOUGHTS + SIGNALS in ring buffer for Gemini to read
        if topic in ('THOUGHTS', 'SIGNALS'):
            self._thoughts_buffer.append(message)
            self._latest_by_sender[sender] = message
        elif topic == 'ORACLE_FORECAST':
            self._latest_oracle_forecast = payload if isinstance(payload, dict) else {'raw': payload}
        elif topic == 'HEALTH_REPORT':
            self._doctor_reports.append(message)

        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.log('ERROR', 'CognitiveBus', f"Subscriber error on {topic}: {e}")

        for callback in self.subscribers['ALL']:
            try:
                callback(message)
            except Exception:
                pass

    def get_recent_thoughts(self, n: int = 30) -> list:
        """Return last N THOUGHTS/SIGNALS messages for Gemini context."""
        return list(self._thoughts_buffer)[-n:]

    def get_latest_by_sender(self) -> dict:
        """Return latest message per part/sender. Used for per-part status."""
        return dict(self._latest_by_sender)

    def get_latest_oracle_forecast(self) -> dict:
        """Return latest Market Oracle forecast."""
        return dict(self._latest_oracle_forecast)

    def get_doctor_reports(self, n: int = 10) -> list:
        """Return last N Doctor health reports. Used by HUD and Gemini advisor."""
        return list(self._doctor_reports)[-n:]

    def report_error(self, part_name: str, error: Exception, context: str = "", try_ai: bool = False, try_ollama: bool = False, **kwargs):
        """
        Call this inside any part's except block.
        Automatically:
        1. Determines severity (WARNING vs CRITICAL).
        2. Logs full traceback to jarvis_thoughts.log.
        3. Tries to call LegacyLLM to explain what went wrong.
        4. Publishes on HEALTH channel so other parts can react.
        
        Usage inside any part:
            try:
                ...
            except Exception as e:
                if hasattr(self, 'bus') and self.bus:
                    self.bus.report_error('Part3_Institutional', e, context='generate_mtf_signals')
        """
        severity = self.health_monitor.record_error(part_name, error, context)
        tb_str = traceback.format_exc()
        
        # Build diagnostic message
        diag = (
            f"[{severity}] {part_name} | Context: {context or 'unknown'} | "
            f"Error: {type(error).__name__}: {str(error)}"
        )
        
        # Log full traceback
        self.logger.log('HEALTH', part_name, diag)
        self.logger.log('TRACEBACK', part_name, tb_str.replace('\n', ' | '))
        
        # Try LegacyLLM diagnosis in background (non-blocking)
        if False:  # Retired model diagnosis; do not spawn a thread.
            threading.Thread(
                target=self._ai_diagnose,
                args=(part_name, type(error).__name__, str(error), context, severity),
                daemon=True
            ).start()
        
        # Publish on HEALTH channel for other subscribers
        self.publish('HEALTH', part_name, {
            'severity': severity,
            'error_type': type(error).__name__,
            'error_msg': str(error),
            'context': context,
            'error_count': self.health_monitor._error_counts[part_name]
        })

    def report_ok(self, part_name: str, context: str = ""):
        """Call when a part successfully completes after having had errors."""
        self.health_monitor.record_ok(part_name)
        self.logger.log('HEALTH', part_name, f"[RECOVERED] {part_name} is working again. Context: {context}")
        
    def _ai_diagnose(self, part_name: str, error_type: str, error_msg: str,
                         context: str, severity: str):
        """Retired model diagnosis. Structured local HEALTH event is sufficient."""
        return None

    def get_system_health(self) -> dict:
        """Returns a snapshot of all part health statuses."""
        return self.health_monitor.get_status()
    
    def print_health_report(self):
        """Print a formatted health report to console."""
        statuses = self.get_system_health()
        print("\n" + "="*55)
        print("  JARVIS SYSTEM HEALTH REPORT")
        print("="*55)
        if not statuses:
            print("  All parts healthy (no errors recorded).")
        else:
            for part, status in statuses.items():
                icon = "[!!]" if "CRITICAL" in status else "[!]" if "WARNING" in status else "[OK]"
                print(f"  {icon}  {part}: {status}")
        print("="*55 + "\n")

    def shutdown(self):
        self.logger.log('SYSTEM', 'CognitiveBus', 'Shutting down.')
        self.logger.stop()
