#!/usr/bin/env python3
"""
JARVIS Self-Healing AI Doctor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Live system monitor + Gemini-powered auto-diagnosis + auto-fix.

Every 60 seconds:
  1. 12 Parts health check (import check + CognitiveBus staleness)
  2. Log file scan (ERROR / CRITICAL lines)
  3. System resources (RAM / CPU / disk via psutil)
  4. Dependency check (LegacyLLM server, Delta API)
  5. If problem found -> Gemini diagnosis -> auto-apply safe fix
  6. Telegram CRITICAL alerts
  7. Print Doctor Report to console
  8. Publish HEALTH_REPORT on CognitiveBus

API Key : DOCTOR_API_KEY (from .env)
Model   : DOCTOR_GEMINI_MODEL (default: gemini-3.6-flash)
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import json
import time
import logging
import threading
import subprocess
import traceback
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("JarvisDoctor")

# ── Config ─────────────────────────────────────────────────────────
DOCTOR_API_KEY  = os.environ.get("DOCTOR_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
DOCTOR_MODEL    = os.environ.get("DOCTOR_GEMINI_MODEL", "gemini-3.6-flash")
DOCTOR_ENABLED  = os.environ.get("DOCTOR_ENABLED", "true").lower() == "true"
# Health observations may be reported, but model-produced plans must never
# pip-install packages or mutate live-trading environment/configuration.
DOCTOR_AUTO_FIX = False
DOCTOR_INTERVAL = int(os.environ.get("DOCTOR_CHECK_INTERVAL", "60"))
DOCTOR_COOLDOWN = int(os.environ.get("DOCTOR_COOLDOWN", "600"))
STALE_THRESHOLD = int(os.environ.get("DOCTOR_STALE_THRESHOLD", "300"))
TELEGRAM_ALERTS = os.environ.get("DOCTOR_TELEGRAM_ALERTS", "true").lower() == "true"

_HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(_HERE, "logs")
LOG_FILE = os.path.join(LOG_DIR, "jarvis_thoughts.log")

# Colors
R  = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
C  = '\033[96m'; DG = '\033[90m'
BD = '\033[1m';  RST = '\033[0m'

# 12 Parts registry
PARTS_REGISTRY = [
    {"name": "Part1_Breakout",      "module": "part1_FIXED"},
    {"name": "Part2_Neural",        "module": "part2_FIXED"},
    {"name": "Part3_Institutional", "module": "part3_FIXED"},
    {"name": "Part4_Backtest",      "module": "part4_FIXED"},
    {"name": "Part5_Fusion",        "module": "part5_FIXED"},
    {"name": "Part6_Trend",         "module": "part6_FIXED"},
    {"name": "Part7_Volatility",    "module": "part7_FIXED"},
    {"name": "Part8_Pattern",       "module": "part8_FIXED"},
    {"name": "Part9_Orderflow",     "module": "part9_FIXED"},
    {"name": "Part10_Sentiment",    "module": "part10_FIXED"},
    {"name": "Part11_Confidence",   "module": "part11_FIXED"},
    {"name": "Part12_Execution",    "module": "part12_FIXED"},
]


# ══════════════════════════════════════════════════════════════════════
class DoctorIssue:
    """Represents one detected problem."""
    def __init__(self, source: str, level: str, error_type: str,
                 message: str, traceback_str: str = ""):
        self.source        = source
        self.level         = level          # CRITICAL / WARNING
        self.error_type    = error_type
        self.message       = message
        self.traceback_str = traceback_str
        self.ts            = datetime.now()
        self.fix_applied: Optional[str] = None
        self.fix_result:  Optional[str] = None

    def fingerprint(self) -> str:
        return f"{self.source}::{self.error_type}::{self.message[:60]}"


# ══════════════════════════════════════════════════════════════════════
class DoctorMonitor:
    """
    JARVIS Self-Healing AI Doctor.
    Call init_doctor(bus, live_trader) once at startup — runs forever.
    """

    def __init__(self, bus=None, live_trader=None):
        self.bus         = bus
        self.live_trader = live_trader
        self.is_running  = False
        self._thread: Optional[threading.Thread] = None
        self._lock       = threading.Lock()

        # Dedup cooldown: fingerprint -> last_seen
        self._seen: Dict[str, datetime] = {}
        # Recent issues ring-buffer for display
        self._recent: deque = deque(maxlen=20)
        # Part status map: name -> "OK"/"STALE"/"IMPORT_FAIL"/"SILENT"
        self._part_status: Dict[str, str] = {}

        self._checks_done    = 0
        self._fixes_done     = 0
        self._last_report_ts: Optional[datetime] = None

        # Subscribe to CognitiveBus HEALTH topic
        if self.bus:
            try:
                self.bus.subscribe("HEALTH", self._on_health_event)
            except Exception:
                pass

        self._ensure_psutil()
        logger.info("[Doctor] JARVIS Self-Healing Doctor initialized.")

    # ── Helpers ────────────────────────────────────────────────────
    def _ensure_psutil(self):
        try:
            import psutil  # noqa
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "psutil", "-q"],
                    check=True, capture_output=True, timeout=60
                )
                logger.info("[Doctor] psutil installed.")
            except Exception as e:
                logger.error(f"[Doctor] psutil install failed: {e}")

    def _should_report(self, fp: str) -> bool:
        with self._lock:
            last = self._seen.get(fp)
            if last and (datetime.now() - last).total_seconds() < DOCTOR_COOLDOWN:
                return False
            self._seen[fp] = datetime.now()
            return True

    def _add_issue(self, issue: DoctorIssue) -> bool:
        if self._should_report(issue.fingerprint()):
            self._recent.append(issue)
            return True
        return False

    # ── CognitiveBus callback ──────────────────────────────────────
    def _on_health_event(self, message: dict):
        payload = str(message.get("payload", ""))
        sender  = message.get("sender", "UNKNOWN")
        if "CRITICAL" in payload.upper() or "CRASH" in payload.upper():
            issue = DoctorIssue(source=sender, level="CRITICAL",
                                error_type="HEALTH_EVENT", message=payload[:300])
            if self._add_issue(issue):
                self._handle_issue(issue)

    # ── LAYER 1: Parts Health ──────────────────────────────────────
    def _check_parts_health(self) -> List[DoctorIssue]:
        issues = []
        for part in PARTS_REGISTRY:
            name   = part["name"]
            module = part["module"]
            # Import check
            try:
                import importlib
                importlib.import_module(module)
            except ImportError as e:
                self._part_status[name] = "IMPORT_FAIL"
                issues.append(DoctorIssue(source=name, level="CRITICAL",
                    error_type="ImportError", message=str(e),
                    traceback_str=traceback.format_exc()))
                continue
            except Exception as e:
                self._part_status[name] = "IMPORT_ERR"
                issues.append(DoctorIssue(source=name, level="WARNING",
                    error_type=type(e).__name__, message=str(e)))
                continue

            # CognitiveBus staleness check
            self._part_status[name] = "OK"
            if self.bus:
                try:
                    latest = self.bus.get_latest_by_sender()
                    key = next(
                        (k for k in latest
                         if name.lower().split("_")[0] in k.lower()), None
                    )
                    if key:
                        ts_str = latest[key].get("timestamp", "")
                        if ts_str:
                            age = (datetime.now() - datetime.fromisoformat(ts_str)).total_seconds()
                            if age > STALE_THRESHOLD:
                                self._part_status[name] = "STALE"
                                issues.append(DoctorIssue(source=name, level="WARNING",
                                    error_type="STALE_SIGNAL",
                                    message=f"No signal in {int(age)}s (limit {STALE_THRESHOLD}s)"))
                except Exception:
                    pass  # bus check failed gracefully

        return issues

    # ── LAYER 2: Log Scan ──────────────────────────────────────────
    def _scan_log_file(self, n_lines: int = 200) -> List[DoctorIssue]:
        issues = []
        if not os.path.exists(LOG_FILE):
            return issues
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                recent = deque(f, maxlen=n_lines)
            for line in recent:
                line = line.strip()
                if not line:
                    continue
                up = line.upper()
                if "CRITICAL" in up:
                    level = "CRITICAL"
                elif "ERROR" in up or "EXCEPTION" in up:
                    level = "WARNING"
                else:
                    continue
                parts = line.split("]")
                sender = parts[2].strip().lstrip("[") if len(parts) > 2 else "SYSTEM"
                msg    = parts[-1].strip() if parts else line
                issues.append(DoctorIssue(source=sender, level=level,
                                           error_type="LOG_SCAN", message=msg[:300]))
        except Exception as e:
            logger.debug(f"[Doctor] Log scan: {e}")
        return issues

    # ── LAYER 3: System Resources ──────────────────────────────────
    def _check_system_resources(self) -> List[DoctorIssue]:
        issues = []
        try:
            import psutil
            ram = psutil.virtual_memory()
            if ram.percent > 95:
                issues.append(DoctorIssue("SYSTEM","CRITICAL","RAM_CRITICAL",
                    f"RAM {ram.percent:.1f}% -- OOM risk!"))
            elif ram.percent > 85:
                issues.append(DoctorIssue("SYSTEM","WARNING","RAM_HIGH",
                    f"RAM {ram.percent:.1f}%"))

            cpu = psutil.cpu_percent(interval=1)
            if cpu > 90:
                issues.append(DoctorIssue("SYSTEM","WARNING","CPU_HIGH",
                    f"CPU {cpu:.1f}%"))

            disk = psutil.disk_usage(_HERE)
            free_gb = disk.free / (1024**3)
            if free_gb < 1.0:
                issues.append(DoctorIssue("SYSTEM","CRITICAL","DISK_FULL",
                    f"Only {free_gb:.1f}GB disk free!"))
            elif free_gb < 3.0:
                issues.append(DoctorIssue("SYSTEM","WARNING","DISK_LOW",
                    f"Low disk {free_gb:.1f}GB"))
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[Doctor] Resources: {e}")
        return issues

    # ── LAYER 4: Dependency Check ──────────────────────────────────
    def _check_dependencies(self) -> List[DoctorIssue]:
        issues = []
        # LegacyLLM retired; dependency is not required by the deterministic runtime.

        # Delta API
        try:
            import requests
            r = requests.get("https://api.delta.exchange/v2/products",
                             timeout=5, params={"page_size": 1})
            if r.status_code not in (200, 429):
                issues.append(DoctorIssue("DeltaAPI","WARNING","API_DEGRADED",
                    f"HTTP {r.status_code}"))
        except Exception as e:
            issues.append(DoctorIssue("DeltaAPI","WARNING","CONNECTION_FAILED", str(e)))

        return issues

    # ── LAYER 5: Gemini Diagnosis ──────────────────────────────────
    def _diagnose_with_gemini(self, issue: DoctorIssue) -> Optional[dict]:
        """Legacy diagnosis retired; no model call or model-proposed auto-fix."""
        return None
        if not DOCTOR_API_KEY:
            return None
        try:
            import requests
            ok_cnt = sum(1 for v in self._part_status.values() if v == "OK")
            sys_state = f"Parts OK: {ok_cnt}/{len(PARTS_REGISTRY)}"
            try:
                import psutil
                ram = psutil.virtual_memory()
                sys_state += f" | RAM {ram.percent:.1f}% | CPU {psutil.cpu_percent():.1f}%"
            except Exception:
                pass

            prompt = (
                f"JARVIS DOCTOR Auto-Diagnosis [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n"
                f"=== PROBLEM ===\n"
                f"Source   : {issue.source}\n"
                f"Error    : {issue.error_type}\n"
                f"Severity : {issue.level}\n"
                f"Message  : {issue.message}\n"
                f"{('Traceback:' + chr(10) + issue.traceback_str[:800]) if issue.traceback_str else ''}\n\n"
                f"=== SYSTEM STATE ===\n{sys_state}\n\n"
                f"Return ONLY valid JSON (no markdown fences):\n"
                '{\n'
                '  "diagnosis": "root cause in one line",\n'
                '  "severity": "CRITICAL|WARNING|INFO",\n'
                '  "fix_type": "pip_install|config_fix|log_rotate|manual|none",\n'
                '  "fix_package": "pkg_name_if_pip_install",\n'
                '  "fix_env_key": "KEY_if_config_fix",\n'
                '  "fix_env_val": "value_if_config_fix",\n'
                '  "safe_auto_apply": true,\n'
                '  "explanation": "2-3 sentence explanation",\n'
                '  "telegram_summary": "1-line Telegram alert"\n'
                '}'
            )

            url = (f"https://generativelanguage.googleapis.com/v1beta"
                   f"/models/{DOCTOR_MODEL}:generateContent")
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json",
                         "X-goog-api-key": DOCTOR_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}},
                timeout=30
            )
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Strip markdown fences
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:])
                if text.endswith("```"):
                    text = text[:-3]
                return json.loads(text.strip())
            else:
                logger.warning(f"[Doctor] Gemini {resp.status_code}")
        except json.JSONDecodeError:
            logger.warning("[Doctor] Gemini non-JSON response")
        except Exception as e:
            logger.error(f"[Doctor] Gemini error: {e}")
        return None

    # ── LAYER 6: Auto-Fix ──────────────────────────────────────────
    def _handle_issue(self, issue: DoctorIssue):
        if not self._add_issue(issue):
            return  # cooldown active

        lcol = R if issue.level == "CRITICAL" else Y
        print(f"\n{lcol}{BD}[DOCTOR] [{issue.level}] {issue.source} -- {issue.error_type}{RST}")
        print(f"  {DG}{issue.message[:120]}{RST}")

        fix_plan = self._diagnose_with_gemini(issue)
        if fix_plan:
            issue.fix_applied = fix_plan.get("fix_type", "none")
            print(f"  {C}Gemini diagnosis: {fix_plan.get('diagnosis', '')}{RST}")
            print(f"  {DG}{fix_plan.get('explanation', '')}{RST}")
            if DOCTOR_AUTO_FIX and fix_plan.get("safe_auto_apply"):
                self._apply_fix(issue, fix_plan)
        else:
            issue.fix_applied = "no_gemini"
            print(f"  {Y}Gemini unavailable -- issue logged{RST}")

        if issue.level == "CRITICAL" and TELEGRAM_ALERTS:
            self._send_telegram_alert(issue, fix_plan)

        # Publish to CognitiveBus
        if self.bus:
            try:
                self.bus.publish("HEALTH_REPORT", "JarvisDoctor", {
                    "source":      issue.source,
                    "level":       issue.level,
                    "error_type":  issue.error_type,
                    "message":     issue.message,
                    "fix_applied": issue.fix_applied,
                    "fix_result":  issue.fix_result,
                    "ts":          issue.ts.isoformat()
                })
            except Exception:
                pass

    def _apply_fix(self, issue: DoctorIssue, fix_plan: dict):
        fix_type = fix_plan.get("fix_type", "none")
        try:
            if fix_type == "pip_install":
                pkg = fix_plan.get("fix_package", "")
                if pkg:
                    print(f"  {G}Auto-fix: pip install {pkg}...{RST}")
                    r = subprocess.run(
                        [sys.executable, "-m", "pip", "install", pkg, "-q"],
                        capture_output=True, text=True, timeout=120
                    )
                    if r.returncode == 0:
                        issue.fix_result = f"pip install {pkg} SUCCESS"
                        print(f"  {G}OK {issue.fix_result}{RST}")
                        self._fixes_done += 1
                    else:
                        issue.fix_result = f"pip install {pkg} FAILED: {r.stderr[:80]}"
                        print(f"  {R}FAIL {issue.fix_result}{RST}")

            elif fix_type == "log_rotate":
                self._rotate_logs()
                issue.fix_result = "Log rotated"
                print(f"  {G}OK {issue.fix_result}{RST}")
                self._fixes_done += 1

            elif fix_type == "config_fix":
                key = fix_plan.get("fix_env_key", "")
                val = fix_plan.get("fix_env_val", "")
                if key:
                    os.environ[key] = str(val)
                    issue.fix_result = f"ENV {key} updated (runtime)"
                    print(f"  {G}OK {issue.fix_result}{RST}")
                    self._fixes_done += 1

            else:
                issue.fix_result = f"fix_type='{fix_type}' -- manual review needed"
                print(f"  {Y}INFO: {issue.fix_result}{RST}")

        except Exception as e:
            issue.fix_result = f"Fix exception: {e}"
            print(f"  {R}FAIL {issue.fix_result}{RST}")

    def _rotate_logs(self):
        try:
            if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 50 * 1024 * 1024:
                backup = LOG_FILE.replace(".log", f"_rotated_{int(time.time())}.log")
                os.rename(LOG_FILE, backup)
                logger.info(f"[Doctor] Log rotated -> {backup}")
        except Exception as e:
            logger.error(f"[Doctor] Rotate error: {e}")

    def _send_telegram_alert(self, issue: DoctorIssue, fix_plan: Optional[dict]):
        try:
            from telegram_notifier import send_message
            summary = (fix_plan or {}).get("telegram_summary") or issue.message[:100]
            msg = (
                f"JARVIS DOCTOR ALERT\n"
                f"Source: {issue.source}\n"
                f"Level : {issue.level}\n"
                f"Error : {issue.error_type}\n"
                f"Info  : {summary}\n"
                f"Fix   : {issue.fix_result or 'none applied'}"
            )
            send_message(msg)
        except Exception:
            pass

    # ── Console Report ──────────────────────────────────────────────
    def print_doctor_report(self):
        W    = 68
        now  = datetime.now().strftime("%H:%M IST")
        ok_c = sum(1 for v in self._part_status.values() if v == "OK")
        tot  = len(PARTS_REGISTRY)
        pct  = int(ok_c / tot * 100) if tot else 0
        hcol = G if pct >= 80 else Y if pct >= 60 else R

        print(f"\n{C}+{'='*W}+{RST}")
        print(f"{C}|{BD}  JARVIS DOCTOR [{now}]  Health: {hcol}{pct}% ({ok_c}/{tot} parts OK){RST}{C}{' '*(W-45)}|{RST}")
        print(f"{C}+{'-'*W}+{RST}")

        for part in PARTS_REGISTRY:
            nm = part["name"]
            st = self._part_status.get(nm, "UNKNOWN")
            if   st == "OK":              sym = f"{G}OK  {RST}"
            elif st in ("STALE","SILENT"): sym = f"{Y}WARN{RST}"
            else:                          sym = f"{R}FAIL{RST}"
            pad = max(0, W - len(nm) - 12)
            print(f"{C}|{RST}  {sym} {nm}{' '*pad}{C}|{RST}")

        if self._recent:
            print(f"{C}+{'-'*W}+{RST}")
            print(f"{C}|{BD}  RECENT ISSUES{RST}{C}{' '*(W-14)}|{RST}")
            for iss in list(self._recent)[-4:]:
                lcol = R if iss.level == "CRITICAL" else Y
                line = f"  {lcol}[{iss.level}]{RST} {iss.source}: {iss.error_type}"
                if iss.fix_result:
                    line += f"  -> {G}{iss.fix_result[:30]}{RST}"
                print(f"{C}|{RST}{line}")

        print(f"{C}+{'='*W}+{RST}")
        print(f"{C}|{RST}  Checks done: {self._checks_done}  |  Auto-fixes: {G}{self._fixes_done}{RST}{C}{' '*(W-38)}|{RST}")
        print(f"{C}+{'='*W}+{RST}\n")

    # ── Main Loop ───────────────────────────────────────────────────
    def _run_loop(self):
        logger.info(f"[Doctor] Loop started (interval={DOCTOR_INTERVAL}s).")
        first = True
        while self.is_running:
            try:
                self._checks_done += 1
                issues: List[DoctorIssue] = []

                issues.extend(self._check_parts_health())
                issues.extend(self._scan_log_file()[:3])       # cap 3/cycle
                issues.extend(self._check_system_resources())
                if self._checks_done % 5 == 1:                 # every 5 min
                    issues.extend(self._check_dependencies())

                for iss in issues:
                    self._handle_issue(iss)

                # Print report every 5 min or on first run
                if first or (
                    self._last_report_ts is None or
                    (datetime.now() - self._last_report_ts).total_seconds() >= 300
                ):
                    self.print_doctor_report()
                    self._last_report_ts = datetime.now()
                    first = False

            except Exception as e:
                logger.error(f"[Doctor] Loop error: {e}\n{traceback.format_exc()}")

            time.sleep(DOCTOR_INTERVAL)

    # ── Start / Stop ────────────────────────────────────────────────
    def start(self) -> "DoctorMonitor":
        if not DOCTOR_ENABLED:
            logger.info("[Doctor] Disabled (DOCTOR_ENABLED=false).")
            return self
        if self.is_running:
            return self
        self.is_running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="JarvisDoctor"
        )
        self._thread.start()
        logger.info("[Doctor] JARVIS Self-Healing Doctor ONLINE.")
        return self

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)


# ── Singleton ───────────────────────────────────────────────────────
_doctor_instance: Optional[DoctorMonitor] = None

def init_doctor(bus=None, live_trader=None) -> DoctorMonitor:
    global _doctor_instance
    if _doctor_instance is None:
        _doctor_instance = DoctorMonitor(bus=bus, live_trader=live_trader)
        _doctor_instance.start()
    return _doctor_instance

def get_doctor() -> Optional[DoctorMonitor]:
    return _doctor_instance


# ── Standalone Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"  JARVIS SELF-HEALING DOCTOR -- Standalone Diagnostic Mode")
    print(f"  API: {DOCTOR_MODEL}  Key: {'SET' if DOCTOR_API_KEY else 'NOT SET'}")
    print(f"{'='*70}\n")

    doc = DoctorMonitor(bus=None)
    doc.start()
    print("Doctor is running. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nDoctor stopped.")
        doc.stop()

