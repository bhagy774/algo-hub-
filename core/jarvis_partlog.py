"""Per-Part log files + logging of silently swallowed exceptions.

Behaviour of the callers is unchanged: log_swallowed() never raises.
Files: logs/partN.log (rotating). Each distinct (part, function, line, error type)
is written the 1st time, then every 500th repeat with a counter.
"""
import logging, os, sys, threading, time, traceback
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.environ.get("JARVIS_LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_loggers = {}
_counts = {}
_lock = threading.Lock()


def _part_name(module_name):
    base = (module_name or "unknown").split(".")[-1]
    for suf in ("_FIXED", "_fixed"):
        base = base.replace(suf, "")
    return base


def get_part_logger(module_name):
    name = _part_name(module_name)
    with _lock:
        lg = _loggers.get(name)
        if lg is not None:
            return lg
        lg = logging.getLogger("jarvis.part." + name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            h = RotatingFileHandler(os.path.join(_LOG_DIR, name + ".log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            lg.addHandler(h)
        except Exception:
            lg.addHandler(logging.NullHandler())
        _loggers[name] = lg
        return lg


def log_swallowed(module_name, lineno=0):
    try:
        et, ev, tb = sys.exc_info()
        if et is None or issubclass(et, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            return
        func = "?"
        line = lineno
        if tb is not None:
            last = traceback.extract_tb(tb)[-1]
            func, line = last.name, last.lineno
        key = (_part_name(module_name), func, lineno, et.__name__)
        with _lock:
            c = _counts.get(key, 0) + 1
            _counts[key] = c
        if c == 1 or c % 500 == 0:
            get_part_logger(module_name).warning(
                "swallowed %s in %s() handler@line %s (raised@%s) x%d: %s",
                et.__name__, func, lineno, line, c, str(ev)[:300])
    except Exception:
        pass


class _SafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe RotatingFileHandler that never crashes on WinError 32 during rollover."""
    def doRollover(self):
        try:
            super().doRollover()
        except OSError:
            try:
                if self.stream is None:
                    self.stream = self._open()
            except Exception:
                pass


_TERMINAL_LOG_TAGS = (
    "[MODE_CANDIDATES]",
    "[MODE_FALLBACK]",
    "[SAFE_ENTRY_ENGINE]",
    "[SAFE_ENTRY_ANCHORS]",
    "[CONTINUOUS_REGIME_SUMMARY]",
    "[PART11_12_LIVE]",
    "[XGB_SHADOW]",
    "[XGB_48_FEATURES]",
    "[PORTFOLIO_CORRELATION_GUARD]",
    "[DUAL_COIN]",
    "[INSTITUTIONAL_CONTEXT]",
    "[CANDLE_CACHE_RECONCILE]",
    "[PULLBACK_ZONE_TRACKER]",
    "[ARBITER]",
)


# ---- Per-source-file logging: every logging record from project file X.py goes to logs/files/X.log ----
class _PerFileHandler(logging.Handler):
    def __init__(self, root_dir, out_dir):
        super().__init__(level=logging.INFO)
        self._root = os.path.normcase(os.path.abspath(root_dir))
        self._out = out_dir
        self._h = {}
        self._l = threading.Lock()
        self._fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s")
    def emit(self, record):
        try:
            path = os.path.normcase(os.path.abspath(record.pathname or ""))
            if path.startswith(self._root) and "jenv" not in path and "site-packages" not in path:
                name = os.path.splitext(os.path.basename(path))[0]
            elif record.levelno >= logging.WARNING:
                name = "_external_libs"
            else:
                return
            with self._l:
                h = self._h.get(name)
                if h is None:
                    os.makedirs(self._out, exist_ok=True)
                    h = _SafeRotatingFileHandler(os.path.join(self._out, name + ".log"), maxBytes=10_000_000, backupCount=2, encoding="utf-8", delay=True)
                    h.setFormatter(self._fmt)
                    self._h[name] = h
            h.emit(record)
            try:
                msg = record.getMessage()
                if any(t in msg for t in _TERMINAL_LOG_TAGS):
                    tlog = os.path.join(_LOG_DIR, "jarvis_terminal.log")
                    os.makedirs(_LOG_DIR, exist_ok=True)
                    with open(tlog, "a", encoding="utf-8", errors="replace") as tf:
                        tf.write(self._fmt.format(record) + "\n")
            except Exception:
                pass
        except Exception:
            pass


def install_per_file_logging(root_dir=None):
    """Attach per-file log handler to the root logger. Console level stays unchanged (WARNING)."""
    try:
        root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, _PerFileHandler):
                return
        for h in root.handlers:
            if h.level == logging.NOTSET:
                h.setLevel(root.level if root.level else logging.WARNING)
        root.addHandler(_PerFileHandler(root_dir, os.path.join(_LOG_DIR, "files")))
        root.setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        def _hook(et, ev, tb):
            logging.getLogger("jarvis.uncaught").error("uncaught", exc_info=(et, ev, tb))
            sys.__excepthook__(et, ev, tb)
        sys.excepthook = _hook
    except Exception:
        pass


# ---- print()/stderr capture: every print from project file X.py is also written to logs/files/X.log ----
class _FileAttributedTee:
    _tl = threading.local()
    _handles = {}
    _lock = threading.Lock()
    def __init__(self, stream, root_dir, out_dir, tag):
        self._s = stream
        self._root = os.path.normcase(os.path.abspath(root_dir))
        self._out = out_dir
        self._tag = tag
        self._n = 0
    def __getattr__(self, k):
        return getattr(self._s, k)
    def _who(self):
        try:
            f = sys._getframe(2)
            while f is not None:
                fn = os.path.normcase(os.path.abspath(f.f_code.co_filename))
                if fn.endswith(os.path.join("logging", "__init__.py")):
                    return None
                if fn.startswith(self._root) and "jenv" not in fn and "site-packages" not in fn \
                        and os.path.basename(fn) != "jarvis_partlog.py":
                    return os.path.splitext(os.path.basename(fn))[0]
                f = f.f_back
        except Exception:
            pass
        return "_unattributed"
    def write(self, text):
        n = self._s.write(text)
        try:
            if not text:
                return n
            if text.strip() == "":
                name = getattr(self._tl, "last", None)
            else:
                name = self._who()
                self._tl.last = name
            if name:
                with self._lock:
                    p = os.path.join(self._out, name + ".log")
                    os.makedirs(self._out, exist_ok=True)
                    with open(p, "a", encoding="utf-8", errors="replace") as h:
                        if text.strip() == "":
                            h.write(text)
                        else:
                            h.write("%s %s " % (time.strftime("%Y-%m-%d %H:%M:%S"), self._tag) + text)
        except Exception:
            pass
        return n
    def flush(self):
        try: self._s.flush()
        except Exception: pass


def install_print_capture(root_dir=None):
    try:
        root_dir = root_dir or os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(_LOG_DIR, "files")
        if not isinstance(sys.stdout, _FileAttributedTee):
            sys.stdout = _FileAttributedTee(sys.stdout, root_dir, out, "OUT")
        if not isinstance(sys.stderr, _FileAttributedTee):
            sys.stderr = _FileAttributedTee(sys.stderr, root_dir, out, "ERR")
    except Exception:
        pass
