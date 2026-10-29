"""Offline tests for the clean output mode fix (JARVIS_OUTPUT_MODE)."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_clean_mode_is_default():
    """Default output mode must be clean (single decision), not verbose."""
    import jarvis_FIXED as jf
    assert jf.JARVIS_OUTPUT_MODE == "clean"
    assert jf.JARVIS_VERBOSE is False


def test_run_quietly_suppresses_chatter_and_returns_result(tmp_path, monkeypatch):
    """In clean mode, engine chatter is captured to log, not printed."""
    import jarvis_FIXED as jf
    monkeypatch.chdir(tmp_path)

    def noisy_engine():
        print("PART 1: BULLISH")
        print("PART 2: NEURAL PATTERN xyz")
        return {"direction": "CALL", "confidence": 80}

    captured = io.StringIO()
    old = sys.stdout
    sys.stdout = captured
    try:
        result = jf._run_quietly(noisy_engine)
    finally:
        sys.stdout = old

    assert result == {"direction": "CALL", "confidence": 80}
    assert captured.getvalue() == ""  # terminal par koi chatter nahi
    log_file = tmp_path / "logs" / "engine_chatter.log"
    assert log_file.exists()
    assert "PART 1: BULLISH" in log_file.read_text()


def test_vprint_silent_in_clean_mode():
    import jarvis_FIXED as jf
    captured = io.StringIO()
    old = sys.stdout
    sys.stdout = captured
    try:
        jf._vprint("hidden detail")
    finally:
        sys.stdout = old
    assert captured.getvalue() == ""


def test_verbose_mode_shows_everything(monkeypatch):
    """With JARVIS_OUTPUT_MODE=verbose, chatter prints normally."""
    monkeypatch.setenv("JARVIS_OUTPUT_MODE", "verbose")
    import importlib
    import jarvis_FIXED as jf
    importlib.reload(jf)
    try:
        assert jf.JARVIS_VERBOSE is True

        def noisy():
            print("verbose detail")
            return 42

        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            result = jf._run_quietly(noisy)
        finally:
            sys.stdout = old
        assert result == 42
        assert "verbose detail" in captured.getvalue()
    finally:
        monkeypatch.delenv("JARVIS_OUTPUT_MODE", raising=False)
        importlib.reload(jf)


def test_compact_status_method_exists():
    import jarvis_FIXED as jf
    import inspect
    assert any("_print_compact_status" in name for name, _ in
               inspect.getmembers(jf, predicate=inspect.isclass)) or \
           "_print_compact_status" in open(jf.__file__, encoding="utf-8").read()
