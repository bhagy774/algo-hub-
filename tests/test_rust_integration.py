import importlib

import pytest

import jarvis_rust_integration as integration


def test_disabled_path_matches_python_reference(monkeypatch):
    monkeypatch.setenv("JARVIS_RUST_MATH", "0")
    module = importlib.reload(integration)
    assert module.rust_math_enabled() is False
    assert module.calculate_sma([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)


def test_enabled_path_uses_rust_or_logs_explicit_fallback(monkeypatch, caplog):
    monkeypatch.setenv("JARVIS_RUST_MATH", "1")
    module = importlib.reload(integration)
    assert module.rust_math_enabled() is True
    with caplog.at_level("WARNING"):
        assert module.calculate_sma([1, 2, 3, 4, 5], 3) == pytest.approx(4.0)
    try:
        import jarvis_rust  # noqa: F401
        if not hasattr(jarvis_rust, "calculate_sma"):
            raise ImportError("jarvis_rust is uncompiled")
    except ImportError:
        assert "jarvis_rust is unavailable" in caplog.text
    else:
        import jarvis_rust
        if callable(getattr(jarvis_rust, "calculate_sma", None)):
            assert "jarvis_rust is unavailable" not in caplog.text
        else:
            assert "jarvis_rust is unavailable" in caplog.text
    monkeypatch.setenv("JARVIS_RUST_MATH", "0")
    importlib.reload(integration)


def test_invalid_data_never_falls_back(monkeypatch):
    monkeypatch.setenv("JARVIS_RUST_MATH", "1")
    module = importlib.reload(integration)
    with pytest.raises(ValueError):
        module.calculate_sma([1, float("nan"), 3], 3)
    with pytest.raises(ValueError):
        module.calculate_sma([1, 2], 3)
    monkeypatch.setenv("JARVIS_RUST_MATH", "0")
    importlib.reload(integration)
