"""Centralized, honest runtime backend detection for Jarvis.

The result describes the backend selected by this process.  It deliberately does
not claim that a remote LegacyLLM server or every optional engine uses that
backend; those are reported separately by their owners.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import logging
import os

logger = logging.getLogger("JarvisRuntime")


@dataclass(frozen=True)
class RuntimeBackend:
    requested: str
    backend: str
    device: str
    device_name: str
    device_count: int
    memory_total_mb: Optional[int]
    detected: bool
    fallback_reason: Optional[str] = None
    torch_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def detect_backend(requested: Optional[str] = None) -> RuntimeBackend:
    """Select CUDA, then MPS, then CPU without inventing a GPU.

    ``JARVIS_DEVICE=auto`` (the default) is the only mode that silently chooses
    a supported accelerator.  Explicit unavailable requests fall back to CPU
    and preserve a reason for telemetry.
    """
    requested = (requested or os.getenv("JARVIS_DEVICE", "auto")).strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        requested = "auto"
    torch = _torch()
    if torch is None:
        return RuntimeBackend(requested, "cpu", "cpu", "cpu", 0, None, False,
                              "PyTorch unavailable", False)

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_ok else 0
    mps_ok = False
    try:
        mps_ok = bool(torch.backends.mps.is_available())
    except Exception:
        pass

    chosen = "cpu"
    reason = None
    if requested == "cpu":
        chosen = "cpu"
    elif requested == "cuda":
        if cuda_ok:
            chosen = "cuda"
        else:
            reason = "CUDA requested but unavailable"
    elif requested == "mps":
        if mps_ok:
            chosen = "mps"
        else:
            reason = "MPS requested but unavailable"
    elif cuda_ok:
        chosen = "cuda"
    elif mps_ok:
        chosen = "mps"
    else:
        reason = "No supported accelerator detected"

    name = "cpu"
    memory_mb = None
    if chosen == "cuda":
        try:
            idx = 0
            name = str(torch.cuda.get_device_name(idx))
            memory_mb = int(torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024))
        except Exception as exc:
            name = "cuda"
            reason = reason or f"CUDA telemetry unavailable: {type(exc).__name__}"
    elif chosen == "mps":
        name = "Apple MPS"

    device = chosen if chosen != "cuda" else "cuda:0"
    return RuntimeBackend(requested, chosen, device, name, cuda_count,
                          memory_mb, chosen != "cpu", reason, True)


def torch_device(runtime: Optional[RuntimeBackend] = None):
    """Return a real torch.device when possible, otherwise the safe string."""
    runtime = runtime or detect_backend()
    torch = _torch()
    if torch is None:
        return runtime.device
    try:
        return torch.device(runtime.device)
    except Exception:
        return torch.device("cpu")


def safe_device(value: Any, fallback: str = "cpu"):
    """Normalize strings and device-like objects before native engine use."""
    torch = _torch()
    if torch is None:
        return fallback
    try:
        if hasattr(value, "type"):
            return value
        return torch.device(str(value or fallback))
    except Exception:
        return torch.device(fallback)


def runtime_status() -> Dict[str, Any]:
    """Small telemetry-safe status object for dashboard/LegacyLLM context."""
    return detect_backend().to_dict()
