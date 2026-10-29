"""Shared bounded runtime validation/CPU-kernel for twelve task-specific models.

Part-specific features, architecture contracts, heads and targets are versioned
separately per task. Runtime has no PyTorch/CUDA/network/model-construction path.
Only validated local JSON weights can produce diagnostic, never-authoritative
advisories; central Jarvis policy retains all strategy and execution authority.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

FORMAT = "jarvis-advisory-task-model"
FORMAT_VERSION = 3
FEATURE_SCHEMA = "jarvis-task-specific-features-v4-quantitative"
from neural_parts import PARTS, FEATURE_NAMES, MODEL_SPECS, TASKS, prepare_features
from neural_parts.common import TIMEFRAMES, mtf_signal_matrix as _mtf_signal_matrix
_TIMEFRAME_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}
_MAX_ARTIFACT_BYTES = 128 * 1024
_MAX_ARTIFACTS = 16
_MAX_PREDICTIONS = 256
# Re-train rather than keep using a predictor whose source history is too old.
_MAX_TRAINING_DATA_AGE_SECONDS = 90 * 24 * 60 * 60
_MAX_NUMERIC_ARTIFACT_VALUE = 1_000_000.0
_LOCK = threading.RLock()
_INFERENCE_SLOT = threading.BoundedSemaphore(1)
_ARTIFACT_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_PREDICTION_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def feature_vector(part_id: str, data: Any) -> Tuple[float, ...]:
    """Dispatch to the named Part-owned feature projection."""
    if part_id not in FEATURE_NAMES:
        raise ValueError("unknown_part_id")
    values = prepare_features(part_id, data)
    if len(values) != len(FEATURE_NAMES[part_id]) or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("feature_schema_internal_error")
    return tuple(float(v) for v in values)


def _identity(context: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    identity = context.get("analysis_identity") or context.get("identity")
    if isinstance(identity, Mapping):
        raw = (identity.get("venue"), identity.get("market_type"), identity.get("instrument_id"), identity.get("symbol"))
    elif isinstance(identity, (list, tuple)) and len(identity) == 4:
        raw = tuple(identity)
    else:
        return None
    venue, market_type, instrument_id, symbol = (str(x or "").strip() for x in raw)
    tf = str(context.get("timeframe") or "").strip()
    expected_symbol = str(context.get("selected_symbol") or context.get("symbol") or "").strip()
    if not all((venue, market_type, instrument_id, symbol, tf)) or tf not in _TIMEFRAME_SECONDS:
        return None
    if venue.lower() in ("unknown", "none", "null") or market_type.lower() in ("unknown", "none", "null"):
        return None
    if symbol.upper() != expected_symbol.upper():
        return None
    return {"venue": venue.lower(), "market_type": market_type.lower(),
            "instrument_id": instrument_id, "symbol": symbol.upper(), "timeframe": tf}


def validate_mtf_evidence(evidence: Any, identity: Mapping[str, str],
                         analysis_timestamp: float, *, now: Optional[float] = None,
                         is_backtest: bool = False) -> None:
    """Fail closed on incomplete, cross-symbol or stale 8×10 Part matrices."""
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("frames"), Mapping):
        raise ValueError("mtf_evidence_envelope_required")
    expected_identity = {k: identity[k] for k in ("venue", "market_type", "instrument_id", "symbol")}
    provided = evidence.get("identity")
    if not isinstance(provided, Mapping) or dict(provided) != expected_identity:
        raise ValueError("mtf_evidence_identity_mismatch")
    if str(evidence.get("symbol", "")).upper() != identity["symbol"].upper():
        raise ValueError("mtf_evidence_symbol_mismatch")
    timestamps = evidence.get("closed_timestamps")
    frames = evidence["frames"]
    if not isinstance(timestamps, Mapping) or set(timestamps) != set(TIMEFRAMES):
        raise ValueError("mtf_evidence_timestamp_coverage_invalid")
    if set(frames) != set(TIMEFRAMES):
        raise ValueError("incomplete_eight_frame_evidence")
    evaluation_time = now if now is not None else time.time()
    for tf in TIMEFRAMES:
        stamp = _timestamp_epoch(timestamps[tf])
        interval = _TIMEFRAME_SECONDS[tf]
        if stamp > analysis_timestamp + interval + 2.0:
            raise ValueError("mtf_frame_timestamp_after_analysis")
        if not is_backtest and (stamp > evaluation_time + 30.0 or evaluation_time - stamp > 3 * interval + 120.0):
            raise ValueError("mtf_frame_stale_or_future")
        frame = frames[tf]
        if not isinstance(frame, Mapping):
            raise ValueError("invalid_frame_evidence")
        for part in PARTS[:10]:
            result = frame.get(part)
            if not isinstance(result, Mapping):
                raise ValueError("missing_part_evidence")
            if result.get("symbol") and str(result["symbol"]).upper() != identity["symbol"].upper():
                raise ValueError("evidence_symbol_mismatch")
            if result.get("timeframe") and str(result["timeframe"]) != tf:
                raise ValueError("evidence_timeframe_mismatch")
    _mtf_signal_matrix(evidence)


def build_mtf_advisory_evidence(part_results: Any, candle_frames: Any,
                                context: Mapping[str, Any]) -> Dict[str, Any]:
    """Create an isolated envelope without mutating deterministic Part results."""
    identity = _identity(context)
    if identity is None or not isinstance(part_results, Mapping) or not isinstance(candle_frames, Mapping):
        raise ValueError("mtf_evidence_context_invalid")
    if set(part_results) != set(TIMEFRAMES) or set(candle_frames) != set(TIMEFRAMES):
        raise ValueError("incomplete_eight_frame_evidence")
    stamps = {}
    for tf in TIMEFRAMES:
        frame = candle_frames[tf]
        if not hasattr(frame, "index") or not len(frame.index):
            raise ValueError("mtf_closed_frame_timestamp_missing")
        stamps[tf] = _timestamp_epoch(frame.index[-1])
    payload = {"identity": {k: identity[k] for k in ("venue", "market_type", "instrument_id", "symbol")},
               "symbol": identity["symbol"], "closed_timestamps": stamps,
               "frames": {tf: dict(part_results[tf]) for tf in TIMEFRAMES}}
    return payload


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", value)[:80] or "_"


def artifact_path(identity: Mapping[str, str], part_id: str, root: Optional[str] = None) -> Path:
    if part_id not in FEATURE_NAMES:
        raise ValueError("unknown_part_id")
    folder = Path(root or os.environ.get("JARVIS_NEURAL_ARTIFACT_DIR", "models/advisory"))
    name = "__".join([part_id, identity["venue"], identity["market_type"], identity["instrument_id"], identity["symbol"], identity["timeframe"]])
    return folder / (_slug(name) + ".json")


def _finite_vector(value: Any, expected: int, name: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError("invalid_" + name)
    out = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in out):
        raise ValueError("non_finite_" + name)
    if any(abs(x) > _MAX_NUMERIC_ARTIFACT_VALUE for x in out):
        raise ValueError("artifact_numeric_value_out_of_bounds")
    return out


def _load_artifact(path: Path, part_id: str, identity: Mapping[str, str]) -> Dict[str, Any]:
    st = path.stat()
    if st.st_size <= 0 or st.st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError("artifact_size_invalid")
    raw = path.read_bytes()
    if len(raw) != st.st_size:
        raise ValueError("artifact_changed_during_read")
    digest = hashlib.sha256(raw).hexdigest()
    # Identity is part of the cache key as well as an artifact field check.
    # This prevents a colliding/truncated path from reusing another identity's
    # already-cached model without re-validating its metadata.
    key = (str(path.resolve()), digest, st.st_size, part_id, tuple(sorted(identity.items())))
    with _LOCK:
        cached = _ARTIFACT_CACHE.get(key)
        if cached is not None:
            _ARTIFACT_CACHE.move_to_end(key)
            return cached
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or obj.get("format") != FORMAT or obj.get("format_version") != FORMAT_VERSION:
        raise ValueError("artifact_format_unsupported")
    if obj.get("feature_schema") != FEATURE_SCHEMA or obj.get("part_id") != part_id:
        raise ValueError("artifact_feature_schema_or_part_mismatch")
    if obj.get("identity") != dict(identity):
        raise ValueError("artifact_identity_mismatch")
    if obj.get("feature_names") != list(FEATURE_NAMES[part_id]):
        raise ValueError("artifact_feature_names_mismatch")
    spec = MODEL_SPECS[part_id]
    contract = {"task": spec["task"], "dims": list(spec["dims"]),
                "activations": list(spec["acts"]), "labels": list(spec["labels"]), "kind": spec["kind"]}
    if obj.get("model_contract") != contract:
        raise ValueError("artifact_model_contract_mismatch")
    norm = obj.get("normalization")
    if not isinstance(norm, dict):
        raise ValueError("artifact_normalization_missing")
    input_dim = len(FEATURE_NAMES[part_id])
    mean = _finite_vector(norm.get("mean"), input_dim, "normalization_mean")
    std = _finite_vector(norm.get("std"), input_dim, "normalization_std")
    if any(x <= 0 for x in std):
        raise ValueError("artifact_normalization_std_invalid")
    layers = obj.get("layers")
    dims, acts = spec["dims"], spec["acts"]
    if not isinstance(layers, list) or len(layers) != len(dims) - 1:
        raise ValueError("artifact_architecture_invalid")
    parsed_layers = []
    for layer, in_dim, out_dim, act in zip(layers, dims[:-1], dims[1:], acts):
        if not isinstance(layer, dict) or layer.get("activation") != act:
            raise ValueError("artifact_activation_invalid")
        weights = layer.get("weights")
        if not isinstance(weights, list) or len(weights) != out_dim:
            raise ValueError("artifact_weights_invalid")
        w = tuple(_finite_vector(row, in_dim, "weights") for row in weights)
        b = _finite_vector(layer.get("bias"), out_dim, "bias")
        parsed_layers.append((w, b, act))

    use_gpu = os.environ.get("JARVIS_NEURAL_GPU", "0") == "1"
    gpu_layers = None
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_layers = []
                for w, b, act in parsed_layers:
                    t_w = torch.tensor(w, dtype=torch.float32, device=device)
                    t_b = torch.tensor(b, dtype=torch.float32, device=device)
                    gpu_layers.append((t_w, t_b, act))
        except Exception:
            gpu_layers = None

    training = obj.get("training")
    if not isinstance(training, dict) or training.get("status") != "validated" or training.get("split") != "chronological_purged" or training.get("approved_for_advisory") is not True:
        raise ValueError("artifact_not_validation_gated")
    digest = training.get("dataset_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("artifact_dataset_provenance_missing")
    if not isinstance(training.get("trainer_version"), str) or not training.get("trainer_version"):
        raise ValueError("artifact_trainer_version_missing")
    try:
        trained_at = datetime.fromisoformat(str(training.get("trained_at_utc")).replace("Z", "+00:00"))
        if trained_at.tzinfo is None:
            raise ValueError
    except Exception as exc:
        raise ValueError("artifact_training_time_invalid") from exc
    try:
        dataset_last_timestamp = float(training.get("dataset_last_timestamp"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("artifact_dataset_cutoff_missing") from exc
    if not math.isfinite(dataset_last_timestamp) or dataset_last_timestamp <= 0:
        raise ValueError("artifact_dataset_cutoff_invalid")
    trained_epoch = trained_at.timestamp()
    interval = _TIMEFRAME_SECONDS[identity["timeframe"]]
    if dataset_last_timestamp > trained_epoch + interval:
        raise ValueError("artifact_dataset_cutoff_after_training")
    for key_name in ("train_samples", "validation_samples", "test_samples"):
        value = training.get(key_name)
        if not isinstance(value, int) or value < (200 if key_name == "train_samples" else 50):
            raise ValueError("artifact_sample_count_below_minimum")
    for m in ("validation_accuracy", "validation_majority_accuracy", "test_accuracy", "test_majority_accuracy"):
        value = training.get(m)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("artifact_metrics_invalid")
    if (training["validation_accuracy"] < training["validation_majority_accuracy"] + 0.02
            or training["test_accuracy"] < training["test_majority_accuracy"] + 0.02):
        raise ValueError("artifact_out_of_sample_gate_failed")
    model = {"mean": mean, "std": std, "layers": tuple(parsed_layers), "training": training}
    if gpu_layers is not None:
        model["gpu_layers"] = tuple(gpu_layers)
    with _LOCK:
        _ARTIFACT_CACHE[key] = model
        _ARTIFACT_CACHE.move_to_end(key)
        while len(_ARTIFACT_CACHE) > _MAX_ARTIFACTS:
            _ARTIFACT_CACHE.popitem(last=False)
    return model


def _index_time(data: Any) -> Optional[float]:
    try:
        if isinstance(data, (list, tuple)) and data:
            value = data[-1].get("timestamp", data[-1].get("time"))
        elif hasattr(data, "index") and not callable(data.index) and len(data.index):
            value = data.index[-1]
        else:
            return None
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value if value > 1e12 else value * 1000.0)
        if isinstance(value, str):
            text = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    except Exception:
        return None
    return None


def _timestamp_epoch(value: Any) -> float:
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        stamp = float(value)
        return stamp / 1000.0 if stamp > 1e12 else stamp
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    raise ValueError("closed_candle_timestamp_missing")


def _validate_closed_series(data: Any, interval: int) -> None:
    if isinstance(data, (list, tuple)):
        if len(data) < 64 or any(not isinstance(row, Mapping) for row in data[-64:]):
            raise ValueError("insufficient_closed_candles")
        tail = data[-64:]
        stamps = [_timestamp_epoch(row.get("timestamp", row.get("time"))) for row in tail]
        closed_values = [row.get("is_closed", True) for row in tail]
    elif hasattr(data, "index") and not callable(data.index):
        tail = data.tail(64)
        if len(tail) < 64:
            raise ValueError("insufficient_closed_candles")
        stamps = [_timestamp_epoch(value) for value in tail.index]
        closed_values = tail["is_closed"].tolist() if "is_closed" in tail.columns else [True] * 64
    else:
        raise ValueError("closed_candle_timestamp_missing")
    if not all(math.isfinite(stamp) for stamp in stamps):
        raise ValueError("closed_candle_timestamp_invalid")
    if any(current <= previous or abs((current - previous) - interval) > 2.0
           for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("closed_candle_sequence_invalid")
    if any(str(value).strip().lower() not in ("1", "true", "yes")
           for value in closed_values):
        raise ValueError("forming_candle_rejected")
    # Reject non-finite raw OHLCV explicitly before quantitative projection so
    # malformed bars cannot be misreported as merely insufficient history.
    if isinstance(data, (list, tuple)):
        for row in data[-64:]:
            for field in ("open", "high", "low", "close", "volume"):
                raw = row.get(field, 0.0)
                try:
                    value = float(raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("invalid_ohlcv") from exc
                if not math.isfinite(value):
                    raise ValueError("non_finite_ohlcv")


def _softmax(logits: Sequence[float]) -> Tuple[float, ...]:
    if len(logits) < 2 or not all(math.isfinite(x) for x in logits):
        raise ValueError("non_finite_inference")
    m = max(logits)
    exps = [math.exp(max(-80.0, min(80.0, x - m))) for x in logits]
    den = sum(exps)
    return tuple(x / den for x in exps)


def _infer(model: Mapping[str, Any], vector: Sequence[float], part_id: str) -> Tuple[float, ...]:
    use_gpu = os.environ.get("JARVIS_NEURAL_GPU", "0") == "1"
    run_on_cpu = True

    if use_gpu and "gpu_layers" in model:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")

                # Preprocess input (normalize)
                x = [(v - m) / s for v, m, s in zip(vector, model["mean"], model["std"])]
                x = [max(-8.0, min(8.0, v)) for v in x]
                t_x = torch.tensor(x, dtype=torch.float32, device=device)

                # Forward pass on GPU using cached tensors
                for t_w, t_b, activation in model["gpu_layers"]:
                    t_x = torch.matmul(t_x, t_w.t()) + t_b

                    if activation == "relu":
                        t_x = torch.relu(t_x)
                    elif activation == "tanh":
                        t_x = torch.tanh(t_x)
                    elif activation != "linear":
                        raise ValueError("unsupported_artifact_activation")

                x = t_x.cpu().tolist()

                if not all(math.isfinite(v) for v in x):
                    raise ValueError("non_finite_inference")

                run_on_cpu = False
        except Exception:
            # Fall back to CPU on ANY PyTorch/CUDA runtime error or ImportError
            run_on_cpu = True

    if run_on_cpu:
        x = [max(-8.0, min(8.0, (v - m) / s)) for v, m, s in zip(vector, model["mean"], model["std"])]
        for weights, bias, activation in model["layers"]:
            x = [sum(wi * xi for wi, xi in zip(row, x)) + b for row, b in zip(weights, bias)]
            if not all(math.isfinite(v) for v in x):
                raise ValueError("non_finite_inference")
            if activation == "relu":
                x = [max(0.0, v) for v in x]
            elif activation == "tanh":
                x = [math.tanh(v) for v in x]
            elif activation != "linear":
                raise ValueError("unsupported_artifact_activation")

    spec = MODEL_SPECS[part_id]
    if spec["kind"] == "sigmoid":
        if len(x) != 1:
            raise ValueError("binary_head_shape_invalid")
        z = max(-80.0, min(80.0, x[0]))
        probability_like_score = 1.0 / (1.0 + math.exp(-z))
        return (1.0 - probability_like_score, probability_like_score)
    scores = _softmax(x)
    if len(scores) != len(spec["labels"]):
        raise ValueError("classification_head_shape_invalid")
    return scores


def predict_advisory(data: Any, part_id: str, context: Optional[Mapping[str, Any]] = None,
                     *, now: Optional[float] = None, artifact_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return an explicit status; unavailable/malformed artifacts never infer."""
    c = context if isinstance(context, Mapping) else {}
    if part_id not in FEATURE_NAMES:
        return {"status": "unavailable", "reason": "unknown_part_id"}
    identity = _identity(c)
    if identity is None:
        return {"status": "unavailable", "reason": "full_analysis_identity_required"}
    if not bool(c.get("is_backtest_mode")):
        fetched = c.get("snapshot_fetched_at")
        try:
            fetched = float(fetched)
        except (TypeError, ValueError, OverflowError):
            return {"status": "unavailable", "reason": "snapshot_freshness_missing"}
        if not math.isfinite(fetched) or abs((now if now is not None else time.time()) - fetched) > 180.0:
            return {"status": "unavailable", "reason": "stale_snapshot"}
    is_mtf_model = part_id in ("part11_fusion", "part12_confidence")
    try:
        ts = _timestamp_epoch(c.get("closed_candle_timestamp")) if c.get("closed_candle_timestamp") is not None else _index_time(data)
    except Exception:
        ts = None
    if ts is None:
        return {"status": "unavailable", "reason": "closed_candle_timestamp_missing"}
    is_backtest = bool(c.get("is_backtest_mode"))
    if not is_backtest:
        interval = _TIMEFRAME_SECONDS[identity["timeframe"]]
        if (now if now is not None else time.time()) - ts > 3 * interval + 120 or ts > (now if now is not None else time.time()) + 30:
            return {"status": "unavailable", "reason": "closed_candles_stale_or_future"}
    try:
        if not is_mtf_model:
            _validate_closed_series(data, _TIMEFRAME_SECONDS[identity["timeframe"]])
        else:
            # Each separate MTF head requires provenance, all eight closed
            # frames, ten named Part outputs per frame, and current timestamps.
            validate_mtf_evidence(data, identity, ts, now=now, is_backtest=is_backtest)
        attrs = getattr(data, "attrs", {}) or {}
        if attrs.get("symbol") and str(attrs["symbol"]).upper() != identity["symbol"]:
            return {"status": "unavailable", "reason": "frame_symbol_mismatch"}
        if attrs.get("timeframe") and str(attrs["timeframe"]) != identity["timeframe"]:
            return {"status": "unavailable", "reason": "frame_timeframe_mismatch"}
        vector = feature_vector(part_id, data)
        path = artifact_path(identity, part_id, artifact_dir)
        try:
            st = path.stat()
        except FileNotFoundError:
            return {"status": "unavailable", "reason": "trained_artifact_missing"}
        cache_key = (part_id, tuple(identity.items()), ts, tuple(vector), str(path.resolve()), st.st_mtime_ns, st.st_size)
        with _LOCK:
            cached = _PREDICTION_CACHE.get(cache_key)
            if cached is not None:
                _PREDICTION_CACHE.move_to_end(cache_key)
                return dict(cached)
        model = _load_artifact(path, part_id, identity)
        evaluation_time = now if now is not None else time.time()
        training = model["training"]
        trained_at = datetime.fromisoformat(str(training["trained_at_utc"]).replace("Z", "+00:00")).timestamp()
        dataset_last_timestamp = float(training["dataset_last_timestamp"])
        if trained_at > evaluation_time + 300.0:
            return {"status": "unavailable", "reason": "artifact_training_time_in_future"}
        if ts <= dataset_last_timestamp:
            return {"status": "unavailable", "reason": "artifact_training_data_overlap"}
        if is_backtest and trained_at > ts:
            return {"status": "unavailable", "reason": "artifact_not_available_at_backtest_time"}
        if not is_backtest and evaluation_time - dataset_last_timestamp > _MAX_TRAINING_DATA_AGE_SECONDS:
            return {"status": "unavailable", "reason": "stale_artifact_training_data"}
        with _INFERENCE_SLOT:
            # Recheck inside the single inference slot so concurrent repeats of
            # one identity/candle cannot trigger duplicate model executions.
            with _LOCK:
                cached = _PREDICTION_CACHE.get(cache_key)
                if cached is not None:
                    _PREDICTION_CACHE.move_to_end(cache_key)
                    return dict(cached)
            scores = _infer(model, vector, part_id)
            spec = MODEL_SPECS[part_id]
            labels = spec["labels"]
            max_score = max(scores)
            winning_index = max(range(len(scores)), key=lambda i: scores[i])
            prediction = labels[winning_index] if max_score >= 0.55 else "uncertain"
            result = {"status": "available", "task": spec["task"],
                      "prediction": prediction,
                      "scores_uncalibrated": {label: scores[i] for i, label in enumerate(labels)},
                      "head_kind": spec["kind"],
                      "role": "advisory_only_no_execution_authority",
                      "model_version": str(model["training"].get("model_version", "3")),
                      "task_output": TASKS[part_id].interpret_scores(scores, prediction)}
            if part_id == "part11_fusion":
                result["direction"] = {"sell": "SELL", "neutral": "NEUTRAL", "buy": "BUY"}.get(prediction, "NEUTRAL")
            if part_id == "part12_confidence":
                result["consensus_correctness_score_uncalibrated"] = scores[1]
            with _LOCK:
                _PREDICTION_CACHE[cache_key] = result
                _PREDICTION_CACHE.move_to_end(cache_key)
                while len(_PREDICTION_CACHE) > _MAX_PREDICTIONS:
                    _PREDICTION_CACHE.popitem(last=False)
        return dict(result)
    except Exception as exc:
        reason = str(exc) if str(exc) and re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", str(exc)) else "artifact_or_feature_validation_failed"
        return {"status": "unavailable", "reason": reason}


def annotate_part_result(result: Any, data: Any, part_id: str,
                         context: Optional[Mapping[str, Any]] = None,
                         *, artifact_dir: Optional[str] = None,
                         now: Optional[float] = None) -> Any:
    """Attach diagnostic-only model metadata without touching deterministic fields."""
    if not isinstance(result, dict):
        return result
    out = dict(result)
    out["neural_advisory"] = predict_advisory(data, part_id, context, now=now, artifact_dir=artifact_dir)
    return out


def annotate_mtf_result(result: Any, evidence: Any, part_id: str,
                        context: Optional[Mapping[str, Any]] = None,
                        *, artifact_dir: Optional[str] = None,
                        now: Optional[float] = None) -> Any:
    """Attach a task-specific advisory to Parts 11/12 from the full 8×10 matrix."""
    if part_id not in ("part11_fusion", "part12_confidence"):
        raise ValueError("mtf_advisory_only_for_parts_11_12")
    if not isinstance(result, dict):
        return result
    out = dict(result)
    out["neural_advisory"] = predict_advisory(evidence, part_id, context, now=now, artifact_dir=artifact_dir)
    return out


def not_applicable_result(part_id: str) -> Dict[str, str]:
    return {"status": "unavailable", "reason": f"{part_id}_requires_trained_task_artifact",
            "role": "advisory_only_no_execution_authority"}


def clear_caches() -> None:
    """Test/support hook; caches are bounded regardless of caller use."""
    with _LOCK:
        _ARTIFACT_CACHE.clear()
        _PREDICTION_CACHE.clear()


def part_advisory_entry(part_id: str, *, data_parameter: str = "data",
                        context_parameter: str = "context",
                        evidence_context_key: Optional[str] = None):
    """Attach diagnostic advisory metadata after the deterministic Part returns.

    The wrapped analyzer remains the sole source of its original result. Missing,
    stale, invalid, or failed neural artifacts are surfaced as unavailable; no
    advisory value can rewrite deterministic signal/confidence fields.
    """
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            if not isinstance(result, dict):
                return result
            output = dict(result)
            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                data = bound.arguments.get(data_parameter)
                context = bound.arguments.get(context_parameter)
                if not isinstance(context, Mapping):
                    context = {}
                advisory_data = (context.get(evidence_context_key)
                                 if evidence_context_key else data)
                output["neural_advisory"] = predict_advisory(
                    advisory_data, part_id, context
                )
            except Exception:
                output["neural_advisory"] = {
                    "status": "unavailable",
                    "reason": "advisory_inference_error",
                }
            return output

        return wrapped
    return decorate
