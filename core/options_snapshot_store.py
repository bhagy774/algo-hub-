"""Append-only point-in-time options snapshots and leakage-safe offline replay.

This module never queries a historical provider API or synthesizes options
history from candles. Captures begin when the runtime fetches chains; a
persistent deployment volume is required to retain them across restarts.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from functools import lru_cache
from threading import Lock
from typing import Any, Iterator, Mapping

from options_chain import parse_instant


_WRITE_LOCK = Lock()
_MAX_ROTATIONS = 3
_DEFAULT_MAX_BYTES = 134_217_728  # 128 MiB per JSONL generation


@contextmanager
def _process_file_lock(path: str) -> Iterator[None]:
    """Serialize append+rotation across OS processes, not just Python threads."""
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_snapshot(path: str, chain: Mapping[str, Any], *, captured_at: datetime | None = None) -> bool:
    """Append a normalized provider snapshot; return False for invalid inputs."""
    if not path or not isinstance(chain, Mapping):
        return False
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        return False
    row = {
        "schema": "jarvis.options.snapshot.v1",
        "captured_at": captured.astimezone(timezone.utc).isoformat(),
        "underlying": str(chain.get("underlying") or "").upper(),
        "providers": chain.get("providers", {}),
    }
    if not row["underlying"] or not isinstance(row["providers"], dict):
        return False
    try:
        payload = (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        maximum = int(os.environ.get("JARVIS_OPTIONS_SNAPSHOT_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))
        if maximum < 1024 or len(payload) > maximum:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with _WRITE_LOCK:
            with _process_file_lock(path):
                try:
                    current_size = os.path.getsize(path)
                except FileNotFoundError:
                    current_size = 0
                if current_size + len(payload) > maximum:
                    oldest = f"{path}.{_MAX_ROTATIONS}"
                    if os.path.exists(oldest):
                        os.remove(oldest)
                    for generation in range(_MAX_ROTATIONS - 1, 0, -1):
                        source = f"{path}.{generation}"
                        target = f"{path}.{generation + 1}"
                        if os.path.exists(source):
                            os.replace(source, target)
                    if os.path.exists(path):
                        os.replace(path, f"{path}.1")
                with open(path, "ab") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        return True
    except (OSError, TypeError, ValueError, ImportError):
        return False


@lru_cache(maxsize=4)
def _load_snapshot_rows(path: str, modified_ns: int, size: int) -> tuple[dict[str, Any], ...]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    return tuple(rows)


def _target_asset(value: Any) -> str:
    text = str(value or "").upper().replace("/", "").replace("-", "").replace("_", "")
    for quote in ("USDT", "USDC", "USD"):
        if text.endswith(quote):
            return text[:-len(quote)]
    return text


def _admissible_provider_names(row: Mapping[str, Any], underlying: str,
                               decision_time: datetime, venue: str | None,
                               max_age_seconds: int) -> list[str]:
    providers = row.get("providers")
    if not isinstance(providers, dict):
        return []
    names = [venue] if venue else list(providers)
    admitted = []
    for name in names:
        provider = providers.get(name)
        if not isinstance(provider, dict):
            continue
        validation = provider.get("validation")
        if (not isinstance(validation, dict) or not validation.get("usable")
                or not validation.get("complete")):
            continue
        if str(provider.get("venue") or name) != str(name):
            continue
        as_of = parse_instant(provider.get("retrieved_at") or validation.get("retrieved_at"))
        if as_of is None or as_of > decision_time or (decision_time - as_of).total_seconds() > max_age_seconds:
            continue
        contracts = provider.get("contracts")
        if not isinstance(contracts, list) or not contracts:
            continue
        valid = True
        for contract in contracts:
            if not isinstance(contract, dict) or _target_asset(contract.get("underlying")) != underlying:
                valid = False
                break
            observed = parse_instant(contract.get("observed_at"))
            if (observed is None or observed > decision_time or observed > as_of
                    or (decision_time - observed).total_seconds() > max_age_seconds):
                valid = False
                break
            try:
                expiry = datetime.fromisoformat(str(contract.get("expiry"))).date()
            except (ValueError, TypeError):
                valid = False
                break
            # Expiry is a date, not an exact settlement instant; same-day
            # contracts are conservatively excluded without venue settlement data.
            if expiry <= decision_time.date():
                valid = False
                break
        if valid:
            admitted.append(str(name))
    return admitted


def _valid_replay_row(row: Mapping[str, Any], underlying: str, decision_time: datetime,
                      venue: str | None, max_age_seconds: int) -> tuple[bool, str, list[str]]:
    if row.get("schema") != "jarvis.options.snapshot.v1":
        return False, "unsupported_snapshot_schema", []
    if _target_asset(row.get("underlying")) != underlying:
        return False, "wrong_underlying", []
    captured = parse_instant(row.get("captured_at"))
    if captured is None or captured > decision_time:
        return False, "future_or_invalid_capture_time", []
    valid_names = _admissible_provider_names(row, underlying, decision_time, venue, max_age_seconds)
    if not valid_names:
        return False, "no_complete_fresh_point_in_time_provider_snapshot", []
    return True, "valid_point_in_time_snapshot", valid_names


def replay_snapshot(path: str, underlying: str, decision_time: datetime, *,
                    venue: str | None = None, max_age_seconds: int = 600) -> dict[str, Any]:
    """Return the newest admissible recorded snapshot at/before decision time."""
    if decision_time.tzinfo is None:
        return {"available": False, "reason": "decision_time_timezone_required", "not_ohlc_derived": True}
    decision = decision_time.astimezone(timezone.utc)
    target = _target_asset(underlying)
    if not path:
        return {"available": False, "reason": "historical_provider_data_unavailable", "not_ohlc_derived": True, "source": "none"}
    candidates = []
    rejected = []
    archives = [os.path.abspath(path)] + [f"{os.path.abspath(path)}.{i}" for i in range(1, _MAX_ROTATIONS + 1)]
    found_archive = False
    try:
        for archive in archives:
            if not os.path.isfile(archive):
                continue
            found_archive = True
            stat = os.stat(archive)
            rows = _load_snapshot_rows(archive, stat.st_mtime_ns, stat.st_size)
            for row in rows:
                ok, reason, valid_names = _valid_replay_row(row, target, decision, venue, max_age_seconds)
                if ok:
                    candidates.append((row, valid_names))
                else:
                    rejected.append(reason)
        if not found_archive:
            return {"available": False, "reason": "historical_provider_data_unavailable", "not_ohlc_derived": True, "source": "none"}
    except OSError:
        return {"available": False, "reason": "snapshot_store_read_failed", "not_ohlc_derived": True, "source": "none"}
    if not candidates:
        return {"available": False, "reason": "no_admissible_historical_options_snapshot", "not_ohlc_derived": True, "source": "none", "rejected_reasons": sorted(set(rejected))}
    selected, valid_names = max(candidates, key=lambda item: parse_instant(item[0].get("captured_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return {"available": True, "snapshot": selected, "admissible_providers": valid_names,
            "not_ohlc_derived": True, "source": "recorded_options_snapshots",
            "rejected_reasons": sorted(set(rejected))}


def replay_institutional_bias(path: str, underlying: str, decision_time: datetime, *,
                              venue: str | None = None, max_age_seconds: int = 600) -> dict[str, Any]:
    """Compute OI/PCR bias from one admissible, complete historical provider chain."""
    replay = replay_snapshot(path, underlying, decision_time, venue=venue,
                             max_age_seconds=max_age_seconds)
    unavailable = {"available": False, "bias": "NEUTRAL", "score": 0,
                   "source": "recorded_options_snapshots", "not_ohlc_derived": True}
    if not replay.get("available"):
        return {**unavailable, "reason": replay.get("reason", "historical_options_unavailable")}
    snapshot = replay["snapshot"]
    providers = snapshot.get("providers", {})
    admitted = set(replay.get("admissible_providers") or [])
    preferred = [venue] if venue else ["Delta", "Binance Options"]
    provider_name = next((name for name in preferred if name in admitted), None)
    provider = providers.get(provider_name) if provider_name else None
    if not isinstance(provider, dict):
        return {**unavailable, "reason": "no_admissible_historical_provider_chain"}

    metrics = provider.get("metrics") if isinstance(provider.get("metrics"), dict) else {}
    contracts = provider.get("contracts")
    if (not isinstance(contracts, list) or not contracts
            or metrics.get("open_interest_complete") is not True
            or metrics.get("put_call_ratio") is None
            or metrics.get("pcr_available") is not True):
        return {**unavailable, "reason": "complete_historical_open_interest_and_pcr_required",
                "source": provider_name}

    target = _target_asset(underlying)
    expiry_rows: dict[str, dict[str, float]] = {}
    for contract in contracts:
        if not isinstance(contract, dict) or _target_asset(contract.get("underlying")) != target:
            return {**unavailable, "reason": "historical_contract_identity_mismatch", "source": provider_name}
        side = contract.get("type")
        if side not in {"call", "put"}:
            return {**unavailable, "reason": "historical_contract_side_missing", "source": provider_name}
        try:
            oi_value = float(contract.get("open_interest"))
        except (TypeError, ValueError, OverflowError):
            return {**unavailable, "reason": "historical_open_interest_incomplete", "source": provider_name}
        if not math.isfinite(oi_value) or oi_value < 0:
            return {**unavailable, "reason": "historical_open_interest_invalid", "source": provider_name}
        expiry = str(contract.get("expiry") or "")
        if not expiry:
            return {**unavailable, "reason": "historical_expiry_missing", "source": provider_name}
        bucket = expiry_rows.setdefault(expiry, {"calls": 0.0, "puts": 0.0})
        bucket["calls" if side == "call" else "puts"] += oi_value

    score = 0
    reasons = []
    for expiry, totals in sorted(expiry_rows.items()):
        if totals["calls"] + totals["puts"] < 100 or totals["calls"] <= 0:
            continue
        pcr = totals["puts"] / totals["calls"]
        if pcr > 1.3:
            score -= 2
            reasons.append(f"[{expiry}] high put/call open-interest ratio")
        elif pcr < 0.65:
            score += 2
            reasons.append(f"[{expiry}] low put/call open-interest ratio")
    bias = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
    pcr = metrics.get("put_call_ratio")
    try:
        pcr = float(pcr)
    except (TypeError, ValueError, OverflowError):
        return {**unavailable, "reason": "historical_pcr_invalid", "source": provider_name}
    if not math.isfinite(pcr) or pcr < 0:
        return {**unavailable, "reason": "historical_pcr_invalid", "source": provider_name}
    max_pain_data = metrics.get("max_pain", {}) if isinstance(metrics.get("max_pain"), dict) else {}
    return {"available": True, "bias": bias, "score": score, "reasons": reasons,
            "pcr": pcr,
            "max_pain": max_pain_data.get("strike") if max_pain_data.get("available") else None,
            "source": provider_name, "retrieved_at": provider.get("retrieved_at"),
            "decision_time": decision_time.astimezone(timezone.utc).isoformat(),
            "snapshot_captured_at": snapshot.get("captured_at"),
            "underlying": target, "expiry_open_interest": expiry_rows,
            "not_ohlc_derived": True, "historical": True}
