"""Bounded, auditable OHLCV statistics for Jarvis Parts 1–12.

All calculations consume closed OHLCV bars only. The module intentionally uses
Python's standard library, so analysis itself has no CUDA, NumPy, pandas, network,
or learned-weight dependency. OHLCV-derived flow is explicitly a proxy, never
exchange trade-tape/order-book data. Values called scores are evidence summaries,
not calibrated probabilities or performance claims.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MAX_BARS = 512
_EPS = 1e-12


def _finite(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def candles_from(data: Any, limit: int = MAX_BARS) -> List[Dict[str, float]]:
    """Extract and validate chronological OHLCV rows; reject malformed input."""
    if data is None or limit < 1:
        return []
    try:
        if hasattr(data, "to_dict"):
            rows = data.to_dict(orient="records")
        elif isinstance(data, Mapping) and isinstance(data.get("price_action"), Sequence):
            # Original Part 1 adapter's documented envelope.
            rows = list(data["price_action"])
        elif isinstance(data, Mapping):
            cols = {str(k): list(v) for k, v in data.items() if hasattr(v, "__iter__") and not isinstance(v, (str, bytes))}
            keys = set(cols)
            if not {"open", "high", "low", "close"}.issubset(keys):
                return []
            n = min(len(cols[k]) for k in ("open", "high", "low", "close"))
            if "volume" in cols:
                n = min(n, len(cols["volume"]))
            rows = [{k: cols[k][i] for k in cols} for i in range(n)]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            rows = list(data)
        else:
            return []
    except Exception:
        return []
    if not rows:
        return []
    out: List[Dict[str, float]] = []
    for row in rows[-limit:]:
        if not isinstance(row, Mapping):
            return []
        values = {k: _finite(row.get(k)) for k in ("open", "high", "low", "close")}
        if any(v is None for v in values.values()):
            return []
        o, h, l, c = (values[k] for k in ("open", "high", "low", "close"))
        v = _finite(row.get("volume", 0.0))
        if v is None or c <= 0 or min(o, h, l) <= 0 or v < 0 or h < max(o, c) or l > min(o, c) or h < l:
            return []
        out.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return out


def quantile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def median_abs_deviation(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    mid = statistics.median(vals)
    return statistics.median(abs(v - mid) for v in vals)


def robust_scale(values: Iterable[float], floor: float = _EPS) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return max(floor, abs(vals[0]) * 1e-6 if vals else floor)
    scale = 1.4826 * median_abs_deviation(vals)
    if scale <= floor:
        # Degenerate MAD is not treated as zero noise if a nonconstant series exists.
        scale = statistics.pstdev(vals)
    return max(float(floor), float(scale))


def _log_returns(rows: Sequence[Mapping[str, float]]) -> List[float]:
    return [math.log(float(b["close"]) / float(a["close"])) for a, b in zip(rows, rows[1:])]


def _theil_sen(log_prices: Sequence[float]) -> float:
    """Median pairwise log-price slope; bounded at 40 points for predictable cost."""
    ys = list(log_prices[-40:])
    slopes = [(ys[j] - ys[i]) / (j - i) for i in range(len(ys)) for j in range(i + 1, len(ys))]
    return statistics.median(slopes) if slopes else 0.0


def _local_linear_filter(log_prices: Sequence[float], return_scale: float) -> Dict[str, float]:
    """Two-state local-linear Kalman filter with robust, data-scaled noise."""
    ys = list(log_prices)
    if len(ys) < 3:
        return {"level": ys[-1] if ys else 0.0, "velocity": 0.0, "innovation_z": 0.0, "velocity_z": 0.0}
    vel0 = statistics.median([ys[i] - ys[i - 1] for i in range(1, min(len(ys), 8))])
    x0, x1 = ys[0], vel0
    # Symmetric 2x2 covariance represented by p00,p01,p11.
    p00, p01, p11 = max(return_scale ** 2, 1e-8), 0.0, max(return_scale ** 2, 1e-8)
    r = max(return_scale ** 2, 1e-10)
    q_level, q_velocity = max(r * 0.04, 1e-12), max(r * 0.002, 1e-13)
    innovation_z = 0.0
    for y in ys[1:]:
        # F=[[1,1],[0,1]]
        xp0, xp1 = x0 + x1, x1
        pp00 = p00 + 2.0 * p01 + p11 + q_level
        pp01 = p01 + p11
        pp11 = p11 + q_velocity
        innovation = y - xp0
        s = max(pp00 + r, 1e-12)
        k0, k1 = pp00 / s, pp01 / s
        x0, x1 = xp0 + k0 * innovation, xp1 + k1 * innovation
        p00 = max((1.0 - k0) * pp00, 1e-12)
        p01 = (1.0 - k0) * pp01
        p11 = max(pp11 - k1 * pp01, 1e-12)
        innovation_z = innovation / math.sqrt(s)
    velocity_sd = math.sqrt(max(p11, return_scale ** 2 / max(len(ys), 1), 1e-12))
    return {"level": x0, "velocity": x1, "innovation_z": innovation_z, "velocity_z": x1 / velocity_sd}


def _change_point_score(returns: Sequence[float]) -> Dict[str, float]:
    """Best one-break mean-shift evidence, penalized against the no-break model."""
    vals = list(returns[-64:])
    n = len(vals)
    if n < 12:
        return {"score": 0.0, "location": 0.0, "bic_gain": 0.0}
    base = statistics.fmean(vals)
    rss0 = sum((v - base) ** 2 for v in vals) + 1e-15
    best = (0.0, 0, rss0)
    for k in range(max(4, n // 5), min(n - 3, 4 * n // 5) + 1):
        a, b = vals[:k], vals[k:]
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        rss = sum((v - ma) ** 2 for v in a) + sum((v - mb) ** 2 for v in b) + 1e-15
        gain = n * math.log(rss0 / rss) - 2.0 * math.log(n)  # BIC penalty, not a p-value.
        if gain > best[0]:
            best = (gain, k, rss)
    z = (statistics.fmean(vals[best[1]:]) - statistics.fmean(vals[:best[1]])) / robust_scale(vals) if best[1] else 0.0
    return {"score": max(0.0, best[0]), "location": best[1] / n if best[1] else 0.0, "bic_gain": max(0.0, best[0]), "shift_z": z}


def _volatility_state(returns: Sequence[float], scale: float) -> Dict[str, float]:
    """Two-state Gaussian forward filter; posterior is conditional on stated priors."""
    vals = list(returns[-96:])
    if not vals:
        return {"high_state_posterior": 0.5, "short_long_ratio": 0.0}
    lo_sd, hi_sd = max(scale * 0.70, 1e-7), max(scale * 2.25, 1.5 * scale, 1e-7)
    p_low, p_high = 0.5, 0.5
    for r in vals:
        ll = []
        for sd in (lo_sd, hi_sd):
            ll.append(-math.log(sd) - 0.5 * (r / sd) ** 2)
        mx = max(ll)
        e0, e1 = math.exp(max(-700.0, ll[0] - mx)), math.exp(max(-700.0, ll[1] - mx))
        prior_low = p_low * 0.975 + p_high * 0.08
        prior_high = p_low * 0.025 + p_high * 0.92
        denom = max(prior_low * e0 + prior_high * e1, 1e-300)
        p_low, p_high = prior_low * e0 / denom, prior_high * e1 / denom
    recent = vals[-8:]
    long = vals[-48:]
    short_rv = math.sqrt(statistics.fmean(v * v for v in recent)) if recent else 0.0
    long_rv = math.sqrt(statistics.fmean(v * v for v in long)) if long else 0.0
    return {"high_state_posterior": min(1.0, max(0.0, p_high)), "short_long_ratio": short_rv / max(long_rv, 1e-12)}


def close_return_trend(closes: Any) -> Dict[str, Any]:
    """Robust close-only log-slope evidence for legacy close-series call sites."""
    try:
        if hasattr(closes, "detach"):
            closes = closes.detach().cpu()
        if hasattr(closes, "tolist"):
            closes = closes.tolist()
        values = [_finite(x) for x in closes]
    except Exception:
        values = []
    if not values or any(v is None or v <= 0 for v in values) or len(values) < 3:
        return {"available": False, "n": len(values), "slope": 0.0, "return_scale": 0.0, "trend_score": 0.0}
    logs = [math.log(v) for v in values[-128:]]
    returns = [b - a for a, b in zip(logs, logs[1:])]
    scale = robust_scale(returns)
    slope = _theil_sen(logs)
    score = slope / max(scale, 1e-12) * math.sqrt(min(len(returns), 64))
    return {"available": True, "n": len(logs), "slope": slope,
            "return_scale": scale, "trend_score": score}


def quantitative_features(data: Any, limit: int = MAX_BARS) -> Dict[str, Any]:
    """Return task-useful, finite, bounded multi-scale OHLCV statistics."""
    rows = candles_from(data, limit=limit)
    if len(rows) < 2:
        return {"available": False, "reason": "insufficient_or_invalid_ohlcv", "n": len(rows)}
    closes = [r["close"] for r in rows]
    opens = [r["open"] for r in rows]
    highs = [r["high"] for r in rows]
    lows = [r["low"] for r in rows]
    volumes = [r["volume"] for r in rows]
    rets = _log_returns(rows)
    rscale = robust_scale(rets)
    rv = math.sqrt(statistics.fmean(x * x for x in rets)) if rets else 0.0
    downside = math.sqrt(statistics.fmean(min(x, 0.0) ** 2 for x in rets)) if rets else 0.0
    absr = [abs(x) for x in rets]
    bipower = (math.pi / 2.0) * sum(abs(a) * abs(b) for a, b in zip(rets, rets[1:])) if len(rets) > 1 else 0.0
    rv2 = sum(x * x for x in rets)
    jump_share = min(1.0, max(0.0, (rv2 - bipower) / max(rv2, 1e-15)))
    price = closes[-1]
    intrabar_ranges = [math.log(max(r["high"], r["close"]) / min(r["low"], r["close"])) for r in rows]
    range_scale_pct = max(quantile(intrabar_ranges[-64:], 0.90), quantile(absr[-64:], 0.90), rscale)
    logprices = [math.log(c) for c in closes]
    state = _local_linear_filter(logprices[-96:], rscale)
    slope = _theil_sen(logprices[-64:])
    change = _change_point_score(rets)
    vol_state = _volatility_state(rets, max(rscale, 1e-8))
    # Geometry and signed-pressure estimates are OHLCV proxies, not measured order flow.
    geometry = []
    signed_flow = []
    for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
        span = max(h - l, price * 1e-12)
        body = c - o
        loc = min(1.0, max(-1.0, (2.0 * c - h - l) / span))
        geometry.append({"body_fraction": min(1.0, abs(body) / span), "close_location": loc, "upper_wick_fraction": max(0.0, (h - max(o, c)) / span), "lower_wick_fraction": max(0.0, (min(o, c) - l) / span)})
        signed_flow.append(loc * v)
    logvol = [math.log1p(v) for v in volumes]
    vol_base = logvol[-33:-1] or logvol[:-1]
    vol_z = (logvol[-1] - statistics.median(vol_base)) / robust_scale(vol_base) if vol_base else 0.0
    flow_den = sum(volumes[-20:])
    flow_imbalance = sum(signed_flow[-20:]) / max(flow_den, 1e-12)
    recent_signs = [1 if x > 0 else -1 if x < 0 else 0 for x in rets[-20:]]
    sign_persistence = sum(1 for a, b in zip(recent_signs, recent_signs[1:]) if a and a == b) / max(1, sum(1 for a, b in zip(recent_signs, recent_signs[1:]) if a and b))
    return {
        "available": True, "n": len(rows), "price": price,
        "returns": rets[-96:], "return_scale": rscale, "realized_vol": rv,
        "downside_vol": downside, "bipower_variation": bipower,
        "jump_share": jump_share, "range_scale_pct": range_scale_pct,
        "range_scale_price": price * range_scale_pct,
        "theil_sen_log_slope": slope, "kalman": state,
        "trend_score": slope / max(rscale, 1e-12) * math.sqrt(min(len(rets), 64)),
        "change_point": change, "volatility_state": vol_state,
        "volume_surprise_robust_z": max(-12.0, min(12.0, vol_z)),
        "flow_imbalance_proxy": min(1.0, max(-1.0, flow_imbalance)),
        "flow_label": "close-location-times-volume OHLCV proxy; not trade-tape delta",
        "sign_persistence": sign_persistence,
        "candle": geometry[-1], "candles": rows,
        "drawdown_from_peak": price / max(closes) - 1.0,
        "tail_return_q05": quantile(rets[-64:], 0.05),
        "tail_return_q95": quantile(rets[-64:], 0.95),
    }


def price_volume_distribution(data: Any, bins: int = 24, limit: int = 128) -> Dict[str, Any]:
    """Range-overlap volume histogram and shortest contiguous 70% mass interval.

    Bar volume is allocated uniformly across each bar's reported high-low range;
    this is only a distribution proxy, not traded-at-price data.
    """
    rows = candles_from(data, limit)
    if not rows or bins < 4:
        return {"available": False, "reason": "insufficient_or_invalid_ohlcv"}
    lo = min(r["low"] for r in rows)
    hi = max(r["high"] for r in rows)
    if hi <= lo:
        return {"available": False, "reason": "flat_price_range"}
    width = (hi - lo) / bins
    mass = [0.0] * bins
    for row in rows:
        a, b, v = row["low"], row["high"], row["volume"]
        if b <= a:
            ix = min(bins - 1, max(0, int((row["close"] - lo) / width)))
            mass[ix] += v
        else:
            covered = []
            for i in range(bins):
                left, right = lo + i * width, lo + (i + 1) * width
                overlap = max(0.0, min(b, right) - max(a, left))
                if overlap:
                    covered.append((i, overlap))
            denom = sum(x for _, x in covered)
            for i, overlap in covered:
                mass[i] += v * overlap / max(denom, 1e-12)
    total = sum(mass)
    if total <= 0:
        return {"available": False, "reason": "zero_volume"}
    poc = max(range(bins), key=lambda i: (mass[i], -i))
    best = (float("inf"), 0, bins - 1)
    for left in range(bins):
        accumulated = 0.0
        for right in range(left, bins):
            accumulated += mass[right]
            if accumulated >= 0.70 * total:
                span = right - left
                if span < best[0]:
                    best = (span, left, right)
                break
    _, left, right = best
    close = rows[-1]["close"]
    flow = []
    for r in rows[-20:]:
        rng = max(r["high"] - r["low"], 1e-12)
        flow.append((2.0 * r["close"] - r["high"] - r["low"]) / rng * r["volume"])
    return {"available": True, "poc": lo + (poc + 0.5) * width,
            "value_low": lo + left * width, "value_high": lo + (right + 1) * width,
            "close_location_in_value": (close - (lo + left * width)) / max((right + 1 - left) * width, 1e-12),
            "volume_surprise_robust_z": quantitative_features(rows).get("volume_surprise_robust_z", 0.0),
            "flow_imbalance_proxy": sum(flow) / max(sum(r["volume"] for r in rows[-20:]), 1e-12),
            "profile_label": "range-overlap volume proxy; not traded-at-price data",
            "mass": mass, "bin_width": width}


def structural_zones(data: Any, limit: int = 128) -> Dict[str, Any]:
    """Recency-weighted clustering of confirmed local extrema with adaptive width."""
    f = quantitative_features(data, limit=limit)
    if not f.get("available") or f["n"] < 12:
        return {"available": False, "reason": "insufficient_or_invalid_ohlcv"}
    rows = f["candles"]
    eps = max(f["range_scale_price"] * 0.15, f["price"] * 1e-6)
    pivots: List[Tuple[float, float, str]] = []
    start = max(2, len(rows) - 96)
    for i in range(start, len(rows) - 2):
        weight = 0.5 + (i - start) / max(len(rows) - start, 1)
        if rows[i]["high"] >= max(rows[j]["high"] for j in range(i - 2, i + 3) if j != i):
            pivots.append((rows[i]["high"], weight, "resistance"))
        if rows[i]["low"] <= min(rows[j]["low"] for j in range(i - 2, i + 3) if j != i):
            pivots.append((rows[i]["low"], weight, "support"))
    clusters: List[Dict[str, Any]] = []
    for price, weight, side in sorted(pivots, key=lambda x: x[0]):
        candidates = [c for c in clusters if c["side"] == side and abs(c["center"] - price) <= eps]
        if candidates:
            c = min(candidates, key=lambda x: abs(x["center"] - price))
            c["center"] = (c["center"] * c["weight"] + price * weight) / (c["weight"] + weight)
            c["weight"] += weight
            c["touches"] += 1
        else:
            clusters.append({"center": price, "weight": weight, "touches": 1, "side": side})
    current = f["price"]
    for c in clusters:
        c["distance"] = c["center"] - current
        c["strength"] = min(1.0, c["weight"] / 4.0) * min(1.0, c["touches"] / 3.0)
    supports = sorted((c for c in clusters if c["center"] <= current), key=lambda c: current - c["center"])
    resistances = sorted((c for c in clusters if c["center"] >= current), key=lambda c: c["center"] - current)
    return {"available": True, "bandwidth": eps,
            "nearest_support": supports[0] if supports else None,
            "nearest_resistance": resistances[0] if resistances else None,
            "clusters": clusters[:64], "label": "confirmed local-extrema density; zones are estimates"}


def sequential_direction_evidence(data: Any, alpha: float = 1.0) -> Dict[str, Any]:
    """Laplace-smoothed next-sign evidence from rolling direction transitions."""
    f = quantitative_features(data)
    if not f.get("available") or len(f["returns"]) < 8:
        return {"available": False, "reason": "insufficient_or_invalid_ohlcv"}
    signs = [1 if r > 0 else 0 for r in f["returns"] if abs(r) > 1e-15]
    transitions = [[alpha, alpha], [alpha, alpha]]
    for a, b in zip(signs, signs[1:]):
        transitions[a][b] += 1.0
    state = signs[-1]
    p_up = transitions[state][1] / sum(transitions[state])
    return {"available": True, "next_up_frequency": p_up,
            "transition_counts": transitions, "n": len(signs),
            "label": "smoothed historical frequency, not a calibrated forecast"}


def _hurst_proxy(returns: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Variance-ratio based Hurst estimate H=0.5*ln(Var4/Var1)/ln(4); None if too short."""
    r = [x for x in returns if isinstance(x, (int, float)) and math.isfinite(x)]
    if len(r) < 32:
        return None, None
    n = len(r)
    mu = sum(r) / n
    q = 4
    v1 = sum((x - mu) ** 2 for x in r) / (n - 1)
    if v1 <= 1e-18:
        return None, None
    m = q * (n - q + 1) * (1.0 - q / n)  # Lo-MacKinlay unbiased overlapping estimator
    vq = sum((sum(r[i:i + q]) - q * mu) ** 2 for i in range(n - q + 1)) / m
    vr = vq / v1
    h = 0.5 + 0.5 * math.log(max(vr, 1e-9)) / math.log(float(q))
    return max(0.2, min(0.8, h)), vr


def _tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


def continuous_part_score(part: str, f: Mapping[str, Any], tele: Mapping[str, Any]) -> Optional[float]:
    """Smooth directional conviction in [-1,+1] for Parts 1-6 and 8-10 (None otherwise).

    Calibration constants are initial design values, NOT validated edge.
    """
    def g(k, d=0.0):
        v = tele.get(k, d)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else d
    trend = g("trend_score")
    velz = g("kalman_velocity_z")
    imb = g("flow_imbalance_proxy")
    h = tele.get("hurst_proxy")
    cl = g("close_location")
    if part == "1":
        return _tanh(0.55 * g("breakout_intensity") + 0.45 * trend)
    if part == "2":
        dd, sd = tele.get("demand_distance_pct"), tele.get("supply_distance_pct")
        if isinstance(dd, (int, float)) and isinstance(sd, (int, float)) and (abs(dd) + abs(sd)) > 1e-9:
            bias = (abs(sd) - abs(dd)) / (abs(sd) + abs(dd))
            return _tanh(1.2 * bias + 0.8 * cl)
        return _tanh(0.8 * cl)
    if part == "3":
        shape = cl * (0.5 + g("body_fraction"))
        return _tanh(0.30 * g("pressure_innovation_z") + 0.8 * shape)
    if part == "4":
        sgn = 1.0 if imb >= 0 else -1.0
        return _tanh(0.60 * (imb / 0.15) + 0.40 * (g("volume_surprise_robust_z") / 2.0) * sgn)
    if part == "5":
        return _tanh(0.5 * velz + 0.35 * trend)
    if part == "6":
        return _tanh(0.60 * trend + 0.40 * velz)
    if part == "8":
        return _tanh(0.5 * (g("high_break_scale") - g("low_break_scale")) + 0.4 * velz)
    if part == "9":
        sgn = 1.0 if imb >= 0 else -1.0
        return _tanh(0.65 * (imb / 0.12) + 0.35 * ((g("sign_persistence", 0.5) - 0.50) / 0.08) * sgn)
    if part == "10":
        seq = tele.get("sequential_direction") or {}
        nu = seq.get("next_up_frequency") if isinstance(seq, Mapping) else None
        if not isinstance(nu, (int, float)):
            return _tanh(0.6 * cl * g("body_fraction"))
        return _tanh(((nu - 0.50) / 0.08) * 0.70 + (cl * 2.0) * g("body_fraction") * 0.30)
    return None


TIMEFRAME_HORIZON_MAP: Dict[str, str] = {
    "1m": "fast", "3m": "fast", "5m": "short", "15m": "short",
    "30m": "session", "1h": "session", "2h": "session",
    "4h": "swing", "6h": "swing", "8h": "swing", "12h": "swing",
    "1d": "macro", "3d": "macro", "1w": "macro", "1M": "macro",
}

# What each Part is PRIMARILY responsible for on each horizon (documentation + telemetry).
HORIZON_PART_ROLES: Dict[str, Dict[str, str]] = {
    "fast": {"1": "micro_breakout_trigger", "3": "candle_effort_trigger", "9": "flow_trigger", "10": "streak_trigger",
             "2": "confirm_only", "4": "confirm_only", "5": "strict_filter", "6": "strict_filter", "8": "strict_filter"},
    "short": {"1": "range_breakout", "2": "intraday_zone", "4": "value_area_escape", "5": "momentum", "6": "trend",
              "8": "bos_choch", "9": "cvd", "3": "confirm_only", "10": "confirm_only"},
    "session": {"2": "session_zone", "4": "poc_value_area", "5": "regime", "6": "session_trend", "8": "session_structure",
                "9": "accumulation", "1": "confirm_only", "3": "trend_aligned_only", "10": "trend_aligned_only"},
    "swing": {"2": "major_zone", "4": "value_area", "5": "state_velocity", "6": "swing_trend", "8": "swing_structure",
              "9": "confirm_only", "1": "confirm_only", "3": "trend_aligned_only", "10": "trend_aligned_only"},
    "macro": {"2": "major_zone", "4": "macro_value", "5": "macro_drift", "6": "macro_trend", "8": "macro_structure",
              "1": "confirm_only", "3": "trend_aligned_only", "9": "trend_aligned_only", "10": "trend_aligned_only"},
}


def resolve_timeframe(data: Any = None, timeframe: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    tf = timeframe
    if not tf and isinstance(context, Mapping):
        tf = context.get("timeframe") or context.get("native_timeframe")
    if not tf and data is not None and not isinstance(data, (list, tuple)):
        attrs = getattr(data, "attrs", None)
        if isinstance(attrs, Mapping):
            tf = attrs.get("timeframe")
    tf = str(tf).strip() if tf else None
    return tf if tf in TIMEFRAME_HORIZON_MAP else None


def part_signal(part: str, data: Any, *, timeframe: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Task-specific deterministic evidence; thresholds adapt to the timeframe horizon when known."""
    _tf = resolve_timeframe(data, timeframe, context)
    hz = TIMEFRAME_HORIZON_MAP.get(_tf, "default")
    role = (HORIZON_PART_ROLES.get(hz) or {}).get(str(part), "primary")
    fast, slow = hz == "fast", hz in ("session", "swing", "macro")
    # strict: stricter bar for noisy 1m/3m trend-type parts; relaxed: Part3/10 single-bar noise on slow frames
    k_strict = 1.25 if fast else 1.0
    k_oppose = 0.70 if slow else 1.0
    f = quantitative_features(data)
    if not f.get("available"):
        return {"signal": 0, "confidence": 5.0, "thought": f"Part{part}: invalid/insufficient OHLCV", "telemetry": {"status": f.get("reason", "unavailable")}}
    rows, price = f["candles"], f["price"]
    retz = f["returns"][-1] / max(f["return_scale"], 1e-12) if f["returns"] else 0.0
    trend = f["trend_score"]
    volz = f["volume_surprise_robust_z"]
    tele = {k: f[k] for k in ("realized_vol", "downside_vol", "jump_share", "trend_score", "volume_surprise_robust_z", "flow_imbalance_proxy", "sign_persistence", "drawdown_from_peak")}
    tele["flow_label"] = f["flow_label"]
    _c = f["candle"]
    tele.update({
        "short_long_ratio": f["volatility_state"].get("short_long_ratio"),
        "kalman_velocity_z": f["kalman"].get("velocity_z"),
        "range_scale_pct": f["range_scale_pct"],
        "atr_pct": f["range_scale_pct"] * 100.0,
        "wick_ratio": _c["upper_wick_fraction"] + _c["lower_wick_fraction"],
        "wick_asymmetry": _c["upper_wick_fraction"] - _c["lower_wick_fraction"],
        "timeframe": _tf, "timeframe_horizon": hz, "timeframe_role": role,
    })
    sig, conf, text = 0, 5.0, "neutral, no decisive evidence"

    if part == "1":
        prior = rows[-(16 if fast else 29 if slow else 21):-1]
        if len(prior) >= 10:
            old_high, old_low = max(r["high"] for r in prior), min(r["low"] for r in prior)
            scale = max(f["range_scale_price"], price * 1e-6)
            up, down = (price - old_high) / scale, (old_low - price) / scale
            volz = volz if not fast else volz - 0.35  # micro breakout needs real volume
            tele["breakout_intensity"] = max(up, 0.0) - max(down, 0.0)
            if up >= 0.0 and trend > 0.75 and volz > -0.5:
                sig, conf, text = 1, min(85.0, 55.0 + 5.0 * min(up, 3.0) + min(trend, 3.0) * 3.0), "multi-scale range escape with trend/change evidence"
            elif down >= 0.0 and trend < -0.75 and volz > -0.5:
                sig, conf, text = -1, min(85.0, 55.0 + 5.0 * min(down, 3.0) + min(abs(trend), 3.0) * 3.0), "multi-scale downside range escape with trend/change evidence"
        tele["breakout_basis"] = "prior-range escape / robust return scale"
    elif part == "2":
        zones = structural_zones(rows)
        tele["zones"] = zones
        if zones.get("available"):
            sup, res = zones.get("nearest_support"), zones.get("nearest_resistance")
            bandwidth = zones["bandwidth"]
            if sup:
                tele["demand_distance_pct"] = (price - sup["center"]) / price * 100.0
                tele["retest_count"] = sup["touches"]
            if res:
                tele["supply_distance_pct"] = (res["center"] - price) / price * 100.0
            if sup and sup["touches"] >= 2 and abs(price - sup["center"]) <= bandwidth and f["candle"]["close_location"] > 0.25:
                sig, conf, text = 1, min(75.0, 55.0 + 10.0 * sup["strength"]), "support-density reaction with positive auction close-location"
            elif res and res["touches"] >= 2 and abs(price - res["center"]) <= bandwidth and f["candle"]["close_location"] < -0.25:
                sig, conf, text = -1, min(75.0, 55.0 + 10.0 * res["strength"]), "resistance-density rejection with negative auction close-location"
    elif part == "3":
        c = f["candle"]
        shape_score = c["close_location"] * (0.5 + c["body_fraction"])
        history = [g["body_fraction"] for g in _geometry(rows[:-1])[-32:]]
        body_z = (c["body_fraction"] - statistics.median(history)) / robust_scale(history) if history else 0.0
        p_innov_z = max(-12.0, min(12.0, f["flow_imbalance_proxy"] / max(f["return_scale"] * math.sqrt(max(len(f["returns"][-20:]), 1)), 1e-6)))
        tele.update({"body_fraction": c["body_fraction"], "close_location": c["close_location"], "body_robust_z": body_z, "pressure_innovation_z": p_innov_z})
        if (body_z >= 0.75 and abs(c["close_location"]) >= 0.55 and abs(retz) >= 0.60) or (abs(p_innov_z) >= 1.1 and abs(c["close_location"]) >= 0.55):
            cand_sig = 1 if shape_score > 0 else -1
            tg3 = 0.85 if slow else 1.25
            counter_strong_trend = (cand_sig < 0 and trend >= tg3) or (cand_sig > 0 and trend <= -tg3)
            climax_reversal = abs(body_z) >= 2.2 and abs(p_innov_z) >= 1.8 and abs(c["close_location"]) >= 0.75
            sig = cand_sig if (not counter_strong_trend or climax_reversal) else 0
            conf = min(75.0, 55.0 + min(abs(body_z), 2.0) * 5.0 + abs(c["close_location"]) * 5.0)
            text = "robustly unusual candle effort/close-location evidence"
    elif part == "4":
        profile = price_volume_distribution(rows)
        tele["volume_profile"] = profile
        if profile.get("available"):
            imb = profile["flow_imbalance_proxy"]
            surprise = profile["volume_surprise_robust_z"]
            poc_price = profile.get("poc", profile.get("poc_price", price))
            tele.update({"poc_price": poc_price, "value_high": profile.get("value_high"), "value_low": profile.get("value_low")})
            if price > profile["value_high"] and imb > 0.10 and surprise >= 0.0:
                sig, conf, text = 1, min(75.0, 55.0 + min(surprise, 3.0) * 4.0 + imb * 10.0), "volume-distribution escape with OHLCV pressure proxy"
            elif price < profile["value_low"] and imb < -0.10 and surprise >= 0.0:
                sig, conf, text = -1, min(75.0, 55.0 + min(surprise, 3.0) * 4.0 + abs(imb) * 10.0), "volume-distribution breakdown with OHLCV pressure proxy"
            elif price > poc_price and imb >= 0.10 and surprise >= 0.25 and trend >= 1.0:
                sig, conf, text = 1, min(72.0, 55.0 + min(surprise, 3.0) * 3.5 + imb * 10.0), "POC volume trend acceptance above point of control"
            elif price < poc_price and imb <= -0.10 and surprise >= 0.25 and trend <= -1.0:
                sig, conf, text = -1, min(72.0, 55.0 + min(surprise, 3.0) * 3.5 + abs(imb) * 10.0), "POC volume trend breakdown below point of control"
    elif part == "5":
        cp = f["change_point"]
        velz = f["kalman"]["velocity_z"]
        # This is a statistical evidence score, not learned ML or a probability.
        tele.update({"kalman_velocity_z": velz, "break_shift_bic_gain": cp["bic_gain"], "break_shift_z": cp["shift_z"]})
        if (abs(velz) >= 1.5 * k_strict and abs(trend) >= 1.5 * k_strict) or (abs(trend) >= 1.65 * k_strict and (velz * trend > 0) and abs(velz) >= 0.45 and cp["bic_gain"] >= 1.0):
            sig = 1 if velz > 0 else -1
            conf, text = min(75.0, 55.0 + min(abs(velz), 3.0) * 4.0 + min(cp["bic_gain"], 4.0) * 2.0), "state-space velocity plus penalized return-mean-shift evidence"
    elif part == "6":
        velz = f["kalman"]["velocity_z"]
        signs = [1 if r > 0 else -1 if r < 0 else 0 for r in f["returns"][-20:]]
        coherence = abs(sum(signs)) / max(1, sum(bool(x) for x in signs))
        tele.update({"posterior_velocity_z": velz, "directional_coherence": coherence})
        if (abs(velz) >= 1.0 * k_strict and coherence >= 0.15 and abs(trend) >= 1.1 * k_strict) or (abs(trend) >= 1.45 * k_strict and coherence >= 0.25 and (velz * trend > 0) and abs(velz) >= 0.35):
            sig = 1 if velz > 0 else -1
            conf, text = min(75.0, 55.0 + min(abs(velz), 3.0) * 5.0 + coherence * 5.0), "adaptive local-linear trend posterior and directional coherence"
    elif part == "7":
        vs = f["volatility_state"]
        tele.update(vs)
        tail = max(abs(f["tail_return_q05"]), abs(f["tail_return_q95"]))
        if (f["range_scale_pct"] >= 0.05 or
                (f["range_scale_pct"] >= 0.015 and vs["short_long_ratio"] >= 2.0) or
                vs["high_state_posterior"] >= 0.90 and
                (vs["short_long_ratio"] >= 2.5 or tail >= 4.0 * f["return_scale"])): 
            sig, conf, text = 0, 5.0, "high-volatility-state risk veto (model-conditional posterior plus realized-tail evidence)"
            tele["entry_blocked"] = True
            tele["risk_veto"] = True
        elif vs["short_long_ratio"] >= 1.5 and abs(retz) >= 1.25:
            sig = 1 if retz > 0 else -1
            conf, text = min(70.0, 55.0 + min(vs["short_long_ratio"], 2.5) * 4.0), "realized-variance expansion with directional return innovation"
    elif part == "8":
        zones = structural_zones(rows)
        tele["zones"] = zones
        sup = zones.get("nearest_support") if isinstance(zones, dict) else None
        res = zones.get("nearest_resistance") if isinstance(zones, dict) else None
        sup_center = sup.get("center") if isinstance(sup, dict) else None
        res_center = res.get("center") if isinstance(res, dict) else None
        tele.update({"nearest_support": sup_center, "nearest_resistance": res_center, "kalman_velocity_z": f["kalman"]["velocity_z"]})
        if len(rows) >= 12:
            prior = rows[-33:-1]
            prior12 = rows[-13:-1]
            if prior:
                span = max(f["range_scale_price"], price * 1e-6)
                high_break = (price - max(r["high"] for r in prior)) / span
                low_break = (min(r["low"] for r in prior) - price) / span
                hb12 = (price - max(r["high"] for r in prior12)) / span if prior12 else high_break
                lb12 = (min(r["low"] for r in prior12) - price) / span if prior12 else low_break
                velz = f["kalman"]["velocity_z"]
                tele.update({"high_break_scale": high_break, "low_break_scale": low_break, "high_break_12": hb12, "low_break_12": lb12})
                if (high_break >= 0.0 or hb12 >= 0.0) and velz >= 0.75 * k_strict and volz >= -0.5:
                    sig, conf, text = 1, min(75.0, 55.0 + min(max(high_break, hb12, 0.2), 3.0) * 5.0), "confirmed structural BOS break with state-space trend"
                elif (low_break >= 0.0 or lb12 >= 0.0) and velz <= -0.75 * k_strict and volz >= -0.5:
                    sig, conf, text = -1, min(75.0, 55.0 + min(max(low_break, lb12, 0.2), 3.0) * 5.0), "confirmed structural downside BOS break with state-space trend"
                elif ((trend >= 1.2 * k_strict and velz >= 0.8) or (trend >= 1.35 * k_strict and velz >= 0.40)) and (sup_center is None or price >= sup_center):
                    sig, conf, text = 1, min(72.0, 55.0 + min(velz, 2.5) * 4.0), "structural HH/HL trend continuation above support zone"
                elif ((trend <= -1.2 * k_strict and velz <= -0.8) or (trend <= -1.35 * k_strict and velz <= -0.40)) and (res_center is None or price <= res_center):
                    sig, conf, text = -1, min(72.0, 55.0 + min(abs(velz), 2.5) * 4.0), "structural LH/LL trend continuation below resistance zone"
    elif part == "9":
        imb = f["flow_imbalance_proxy"]
        vol_surp = f.get("volume_surprise_robust_z", 0.0)
        tele["pressure_innovation_z"] = max(-12.0, min(12.0, imb / max(f["return_scale"] * math.sqrt(max(len(f["returns"][-20:]), 1)), 1e-6)))
        tele["flow_is_measured_orderbook"] = False
        price_change = math.log(price / rows[max(0, len(rows) - (10 if fast else 20))]["close"])
        if (imb >= 0.10 and f["sign_persistence"] >= 0.52 and trend >= 1.0) or (imb >= 0.15 and vol_surp >= 0.5 and trend >= 0.5):
            sig, conf, text = 1, min(75.0, 55.0 + min(imb, 0.8) * 15.0), "persistent positive close-location volume proxy aligned with robust trend"
        elif (imb <= -0.10 and f["sign_persistence"] >= 0.52 and trend <= -1.0) or (imb <= -0.15 and vol_surp >= 0.5 and trend <= -0.5):
            sig, conf, text = -1, min(75.0, 55.0 + min(abs(imb), 0.8) * 15.0), "persistent negative close-location volume proxy aligned with robust trend"
        elif price_change < -2.0 * f["return_scale"] and imb > 0.15 and trend > -1.35 and f["kalman"]["velocity_z"] >= -0.55:
            sig, conf, text = 1, 60.0, "price/volume-proxy divergence evidence; not exchange order flow"
        elif price_change > 2.0 * f["return_scale"] and imb < -0.15 and trend < 1.35 and f["kalman"]["velocity_z"] <= 0.55:
            sig, conf, text = -1, 60.0, "price/volume-proxy divergence evidence; not exchange order flow"
    elif part == "10":
        seq = sequential_direction_evidence(rows)
        tele["sequential_direction"] = seq
        g = f["candle"]
        tele.update({"body_fraction": g["body_fraction"], "close_location": g["close_location"]})
        seq_dir = (1 if seq.get("next_up_frequency", 0.5) > 0.5 else -1) if seq.get("available") else 0
        seq_opposes_trend = (seq_dir < 0 and trend >= 1.2 * k_oppose) or (seq_dir > 0 and trend <= -1.2 * k_oppose)
        if seq.get("available") and not seq_opposes_trend and abs(seq["next_up_frequency"] - 0.5) >= 0.045 and g["body_fraction"] >= 0.50 and abs(g["close_location"]) >= 0.45 and ((seq["next_up_frequency"] > 0.5 and g["close_location"] > 0) or (seq["next_up_frequency"] < 0.5 and g["close_location"] < 0)):
            sig = 1 if seq["next_up_frequency"] > 0.5 else -1
            conf, text = 55.0 + min(20.0, abs(seq["next_up_frequency"] - 0.5) * 40.0), "smoothed sequential return-sign frequency with current candle geometry"
        elif g["body_fraction"] >= 0.65 and abs(g["close_location"]) >= 0.60 and abs(trend) >= 1.0 and (g["close_location"] * trend > 0):
            sig = 1 if g["close_location"] > 0 else -1
            conf, text = min(72.0, 56.0 + g["body_fraction"] * 12.0), "high-conviction candle geometry expansion aligned with trend"
    elif part in ("11", "12"):
        text = "deterministic multi-frame evidence fusion remains the authority"
    _h, _vr = _hurst_proxy(f["returns"])
    tele["hurst_proxy"] = _h
    tele["variance_ratio"] = _vr
    _cs = continuous_part_score(part, f, tele)
    tele["continuous_score"] = None if _cs is None else round(_cs, 4)
    return {"signal": sig, "confidence": round(conf, 1), "thought": f"Part{part}: {text}", "telemetry": tele, "continuous_score": tele["continuous_score"]}


def _geometry(rows: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    out = []
    for r in rows:
        span = max(r["high"] - r["low"], r["close"] * 1e-12)
        out.append({"body_fraction": min(1.0, abs(r["close"] - r["open"]) / span),
                    "close_location": min(1.0, max(-1.0, (2 * r["close"] - r["high"] - r["low"]) / span)),
                    "upper_wick_fraction": max(0.0, (r["high"] - max(r["open"], r["close"])) / span),
                    "lower_wick_fraction": max(0.0, (min(r["open"], r["close"]) - r["low"]) / span)})
    return out


def live_feature_snapshot(data: Any, limit: int = 128) -> Dict[str, Any]:
    """Bounded Python-native diagnostic snapshot for Part 7's live data adapter."""
    f = quantitative_features(data, limit=min(128, max(1, int(limit))))
    if not f.get("available"):
        return {"available": False, "data_status": f.get("reason", "invalid_ohlcv"), "compute_device": "cpu"}
    profile = price_volume_distribution(f["candles"], limit=128)
    zones = structural_zones(f["candles"], limit=128)
    sequential = sequential_direction_evidence(f["candles"])
    return {
        "available": True,
        "n_closed_bars": f["n"],
        "data_status": "valid_closed_ohlcv",
        "compute_device": "cpu",
        "signal_role": "advisory_only_no_execution_authority",
        "log_returns": list(f["returns"]),
        "realized_volatility": f["realized_vol"],
        "downside_volatility": f["downside_vol"],
        "bipower_variation": f["bipower_variation"],
        "jump_variation_share": f["jump_share"],
        "robust_return_scale": f["return_scale"],
        "range_scale_price": f["range_scale_price"],
        "robust_trend_score": f["trend_score"],
        "state_velocity_z": f["kalman"]["velocity_z"],
        "state_innovation_z": f["kalman"]["innovation_z"],
        "change_point_bic_gain": f["change_point"]["bic_gain"],
        "high_variance_state_posterior": f["volatility_state"]["high_state_posterior"],
        "short_long_variance_ratio": f["volatility_state"]["short_long_ratio"],
        "volume_surprise_robust_z": f["volume_surprise_robust_z"],
        "ohlcv_flow_imbalance_proxy": f["flow_imbalance_proxy"],
        "flow_proxy_label": f["flow_label"],
        "candle_geometry": dict(f["candle"]),
        "price_volume_distribution": profile,
        "structural_zones": zones,
        "sequential_direction_evidence": sequential,
    }


def live_directional_advisory(features: Any) -> str:
    """Conservative uncalibrated direction hint; risk vetoes return HOLD."""
    if not isinstance(features, Mapping) or not features.get("available"):
        return "HOLD"
    velocity = _finite(features.get("state_velocity_z"))
    trend = _finite(features.get("robust_trend_score"))
    break_gain = _finite(features.get("change_point_bic_gain"))
    high_state = _finite(features.get("high_variance_state_posterior"))
    variance_ratio = _finite(features.get("short_long_variance_ratio"))
    if None in (velocity, trend, break_gain, high_state, variance_ratio):
        return "HOLD"
    if high_state >= 0.90 or variance_ratio >= 2.5:
        return "HOLD"
    if velocity >= 1.5 and trend >= 1.0 and break_gain >= 2.0:
        return "BUY"
    if velocity <= -1.5 and trend <= -1.0 and break_gain >= 2.0:
        return "SELL"
    return "HOLD"


def risk_scale_price(data: Any, price: Optional[float] = None, floor_pct: float = 0.001) -> float:
    """Conservative price-unit risk scale from empirical range/return quantiles."""
    f = quantitative_features(data)
    p = _finite(price) if price is not None else (f.get("price") if f.get("available") else None)
    if not p or p <= 0:
        return 0.0
    floor = p * max(0.0, float(floor_pct))
    if not f.get("available"):
        return floor
    return max(floor, f["range_scale_price"], quantile([r["high"] - r["low"] for r in f["candles"][-64:]], 0.85))
