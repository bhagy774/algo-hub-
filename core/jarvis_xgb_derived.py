"""Real candle-derived values for XGBoost feature slots that Parts do not emit.
Shadow/advisory only. Anything not computable from the supplied frames is omitted (never fabricated)."""
import math
import numpy as np


def _tr(df):
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    pc = c.shift(1)
    return (h - l).combine((h - pc).abs(), max).combine((l - pc).abs(), max)


def _adx(df, n=14):
    h, l = df['high'].astype(float), df['low'].astype(float)
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = _tr(df).ewm(alpha=1.0 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return float(dx.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1])


def _ok(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _compute_base(df15, df1h, part_signals=None):
    out = {}

    def put(part, key, val):
        v = _ok(val)
        if v is not None:
            out.setdefault(part, {})[key] = v

    try:
        d = df15
        if d is not None and len(d) >= 60:
            c = d['close'].astype(float)
            h, l, o, v = d['high'].astype(float), d['low'].astype(float), d['open'].astype(float), d['volume'].astype(float)
            price = float(c.iloc[-1])
            atr14 = float(_tr(d).tail(14).mean())
            atr20 = float(_tr(d).tail(20).mean())
            put('part7_volatility', 'atr_pct', atr14 / price * 100.0)
            put('part1_breakout', 'squeeze_ratio', (4.0 * float(c.tail(20).std())) / (3.0 * atr20) if atr20 else None)
            rng = float(h.iloc[-1] - l.iloc[-1])
            if rng > 0:
                up_w = float(h.iloc[-1] - max(o.iloc[-1], c.iloc[-1]))
                lo_w = float(min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1])
                put('part1_breakout', 'wick_ratio', (up_w + lo_w) / rng)
                put('part10_candlestats', 'wick_asymmetry', (up_w - lo_w) / rng)
                put('part10_candlestats', 'body_ratio', abs(float(c.iloc[-1] - o.iloc[-1])) / rng)
            run, sgn = 0, 0
            for i in range(len(d) - 1, max(len(d) - 12, 0), -1):
                s = 1 if c.iloc[i] > o.iloc[i] else -1 if c.iloc[i] < o.iloc[i] else 0
                if s == 0:
                    break
                if sgn == 0:
                    sgn = s
                if s != sgn:
                    break
                run += 1
            put('part10_candlestats', 'consecutive_bars', sgn * run)
            put('part4_volume', 'relative_volume', float(v.iloc[-1]) / float(v.iloc[-21:-1].mean()) if float(v.iloc[-21:-1].mean()) > 0 else None)
            n14 = 14
            hh, ll = float(h.tail(n14).max()), float(l.tail(n14).min())
            if hh > ll:
                put('part7_volatility', 'choppiness_index', 100.0 * math.log10(float(_tr(d).tail(n14).sum()) / (hh - ll)) / math.log10(n14))
            hl = np.log((h.tail(20) / l.tail(20)).astype(float).values) ** 2
            park = math.sqrt(float(hl.mean()) / (4.0 * math.log(2)))
            sd = float(np.log(c / c.shift(1)).tail(20).std())
            put('part7_volatility', 'parkinson_vol', park / sd if sd > 0 else None)
            y = np.log(c.tail(30).values.astype(float))
            x = np.arange(len(y), dtype=float)
            xm = x - x.mean()
            slope = float((xm * (y - y.mean())).sum() / (xm ** 2).sum())
            resid = y - (y.mean() + slope * xm)
            se = math.sqrt(float((resid ** 2).sum()) / (len(y) - 2) / float((xm ** 2).sum()))
            put('part5_ml', 'trend_tstat', slope / se if se > 0 else None)
            put('part6_trend', 'adx_15m', _adx(d))
            h60, l60 = float(h.tail(60).max()), float(l.tail(60).min())
            dz_hi, sz_lo = l60 + 0.5 * atr14, h60 - 0.5 * atr14
            put('part2_zone', 'demand_distance_pct', (price - dz_hi) / price * 100.0)
            put('part2_zone', 'supply_distance_pct', (sz_lo - price) / price * 100.0)
            dem_touch = int((l.tail(60) <= dz_hi).sum())
            sup_touch = int((h.tail(60) >= sz_lo).sum())
            put('part2_zone', 'retest_count', dem_touch if abs(price - dz_hi) <= abs(sz_lo - price) else sup_touch)
            best = None
            hv, lv = h.tail(52).values, l.tail(52).values
            for i in range(2, len(hv)):
                if lv[i] > hv[i - 2] and not (lv[i + 1:] <= hv[i - 2]).any():
                    mid = (lv[i] + hv[i - 2]) / 2.0
                elif hv[i] < lv[i - 2] and not (hv[i + 1:] >= lv[i - 2]).any():
                    mid = (hv[i] + lv[i - 2]) / 2.0
                else:
                    continue
                dist = abs(price - mid) / price * 100.0
                if best is None or dist < best:
                    best = dist
            put('part8_structure', 'fvg_distance_pct', best)
    except Exception:
        pass
    try:
        if df1h is not None and len(df1h) >= 60:
            put('part6_trend', 'adx_1h', _adx(df1h))
    except Exception:
        pass
    try:
        sigs = [float(s) for s in (part_signals or []) if _ok(s) is not None]
        if len(sigs) >= 4:
            put('part11_fusion', 'quorum_ratio', sum(1 for s in sigs if s != 0) / len(sigs))
            put('part11_fusion', 'dispersion_variance', float(np.var(sigs)))
    except Exception:
        pass
    return out


def _extra(df15):
    import numpy as np
    out = {}
    if df15 is None or len(df15) < 40:
        return out
    c = df15['close'].astype(float).values; o = df15['open'].astype(float).values
    h = df15['high'].astype(float).values; l = df15['low'].astype(float).values; v = df15['volume'].astype(float).values
    tail = c[-120:]; q = max(float(np.var(np.diff(tail))), 1e-12) * 0.05; r = max(float(np.var(np.diff(tail))), 1e-12)
    x = tail[0]; p = r
    for z in tail[1:]:
        p += q; k = p / (p + r); x = x + k * (z - x); p = (1 - k) * p
    out['part5_ml'] = {'kalman_error': float((c[-1] - x) / c[-1] * 100.0)}
    body = np.abs(c - o); mb = float(np.mean(body[-21:-1])) or 1e-12
    ob = 0.0
    for i in range(len(c) - 1, len(c) - 11, -1):
        if body[i] >= 1.5 * mb:
            ob = 1.0 if c[i] > o[i] else -1.0; break
    out['part8_structure'] = {'order_block_status': ob}
    rng = max(h[-1] - l[-1], 1e-12); vm = float(np.mean(v[-21:-1])) or 1e-12
    out['part9_orderflow'] = {'iceberg_flag': 1.0 if (v[-1] > 2.0 * vm and body[-1] / rng < 0.35) else 0.0}
    return out


def compute(df15, df1h, part_signals=None, inst_ctx=None):
    res = _compute_base(df15, df1h, part_signals)
    try:
        if isinstance(res, dict):
            for part, d in _extra(df15).items():
                cur = res.setdefault(part, {})
                if isinstance(cur, dict):
                    for k, val in d.items():
                        cur.setdefault(k, val)
            if isinstance(inst_ctx, dict):
                p3 = res.setdefault('part3_psychology', {})
                p4 = res.setdefault('part4_volume', {})
                fz = _ok(inst_ctx.get('funding_z'))
                if fz is not None:
                    p3.setdefault('funding_rate_zscore', fz)
                tla = _ok(inst_ctx.get('top_long_account'))
                if tla is not None:
                    p3.setdefault('retail_long_skew', (tla - 0.5) * 2.0)
                oi1h = _ok(inst_ctx.get('oi_delta_1h_pct'))
                if oi1h is not None:
                    p3.setdefault('liquidation_gravity', oi1h)
                bki = _ok(inst_ctx.get('book_imbalance'))
                if bki is not None:
                    p4.setdefault('bid_ask_imbalance', bki)
    except Exception:
        pass
    return res
