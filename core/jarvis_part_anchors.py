"""Real, candle-derived price anchors + continuous scores for Parts 1/2/4/8.
Every value is computed from the supplied OHLCV frame; unavailable -> None. Never fabricated.
Advisory sensor data only: jarvis_FIXED.py keeps final authority."""
import math


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _atr(df, n=14):
    try:
        h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
        pc = c.shift(1)
        tr = (h - l).combine((h - pc).abs(), max).combine((l - pc).abs(), max)
        return _f(tr.tail(n).mean())
    except Exception:
        return None


def _edge_score(pos):
    """pos in [-1,1] (range position). Dead zone until |pos|>0.6, then scaled to 1.0 at the edge."""
    a = abs(pos)
    if a <= 0.6:
        return 0.0
    return math.copysign(min(1.0, (a - 0.6) / 0.4), pos)


def _profile(df, bins=40, n=96):
    d = df.tail(n)
    lo, hi = _f(d['low'].min()), _f(d['high'].max())
    if lo is None or hi is None or hi <= lo:
        return None
    step = (hi - lo) / bins
    vol = [0.0] * bins
    for _, r in d.iterrows():
        tp = (float(r['high']) + float(r['low']) + float(r['close'])) / 3.0
        i = min(bins - 1, max(0, int((tp - lo) / step)))
        vol[i] += float(r['volume'])
    tot = sum(vol)
    if tot <= 0:
        return None
    pi = max(range(bins), key=lambda i: vol[i])
    lo_i = hi_i = pi
    acc = vol[pi]
    while acc < 0.7 * tot and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        dn = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return {'poc_price': lo + (pi + 0.5) * step, 'val_price': lo + lo_i * step, 'vah_price': lo + (hi_i + 1) * step}


def _swings(df, n=72, k=2):
    d = df.tail(n)
    h, l = d['high'].astype(float).tolist(), d['low'].astype(float).tolist()
    sh = sl = None
    for i in range(k, len(d) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            sh = h[i]
        if l[i] == min(l[i - k:i + k + 1]):
            sl = l[i]
    return sh, sl


def _enrich_raw(name, item, df):
    """Mutates item: adds 'anchors' (price levels) and, for part1/part2, 'continuous_score' when missing."""
    if not isinstance(item, dict) or df is None or len(df) < 40:
        return item
    try:
        anchors = {}
        close = _f(df['close'].iloc[-1])
        atr = _atr(df)
        if name == 'part1_breakout':
            h20, l20 = _f(df['high'].tail(20).max()), _f(df['low'].tail(20).min())
            anchors.update({'resistance_level': h20, 'support_level': l20, 'source': 'donchian20'})
            if close and h20 and l20 and h20 > l20 and item.get('continuous_score') is None:
                pos = (close - (h20 + l20) / 2.0) / ((h20 - l20) / 2.0)
                item['continuous_score'] = round(_edge_score(max(-1.0, min(1.0, pos))) + 0.0, 3)
        elif name == 'part2_zone':
            h60, l60 = _f(df['high'].tail(60).max()), _f(df['low'].tail(60).min())
            if h60 and l60 and atr:
                anchors.update({'demand_zone_low': l60, 'demand_zone_high': l60 + 0.5 * atr,
                                'supply_zone_low': h60 - 0.5 * atr, 'supply_zone_high': h60, 'source': 'range_extreme_60'})
                if close and h60 > l60 and item.get('continuous_score') is None:
                    pos = (close - (h60 + l60) / 2.0) / ((h60 - l60) / 2.0)
                    item['continuous_score'] = round(-_edge_score(max(-1.0, min(1.0, pos))) + 0.0, 3)  # mean-reversion at zones
        elif name == 'part4_volume':
            p = _profile(df)
            if p:
                anchors.update(p)
                anchors['source'] = 'volprofile96'
        elif name == 'part8_structure':
            sh, sl = _swings(df)
            anchors.update({'swing_high': sh, 'swing_low': sl, 'source': 'pivots72'})
        if anchors:
            item['anchors'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in anchors.items() if v is not None}
    except Exception:
        pass
    return item


def collect(parts_by_tf, price, tfs=('15m', '1h')):
    """Return (supports, resistances, sources) from all part anchors for the given frames."""
    sup, res, src = [], [], []
    keys = ('resistance_level', 'support_level', 'demand_zone_low', 'demand_zone_high', 'supply_zone_low',
            'supply_zone_high', 'poc_price', 'val_price', 'vah_price', 'swing_high', 'swing_low')
    for tf in tfs:
        for nm, it in (parts_by_tf.get(tf) or {}).items():
            if not nm.startswith('part') or '_' not in nm or not isinstance(it, dict):
                continue
            a = it.get('anchors') or {}
            for k in keys:
                v = _f(a.get(k))
                if v is None or not (0.5 * price < v < 1.5 * price):
                    continue
                (sup if v < price else res).append(v)
                src.append('%s:%s@%s' % (nm.split('_')[0], k, tf))
    return sorted(set(sup)), sorted(set(res)), src


def _sig_num(v):
    if isinstance(v, str):
        u = v.strip().upper()
        return 1 if u in ('BUY', 'CALL', 'LONG', 'BULLISH') else (-1 if u in ('SELL', 'PUT', 'SHORT', 'BEARISH') else 0)
    try:
        return 1 if float(v) > 0 else (-1 if float(v) < 0 else 0)
    except Exception:
        return 0


def enrich(name, item, df):
    """Wrapper: continuous_score must be sign-consistent with the Part's own discrete signal."""
    item = _enrich_raw(name, item, df)
    try:
        if name in ('part1_breakout', 'part2_zone') and isinstance(item, dict) and item.get('continuous_score') is not None:
            cs = float(item['continuous_score']); sg = _sig_num(item.get('signal'))
            if sg == 0:
                cs = max(-0.15, min(0.15, cs)) if abs(cs) <= 0.15 else 0.0
            else:
                cs = sg * max(0.05, abs(cs))
            item['continuous_score'] = round(max(-1.0, min(1.0, cs)), 3)
    except Exception:
        pass
    return item
