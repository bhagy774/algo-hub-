# Central Safe-Entry engine (sensor used by jarvis_FIXED.py). Levels come from real candles/option walls; nothing fabricated.
# SCALP and SWING use their OWN reference frames (SCALP: 5m fair value/ATR + 5m/15m structure;
# SWING: 1h fair value/ATR + 1h/4h structure). All ATR-based limits are expressed in the ATR of the mode frame.
import math

COST_PCT = 0.14
MIN_RR = {'SCALP': 1.5, 'SWING': 1.8}
MAX_EXT_ATR = {'SCALP': 0.85, 'SWING': 1.0}     # SCALP measured vs FAST anchor (5m EMA9/EMA20/2h VWAP), SWING vs 1h fair value
SCALP_EXHAUST_ATR = 1.8                          # beyond this from nearest fast anchor = exhaustion -> BLOCKED
SCALP_SL_MIN_PCT, SCALP_SL_MAX_PCT = 0.25, 0.75  # scalp stop distance bounds (% of entry); also >= 0.8 ATR(5m)
SCALP_FAST_VWAP_BARS = 24                        # 5m x24 = 2h rolling VWAP
SL_BUF_ATR = {'SCALP': 0.25, 'SWING': 0.35}
WALL_MIN_ATR = {'SCALP': 0.85, 'SWING': 0.5}
RISK_MIN_ATR = {'SCALP': 0.7, 'SWING': 0.4}
RISK_MAX_ATR = {'SCALP': 5.0, 'SWING': 3.0}
OFFSET_ATR = {'SCALP': 0.17, 'SWING': 0.1}
VWAP_BARS = {'SCALP': 36, 'SWING': 24}          # 5m x36 = 3h ; 1h x24 = 24h
BAND_ATR = 0.25
PRIMARY_TF = {'SCALP': '5m', 'SWING': '1h'}
SECONDARY_TF = {'SCALP': '15m', 'SWING': '4h'}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def _atr(df, n=14):
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    pc = c.shift(1)
    tr = (h - l).combine((h - pc).abs(), max).combine((l - pc).abs(), max)
    return _num(tr.rolling(n).mean().iloc[-1])


def _vwap(df, n=96):
    d = df.tail(n)
    tp = (d['high'].astype(float) + d['low'].astype(float) + d['close'].astype(float)) / 3.0
    vol = d['volume'].astype(float)
    s = vol.sum()
    return _num((tp * vol).sum() / s) if s > 0 else None


def _pivots(df, n=72, k=2):
    d = df.tail(n)
    h, l = d['high'].astype(float).tolist(), d['low'].astype(float).tolist()
    highs, lows = [], []
    for i in range(k, len(d) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            highs.append(h[i])
        if l[i] == min(l[i - k:i + k + 1]):
            lows.append(l[i])
    return highs, lows


def _block(out, why):
    out['status'] = 'BLOCKED_UNSAFE_LOCATION'
    out['reasons'].append(why)
    return out


def evaluate_safe_entry(coin, mode, direction, price, frames, options_walls=None, part_anchors=None, breakout_confirmed=False):
    mode = 'SWING' if str(mode).upper() == 'SWING' else 'SCALP'
    long_ = str(direction).upper() in ('CALL', 'BUY')
    out = {'coin': coin, 'mode': mode, 'direction': 'BUY' if long_ else 'SELL', 'price': price,
           'status': 'BLOCKED_UNSAFE_LOCATION', 'reasons': [], 'zone': None, 'anchor': None,
           'sl': None, 'tp1': None, 'tp2': None, 'net_rr': None, 'ext_atr': None,
           'opposing_wall': None, 'wall_source': None, 'wall_dist_atr': None, 'atr': None,
           'frame': PRIMARY_TF[mode]}
    try:
        price = float(price)
        pf, sf = PRIMARY_TF[mode], SECONDARY_TF[mode]
        fm = frames.get(pf)
        if fm is None or len(fm) < 40:
            return _block(out, '%s frame unavailable' % pf)
        atrm = _atr(fm)
        out['atr'] = round(atrm, 4) if atrm else None
        vwap = _vwap(fm, VWAP_BARS[mode])
        ema20 = _num(fm['close'].astype(float).ewm(span=20, adjust=False).mean().iloc[-1])
        if not (atrm and vwap and ema20):
            return _block(out, 'atr/vwap/ema unavailable')
        fair = (vwap + ema20) / 2.0
        if mode == 'SCALP':
            ema9 = _num(fm['close'].astype(float).ewm(span=9, adjust=False).mean().iloc[-1])
            vw2 = _vwap(fm, SCALP_FAST_VWAP_BARS)
            cands = {'5m EMA9': ema9, '5m EMA20': ema20, '5m 2h-VWAP': vw2}
            cands = {k: v for k, v in cands.items() if v}
            def _ex(a):
                return ((price - a) if long_ else (a - price)) / atrm
            behind = {k: v for k, v in cands.items() if _ex(v) >= 0}
            pool = behind if behind else cands
            an_name, an_val = min(pool.items(), key=lambda kv: abs(_ex(kv[1])))
            fair = an_val
            out['fast_anchor'] = an_name
            out['anchor'] = an_name
        else:
            out['anchor'] = '%s VWAP/EMA20 fair value' % pf
        ext = ((price - fair) if long_ else (fair - price)) / atrm
        out['ext_atr'] = round(ext, 3)
        out['fair'] = round(fair, 4)
        highs, lows = _pivots(fm, 48 if mode == 'SCALP' else 72, 4 if mode == 'SCALP' else 2)  # SCALP: only 4-bar-each-side (20m) 5m pivots count as walls, not 2-bar noise
        lvl_src = {}
        for _v in highs + lows:
            lvl_src[round(_v, 4)] = '%s_pivot_k%d' % (pf, 4 if mode == 'SCALP' else 2)
        fs = frames.get(sf)
        if fs is not None and len(fs) >= 40:
            h2, l2 = _pivots(fs)
            for _v in h2 + l2:
                lvl_src[round(_v, 4)] = '%s_pivot_k2' % sf
            highs, lows = highs + h2, lows + l2
        ow = options_walls or {}
        sup, res = _num(ow.get('support')), _num(ow.get('resistance'))
        if sup and not (0.5 * price < sup < 1.5 * price):
            sup = None
        if res and not (0.5 * price < res < 1.5 * price):
            res = None
        if sup:
            lvl_src[round(sup, 4)] = 'options_support_wall'
        if res:
            lvl_src[round(res, 4)] = 'options_resistance_wall'
        supports = sorted([x for x in lows + ([sup] if sup else []) if x < price])
        resistances = sorted([x for x in highs + ([res] if res else []) if x > price])
        if part_anchors:
            _ps, _pr = part_anchors[0], part_anchors[1]
            _psrc = part_anchors[3] if len(part_anchors) > 3 and isinstance(part_anchors[3], dict) else {}
            for _v in _ps + _pr:
                lvl_src[round(_v, 4)] = _psrc.get(round(_v, 4), 'part_anchor_15m_1h')
            supports = sorted(set(supports + [x for x in _ps if x < price]))
            resistances = sorted(set(resistances + [x for x in _pr if x > price]))
            out['part_anchor_n'] = [len(_ps), len(_pr)]
        entry = price
        out['entry'] = round(entry, 4)
        band = BAND_ATR * atrm
        off = OFFSET_ATR[mode] * atrm
        if mode == 'SCALP' and ext > SCALP_EXHAUST_ATR:
            return _block(out, 'exhaustion: %.2f ATR(5m) from nearest fast anchor (%s) > %.1f' % (ext, out.get('fast_anchor'), SCALP_EXHAUST_ATR))
        if mode == 'SCALP' and breakout_confirmed:
            out['status'] = 'IMMEDIATE_SAFE_ENTRY'
            out['zone'] = [round(price - band, 4), round(price + band, 4)]
            out['archetype'] = 'MICRO_BREAKOUT'
        elif ext > MAX_EXT_ATR[mode]:
            entry = fair
            out['entry'] = round(entry, 4)
            out['status'] = 'WAIT_FOR_PULLBACK'
            out['zone'] = [round(fair - band, 4), round(fair + band, 4)]
            out['anchor'] = ('%s fast anchor' % out.get('fast_anchor')) if mode == 'SCALP' else '%s VWAP/EMA20 fair value' % pf
            out['optimal_entry'] = round(entry, 4)
            supports = [x for x in supports if x < entry]
            resistances = [x for x in resistances if x > entry]
        else:
            out['status'] = 'IMMEDIATE_SAFE_ENTRY'
            out['zone'] = [round(price - band, 4), round(price + band, 4)]
        if long_:
            inv = [x for x in supports if x < entry - off]
            if mode == 'SCALP':
                _mn = max(SCALP_SL_MIN_PCT / 100.0 * entry, 0.8 * atrm)
                inv = [x for x in inv if (entry - (x - SL_BUF_ATR[mode] * atrm)) >= _mn]
            sl_lvl = inv[-1] if inv else None
            sl = (sl_lvl - SL_BUF_ATR[mode] * atrm) if sl_lvl else None
            obs = [x for x in resistances if x > entry]
        else:
            inv = [x for x in resistances if x > entry + off]
            if mode == 'SCALP':
                _mn = max(SCALP_SL_MIN_PCT / 100.0 * entry, 0.8 * atrm)
                inv = [x for x in inv if ((x + SL_BUF_ATR[mode] * atrm) - entry) >= _mn]
            sl_lvl = inv[0] if inv else None
            sl = (sl_lvl + SL_BUF_ATR[mode] * atrm) if sl_lvl else None
            obs = sorted([x for x in supports if x < entry], reverse=True)
        if obs:
            first_obs = obs[0]
            out['opposing_wall'] = round(first_obs, 4)
            out['wall_source'] = lvl_src.get(round(first_obs, 4), 'structure')
            out['wall_dist_atr'] = round(abs(first_obs - entry) / atrm, 3)
        if sl is None:
            return _block(out, 'no structural invalidation level')
        out['sl'], out['sl_anchor'] = round(sl, 4), round(sl_lvl, 4)
        out['sl_source'] = lvl_src.get(round(sl_lvl, 4), 'structure')
        risk = abs(entry - sl)
        if obs:
            sgn_p = 1.0 if long_ else -1.0
            tp1_p = obs[0] - sgn_p * off
            nxt_p = [x for x in obs if abs(x - entry) > abs(obs[0] - entry) + WALL_MIN_ATR[mode] * atrm]
            tp2_p = (nxt_p[0] - sgn_p * off) if nxt_p else None
            out['tp1'], out['tp2'] = round(tp1_p, 4), (round(tp2_p, 4) if tp2_p else None)
            if risk > 0:
                out['net_rr'] = round((abs(tp1_p - entry) / entry * 100.0 - COST_PCT) / (risk / entry * 100.0), 3)
        if mode == 'SCALP' and risk / entry * 100.0 > SCALP_SL_MAX_PCT:
            return _block(out, 'scalp SL too wide (%.2f%% > %.2f%%)' % (risk / entry * 100.0, SCALP_SL_MAX_PCT))
        if risk < RISK_MIN_ATR[mode] * atrm and mode != 'SCALP':
            return _block(out, 'SL inside noise (<%.1f ATR %s)' % (RISK_MIN_ATR[mode], pf))
        if risk > RISK_MAX_ATR[mode] * atrm:
            return _block(out, 'SL too wide (>%.1f ATR %s)' % (RISK_MAX_ATR[mode], pf))
        if not obs:
            return _block(out, 'no opposing structure (target unbounded)')
        first = obs[0]
        sgn = 1.0 if long_ else -1.0
        tp1_initial = first - sgn * off
        rr_initial = (abs(tp1_initial - entry) / entry * 100.0 - COST_PCT) / (risk / entry * 100.0) if risk > 0 else 0.0
        if mode == 'SCALP' and (abs(first - entry) < WALL_MIN_ATR[mode] * atrm or rr_initial < MIN_RR[mode]):
            # Phase 10 rule: minor 5m pivots should not act as hard blocking opposing walls
            # if 15m/1h major pivot / POC / liquidity pool room is clear and net R:R >= 1.5 is achievable.
            major_obs = [x for x in obs if lvl_src.get(round(x, 4), '') != '5m_pivot_k4']
            for maj_cand in major_obs:
                if WALL_MIN_ATR[mode] * atrm <= abs(maj_cand - entry) <= 3.0 * atrm:  # never chase a distant wall past a real 5m wall
                    tp1_m = maj_cand - sgn * off
                    rr_m = (abs(tp1_m - entry) / entry * 100.0 - COST_PCT) / (risk / entry * 100.0)
                    if rr_m >= MIN_RR[mode]:
                        obs = [x for x in obs if abs(x - entry) >= abs(maj_cand - entry)]
                        first = maj_cand
                        out['opposing_wall'] = round(first, 4)
                        out['wall_source'] = lvl_src.get(round(first, 4), 'major_structure') + '(bypassed_minor_5m_pivot)'
                        out['wall_dist_atr'] = round(abs(first - entry) / atrm, 3)
                        break
        if abs(first - entry) < WALL_MIN_ATR[mode] * atrm:
            return _block(out, 'opposing wall %.4f (%s) closer than %.2f ATR %s (dist=%.2f ATR)' % (
                first, out.get('wall_source') or 'structure', WALL_MIN_ATR[mode], pf, abs(first - entry) / atrm))
        tp1 = first - sgn * off
        nxt = [x for x in obs if abs(x - entry) > abs(first - entry) + WALL_MIN_ATR[mode] * atrm]
        tp2 = (nxt[0] - sgn * off) if nxt else None
        if mode == 'SCALP' and abs(tp1 - entry) > 3.0 * atrm:
            # Realistic scalp TP1 (<=3 ATR of 5m): the far wall becomes TP2 so profit is booked before mean reversion.
            tp2 = tp1
            tp1 = entry + sgn * 3.0 * atrm
        out['tp1'], out['tp2'] = round(tp1, 4), (round(tp2, 4) if tp2 else None)
        net_rr = (abs(tp1 - entry) / entry * 100.0 - COST_PCT) / (risk / entry * 100.0)
        out['net_rr'] = round(net_rr, 3)
        out['entry'] = round(entry, 4)
        if net_rr < MIN_RR[mode]:
            return _block(out, 'post-cost R:R %.2f < %.1f' % (net_rr, MIN_RR[mode]))
        return out
    except Exception as e:
        return _block(out, 'engine error %s' % type(e).__name__)
