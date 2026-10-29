"""
DUAL MODE BACKTEST
==================
1) INSTITUTIONAL BIAS  (1h)  -> Part 1 + Part 2 (all brains) + EMA trend
2) SWING MODE          (1h)  -> trade only when institutional bias is STRONG
3) SCALP MODE          (5m)  -> fast brains (momentum / S-R bounce / volatility)
   RULE: a scalp is NEVER taken opposite to the institutional bias.
         Bias BUY  -> only BUY scalps
         Bias SELL -> only SELL scalps
         Bias NEUTRAL (range) -> both sides allowed (range scalps)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from part1_FIXED import SmartBreakoutAI
from part2_FIXED import AdvancedAnalysisSystem

SYMBOL = "BTCUSDT"
N_5M = 2000          # ~7 days of 5m candles
SCALP_TP, SCALP_SL, SCALP_HOLD = 0.0025, 0.0015, 12     # 0.25% / 0.15% / 1h max
SWING_TP, SWING_SL, SWING_HOLD = 0.0150, 0.0075, 48     # 1.5% / 0.75% / 48h max


# ---------------------------------------------------------------- data
def fetch(interval, total, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    out = []
    while len(out) < total:
        params = {"symbol": SYMBOL, "interval": interval, "limit": min(1000, total - len(out))}
        if end_time:
            params["endTime"] = int(end_time)
        data = requests.get(url, params=params, timeout=20).json()
        if not data:
            break
        out = data + out
        end_time = data[0][0] - 1
    df = pd.DataFrame(out).iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    return df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)


def to_dir(v):
    s = str(v).upper()
    if s in ('CALL', 'BUY', 'LONG') or (isinstance(v, (int, float)) and v > 0):
        return "BUY"
    if s in ('PUT', 'SELL', 'SHORT') or (isinstance(v, (int, float)) and v < 0):
        return "SELL"
    return "NEUTRAL"


# ---------------------------------------------------------------- institutional
def institutional_bias(part1, part2, h1):
    """Returns (bias, strength 0..3) from Part1, Part2 and 1h EMA trend."""
    votes = []
    try:
        votes.append(to_dir(part1.analyze(h1).get('signal', 0)))
    except Exception:
        votes.append("NEUTRAL")
    try:
        sigs = part2.process_market_data(h1, h1, h1, h1.iloc[-1].to_dict())
        valid = [s for s in (sigs or []) if isinstance(s, tuple) and len(s) >= 2]
        votes.append(to_dir(max(valid, key=lambda x: x[1])[0]) if valid else "NEUTRAL")
    except Exception:
        votes.append("NEUTRAL")
    c = h1['close']
    e50, e200 = c.ewm(span=50).mean().iloc[-1], c.ewm(span=200).mean().iloc[-1]
    votes.append("BUY" if c.iloc[-1] > e50 > e200 else "SELL" if c.iloc[-1] < e50 < e200 else "NEUTRAL")

    b, s = votes.count("BUY"), votes.count("SELL")
    if b >= 2 and s == 0:
        return "BUY", b, votes
    if s >= 2 and b == 0:
        return "SELL", s, votes
    return "NEUTRAL", 0, votes


# ---------------------------------------------------------------- fast scalp brains
def scalp_signal(w):
    """3 fast brains on 5m; need 2 of 3 to agree."""
    c, h, l = w['close'], w['high'], w['low']
    votes = []
    # Brain 1: Momentum (EMA9/21 + RSI)
    e9, e21 = c.ewm(span=9).mean(), c.ewm(span=21).mean()
    d = c.diff()
    rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs.iloc[-1])
    votes.append("BUY" if e9.iloc[-1] > e21.iloc[-1] and 50 < rsi < 70 else
                 "SELL" if e9.iloc[-1] < e21.iloc[-1] and 30 < rsi < 50 else "NEUTRAL")
    # Brain 2: Support/Resistance bounce (range scalp)
    sup, res = l.iloc[-30:-1].min(), h.iloc[-30:-1].max()
    rng = max(res - sup, 1e-9)
    pos = (c.iloc[-1] - sup) / rng
    bull = c.iloc[-1] > w['open'].iloc[-1]
    votes.append("BUY" if pos < 0.2 and bull else "SELL" if pos > 0.8 and not bull else "NEUTRAL")
    # Brain 3: Volatility expansion candle direction
    tr = (h - l)
    big = tr.iloc[-1] > tr.iloc[-20:-1].mean() * 1.3
    votes.append(("BUY" if bull else "SELL") if big else "NEUTRAL")

    b, s = votes.count("BUY"), votes.count("SELL")
    return ("BUY" if b >= 2 else "SELL" if s >= 2 else "NEUTRAL"), votes


def simulate(df, idx, direction, tp_pct, sl_pct, hold):
    entry = df['close'].iloc[idx]
    tp = entry * (1 + tp_pct) if direction == "BUY" else entry * (1 - tp_pct)
    sl = entry * (1 - sl_pct) if direction == "BUY" else entry * (1 + sl_pct)
    for j in range(idx + 1, min(idx + 1 + hold, len(df))):
        hi, lo = df['high'].iloc[j], df['low'].iloc[j]
        if direction == "BUY":
            if lo <= sl: return "LOSS", j
            if hi >= tp: return "WIN", j
        else:
            if hi >= sl: return "LOSS", j
            if lo <= tp: return "WIN", j
    ex = df['close'].iloc[min(idx + hold, len(df) - 1)]
    win = ex > entry if direction == "BUY" else ex < entry
    return ("WIN" if win else "LOSS") + "(time)", min(idx + hold, len(df) - 1)


# ---------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("DUAL MODE BACKTEST  |  Institutional bias gates every scalp")
    print("=" * 60)
    m5 = fetch("5m", N_5M)
    h1 = fetch("1h", N_5M // 12 + 260)
    print(f"[+] 5m candles: {len(m5)} | 1h candles: {len(h1)}")

    part1, part2 = SmartBreakoutAI(), AdvancedAnalysisSystem()

    # --- compute institutional bias once per closed 1h candle
    print("[*] Computing institutional bias per hour (Part1 + Part2 + EMA)...")
    start_ts = m5['timestamp'].iloc[0]
    bias_map = {}
    hrs = h1.index[(h1['timestamp'] >= start_ts - 3600_000) & (h1.index >= 200)]
    for n, k in enumerate(hrs):
        bias_map[h1['timestamp'].iloc[k]] = institutional_bias(part1, part2, h1.iloc[k - 200:k + 1].reset_index(drop=True))
        if n % 20 == 0:
            print(f"    hour {n}/{len(hrs)}  bias={bias_map[h1['timestamp'].iloc[k]][0]}")
    bias_ts = sorted(bias_map)

    def bias_at(ts):
        # last fully CLOSED hour before ts (no look-ahead)
        prev = [t for t in bias_ts if t + 3600_000 <= ts]
        return bias_map[prev[-1]] if prev else ("NEUTRAL", 0, [])

    # --- SWING MODE on 1h
    swing = []
    k_list = [h1.index[h1['timestamp'] == t][0] for t in bias_ts]
    skip_until = -1
    for k in k_list:
        if k <= skip_until:
            continue
        bias, strength, _ = bias_map[h1['timestamp'].iloc[k]]
        if bias != "NEUTRAL" and strength >= 3:
            res, ex = simulate(h1, k, bias, SWING_TP, SWING_SL, SWING_HOLD)
            swing.append(res); skip_until = ex
            print(f"  [SWING] {bias} @ {h1['close'].iloc[k]:.2f} -> {res}")

    # --- SCALP MODE on 5m (gated by institutional bias)
    scalp, blocked = [], 0
    i = 60
    while i < len(m5) - 1:
        sig, _ = scalp_signal(m5.iloc[i - 60:i + 1])
        if sig != "NEUTRAL":
            bias = bias_at(m5['timestamp'].iloc[i] + 300_000)[0]
            if bias != "NEUTRAL" and sig != bias:
                blocked += 1          # opposite to institutions -> NOT taken
            else:
                res, ex = simulate(m5, i, sig, SCALP_TP, SCALP_SL, SCALP_HOLD)
                scalp.append((res, sig, bias))
                i = ex
        i += 1

    def stats(lst):
        w = sum(1 for r in lst if r.startswith("WIN"))
        return len(lst), w, (w / len(lst) * 100 if lst else 0)

    print("\n" + "=" * 60)
    t, w, wr = stats(swing)
    print(f"SWING  (institutional, 1h): trades={t} wins={w} winrate={wr:.1f}%")
    t, w, wr = stats([r for r, _, _ in scalp])
    print(f"SCALP  (5m, bias-gated)   : trades={t} wins={w} winrate={wr:.1f}%")
    for b in ("BUY", "SELL", "NEUTRAL"):
        t, w, wr = stats([r for r, _, bb in scalp if bb == b])
        print(f"   with bias {b:7}: trades={t} wins={w} winrate={wr:.1f}%")
    print(f"Scalps BLOCKED (opposite to institutions): {blocked}")
    print(f"R:R  scalp 1:{SCALP_TP/SCALP_SL:.2f} | swing 1:{SWING_TP/SWING_SL:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
