"""Backtest for mode_engine.py on Binance history (no look-ahead)."""
import os, sys, time
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import pandas as pd
from test_dual_mode_backtest import fetch, simulate
from mode_engine import ModeEngine, CLOCK

N_5M = int(os.environ.get("N_5M", 2000))
SCALP_RR, SWING_RR = 1.67, 2.0


def main():
    t0 = time.time()
    m5 = fetch("5m", N_5M)
    h1 = fetch("1h", N_5M // 12 + 260)
    print(f"[+] 5m={len(m5)}  1h={len(h1)}")
    eng = ModeEngine()

    # ---- hourly: mode + institutional bias + swing entries
    start = m5['timestamp'].iloc[0] - 3600_000
    hourly = {}
    swing, skip = [], -1
    swing_votes_hist = Counter()
    for k in h1.index[(h1['timestamp'] >= start) & (h1.index >= 200)]:
        win = h1.iloc[k - 200:k + 1].reset_index(drop=True)
        close_ts = h1['timestamp'].iloc[k] + 3600_000
        CLOCK.now = close_ts / 1000
        mode, r1, r2 = eng.mode(win)
        bias, entry, votes = eng.institutional_bias(win)
        hourly[close_ts] = (mode, bias)
        swing_votes_hist[max(list(votes.values()).count("BUY"), list(votes.values()).count("SELL"))] += 1
        if entry and k > skip and k < len(h1) - 1:
            atr = (h1['high'] - h1['low']).iloc[k - 14:k + 1].mean()
            sl = 1.5 * atr / h1['close'].iloc[k]
            res, ex = simulate(h1, k, entry, sl * SWING_RR, sl, 48)
            swing.append(res); skip = ex
            print(f"  [SWING] {entry} @ {h1['close'].iloc[k]:.1f}  {votes}  -> {res}")
    ts_sorted = sorted(hourly)
    print(f"[+] hourly done ({time.time()-t0:.0f}s). modes={Counter(m for m,_ in hourly.values())} "
          f"bias={Counter(b for _,b in hourly.values())}")

    def ctx(ts):
        prev = [t for t in ts_sorted if t <= ts]
        return hourly[prev[-1]] if prev else ("RANGE", "NEUTRAL")

    # ---- 5m scalps
    scalps, blocked, agree_hist = [], Counter(), Counter()
    i = 150
    while i < len(m5) - 1:
        bar_close = m5['timestamp'].iloc[i] + 300_000
        CLOCK.now = bar_close / 1000
        mode, bias = ctx(bar_close)
        d, reason, votes, stop = eng.scalp(m5.iloc[i - 150:i + 1].reset_index(drop=True), mode, bias)
        v = list(votes.values())
        agree_hist[max(v.count("BUY"), v.count("SELL"))] += 1
        if reason:
            blocked[reason] += 1
        if d:
            sl = stop / m5['close'].iloc[i]
            res, ex = simulate(m5, i, d, sl * SCALP_RR, sl, 12)
            scalps.append((res, mode, bias))
            print(f"  [SCALP] {d} @ {m5['close'].iloc[i]:.1f} mode={mode} bias={bias} sl={sl*100:.2f}% -> {res}")
            i = ex
        i += 1
        if i % 300 == 0:
            print(f"   ..5m {i}/{len(m5)}  ({time.time()-t0:.0f}s)")

    def st(lst):
        w = sum(1 for r in lst if r.startswith("WIN"))
        return f"trades={len(lst)} wins={w} winrate={(w/len(lst)*100 if lst else 0):.1f}%"

    print("\n" + "=" * 64)
    print(f"SWING (6 swing brains, >=4 agree, 1h, RR 1:{SWING_RR}): {st(swing)}")
    print(f"SCALP (8 scalp brains, >=5 agree, 5m, RR 1:{SCALP_RR}): {st([r for r,_,_ in scalps])}")
    for m in ("TREND", "RANGE"):
        print(f"   in {m:5} mode: {st([r for r,mm,_ in scalps if mm == m])}")
    print(f"Blocked scalps: {dict(blocked)}")
    print(f"Swing max-agree histogram (of 6): {dict(sorted(swing_votes_hist.items()))}")
    print(f"Scalp max-agree histogram (of 8): {dict(sorted(agree_hist.items()))}")
    print(f"Breakeven winrate: scalp {100/(1+SCALP_RR):.1f}%  swing {100/(1+SWING_RR):.1f}%")
    print(f"Time: {time.time()-t0:.0f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
