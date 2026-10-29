"""
JARVIS FULL PIPELINE DIAGNOSTIC TEST
=====================================
Live data thi signal tak every step trace kare chhe.
Checks: real data -> parts -> HTF aggregation -> final signal

Run: python jarvis_diagnostic_test.py
"""
import sys, os, time, json
sys.path.insert(0, r'c:\jarvis')
os.environ['JARVIS_PURE_ALGO'] = 'true'
os.environ['JARVIS_OUTPUT_MODE'] = 'clean'

print("=" * 70)
print("JARVIS FULL PIPELINE DIAGNOSTIC TEST")
print("=" * 70)

# ─── STEP 1: Import quantitative_math directly ───────────────────────────────
print("\n[STEP 1] Testing quantitative_math directly with live Binance data...")
try:
    import requests
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": 100},
        timeout=10
    )
    klines = r.json()
    # Convert to list of dicts
    candles = []
    for k in klines:
        candles.append({
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close":float(k[4]),
            "volume":float(k[5]),
        })
    print(f"  [OK] Fetched {len(candles)} live BTCUSDT 1m candles from Binance")
    print(f"  Latest close: ${candles[-1]['close']:,.2f}")
except Exception as e:
    print(f"  [FAIL] Binance fetch error: {e}")
    candles = []

# ─── STEP 2: Run quantitative_math on each part ──────────────────────────────
if candles:
    print("\n[STEP 2] Running quantitative_math on live candles for each part...")
    try:
        import quantitative_math as qm
        for part_num in ["1", "3", "4", "5", "6", "7", "8", "9", "10"]:
            res = qm.part_signal(candles, part_num)
            sig = res.get('signal', 0)
            conf = res.get('confidence', 0)
            thought = res.get('thought', 'N/A')
            tele = res.get('telemetry', {})
            direction = "BUY  " if sig > 0 else ("SELL " if sig < 0 else "NEUT ")
            print(f"  Part{part_num:>2}: [{direction}] conf={conf:.1f}%  | {thought[:60]}")
            # Print key telemetry values for parts that are neutral
            if sig == 0 and tele:
                ts = tele.get('trend_score', None)
                vz = tele.get('volume_surprise_robust_z', None)
                kv = tele.get('kalman_velocity_z', None)
                fp = tele.get('flow_imbalance_proxy', None)
                details = []
                if ts is not None: details.append(f"trend={ts:.2f}")
                if vz is not None: details.append(f"vol_z={vz:.2f}")
                if kv is not None: details.append(f"kalman_v={kv:.2f}")
                if fp is not None: details.append(f"flow={fp:.2f}")
                if details:
                    print(f"         -> Why Neutral? {' | '.join(details)}")
    except Exception as e:
        print(f"  [FAIL] quantitative_math error: {e}")
        import traceback; traceback.print_exc()

# ─── STEP 3: Simulate HTF aggregation ────────────────────────────────────────
if candles:
    print("\n[STEP 3] Simulating HTF aggregation (what jarvis_FIXED.py does internally)...")
    try:
        import pandas as pd
        import numpy as np
        import quantitative_math as qm

        # Build a 1m dataframe
        df = pd.DataFrame(candles)
        df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='1min')

        # Resample to higher timeframes
        ohlcv_agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        tf_rules = {'1m': None, '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h'}
        tf_weights = {'1m': 1.0, '5m': 2.0, '15m': 3.0, '30m': 4.0, '1h': 5.0}

        mtf_data = {'1m': df}
        for tf, rule in tf_rules.items():
            if rule:
                resampled = df.resample(rule).agg(ohlcv_agg).dropna()
                if len(resampled) >= 3:
                    mtf_data[tf] = resampled

        print(f"  Timeframes built: {[(tf, len(d)) for tf, d in mtf_data.items()]}")

        # For each part, check all timeframes
        print("\n  Per-Part HTF breakdown:")
        for part_num in ["1", "5", "6", "8"]:
            signals_per_tf = {}
            for tf_name, tf_df in mtf_data.items():
                rows = []
                for _, row in tf_df.iterrows():
                    rows.append({
                        'open': float(row['open']), 'high': float(row['high']),
                        'low': float(row['low']), 'close': float(row['close']),
                        'volume': float(row['volume'])
                    })
                if len(rows) >= 3:
                    res = qm.part_signal(rows, part_num)
                    signals_per_tf[tf_name] = res.get('signal', 0)

            # Calculate HTF avg
            htf_sum = sum(signals_per_tf.get(tf, 0) * w
                         for tf, w in tf_weights.items() if tf != '1m')
            htf_total = sum(w for tf, w in tf_weights.items()
                           if tf != '1m' and tf in signals_per_tf)
            htf_avg = htf_sum / htf_total if htf_total > 0 else 0.0

            base_1m = signals_per_tf.get('1m', 0)

            # Apply Option A HTF logic
            if base_1m == 0:
                if abs(htf_avg) >= 0.40:
                    final = 1 if htf_avg > 0 else -1
                    decision = f"HTF-LED ({'+' if htf_avg>0 else ''}{htf_avg:.2f})"
                else:
                    final = 0
                    decision = f"NEUTRAL (HTF avg={htf_avg:.2f} < 0.40 threshold)"
            else:
                final = base_1m
                decision = f"1m-based ({'BUY' if final > 0 else 'SELL'})"

            tf_str = " | ".join([f"{tf}={'B' if s>0 else ('S' if s<0 else 'N')}"
                                  for tf, s in signals_per_tf.items()])
            print(f"  Part {part_num}: {tf_str}")
            print(f"         HTF avg={htf_avg:+.3f} | 1m={base_1m} | FINAL={final} -> {decision}")

    except Exception as e:
        print(f"  [FAIL] HTF simulation error: {e}")
        import traceback; traceback.print_exc()

# ─── STEP 4: Check MULTICOIN scheduler fix ───────────────────────────────────
print("\n[STEP 4] Checking MULTICOIN scheduler fix in jarvis_multicoin_pipeline.py...")
try:
    with open(r'c:\jarvis\jarvis_multicoin_pipeline.py', 'r', encoding='utf-8') as f:
        content = f.read()
    if '"interpreter shutdown" in str(exc)' in content:
        print("  [OK] MULTICOIN fix is in place - interpreter shutdown will stop loop cleanly")
    else:
        print("  [WARN] MULTICOIN fix NOT found - spam errors will continue on Ctrl+C")
except Exception as e:
    print(f"  [FAIL] {e}")

# ─── STEP 5: Check HTF fix in jarvis_FIXED.py ────────────────────────────────
print("\n[STEP 5] Checking HTF-led fix in jarvis_FIXED.py...")
try:
    with open(r'c:\jarvis\jarvis_FIXED.py', 'r', encoding='utf-8') as f:
        content = f.read()
    if 'HTF-led: 1m flat but HTF avg=' in content:
        print("  [OK] Option A HTF-led fix is active in jarvis_FIXED.py!")
    else:
        print("  [WARN] HTF-led fix NOT found - signals will still be blocked by 1m neutral")
    if 'abs(htf_avg) >= 0.40' in content:
        print("  [OK] HTF threshold (0.40) is correctly set")
    if 'JARVIS_PURE_ALGO' in content:
        print("  [OK] PURE_ALGO mode is enabled - no Ollama calls")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("""
INTERPRETATION GUIDE:
  If Part signals show 'NEUT' on 1m but HTF avg >= 0.40:
    → HTF-Led fix kaam karase, signal generate thase

  If HTF avg < 0.40 on all parts:
    → Market truly sideways/consolidating - Neutral is correct!

  If trend_score < 1.25 on all parts:
    → Market maa koi strong direction nathi abhi
""")
