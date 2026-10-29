#!/usr/bin/env python3
"""
Quick test: Check if Binance provides all data needed by Jarvis parts
"""
import sys, time
sys.path.insert(0, '/root/jarvis-scalp')

print("=" * 60)
print("  JARVIS BINANCE DATA TEST")
print("=" * 60)

from binance_data import BinanceData
bd = BinanceData()

# ─── 1. LIVE PRICE ─────────────────────────────────────────
print("\n[1] LIVE PRICE TEST")
price = bd.get_live_price("BTCUSDT")
if price > 100:
    print(f"  ✅ BTC Live Price: ${price:,.2f}")
else:
    print(f"  ❌ Price failed: {price}")

# ─── 2. CANDLES FOR ALL TIMEFRAMES ─────────────────────────
print("\n[2] CANDLE DATA TEST (All Timeframes)")
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"]
all_ok = True
for tf in timeframes:
    candles = bd.get_historical_candles("BTCUSDT", tf, 100)
    if candles and len(candles) > 10:
        last_close = candles[-1]['close']
        has_volume = candles[-1].get('volume', 0) > 0
        vol_str = "📊 has volume" if has_volume else "⚠️  volume=0"
        print(f"  ✅ {tf:4s}: {len(candles)} candles | Last close: ${last_close:,.2f} | {vol_str}")
    else:
        print(f"  ❌ {tf}: FAILED (got {len(candles) if candles else 0} candles)")
        all_ok = False

# ─── 3. OHLCV COLUMN CHECK ─────────────────────────────────
print("\n[3] DATA FORMAT CHECK (Parts compatibility)")
import pandas as pd
candles = bd.get_historical_candles("BTCUSDT", "1m", 500)
df = pd.DataFrame(candles)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.index = pd.to_datetime(df['time'], unit='s')
df = df.sort_index()

required_cols = ['open', 'high', 'low', 'close', 'volume']
for col in required_cols:
    ok = col in df.columns and df[col].notna().sum() > 10
    print(f"  {'✅' if ok else '❌'} Column '{col}': {df[col].notna().sum()} valid rows")

print(f"\n  DataFrame shape: {df.shape}")
print(f"  Time range: {df.index[0]} → {df.index[-1]}")

# ─── 4. PRICE ACCURACY ─────────────────────────────────────
print("\n[4] PRICE ACCURACY vs LAST CANDLE")
last_candle_close = float(df['close'].iloc[-1])
live_price = bd.get_live_price("BTCUSDT")
diff = abs(live_price - last_candle_close)
diff_pct = (diff / live_price) * 100
print(f"  Live Price:  ${live_price:,.2f}")
print(f"  Last Candle: ${last_candle_close:,.2f}")
print(f"  Difference:  ${diff:.2f} ({diff_pct:.3f}%)")
if diff_pct < 0.5:
    print("  ✅ Price match - data is LIVE!")
else:
    print("  ⚠️  Price gap > 0.5% - candle might be delayed")

# ─── FINAL RESULT ──────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok and price > 100:
    print("  🎉 ALL TESTS PASSED! Binance data is ready for all parts!")
else:
    print("  ⚠️  Some tests failed. Check above.")
print("=" * 60)
