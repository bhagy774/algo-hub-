import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from binance_data import get_ohlcv_binance
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def run_test():
    print("Fetching LIVE 15m candles for BTCUSDT...")
    df_15m = get_ohlcv_binance(symbol="BTCUSDT", resolution="15m", limit=300)
    df_1h = get_ohlcv_binance(symbol="BTCUSDT", resolution="1h", limit=300)
    df_4h = get_ohlcv_binance(symbol="BTCUSDT", resolution="4h", limit=300)
    
    if df_15m is None or len(df_15m) == 0:
        print("Failed to fetch data.")
        return
        
    print("Injecting a MASSIVE BULLISH SETUP (Double Bottom + Volume Spike) into recent data...")
    # Just like we did for Part 1 yesterday!
    recent_low = df_15m['low'].min()
    base_price = df_15m.iloc[-10]['close']
    
    # Create a clear 'W' bottom
    # Leg 1 down
    df_15m.iloc[-5, df_15m.columns.get_loc('close')] = recent_low * 1.001
    df_15m.iloc[-5, df_15m.columns.get_loc('low')] = recent_low * 0.999
    
    # Bounce up
    df_15m.iloc[-4, df_15m.columns.get_loc('close')] = recent_low * 1.02
    
    # Leg 2 down (Double bottom)
    df_15m.iloc[-3, df_15m.columns.get_loc('close')] = recent_low * 1.002
    df_15m.iloc[-3, df_15m.columns.get_loc('low')] = recent_low * 0.998
    
    # Slight recovery
    df_15m.iloc[-2, df_15m.columns.get_loc('close')] = recent_low * 1.01
    
    # MASSIVE BREAKOUT CANDLE (Bullish Engulfing)
    df_15m.iloc[-1, df_15m.columns.get_loc('open')] = recent_low * 1.01
    df_15m.iloc[-1, df_15m.columns.get_loc('close')] = recent_low * 1.06  # 6% pump
    df_15m.iloc[-1, df_15m.columns.get_loc('high')] = recent_low * 1.065
    df_15m.iloc[-1, df_15m.columns.get_loc('low')] = recent_low * 1.005
    df_15m.iloc[-1, df_15m.columns.get_loc('volume')] = df_15m['volume'].mean() * 15.0 # 15x volume spike!
    
    current_candle = df_15m.iloc[-1].to_dict()
    
    print("\n--- Running PART 2 (Advanced Analysis System) on INJECTED DATA ---")
    part2 = AdvancedAnalysisSystem()
    signals = part2.process_market_data(df_15m, df_1h, df_4h, current_candle)
    
    part2_signal_val = 0
    part2_confidence = 0.0
    part2_thought = "NO_SIGNAL"
    
    if signals and isinstance(signals, list):
        valid = [s for s in signals if isinstance(s, tuple) and len(s) >= 3]
        if valid:
            best = max(valid, key=lambda x: x[1])
            part2_signal_val = 1 if str(best[0]).upper() in ['CALL', 'BUY', 'LONG'] else -1 if str(best[0]).upper() in ['PUT', 'SELL', 'SHORT'] else 0
            part2_confidence = best[1]
            part2_thought = f"Part2(15m): {best[2]}"
            
    print(f"Part 2 Signal: {part2_signal_val}")
    print(f"Part 2 Confidence: {part2_confidence:.2f}")
    print(f"Part 2 Thought:\n{part2_thought}")
    
    print("\n--- Running PART 5 (XGBoost ML) ---")
    part5 = Part5ML()
    context = {'symbol': 'BTCUSDT', 'timeframe': '15m'}
    res_part5 = part5.analyze(df_15m, context=context)
    
    print("XGBoost Signal:", res_part5.get('signal'))
    print("XGBoost Thought:", res_part5.get('thought'))
    
    print("\n--- Running LAYA (Advisory Engine) ---")
    decision_direction = "BUY" if part2_signal_val > 0 else "SELL" if part2_signal_val < 0 else "NO_TRADE"
    
    snapshot = {
        'symbol': 'BTCUSDT',
        'direction': decision_direction,
        'confidence': part2_confidence,
        'price': current_candle['close'],
        'reasons': [part2_thought, res_part5.get('thought', '')]
    }
    
    try:
        advisory = advise(
            snapshot, 
            symbol="BTCUSDT", 
            deterministic_decision=decision_direction,
            enabled=True
        )
        print("Laya Status:", advisory.status)
        print("Laya Suggestion:", advisory.suggestion)
        print("Laya Commentary:\n" + getattr(advisory, 'rationale', 'N/A'))
    except Exception as e:
        print(f"Laya encountered an error: {e}")

if __name__ == "__main__":
    run_test()
