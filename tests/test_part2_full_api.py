import json
import pandas as pd
import numpy as np
import warnings
import os
import sys

# Suppress some noisy warnings for cleaner test output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from binance_data import get_ohlcv_binance
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def request_runtime_advisory(snapshot, symbol, deterministic_decision):
    return advise(snapshot, symbol=symbol, deterministic_decision=deterministic_decision)

def run_test():
    print("Fetching LIVE candles for BTCUSDT from Binance API...")
    # Fetch data as Jarvis orchestrator does
    df_15m = get_ohlcv_binance(symbol="BTCUSDT", resolution="15m", limit=300)
    df_1h = get_ohlcv_binance(symbol="BTCUSDT", resolution="1h", limit=300)
    df_4h = get_ohlcv_binance(symbol="BTCUSDT", resolution="4h", limit=300)
    
    if df_15m is None or len(df_15m) == 0:
        print("Failed to fetch data from Binance!")
        return
        
    print(f"Data fetched! 15m: {len(df_15m)} candles, 1h: {len(df_1h)} candles, 4h: {len(df_4h)} candles.")
    
    current_candle = df_15m.iloc[-1].to_dict()
    
    # 1. Run Part 2
    print("\n--- Running PART 2 (Advanced Analysis System - 16 Brains) ---")
    part2 = AdvancedAnalysisSystem()
    print("Running process_market_data()...")
    signals = part2.process_market_data(df_15m, df_1h, df_4h, current_candle)
    
    part2_thought = "NO_SIGNAL"
    part2_signal_val = 0
    part2_confidence = 0.0
    
    if signals and isinstance(signals, list):
        valid = [s for s in signals if isinstance(s, tuple) and len(s) >= 3]
        if valid:
            best = max(valid, key=lambda x: x[1])
            part2_signal_val = 1 if str(best[0]).upper() in ['CALL', 'BUY', 'LONG'] else -1
            part2_confidence = best[1]
            part2_thought = f"Part2(15m): {best[2]}"
            
    print(f"Part 2 Final Decision: {part2_signal_val}")
    print(f"Part 2 Confidence: {part2_confidence:.2f}")
    print(f"Part 2 Reason/Thought:\n{part2_thought}")
    
    # 2. Run XGBoost (Part 5)
    print("\n--- Running PART 5 (XGBoost Tabular Alpha) ---")
    part5 = Part5ML()
    context = {'symbol': 'BTCUSDT', 'timeframe': '15m'}
    res_part5 = part5.analyze(df_15m, context=context)
    
    print("XGBoost Signal:", res_part5.get('signal'))
    print("XGBoost Thought:", res_part5.get('thought'))
    
    # 3. Run Laya Auditor
    print("\n--- Running LAYA (HuggingFace/AI Router) ---")
    decision_direction = "BUY" if part2_signal_val > 0 else "SELL" if part2_signal_val < 0 else "NO_TRADE"
    
    print(f"Sending decision snapshot to Laya for audit...")
    advisory = request_runtime_advisory({
        'symbol': 'BTCUSDT',
        'direction': decision_direction,
        'confidence': part2_confidence,
        'price': current_candle['close'],
        'reasons': [part2_thought, res_part5.get('thought', '')]
    }, symbol='BTCUSDT', deterministic_decision=decision_direction)
    
    print("Laya Status:", advisory.status)
    print("Laya Suggestion:", advisory.suggestion)
    print("Laya Commentary:\n" + getattr(advisory, 'rationale', 'N/A'))

if __name__ == "__main__":
    run_test()
