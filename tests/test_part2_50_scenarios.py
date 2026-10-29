import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress huggingface logs if possible
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def generate_synthetic_data(scenario_type, base_price=50000.0, length=100):
    """Generates synthetic 15m OHLCV data to force specific signals."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='15min')
    df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Base random walk
    np.random.seed()
    closes = [base_price]
    for _ in range(1, length):
        change = np.random.normal(0, base_price * 0.001)
        closes.append(closes[-1] + change)
        
    df['close'] = closes
    df['open'] = df['close'].shift(1).fillna(base_price)
    df['high'] = df[['open', 'close']].max(axis=1) + (base_price * 0.001)
    df['low'] = df[['open', 'close']].min(axis=1) - (base_price * 0.001)
    df['volume'] = np.random.normal(100, 20, length)
    
    # Inject Patterns at the end
    if scenario_type == 'BUY':
        # Double bottom + Bullish Engulfing
        low_pt = df['low'].min() * 0.98
        df.iloc[-5, df.columns.get_loc('low')] = low_pt
        df.iloc[-5, df.columns.get_loc('close')] = low_pt * 1.01
        
        df.iloc[-4, df.columns.get_loc('close')] = low_pt * 1.03
        
        df.iloc[-3, df.columns.get_loc('low')] = low_pt
        df.iloc[-3, df.columns.get_loc('close')] = low_pt * 1.005
        
        df.iloc[-1, df.columns.get_loc('open')] = low_pt * 1.005
        df.iloc[-1, df.columns.get_loc('close')] = low_pt * 1.05
        df.iloc[-1, df.columns.get_loc('high')] = low_pt * 1.055
        df.iloc[-1, df.columns.get_loc('volume')] = 1000.0 # Huge volume
        
    elif scenario_type == 'SELL':
        # Double Top + Bearish Engulfing
        high_pt = df['high'].max() * 1.02
        df.iloc[-5, df.columns.get_loc('high')] = high_pt
        df.iloc[-5, df.columns.get_loc('close')] = high_pt * 0.99
        
        df.iloc[-4, df.columns.get_loc('close')] = high_pt * 0.97
        
        df.iloc[-3, df.columns.get_loc('high')] = high_pt
        df.iloc[-3, df.columns.get_loc('close')] = high_pt * 0.995
        
        df.iloc[-1, df.columns.get_loc('open')] = high_pt * 0.995
        df.iloc[-1, df.columns.get_loc('close')] = high_pt * 0.95
        df.iloc[-1, df.columns.get_loc('low')] = high_pt * 0.945
        df.iloc[-1, df.columns.get_loc('volume')] = 1000.0 # Huge volume
        
    elif scenario_type == 'NEUTRAL':
        # Flat choppy range for entire dataset
        for i in range(length):
            df.iloc[i, df.columns.get_loc('open')] = base_price
            df.iloc[i, df.columns.get_loc('close')] = base_price * (1 + np.random.normal(0, 0.0001))
            df.iloc[i, df.columns.get_loc('high')] = base_price * 1.0005
            df.iloc[i, df.columns.get_loc('low')] = base_price * 0.9995
            df.iloc[i, df.columns.get_loc('volume')] = 10.0 # Very low volume
            
    df = df.astype(float)
    return df

def run_50_tests():
    print("Initializing Jarvis Part 2 and Part 5 Engines...")
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()
    
    scenarios = (['BUY'] * 15) + (['SELL'] * 15) + (['NEUTRAL'] * 20)
    np.random.shuffle(scenarios)
    
    results = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
    laya_successes = 0
    
    for i, s_type in enumerate(scenarios):
        print(f"\n--- Test {i+1}/50: Target [{s_type}] ---")
        df_15m = generate_synthetic_data(s_type)
        df_1h = df_15m.copy() # Mock higher tf
        df_4h = df_15m.copy()
        current_candle = df_15m.iloc[-1].to_dict()
        
        # 1. Run Part 2
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
                part2_thought = f"Part2: {best[2]}"
                
        decision_direction = "BUY" if part2_signal_val > 0 else "SELL" if part2_signal_val < 0 else "NEUTRAL"
        
        if part2_signal_val == 0 and part2_confidence < 3.0:
            decision_direction = "NEUTRAL"
            
        results[decision_direction] += 1
        print(f"Part 2 Decision: {decision_direction} (Conf: {part2_confidence:.2f}) | Thought: {part2_thought}")
        
        # 2. Run XGBoost
        context = {'symbol': 'BTCUSDT', 'timeframe': '15m'}
        res_part5 = part5.analyze(df_15m, context=context)
        print(f"XGBoost Signal: {res_part5.get('signal')}")
        
        # 3. Run Laya
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
            if advisory.status == 'ok':
                laya_successes += 1
            print(f"Laya Status: {advisory.status}")
        except Exception as e:
            pass
            
    print("\n==================================================")
    print("50 TESTS SUMMARY REPORT")
    print("==================================================")
    print(f"Total Tests Run: 50")
    print(f"BUY Signals Generated: {results['BUY']}")
    print(f"SELL Signals Generated: {results['SELL']}")
    print(f"NEUTRAL Signals Generated: {results['NEUTRAL']}")
    print(f"Laya Audit Successes: {laya_successes} / 50")
    print("==================================================")

if __name__ == '__main__':
    run_50_tests()
