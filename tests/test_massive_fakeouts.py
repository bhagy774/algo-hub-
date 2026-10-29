import json
import pandas as pd
import numpy as np
from binance_data import get_binance_data, get_ohlcv_binance
from jarvis_FIXED import Part1Breakout, Part5ML, Part2Zone
from jarvis_laya_advisor import advise
import time

def inject_bullish_trap(df, idx):
    # Base price before trap
    base_price = df.iloc[idx-5]['close']
    
    # Buildup
    df.iloc[idx-4, df.columns.get_loc('close')] = base_price * 1.002
    df.iloc[idx-4, df.columns.get_loc('high')] = max(df.iloc[idx-4]['high'], base_price * 1.002)
    df.iloc[idx-4, df.columns.get_loc('low')] = min(df.iloc[idx-4]['low'], min(df.iloc[idx-4]['open'], base_price * 1.002))
    
    df.iloc[idx-3, df.columns.get_loc('close')] = base_price * 1.005
    df.iloc[idx-3, df.columns.get_loc('high')] = max(df.iloc[idx-3]['high'], base_price * 1.006)
    df.iloc[idx-3, df.columns.get_loc('low')] = min(df.iloc[idx-3]['low'], min(df.iloc[idx-3]['open'], base_price * 1.005))
    
    # Fakeout (Pump breaking resistance)
    recent_high = max(df['high'].iloc[idx-20:idx-3])
    df.iloc[idx-2, df.columns.get_loc('open')] = df.iloc[idx-3]['close']
    df.iloc[idx-2, df.columns.get_loc('close')] = recent_high * 1.01
    df.iloc[idx-2, df.columns.get_loc('high')] = max(recent_high * 1.02, recent_high * 1.01, df.iloc[idx-3]['close'])
    df.iloc[idx-2, df.columns.get_loc('low')] = min(df.iloc[idx-3]['close'], recent_high * 1.01)
    df.iloc[idx-2, df.columns.get_loc('volume')] = df['volume'].mean() * 5.0
    
    # Dump (The Trap springs)
    df.iloc[idx-1, df.columns.get_loc('open')] = recent_high * 1.01
    df.iloc[idx-1, df.columns.get_loc('close')] = base_price * 0.98
    df.iloc[idx-1, df.columns.get_loc('high')] = max(recent_high * 1.01, base_price * 0.98)
    df.iloc[idx-1, df.columns.get_loc('low')] = min(base_price * 0.97, base_price * 0.98)
    df.iloc[idx-1, df.columns.get_loc('volume')] = df['volume'].mean() * 8.0
    
    # Actual current candle continuing dump
    df.iloc[idx, df.columns.get_loc('open')] = base_price * 0.98
    df.iloc[idx, df.columns.get_loc('close')] = base_price * 0.95
    df.iloc[idx, df.columns.get_loc('high')] = max(base_price * 0.98, base_price * 0.95)
    df.iloc[idx, df.columns.get_loc('low')] = min(base_price * 0.94, base_price * 0.95)
    df.iloc[idx, df.columns.get_loc('volume')] = df['volume'].mean() * 10.0
    return df

def inject_bearish_trap(df, idx):
    base_price = df.iloc[idx-5]['close']
    
    # Buildup
    df.iloc[idx-4, df.columns.get_loc('close')] = base_price * 0.998
    df.iloc[idx-4, df.columns.get_loc('high')] = max(df.iloc[idx-4]['high'], max(df.iloc[idx-4]['open'], base_price * 0.998))
    df.iloc[idx-4, df.columns.get_loc('low')] = min(df.iloc[idx-4]['low'], base_price * 0.998)
    
    df.iloc[idx-3, df.columns.get_loc('close')] = base_price * 0.995
    df.iloc[idx-3, df.columns.get_loc('low')] = min(df.iloc[idx-3]['low'], base_price * 0.994)
    df.iloc[idx-3, df.columns.get_loc('high')] = max(df.iloc[idx-3]['high'], max(df.iloc[idx-3]['open'], base_price * 0.995))
    
    # Fakeout (Dump breaking support)
    recent_low = min(df['low'].iloc[idx-20:idx-3])
    df.iloc[idx-2, df.columns.get_loc('open')] = df.iloc[idx-3]['close']
    df.iloc[idx-2, df.columns.get_loc('close')] = recent_low * 0.99
    df.iloc[idx-2, df.columns.get_loc('low')] = min(recent_low * 0.98, recent_low * 0.99, df.iloc[idx-3]['close'])
    df.iloc[idx-2, df.columns.get_loc('high')] = max(df.iloc[idx-3]['close'], recent_low * 0.99)
    df.iloc[idx-2, df.columns.get_loc('volume')] = df['volume'].mean() * 5.0
    
    # Pump (The Trap springs)
    df.iloc[idx-1, df.columns.get_loc('open')] = recent_low * 0.99
    df.iloc[idx-1, df.columns.get_loc('close')] = base_price * 1.02
    df.iloc[idx-1, df.columns.get_loc('high')] = max(base_price * 1.03, base_price * 1.02)
    df.iloc[idx-1, df.columns.get_loc('low')] = min(recent_low * 0.99, base_price * 1.02)
    df.iloc[idx-1, df.columns.get_loc('volume')] = df['volume'].mean() * 8.0
    
    # Actual current candle continuing pump
    df.iloc[idx, df.columns.get_loc('open')] = base_price * 1.02
    df.iloc[idx, df.columns.get_loc('close')] = base_price * 1.05
    df.iloc[idx, df.columns.get_loc('high')] = max(base_price * 1.06, base_price * 1.05)
    df.iloc[idx, df.columns.get_loc('low')] = min(base_price * 1.02, base_price * 1.05)
    df.iloc[idx, df.columns.get_loc('volume')] = df['volume'].mean() * 10.0
    return df

def inject_clean_bullish_breakout(df, idx):
    base_price = df.iloc[idx-5]['close']
    recent_high = max(df['high'].iloc[idx-20:idx-1])
    
    df.iloc[idx-1, df.columns.get_loc('close')] = recent_high * 0.995
    df.iloc[idx-1, df.columns.get_loc('high')] = max(df.iloc[idx-1]['high'], recent_high * 0.995)
    df.iloc[idx-1, df.columns.get_loc('low')] = min(df.iloc[idx-1]['low'], recent_high * 0.995)
    
    df.iloc[idx, df.columns.get_loc('open')] = recent_high * 0.995
    df.iloc[idx, df.columns.get_loc('close')] = recent_high * 1.05
    df.iloc[idx, df.columns.get_loc('high')] = max(recent_high * 1.06, recent_high * 1.05)
    df.iloc[idx, df.columns.get_loc('low')] = min(recent_high * 0.995, recent_high * 1.05)
    df.iloc[idx, df.columns.get_loc('volume')] = df['volume'].mean() * 10.0
    return df

def run_rolling_test():
    print("Fetching 1000 LIVE 5m candles for BTCUSDT to satisfy XGBoost OHLCV requirements...")
    df = get_ohlcv_binance(symbol="BTCUSDT", resolution="5m", limit=1000)
    df.attrs['timeframe'] = '5m'
    df.attrs['symbol'] = 'BTCUSDT'
    
    print("Injecting Scenarios at specific indices to guarantee traps and breakouts...")
    test_indices = []
    
    # 10 Bullish Traps
    for i in range(10):
        idx = 520 + i * 15
        df = inject_bullish_trap(df, idx)
        test_indices.append(('Bullish Trap (Fakeout)', idx, 0)) # expected signal 0
        
    # 10 Bearish Traps
    for i in range(10):
        idx = 670 + i * 15
        df = inject_bearish_trap(df, idx)
        test_indices.append(('Bearish Trap (Fakeout)', idx, 0)) # expected signal 0
        
    # 5 Clean Bullish Breakouts
    for i in range(5):
        idx = 820 + i * 15
        df = inject_clean_bullish_breakout(df, idx)
        test_indices.append(('Clean Bullish Breakout', idx, 1)) # expected signal 1

    test_indices.sort(key=lambda x: x[1])
    
    # 10 Bullish Traps
    for i in range(10):
        idx = 520 + i * 15
        df = inject_bullish_trap(df, idx)
        test_indices.append(("Bullish Trap (Fakeout)", idx, 0)) # expected signal 0
        
    # 10 Bearish Traps
    for i in range(10):
        idx = 670 + i * 15
        df = inject_bearish_trap(df, idx)
        test_indices.append(("Bearish Trap (Fakeout)", idx, 0)) # expected signal 0
        
    # 5 Clean Bullish Breakouts
    for i in range(5):
        idx = 820 + i * 15
        df = inject_clean_bullish_breakout(df, idx)
        test_indices.append(("Clean Bullish Breakout", idx, 1)) # expected signal 1

    test_indices.sort(key=lambda x: x[1])
    
    part1 = Part1Breakout()
    part2 = Part2Zone()
    part5 = Part5ML()
    
    scorecard = {"Part1_Correct": 0, "Part2_Correct": 0, "Part5_Correct": 0, "Laya_Correct": 0, "Total": len(test_indices)}
    
    for name, idx, expected_signal in test_indices:
        print(f"\n{'='*60}")
        print(f" TESTING: {name} at Index {idx} ")
        print(f"{'='*60}")
        
        # STRICT NO-CHEATING: Model ONLY sees up to `idx`
        # Using 500 candles to give plenty of data to XGBoost
        window_df = df.iloc[idx-500:idx+1].copy()
        window_df.attrs = df.attrs.copy()
        
        # Create context
        context = {
            'timeframe': '5m',
            'symbol': 'BTCUSDT',
            'real_funding': {'lastFundingRate': '0.0001'},
            'real_orderbook': {
                "bids": [[window_df.iloc[-1]['close'] * 0.99, "50.0"]],
                "asks": [[window_df.iloc[-1]['close'] * 1.01, "50.0"]]
            }
        }
        
        print("\n--- PART 1 (SmartBreakoutAI) ---")
        res_part1 = part1.analyze(window_df.copy(), context=context)
        p1_sig = res_part1.get('signal', 0)
        print(f"Signal: {p1_sig}")
        print(f"Thought: {res_part1.get('thought')}")
        
        if p1_sig == expected_signal or (expected_signal == 0 and p1_sig == 0):
            scorecard["Part1_Correct"] += 1
        
        print("\n--- PART 5 (XGBoost ML) ---")
        context['part_results'] = {'part1_breakout': res_part1}
        res_part5 = part5.analyze(window_df.copy(), context=context)
        p5_sig = res_part5.get('signal', 0)
        print(f"Signal: {p5_sig}")
        print(f"Thought: {res_part5.get('thought')}")
        
        if p5_sig == expected_signal or (expected_signal == 0 and p5_sig == 0):
            scorecard["Part5_Correct"] += 1
            
        print("\n--- PART 2 (Zone & Analysis) ---")
        # Ensure we pass active_timeframe='1m' and context['mtf_datasets'] so Part 2 runs the live engine
        context['timeframe'] = '1m'
        context['mtf_datasets'] = {'1m': window_df.copy()}
        res_part2 = part2.analyze(window_df.copy(), context=context)
        p2_sig = res_part2.get('signal', 0)
        print(f"Signal: {p2_sig}")
        print(f"Thought: {res_part2.get('thought')}")
        
        if p2_sig == expected_signal or (expected_signal == 0 and p2_sig == 0):
            scorecard["Part2_Correct"] += 1
        
        print("\n--- LAYA (Advisory Engine) ---")
        decision_packet = {
            'trade_signal': {
                'direction': 'BUY' if p1_sig > 0 else 'SELL' if p1_sig < 0 else 'NO_TRADE',
                'confidence_score': 85
            },
            'system_state': {'regime': res_part1.get('regime', {}).get('regime_type', 'UNKNOWN')}
        }
        
        snapshot = {
            'symbol': 'BTCUSDT',
            'decision': decision_packet,
            'parts': {'part1': res_part1, 'part5': res_part5},
            'context': context
        }
        
        try:
            advisory = advise(
                snapshot, 
                symbol="BTCUSDT", 
                deterministic_decision=decision_packet['trade_signal']['direction'],
                enabled=True
            )
            laya_sug = advisory.suggestion
            print(f"Laya Suggestion: {laya_sug}")
            print(f"Laya Confidence: {advisory.confidence}")
            
            laya_is_correct = False
            if expected_signal == 0 and laya_sug == 'NO_TRADE':
                laya_is_correct = True
            elif expected_signal == 1 and laya_sug == 'BUY':
                laya_is_correct = True
            elif expected_signal == -1 and laya_sug == 'SELL':
                laya_is_correct = True
                
            if laya_is_correct:
                scorecard["Laya_Correct"] += 1
                
        except Exception as e:
            print(f"Laya API Error: {e}")

    print("\n" + "="*60)
    print(" MASSIVE FAKEOUT TEST SCORECARD ")
    print("="*60)
    print(f"Total Scenarios Tested: {scorecard['Total']} (20 Fakeouts, 5 True Breakouts)")
    print(f"Part 1 Correct Detections: {scorecard['Part1_Correct']}/{scorecard['Total']}")
    print(f"Part 2 Correct Detections: {scorecard['Part2_Correct']}/{scorecard['Total']}")
    print(f"Part 5 Correct Detections: {scorecard['Part5_Correct']}/{scorecard['Total']}")
    print(f"Laya Correct Advisory:     {scorecard['Laya_Correct']}/{scorecard['Total']}")
    print("="*60)

if __name__ == "__main__":
    run_rolling_test()
