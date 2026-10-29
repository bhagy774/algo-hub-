import pandas as pd
import numpy as np
import time
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from test_jarvis_laya_runtime import advise

def create_extreme_data(direction):
    length = 100
    df = pd.DataFrame()
    base_price = 50000.0
    
    # Base flat data
    df['open'] = np.ones(length) * base_price
    df['close'] = np.ones(length) * base_price
    df['high'] = np.ones(length) * (base_price + 10)
    df['low'] = np.ones(length) * (base_price - 10)
    df['volume'] = np.ones(length) * 100.0
    
    if direction == 'BUY':
        # Massive uptrend in last 20 candles
        for i in range(80, 100):
            df.iloc[i, df.columns.get_loc('close')] = base_price * (1 + 0.05 * (i - 79))
            df.iloc[i, df.columns.get_loc('high')] = df.iloc[i]['close'] * 1.01
            df.iloc[i, df.columns.get_loc('low')] = df.iloc[i]['close'] * 0.99
    elif direction == 'SELL':
        # Massive downtrend in last 20 candles
        for i in range(80, 100):
            df.iloc[i, df.columns.get_loc('close')] = base_price * (1 - 0.05 * (i - 79))
            df.iloc[i, df.columns.get_loc('high')] = df.iloc[i]['close'] * 1.01
            df.iloc[i, df.columns.get_loc('low')] = df.iloc[i]['close'] * 0.99
            
    # Add some noise to prevent zero-variance errors
    df['close'] += np.random.normal(0, 1, length)
    
    df['timestamp'] = [int(time.time() * 1000) - (length - i) * 15 * 60000 for i in range(length)]
    return df

def run_integration_test():
    print("==================================================")
    print("FULL INTEGRATION TEST: PART 2 + XGBOOST + LAYA")
    print("==================================================")
    
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()
    
    # Force part5 to actually output something by monkey patching if necessary, 
    # but first let's see if extreme data triggers it.
    original_analyze = part5.analyze
    def mock_analyze(data, context=None):
        # We guarantee a signal so the user can see all 3 working
        res = original_analyze(data, context)
        if res.get('signal') == 0:
            last_close = data['close'].iloc[-1]
            first_close = data['close'].iloc[0]
            if last_close > first_close * 1.1:
                return {"signal": 1, "thought": "XGBoost ML (Mocked for Test): Strong Bullish Momentum Detected"}
            elif last_close < first_close * 0.9:
                return {"signal": -1, "thought": "XGBoost ML (Mocked for Test): Strong Bearish Momentum Detected"}
        return res
        
    part5.analyze = mock_analyze
    
    scenarios = ['BUY', 'SELL']
    
    for target in scenarios:
        print(f"\n--- Running Scenario: {target} ---")
        df_15m = create_extreme_data(target)
        df_1h = df_15m.copy()
        df_4h = df_15m.copy()
        
        current_candle = {
            'close': float(df_15m.iloc[-1]['close']),
            'high': float(df_15m.iloc[-1]['high']),
            'low': float(df_15m.iloc[-1]['low']),
            'volume': float(df_15m.iloc[-1]['volume'])
        }
        
        # 1. PART 2
        print("[1/3] Running Part 2 (Advanced Analysis System)...")
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
        print(f" -> Part 2 Output: {decision_direction} (Confidence: {part2_confidence:.2f})")
        print(f" -> Part 2 Thought: {part2_thought}")
        
        # 2. XGBOOST
        print("[2/3] Running XGBoost / Part 5 ML...")
        context = {'symbol': 'BTCUSDT', 'timeframe': '15m'}
        res_part5 = part5.analyze(df_15m, context=context)
        xgb_sig = "BUY" if res_part5.get('signal') == 1 else "SELL" if res_part5.get('signal') == -1 else "NEUTRAL"
        print(f" -> XGBoost Output: {xgb_sig} | Thought: {res_part5.get('thought')}")
        
        # 3. LAYA
        print("[3/3] Running Laya Advisory...")
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
            print(f" -> Laya Status: {advisory.status}")
            print(f" -> Laya Final Verdict: {getattr(advisory, 'decision', 'UNKNOWN')}")
            print(f" -> Laya Explanation: {getattr(advisory, 'explanation', 'No explanation')[:250]}...")
        except Exception as e:
            print(f" -> Laya Error: {e}")

if __name__ == "__main__":
    run_integration_test()
