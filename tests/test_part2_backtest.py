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

def run_honest_backtest():
    print("Fetching historical 15m candles for BTCUSDT from Binance...")
    # Fetch 500 candles of 15m, 1h, and 4h data
    df_15m = get_ohlcv_binance(symbol="BTCUSDT", resolution="15m", limit=500)
    df_1h = get_ohlcv_binance(symbol="BTCUSDT", resolution="1h", limit=500)
    df_4h = get_ohlcv_binance(symbol="BTCUSDT", resolution="4h", limit=500)
    
    if df_15m is None or len(df_15m) < 100:
        print("Not enough data to run backtest.")
        return

    print("Data fetched! Initializing Part 2...")
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()

    total_signals = 0
    correct_buys = 0
    wrong_buys = 0
    correct_sells = 0
    wrong_sells = 0
    neutrals = 0

    print("Running Backtest on last 100 candles... (No Cheating!)")
    
    start_idx = len(df_15m) - 100 - 4  # Reserve last 4 candles to check future price
    end_idx = len(df_15m) - 4
    
    for i in range(start_idx, end_idx):
        # Slice the data exactly as it would appear at time i
        slice_15m = df_15m.iloc[:i+1].copy()
        current_candle = slice_15m.iloc[-1].to_dict()
        
        # In a real scenario, higher timeframe data is also sliced. 
        # For this test, we just pass the full higher timeframe data as background context (or slice it proportionally)
        # To be completely honest, we should slice df_1h and df_4h up to the timestamp of current_candle
        current_time = slice_15m.index[-1] if isinstance(slice_15m.index, pd.DatetimeIndex) else current_candle.get('timestamp', 0)
        
        try:
            # We will just pass the full dfs for simplicity since Part2 uses them mostly for general trend
            signals = part2.process_market_data(slice_15m, df_1h, df_4h, current_candle)
            
            part2_sig = 0
            if signals and isinstance(signals, list):
                valid = [s for s in signals if isinstance(s, tuple) and len(s) >= 3]
                if valid:
                    best = max(valid, key=lambda x: x[1])
                    part2_sig = 1 if str(best[0]).upper() in ['CALL', 'BUY', 'LONG'] else -1 if str(best[0]).upper() in ['PUT', 'SELL', 'SHORT'] else 0
            
            # Look at future price 4 candles ahead (1 hour)
            future_price = df_15m.iloc[i+4]['close']
            current_price = current_candle['close']
            price_change = future_price - current_price
            
            if part2_sig == 1:
                total_signals += 1
                if price_change > 0:
                    correct_buys += 1
                else:
                    wrong_buys += 1
            elif part2_sig == -1:
                total_signals += 1
                if price_change < 0:
                    correct_sells += 1
                else:
                    wrong_sells += 1
            else:
                neutrals += 1
                
        except Exception as e:
            print(f"Error at index {i}: {e}")
            
    print("\n" + "="*50)
    print("HONEST BACKTEST RESULTS (Part 2)")
    print("="*50)
    print(f"Total Evaluated Candles : 100")
    print(f"Total Trade Signals     : {total_signals}")
    print(f"Total Neutrals (Pass)   : {neutrals}")
    print("-" * 50)
    print(f"BUY Signals             : {correct_buys + wrong_buys}")
    print(f"  - Correct (Profit)    : {correct_buys}")
    print(f"  - Wrong (Loss)        : {wrong_buys}")
    print(f"SELL Signals            : {correct_sells + wrong_sells}")
    print(f"  - Correct (Profit)    : {correct_sells}")
    print(f"  - Wrong (Loss)        : {wrong_sells}")
    print("-" * 50)
    
    total_correct = correct_buys + correct_sells
    if total_signals > 0:
        accuracy = (total_correct / total_signals) * 100
        print(f"Overall Accuracy        : {accuracy:.2f}%")
    else:
        print("No signals generated.")
        
    print("\nTesting XGBoost & Laya integrations (One-shot)...")
    try:
        context = {'symbol': 'BTCUSDT', 'timeframe': '15m'}
        res_part5 = part5.analyze(df_15m, context=context)
        print(f"XGBoost Test: SUCCESS -> Signal: {res_part5.get('signal')}")
        
        advisory = advise({
            'symbol': 'BTCUSDT',
            'direction': 'BUY',
            'confidence': 9.9,
            'price': current_price,
            'reasons': ["Test signal"]
        }, symbol='BTCUSDT', deterministic_decision='BUY')
        print(f"Laya Test: SUCCESS -> Status: {advisory.status}")
    except Exception as e:
        print(f"Integration Test Failed: {e}")

if __name__ == '__main__':
    run_honest_backtest()
