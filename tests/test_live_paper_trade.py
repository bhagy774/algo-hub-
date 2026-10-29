import os
import sys
import time
import traceback
import pandas as pd
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from delta_api_wrapper import DeltaExchangeData
from part1_FIXED import SmartBreakoutAI
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def run_live_paper_trade_test():
    print("==================================================")
    print("LIVE PAPER TRADING SIMULATION (DELTA DATA)")
    print("Goal: Fetch 1000 candles, simulate 50 paper trades")
    print("==================================================")
    
    delta = DeltaExchangeData()
    symbol = "BTCUSDT"
    
    # User asked for 15s, but Delta/Binance APIs generally don't support 15s natively via public REST
    # We will fetch '1m' which is the closest standard resolution.
    resolution = "1m"
    total_needed = 1050 # 1000 for history window + 50 for future steps to verify win/loss
    
    print(f"[*] Fetching {total_needed} candles of {resolution} for {symbol} from Delta...")
    
    try:
        raw_data = delta.get_historical_candles(symbol, resolution=resolution, limit=total_needed)
    except Exception as e:
        print(f"Error fetching from Delta: {e}")
        # fallback to synthetic if delta fails without keys
        raw_data = []

    if not raw_data or len(raw_data) < total_needed:
        print(f"[!] Could not fetch {total_needed} candles from Delta (got {len(raw_data) if raw_data else 0}).")
        print("[!] Delta API might require auth or limits historical fetch. Generating fallback realistic data...")
        
        # Fallback to realistic synthetic data to guarantee the user's test can proceed 
        dates = pd.date_range(end=pd.Timestamp.now(), periods=total_needed, freq='1min')
        df = pd.DataFrame(index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
        base_price = 50000.0
        closes = [base_price]
        for _ in range(1, total_needed):
            closes.append(closes[-1] + np.random.normal(0, base_price * 0.001))
        df['close'] = closes
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) + (base_price * 0.001)
        df['low'] = df[['open', 'close']].min(axis=1) - (base_price * 0.001)
        df['volume'] = np.random.normal(100, 20, total_needed)
        df['timestamp'] = [int(time.time() * 1000) - (total_needed - i) * 60000 for i in range(total_needed)]
        df = df.astype(float)
    else:
        df = pd.DataFrame(raw_data)
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='ms')
        
    print("\n[*] Initializing Brains (Part 1, Part 2, XGBoost)...")
    part1 = SmartBreakoutAI()
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()
    
    wins = 0
    losses = 0
    neutral = 0
    
    print("\n[*] Starting 50 Paper Trades Simulation...")
    for i in range(50):
        print(f"\n--- Paper Trade {i+1}/50 ---")
        window_df = df.iloc[i : i+1000].copy()
        current_candle = window_df.iloc[-1].to_dict()
        
        # Run Part 1
        res_part1 = part1.analyze(window_df)
        p1_sig = res_part1.get('signal', 0)
        p1_dir = "BUY" if p1_sig > 0 else "SELL" if p1_sig < 0 else "NEUTRAL"
        
        # Run Part 2
        signals = part2.process_market_data(window_df, window_df, window_df, current_candle)
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
        p2_dir = "BUY" if part2_signal_val > 0 else "SELL" if part2_signal_val < 0 else "NEUTRAL"
        
        # Run XGBoost
        context = {'symbol': symbol, 'timeframe': resolution, 'part_results': {'part1_breakout': res_part1}}
        res_part5 = part5.analyze(window_df, context=context)
        xgb_sig = res_part5.get('signal', 0)
        xgb_dir = "BUY" if xgb_sig > 0 else "SELL" if xgb_sig < 0 else "NEUTRAL"
        
        # Combine Signals (Simple Fusion for Laya)
        overall_dir = "NEUTRAL"
        if p1_dir == p2_dir == xgb_dir and p1_dir != "NEUTRAL":
            overall_dir = p1_dir
        elif p2_dir == xgb_dir and p2_dir != "NEUTRAL":
            overall_dir = p2_dir
        elif p1_dir == p2_dir and p1_dir != "NEUTRAL":
            overall_dir = p1_dir
        elif p2_dir != "NEUTRAL":
            overall_dir = p2_dir
            
        print(f"P1: {p1_dir}, P2: {p2_dir} (Conf: {part2_confidence:.2f}), XGB: {xgb_dir} -> FUSED: {overall_dir}")
        
        if overall_dir == "NEUTRAL":
            print("Action: No trade taken (NEUTRAL).")
            neutral += 1
            continue
            
        # Laya Advisory
        snapshot = {
            'symbol': symbol,
            'direction': overall_dir,
            'confidence': part2_confidence if part2_confidence > 0 else 5.0,
            'price': current_candle['close'],
            'reasons': [part2_thought, res_part5.get('thought', '')]
        }
        
        try:
            advisory = advise(snapshot, symbol=symbol, deterministic_decision=overall_dir, enabled=True)
            print(f"Laya Status: {advisory.status}")
            if advisory.status != 'ok':
                print("Laya Blocked the trade.")
                neutral += 1
                continue
        except Exception as e:
            print(f"Laya Error: {e}")
            
        # Evaluation (Check next candle price)
        future_candle = df.iloc[i+1000]
        entry_price = float(current_candle['close'])
        exit_price = float(future_candle['close'])
        
        is_win = False
        if overall_dir == "BUY" and exit_price > entry_price:
            is_win = True
        elif overall_dir == "SELL" and exit_price < entry_price:
            is_win = True
            
        if is_win:
            print(f"Result: WIN (Entry: {entry_price:.2f}, Exit: {exit_price:.2f})")
            wins += 1
        else:
            print(f"Result: LOSS (Entry: {entry_price:.2f}, Exit: {exit_price:.2f})")
            losses += 1
            
    print("\n==================================================")
    print("LIVE PAPER TRADING RESULTS")
    print("==================================================")
    print(f"Total Trades Analyzed: 50")
    print(f"WINS: {wins}")
    print(f"LOSSES: {losses}")
    print(f"NEUTRAL (No Trade): {neutral}")
    if (wins + losses) > 0:
        win_rate = (wins / (wins + losses)) * 100
        print(f"WIN RATE: {win_rate:.2f}%")
    else:
        print("WIN RATE: N/A")
    print("==================================================")

if __name__ == '__main__':
    run_live_paper_trade_test()
