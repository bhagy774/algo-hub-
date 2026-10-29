import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
import requests

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from part1_FIXED import SmartBreakoutAI
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def fetch_binance_1s_candles(symbol="BTCUSDT", limit=1000, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1s",
        "limit": limit
    }
    if end_time:
        params["endTime"] = int(end_time)
        
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Clean up types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def run_live_1s_paper_trade():
    print("==================================================")
    print("LIVE PAPER TRADING SIMULATION (BINANCE 1-SECOND)")
    print("Goal: Let AI independently find and execute exactly 5 trades.")
    print("Requirement: Part1, Part2, XGBoost and Laya MUST ALL AGREE.")
    print("==================================================")
    
    symbol = "BTCUSDT"
    resolution = "1s"
    
    # We need a lot of history because we are waiting for a real setup, not forcing trades.
    # Let's fetch 3000 candles (50 minutes of 1s data).
    print(f"[*] Fetching 3000 candles of {resolution} for {symbol} from Binance...")
    
    try:
        df3 = fetch_binance_1s_candles(symbol=symbol, limit=1000)
        df2 = fetch_binance_1s_candles(symbol=symbol, limit=1000, end_time=df3['timestamp'].iloc[0] - 1)
        df1 = fetch_binance_1s_candles(symbol=symbol, limit=1000, end_time=df2['timestamp'].iloc[0] - 1)
        
        df = pd.concat([df1, df2, df3]).sort_values('timestamp').reset_index(drop=True)
        df.index = pd.to_datetime(df['timestamp'], unit='ms')
    except Exception as e:
        print(f"Error fetching 1s from Binance: {e}")
        return

    print(f"[+] Successfully fetched {len(df)} candles.")
    
    print("\n[*] Initializing Brains (Part 1, Part 2, XGBoost)...")
    part1 = SmartBreakoutAI()
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()
    
    trades_taken = 0
    wins = 0
    losses = 0
    target_trades = 5
    
    print("\n[*] Scanning market... AI will only trade when they find an opportunity and ALL AGREE.")
    
    i = 0
    # Stop when we don't have enough future candles (1000 for history + 60 for evaluation)
    while i < (len(df) - 1060):
        window_df = df.iloc[i : i+1000].copy()
        current_candle = window_df.iloc[-1].to_dict()
        
        # 1. Part 1 
        res_part1 = part1.analyze(window_df)
        p1_sig = res_part1.get('signal', 0)
        p1_dir = "BUY" if p1_sig > 0 else "SELL" if p1_sig < 0 else "NEUTRAL"
        
        # 2. Part 2
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
        
        # 3. XGBoost
        context = {'symbol': symbol, 'timeframe': resolution, 'part_results': {'part1_breakout': res_part1}}
        res_part5 = part5.analyze(window_df, context=context)
        xgb_sig = res_part5.get('signal', 0)
        xgb_dir = "BUY" if xgb_sig > 0 else "SELL" if xgb_sig < 0 else "NEUTRAL"
        
        # Check if ALL agree
        all_agree = (p1_dir != "NEUTRAL") and (p1_dir == p2_dir == xgb_dir)
        
        if not all_agree:
            # Move 1 second forward and keep scanning
            i += 1
            # Optional: Print progress every 60 seconds so user knows it's alive
            if i % 60 == 0:
                print(f"[Scan] Checked {i} candles... No consensus entry found yet.")
            continue
            
        print(f"\n--- ⚡ OPPORTUNITY DETECTED! ---")
        print(f"P1: {p1_dir}, P2: {p2_dir}, XGB: {xgb_dir}")
        print(f"All Brains Agreed on {p1_dir}!")
        
        overall_dir = p1_dir
            
        # 4. Laya Advisory Audit
        snapshot = {
            'symbol': symbol,
            'direction': overall_dir,
            'confidence': part2_confidence if part2_confidence > 0 else 5.0,
            'price': current_candle['close'],
            'reasons': [part2_thought, res_part5.get('thought', ''), 'Trade Mode: SCALP']
        }
        
        advisory_ok = False
        try:
            advisory = advise(snapshot, symbol=symbol, deterministic_decision=overall_dir, enabled=True)
            print(f"Laya Status: {advisory.status}")
            if advisory.status == 'ok':
                advisory_ok = True
            else:
                print("Laya Blocked the trade. Continuing scan...")
        except Exception as e:
            print(f"Laya Error: {e}. Continuing scan...")
            
        if not advisory_ok:
            i += 1
            continue
            
        # ALL APPROVED. EXECUTE TRADE.
        trades_taken += 1
        print(f"\n✅ TRADE #{trades_taken} EXECUTED: {overall_dir} at {current_candle['close']}")
        
        # Evaluation (Check next 60 1s candles for TP/SL hit)
        entry_price = float(current_candle['close'])
        tp_margin = entry_price * 0.0005  # 0.05% TP
        sl_margin = entry_price * 0.0003  # 0.03% SL
        
        tp_price = entry_price + tp_margin if overall_dir == "BUY" else entry_price - tp_margin
        sl_price = entry_price - sl_margin if overall_dir == "BUY" else entry_price + sl_margin
        
        future_candles = df.iloc[i+1000 : i+1060]
        hit_tp = False
        hit_sl = False
        exit_price = entry_price
        exit_offset = 60 # Default jump is 60 if it times out
        
        for idx, row in future_candles.iterrows():
            high_price = float(row['high'])
            low_price = float(row['low'])
            step = future_candles.index.get_loc(idx) + 1
            
            if overall_dir == "BUY":
                if low_price <= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                    exit_offset = step
                    break
                elif high_price >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_offset = step
                    break
            elif overall_dir == "SELL":
                if high_price >= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                    exit_offset = step
                    break
                elif low_price <= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_offset = step
                    break
                    
        if not hit_tp and not hit_sl and len(future_candles) > 0:
            exit_price = float(future_candles.iloc[-1]['close'])
            if overall_dir == "BUY" and exit_price > entry_price:
                hit_tp = True
            elif overall_dir == "SELL" and exit_price < entry_price:
                hit_tp = True
                
        if hit_tp:
            print(f"Result: WIN (Entry: {entry_price:.2f}, Exit: {exit_price:.2f}) [TP HIT]")
            wins += 1
        else:
            print(f"Result: LOSS (Entry: {entry_price:.2f}, Exit: {exit_price:.2f}) [SL HIT]")
            losses += 1
            
        if trades_taken >= target_trades:
            print("\n[+] Target 5 trades reached. Stopping simulation.")
            break
            
        # Jump ahead to the end of the evaluated trade so we don't open overlapping trades.
        i += exit_offset

    print("\n==================================================")
    print("LIVE PAPER TRADING RESULTS (AUTONOMOUS 5 TRADES)")
    print("==================================================")
    print(f"Total Trades Taken: {trades_taken}")
    print(f"WINS: {wins}")
    print(f"LOSSES: {losses}")
    if trades_taken > 0:
        win_rate = (wins / trades_taken) * 100
        print(f"WIN RATE: {win_rate:.2f}%")
    else:
        print("No consensus trades were found in the available timeframe.")
    print("==================================================")

if __name__ == '__main__':
    run_live_1s_paper_trade()
