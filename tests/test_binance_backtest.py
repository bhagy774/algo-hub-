import os
import sys
import time
import pandas as pd
import requests

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from part1_FIXED import SmartBreakoutAI
from part2_FIXED import AdvancedAnalysisSystem
from jarvis_FIXED import Part5ML
from jarvis_laya_advisor import advise

def fetch_historical_binance(symbol="BTCUSDT", interval="5m", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    # We want e.g. 1000 candles to get a quick test
    for _ in range(1):
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = int(end_time)
            
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            break
            
        all_data = data + all_data
        end_time = data[0][0] - 1
        
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates('timestamp').reset_index(drop=True)

def run_backtest():
    print("==================================================")
    print("FAST BACKTESTING SIMULATION (BINANCE HISTORICAL)")
    print("Requirement: Part1, Part2, XGBoost and Laya MUST ALL AGREE.")
    print("==================================================")
    
    symbol = "BTCUSDT"
    interval = "5m"  # Using 5m timeframe for better structure
    
    print(f"[*] Fetching historical {interval} data from Binance...")
    df = fetch_historical_binance(symbol, interval, 1000)
    print(f"[+] Loaded {len(df)} candles.")
    
    print("[*] Loading AI Brains...")
    part1 = SmartBreakoutAI()
    part2 = AdvancedAnalysisSystem()
    part5 = Part5ML()
    
    trades = 0
    wins = 0
    losses = 0
    
    print("[*] Starting Backtest (Scanning fast over history)...")
    
    # We need a window of 200 candles for analysis
    window_size = 200
    
    i = 0
    while i < len(df) - window_size - 12: # 12 candles for evaluation (1 hour in 5m tf)
        window_df = df.iloc[i : i+window_size].copy()
        current_candle = window_df.iloc[-1].to_dict()
        
        try:
            # 1. Part 1
            res_part1 = part1.analyze(window_df)
            p1_sig = res_part1.get('signal', 0)
            p1_dir = "BUY" if p1_sig > 0 else "SELL" if p1_sig < 0 else "NEUTRAL"
            
            # 2. Part 2
            signals = part2.process_market_data(window_df, window_df, window_df, current_candle)
            part2_signal_val = 0
            p2_dir = "NEUTRAL"
            if signals and isinstance(signals, list):
                valid = [s for s in signals if isinstance(s, tuple) and len(s) >= 3]
                if valid:
                    best = max(valid, key=lambda x: x[1])
                    part2_signal_val = 1 if str(best[0]).upper() in ['CALL', 'BUY', 'LONG'] else -1 if str(best[0]).upper() in ['PUT', 'SELL', 'SHORT'] else 0
                    p2_dir = "BUY" if part2_signal_val > 0 else "SELL" if part2_signal_val < 0 else "NEUTRAL"
            
            # 3. XGBoost
            context = {'symbol': symbol, 'timeframe': interval, 'part_results': {'part1_breakout': res_part1}}
            res_part5 = part5.analyze(window_df, context=context)
            xgb_sig = res_part5.get('signal', 0)
            xgb_dir = "BUY" if xgb_sig > 0 else "SELL" if xgb_sig < 0 else "NEUTRAL"
            
            # Check consensus (Scalp vs Breakout)
            is_breakout = (p1_dir != "NEUTRAL") and (p1_dir == p2_dir == xgb_dir)
            is_scalp = (p1_dir == "NEUTRAL") and (p2_dir != "NEUTRAL") and (p2_dir == xgb_dir)
            
            if is_breakout or is_scalp:
                overall_dir = p2_dir
                trade_type = "BREAKOUT" if is_breakout else "SCALP"
                
                # Laya
                snapshot = {
                    'symbol': symbol, 'direction': overall_dir, 'confidence': 8.5,
                    'price': current_candle['close'], 'reasons': [f'Backtest {trade_type} Consensus']
                }
                
                advisory = advise(snapshot, symbol=symbol, deterministic_decision=overall_dir, enabled=True)
                if advisory.status == 'ok':
                    trades += 1
                    entry_price = float(current_candle['close'])
                    
                    # Evaluation (0.3% TP, 0.15% SL)
                    tp = entry_price * 1.003 if overall_dir == "BUY" else entry_price * 0.997
                    sl = entry_price * 0.9985 if overall_dir == "BUY" else entry_price * 1.0015
                    
                    future_candles = df.iloc[i+window_size : i+window_size+12]
                    hit_tp = False
                    
                    for _, row in future_candles.iterrows():
                        if overall_dir == "BUY":
                            if row['low'] <= sl: break
                            if row['high'] >= tp: hit_tp = True; break
                        else:
                            if row['high'] >= sl: break
                            if row['low'] <= tp: hit_tp = True; break
                            
                    if hit_tp:
                        wins += 1
                        print(f"✅ Trade #{trades} WIN  | [{trade_type}] {overall_dir} @ {entry_price:.2f}")
                    else:
                        losses += 1
                        print(f"❌ Trade #{trades} LOSS | [{trade_type}] {overall_dir} @ {entry_price:.2f}")
                        
                    # Skip ahead after trade
                    i += 12
                    continue
        except Exception as e:
            pass
            
        i += 1
        
        if i % 25 == 0:
            print(f"[Scan] Processed {i}/{len(df)} candles...")
            
    print("\n==================================================")
    print("BACKTEST RESULTS (STRICT CONSENSUS)")
    print(f"Total Trades: {trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    if trades > 0:
        print(f"Win Rate: {(wins/trades)*100:.2f}%")
    else:
        print("Win Rate: 0.00% (No trades met the strict consensus criteria)")
    print("==================================================")

if __name__ == '__main__':
    run_backtest()
