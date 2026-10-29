import json
import pandas as pd
import numpy as np
from binance_data import get_binance_data, get_ohlcv_binance
from jarvis_FIXED import Part1Breakout, Part5ML
from jarvis_laya_advisor import advise

def run_test():
    print("Fetching LIVE 5m candles for BTCUSDT...")
    df = get_ohlcv_binance(symbol="BTCUSDT", resolution="5m", limit=60)
    
    # ── MOCK A MASSIVE BREAKOUT ──
    print("Injecting a MASSIVE BULLISH BREAKOUT into the recent data...")
    recent_high = df['high'].max()
    base_price = df.iloc[-4]['close']
    
    # Last 3 candles surge past the recent high
    df.iloc[-3, df.columns.get_loc('open')] = base_price
    df.iloc[-3, df.columns.get_loc('close')] = base_price * 1.01
    df.iloc[-3, df.columns.get_loc('high')] = base_price * 1.011
    df.iloc[-3, df.columns.get_loc('low')] = base_price * 0.999
    
    # Ensure prev_close is LOW so the breakout happens EXACTLY on the last candle
    df.iloc[-2, df.columns.get_loc('open')] = base_price * 1.01
    df.iloc[-2, df.columns.get_loc('close')] = base_price * 0.98
    df.iloc[-2, df.columns.get_loc('high')] = base_price * 1.015
    df.iloc[-2, df.columns.get_loc('low')] = base_price * 0.97
    
    # The massive breakout candle!
    df.iloc[-1, df.columns.get_loc('open')] = base_price * 0.98
    df.iloc[-1, df.columns.get_loc('close')] = recent_high * 1.05  # 5% above previous absolute high
    df.iloc[-1, df.columns.get_loc('high')] = recent_high * 1.06
    df.iloc[-1, df.columns.get_loc('low')] = base_price * 0.975
    df.iloc[-1, df.columns.get_loc('volume')] = df['volume'].mean() * 10.0 # Massive volume
    
    df.attrs['timeframe'] = '5m'
    df.attrs['symbol'] = 'BTCUSDT'
    
    print("Mocking massive BUY pressure in Orderbook...")
    # 90% Bids, 10% Asks
    mock_orderbook = {
        "bids": [[recent_high * 1.02, "10.0"], [recent_high * 1.01, "15.0"], [recent_high, "50.0"]],
        "asks": [[recent_high * 1.031, "1.0"], [recent_high * 1.035, "2.0"]]
    }
    
    context = {
        'real_orderbook': mock_orderbook,
        'real_funding': {'lastFundingRate': '0.0001'},
        'timeframe': '5m',
        'symbol': 'BTCUSDT'
    }
    
    print("\n--- Running PART 1 (SmartBreakoutAI) on BREAKOUT DATA ---")
    part1 = Part1Breakout()
    res_part1 = part1.analyze(df, context=context)
    print("Part 1 Signal:", res_part1.get('signal'))
    print("Part 1 Thought:", res_part1.get('thought'))
    
    brain_support = res_part1.get('brain_support', {})
    mini_v3 = brain_support.get('mini_v3', {})
    if mini_v3:
        print(f"Mini V3 Brain (Real Tick Momentum): {mini_v3.get('tick_momentum'):.4f}")
        
    print("\n--- Running PART 5 (XGBoost ML) ---")
    context['part_results'] = {'part1_breakout': res_part1}
    part5 = Part5ML()
    res_part5 = part5.analyze(df, context=context)
    print("XGBoost Signal:", res_part5.get('signal'))
    print("XGBoost Thought:", res_part5.get('thought'))
    
    print("\n--- Running LAYA (Advisory Engine) ---")
    decision_packet = {
        'trade_signal': {
            'direction': 'BUY' if res_part1.get('signal', 0) > 0 else 'SELL' if res_part1.get('signal', 0) < 0 else 'NO_TRADE',
            'confidence_score': 85
        },
        'system_state': {'regime': res_part1.get('regime', {}).get('regime_type', 'UNKNOWN')}
    }
    part_snapshots = {
        'part1': res_part1,
        'part5': res_part5
    }
    
    snapshot = {
        'symbol': 'BTCUSDT',
        'decision': decision_packet,
        'parts': part_snapshots,
        'context': context
    }
    
    # Call advise synchronously
    try:
        advisory = advise(
            snapshot, 
            symbol="BTCUSDT", 
            deterministic_decision=decision_packet['trade_signal']['direction'],
            enabled=True
        )
        print("Laya Status:", advisory.status)
        print("Laya Suggestion:", advisory.suggestion)
        print("Laya Confidence:", advisory.confidence)
        print("Laya Commentary:\n" + getattr(advisory, 'rationale', 'N/A'))
    except Exception as e:
        print(f"Laya encountered an error: {e}")

if __name__ == "__main__":
    run_test()
