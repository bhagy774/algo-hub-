import json
import pandas as pd
import numpy as np
from binance_data import get_binance_data, get_ohlcv_binance
from jarvis_FIXED import Part1Breakout, Part5ML
from jarvis_laya_advisor import advise

def generate_scenario(df, scenario_type):
    df_mock = df.copy()
    base_price = df_mock.iloc[-4]['close']
    recent_high = df_mock['high'].max()
    recent_low = df_mock['low'].min()
    
    context = {
        'timeframe': '5m',
        'symbol': 'BTCUSDT',
        'real_funding': {'lastFundingRate': '0.0001'}
    }
    
    if scenario_type == "BULLISH_BREAKOUT":
        # Massive Bullish Breakout
        df_mock.iloc[-3, df_mock.columns.get_loc('close')] = base_price * 1.01
        
        # Penultimate candle stays below resistance
        df_mock.iloc[-2, df_mock.columns.get_loc('open')] = base_price * 1.01
        df_mock.iloc[-2, df_mock.columns.get_loc('close')] = base_price * 0.98
        
        # Breakout candle!
        df_mock.iloc[-1, df_mock.columns.get_loc('open')] = base_price * 0.98
        df_mock.iloc[-1, df_mock.columns.get_loc('close')] = recent_high * 1.05
        df_mock.iloc[-1, df_mock.columns.get_loc('high')] = recent_high * 1.06
        df_mock.iloc[-1, df_mock.columns.get_loc('low')] = base_price * 0.97
        df_mock.iloc[-1, df_mock.columns.get_loc('volume')] = df_mock['volume'].mean() * 10.0
        
        # Massive BUY pressure in orderbook
        context['real_orderbook'] = {
            "bids": [[recent_high * 1.02, "50.0"], [recent_high * 1.01, "30.0"]],
            "asks": [[recent_high * 1.06, "1.0"]]
        }
        
    elif scenario_type == "BEARISH_BREAKDOWN":
        # Massive Bearish Breakdown
        df_mock.iloc[-3, df_mock.columns.get_loc('close')] = base_price * 0.99
        
        # Penultimate candle stays above support
        df_mock.iloc[-2, df_mock.columns.get_loc('open')] = base_price * 0.99
        df_mock.iloc[-2, df_mock.columns.get_loc('close')] = base_price * 1.02
        
        # Breakdown candle!
        df_mock.iloc[-1, df_mock.columns.get_loc('open')] = base_price * 1.02
        df_mock.iloc[-1, df_mock.columns.get_loc('close')] = recent_low * 0.95
        df_mock.iloc[-1, df_mock.columns.get_loc('high')] = base_price * 1.03
        df_mock.iloc[-1, df_mock.columns.get_loc('low')] = recent_low * 0.94
        df_mock.iloc[-1, df_mock.columns.get_loc('volume')] = df_mock['volume'].mean() * 10.0
        
        # Massive SELL pressure in orderbook
        context['real_orderbook'] = {
            "bids": [[recent_low * 0.94, "1.0"]],
            "asks": [[recent_low * 0.98, "50.0"], [recent_low * 0.99, "30.0"]]
        }
        
    elif scenario_type == "FAKEOUT":
        # Bullish Fakeout (Pierces resistance then dumps)
        df_mock.iloc[-3, df_mock.columns.get_loc('close')] = base_price * 1.01
        
        df_mock.iloc[-2, df_mock.columns.get_loc('open')] = base_price * 1.01
        df_mock.iloc[-2, df_mock.columns.get_loc('close')] = recent_high * 1.02 # Fakeout breakout
        df_mock.iloc[-2, df_mock.columns.get_loc('high')] = recent_high * 1.03
        
        # Dump candle!
        df_mock.iloc[-1, df_mock.columns.get_loc('open')] = recent_high * 1.02
        df_mock.iloc[-1, df_mock.columns.get_loc('close')] = base_price * 0.95 # Reverses below base
        df_mock.iloc[-1, df_mock.columns.get_loc('high')] = recent_high * 1.025
        df_mock.iloc[-1, df_mock.columns.get_loc('low')] = base_price * 0.94
        df_mock.iloc[-1, df_mock.columns.get_loc('volume')] = df_mock['volume'].mean() * 8.0
        
        # Fakeout orderbook (Trapped buyers, heavy asks)
        context['real_orderbook'] = {
            "bids": [[base_price * 0.95, "2.0"]],
            "asks": [[recent_high * 1.01, "60.0"], [recent_high * 1.02, "40.0"]]
        }
        
    return df_mock, context


def run_scenario(scenario_name, df_base):
    print(f"\n{'='*50}")
    print(f" TESTING SCENARIO: {scenario_name} ")
    print(f"{'='*50}")
    
    df, context = generate_scenario(df_base, scenario_name)
    
    # 1. PART 1
    print("\n--- Running PART 1 (SmartBreakoutAI) ---")
    part1 = Part1Breakout()
    res_part1 = part1.analyze(df, context=context)
    
    print(f"Part 1 Signal: {res_part1.get('signal')}")
    print(f"Part 1 Thought: {res_part1.get('thought')}")
    
    brain_support = res_part1.get('brain_support', {})
    mini_v3 = brain_support.get('mini_v3', {})
    if mini_v3:
        print(f"Real Tick Momentum (from Orderbook): {mini_v3.get('tick_momentum'):.4f}")
        
    # 2. XGBOOST
    print("\n--- Running PART 5 (XGBoost ML) ---")
    context['part_results'] = {'part1_breakout': res_part1}
    part5 = Part5ML()
    res_part5 = part5.analyze(df, context=context)
    print(f"XGBoost Signal: {res_part5.get('signal')}")
    print(f"XGBoost Thought: {res_part5.get('thought')}")
    
    # 3. LAYA
    print("\n--- Running LAYA (Advisory Engine) ---")
    direction = 'NO_TRADE'
    if res_part1.get('signal', 0) > 0: direction = 'BUY'
    elif res_part1.get('signal', 0) < 0: direction = 'SELL'
    
    decision_packet = {
        'trade_signal': {
            'direction': direction,
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
    
    try:
        advisory = advise(
            snapshot, 
            symbol="BTCUSDT", 
            deterministic_decision=direction,
            enabled=True
        )
        print(f"Laya Status: {advisory.status}")
        print(f"Laya Suggestion: {advisory.suggestion}")
        print(f"Laya Confidence: {advisory.confidence}")
        print(f"Laya Commentary:\n{getattr(advisory, 'rationale', 'N/A')}")
    except Exception as e:
        print(f"Laya encountered an error: {e}")


def main():
    print("Fetching base LIVE 5m candles for BTCUSDT...")
    df_base = get_ohlcv_binance(symbol="BTCUSDT", resolution="5m", limit=60)
    df_base.attrs['timeframe'] = '5m'
    df_base.attrs['symbol'] = 'BTCUSDT'
    
    scenarios = ["BULLISH_BREAKOUT", "BEARISH_BREAKDOWN", "FAKEOUT"]
    for s in scenarios:
        run_scenario(s, df_base)


if __name__ == "__main__":
    main()
