import json
from part1_FIXED import SmartBreakoutAI
from binance_data import LiveDataManager

def test_part1():
    engine = SmartBreakoutAI()
    
    # Generate some fake breakout market data
    import pandas as pd
    import numpy as np
    
    # 20 flat candles then 1 big breakout candle
    closes = [100.0] * 20 + [105.0]
    highs = [100.1] * 20 + [105.5]
    lows = [99.9] * 20 + [100.0]
    opens = [100.0] * 20 + [100.5]
    volumes = [1000] * 20 + [50000]
    
    candles = []
    for i in range(21):
        candles.append({
            'open': opens[i],
            'high': highs[i],
            'low': lows[i],
            'close': closes[i],
            'volume': volumes[i],
            'is_closed': True
        })
        
    market_data = {
        'price_action': candles
    }
    
    res = engine.analyze(market_data)
    print("Part 1 Output:")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test_part1()
