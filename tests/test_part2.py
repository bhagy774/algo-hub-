import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "c:\\jarvis")
from jarvis_FIXED import Part2Zone

def run_test():
    n_candles = 100
    base_price = 50000.0

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_candles, freq='15min'),
        'open': np.random.normal(base_price, 100, n_candles),
        'high': np.random.normal(base_price + 200, 100, n_candles),
        'low': np.random.normal(base_price - 200, 100, n_candles),
        'close': np.random.normal(base_price, 100, n_candles),
        'volume': np.random.normal(1000, 200, n_candles)
    })
    
    # Simulate a "fakeout trap" that the guard will catch
    # for i in range(-5, 0):
    #     df.loc[df.index[i], 'open'] = base_price
    #     df.loc[df.index[i], 'close'] = base_price + 300 # 0.6% pump
    # df.loc[df.index[-1], 'close'] = base_price - 300 # 0.6% dump
    # df.loc[df.index[-1], 'volume'] = 2000 # High volume
    
    df.attrs['timeframe'] = '15m'

    context = {
        'timeframe': '15m',
        'mtf_datasets': {
            '15m': df.copy(),
            '1h': df.copy(),
            '4h': df.copy()
        }
    }

    zone = Part2Zone()
    res = zone.analyze(df, context)
    print("Result of Part2Zone analyze (with fakeout trap simulation):")
    print(res)

if __name__ == '__main__':
    run_test()
