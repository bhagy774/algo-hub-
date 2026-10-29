import pandas as pd
from jarvis_backtester import JarvisFullBacktester, BacktestConfig

print("Loading data...")
df = pd.read_csv("data/BTCUSDT_1m_20260716_20260914.csv", nrows=2000)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Initializing backtester...")
config = BacktestConfig(starting_capital=1000.0, fee_bps=10.0, slippage_bps=5.0)
backtester = JarvisFullBacktester(config=config)
print("Running backtest on 2000 candles...")
results = backtester.run_backtest(df, 'BTCUSDT', '1m')

print("\n--- RESULTS ---")
print(f"Total Trades: {results['total_trades']}")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Net PnL: ${results['net_pnl']:.2f}")
print("Audit complete.")
