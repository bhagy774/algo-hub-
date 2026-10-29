"""8-Timeframe Omni Test for ModeEngine master_evaluate."""
from test_dual_mode_backtest import fetch
from mode_engine import ModeEngine

def test_omni_timeframes():
    print("[+] Fetching 8 Timeframes from Binance API (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d)...")
    tf_dict = {
        "1min": fetch("1m", 150),
        "3min": fetch("3m", 150),
        "5min": fetch("5m", 150),
        "15min": fetch("15m", 150),
        "30min": fetch("30m", 150),
        "1h": fetch("1h", 200),
        "4h": fetch("4h", 200),
        "1d": fetch("1d", 100),
    }

    engine = ModeEngine()
    result = engine.master_evaluate(tf_dict["5min"], tf_dict["1h"], tf_dict=tf_dict)

    print("\n================ OMNI 8-TIMEFRAME ENGINE RESULT ================")
    for k, v in result.items():
        if k != "all_brain_votes":
            print(f"  {k:22}: {v}")
    print("\n  Sample Brain Votes across 8 TFs:")
    for b_name in list(result["all_brain_votes"].keys())[:10]:
        print(f"    - {b_name:25}: {result['all_brain_votes'][b_name]}")
    print("=================================================================")

if __name__ == "__main__":
    test_omni_timeframes()
