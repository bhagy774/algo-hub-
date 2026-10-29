"""Direct Verification Script for Part 1 and Part 2 Native Integration."""
from test_dual_mode_backtest import fetch
from part1_FIXED import SmartBreakoutAI
from part2_FIXED import AdvancedAnalysisSystem

def test_parts_native():
    print("[+] Fetching candles for Part 1 and Part 2 native test...")
    m5 = fetch("5m", 150)
    h1 = fetch("1h", 200)
    tf_dict = {"5min": m5, "1h": h1}

    p1_sys = SmartBreakoutAI()
    p2_sys = AdvancedAnalysisSystem()

    res1 = p1_sys.analyze_master_consensus(m5, h1, tf_dict=tf_dict)
    res2 = p2_sys.process_master_consensus(m5, h1, tf_dict=tf_dict)

    print("\n--- Part 1 Native Master Consensus ---")
    print(f"  Action: {res1['action']} | Type: {res1['trade_type']} | Direction: {res1['direction']} | Voted: {res1['total_brains_voted']}")

    print("\n--- Part 2 Native Master Consensus ---")
    print(f"  Action: {res2['action']} | Type: {res2['trade_type']} | Direction: {res2['direction']} | Voted: {res2['total_brains_voted']}")

if __name__ == "__main__":
    test_parts_native()
