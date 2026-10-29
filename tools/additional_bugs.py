# Now check: does DeepSeek signal logic make sense?
# add_tf_sig("P1_DeepSeek", "BUY" if ds_res.get("fakeout_block", 0) < 0.3 and t_res["trend_direction"] == 1
#            else "SELL" if ds_res.get("fakeout_block", 0) < 0.3 else "NEUTRAL", 0.7)
# fakeout_block is 0..1 where LOW means LOW fakeout risk
# If fakeout_block < 0.3 (low risk of fakeout) AND uptrend -> BUY = OK
# If fakeout_block < 0.3 AND NOT uptrend -> SELL (but wrong! should check downtrend)
# Same bug pattern as P1_Risk and P1_Reversal - if fakeout_block < 0.3 but no downtrend -> SELL is wrong
print('BUG also in P1_DeepSeek: SELL when fakeout_block < 0.3 regardless of downtrend')
print('FIX: Check t_res["trend_direction"] == -1 for SELL')
print()
# EvolutionBrain: same pattern - SELL if support_score > 0.5 but no direction check
print('BUG also in P1_Evolution: SELL when support_score > 0.5 but no downtrend check')
print('FIX: Check t_res["trend_direction"] == -1 for SELL')
