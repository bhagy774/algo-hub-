# BUG: line 476-477: p1_dir is set based on meta_score but also has a bug
# p1_dir = 1 if meta_score > 0.1 and t_res['trend_direction'] == 1 else -1 if meta_score > 0.1 else 0
# If meta_score > 0.1 but trend_direction != 1, p1_dir = -1 (SELL) even in sideways or uptrend
# Should be: -1 only if trend_direction == -1
print('BUG in seed p1_dir calculation (line 477):')
print('  -1 if meta_score > 0.1 else 0 => missing downtrend check')
print('  Should be: -1 if meta_score > 0.1 and t_res["trend_direction"] == -1 else 0')
print()
# Also check: what happens to P3_Institutional brain_tf_signals entry?
# Currently P3 is used ONLY as seed for Parts 4-12 bridge, and as super-vote in _master_evaluate
# P3 does NOT get its own brain_tf_signals entry (no add_tf_sig("P3_Institutional", ...))
# This means the 29 brains count excludes P3 (only if extended parts disabled)
# With extended parts, P3 gets its super-vote of +3 in _master_evaluate
# This is by design but we should ADD P3 to brain_tf_signals for transparency
print('DESIGN NOTE: P3_Institutional has NO brain_tf_signals entry')
print('  It contributes via super-vote (+3) in _master_evaluate separately')
print('  Consider adding add_tf_sig for P3 for better visibility')
