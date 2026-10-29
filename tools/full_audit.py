# Comprehensive audit of all bugs in mode_engine.py

print('=== FULL AUDIT OF mode_engine.py vs actual Part APIs ===')
print()

# BUG 1: P1_Risk uses risk_level but RiskBrain returns risk_score (float 0-1)
print('BUG 1 (CRITICAL): P1_Risk signal logic')
print('  mode_engine: rk_res.get("risk_level") == "LOW"')
print('  RiskBrain returns: risk_score (float 0..1), NOT "risk_level"')
print('  EFFECT: P1_Risk always returns NEUTRAL (risk_level is always None)')
print()

# BUG 2: P1_Reversal signal logic is inverted/wrong for non-uptrend case
print('BUG 2 (LOGIC): P1_Reversal BUY when no uptrend')
print('  mode_engine: if reversal_prob > 0.001 and t_res["trend_direction"] == 1 -> SELL')
print('               elif reversal_prob > 0.001 -> BUY  (WRONG - should check downtrend)')
print('  FIX: elif reversal_prob > 0.001 and t_res["trend_direction"] == -1 -> BUY')
print()

# BUG 3: MetaFusionBrain receives 12 brain results dict - OK
print('OK: MetaFusionBrain receives correct dict with brain results containing support_score')
print()

# BUG 4: Part 5 veto logic - signals check
print('BUG 4 (LOGIC): Part 5 used as BLACK SWAN veto but uses InstitutionalFusionEngine')
print('  part5 is MLEngineGPU + InstitutionalFusionEngine')
print('  InstitutionalFusionEngine.process_market_data is the veto - but m1 required')
print('  If m1 is None (no 1m frame), Part 5 veto is silently SKIPPED -> risk')
print()

# BUG 5: p3_institutional called TWICE in master_evaluate  
print('BUG 5 (PERFORMANCE): p3_institutional.generate_mtf_signals called twice')
print('  - Once inside evaluate_all_brains (per-timeframe)')
print('  - Once inside _master_evaluate (for super-vote)')
print('  FIX: Cache first call result, reuse it')
print()

# Check what evaluate_all_brains RETURNS vs what is used
print('BUG 6 (CRITICAL): evaluate_all_brains returns votes for Parts 4-10 via add_tf_sig')
print('  BUT Part 3 is only used inside evaluate_all_brains as seed for part_bridge')
print('  Part 3 does NOT get added to brain_tf_signals directly (no add_tf_sig for P3_Institutional)')
print('  EFFECT: P3_Institutional appears in xgb_acc shadow but NOT in vote tally brain_tf_signals')
print('  However, in _master_evaluate, P3 gets a super-vote of +3 separately - this is intentional design')
print()
