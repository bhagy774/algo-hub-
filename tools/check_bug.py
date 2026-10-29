# Now cross-check mode_engine.py signal logic with actual return keys

# From audit:
# TrendBrain: support_score, trend_direction, trend_strength, etc
# VolatilityBrain: volatility_regime, support_score, volatility_score 
# StrengthBrain: overall_strength, support_score
# RiskBrain: stop_distance, risk_score, risk_level??? (mode_engine uses risk_level)
# ReversalBrain: reversal_probability, support_score
# RegimeBrain: regime_type, regime_confidence
# DeepSeekBrain: fakeout_block, support_score
# EvolutionBrain: support_score, performance_score
# MemoryBrain: support_score (mode_engine uses support_score + .get negative check)
# SelfHealingBrain: health_index
# MiniR1Brain: trap_probability
# MiniV3Brain: tick_momentum

# CRITICAL BUG CHECK: mode_engine uses risk_level but RiskBrain returns risk_score, not risk_level!
print('Known keys from RiskBrain:', ['stop_distance', 'fakeout_probability', 'position_size', 'avg_risk', 'support_score', 'risk_score'])
print('mode_engine uses: rk_res.get(\"risk_level\")')
print('BUG: risk_level not in RiskBrain - should be risk_score or derived')
