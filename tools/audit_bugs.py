# Full audit of mode_engine.py signal logic vs actual Part 1 returned keys

# From Part 1 audit:
actual_keys = {
    'TrendBrain': ['trend_strength', 'momentum', 'trend_quality', 'support_score', 'trend_direction'],
    'VolatilityBrain': ['volatility_trend', 'volatility_regime', 'support_score', 'volatility_score', 'breakout_potential'],
    'StrengthBrain': ['breakout_strength', 'volume_confirmation', 'strength_momentum', 'momentum_strength', 'support_score', 'overall_strength'],
    'RiskBrain': ['stop_distance', 'fakeout_probability', 'position_size', 'avg_risk', 'support_score', 'risk_score'],  # NO risk_level!
    'ReversalBrain': ['divergence_detected', 'volume_momentum', 'exhaustion_signals', 'support_score', 'price_momentum', 'reversal_probability'],
    'RegimeBrain': ['support_score', 'regime_confidence', 'transition_phase', 'regime_type'],
    'DeepSeekBrain': ['fakeout_block', 'contradiction_score', 'noise_level', 'support_score', 'correction_factor', 'threshold_boost'],
    'EvolutionBrain': ['adaptation_factor', 'generation', 'support_score', 'performance_score', 'learning_rate'],
    'MemoryBrain': ['prediction_confidence', 'memory_strength', 'long_term_pattern', 'short_term_recall', 'support_score'],
    'SelfHealingBrain': ['error_rate', 'stability_score', 'health_index', 'support_score', 'recovery_factor'],
    'MetaFusionBrain': ['support_score', 'consensus_level', 'weight_distribution', 'fusion_score'],
    'MiniR1Brain': ['reasoning_depth', 'psychology_score', 'volatility_override', 'support_score', 'trap_probability'],
    'MiniV3Brain': ['pressure_detection', 'microstructure_score', 'support_score', 'wick_imbalance', 'tick_momentum'],
}

# From mode_engine.py add_tf_sig lines:
mode_engine_access = {
    'P1_Trend': ['support_score', 'trend_direction'],  # t_res
    'P1_Volatility': ['volatility_regime', 'trend_direction'],  # v_res + t_res
    'P1_Strength': ['overall_strength', 'trend_direction'],  # st_res + t_res
    'P1_Risk': ['risk_level', 'trend_direction'],  # rk_res - BUG: risk_level not in RiskBrain
    'P1_Reversal': ['reversal_probability', 'trend_direction'],  # rv_res + t_res
    'P1_Regime': ['regime_type', 'regime_confidence'],  # reg_res
    'P1_DeepSeek': ['fakeout_block', 'trend_direction'],  # ds_res + t_res
    'P1_Evolution': ['support_score', 'trend_direction'],  # evo_res + t_res
    'P1_Memory': ['support_score'],  # mem_res
    'P1_SelfHealing': ['health_index', 'trend_direction'],  # heal_res + t_res
    'P1_MiniR1': ['trap_probability', 'trend_direction'],  # r1_res + t_res
    'P1_MiniV3': ['tick_momentum'],  # v3_res
    'P1_MetaFusion': ['support_score', 'trend_direction'],  # mf_res + t_res
}

print('=== BUGS FOUND ===')
print()
print('1. P1_Risk uses rk_res.get("risk_level") but RiskBrain returns risk_score (not risk_level)')
print('   FIX: Compare risk_score thresholds instead of checking for "LOW" string')
print()
print('2. MetaFusionBrain.analyze_meta_fusion signature is analyze_meta_fusion(all_brain_support)')
print('   Check if mode_engine passes the correct keys')
print()

# Also check: what does add_tf_sig for P1_Reversal say?
# add_tf_sig("P1_Reversal", "SELL" if rv_res["reversal_probability"] > 0.001 and t_res["trend_direction"] == 1 
#            else "BUY" if rv_res["reversal_probability"] > 0.001 else "NEUTRAL", rv_res["reversal_probability"])
# BUG: A reversal signal suggests market will reverse, so if trend is UP and reversal is detected -> should be SELL
# This is actually correct - reversal while bullish means going down = SELL. But "BUY" when no trend given?
# If reversal_probability > 0.001 AND trend == 1 -> SELL (correct)
# elif reversal_probability > 0.001 -> BUY (WRONG! No trend direction, can be sell)
# More correct: else NEUTRAL
print('3. P1_Reversal BUY signal logic is wrong: if reversal_prob > 0.001 and not uptrend -> BUY is illogical')
print('   Should be: if uptrend and reversal -> SELL; if downtrend and reversal -> BUY; else NEUTRAL')
