# The Part 5 call in mode_engine uses: p5_res.get('fused_signal', 'NO TRADE')
# Part 5 process_market_data signature: (df_1min, df_5min, df_15min, market_regime='RANGING')
# mode_engine calls: self.p5_fusion.process_market_data(m1, m5_data, m15)
# This is CORRECT (3 positional args + default regime)
# The fused_signal returns 'CALL'/'PUT'/'NO TRADE' but mode_engine checks for 'BUY'/'SELL'
# BUG: p5 returns CALL/PUT but mode_engine checks for BUY/SELL!
print('BUG (CRITICAL): Part 5 returns fused_signal as CALL/PUT/NO TRADE')
print('  mode_engine checks: if p5_signal in ["BUY", "SELL"]')
print('  But Part 5 InstitutionalFusionEngine returns "CALL" or "PUT" not "BUY"/"SELL"')
print('  EFFECT: Part 5 veto NEVER fires even if it detects a black swan!')
print()
print('FIX: Check for "CALL" -> "BUY" and "PUT" -> "SELL" mapping')
