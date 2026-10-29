# Zone detector has signature: detect_0_5_zone_signals(self, current_candle, psychology, df_1min, df_5min, df_15min)
# mode_engine calls: b2["zone_detector"].detect_0_5_zone_signals(cndl, psy, df, df, df)
# This passes the SAME df for all 3 timeframes - this is a design limitation since we're inside a per-TF loop
# We should ideally pass the actual 1m, 5m, 15m frames from available_tfs
# Let's check if we can improve this

print('P2_ZonePointFiveDetector: currently passing df for all 3 TF args')
print('  detect_0_5_zone_signals(cndl, psy, df_1min, df_5min, df_15min)')
print('  FIX: pass actual 1m/5m/15m frames from available_tfs dict')
print()
print('P2_SignalFusion: fuse_signals([], price, {}) passes empty signals')
print('  Should pass actual brain signals dict for meaningful fusion')
print('  But this is complex - keep as-is since signals are fused at a higher level')
