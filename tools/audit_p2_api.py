import ast

# Check Part 2 brain method signatures
with open('part2_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

target_classes = {
    'ZonePointFiveDetectorGPU': 'detect_0_5_zone_signals',
    'CandlePsychologyMasterGPU': 'analyze_candle_psychology',
    'VolumeProfileBrainGPU': 'detect_volume_signals',
    'MarketStructureBrainGPU': 'get_structure_signals',
    'OrderFlowBrainGPU': 'get_orderflow_signals',
    'MomentumOscillatorBrainGPU': 'get_momentum_signals',
    'VolatilityRegimeBrainGPU': 'get_volatility_signals',
    'CycleAnalysisBrainGPU': 'get_cycle_signals',
    'CorrelationMatrixBrainGPU': 'get_correlation_signals',
    'PatternRecognitionBrainGPU': 'get_pattern_signals',
    'SupportResistanceBrainGPU': 'get_sr_signals',
    'TrendAnalysisBrainGPU': 'get_trend_signals',
    'MarketRegimeBrainGPU': 'get_regime_signals',
    'PriceActionBrainGPU': 'get_pa_signals',
    'InstitutionalFlowBrainGPU': 'get_institutional_signals',
    'SignalFusionBrainGPU': 'fuse_signals',
}

print('=== Part 2 Brain Method Check ===')
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in target_classes:
        target_method = target_classes[cls.name]
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        if target_method in methods:
            method = methods[target_method]
            args = [a.arg for a in method.args.args]
            print(f'{cls.name}.{target_method}({args}): OK')
        else:
            available = [n for n in methods.keys() if not n.startswith('_')]
            print(f'{cls.name}: MISSING {target_method}! Available: {available}')
