import ast

# Check Part 2 first-call methods (analyze methods before get_signals)
with open('part2_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

first_call_methods = {
    'VolumeProfileBrainGPU': 'analyze_volume_profile',
    'MarketStructureBrainGPU': 'analyze_market_structure',
    'OrderFlowBrainGPU': 'analyze_order_flow',
    'MomentumOscillatorBrainGPU': 'calculate_all_oscillators',
    'VolatilityRegimeBrainGPU': 'analyze_volatility_regime',
    'CycleAnalysisBrainGPU': 'analyze_market_cycles',
    'CorrelationMatrixBrainGPU': 'analyze_correlations',
    'PatternRecognitionBrainGPU': 'recognize_chart_patterns',
    'SupportResistanceBrainGPU': 'calculate_support_resistance',
    'TrendAnalysisBrainGPU': 'analyze_multi_timeframe_trends',
    'MarketRegimeBrainGPU': 'detect_market_regime',
    'PriceActionBrainGPU': 'analyze_price_action',
    'InstitutionalFlowBrainGPU': 'analyze_institutional_flow',
}

print('=== Part 2 First-Call Method Check ===')
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in first_call_methods:
        target_method = first_call_methods[cls.name]
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        if target_method in methods:
            method = methods[target_method]
            args = [a.arg for a in method.args.args]
            print(f'{cls.name}.{target_method}({args}): OK')
        else:
            available = [n for n in methods.keys() if not n.startswith('_')]
            print(f'{cls.name}: MISSING {target_method}! Available: {available}')

# Also check MarketRegimeBrainGPU.get_regime_signals - mode_engine passes 2 args
# mode_engine: b2[\"market_regime\"].get_regime_signals(price, psy)
# but Part 2: get_regime_signals(['self', 'current_price', 'psychology', 'current_regime'])
# POSSIBLE BUG: current_regime is missing! It has 4 params including default?
print()
print('=== Checking MarketRegimeBrainGPU.get_regime_signals default param ===')
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == 'MarketRegimeBrainGPU':
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'get_regime_signals':
                args = [a.arg for a in method.args.args]
                defaults = method.args.defaults
                print(f'  args: {args}')
                print(f'  num defaults: {len(defaults)} (from right)')
                # If there are defaults, show which args have defaults
                if defaults:
                    num_required = len(args) - len(defaults)
                    req = args[:num_required]
                    opt = args[num_required:]
                    print(f'  required: {req}')
                    print(f'  optional: {opt}')
