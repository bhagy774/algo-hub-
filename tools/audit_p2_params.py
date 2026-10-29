import ast

# Check critical missing params:
# 1. MarketRegimeBrainGPU.get_regime_signals needs current_regime - mode_engine passes 2 args (price, psy)
# 2. OrderFlowBrainGPU.analyze_order_flow needs bid_ask_data - mode_engine passes 1 arg (df)
# 3. MarketRegimeBrainGPU.detect_market_regime needs volume_data - mode_engine passes 1 arg (df)
# 4. InstitutionalFlowBrainGPU.analyze_institutional_flow needs trade_data - mode_engine passes 1 arg (df)

with open('part2_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ['MarketRegimeBrainGPU', 'OrderFlowBrainGPU', 'InstitutionalFlowBrainGPU']:
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name in ['get_regime_signals', 'analyze_order_flow', 'detect_market_regime', 'analyze_institutional_flow']:
                args = [a.arg for a in method.args.args]
                defaults = method.args.defaults
                num_optional = len(defaults)
                num_required = len(args) - num_optional
                required = args[:num_required]
                optional = args[num_required:]
                print(f'{cls.name}.{method.name}: required={required}, optional={optional}')
