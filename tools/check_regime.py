# BUG: MarketRegimeBrainGPU.get_regime_signals requires 3 non-self args: current_price, psychology, current_regime
# mode_engine calls: b2["market_regime"].get_regime_signals(price, psy)  -- MISSING current_regime!
# This will always raise TypeError at runtime (but it's wrapped in try/except in eval_p2_tf)
# So P2_MarketRegime silently always returns (NEUTRAL, 0)

# Let's look at what current_regime should be
import ast
with open('part2_FIXED.py', 'r', encoding='utf-8') as f:
    content = f.read()
    tree = ast.parse(content)

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == 'MarketRegimeBrainGPU':
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'get_regime_signals':
                # Get first 15 lines of the method body
                lines = content.splitlines()
                start = method.lineno - 1
                end = min(start + 20, len(lines))
                print(f'get_regime_signals body:')
                for l in lines[start:end]:
                    print(l)
