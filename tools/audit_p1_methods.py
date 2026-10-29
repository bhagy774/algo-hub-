import ast

# Check Part 1 brain methods
with open('part1_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ['TrendBrain', 'VolatilityBrain', 'StrengthBrain', 'RiskBrain', 
                                                       'ReversalBrain', 'RegimeBrain', 'DeepSeekBrain', 'EvolutionBrain', 
                                                       'MemoryBrain', 'SelfHealingBrain', 'MetaFusionBrain', 'MiniR1Brain', 'MiniV3Brain']:
        methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        print(f'{cls.name}: {methods}')
