import ast

# Check Part 1 analyze methods signatures and return keys
with open('part1_FIXED.py', 'r', encoding='utf-8') as f:
    content = f.read()
    tree = ast.parse(content)
    lines = content.splitlines()

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ['TrendBrain', 'VolatilityBrain', 'StrengthBrain', 'RiskBrain', 
                                                       'ReversalBrain', 'RegimeBrain', 'DeepSeekBrain', 'EvolutionBrain', 
                                                       'MemoryBrain', 'SelfHealingBrain', 'MetaFusionBrain', 'MiniR1Brain', 'MiniV3Brain']:
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name.startswith('analyze'):
                # Get the method args
                args = [a.arg for a in method.args.args]
                # Find all dict keys returned
                return_keys = []
                for node in ast.walk(method):
                    if isinstance(node, ast.Dict):
                        for k in node.keys:
                            if isinstance(k, ast.Constant):
                                return_keys.append(k.value)
                print(f'{cls.name}.{method.name}({args}):')
                print(f'  returns keys: {list(set(return_keys))[:15]}')
                print()
