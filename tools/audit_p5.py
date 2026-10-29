# Check part5_FIXED InstitutionalFusionEngine methods and process_market_data signature
import ast
with open('part5_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name in ['InstitutionalFusionEngine', 'MLEngineGPU']:
        methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        print(f'{cls.name} methods: {methods}')
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'process_market_data':
                args = [a.arg for a in method.args.args]
                print(f'  process_market_data args: {args}')
