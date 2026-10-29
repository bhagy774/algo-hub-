# Check part3_FIXED InstitutionalTradingEngineGPU methods
import ast
with open('part3_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == 'InstitutionalTradingEngineGPU':
        methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        print(f'InstitutionalTradingEngineGPU methods: {methods}')
