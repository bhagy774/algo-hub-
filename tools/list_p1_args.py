import ast

def get_args(method_node):
    return [a.arg for a in method_node.args.args if a.arg != 'self']

with open('part1_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

expected = {}
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and 'Brain' in cls.name:
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name.startswith('analyze_'):
                expected[cls.name] = get_args(method)

for k, v in expected.items():
    print(f"{k}: {v}")
