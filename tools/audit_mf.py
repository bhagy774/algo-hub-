import ast

# Check MetaFusionBrain.analyze_meta_fusion params
with open('part1_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == 'MetaFusionBrain':
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'analyze_meta_fusion':
                args = [a.arg for a in method.args.args]
                print(f'MetaFusionBrain.analyze_meta_fusion args: {args}')
                # Find what keys it accesses from all_brain_support
                for node in ast.walk(method):
                    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                        if hasattr(node, 'value') and isinstance(node.value, ast.Name) and node.value.id == 'all_brain_support':
                            print(f'  accesses key: {node.slice.value}')
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr == 'get' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'all_brain_support':
                            if node.args and isinstance(node.args[0], ast.Constant):
                                print(f'  .get key: {node.args[0].value}')
