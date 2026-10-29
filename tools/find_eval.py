import ast
import json

with open('mode_engine.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

engine_class = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModeEngine'][0]
eval_method = [n for n in engine_class.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_all_brains'][0]

print(f"evaluate_all_brains lines: {eval_method.lineno} to {eval_method.end_lineno}")
