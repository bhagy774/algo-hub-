import ast
import os

parts_brains = {}
for i in range(1, 13):
    part_file = f'part{i}_FIXED.py'
    if not os.path.exists(part_file):
        continue
    with open(part_file, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    # Filter only those that seem like brains or engines
    brains = [c for c in classes if 'Brain' in c or 'Engine' in c or 'Detector' in c or 'Master' in c or 'Predictor' in c or 'System' in c or 'Manager' in c]
    parts_brains[f'Part {i}'] = brains

for p, b in parts_brains.items():
    print(f'{p}: {b}')
