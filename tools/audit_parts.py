import ast, sys

parts_info = {}

for partfile in ['part1_FIXED.py', 'part2_FIXED.py', 'part3_FIXED.py', 'part4_FIXED.py', 'part5_FIXED.py', 
                 'part6_FIXED.py', 'part7_FIXED.py', 'part8_FIXED.py', 'part9_FIXED.py', 'part10_FIXED.py',
                 'part11_FIXED.py', 'part12_FIXED.py']:
    try:
        with open(partfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        parts_info[partfile] = classes
    except Exception as e:
        parts_info[partfile] = [f'ERROR: {e}']

for fname, classes in parts_info.items():
    print(f'{fname}: {classes}')
