import re

with open('mode_part_bridge.py', 'r', encoding='utf-8') as f:
    bridge_lines = f.readlines()

# Extract imports and code from bridge
bridge_code = []
imports = []
in_docstring = False

for line in bridge_lines:
    if line.startswith('"""ModeEngine'):
        in_docstring = True
        continue
    if in_docstring and '"""' in line:
        in_docstring = False
        continue
    if in_docstring:
        continue
    if line.startswith('from __future__'):
        continue
        
    if line.startswith('import ') or line.startswith('from typing import'):
        imports.append(line)
    else:
        bridge_code.append(line)

with open('mode_engine.py', 'r', encoding='utf-8') as f:
    engine_content = f.read()

# Remove the imports
engine_content = engine_content.replace('from mode_part_bridge import PartEngineBridge', '')
engine_content = engine_content.replace('from mode_part_bridge import VOTE_NAMES, direction', '')

# Insert the code
# Find line 18 (after 'import part3_FIXED as p3')
engine_lines = engine_content.splitlines(True)
insert_idx = 0
for i, line in enumerate(engine_lines):
    if line.startswith('import part3_FIXED as p3'):
        insert_idx = i + 1
        break

engine_lines[insert_idx:insert_idx] = ['\n'] + imports + ['\n'] + bridge_code + ['\n']

with open('mode_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(engine_lines)
print("Merged mode_part_bridge into mode_engine.")
