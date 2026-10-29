import ast
import re

with open('mode_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# find all add_tf_sig calls in evaluate_all_brains
matches = re.findall(r'add_tf_sig\("([^"]+)"', content)
print("Signals added via add_tf_sig:")
for m in matches:
    print(m)

print("\nCount:", len(matches))
