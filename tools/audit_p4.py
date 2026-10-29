# Also check Part 3 add_tf_sig - does mode_engine add P3 to brain_tf_signals?
# Looking at mode_engine lines 460-495, P3 is only seed for part_bridge, not brain_tf_signals
# But in _master_evaluate P3 votes separately (super-vote +3)
# This is by design but we should also check P4-P10 add_tf_sig

# Check Part 4 VolumeProfileEngineGPU.analyze method signature  
import ast
with open('part4_FIXED.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == 'VolumeProfileEngineGPU':
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == 'analyze':
                args = [a.arg for a in method.args.args]
                defaults = [d for d in method.args.defaults]
                print(f'VolumeProfileEngineGPU.analyze args: {args}')
                # Find return keys
                return_keys = []
                for node in ast.walk(method):
                    if isinstance(node, ast.Dict):
                        for k in node.keys:
                            if isinstance(k, ast.Constant):
                                return_keys.append(k.value)
                print(f'  returns keys: {list(set(return_keys))[:10]}')
