# Check all Part 4-12 analyze methods to see if they all have context param
import ast

parts = {
    'part4_FIXED.py': 'VolumeProfileEngineGPU',
    'part5_FIXED.py': 'MLEngineGPU',
    'part6_FIXED.py': 'TrendEngineGPU',
    'part7_FIXED.py': 'VolatilityEngineGPU',
    'part8_FIXED.py': 'MarketStructureEngineGPU',
    'part9_FIXED.py': 'OrderflowEngineGPU',
    'part10_FIXED.py': 'CandleStatsEngineGPU',
    'part11_FIXED.py': 'SignalFusionEngineGPU',
    'part12_FIXED.py': 'ConfidenceEngineGPU',
}

for partfile, cls_name in parts.items():
    try:
        with open(partfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name == 'analyze':
                        args = [a.arg for a in method.args.args]
                        has_context = 'context' in args
                        print(f'{cls_name}.analyze({args}) -> has context: {has_context}')
    except Exception as e:
        print(f'{partfile}: ERROR {e}')
