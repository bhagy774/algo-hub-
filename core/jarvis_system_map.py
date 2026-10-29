"""Generate SYSTEM_MAP.json: what every .py file in the repo does (module docstring,
classes, top-level functions). Run: python jarvis_system_map.py
Documentation/inspection aid only; XGBoost consumes numeric Part outputs (see
jarvis_part_snapshot.py), not source text."""
import ast, json, sys
from pathlib import Path

def main(root="."):
    out = {}
    for p in sorted(Path(root).rglob("*.py")):
        if any(s in p.parts for s in (".git", "venv", "__pycache__", "tests")):
            continue
        try:
            t = ast.parse(p.read_text(errors="ignore"))
        except Exception as e:
            out[str(p)] = {"error": type(e).__name__}; continue
        out[str(p)] = {
            "doc": (ast.get_docstring(t) or "").strip().split("\n")[0][:200],
            "classes": [n.name for n in t.body if isinstance(n, ast.ClassDef)][:30],
            "functions": [n.name for n in t.body if isinstance(n, ast.FunctionDef)][:40],
        }
    Path("SYSTEM_MAP.json").write_text(json.dumps(out, indent=1))
    print(f"mapped {len(out)} files")

if __name__ == "__main__":
    main(*sys.argv[1:])
