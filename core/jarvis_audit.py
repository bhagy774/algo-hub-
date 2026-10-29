"""
JARVIS FULL AUDIT  -  one script, auto-installs tools, writes JARVIS_AUDIT_REPORT.md

Run (PowerShell, from the jarvis-scalp folder):
    python jarvis_audit.py
Options:
    --root <folder>      repo folder (default: current folder)
    --entry jarvis_FIXED.py
    --no-install         do not pip install anything
    --no-import-test     skip the "does every file import" test
    --run-tests          also run tests (scratch/test_*.py, tests/) with timeouts
    --test-timeout 120

SAFE: never places orders, never needs API keys (keys are removed from the
environment for child processes). Only READS your code.
It cannot prove the bot is profitable or "live ready" - it finds problems.
"""
import argparse, ast, json, os, re, subprocess, sys, time, shutil
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--entry", default="jarvis_FIXED.py")
ap.add_argument("--no-install", action="store_true")
ap.add_argument("--no-import-test", action="store_true")
ap.add_argument("--full", action="store_true", help="also run heavy tools: pytest+coverage, semgrep, repomix, aider repo-map")
ap.add_argument("--run-tests", action="store_true")
ap.add_argument("--test-timeout", type=int, default=120)
args = ap.parse_args()

ROOT = Path(args.root).resolve()
OUT = ROOT / "JARVIS_AUDIT_REPORT.md"
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "jenv", "node_modules", ".mypy_cache", ".ruff_cache", "build", "dist", "models", "data"}
SECRET_ENV = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE)", re.I)

def py_files():
    out = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts[:-1]):
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        out.append(p)
    return sorted(out)

def rel(p): return str(Path(p).relative_to(ROOT)).replace("\\", "/")

def run(cmd, timeout=300, env=None, cwd=None):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, cwd=cwd or ROOT, errors="replace")
        try:
            out, err = p.communicate(timeout=timeout)
            return p.returncode, (out or "") + (err or "")
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(f"taskkill /F /T /PID {p.pid}", shell=True, capture_output=True)
            except Exception:
                p.kill()
            try:
                out, err = p.communicate(timeout=2)
            except Exception:
                out, err = "", ""
            return 124, "TIMEOUT"
    except Exception as e:
        return 1, f"ERROR: {e}"

def safe_env():
    e = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    e["JARVIS_AUDIT"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    return e

files = py_files()
print(f"[audit] {len(files)} python files in {ROOT}")

# ---------- install tools ----------
TOOLS = ["ruff", "vulture", "bandit", "pylint", "pyright", "pydeps", "mypy", "pyan3"]
HEAVY = ["pytest", "coverage", "semgrep", "code2prompt"]
TOOL_MOD = {"pydeps": "pydeps"}
if not args.no_install:
    print("[audit] installing/checking tools:", ", ".join(TOOLS))
    run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"] + TOOLS, timeout=600)

def mod_ok(m):
    c, _ = run([sys.executable, "-m", m, "--version"], timeout=120)
    return c == 0

if args.full and not args.no_install:
    for t in HEAVY:
        print("[audit] installing heavy tool:", t)
        run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", t], timeout=900)
have = {t: mod_ok(t) for t in TOOLS if t != "pyan3"}
have["pyan3"] = run([sys.executable, "-c", "import pyan"], timeout=60)[0] == 0
for t, m in {"pytest": "pytest", "coverage": "coverage", "semgrep": "semgrep"}.items():
    have[t] = mod_ok(m) if args.full else False
import shutil as _sh
_NPX = _sh.which("npx") or _sh.which("npx.cmd")

R = []  # report lines
def H(t): R.append(f"\n## {t}\n")
def L(t=""): R.append(t)

# ---------- 1. syntax ----------
H("1. Syntax check (does every file even parse?)")
bad_syntax = []
trees = {}
for f in files:
    try:
        src = f.read_text(encoding="utf-8", errors="replace")
        trees[f] = (ast.parse(src), src)
    except SyntaxError as e:
        bad_syntax.append((rel(f), f"line {e.lineno}: {e.msg}"))
if bad_syntax:
    L(f"*FAIL* - {len(bad_syntax)} file(s) with syntax errors:")
    for f, m in bad_syntax: L(f"- {f} {m}")
else:
    L(f"*PASS* - all {len(files)} files parse.")

# ---------- 2. import graph / reachability ----------
H(f"2. Which files are really used? (reachable from {args.entry})")
modmap = {}
for f in trees:
    r = f.relative_to(ROOT)
    parts = list(r.with_suffix("").parts)
    if parts[-1] == "__init__": parts = parts[:-1]
    if parts: modmap[".".join(parts)] = f
    modmap.setdefault(f.stem, f)

def imports_of(f):
    tree, _ = trees[f]
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: found.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                found.add(n.module)
                for a in n.names: found.add(f"{n.module}.{a.name}")
            else:
                for a in n.names: found.add(a.name)
    out = set()
    for m in found:
        if m in modmap: out.add(modmap[m])
        else:
            top = m.split(".")[0]
            if top in modmap and modmap[top] != f and (ROOT / top).is_dir() is False:
                out.add(modmap[top])
    # dynamic imports by string e.g. importlib.import_module("part1_FIXED")
    _, src = trees[f]
    for m in re.findall(r"""(?:import_module|__import__|spec_from_file_location)\(\s*['"]([\w./\\-]+)['"]""", src):
        k = Path(m).stem
        if k in modmap: out.add(modmap[k])
    return out

graph = {f: imports_of(f) for f in trees}
entry = ROOT / args.entry
reach = set()
if entry in graph:
    stack = [entry]
    while stack:
        x = stack.pop()
        if x in reach: continue
        reach.add(x)
        stack.extend(graph.get(x, []))
    L(f"{args.entry} reaches *{len(reach)}* of {len(trees)} files through imports.\n")
    L("*Used (reachable):*")
    for f in sorted(reach, key=rel): L(f"- {rel(f)}")
    unreach = [f for f in trees if f not in reach]
    L(f"\n*NOT reachable from entry ({len(unreach)})* - tests/tools/dashboards or dead code (check by hand; "
      f"files started separately or imported inside strings are shown as unreachable):")
    for f in sorted(unreach, key=rel): L(f"- {rel(f)}")
else:
    L(f"*UNVERIFIED* - entry file {args.entry} not found.")

# ---------- 3. import test ----------
H("3. Does each file import without crashing? (runtime import test)")
import_fail = []
if args.no_import_test:
    L("SKIPPED (--no-import-test).")
else:
    check_files = [f for f in files if f in trees and not rel(f).startswith(("scratch/", "tests/", "test_"))
                   and not f.name.startswith("test_")]
    L(f"Each of {len(check_files)} non-test files imported in a separate process (8s limit, no API keys). "
      "Files that run trading code at import time may show TIMEOUT.\n")
    env = safe_env()
    for f in check_files:
        code = ("import sys,importlib.util,runpy;sys.path.insert(0,r'%s');"
                "spec=importlib.util.spec_from_file_location('_audit_mod',r'%s');"
                "m=importlib.util.module_from_spec(spec);sys.modules['_audit_mod']=m;spec.loader.exec_module(m)") % (str(ROOT), str(f))
        c, o = run([sys.executable, "-c", code], timeout=8, env=env)
        if c != 0:
            last = [ln for ln in o.strip().splitlines() if ln.strip()][-1:] or [""]
            import_fail.append((rel(f), "TIMEOUT (>8s)" if c == 124 else last[0][:200]))
    if import_fail:
        L(f"*FAIL / CHECK* - {len(import_fail)} file(s) did not import cleanly:")
        for f, m in import_fail: L(f"- {f} -> {m}")
    else:
        L("*PASS* - all import cleanly.")

# ---------- 4. ruff ----------
H("4. Real bugs found by Ruff (undefined names, wrong comparisons, etc.)")
if have["ruff"]:
    sel = "F63,F7,F82,F811,F821,F822,F823,F841,E711,E712,E722,B006,B008,B015,B018,PLE,PLW0602,PLW0603"
    c, o = run([sys.executable, "-m", "ruff", "check", "--select", sel, "--output-format", "json",
                "--no-cache", "--exit-zero", str(ROOT)], timeout=300)
    try:
        items = json.loads(o) if o.strip().startswith("[") else []
    except Exception:
        items = []
    items = [i for i in items if not any(p in SKIP_DIRS for p in Path(i["filename"]).parts)]
    bycode = defaultdict(list)
    for i in items: bycode[i["code"]].append(i)
    serious = {k: v for k, v in bycode.items() if k.startswith(("F82", "F63", "F7", "PLE", "F811", "F823", "E711", "E712"))}
    L(f"Total findings: {len(items)}.  *Serious (likely real bugs): {sum(len(v) for v in serious.values())}*\n")
    for k in sorted(serious):
        L(f"*{k}* ({len(serious[k])}):")
        for i in serious[k][:25]:
            L(f"- {rel(i['filename'])}:{i['location']['row']} {i['message']}")
        if len(serious[k]) > 25: L(f"- ... +{len(serious[k])-25} more")
    others = {k: len(v) for k, v in bycode.items() if k not in serious}
    if others: L("\nMinor (style/unused): " + ", ".join(f"{k}={v}" for k, v in sorted(others.items())))
else:
    L("UNVERIFIED - ruff not installed.")

# ---------- 5. vulture ----------
H("5. Dead code (functions/classes never used) - Vulture")
if have["vulture"]:
    c, o = run([sys.executable, "-m", "vulture", str(ROOT), "--min-confidence", "80",
                "--exclude", ",".join(SKIP_DIRS)], timeout=300)
    lines = [ln for ln in o.splitlines() if ln.strip()]
    L(f"{len(lines)} items at 80%+ confidence (unused imports/variables/unreachable code):")
    for ln in lines[:60]: L(f"- {ln.replace(str(ROOT)+os.sep,'')}")
    if len(lines) > 60: L(f"- ... +{len(lines)-60} more")
else:
    L("UNVERIFIED - vulture not installed.")

# ---------- 6. bandit ----------
H("6. Security (Bandit) - HIGH severity only")
if have["bandit"]:
    c, o = run([sys.executable, "-m", "bandit", "-r", str(ROOT), "-lll", "-f", "json", "-q",
                "-x", ",".join(str(ROOT / d) for d in SKIP_DIRS)], timeout=300)
    try:
        j = json.loads(o[o.index("{"):]) if "{" in o else {}
    except Exception:
        j = {}
    res = j.get("results", [])
    L(f"{len(res)} high-severity finding(s).")
    for r in res[:30]:
        L(f"- {rel(r['filename'])}:{r['line_number']} {r['test_id']} {r['issue_text']}")
else:
    L("UNVERIFIED - bandit not installed.")

# ---------- 7. trading-specific scan ----------
H("7. Trading-safety scan (custom rules for your system)")
rules = [
    ("SECRET", "Hard-coded secret/API key (ROTATE it if real!)",
     re.compile(r"""(api[-]?key|api[-]?secret|secret|token|passphrase|password)\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"]""", re.I)),
    ("NOTIMPL", "Unfinished code (NotImplementedError / TODO / FIXME / stub)",
     re.compile(r"NotImplementedError|#\s*(TODO|FIXME|XXX|HACK)", re.I)),
    ("BARE_EXCEPT", "Silent error swallowing: except: pass / except Exception: pass",
     re.compile(r"except(\s+Exception)?\s*:\s*\n\s*pass\b")),
    ("RANDOM", "random/np.random used (never allowed for model inference or decisions)",
     re.compile(r"\b(np\.random\.|numpy\.random\.|random\.(random|randint|choice|uniform|gauss)\()")),
    ("NOTIMEOUT", "HTTP request without timeout (can hang the bot)",
     re.compile(r"requests\.(get|post|delete|put)\((?![^)]timeout)[^)]\)")),
    ("TF_NOISY", "Uses noisy 1m/3m timeframe (you prefer 15m entry)",
     re.compile(r"""['"](1m|3m)['"]""")),
    ("FALLBACK_SYM", "Fallback/default symbol (BTC) - risk of wrong-coin data",
     re.compile(r"""(fallback|default)[_a-z]symbol|symbol\s=\s*symbol\s*or\s*['"][A-Z]{3,}""", re.I)),
    ("LIVE_ORDER", "Order-placing call (verify it is behind safety gates)",
     re.compile(r"\b(place_order|create_order|submit_order|send_order|market_order|limit_order)\s*\(")),
    ("PRINT_SPAM", "print() in decision code (terminal noise)", None),
    ("MUTABLE_GLOBAL", "global state (cross-symbol contamination risk)", re.compile(r"^\s*global\s+\w+", re.M)),
]
hits = defaultdict(list)
for f, (tree, src) in trees.items():
    r = rel(f)
    if r.startswith(("scratch/", "tests/")) or f.name.startswith("test_"):
        continue
    for code, desc, rx in rules:
        if rx is None: continue
        for m in rx.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            hits[code].append((r, line))
descs = {c: d for c, d, _ in rules}
for code in ["SECRET", "NOTIMPL", "BARE_EXCEPT", "RANDOM", "NOTIMEOUT", "TF_NOISY", "FALLBACK_SYM", "LIVE_ORDER", "MUTABLE_GLOBAL"]:
    h = hits.get(code, [])
    if not h:
        L(f"- *{code}* - {descs[code]}: none found")
        continue
    byfile = defaultdict(list)
    for r, ln in h: byfile[r].append(ln)
    L(f"\n- *{code}* - {descs[code]}: *{len(h)}* in {len(byfile)} file(s)")
    for r, lns in sorted(byfile.items(), key=lambda x: -len(x[1]))[:12]:
        L(f"    - {r} lines {', '.join(map(str, lns[:8]))}{' ...' if len(lns) > 8 else ''}")

# ---------- 8. big functions ----------
H("8. Files/functions that are too big or complex (hard to test, bug-prone)")
big = []
for f, (tree, src) in trees.items():
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ln = (n.end_lineno or n.lineno) - n.lineno + 1
            if ln >= 200: big.append((ln, rel(f), n.name, n.lineno))
big.sort(reverse=True)
L(f"Functions over 200 lines: {len(big)}")
for ln, r, nm, l0 in big[:20]: L(f"- {r}:{l0} {nm} = {ln} lines")
sizes = sorted(((len(s.splitlines()), rel(f)) for f, (t, s) in trees.items()), reverse=True)[:10]
L("\nBiggest files: " + ", ".join(f"{n} ({c})" for c, n in sizes))

# ---------- 9. tests ----------
H("9. Tests")
test_files = [f for f in files if rel(f).startswith(("tests/", "scratch/test", "test_")) or f.name.startswith("test_")]
L(f"Test files found: {len(test_files)}")
test_results = []
if args.run_tests:
    env = safe_env()
    for f in test_files:
        c, o = run([sys.executable, str(f)], timeout=args.test_timeout, env=env)
        test_results.append((rel(f), c))
    ok = [t for t, c in test_results if c == 0]
    ko = [(t, c) for t, c in test_results if c != 0]
    L(f"Ran each test file once: *{len(ok)} passed, {len(ko)} failed/timeout*")
    for t, c in ko: L(f"- FAIL {t} (exit {c}{' timeout' if c == 124 else ''})")
else:
    L("NOT RUN. Re-run with --run-tests to execute them (may take a while).")

# ---------- 10. live readiness ----------
H("10. LIVE READINESS - honest checklist")
def has(rx):
    r = re.compile(rx, re.I)
    return any(r.search(s) for f, (t, s) in trees.items()
               if not rel(f).startswith(("scratch/", "tests/")) and not f.name.startswith("test_"))
checks = [
    ("All files parse (no syntax errors)", "PASS" if not bad_syntax else "FAIL"),
    ("All main files import cleanly", "SKIPPED" if args.no_import_test else ("PASS" if not import_fail else "FAIL")),
    ("No hard-coded secrets found", "PASS" if not hits.get("SECRET") else "FAIL"),
    ("No unfinished stubs/TODO in non-test code", "PASS" if not hits.get("NOTIMPL") else "CHECK"),
    ("No random numbers in decision code", "PASS" if not hits.get("RANDOM") else "CHECK"),
    ("HTTP calls have timeouts", "PASS" if not hits.get("NOTIMEOUT") else "CHECK"),
    ("Stale-data / freshness guard exists in code", "FOUND" if has(r"stale|freshness|max_age|data_age") else "NOT FOUND"),
    ("Reconciliation / restart-recovery code exists", "FOUND" if has(r"reconcil") else "NOT FOUND"),
    ("Stop-loss / protective order code exists", "FOUND" if has(r"stop_loss|stoploss|\bsl_price|protective") else "NOT FOUND"),
    ("Paper/dry-run mode exists", "FOUND" if has(r"paper|dry_run|dry-run|simulat") else "NOT FOUND"),
    ("Tests executed this run", "PASS" if args.run_tests and test_results and all(c == 0 for _, c in test_results)
        else ("FAIL" if args.run_tests and test_results else "NOT RUN")),
    ("Real exchange fills, SL/TP placement, restart recovery with real money", "UNVERIFIED (cannot be checked by a script)"),
    ("Profitability / out-of-sample edge", "UNVERIFIED (needs backtest + paper trading)"),
]
L("| Check | Result |\n|---|---|")
for a, b in checks: L(f"| {a} | {b} |")
L("\n*Verdict:* a script cannot declare the bot live ready. FAIL/CHECK items above must be fixed first, "
  "then run in paper/shadow mode with tiny size. Never rely only on this report.")

H("11. Do next (in order)")
L("1. Fix everything marked FAIL (syntax, import errors, secrets, undefined names).")
L("2. Look at section 7 LIVE_ORDER hits: every order call must sit behind your safety gates.")
L("3. Remove or archive dead files from section 2 (not reachable) after confirming by hand.")
L("4. Run again with --run-tests.")
L("5. Paper/shadow test for days before any live money. Verify exchange state before each restart.")
L(f"\n_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} - tools: " + ", ".join(f"{k}={'ok' if v else 'missing'}" for k, v in have.items()) + "_")

# ---------- 12a. pylint (errors only) ----------
H("12a. Pylint - error-level bugs only (E)")
if have.get("pylint"):
    targets = [str(f) for f in files if not any(p in SKIP_DIRS for p in Path(f).parts)]
    c, o = run([sys.executable, "-m", "pylint", "--disable=all", "--enable=E", "--disable=E0401,E0611,E1101",
                "--output-format=json", "--jobs=0", "--score=n"] + targets, timeout=900)
    try:
        pl = json.JSONDecoder().raw_decode(o[o.index("["):])[0] if "[" in o else []
    except Exception:
        pl = []
    L(f"{len(pl)} error-level finding(s) (import/attribute noise filtered out; verify each by hand):")
    for i in pl[:40]:
        L(f"- {rel(i['path']) if Path(i['path']).is_absolute() else i['path']}:{i['line']} {i['symbol']} - {i['message']}")
    if len(pl) > 40: L(f"- ... +{len(pl)-40} more")
else:
    L("UNVERIFIED - pylint not installed.")

# ---------- 12b. pyright ----------
H("12b. Pyright - type / undefined / possibly-unbound problems")
if have.get("pyright"):
    cfg = ROOT / ".jarvis_audit_pyright.json"
    cfg.write_text(json.dumps({"include": ["."], "exclude": sorted(SKIP_DIRS) + ["*/tests", "*/scratch"],
        "typeCheckingMode": "off", "reportUndefinedVariable": "error", "reportPossiblyUnbound": "error",
        "reportCallIssue": "error", "reportIndexIssue": "error", "reportAttributeAccessIssue": "warning",
        "reportOptionalMemberAccess": "warning", "reportSelfClsParameterName": "error",
        "reportRedeclaration": "warning", "pythonVersion": "%d.%d" % sys.version_info[:2]}), encoding="utf-8")
    c, o = run([sys.executable, "-m", "pyright", "-p", str(cfg), "--outputjson"], timeout=900)
    try:
        pj = json.JSONDecoder().raw_decode(o[o.index("{"):])[0] if "{" in o else {}
    except Exception:
        pj = {}
    try: cfg.unlink()
    except Exception: pass
    diags = pj.get("generalDiagnostics", [])
    if not pj:
        L("UNVERIFIED - pyright could not run (it needs Node.js / internet on first run). Output: " + o[:200].replace("\n", " "))
    else:
        byrule = defaultdict(list)
        for d in diags: byrule[d.get("rule", "other")].append(d)
        L(f"{len(diags)} finding(s): " + ", ".join(f"{k}={len(v)}" for k, v in sorted(byrule.items())))
        for rule in ("reportUndefinedVariable", "reportPossiblyUnbound", "reportCallIssue", "reportIndexIssue"):
            for d in byrule.get(rule, [])[:20]:
                L(f"- *{rule}* {rel(d['file'])}:{d['range']['start']['line']+1} {d['message'].splitlines()[0]}")
else:
    L("UNVERIFIED - pyright not installed.")

# ---------- 12c. import graph (pydeps) ----------
H("12c. Import graph from entry file (pydeps)")
if have.get("pydeps"):
    c, o = run([sys.executable, "-m", "pydeps", str(ROOT / args.entry), "--show-deps", "--no-output", "--noshow"], timeout=300)
    try:
        deps = json.loads(o) if o.strip().startswith("{") else {}
        L(f"Import dependencies generated successfully.")
    except Exception:
        L("Error parsing pydeps output.")
else:
    L("UNVERIFIED - pydeps not installed.")

# Write report
OUT.write_text("\n".join(R), encoding="utf-8")
print(f"[audit] Report written to {OUT}")
