"""Isolated tests for scripts/audit_system.py; never import the trading app."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve()
AUDIT = HERE.parents[1] / "scripts" / "audit_system.py"


class AuditRunnerTests(unittest.TestCase):
    def run_audit(self, source: str, extra: str = ""):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "jarvis_FIXED.py").write_text(source, encoding="utf-8")
            out = root / "report.json"
            md = root / "report.md"
            cp = subprocess.run(
                [sys.executable, str(AUDIT), "--offline", "--root", str(root),
                 "--json", str(out), "--markdown", str(md)],
                capture_output=True, text=True, timeout=15,
            )
            return cp, json.loads(out.read_text(encoding="utf-8"))

    def test_safe_fixture_and_no_network_guard(self):
        cp, report = self.run_audit('''\n# Importing this fixture would be unsafe; the auditor must never import it.\nraise RuntimeError("fixture application imported")\ndef main():\n    return 0\nif __name__ == "__main__":\n    main()\n''')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(report["counts"]["FAIL"], 0)
        self.assertTrue(report["offline"])

    def test_syntax_failure_nonzero(self):
        cp, report = self.run_audit("def broken(:\n")
        self.assertNotEqual(cp.returncode, 0)
        self.assertGreaterEqual(report["counts"]["FAIL"], 1)

    def test_redaction_never_emits_secret_value(self):
        cp, report = self.run_audit('''\nAPI_KEY = "DO_NOT_PRINT_THIS_VALUE"\ndef main(): pass\n''')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        raw = json.dumps(report)
        self.assertNotIn("DO_NOT_PRINT_THIS_VALUE", raw)
        self.assertIn("REDACTED", raw)

    def test_unclassified_missing_dependency_is_not_claimed_local_clean(self):
        cp, report = self.run_audit('''\nimport definitely_missing_audit_dependency\ndef main(): pass\n''')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        wiring = next(x for x in report["findings"] if x["check"] == "import-wiring")
        self.assertEqual(wiring["status"], "UNVERIFIED")
        dep = next(x for x in report["findings"] if x["check"] == "dependency-classification")
        self.assertEqual(dep["status"], "UNVERIFIED")
        self.assertIn("definitely_missing_audit_dependency", dep["evidence"])

    def test_static_only_paper_and_restart_evidence_is_unverified(self):
        cp, report = self.run_audit('''
PAPER_CONFIG = {"save_file": "paper_trades.json"}
def _check_paper_trades():
    return "paper", "reduce_only"
def _save_paper_state():
    return "persist"
def _load_paper_state():
    return "load"
def reconcile():
    return "reconcile"
def main(): pass
''')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        findings = {x["check"]: x for x in report["findings"]}
        self.assertEqual(findings["paper-isolation"]["status"], "UNVERIFIED")
        self.assertEqual(findings["restart-reconciliation"]["status"], "UNVERIFIED")
        self.assertNotEqual(findings["paper-isolation"]["status"], "PASS")
        self.assertNotEqual(findings["restart-reconciliation"]["status"], "PASS")
        self.assertIn("Static evidence only", findings["paper-isolation"]["message"])

    def test_secret_candidates_are_review_required_and_redacted(self):
        cp, report = self.run_audit('''\nAPI_KEY = "DO_NOT_PRINT_THIS_VALUE"\ndef main(): pass\n''')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        secret = next(x for x in report["findings"] if x["check"] == "secrets-scan")
        self.assertEqual(secret["status"], "UNVERIFIED")
        self.assertIn("review required", secret["message"])
        self.assertNotIn("DO_NOT_PRINT_THIS_VALUE", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
