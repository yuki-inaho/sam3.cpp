"""Artifact validators must reject invalid evidence even in optimized Python."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from verify_synthetic_results import verify


class ValidationTests(unittest.TestCase):
    def test_incomplete_report_is_rejected_with_and_without_optimization(self):
        with tempfile.TemporaryDirectory(prefix="sam31-validation-") as temporary:
            root = Path(temporary)
            (root / "report.json").write_text(json.dumps({"complete": False, "model": {"test_fixture": False}}))
            with self.assertRaisesRegex(ValueError, "real checkpoint required"):
                verify(root, root, 1, .8)
            result = subprocess.run([sys.executable, "-O", str(TOOLS / "verify_synthetic_results.py"),
                                     "--output", str(root), "--inputs", str(root), "--frames", "1"],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("real checkpoint required", result.stderr)
