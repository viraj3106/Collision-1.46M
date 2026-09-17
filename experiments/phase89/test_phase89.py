import os
import sys
import json
import unittest
import hashlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PHASE89_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "collision-10m", "model.pt")

class TestPhase89Forensics(unittest.TestCase):

    def test_model_sha256_integrity(self):
        """Verify production model SHA256 is unchanged."""
        sha = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        expected_sha = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
        self.assertEqual(sha.hexdigest(), expected_sha)

    def test_deliverables_exist_if_run(self):
        """Check if phase89 deliverables exist."""
        report_path = os.path.join(PHASE89_DIR, "phase89_forensic_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# PHASE 89 — GENERATION FAILURE FORENSIC REPORT", content)
            self.assertIn("PHASE_89_ROOT_CAUSE_IDENTIFIED", content)

if __name__ == "__main__":
    unittest.main()
