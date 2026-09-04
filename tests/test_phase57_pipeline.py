import unittest
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE57_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase57")

class TestPhase57Pipeline(unittest.TestCase):
    def test_website_landing_page_file_exists(self):
        app_path = os.path.join(PROJECT_ROOT, "website", "src", "App.tsx")
        self.assertTrue(os.path.exists(app_path), "App.tsx landing file missing")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("COLLISION", content)
        self.assertIn("Small AI. Built from scratch.", content)
        self.assertIn("TRY COLLISION → PLAYGROUND", content)
        self.assertIn("GET API KEY → DEVELOPER PORTAL", content)

    def test_data_status_and_diversity_reports_phase57(self):
        status = generate_data_status_reports(phase_dir=PHASE57_DIR)
        self.assertEqual(status["target_clean_records"], 100)
        self.assertIn("current_clean_records", status)
        
        diversity_path = os.path.join(PHASE57_DIR, "diversity_report.json")
        self.assertTrue(os.path.exists(diversity_path), "diversity_report.json missing")

    def test_training_readiness_gate_phase57(self):
        gate = check_training_readiness_gate(phase_dir=PHASE57_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_57_DATA_COLLECTION_ACTIVE")
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

if __name__ == "__main__":
    unittest.main()
