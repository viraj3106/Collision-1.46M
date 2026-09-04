import unittest
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE55_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase55")

class TestPhase55Pipeline(unittest.TestCase):
    def test_production_integrity_phase55(self):
        integrity_path = os.path.join(PHASE55_DIR, "production_integrity.json")
        self.assertTrue(os.path.exists(integrity_path), "production_integrity.json missing")
        with open(integrity_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["trainable_parameter_count"], 10282304)
        self.assertEqual(data["production_sha256"], "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")
        self.assertEqual(data["served_model_name"], "collision-10m")
        self.assertEqual(data["promoted_research_candidate"], "J52")

    def test_data_status_and_diversity_reports(self):
        status = generate_data_status_reports(phase_dir=PHASE55_DIR)
        self.assertIn("target_clean_records", status)
        self.assertEqual(status["target_clean_records"], 100)
        self.assertIn("domain_distribution", status)

        diversity_path = os.path.join(PHASE55_DIR, "diversity_report.json")
        self.assertTrue(os.path.exists(diversity_path), "diversity_report.json missing")
        with open(diversity_path, "r", encoding="utf-8") as f:
            div_data = json.load(f)
        self.assertEqual(div_data["domain_categories_count"], 11)
        self.assertEqual(div_data["diversity_audit_status"], "NATURAL_DISTRIBUTION_BALANCED")

    def test_training_readiness_gate_phase55(self):
        gate_data = check_training_readiness_gate(phase_dir=PHASE55_DIR)
        self.assertEqual(gate_data["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate_data["phase_verdict"], "PHASE_55_PUBLIC_BETA_LIVE")
        self.assertFalse(gate_data["training_executed"])
        self.assertTrue(gate_data["automatic_training_blocked"])

if __name__ == "__main__":
    unittest.main()
