import unittest
import os
import sys
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records
from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE58_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase58")

class TestPhase58Pipeline(unittest.TestCase):
    def test_production_checkpoint_integrity_phase58(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path), "production model.pt missing")
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(file_hash, "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_privacy_filtering_and_credentials_phase58(self):
        records = [
            {"prompt": "API key col_12345 secret test", "response": "Valid response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Password reset passwd=secret", "response": "Valid response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Clean safe prompt for testing", "response": "Clean safe response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 2)

    def test_diversity_report_metrics_phase58(self):
        status = generate_data_status_reports(phase_dir=PHASE58_DIR)
        self.assertIn("data_diversity_status", status)
        self.assertIn("domain_percentage_distribution", status)
        self.assertIn("zero_record_domains", status)
        self.assertIn("sub_5_record_domains", status)
        self.assertEqual(status["data_diversity_status"], "HIGHLY_CONCENTRATED")

    def test_training_readiness_gate_phase58(self):
        gate = check_training_readiness_gate(phase_dir=PHASE58_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_58_DATA_COLLECTION_ACTIVE")
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

if __name__ == "__main__":
    unittest.main()
