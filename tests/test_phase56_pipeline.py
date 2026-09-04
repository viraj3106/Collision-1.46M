import unittest
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records
from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate
from data.monitor_real_world import print_real_world_data_status

PHASE56_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase56")

class TestPhase56Pipeline(unittest.TestCase):
    def test_consent_enforcement_phase56(self):
        records = [
            {"prompt": "Valid prompt 1", "response": "Valid response 1", "rating": "thumbs_up", "consent": True},
            {"prompt": "Valid prompt 2", "response": "Valid response 2", "rating": "thumbs_up", "consent": False}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("Missing or unverified consent", rejected[0]["rejection_reasons"][0])

    def test_model_version_attribution(self):
        records = [
            {"prompt": "Test prompt text", "response": "Test response text", "rating": "thumbs_up", "consent": True, "model": "collision-10m"},
            {"prompt": "Another prompt text", "response": "Another response text", "rating": "thumbs_up", "consent": True, "model": "J52"}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["source"], "collision-10m")
        self.assertEqual(cleaned[1]["source"], "J52")

    def test_privacy_filtering_and_duplicate_detection(self):
        records = [
            {"prompt": "API key col_secret123", "response": "Valid response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Unique prompt test", "response": "Unique response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Unique prompt test", "response": "Unique response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 2)

    def test_readiness_threshold_and_no_training(self):
        gate = check_training_readiness_gate(phase_dir=PHASE56_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_56_DATA_COLLECTION_ACTIVE")
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

    def test_monitor_script_execution(self):
        status, gate, output_text = print_real_world_data_status(phase_dir=PHASE56_DIR)
        self.assertTrue("COLLISION PUBLIC BETA" in output_text or "REAL-WORLD DATA STATUS" in output_text)
        self.assertIn("Clean", output_text)
        self.assertTrue("Training:" in output_text or "Training ready:" in output_text)

if __name__ == "__main__":
    unittest.main()
