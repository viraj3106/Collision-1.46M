import unittest
import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records
from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

class TestPhase54Pipeline(unittest.TestCase):
    def test_production_integrity_values(self):
        integrity_path = os.path.join(PROJECT_ROOT, "experiments", "phase54", "production_integrity.json")
        self.assertTrue(os.path.exists(integrity_path), "production_integrity.json missing")
        with open(integrity_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["trainable_parameter_count"], 10282304)
        self.assertEqual(data["raw_state_dict_elements"], 13747520)
        self.assertEqual(data["production_sha256"], "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")
        self.assertEqual(data["parameter_count_discrepancy_verdict"], "REPORTING_CALCULATION_BUG")

    def test_privacy_filtering_credentials_and_pii(self):
        records = [
            {"prompt": "User email is dev@example.com", "response": "Valid text response", "rating": "thumbs_up", "consent": True},
            {"prompt": "My API key col_9999 is set", "response": "Valid text response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Here is sk-testkey12345", "response": "Valid text response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Call me at +1 (555) 123-4567", "response": "Valid text response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Clean safe prompt", "response": "Clean safe response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 4)

    def test_training_readiness_gate(self):
        gate_data = check_training_readiness_gate()
        self.assertIn(gate_data["verdict"], ["REAL_WORLD_DATA_NOT_READY", "REAL_WORLD_DATA_READY_FOR_SFT"])
        self.assertFalse(gate_data["training_executed"])
        self.assertEqual(gate_data["promoted_research_candidate"], "J52")

    def test_data_status_reports_generation(self):
        status = generate_data_status_reports()
        self.assertIn("current_clean_record_count", status)
        self.assertIn("target_clean_records", status)
        self.assertIn("remaining_records_required", status)
        self.assertEqual(status["target_clean_records"], 100)

if __name__ == "__main__":
    unittest.main()
