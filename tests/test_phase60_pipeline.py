import unittest
import os
import sys
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records, process_data_pipeline
from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE60_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase60")

class TestPhase60Pipeline(unittest.TestCase):
    def test_01_existing_data_schema_valid(self):
        stats = process_data_pipeline()
        self.assertIn("cleaned_file", stats)
        self.assertIn("rejected_file", stats)

    def test_02_clean_records_counted_correctly(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertEqual(status["clean_records"], 7)
        self.assertEqual(status["current_clean_records"], 7)

    def test_03_rejected_records_counted_correctly(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertEqual(status["rejected_records"], 5)

    def test_04_consent_validation_works(self):
        records = [
            {"prompt": "Test prompt 1", "response": "Response 1", "rating": "thumbs_up", "consent": True},
            {"prompt": "Test prompt 2", "response": "Response 2", "rating": "thumbs_up", "consent": False}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_05_privacy_validation_works(self):
        records = [
            {"prompt": "API key col_12345 secret test", "response": "Valid response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Clean prompt text", "response": "Clean response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_06_duplicate_detection_works(self):
        records = [
            {"prompt": "Identical prompt", "response": "Identical response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Identical prompt", "response": "Identical response", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_07_domain_distribution_calculated_correctly(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertIn("domain_distribution", status)
        self.assertIn("General Knowledge", status["domain_distribution"])
        self.assertEqual(status["domain_distribution"]["General Knowledge"], 7)

    def test_08_conversation_type_distribution_calculated_correctly(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertIn("conversation_type_distribution", status)
        self.assertIn("factual Q&A", status["conversation_type_distribution"])

    def test_09_zero_record_categories_detected(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertIn("zero_record_domains", status)
        self.assertIn("zero_record_conversation_types", status)
        self.assertEqual(len(status["zero_record_domains"]), 10)

    def test_10_concentration_warnings_work(self):
        status = generate_data_status_reports(phase_dir=PHASE60_DIR)
        self.assertIn("concentration_warnings", status)
        self.assertTrue(len(status["concentration_warnings"]) > 0)

    def test_11_readiness_gate_authoritative(self):
        gate = check_training_readiness_gate(phase_dir=PHASE60_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_60_DATA_COLLECTION_ACTIVE")

    def test_12_training_not_executed(self):
        gate = check_training_readiness_gate(phase_dir=PHASE60_DIR)
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

    def test_13_production_checkpoint_untouched(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path), "production model.pt missing")
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(file_hash, "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

if __name__ == "__main__":
    unittest.main()
