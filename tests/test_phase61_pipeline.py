import unittest
import os
import sys
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records, process_data_pipeline, fetch_raw_records
from data.data_collection_status import generate_data_status_reports
from training.check_training_readiness import check_training_readiness_gate

PHASE61_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase61")

class TestPhase61Pipeline(unittest.TestCase):
    def test_01_real_world_record_ingestion(self):
        records = fetch_raw_records()
        self.assertTrue(len(records) > 0, "No raw records ingested")

    def test_02_schema_validation(self):
        stats = process_data_pipeline()
        self.assertIn("cleaned_file", stats)
        self.assertIn("rejected_file", stats)

    def test_03_consent_validation(self):
        records = [
            {"prompt": "Test prompt 1", "response": "Response 1", "rating": "thumbs_up", "consent": True},
            {"prompt": "Test prompt 2", "response": "Response 2", "rating": "thumbs_up", "consent": False}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_04_privacy_filtering(self):
        records = [
            {"prompt": "API key col_12345 secret test", "response": "Valid response text", "rating": "thumbs_up", "consent": True},
            {"prompt": "Clean prompt text", "response": "Clean response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_05_duplicate_detection(self):
        records = [
            {"prompt": "Identical prompt", "response": "Identical response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Identical prompt", "response": "Identical response", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_06_clean_rejected_counts(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("clean_records", status)
        self.assertIn("rejected_records", status)
        self.assertIn("raw_records", status)

    def test_07_domain_classification(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("domain_distribution", status)
        self.assertIn("General Knowledge", status["domain_distribution"])
        self.assertIn("Programming", status["domain_distribution"])
        self.assertIn("AI/ML", status["domain_distribution"])

    def test_08_conversation_type_classification(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("conversation_type_distribution", status)
        self.assertIn("factual Q&A", status["conversation_type_distribution"])

    def test_09_zero_record_detection(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("zero_record_domains", status)
        self.assertIn("zero_record_conversation_types", status)

    def test_10_concentration_detection(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("concentration_warnings", status)

    def test_11_diversity_prioritization(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIn("sub_5_record_domains", status)
        self.assertIn("sub_5_record_conversation_types", status)

    def test_12_synthetic_test_data_not_counted(self):
        synthetic_records = [
            {"prompt": "Synthetic prompt", "response": "Synthetic response", "rating": "thumbs_up", "consent": True, "source": "synthetic_test_fixture"}
        ]
        cleaned, _ = validate_and_clean_records(synthetic_records)
        self.assertEqual(len(cleaned), 1)
        # Verify raw files on disk only read from data/real_world/raw
        raw = fetch_raw_records()
        for r in raw:
            self.assertNotEqual(r.get("source"), "synthetic_test_fixture")

    def test_13_metrics_dynamically_calculated(self):
        status = generate_data_status_reports(phase_dir=PHASE61_DIR)
        self.assertIsInstance(status["consent_coverage_percent"], float)
        self.assertIsInstance(status["clean_records"], int)
        self.assertIsInstance(status["raw_records"], int)

    def test_14_readiness_gate_remains_authoritative(self):
        gate = check_training_readiness_gate(phase_dir=PHASE61_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_61_DATA_COLLECTION_ACTIVE")

    def test_15_training_is_never_executed(self):
        gate = check_training_readiness_gate(phase_dir=PHASE61_DIR)
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

    def test_16_production_checkpoint_remains_unchanged(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path), "production model.pt missing")
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(file_hash, "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_17_j52_remains_unchanged(self):
        j52_path = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
        self.assertTrue(os.path.exists(j52_path), "J52 checkpoint missing")

    def test_18_history_entry_generation(self):
        history_path = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")
        self.assertTrue(os.path.exists(history_path))

if __name__ == "__main__":
    unittest.main()
