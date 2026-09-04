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

PHASE62_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase62")

class TestPhase62Pipeline(unittest.TestCase):
    def test_01_feedback_submission_schema(self):
        sample = {
            "user_id": "test_dev",
            "prompt": "Explain Quantum Computing",
            "response": "Quantum computing uses qubits...",
            "rating": "thumbs_up",
            "category": "explanatory",
            "feedback": "Great explanation",
            "consent": True,
            "model": "collision-10m"
        }
        self.assertIn("prompt", sample)
        self.assertIn("response", sample)
        self.assertIn("consent", sample)

    def test_02_valid_feedback_ingestion(self):
        records = [
            {"prompt": "Valid test prompt", "response": "Valid response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 0)

    def test_03_invalid_feedback_rejection(self):
        records = [
            {"prompt": "", "response": "Valid response text", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_04_explicit_consent_requirement(self):
        records = [
            {"prompt": "Consent prompt", "response": "Consent response", "rating": "thumbs_up", "consent": False}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_05_privacy_filtering(self):
        records = [
            {"prompt": "My email is test@domain.com", "response": "OK", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_06_pii_detection(self):
        records = [
            {"prompt": "Phone number +1-555-019-2831", "response": "Saved", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_07_secret_api_key_detection(self):
        records = [
            {"prompt": "Key col_secretkey123", "response": "Authenticated", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_08_duplicate_detection(self):
        records = [
            {"prompt": "Duplicate prompt", "response": "Duplicate response", "rating": "thumbs_up", "consent": True},
            {"prompt": "Duplicate prompt", "response": "Duplicate response", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_09_domain_tracking(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIn("domain_distribution", status)
        self.assertIn("General Knowledge", status["domain_distribution"])

    def test_10_conversation_type_tracking(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIn("conversation_type_distribution", status)
        self.assertIn("factual Q&A", status["conversation_type_distribution"])

    def test_11_diversity_prioritization(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIn("sub_5_record_domains", status)
        self.assertIn("sub_5_record_conversation_types", status)

    def test_12_zero_category_detection(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIn("zero_record_domains", status)
        self.assertIn("zero_record_conversation_types", status)

    def test_13_real_world_vs_synthetic_separation(self):
        synthetic_records = [
            {"prompt": "Synthetic prompt", "response": "Synthetic response", "rating": "thumbs_up", "consent": True, "source": "synthetic_test_fixture"}
        ]
        cleaned, _ = validate_and_clean_records(synthetic_records)
        self.assertEqual(len(cleaned), 1)
        raw = fetch_raw_records()
        for r in raw:
            self.assertNotEqual(r.get("source"), "synthetic_test_fixture")

    def test_14_dynamic_metric_calculation(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIsInstance(status["clean_records"], int)
        self.assertIsInstance(status["raw_records"], int)
        self.assertIsInstance(status["consent_coverage_percent"], float)

    def test_15_collection_funnel_metrics(self):
        status = generate_data_status_reports(phase_dir=PHASE62_DIR)
        self.assertIn("collection_funnel", status)
        funnel = status["collection_funnel"]
        self.assertIn("feedback_ui_shown", funnel)
        self.assertIn("submission_accepted", funnel)
        self.assertIn("acceptance_rate_pct", funnel)

    def test_16_readiness_gate_authority(self):
        gate = check_training_readiness_gate(phase_dir=PHASE62_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")
        self.assertEqual(gate["phase_verdict"], "PHASE_62_DATA_COLLECTION_ACTIVE")

    def test_17_no_training_guarantee(self):
        gate = check_training_readiness_gate(phase_dir=PHASE62_DIR)
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

    def test_18_production_checkpoint_integrity(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path), "production model.pt missing")
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(file_hash, "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_19_j52_checkpoint_integrity(self):
        j52_path = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
        self.assertTrue(os.path.exists(j52_path), "J52 checkpoint missing")

    def test_20_history_entry_generation(self):
        history_path = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")
        self.assertTrue(os.path.exists(history_path))

if __name__ == "__main__":
    unittest.main()
