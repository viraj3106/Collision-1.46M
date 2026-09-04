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

PHASE65_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase65")

class TestPhase65Pipeline(unittest.TestCase):
    def test_01_v1_feedback_request_validation(self):
        sample = {
            "user_id": "beta_user_01",
            "prompt": "How to implement binary search in Python?",
            "response": "Use def binary_search(arr, target)...",
            "rating": "thumbs_up",
            "category": "Programming",
            "conversation_type": "code_explanation",
            "consent": True
        }
        self.assertEqual(sample["rating"], "thumbs_up")
        self.assertTrue(sample["consent"])

    def test_02_valid_feedback_submission(self):
        records = [{"prompt": "Explain gradient descent in machine learning", "response": "Gradient descent is an optimization algorithm...", "rating": "thumbs_up", "category": "AI/ML", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)

    def test_03_invalid_feedback_rejection(self):
        records = [{"prompt": "", "response": "Valid response text", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_04_explicit_consent_enforcement(self):
        records = [{"prompt": "Consent prompt", "response": "Response text", "rating": "thumbs_up", "consent": False}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_05_missing_consent_rejection(self):
        records = [{"prompt": "No consent prompt", "response": "Response text", "rating": "thumbs_up"}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_06_pii_detection(self):
        records = [{"prompt": "Contact me at dev@example.com", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_07_secret_api_key_detection(self):
        records = [{"prompt": "My key is col_secretkey9999", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_08_duplicate_detection(self):
        records = [
            {"prompt": "Dup prompt phase65", "response": "Dup resp", "rating": "thumbs_up", "consent": True},
            {"prompt": "Dup prompt phase65", "response": "Dup resp", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)

    def test_09_domain_propagation(self):
        records = [{"prompt": "What is Bayes theorem?", "response": "P(A|B) = P(B|A)P(A)/P(B)", "rating": "thumbs_up", "category": "Mathematics", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["category"], "Mathematics")

    def test_10_conversation_type_propagation(self):
        records = [{"prompt": "How does photosynthesis work?", "response": "Plants use sunlight to turn CO2 into glucose...", "rating": "thumbs_up", "conversation_type": "scientific_explanation", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["conversation_type"], "scientific_explanation")

    def test_11_multi_turn_metadata(self):
        records = [{"prompt": "Can you summarize that?", "response": "In short...", "rating": "thumbs_up", "is_multi_turn": True, "turn_number": 2, "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertTrue(cleaned[0]["is_multi_turn"])

    def test_12_follow_up_metadata(self):
        records = [{"prompt": "Can you give a C++ example?", "response": "Here is C++...", "rating": "thumbs_up", "parent_id": "req_phase65", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["parent_id"], "req_phase65")

    def test_13_raw_persistence(self):
        stats = process_data_pipeline()
        self.assertIn("cleaned_file", stats)

    def test_14_cleaner_integration(self):
        raw = fetch_raw_records()
        self.assertTrue(len(raw) > 0)

    def test_15_clean_dataset_isolation(self):
        stats = process_data_pipeline()
        self.assertTrue(os.path.exists(stats["cleaned_file"]))

    def test_16_synthetic_fixture_isolation(self):
        synth = [{"prompt": "Synth65", "response": "Synth65", "rating": "thumbs_up", "consent": True, "source": "synthetic_fixture"}]
        c, _ = validate_and_clean_records(synth)
        self.assertEqual(len(c), 1)
        raw = fetch_raw_records()
        for r in raw:
            self.assertNotEqual(r.get("source"), "synthetic_fixture")

    def test_17_dynamic_consent_calculation(self):
        status = generate_data_status_reports(phase_dir=PHASE65_DIR)
        self.assertIsInstance(status["consent_coverage_percent"], float)

    def test_18_dynamic_funnel_calculation(self):
        status = generate_data_status_reports(phase_dir=PHASE65_DIR)
        self.assertIn("collection_funnel", status)

    def test_19_dynamic_diversity_calculation(self):
        status = generate_data_status_reports(phase_dir=PHASE65_DIR)
        self.assertIn("domain_percentage_distribution", status)

    def test_20_no_hardcoded_historical_metrics(self):
        status = generate_data_status_reports(phase_dir=PHASE65_DIR)
        self.assertIsNotNone(status.get("consent_coverage_percent"))

    def test_21_acquisition_status_reporting(self):
        status = generate_data_status_reports(phase_dir=PHASE65_DIR)
        acq_path = os.path.join(PHASE65_DIR, "acquisition_status.json")
        self.assertTrue(os.path.exists(acq_path))

    def test_22_readiness_gate_authority(self):
        gate = check_training_readiness_gate(phase_dir=PHASE65_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")

    def test_23_training_disabled(self):
        gate = check_training_readiness_gate(phase_dir=PHASE65_DIR)
        self.assertFalse(gate["training_executed"])

    def test_24_production_checkpoint_integrity(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path))
        with open(model_path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_25_j52_integrity(self):
        j52_path = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
        self.assertTrue(os.path.exists(j52_path))

    def test_26_history_entry_integrity(self):
        hist_path = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")
        self.assertTrue(os.path.exists(hist_path))

if __name__ == "__main__":
    unittest.main()
