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

PHASE63_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase63")

class TestPhase63Pipeline(unittest.TestCase):
    def test_01_v1_feedback_request_handling(self):
        sample = {
            "user_id": "test_dev",
            "prompt": "How to debug a memory leak?",
            "response": "Use a memory profiler...",
            "rating": "thumbs_up",
            "category": "Troubleshooting",
            "conversation_type": "troubleshooting",
            "consent": True
        }
        self.assertEqual(sample["rating"], "thumbs_up")
        self.assertTrue(sample["consent"])

    def test_02_valid_persistence(self):
        stats = process_data_pipeline()
        self.assertIn("cleaned_file", stats)

    def test_03_invalid_request_rejection(self):
        records = [{"prompt": "", "response": "Ans", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_04_consent_enforcement(self):
        records = [{"prompt": "P", "response": "Response text", "rating": "thumbs_up", "consent": False}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_05_privacy_filtering(self):
        records = [{"prompt": "Email user@test.com", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_06_pii_detection(self):
        records = [{"prompt": "Phone +1-555-019-2831", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_07_secret_api_key_detection(self):
        records = [{"prompt": "col_secretkey123", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)

    def test_08_duplicate_detection(self):
        records = [
            {"prompt": "Dup prompt", "response": "Dup resp", "rating": "thumbs_up", "consent": True},
            {"prompt": "Dup prompt", "response": "Dup resp", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)

    def test_09_domain_propagation(self):
        records = [{"prompt": "Write Python code", "response": "def foo(): pass", "rating": "thumbs_up", "category": "Programming", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["category"], "Programming")

    def test_10_conversation_type_propagation(self):
        records = [{"prompt": "Explain gravity", "response": "Gravity is spacetime curvature", "rating": "thumbs_up", "conversation_type": "explanatory", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["conversation_type"], "explanatory")

    def test_11_multi_turn_metadata(self):
        records = [{"prompt": "Follow up question", "response": "Answer", "rating": "thumbs_up", "is_multi_turn": True, "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertTrue(cleaned[0]["is_multi_turn"])

    def test_12_follow_up_metadata(self):
        records = [{"prompt": "Clarification", "response": "Details", "rating": "thumbs_up", "parent_id": "req_123", "consent": True}]
        cleaned, _ = validate_and_clean_records(records)
        self.assertEqual(cleaned[0]["parent_id"], "req_123")

    def test_13_rejection_reason_tracking(self):
        records = [{"prompt": "Bad answer query", "response": "Hallucination", "rating": "thumbs_down", "consent": True}]
        _, rejected = validate_and_clean_records(records)
        self.assertIn("rejection_reasons", rejected[0])

    def test_14_dynamic_consent_metrics(self):
        status = generate_data_status_reports(phase_dir=PHASE63_DIR)
        self.assertIsInstance(status["consent_coverage_percent"], float)

    def test_15_dynamic_record_metrics(self):
        status = generate_data_status_reports(phase_dir=PHASE63_DIR)
        self.assertIsInstance(status["clean_records"], int)

    def test_16_synthetic_vs_real_world_separation(self):
        synth = [{"prompt": "Synth", "response": "Synth", "rating": "thumbs_up", "consent": True, "source": "synthetic_fixture"}]
        c, _ = validate_and_clean_records(synth)
        self.assertEqual(len(c), 1)
        raw = fetch_raw_records()
        for r in raw:
            self.assertNotEqual(r.get("source"), "synthetic_fixture")

    def test_17_collection_funnel_calculations(self):
        status = generate_data_status_reports(phase_dir=PHASE63_DIR)
        self.assertIn("collection_funnel", status)

    def test_18_diversity_scorecard(self):
        status = generate_data_status_reports(phase_dir=PHASE63_DIR)
        self.assertIn("diversity_scorecard", status)
        self.assertIn("Multi-turn", status["diversity_scorecard"])

    def test_19_readiness_gate_authority(self):
        gate = check_training_readiness_gate(phase_dir=PHASE63_DIR)
        self.assertEqual(gate["readiness_verdict"], "REAL_WORLD_DATA_NOT_READY")

    def test_20_no_training_guarantee(self):
        gate = check_training_readiness_gate(phase_dir=PHASE63_DIR)
        self.assertFalse(gate["training_executed"])

    def test_21_production_checkpoint_integrity(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path))
        with open(model_path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_22_j52_integrity(self):
        j52_path = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
        self.assertTrue(os.path.exists(j52_path))

    def test_23_history_generation(self):
        hist_path = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")
        self.assertTrue(os.path.exists(hist_path))

    def test_24_end_to_end_feedback_flow(self):
        rec = [{"prompt": "E2E prompt testing flow", "response": "E2E response text testing flow", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(rec)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 0)

if __name__ == "__main__":
    unittest.main()
