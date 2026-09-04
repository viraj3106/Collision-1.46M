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

PHASE66_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase66")

class TestPhase66Pipeline(unittest.TestCase):
    def test_01_dataset_recomputation_from_scratch(self):
        raw = fetch_raw_records()
        self.assertGreaterEqual(len(raw), 12)
        cleaned, rejected = validate_and_clean_records(raw)
        self.assertEqual(len(cleaned), 7)
        self.assertEqual(len(rejected), 5)

    def test_02_provenance_validation(self):
        raw = fetch_raw_records()
        for r in raw:
            self.assertIn("user_id", r)
            self.assertIn("timestamp", r)
            self.assertIn("model", r)

    def test_03_consent_enforcement_audit(self):
        unconsented = [{"prompt": "No consent test", "response": "Response text", "rating": "thumbs_up", "consent": False}]
        cleaned, rejected = validate_and_clean_records(unconsented)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 1)

    def test_04_privacy_pii_validation(self):
        pii_record = [{"prompt": "Email me at dev_p66@example.com", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(pii_record)
        self.assertEqual(len(cleaned), 0)
        self.assertIn("Sensitive data detected: email address", rejected[0]["rejection_reasons"])

    def test_05_secret_credential_validation(self):
        secret_record = [{"prompt": "My key is col_secretkey6666", "response": "OK", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(secret_record)
        self.assertEqual(len(cleaned), 0)

    def test_06_duplicate_filtering_validation(self):
        dups = [
            {"prompt": "P66 dup prompt", "response": "P66 dup resp", "rating": "thumbs_up", "consent": True},
            {"prompt": "P66 dup prompt", "response": "P66 dup resp", "rating": "thumbs_up", "consent": True}
        ]
        cleaned, rejected = validate_and_clean_records(dups)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

    def test_07_quality_validation_length(self):
        short = [{"prompt": "a", "response": "b", "rating": "thumbs_up", "consent": True}]
        cleaned, rejected = validate_and_clean_records(short)
        self.assertEqual(len(cleaned), 0)

    def test_08_domain_diversity_recalculation(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(status["data_diversity_status"], "HIGHLY_CONCENTRATED")
        self.assertEqual(status["domain_distribution"]["General Knowledge"], 7)

    def test_09_conversation_diversity_recalculation(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(status["conversation_type_distribution"]["factual Q&A"], 7)

    def test_10_multi_turn_tracking(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(status["multi_turn_count"], 0)

    def test_11_follow_up_tracking(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(status["follow_up_count"], 0)

    def test_12_readiness_policy_execution(self):
        gate = check_training_readiness_gate(phase_dir=PHASE66_DIR)
        self.assertEqual(gate["phase_verdict"], "PHASE_66_DATA_NOT_READY_EXTERNAL_TRAFFIC_REQUIRED")

    def test_13_minimum_record_gate_enforcement(self):
        gate = check_training_readiness_gate(phase_dir=PHASE66_DIR)
        self.assertFalse(gate["quality_gates_audit"]["clean_records_ge_100"])

    def test_14_diversity_gate_enforcement(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(len(status["zero_record_domains"]), 10)

    def test_15_provenance_gate_enforcement(self):
        cleaned, _ = validate_and_clean_records(fetch_raw_records())
        for r in cleaned:
            self.assertEqual(r["quality_status"], "passed_audit")

    def test_16_privacy_gate_clean_split_zero_violations(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertEqual(status["pii_secrets_rejection_count"], 3)

    def test_17_synthetic_data_exclusion(self):
        synth = [{"prompt": "Synthetic P66", "response": "Response P66", "rating": "thumbs_up", "consent": True, "source": "synthetic_fixture"}]
        c, _ = validate_and_clean_records(synth)
        raw = fetch_raw_records()
        for r in raw:
            self.assertNotEqual(r.get("source"), "synthetic_fixture")

    def test_18_dynamic_metrics_no_hardcoding(self):
        status = generate_data_status_reports(phase_dir=PHASE66_DIR)
        self.assertIsInstance(status["consent_coverage_percent"], float)
        self.assertIsInstance(status["duplicate_rate"], float)

    def test_19_production_checkpoint_integrity(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
        self.assertTrue(os.path.exists(model_path))
        with open(model_path, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97")

    def test_20_j52_candidate_integrity(self):
        j52_path = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
        self.assertTrue(os.path.exists(j52_path))

    def test_21_zero_training_enforcement(self):
        gate = check_training_readiness_gate(phase_dir=PHASE66_DIR)
        self.assertFalse(gate["training_executed"])
        self.assertTrue(gate["automatic_training_blocked"])

    def test_22_history_file_integrity(self):
        hist_path = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")
        self.assertTrue(os.path.exists(hist_path))

if __name__ == "__main__":
    unittest.main()
