import unittest
import os
import sys
import json
import shutil

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import validate_and_clean_records, process_data_pipeline
from training.prepare_real_world_dataset import convert_real_world_to_collision_dataset

class TestRealWorldDataPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(PROJECT_ROOT, "scratch", "test_real_world")
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.cleaned_dir = os.path.join(self.test_dir, "cleaned")
        self.rejected_dir = os.path.join(self.test_dir, "rejected")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.cleaned_dir, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_missing_fields_rejection(self):
        records = [
            {"prompt": "Hello", "response": "", "rating": "thumbs_up"}, # missing response
            {"prompt": "", "response": "Hi", "rating": "thumbs_up"}, # missing prompt
            {"prompt": "Test", "response": "Ans"}, # missing rating
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 3)

    def test_duplicate_handling(self):
        records = [
            {"user_id": "u1", "prompt": "What is AI?", "response": "Artificial intelligence.", "rating": "thumbs_up"},
            {"user_id": "u2", "prompt": "What is AI?", "response": "Artificial intelligence.", "rating": "thumbs_up"},
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("Duplicate prompt-response example", rejected[0]["rejection_reasons"])

    def test_consent_handling(self):
        records = [
            {"user_id": "u1", "prompt": "P1", "response": "R1", "rating": "thumbs_up", "consent": True},
            {"user_id": "u2", "prompt": "P2", "response": "R2", "rating": "thumbs_up", "consent": False},
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn("Consent explicitly declined", rejected[0]["rejection_reasons"][0])

    def test_sensitive_credential_rejection(self):
        records = [
            {"prompt": "My API key is col_12345", "response": "OK", "rating": "thumbs_up"},
            {"prompt": "Reset password=", "response": "Done", "rating": "thumbs_up"}
        ]
        cleaned, rejected = validate_and_clean_records(records)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(len(rejected), 2)

    def test_raw_dataset_preservation_and_conversion(self):
        raw_file = os.path.join(self.raw_dir, "sample.json")
        sample_records = [
            {"user_id": "dev1", "prompt": "Explain gravity", "response": "Gravity is spacetime curvature.", "rating": "thumbs_up", "consent": True},
            {"user_id": "dev2", "prompt": "Bad question", "response": "Bad answer", "rating": "thumbs_down", "consent": True}
        ]
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(sample_records, f)

        # Run cleaning
        cleaned, rejected = validate_and_clean_records(sample_records)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(rejected), 1)

        # Verify raw file remains intact
        with open(raw_file, "r", encoding="utf-8") as f:
            persisted_raw = json.load(f)
        self.assertEqual(len(persisted_raw), 2)

        # Test dataset conversion
        cleaned_jsonl = os.path.join(self.cleaned_dir, "real_world_cleaned.jsonl")
        with open(cleaned_jsonl, "w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(json.dumps(item) + "\n")

        output_jsonl = os.path.join(self.test_dir, "real_world_formatted.jsonl")
        count = convert_real_world_to_collision_dataset(cleaned_jsonl, output_jsonl)
        self.assertEqual(count, 1)
        self.assertTrue(os.path.exists(output_jsonl))

        with open(output_jsonl, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["instruction"], "Explain gravity")
            self.assertEqual(data["response"], "Gravity is spacetime curvature.")

if __name__ == "__main__":
    unittest.main()
