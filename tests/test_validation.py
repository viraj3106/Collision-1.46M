import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path
os.environ["COLLISION_DB_PATH"] = "collision_test_val.db"
os.environ["COLLISION_RATE_LIMIT"] = "100"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer, create_api_key

class TestAPIValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_val.db"):
            os.remove("collision_test_val.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_val.db"):
            os.remove("collision_test_val.db")

    def test_payload_validations(self):
        # Register developer & key
        dev_id = create_developer("test_val@example.com")
        raw_key, key_id = create_api_key(dev_id)
        headers = {"Authorization": f"Bearer {raw_key}"}
        
        # Case A: Empty prompt validation
        payload_empty = {"model": "collision-10m", "prompt": "", "max_tokens": 10}
        res_empty = self.client.post("/v1/generate", json=payload_empty, headers=headers)
        self.assertEqual(res_empty.status_code, 422)
        self.assertEqual(res_empty.json()["error"]["type"], "validation_error")
        
        # Case B: Invalid model selection
        payload_model = {"model": "collision-50m", "prompt": "Artificial intelligence is", "max_tokens": 10}
        res_model = self.client.post("/v1/generate", json=payload_model, headers=headers)
        self.assertEqual(res_model.status_code, 400)
        self.assertIn("not supported", res_model.json()["error"]["message"])
        
        # Case C: Invalid temperature bounds (temperature must be gt 0.0)
        payload_temp = {"model": "collision-10m", "prompt": "Artificial intelligence is", "temperature": -0.5}
        res_temp = self.client.post("/v1/generate", json=payload_temp, headers=headers)
        self.assertEqual(res_temp.status_code, 422)
        self.assertIn("temperature", res_temp.json()["error"]["message"])
        
        # Case D: Oversized context length (exceeds 256 tokens)
        payload_oversized = {"model": "collision-10m", "prompt": "word " * 300, "max_tokens": 10}
        res_oversized = self.client.post("/v1/generate", json=payload_oversized, headers=headers)
        self.assertEqual(res_oversized.status_code, 413)
        self.assertIn("exceeds max context limit", res_oversized.json()["error"]["message"])

if __name__ == "__main__":
    unittest.main()
