import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing app/db components
os.environ["COLLISION_DB_PATH"] = "collision_test_auth.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer, create_api_key, revoke_api_key
from api.limiter import clear_rate_limits

class TestAPIAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set rate limit env right before initializing
        os.environ["COLLISION_RATE_LIMIT"] = "100"
        clear_rate_limits()
        # Remove existing test DB if present
        if os.path.exists("collision_test_auth.db"):
            os.remove("collision_test_auth.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_auth.db"):
            os.remove("collision_test_auth.db")

    def test_generate_auth_endpoints(self):
        os.environ["COLLISION_RATE_LIMIT"] = "100"
        clear_rate_limits()
        
        # Register dev & key
        dev_id = create_developer("test_auth@example.com")
        raw_key, key_id = create_api_key(dev_id)
        
        # Payload for completion
        payload = {
            "model": "collision-10m",
            "prompt": "Artificial intelligence is",
            "max_tokens": 10
        }
        
        # Case A: Missing Authorization Header
        res_missing = self.client.post("/v1/generate", json=payload)
        self.assertEqual(res_missing.status_code, 401)
        self.assertEqual(res_missing.json()["error"]["type"], "authentication_error")
        
        # Case B: Malformed Authorization Header
        res_malformed = self.client.post(
            "/v1/generate", 
            json=payload,
            headers={"Authorization": f"BearerInvalid {raw_key}"}
        )
        self.assertEqual(res_malformed.status_code, 401)
        self.assertIn("Bearer", res_malformed.json()["error"]["message"])
        
        # Case C: Invalid API Key
        res_invalid = self.client.post(
            "/v1/generate", 
            json=payload,
            headers={"Authorization": "Bearer col_invalidkey12345"}
        )
        self.assertEqual(res_invalid.status_code, 401)
        self.assertIn("Invalid API key", res_invalid.json()["error"]["message"])
        
        # Case D: Valid API Key
        res_valid = self.client.post(
            "/v1/generate", 
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"}
        )
        self.assertEqual(res_valid.status_code, 200)
        self.assertIn("text", res_valid.json())
        self.assertIn("X-Request-ID", res_valid.headers)
        
        # Case E: Revoked API Key
        revoke_api_key(key_id)
        res_revoked = self.client.post(
            "/v1/generate", 
            json=payload,
            headers={"Authorization": f"Bearer {raw_key}"}
        )
        self.assertEqual(res_revoked.status_code, 401)
        self.assertIn("revoked", res_revoked.json()["error"]["message"])

if __name__ == "__main__":
    unittest.main()
