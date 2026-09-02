import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path
os.environ["COLLISION_DB_PATH"] = "collision_test_rate.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer, create_api_key
from api.limiter import clear_rate_limits

class TestAPIRateLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set rate limit env right before initializing
        os.environ["COLLISION_RATE_LIMIT"] = "3"
        clear_rate_limits()
        if os.path.exists("collision_test_rate.db"):
            os.remove("collision_test_rate.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_rate.db"):
            os.remove("collision_test_rate.db")

    def test_rate_limiting(self):
        # Ensure correct env variable inside the test execution
        os.environ["COLLISION_RATE_LIMIT"] = "3"
        clear_rate_limits()
        
        # Register developer & key
        dev_id = create_developer("test_rate@example.com")
        raw_key, key_id = create_api_key(dev_id)
        
        payload = {
            "model": "collision-10m",
            "prompt": "Stars are",
            "max_tokens": 5
        }
        headers = {"Authorization": f"Bearer {raw_key}"}
        
        # Send 3 successful requests (should pass)
        for i in range(3):
            res = self.client.post("/v1/generate", json=payload, headers=headers)
            self.assertEqual(res.status_code, 200, f"Request {i+1} failed but should pass")
            
        # The 4th request must be rate limited (HTTP 429)
        res_limited = self.client.post("/v1/generate", json=payload, headers=headers)
        self.assertEqual(res_limited.status_code, 429)
        self.assertEqual(res_limited.json()["error"]["type"], "rate_limit_error")
        self.assertIn("Retry-After", res_limited.headers)

if __name__ == "__main__":
    unittest.main()
