import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path
os.environ["COLLISION_DB_PATH"] = "collision_test_usage.db"
os.environ["COLLISION_RATE_LIMIT"] = "100"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer, create_api_key, get_developer_usage_stats

class TestAPIUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_usage.db"):
            os.remove("collision_test_usage.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_usage.db"):
            os.remove("collision_test_usage.db")

    def test_usage_logging(self):
        # Register developer & key
        dev_id = create_developer("test_usage@example.com")
        raw_key, key_id = create_api_key(dev_id)
        
        # Check stats are initially zero
        initial_stats = get_developer_usage_stats(dev_id)
        self.assertEqual(initial_stats["total_requests"], 0)
        
        payload = {
            "model": "collision-10m",
            "prompt": "An algorithm is",
            "max_tokens": 10
        }
        headers = {"Authorization": f"Bearer {raw_key}"}
        
        # Fire requests
        res1 = self.client.post("/v1/generate", json=payload, headers=headers)
        self.assertEqual(res1.status_code, 200)
        
        res2 = self.client.post("/v1/generate", json=payload, headers=headers)
        self.assertEqual(res2.status_code, 200)
        
        # Retrieve updated stats
        stats = get_developer_usage_stats(dev_id)
        self.assertEqual(stats["total_requests"], 2)
        self.assertGreater(stats["total_prompt_tokens"], 0)
        self.assertGreater(stats["total_completion_tokens"], 0)
        self.assertEqual(stats["total_tokens"], stats["total_prompt_tokens"] + stats["total_completion_tokens"])
        self.assertGreater(stats["avg_latency_ms"], 0.0)

if __name__ == "__main__":
    unittest.main()
