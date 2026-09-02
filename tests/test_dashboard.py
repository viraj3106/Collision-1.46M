import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path
os.environ["COLLISION_DB_PATH"] = "collision_test_dash.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer_with_password, create_session, create_api_key

class TestDeveloperDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_dash.db"):
            os.remove("collision_test_dash.db")
        init_db()
        cls.client = TestClient(app)
        
        # Register dev
        cls.dev_id = create_developer_with_password("dashdev@example.com", "password123")
        cls.session_token = create_session(cls.dev_id)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_dash.db"):
            os.remove("collision_test_dash.db")

    def test_dashboard_stats_endpoint(self):
        # Request usage statistics securely
        res = self.client.get(
            f"/v1/developers/{self.dev_id}/usage",
            headers={"Authorization": f"Bearer {self.session_token}"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        
        # Verify stats default to zero for a new developer
        self.assertEqual(data["total_requests"], 0)
        self.assertEqual(data["total_prompt_tokens"], 0)
        self.assertEqual(data["total_completion_tokens"], 0)
        self.assertEqual(data["total_tokens"], 0)
        self.assertEqual(data["avg_latency_ms"], 0.0)

if __name__ == "__main__":
    unittest.main()
