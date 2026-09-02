import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing app/db
os.environ["COLLISION_DB_PATH"] = "collision_test_iso.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import (
    init_db, 
    create_developer_with_password, 
    create_session, 
    create_api_key
)
from api.limiter import clear_rate_limits

class TestDeveloperIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_iso.db"):
            os.remove("collision_test_iso.db")
        init_db()
        cls.client = TestClient(app)
        
        # 1. Register Developer A and generate session/keys
        cls.dev_a_id = create_developer_with_password("deva@example.com", "passwordA123")
        cls.session_a = create_session(cls.dev_a_id)
        cls.key_a_raw, cls.key_a_id = create_api_key(cls.dev_a_id)
        
        # 2. Register Developer B and generate session/keys
        cls.dev_b_id = create_developer_with_password("devb@example.com", "passwordB123")
        cls.session_b = create_session(cls.dev_b_id)
        cls.key_b_raw, cls.key_b_id = create_api_key(cls.dev_b_id)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_iso.db"):
            os.remove("collision_test_iso.db")

    def test_idor_key_generation(self):
        # Developer A attempts to generate a key for Developer B (dev_b_id)
        res = self.client.post(
            "/v1/keys",
            json={"developer_id": self.dev_b_id},
            headers={"Authorization": f"Bearer {self.session_a}"}
        )
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["error"]["type"], "authorization_error")

    def test_idor_key_listing(self):
        # Developer A attempts to list Developer B's keys
        res = self.client.get(
            f"/v1/developers/{self.dev_b_id}/keys",
            headers={"Authorization": f"Bearer {self.session_a}"}
        )
        self.assertEqual(res.status_code, 403)

    def test_idor_key_revocation(self):
        # Developer A attempts to revoke Developer B's key
        res = self.client.post(
            f"/v1/keys/{self.key_b_id}/revoke",
            headers={"Authorization": f"Bearer {self.session_a}"}
        )
        self.assertEqual(res.status_code, 403)

    def test_idor_usage_access(self):
        # Developer A attempts to view Developer B's usage stats
        res = self.client.get(
            f"/v1/developers/{self.dev_b_id}/usage",
            headers={"Authorization": f"Bearer {self.session_a}"}
        )
        self.assertEqual(res.status_code, 403)

if __name__ == "__main__":
    unittest.main()
