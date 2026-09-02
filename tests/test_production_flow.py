import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing main/database
os.environ["COLLISION_DB_PATH"] = "collision_test_prod_flow.db"
os.environ["COLLISION_RATE_LIMIT"] = "100"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, get_developer_usage_stats
from api.limiter import clear_rate_limits

class TestProductionFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_prod_flow.db"):
            os.remove("collision_test_prod_flow.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_prod_flow.db"):
            os.remove("collision_test_prod_flow.db")

    def test_e2e_developer_flow(self):
        clear_rate_limits()
        
        # 1. Sign up Developer A
        res_signup_a = self.client.post(
            "/v1/auth/signup",
            json={"email": "deva_prod@example.com", "password": "passwordA123"}
        )
        self.assertEqual(res_signup_a.status_code, 200)
        dev_a_id = res_signup_a.json()["id"]
        
        # 2. Log in Developer A
        res_login_a = self.client.post(
            "/v1/auth/login",
            json={"email": "deva_prod@example.com", "password": "passwordA123"}
        )
        self.assertEqual(res_login_a.status_code, 200)
        session_a = res_login_a.json()["session_token"]
        
        # 3. Create API Key for Developer A
        res_key_a = self.client.post(
            "/v1/keys",
            json={"developer_id": dev_a_id},
            headers={"Authorization": f"Bearer {session_a}"}
        )
        self.assertEqual(res_key_a.status_code, 200)
        raw_key_a = res_key_a.json()["api_key"]
        key_a_id = res_key_a.json()["id"]
        
        # 4. Generate completion with Developer A's API Key
        payload = {
            "model": "collision-10m",
            "prompt": "Stars are",
            "max_tokens": 10
        }
        res_gen = self.client.post(
            "/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key_a}"}
        )
        self.assertEqual(res_gen.status_code, 200)
        self.assertIn("text", res_gen.json())
        
        # 5. Verify usage statistics recorded in DB
        res_usage = self.client.get(
            f"/v1/developers/{dev_a_id}/usage",
            headers={"Authorization": f"Bearer {session_a}"}
        )
        self.assertEqual(res_usage.status_code, 200)
        self.assertEqual(res_usage.json()["total_requests"], 1)
        self.assertGreater(res_usage.json()["total_tokens"], 0)
        
        # 6. Sign up Developer B to test isolation
        res_signup_b = self.client.post(
            "/v1/auth/signup",
            json={"email": "devb_prod@example.com", "password": "passwordB123"}
        )
        self.assertEqual(res_signup_b.status_code, 200)
        dev_b_id = res_signup_b.json()["id"]
        
        # 7. Log in Developer B
        res_login_b = self.client.post(
            "/v1/auth/login",
            json={"email": "devb_prod@example.com", "password": "passwordB123"}
        )
        session_b = res_login_b.json()["session_token"]
        
        # 8. Verify Developer B cannot access Developer A's usage
        res_isolation_usage = self.client.get(
            f"/v1/developers/{dev_a_id}/usage",
            headers={"Authorization": f"Bearer {session_b}"}
        )
        self.assertEqual(res_isolation_usage.status_code, 403)
        
        # 9. Revoke Developer A's API Key
        res_revoke = self.client.post(
            f"/v1/keys/{key_a_id}/revoke",
            headers={"Authorization": f"Bearer {session_a}"}
        )
        self.assertEqual(res_revoke.status_code, 200)
        
        # 10. Verify generation with revoked key fails (401 Unauthorized)
        res_gen_after = self.client.post(
            "/v1/generate",
            json=payload,
            headers={"Authorization": f"Bearer {raw_key_a}"}
        )
        self.assertEqual(res_gen_after.status_code, 401)
        self.assertEqual(res_gen_after.json()["error"]["type"], "authentication_error")

if __name__ == "__main__":
    unittest.main()
