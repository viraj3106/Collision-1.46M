import os
import sys
import time
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing app/db
os.environ["COLLISION_DB_PATH"] = "collision_test_sess.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, get_db_connection

class TestDeveloperSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_sess.db"):
            os.remove("collision_test_sess.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_sess.db"):
            os.remove("collision_test_sess.db")

    def test_signup_login_logout_flow(self):
        # 1. Sign up
        res_signup = self.client.post(
            "/v1/auth/signup",
            json={"email": "newdev@example.com", "password": "supersecure123"}
        )
        self.assertEqual(res_signup.status_code, 200)
        self.assertEqual(res_signup.json()["email"], "newdev@example.com")
        
        # 2. Login
        res_login = self.client.post(
            "/v1/auth/login",
            json={"email": "newdev@example.com", "password": "supersecure123"}
        )
        self.assertEqual(res_login.status_code, 200)
        session_token = res_login.json()["session_token"]
        self.assertTrue(session_token.startswith("sess_"))
        
        # 3. Access protected list endpoint using session
        dev_id = res_login.json()["developer_id"]
        res_keys = self.client.get(
            f"/v1/developers/{dev_id}/keys",
            headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res_keys.status_code, 200)
        
        # 4. Logout
        res_logout = self.client.post(
            "/v1/auth/logout",
            headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res_logout.status_code, 200)
        
        # 5. Accessing keys after logout should fail (Unauthorized)
        res_keys_after = self.client.get(
            f"/v1/developers/{dev_id}/keys",
            headers={"Authorization": f"Bearer {session_token}"}
        )
        self.assertEqual(res_keys_after.status_code, 401)
        self.assertEqual(res_keys_after.json()["error"]["type"], "authentication_error")

if __name__ == "__main__":
    unittest.main()
