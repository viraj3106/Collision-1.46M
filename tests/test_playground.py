import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing app/db
os.environ["COLLISION_DB_PATH"] = "collision_test_playground.db"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, get_db_connection

class TestPlayground(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_playground.db"):
            os.remove("collision_test_playground.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_playground.db"):
            os.remove("collision_test_playground.db")

    def test_playground_flow(self):
        # 1. Test unauthenticated playground request is rejected
        res = self.client.post(
            "/v1/playground/generate",
            json={"prompt": "Artificial intelligence is", "max_tokens": 10}
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"]["type"], "authentication_error")

        # 2. Register a user and log in to get session token
        res_signup = self.client.post(
            "/v1/auth/signup",
            json={"email": "playdev@example.com", "password": "supersecure123"}
        )
        self.assertEqual(res_signup.status_code, 200)

        res_login = self.client.post(
            "/v1/auth/login",
            json={"email": "playdev@example.com", "password": "supersecure123"}
        )
        self.assertEqual(res_login.status_code, 200)
        session_token = res_login.json()["session_token"]

        # 3. Test authenticated generation succeeds
        res_gen = self.client.post(
            "/v1/playground/generate",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"prompt": "Artificial intelligence is", "max_tokens": 10, "temperature": 0.7}
        )
        self.assertEqual(res_gen.status_code, 200)
        json_data = res_gen.json()
        self.assertIn("text", json_data)
        self.assertIn("usage", json_data)
        self.assertIn("performance", json_data)
        
        # 4. Test invalid generation parameters are rejected (e.g. temperature <= 0.0)
        res_gen_invalid = self.client.post(
            "/v1/playground/generate",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"prompt": "Artificial intelligence is", "max_tokens": 10, "temperature": 0.0}
        )
        self.assertEqual(res_gen_invalid.status_code, 422) # Validation error
        self.assertEqual(res_gen_invalid.json()["error"]["type"], "validation_error")

if __name__ == "__main__":
    unittest.main()
