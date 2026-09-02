import os
import sys
import unittest
from fastapi.testclient import TestClient

# Set test database path before importing app/db
TEST_DB_PATH = "collision_test_prod_api.db"
os.environ["COLLISION_DB_PATH"] = TEST_DB_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db, create_developer_with_password, create_api_key

class TestProductionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        init_db()
        cls.client = TestClient(app)
        
        # Setup developer and API key for tests
        cls.dev_id = create_developer_with_password("prod_test_dev@example.com", "password12345")
        cls.raw_key, cls.key_id = create_api_key(cls.dev_id)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "collision-10m")
        self.assertIn("device", data)

    def test_readiness_endpoint(self):
        res = self.client.get("/ready")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertIn("checks", data)
        self.assertEqual(data["checks"].get("database"), "ok")
        self.assertEqual(data["checks"].get("model"), "ok")

    def test_model_listing_endpoint(self):
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("data", data)
        self.assertGreaterEqual(len(data["data"]), 1)
        model = data["data"][0]
        self.assertEqual(model["id"], "collision-10m")
        self.assertEqual(model["parameter_count"], 10282304)

    def test_authenticated_generation(self):
        res = self.client.post(
            "/v1/generate",
            headers={"Authorization": f"Bearer {self.raw_key}"},
            json={"prompt": "Machine learning is", "max_tokens": 10, "temperature": 0.7}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model"], "collision-10m")
        self.assertIn("text", data)
        self.assertIn("usage", data)
        self.assertIn("performance", data)
        self.assertIn("X-Request-ID", res.headers)

    def test_invalid_api_key(self):
        res = self.client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer col_invalid_key_12345"},
            json={"prompt": "Test prompt", "max_tokens": 5}
        )
        self.assertEqual(res.status_code, 401)
        data = res.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "authentication_error")
        self.assertNotIn("Traceback", res.text)

    def test_invalid_request_validation(self):
        # Invalid temperature (must be > 0.0)
        res = self.client.post(
            "/v1/generate",
            headers={"Authorization": f"Bearer {self.raw_key}"},
            json={"prompt": "Test prompt", "max_tokens": 5, "temperature": -1.0}
        )
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "validation_error")

    def test_context_length_enforcement(self):
        # Oversized prompt (> 256 tokens)
        long_prompt = "word " * 300
        res = self.client.post(
            "/v1/generate",
            headers={"Authorization": f"Bearer {self.raw_key}"},
            json={"prompt": long_prompt, "max_tokens": 10}
        )
        self.assertEqual(res.status_code, 413)
        data = res.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "validation_error")

    def test_request_id_and_safe_error_responses(self):
        custom_req_id = "req_custom_test_123"
        res = self.client.post(
            "/v1/generate",
            headers={
                "Authorization": "Bearer col_bad_key",
                "X-Request-ID": custom_req_id
            },
            json={"prompt": "Hello"}
        )
        self.assertEqual(res.headers.get("X-Request-ID"), custom_req_id)
        data = res.json()
        self.assertEqual(data["error"]["request_id"], custom_req_id)
        # Ensure filesystem path or raw stack traces are not leaked
        self.assertNotIn("C:\\", res.text)
        self.assertNotIn("Traceback", res.text)

    def test_e2e_production_smoke_test(self):
        # 1. Check health and readiness
        res_h = self.client.get("/health")
        self.assertEqual(res_h.status_code, 200)
        res_r = self.client.get("/ready")
        self.assertEqual(res_r.status_code, 200)

        # 2. Complete feedback submission flow
        res_fb = self.client.post(
            "/v1/feedback",
            json={
                "user_id": "smoketest_user_1",
                "prompt": "Explain neural networks",
                "model": "collision-10m",
                "response": "Neural networks are computational models.",
                "rating": "thumbs_up",
                "feedback": "Great response",
                "category": "explanation",
                "consent": True
            }
        )
        self.assertEqual(res_fb.status_code, 200)
        self.assertEqual(res_fb.json()["status"], "success")

if __name__ == "__main__":
    unittest.main()
