import os
import sys
import unittest
import hashlib
from fastapi.testclient import TestClient

# Set test database path before importing app
os.environ["COLLISION_DB_PATH"] = "collision_test_smoke.db"
os.environ["COLLISION_RATE_LIMIT"] = "100"
os.environ["ADMIN_SECRET"] = "supersecretadmintoken"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from api.database import init_db
from api.dependencies import get_inference_engine

class TestProductionDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("collision_test_smoke.db"):
            os.remove("collision_test_smoke.db")
        init_db()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("collision_test_smoke.db"):
            os.remove("collision_test_smoke.db")

    def test_ready_health_endpoints(self):
        # Ready probe checks database and model loaded states
        res_ready = self.client.get("/ready")
        self.assertEqual(res_ready.status_code, 200)
        self.assertEqual(res_ready.json()["status"], "ready")

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

    def test_model_parameters_and_checksum(self):
        engine = get_inference_engine()
        
        # Verify model parameter count equals exactly 10,282,304
        param_count = sum(p.numel() for p in engine.model.parameters())
        self.assertEqual(param_count, 10282304)
        
        # Verify checkpoint checksum match
        model_pt_path = os.path.join(engine.model_dir, "model.pt")
        sha = hashlib.sha256()
        with open(model_pt_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        self.assertEqual(
            sha.hexdigest(),
            "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
        )

    def test_admin_route_protection(self):
        # Case A: Request without X-Admin-Token
        res_unauth = self.client.post(
            "/v1/developers",
            json={"email": "unauth@example.com"}
        )
        self.assertEqual(res_unauth.status_code, 401)
        self.assertEqual(res_unauth.json()["error"]["type"], "authentication_error")
        
        # Case B: Request with invalid token
        res_invalid = self.client.post(
            "/v1/developers",
            json={"email": "unauth@example.com"},
            headers={"X-Admin-Token": "badtoken"}
        )
        self.assertEqual(res_invalid.status_code, 401)
        
        # Case C: Request with valid token
        res_valid = self.client.post(
            "/v1/developers",
            json={"email": "newadminuser@example.com"},
            headers={"X-Admin-Token": "supersecretadmintoken"}
        )
        self.assertEqual(res_valid.status_code, 200)

if __name__ == "__main__":
    unittest.main()
