import os
import sys
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model"] == "collision-10m"
    assert "device" in data

def test_models():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "collision-10m"

def test_generate_valid():
    payload = {
        "model": "collision-10m",
        "prompt": "What is machine learning?",
        "max_tokens": 15,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert data["model"] == "collision-10m"
    assert "text" in data
    assert len(data["text"]) > 0
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["performance"]["latency_ms"] > 0.0

def test_generate_empty_prompt():
    payload = {
        "model": "collision-10m",
        "prompt": "",
        "max_tokens": 15
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 422  # validation error from pydantic (min_length=1)

def test_generate_invalid_model():
    payload = {
        "model": "collision-50m",
        "prompt": "What is gravity?",
        "max_tokens": 15
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 400
    assert "not supported" in res.json()["detail"]

def test_generate_invalid_temp():
    payload = {
        "model": "collision-10m",
        "prompt": "What is gravity?",
        "max_tokens": 15,
        "temperature": 0.0  # must be gt 0.0
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 422

def test_generate_invalid_top_p():
    payload = {
        "model": "collision-10m",
        "prompt": "What is gravity?",
        "max_tokens": 15,
        "top_p": 1.5  # must be le 1.0
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 422

def test_generate_oversized_input():
    payload = {
        "model": "collision-10m",
        "prompt": "word " * 300,  # exceeds 256 context len
        "max_tokens": 15
    }
    res = client.post("/v1/generate", json=payload)
    assert res.status_code == 400
    assert "exceeds context limit" in res.json()["detail"]

if __name__ == "__main__":
    print("Running API endpoint tests...")
    test_health()
    print("Health check endpoint test passed.")
    test_models()
    print("Models endpoint test passed.")
    test_generate_valid()
    print("Generate completion endpoint test passed.")
    test_generate_empty_prompt()
    print("Empty prompt validation test passed.")
    test_generate_invalid_model()
    print("Invalid model validation test passed.")
    test_generate_invalid_temp()
    print("Invalid temperature validation test passed.")
    test_generate_invalid_top_p()
    print("Invalid top_p validation test passed.")
    test_generate_oversized_input()
    print("Oversized prompt validation test passed.")
    print("All API tests passed successfully!")
