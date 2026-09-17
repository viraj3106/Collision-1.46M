"""
COLLISION Phase 99 — Production API & Engine Stabilization Tests.

Verifies:
- /health and /ready endpoints
- /v1/ask and /ask endpoints (AUTO, LOCAL, WEB, HYBRID, MODEL)
- Request validation (empty, missing, oversized, invalid mode)
- Error handling and stack trace isolation
- Provenance and claim metadata preservation
- CLI functionality (text presentation & --json schema)
- Checkpoint integrity
"""

import os
import sys
import hashlib
import json
import subprocess
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from collision.service import get_collision_service, CollisionService

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

client = TestClient(app)


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()


def test_checkpoint_integrity_phase99():
    """Verify that protected production and research checkpoints remain byte-identical."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    assert os.path.exists(c10m_path), "Production collision-10m model checkpoint missing"
    assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    assert os.path.exists(v9_path), "Research phase91_v9_10m checkpoint missing"
    assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256


def test_health_endpoint():
    """Verify /health returns 200 with expected structure."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data.get("service") == "collision"
    assert "model" in data


def test_ready_endpoint():
    """Verify /ready verifies all runtime components."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["service"] == "collision"
    checks = data["checks"]
    assert checks["model_checkpoint"] is True
    assert checks["local_index"] is True
    assert checks["web_provider"] is True


def test_ask_endpoint_local_rag():
    """Verify POST /v1/ask for local RAG query."""
    payload = {
        "question": "What is the embedding dimension in the COLLISION 10M architecture?",
        "mode": "AUTO"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ANSWERED"
    assert data["mode"] in ("LOCAL", "HYBRID")
    assert "384" in data["answer"] or "dimension" in data["answer"].lower()
    assert len(data["sources"]) > 0
    assert data["sources"][0]["source_type"] == "LOCAL"
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0


def test_ask_endpoint_web_grounding():
    """Verify POST /v1/ask for web grounded query."""
    payload = {
        "question": "What is the latest release version of PyTorch in 2025?",
        "mode": "WEB"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ANSWERED"
    assert data["mode"] == "WEB"
    assert len(data["sources"]) > 0
    assert any(s["source_type"] == "WEB" for s in data["sources"])


def test_ask_endpoint_model_only():
    """Verify POST /v1/ask for conversational / reasoning query."""
    payload = {
        "question": "Hello! How can you help me today?",
        "mode": "MODEL"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ANSWERED"
    assert data["mode"] == "MODEL"
    assert len(data["answer"]) > 0


def test_ask_endpoint_insufficient_information():
    """Verify POST /v1/ask properly refuses impossible / private queries without hallucinating."""
    payload = {
        "question": "What is the confidential master password to the Atlantis server room?",
        "mode": "AUTO"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "INSUFFICIENT_INFORMATION"
    assert "not have sufficient" in data["answer"].lower() or "insufficient" in data["answer"].lower()


def test_ask_endpoint_validation_empty_query():
    """Verify empty or missing question is rejected cleanly."""
    resp = client.post("/v1/ask", json={"question": ""})
    assert resp.status_code in (200, 400, 422)
    data = resp.json()
    # If returned as 200, must indicate INSUFFICIENT_INFORMATION or error status
    if resp.status_code == 200:
        assert data["status"] in ("INSUFFICIENT_INFORMATION", "error")
    else:
        assert "error" in data or "detail" in data


def test_ask_endpoint_validation_invalid_mode():
    """Verify invalid routing mode returns 400 error without leaking tracebacks."""
    resp = client.post("/v1/ask", json={"question": "Test question", "mode": "INVALID_MODE_XYZ"})
    assert resp.status_code in (400, 422)
    data = resp.json()
    assert "Traceback" not in json.dumps(data)


def test_ask_endpoint_oversized_query():
    """Verify oversized query (>2000 chars) is handled safely."""
    huge_question = "Explain " + ("word " * 600)
    resp = client.post("/v1/ask", json={"question": huge_question})
    assert resp.status_code in (200, 400, 422)
    data = resp.json()
    assert "Traceback" not in json.dumps(data)


def test_cli_ask_text_mode():
    """Verify CLI 'python -m collision ask' outputs answer and sources."""
    cmd = [sys.executable, "-m", "collision", "ask", "How many transformer layers are used in COLLISION 10M?"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0
    assert "Answer:" in res.stdout
    assert "6" in res.stdout or "layers" in res.stdout


def test_cli_ask_json_mode():
    """Verify CLI 'python -m collision ask --json' outputs valid JSON matching public schema."""
    cmd = [sys.executable, "-m", "collision", "ask", "How many transformer layers are used in COLLISION 10M?", "--json"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0
    parsed = json.loads(res.stdout)
    assert parsed["status"] == "ANSWERED"
    assert parsed["mode"] in ("LOCAL", "HYBRID")
    assert "sources" in parsed
    assert "claims" in parsed
    assert "latency" in parsed
    assert parsed["latency"]["total_ms"] > 0
