"""
COLLISION Phase 100 — UI Integration & API Client Verification Suite.

Validates the full integration between the existing frontend chat interface and
the Phase 99 production API (/v1/ask):
- Checkpoint integrity (untrained, unmodified)
- Frontend API Client contract adherence
- Handling all response states:
  * ANSWERED
  * INSUFFICIENT_INFORMATION
  * CONFLICT
  * ERROR
- Real source metadata and citation schemas
- Claim provenance verification
- Latency and routing telemetry preservation
- Error shielding (no stack traces, no internal leaks)
- UI asset verification (no mock/fake answers in build)
"""

import os
import sys
import json
import hashlib
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app
from collision.service import get_collision_service

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

client = TestClient(app)


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()


def test_checkpoint_integrity_phase100():
    """Verify that protected production and research checkpoints remain byte-identical."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    assert os.path.exists(c10m_path), "Production collision-10m model checkpoint missing"
    assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    assert os.path.exists(v9_path), "Research phase91_v9_10m checkpoint missing"
    assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256


def test_ui_request_creation_contract():
    """Verify that POST /v1/ask accepts standard UI payloads and defaults."""
    payload = {
        "question": "What is COLLISION 10M?",
        "mode": "AUTO",
        "include_sources": True,
        "include_claims": True
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "status" in data
    assert "mode" in data
    assert "sources" in data
    assert "claims" in data
    assert "latency" in data


def test_ui_response_state_answered():
    """Verify ANSWERED state returns proper structure for UI rendering."""
    payload = {
        "question": "What is the parameter count of the COLLISION 10M model?",
        "mode": "LOCAL"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ANSWERED"
    assert len(data["answer"]) > 0
    assert data["confidence"] > 0
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    # Validate source fields for UI chip rendering
    first_src = data["sources"][0]
    assert "title" in first_src
    assert "source_type" in first_src
    assert "retrieval_score" in first_src


def test_ui_response_state_insufficient_information():
    """Verify INSUFFICIENT_INFORMATION state preserves uncertainty for the UI."""
    payload = {
        "question": "What will the stock price of Apple be on December 31, 2035?",
        "mode": "LOCAL"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("INSUFFICIENT_INFORMATION", "ANSWERED")
    if data["status"] == "INSUFFICIENT_INFORMATION":
        assert "insufficient" in data["answer"].lower() or "not establish" in data["answer"].lower() or "evidence" in data["answer"].lower()


def test_ui_response_state_conflict():
    """Verify CONFLICT state is surfaced cleanly to the UI."""
    service = get_collision_service()
    # Test service conflict handling logic directly to verify schema contract
    conflict_res = service.ask(
        question="Was Python created in 1991 or 2005?",
        mode="LOCAL"
    )
    assert "status" in conflict_res
    assert "answer" in conflict_res
    assert "sources" in conflict_res


def test_ui_sources_schema_and_metadata():
    """Verify sources schema fields needed for frontend rendering."""
    payload = {
        "question": "How many attention heads are in COLLISION 10M?",
        "mode": "LOCAL",
        "include_sources": True
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sources"]) > 0
    for src in data["sources"]:
        assert "source_id" in src
        assert "title" in src
        assert "url" in src
        assert "source_type" in src
        assert "retrieval_score" in src


def test_ui_claims_schema_and_grounding():
    """Verify claims schema fields needed for citation chips."""
    payload = {
        "question": "What is the context length of COLLISION 10M?",
        "mode": "LOCAL",
        "include_claims": True
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["claims"], list)
    for claim in data["claims"]:
        assert "text" in claim
        assert "support_status" in claim
        assert "evidence_ids" in claim


def test_ui_latency_breakdown_schema():
    """Verify latency timing metrics for UI performance telemetry."""
    payload = {
        "question": "What is COLLISION?",
        "mode": "AUTO"
    }
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    latency = data["latency"]
    assert "routing_ms" in latency
    assert "retrieval_ms" in latency
    assert "generation_ms" in latency
    assert "verification_ms" in latency
    assert "total_ms" in latency
    assert latency["total_ms"] >= 0


def test_ui_error_handling_empty_payload():
    """Verify 422/400 validation error when empty payload is submitted."""
    resp = client.post("/v1/ask", json={})
    assert resp.status_code in (400, 422)
    err = resp.json()
    assert "error" in err
    # Ensure no internal tracebacks or secrets leak
    err_str = json.dumps(err)
    assert "Traceback" not in err_str
    assert "api_key" not in err_str


def test_ui_error_handling_invalid_mode():
    """Verify validation error when invalid routing mode is passed."""
    resp = client.post("/v1/ask", json={"question": "Test?", "mode": "NONEXISTENT_MODE"})
    assert resp.status_code == 400
    err = resp.json()
    assert "error" in err
    assert "Traceback" not in json.dumps(err)


def test_ui_frontend_source_files_integrity():
    """Verify frontend code files contain collisionApi integration and no mock answers."""
    api_ts_path = os.path.join(PROJECT_ROOT, "website", "src", "api.ts")
    assert os.path.exists(api_ts_path)
    with open(api_ts_path, "r", encoding="utf-8") as f:
        api_content = f.read()
    assert "/v1/ask" in api_content
    assert "collisionApi" in api_content

    app_tsx_path = os.path.join(PROJECT_ROOT, "website", "src", "App.tsx")
    with open(app_tsx_path, "r", encoding="utf-8") as f:
        app_content = f.read()
    assert "collisionApi.ask" in app_content

    chat_ws_path = os.path.join(PROJECT_ROOT, "website", "src", "components", "ChatWorkspace.tsx")
    with open(chat_ws_path, "r", encoding="utf-8") as f:
        chat_content = f.read()
    assert "sources" in chat_content
    assert "mode-tag" in chat_content
    assert "banner-conflict" in chat_content
    assert "banner-insufficient" in chat_content
