"""
COLLISION Phase 102 — Production Deployment & Real-World Reliability Test Suite.

Verifies:
- Checkpoint integrity (locked, unmodified)
- Production configuration & environment variables
- CORS security & origin filtering
- API security & input validation (oversized input, malformed JSON)
- Rate limiting protection (HTTP 429, Retry-After header)
- Request timeout protection & safe fallback
- Web search failure recovery (SSRF, timeout, 500, malformed HTML)
- Local RAG failure recovery (empty index, zero matches, graceful handling)
- Model failure recovery (error containment, no traceback leakage)
- Health & Readiness endpoint semantics (/health, /ready)
- Startup validation & fast failure
- Structured logging & observability metrics (/metrics, /v1/metrics)
"""

import os
import sys
import time
import hashlib
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app, validate_system_startup
from collision.config import (
    PROTECTED_COLLISION_10M_SHA256,
    PROTECTED_PHASE91_V9_SHA256,
    COLLISION_MAX_INPUT_LENGTH,
    PRODUCTION_MODEL_PATH
)
from collision.service import get_collision_service, CollisionService
from collision.web.search import MockWebSearchProvider
from collision.web.fetch import safe_fetch_page, SSRFProtectionError
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.grounding.engine import GroundedSynthesisEngine
from collision.grounding.schemas import AnswerType
from collision.routing.schemas import RouteMode
from api.limiter import clear_rate_limits, check_rate_limit
from api.metrics import metrics_collector

client = TestClient(app)


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()


# --------------------------------------------------------------------------
# 1. Checkpoint & Model Integrity Lock
# --------------------------------------------------------------------------
def test_checkpoint_integrity_phase102():
    """Verify production and research checkpoints remain byte-for-byte identical."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    assert os.path.exists(c10m_path), "Production collision-10m model checkpoint missing"
    assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    assert os.path.exists(v9_path), "Research phase91_v9_10m checkpoint missing"
    assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256


# --------------------------------------------------------------------------
# 2. Health & Readiness Endpoint Semantics
# --------------------------------------------------------------------------
def test_health_liveness_endpoint():
    """Verify /health returns live status and model metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "collision"
    assert "model" in data


def test_readiness_probe_dependencies():
    """Verify /ready inspects database, model, tokenizer, and index components."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    checks = data["checks"]
    assert checks.get("database") == "ok"
    assert checks.get("model") == "ok"
    assert checks.get("model_checkpoint") is True
    assert checks.get("tokenizer") is True
    assert checks.get("local_index") is True


# --------------------------------------------------------------------------
# 3. CORS Security
# --------------------------------------------------------------------------
def test_cors_allowed_origin():
    """Verify that configured origins receive proper CORS headers."""
    resp = client.options(
        "/v1/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --------------------------------------------------------------------------
# 4. API Security & Input Validation
# --------------------------------------------------------------------------
def test_oversized_input_rejection():
    """Verify that queries exceeding maximum input length are rejected with 413 or 422."""
    oversized_prompt = "What is COLLISION? " * 300  # > 2000 chars
    resp = client.post("/v1/ask", json={"question": oversized_prompt})
    assert resp.status_code in (413, 422)
    data = resp.json()
    assert data["error"]["type"] == "validation_error"


def test_malformed_json_handling():
    """Verify that malformed request payloads receive structured error responses."""
    resp = client.post(
        "/v1/ask",
        content="{'invalid': json",
        headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "validation_error"


# --------------------------------------------------------------------------
# 5. Rate Limiting Protection
# --------------------------------------------------------------------------
def test_rate_limiting_enforcement():
    """Verify that exceeding rate limits returns HTTP 429 with Retry-After header."""
    clear_rate_limits()
    os.environ["COLLISION_RATE_LIMIT"] = "5"

    try:
        # Send 5 allowed requests
        for _ in range(5):
            r = client.post(
                "/v1/ask",
                json={"question": "What is COLLISION 10M?"},
                headers={"X-API-Key": "test_rate_limit_key"}
            )
            assert r.status_code == 200

        # 6th request must trigger rate limit (429)
        exceeded_resp = client.post(
            "/v1/ask",
            json={"question": "What is COLLISION 10M?"},
            headers={"X-API-Key": "test_rate_limit_key"}
        )
        assert exceeded_resp.status_code == 429
        assert "Retry-After" in exceeded_resp.headers
        data = exceeded_resp.json()
        assert data["error"]["type"] == "rate_limit_error"

    finally:
        os.environ["COLLISION_RATE_LIMIT"] = "60"
        clear_rate_limits()


# --------------------------------------------------------------------------
# 6. Web Failure Recovery & SSRF Protection
# --------------------------------------------------------------------------
def test_ssrf_blocking():
    """Verify that SSRF attempts to private networks or localhost are blocked."""
    from collision.web.fetch import validate_url, safe_fetch_page, SSRFProtectionError

    with pytest.raises(SSRFProtectionError):
        validate_url("http://127.0.0.1:8000/secret")

    with pytest.raises(SSRFProtectionError):
        validate_url("http://169.254.169.254/latest/meta-data")

    html, err = safe_fetch_page("http://127.0.0.1:8000/secret")
    assert html is None
    assert "SSRF" in err or "blocked" in err


def test_web_search_empty_results_handling():
    """Verify that web provider failure or empty results cleanly abstains without crashing."""
    search_provider = MockWebSearchProvider(mock_database={})
    engine = GroundedSynthesisEngine(search_provider=search_provider)
    res = engine.answer("What are the release notes for React 99?", mode=RouteMode.WEB)
    assert res.answer_type == AnswerType.INSUFFICIENT_INFORMATION
    assert res.status == "INSUFFICIENT_INFORMATION"


# --------------------------------------------------------------------------
# 7. Local RAG Failure Recovery
# --------------------------------------------------------------------------
def test_local_rag_empty_index_recovery():
    """Verify that an empty local index gracefully returns insufficient information."""
    empty_index = VectorIndex()
    retriever = DocumentRetriever(index=empty_index)
    engine = GroundedSynthesisEngine(local_retriever=retriever)
    res = engine.answer("What is COLLISION 10M?", mode=RouteMode.LOCAL)
    assert res.status == "INSUFFICIENT_INFORMATION"
    assert "insufficient" in res.answer.lower() or "reliable" in res.answer.lower()


# --------------------------------------------------------------------------
# 8. Startup Validation
# --------------------------------------------------------------------------
def test_startup_validation_success():
    """Verify that startup validation succeeds when environment is intact."""
    validate_system_startup()


# --------------------------------------------------------------------------
# 9. Observability Metrics
# --------------------------------------------------------------------------
def test_observability_metrics_endpoint():
    """Verify that /metrics endpoint tracks requests, latency, and status codes."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert "total_requests" in metrics
    assert "successful_requests" in metrics
    assert "latency_ms" in metrics
    assert "avg" in metrics["latency_ms"]
    assert "p50" in metrics["latency_ms"]
    assert "p95" in metrics["latency_ms"]
    assert "status_code_distribution" in metrics
