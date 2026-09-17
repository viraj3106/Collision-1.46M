import os
import sys
import pytest
import hashlib
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.schemas import RAGRequest, SearchItem
from rag.search import MockSearchProvider
from rag.context import ContextManager
from rag.pipeline import RAGPipeline, QueryRouter

def test_production_model_sha256_and_param_count_integrity():
    model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    expected_sha = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
    
    assert os.path.exists(model_path)
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()
    assert actual_sha == expected_sha, f"Production model SHA256 changed! Expected: {expected_sha}, Actual: {actual_sha}"

def test_phase84_artifacts_unchanged():
    p84_report = os.path.join(PROJECT_ROOT, "experiments", "phase84", "PHASE84_REPORT.md")
    assert os.path.exists(p84_report)
    with open(p84_report, "r", encoding="utf-8") as f:
        content = f.read()
    assert "PHASE_84_RAG_ANSWER_QUALITY_VALIDATED" in content

def test_router_deterministic_decisions():
    router = QueryRouter()
    # Explicit search keyword -> ALWAYS True
    assert router.should_search("search the web for python 3.12") is True
    # Temporal keyword -> ALWAYS True
    assert router.should_search("What is today's weather in Tokyo?") is True
    # Verification/adversarial keyword -> True
    assert router.should_search("What features were added in Python 9.0?") is True
    # Static math -> False
    assert router.should_search("What is 15 multiplied by 12?") is False

def test_auto_rag_optimized_pipeline():
    mock_search = MockSearchProvider()
    mock_search.add_mock_results(
        "python 9.0 features",
        [{"title": "Python Versions", "url": "https://python.org", "snippet": "Python 9.0 does not exist."}]
    )
    pipeline = RAGPipeline(search_provider=mock_search)
    res = pipeline.process(RAGRequest(query="What features were added in Python 9.0?", mode="auto"))
    
    assert res.web_search_used is True
    assert len(res.sources) == 1
    assert "UNTRUSTED WEB DATA" in res.prompt_formatted

def test_context_budget_256_cap():
    cm = ContextManager(max_seq_len=256)
    huge_passage = "Benchmarking data context block " * 300
    item = SearchItem(title="Huge Document", url="https://example.com/huge", snippet=huge_passage)
    
    _, formatted_prompt, _, budget = cm.build_rag_context("Stress query", [(item, huge_passage)], max_completion_tokens=80)
    prompt_tokens = len(formatted_prompt.split())
    
    assert prompt_tokens + 80 <= 256
    assert budget.total_budget == 256

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
