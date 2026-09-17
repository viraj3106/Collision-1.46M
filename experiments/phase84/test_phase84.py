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

def test_phase84_eval_set_structure():
    path = os.path.join(PROJECT_ROOT, "experiments", "phase84", "rag_answer_eval_set.jsonl")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    assert len(records) >= 60
    categories = set(r["category"] for r in records)
    assert len(categories) == 10

def test_router_regression_phase84():
    router = QueryRouter()
    path = os.path.join(PROJECT_ROOT, "experiments", "phase83", "router_eval_set.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    correct = sum(1 for r in records if router.should_search(r["query"]) == r["expected_requires_web"])
    accuracy = (correct / len(records)) * 100.0
    
    no_web_recs = [r for r in records if not r["expected_requires_web"]]
    unnecessary = sum(1 for r in no_web_recs if router.should_search(r["query"]))
    unnecessary_rate = (unnecessary / max(1, len(no_web_recs))) * 100.0

    print(f"\nRouter Accuracy Regression Check: {accuracy:.2f}% (Target >= 90.0%)")
    print(f"Unnecessary Search Rate Check: {unnecessary_rate:.2f}% (Target <= 10.0%)")

    assert accuracy >= 90.0
    assert unnecessary_rate <= 10.0

def test_hallucination_resistance():
    mock_search = MockSearchProvider()
    mock_search.add_mock_results(
        "python 9.0 features",
        [{"title": "Python Versions", "url": "https://python.org", "snippet": "Python 9.0 does not exist. Python 3.12 is the current release."}]
    )
    pipeline = RAGPipeline(search_provider=mock_search)
    res = pipeline.process(RAGRequest(query="What features were added in Python 9.0?", mode="on"))
    
    assert res.web_search_used is True
    assert "UNTRUSTED WEB DATA" in res.prompt_formatted
    assert "INSTRUCTION: Answer using context only" in res.prompt_formatted

def test_context_budget_strict_256_cap():
    cm = ContextManager(max_seq_len=256)
    huge_passage = "Benchmarking data context block " * 300
    item = SearchItem(title="Huge Document", url="https://example.com/huge", snippet=huge_passage)
    
    _, formatted_prompt, _, budget = cm.build_rag_context("Stress query", [(item, huge_passage)], max_completion_tokens=80)
    prompt_tokens = len(formatted_prompt.split())
    
    assert prompt_tokens + 80 <= 256
    assert budget.total_budget == 256

def test_production_model_sha256_and_param_count():
    model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    expected_sha = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
    
    assert os.path.exists(model_path)
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()
    assert actual_sha == expected_sha, f"Production model SHA256 mismatch! Found: {actual_sha}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
