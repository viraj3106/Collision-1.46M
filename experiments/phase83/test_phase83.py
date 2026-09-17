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

def test_router_accuracy_target():
    router = QueryRouter()
    eval_set_path = os.path.join(PROJECT_ROOT, "experiments", "phase83", "router_eval_set.jsonl")
    assert os.path.exists(eval_set_path)
    
    with open(eval_set_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) >= 60
    
    correct = 0
    for rec in records:
        pred = router.should_search(rec["query"])
        if pred == rec["expected_requires_web"]:
            correct += 1
            
    accuracy = (correct / len(records)) * 100.0
    print(f"\nRouter Accuracy on 60-example test set: {accuracy:.2f}% ({correct}/{len(records)})")
    assert accuracy >= 90.0, f"Router accuracy {accuracy:.2f}% fell below 90.0% target!"

def test_unnecessary_search_prevention_rate():
    router = QueryRouter()
    eval_set_path = os.path.join(PROJECT_ROOT, "experiments", "phase83", "router_eval_set.jsonl")
    with open(eval_set_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    no_web_records = [r for r in records if not r["expected_requires_web"]]
    unnecessary_searches = sum(1 for r in no_web_records if router.should_search(r["query"]))
    
    unnecessary_rate = (unnecessary_searches / max(1, len(no_web_records))) * 100.0
    print(f"\nUnnecessary Search Rate: {unnecessary_rate:.2f}% ({unnecessary_searches}/{len(no_web_records)})")
    assert unnecessary_rate <= 10.0, f"Unnecessary search rate {unnecessary_rate:.2f}% exceeded 10.0% ceiling!"

def test_explicit_modes_deterministic_behavior():
    pipeline = RAGPipeline(search_provider=MockSearchProvider())
    
    # Mode OFF -> NEVER search
    res_off = pipeline.process(RAGRequest(query="search the web for latest news today", mode="off"))
    assert res_off.web_search_used is False
    
    # Mode ON -> ALWAYS attempt search
    res_on = pipeline.process(RAGRequest(query="What is 2 + 2?", mode="on"))
    assert res_on.web_search_used is True

def test_context_budget_256_cap():
    cm = ContextManager(max_seq_len=256)
    huge_passage = "Advanced COLLISION 25M model benchmarking data context " * 200
    huge_item = SearchItem(title="Stress Test Page", url="https://example.com/stress", snippet=huge_passage)
    
    _, formatted_prompt, _, budget = cm.build_rag_context("Stress query", [(huge_item, huge_passage)], max_completion_tokens=80)
    prompt_tokens = len(formatted_prompt.split())
    
    assert prompt_tokens + 80 <= 256
    assert budget.context_tokens <= 180

def test_model_sha256_integrity():
    model_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    expected_sha = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
    
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()
    assert actual_sha == expected_sha

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
