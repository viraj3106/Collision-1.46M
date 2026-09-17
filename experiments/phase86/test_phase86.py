import os
import json
import pytest
import hashlib
from rag.pipeline import QueryRouter, RAGPipeline

PHASE86_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(PHASE86_DIR, "independent_rag_benchmark.jsonl")
LOCK_PATH = os.path.join(PHASE86_DIR, "experiment_lock.json")
AUDIT_PATH = os.path.join(PHASE86_DIR, "contamination_audit.json")
MODEL_PATH = os.path.abspath(os.path.join(PHASE86_DIR, "..", "..", "models", "collision-10m", "model.pt"))

def test_phase86_benchmark_integrity():
    assert os.path.exists(BENCHMARK_PATH), "Benchmark JSONL file must exist"
    questions = []
    categories = set()
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                questions.append(item)
                categories.add(item["category"])
    
    assert len(questions) == 240, f"Expected 240 benchmark questions, got {len(questions)}"
    assert len(categories) == 12, f"Expected 12 distinct categories, got {len(categories)}"
    
    # Verify each category has exactly 20 questions
    for cat in categories:
        cat_count = sum(1 for q in questions if q["category"] == cat)
        assert cat_count == 20, f"Category {cat} has {cat_count} questions, expected 20"

def test_phase86_model_lock():
    assert os.path.exists(MODEL_PATH), "Production model checkpoint must exist"
    sha256 = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    assert digest == "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97", f"Model SHA256 mismatch: {digest}"

def test_phase86_contamination_audit():
    assert os.path.exists(AUDIT_PATH), "Contamination audit JSON must exist"
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["contamination_rate"] == 0.0, "Contamination rate must be 0.0%"
    assert audit["overlapping_questions_count"] == 0

def test_phase86_router_frozen_rules():
    router = QueryRouter()
    # Test router basic functionality without mutating state
    res_web = router.should_search("What is the current stock price of Apple today?")
    assert res_web is True
    
    res_static = router.should_search("What is the capital of France?")
    assert res_static is False
