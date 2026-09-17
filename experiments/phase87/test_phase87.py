import os
import json
import pytest
import hashlib

PHASE87_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE86_DIR = os.path.abspath(os.path.join(PHASE87_DIR, "..", "phase86"))
BENCHMARK_PATH = os.path.join(PHASE86_DIR, "independent_rag_benchmark.jsonl")
MODEL_PATH = os.path.abspath(os.path.join(PHASE87_DIR, "..", "..", "models", "collision-10m", "model.pt"))
RECALC_PATH = os.path.join(PHASE87_DIR, "recalculated_metrics.json")
EVAL_AUDIT_PATH = os.path.join(PHASE87_DIR, "evaluator_audit.json")

def test_phase87_model_integrity():
    assert os.path.exists(MODEL_PATH), "Model checkpoint must exist"
    sha = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    assert sha.hexdigest() == "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def test_phase87_benchmark_unmodified():
    assert os.path.exists(BENCHMARK_PATH), "Phase 86 benchmark must exist"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 240, "Phase 86 benchmark line count must remain 240"

def test_phase87_recalculated_metrics_exist():
    assert os.path.exists(RECALC_PATH), "Recalculated metrics JSON must exist"
    with open(RECALC_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "AUTO_RAG_PHASE85" in data
    assert data["AUTO_RAG_PHASE85"]["total_questions"] == 240

def test_phase87_evaluator_audit():
    assert os.path.exists(EVAL_AUDIT_PATH), "Evaluator audit JSON must exist"
    with open(EVAL_AUDIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "evaluator_validity_rate" in data
    assert data["total_cases"] == 7
