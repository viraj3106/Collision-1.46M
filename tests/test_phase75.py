import os
import sys
import pytest
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase75.evaluation.taxonomy import FailureTaxonomy, FAILURE_CATEGORIES
from experiments.phase75.evaluation.rubric import ScoringRubric
from experiments.phase75.evaluation.metrics import calculate_metrics_for_generation, calculate_repetition_ratio
from experiments.phase75.evaluation.evaluator import load_model, PROD_MODEL_PATH
from data.tokenize import BPETokenizer

CONFIG_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "phase75_config.yaml")
GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "data", "gold", "gold_eval_set.jsonl")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def test_phase75_config_validity():
    assert os.path.exists(CONFIG_PATH), "phase75_config.yaml missing!"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 75
    assert "research_question" in cfg["experiment"]
    assert "hypothesis" in cfg["experiment"]
    assert len(cfg["failure_taxonomy"]["categories"]) == 10

def test_gold_dataset_exists_and_count():
    assert os.path.exists(GOLD_SET_PATH), "gold_eval_set.jsonl missing!"
    records = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    assert len(records) == 140, f"Expected 140 gold records, got {len(records)}"

def test_gold_dataset_schema_and_categories():
    records = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
                
    categories = set()
    for rec in records:
        assert "id" in rec
        assert "category" in rec
        assert "prompt" in rec
        assert "expected_behavior" in rec
        assert "evaluation_tags" in rec
        categories.add(rec["category"])
        
    assert len(categories) == 14, f"Expected 14 categories, found {len(categories)}"

def test_tokenizer_and_model_loading():
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    assert len(tokenizer.vocab) > 260
    
    model = load_model(PROD_MODEL_PATH)
    assert model is not None
    assert hasattr(model, "eval")

def test_metrics_calculation():
    text = "hello world hello world hello world"
    rep_ratio = calculate_repetition_ratio(text, n=2)
    assert rep_ratio > 0.0
    
    metrics = calculate_metrics_for_generation(
        prompt="Tell me about space.",
        response="Space is vast and fascinating.",
        expected_behavior="General explanation",
        category="explanation",
        loss=1.5
    )
    assert "repetition_ratio" in metrics
    assert "context_retained" in metrics
    assert metrics["generation_length_words"] == 5

def test_scoring_rubric():
    metrics = {"repetition_ratio": 0.05, "context_retained": True}
    scores = ScoringRubric.score_response(
        prompt="My name is Alex.",
        generated_text="Nice to meet you Alex!",
        expected_behavior="Recognize name Alex",
        category="context_retention",
        metrics=metrics
    )
    assert "coherence" in scores
    assert "response_quality" in scores
    assert 0.0 <= scores["response_quality"] <= 4.0

def test_failure_taxonomy_classification():
    metrics = {"repetition_ratio": 0.8, "unigram_repetition": 0.7}
    failures = FailureTaxonomy.classify_response(
        prompt="Repeat word",
        generated_text="word word word word word word word word",
        expected_behavior="No repetition",
        category="repetition_resistance",
        metrics=metrics
    )
    assert "REPETITION" in failures

def test_phase74_artifacts_immutability():
    phase74_test_file = os.path.join(PROJECT_ROOT, "tests", "test_phase74.py")
    assert os.path.exists(phase74_test_file), "Phase 74 test file missing!"
    with open(phase74_test_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "test_checkpoint_immutability" in content
    assert "SYNTHETIC_TEMPLATE_CONTAMINATION" in content
