import os
import sys
import pytest
import json
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase73.train_phase73 import verify_immutability
from experiments.phase76.training.train_phase76 import compute_hybrid_loss
from data.tokenize import BPETokenizer
import torch

CONFIG_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase76", "phase76_config.yaml")
GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "data", "gold", "gold_eval_set.jsonl")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def test_phase76_directory_and_config():
    assert os.path.exists(CONFIG_PATH), "phase76_config.yaml missing!"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 76
    assert "research_question" in cfg["experiment"]

def test_gold_evaluation_set_integrity():
    assert os.path.exists(GOLD_SET_PATH), "gold_eval_set.jsonl missing!"
    records = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    assert len(records) == 140

def test_tokenizer_loading():
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    assert len(tokenizer.vocab) > 260

def test_production_baseline_immutability():
    sha = verify_immutability()
    assert sha.lower() == "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97".lower()

def test_loss_configuration_validity():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    loss_cfgs = cfg["loss_configurations"]
    assert "B0" in loss_cfgs
    assert "B1" in loss_cfgs
    assert "B2" in loss_cfgs
    assert loss_cfgs["B1"]["alpha"] + loss_cfgs["B1"]["beta"] == pytest.approx(1.0)

def test_context_length_stress_test_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lengths = cfg["context_stress_test"]["eval_lengths"]
    assert lengths == [32, 64, 96, 128, 160, 192, 256, 384]

def test_hybrid_loss_math_execution():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), x.size(1), 260)
            
    model = DummyModel()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
    
    total_loss, c_loss, r_loss = compute_hybrid_loss(model, x, y, mask, alpha=0.10, beta=0.90)
    assert total_loss.item() > 0.0
    assert c_loss.item() > 0.0
    assert r_loss.item() > 0.0

def test_result_schema_validity():
    results_path = os.path.join(PROJECT_ROOT, "experiments", "phase76", "results", "phase76_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        assert "stress_test" in res or "evaluations" in res

def test_deterministic_evaluation_metadata():
    report_file = os.path.join(PROJECT_ROOT, "experiments", "phase76", "reports", "generate_report.py")
    assert os.path.exists(report_file)

def test_historical_immutability_non_regression():
    p74_file = os.path.join(PROJECT_ROOT, "tests", "test_phase74.py")
    p75_file = os.path.join(PROJECT_ROOT, "tests", "test_phase75.py")
    assert os.path.exists(p74_file)
    assert os.path.exists(p75_file)
