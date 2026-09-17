import os
import sys
import json
import yaml
import torch
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from experiments.phase80.train_phase80 import CONDITION_SPECS, load_binary_dataset, DATASET_DIR, train_phase80_condition
from experiments.phase80.evaluator import calculate_generation_metrics, compute_phase80_scaling_analysis

def test_phase80_config_validity():
    config_path = os.path.join(PROJECT_ROOT, "experiments", "phase80", "phase80_config.yaml")
    assert os.path.exists(config_path), "Phase 80 config file missing"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 80
    assert "P80-A" in cfg["conditions"]
    assert "P80-E" in cfg["conditions"]

def test_phase80_parameter_count_exact():
    m_cfg = ModelConfig(
        vocab_size=8000,
        max_seq_len=256,
        d_model=512,
        n_layer=10,
        n_head=8,
        d_ff=1024,
        tie_embeddings=True
    )
    model = CollisionTransformer(m_cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 25263936, f"25M exact parameter count mismatch: got {params}, expected 25263936"

def test_phase80_model_logits_forward_pass():
    m_cfg = ModelConfig(
        vocab_size=8000,
        max_seq_len=256,
        d_model=512,
        n_layer=10,
        n_head=8,
        d_ff=1024,
        tie_embeddings=True
    )
    model = CollisionTransformer(m_cfg)
    dummy_input = torch.randint(0, 8000, (2, 32))
    logits, loss = model(dummy_input, targets=dummy_input)
    assert logits.shape == (2, 32, 8000), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss)

def test_phase80_dataset_loading_and_zero_leakage():
    train_data, val_data, test_data = load_binary_dataset(DATASET_DIR)
    assert len(train_data) > 0, "Train dataset is empty"
    assert len(val_data) > 0, "Val dataset is empty"
    assert len(test_data) > 0, "Test dataset is empty"
    
    meta_path = os.path.join(DATASET_DIR, "dataset_metadata.json")
    assert os.path.exists(meta_path), "Dataset metadata missing"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["leakage_val_pct"] == 0.0
    assert meta["leakage_test_pct"] == 0.0

def test_phase80_token_accounting():
    steps = 500
    batch_size = 8
    seq_len = 256
    exact_params = 25263936
    
    tokens_processed = steps * batch_size * seq_len
    assert tokens_processed == 1024000
    tokens_per_param = tokens_processed / exact_params
    assert round(tokens_per_param, 4) == 0.0405

def test_phase80_scaling_analysis_logic():
    mock_train = {
        "P80-A": {"actual_tokens_processed": 1024000, "tokens_per_param": 0.0405, "elapsed_time_s": 10.0},
        "P80-E": {"actual_tokens_processed": 20480000, "tokens_per_param": 0.8106, "elapsed_time_s": 100.0}
    }
    mock_eval = {
        "P80-A": {"val_loss": 3.50, "val_ppl": 33.11, "test_loss": 3.55, "test_ppl": 34.81, "gen_metrics": {"repetition_rate": 0.05}, "probe_evaluation": {"overall_probe_score": 0.40}},
        "P80-E": {"val_loss": 1.50, "val_ppl": 4.48, "test_loss": 1.55, "test_ppl": 4.71, "gen_metrics": {"repetition_rate": 0.00}, "probe_evaluation": {"overall_probe_score": 0.85}}
    }
    
    analysis = compute_phase80_scaling_analysis(mock_train, mock_eval)
    assert "scientific_outcome" in analysis
    assert analysis["val_loss_improvement_pct"] > 50.0

if __name__ == "__main__":
    pytest.main([__file__])
