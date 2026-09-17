import os
import sys
import yaml
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from experiments.phase78.train_phase78 import CANDIDATE_SPECS
from experiments.phase78.evaluator import compute_phase78_scaling_analysis

def test_phase78_config_validity():
    config_path = os.path.join(PROJECT_ROOT, "experiments", "phase78", "phase78_config.yaml")
    assert os.path.exists(config_path), "Phase 78 config file missing"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 78
    assert "P78-A" in cfg["candidates"]
    assert "P78-B" in cfg["candidates"]
    assert "P78-C" in cfg["candidates"]
    assert "P78-D" in cfg["candidates"]
    assert "P78-E" in cfg["candidates"]

def test_phase78_model_construction_and_parameters():
    expected_params = {
        "P78-A": 10282304,
        "P78-B": 10282304,
        "P78-C": 25263936,
        "P78-D": 25263936,
        "P78-E": 48893504
    }
    for key, spec in CANDIDATE_SPECS.items():
        m_cfg = ModelConfig(
            vocab_size=8000,
            max_seq_len=256,
            d_model=spec["d_model"],
            n_layer=spec["n_layer"],
            n_head=spec["n_head"],
            d_ff=spec["d_ff"],
            tie_embeddings=True
        )
        model = CollisionTransformer(m_cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params == expected_params[key], f"{key} parameter count mismatch: got {params}, expected {expected_params[key]}"

def test_phase78_scaling_metrics_logic():
    mock_eval = {
        "COLLISION-10M": {"aggregate_metrics": {"avg_response_quality": 3.13, "avg_coherence": 2.47}, "failure_distribution": {}},
        "P78-A": {"aggregate_metrics": {"avg_response_quality": 2.1, "avg_coherence": 1.2}, "failure_distribution": {}},
        "P78-B": {"aggregate_metrics": {"avg_response_quality": 2.5, "avg_coherence": 1.8}, "failure_distribution": {}},
        "P78-C": {"aggregate_metrics": {"avg_response_quality": 2.7, "avg_coherence": 2.1}, "failure_distribution": {}},
        "P78-D": {"aggregate_metrics": {"avg_response_quality": 2.9, "avg_coherence": 2.3}, "failure_distribution": {}},
        "P78-E": {"aggregate_metrics": {"avg_response_quality": 3.2, "avg_coherence": 2.65}, "failure_distribution": {}}
    }
    mock_train = {
        "P78-A": {"exact_params": 10282304, "tokens_processed": 204800, "best_val_loss": 3.5, "ppl": 33.1, "elapsed_time_s": 1.0},
        "P78-B": {"exact_params": 10282304, "tokens_processed": 819200, "best_val_loss": 3.1, "ppl": 22.2, "elapsed_time_s": 3.0},
        "P78-C": {"exact_params": 25263936, "tokens_processed": 819200, "best_val_loss": 2.9, "ppl": 18.2, "elapsed_time_s": 5.0},
        "P78-D": {"exact_params": 25263936, "tokens_processed": 2048000, "best_val_loss": 2.5, "ppl": 12.2, "elapsed_time_s": 10.0},
        "P78-E": {"exact_params": 48893504, "tokens_processed": 2048000, "best_val_loss": 2.2, "ppl": 9.0, "elapsed_time_s": 15.0}
    }
    
    analysis = compute_phase78_scaling_analysis(mock_eval, mock_train)
    assert "scientific_outcome" in analysis
    assert "matrix_table" in analysis
    assert analysis["scientific_outcome"].startswith("Outcome A")

if __name__ == "__main__":
    pytest.main([__file__])
