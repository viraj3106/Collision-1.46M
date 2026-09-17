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
from experiments.phase76.training.train_phase76 import compute_hybrid_loss
from experiments.phase77.train_phase77 import MODEL_SPECS, train_phase77_model
from experiments.phase77.evaluator import compute_scaling_metrics
from experiments.phase77.generate_report import generate_phase77_report

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def test_config_validity():
    config_path = os.path.join(PROJECT_ROOT, "experiments", "phase77", "phase77_config.yaml")
    assert os.path.exists(config_path), "Phase 77 config file missing"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 77
    assert "control_10m" in cfg["models"]
    assert "candidate_25m" in cfg["models"]
    assert "candidate_35m" in cfg["models"]
    assert "candidate_50m" in cfg["models"]

def test_model_construction_and_parameter_counts():
    expected_counts = {
        "COLLISION-10M": 10282304,
        "J77-25M": 25263936,
        "J77-35M": 34847680,
        "J77-50M": 48893504
    }
    
    for key, spec in MODEL_SPECS.items():
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
        exact_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert exact_params == expected_counts[key], f"{key} parameter count mismatch: got {exact_params}, expected {expected_counts[key]}"

def test_hybrid_loss_calculation():
    m_cfg = ModelConfig(vocab_size=8000, max_seq_len=256, d_model=384, n_layer=6, n_head=8, d_ff=768)
    model = CollisionTransformer(m_cfg)
    x = torch.randint(0, 8000, (2, 32))
    targets = torch.randint(0, 8000, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.float)
    mask[:, :16] = 0.0 # First 16 tokens context, last 16 tokens response
    
    loss, c_loss, r_loss = compute_hybrid_loss(model, x, targets, mask, alpha=0.10, beta=0.90)
    assert loss.item() > 0.0
    assert c_loss.item() > 0.0
    assert r_loss.item() > 0.0

def test_scaling_metrics_computation():
    mock_eval = {
        "COLLISION-10M": {"aggregate_metrics": {"avg_response_quality": 3.1, "avg_coherence": 2.36}, "failure_distribution": {}},
        "J77-25M": {"aggregate_metrics": {"avg_response_quality": 3.2, "avg_coherence": 2.45}, "failure_distribution": {}},
        "J77-35M": {"aggregate_metrics": {"avg_response_quality": 3.25, "avg_coherence": 2.52}, "failure_distribution": {}},
        "J77-50M": {"aggregate_metrics": {"avg_response_quality": 3.3, "avg_coherence": 2.60}, "failure_distribution": {}}
    }
    mock_train = {
        "COLLISION-10M": {"exact_params": 10282304, "elapsed_time_s": 2.0, "final_loss": 2.5, "ppl": 12.18, "d_model": 384, "n_layer": 6, "n_head": 8, "d_ff": 768},
        "J77-25M": {"exact_params": 25263936, "elapsed_time_s": 4.0, "final_loss": 2.3, "ppl": 9.97, "d_model": 512, "n_layer": 10, "n_head": 8, "d_ff": 1024},
        "J77-35M": {"exact_params": 34847680, "elapsed_time_s": 6.0, "final_loss": 2.1, "ppl": 8.17, "d_model": 640, "n_layer": 9, "n_head": 10, "d_ff": 1280},
        "J77-50M": {"exact_params": 48893504, "elapsed_time_s": 8.0, "final_loss": 1.9, "ppl": 6.69, "d_model": 768, "n_layer": 9, "n_head": 12, "d_ff": 1536}
    }
    
    scaling_analysis = compute_scaling_metrics(mock_eval, mock_train)
    assert scaling_analysis["scientific_outcome"].startswith("Outcome A")
    assert scaling_analysis["hypothesis_supported"] is True

if __name__ == "__main__":
    pytest.main([__file__])
