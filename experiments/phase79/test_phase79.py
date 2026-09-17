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
from experiments.phase79.train_phase79 import MODEL_SPECS, load_binary_dataset, DATASET_DIR
from experiments.phase79.evaluator import calculate_generation_metrics, compute_phase79_scaling_analysis

def test_phase79_config_validity():
    config_path = os.path.join(PROJECT_ROOT, "experiments", "phase79", "config.yaml")
    assert os.path.exists(config_path), "Phase 79 config file missing"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["phase"] == 79
    assert "P79-A" in cfg["models"]
    assert "P79-B" in cfg["models"]

def test_phase79_parameter_counts_exact():
    expected_params = {
        "P79-A": 10282304,
        "P79-B": 25263936
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
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params == expected_params[key], f"{key} exact parameter count mismatch: got {params}, expected {expected_params[key]}"

def test_phase79_model_logits_forward_pass():
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
        dummy_input = torch.randint(0, 8000, (2, 32))
        logits, loss = model(dummy_input, targets=dummy_input)
        assert logits.shape == (2, 32, 8000), f"{key} logits shape mismatch: {logits.shape}"
        assert loss is not None and not torch.isnan(loss)

def test_phase79_dataset_loading():
    train_data, val_data, test_data = load_binary_dataset(DATASET_DIR)
    assert len(train_data) > 0, "Train dataset is empty"
    assert len(val_data) > 0, "Val dataset is empty"
    assert len(test_data) > 0, "Test dataset is empty"

def test_phase79_scaling_analysis_logic():
    mock_train = {
        "P79-A": {"exact_params": 10282304, "tokens_processed": 2048000, "tokens_per_param": 0.199, "elapsed_time_s": 10.0, "tokens_per_sec": 204800.0},
        "P79-B": {"exact_params": 25263936, "tokens_processed": 2048000, "tokens_per_param": 0.081, "elapsed_time_s": 20.0, "tokens_per_sec": 102400.0}
    }
    mock_eval = {
        "P79-A": {"val_loss": 3.20, "val_ppl": 24.53, "test_loss": 3.25, "test_ppl": 25.79, "gen_metrics_temp_07": {"repetition_rate": 0.05, "coherence_score": 2.1}, "probe_evaluation": {"overall_probe_score": 0.40}},
        "P79-B": {"val_loss": 2.80, "val_ppl": 16.44, "test_loss": 2.85, "test_ppl": 17.28, "gen_metrics_temp_07": {"repetition_rate": 0.03, "coherence_score": 2.5}, "probe_evaluation": {"overall_probe_score": 0.55}}
    }
    
    analysis = compute_phase79_scaling_analysis(mock_train, mock_eval)
    assert "scientific_outcome" in analysis
    assert analysis["parameter_increase_pct"] > 140.0
    assert analysis["val_loss_improvement_pct"] > 0.0

if __name__ == "__main__":
    pytest.main([__file__])
