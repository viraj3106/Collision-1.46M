"""
COLLISION-1.0B Official Flagship Regression & Verification Test Suite.

Verifies:
1. Default configuration = 1B (999,376,128 parameters)
2. Model construction and tensor shapes
3. Checkpoint existence, format, and SHA-256 integrity
4. Checkpoint loading and forward pass
5. Generation without NaN/Inf
6. Service and API runtime resolution to 1B
"""

import os
import sys
import hashlib
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from collision.config import (
    DEFAULT_CONFIG_PATH,
    PRODUCTION_MODEL_PATH,
    PROTECTED_COLLISION_1B_SHA256,
    validate_checkpoints
)
from collision.service import CollisionService


def test_default_model_config_is_1b():
    config = ModelConfig()
    param_count = config.calculate_parameter_count()
    assert param_count == 999376128, f"Expected 999,376,128, got {param_count}"
    assert config.vocab_size == 32000
    assert config.max_seq_len == 1024
    assert config.d_model == 2048
    assert config.n_layer == 24
    assert config.n_head == 16
    assert config.d_ff == 5376
    assert config.tie_embeddings is True


def test_yaml_config_1b():
    assert os.path.exists(DEFAULT_CONFIG_PATH)
    config = ModelConfig.from_yaml(DEFAULT_CONFIG_PATH)
    param_count = config.calculate_parameter_count()
    assert param_count == 999376128, f"Expected 999,376,128, got {param_count}"


def test_checkpoint_integrity_and_load():
    assert os.path.exists(PRODUCTION_MODEL_PATH), f"Production model missing at {PRODUCTION_MODEL_PATH}"
    assert validate_checkpoints() is True

    # Checkpoint dict inspect
    ckpt = torch.load(PRODUCTION_MODEL_PATH, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "config" in ckpt

    cfg = ModelConfig(**ckpt["config"])
    assert cfg.calculate_parameter_count() == 999376128

    model = CollisionTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 999376128

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_service_runtime_flagship():
    service = CollisionService()
    h = service.health()
    assert h["status"] == "ok"
    assert h["model"] == "collision-1.0b"
    assert h["model_available"] is True

    r = service.ready()
    assert r["status"] == "ready"
