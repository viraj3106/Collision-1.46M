import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase73.train_phase73 import verify_immutability, create_masked_batch, compute_sft_loss
from data.tokenize import BPETokenizer
from model.config import ModelConfig
from model.transformer import CollisionTransformer

EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def test_checkpoint_immutability():
    sha = verify_immutability()
    assert sha.lower() == EXPECTED_SHA256.lower()

def test_response_only_loss_masking():
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    
    p = "Write a quick test prompt."
    r = "This is the response content."
    m_data = create_masked_batch(p, r, tokenizer)
    
    prompt_len = m_data["prompt_len"]
    mask = m_data["loss_mask"]
    
    assert sum(mask[:prompt_len]) == 0, "Prompt tokens must have loss_mask = 0"
    assert sum(mask[prompt_len:]) > 0, "Response tokens must have loss_mask = 1"

def test_kl_gradient_isolation():
    config = ModelConfig(vocab_size=890, n_layer=2, n_head=2, d_model=64, max_seq_len=64)
    cand_model = CollisionTransformer(config)
    base_model = CollisionTransformer(config)
    
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
        
    x = torch.randint(0, 890, (1, 10))
    targets = torch.randint(0, 890, (1, 10))
    mask = torch.ones((1, 10), dtype=torch.float32)
    
    tot_l, _, kl_l = compute_sft_loss(cand_model, base_model, x, targets, mask, kl_weight=0.1)
    tot_l.backward()
    
    assert not torch.isnan(tot_l), "Loss must not be NaN"
    assert not torch.isinf(tot_l), "Loss must not be Inf"
    
    for param in base_model.parameters():
        assert param.grad is None, "Base model parameters must not accumulate gradients"
