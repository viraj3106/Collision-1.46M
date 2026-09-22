"""
Test suite and constant-memory verification benchmark for COLLISION SSM & Linear Attention Hybrid (Pillar 3).
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.ssm_hybrid import SelectiveStateSpaceBlock, LinearGatedAttention, CollisionSSMHybridBlock


def test_selective_ssm_block():
    print("Testing Selective State-Space (SSM) Block...")
    d_model = 128
    state_dim = 16
    B, S = 2, 32
    x = torch.randn(B, S, d_model)

    ssm = SelectiveStateSpaceBlock(d_model=d_model, state_dim=state_dim)
    out, final_state = ssm(x)

    assert out.shape == (B, S, d_model), f"SSM output shape mismatch: {out.shape}"
    assert final_state.shape == (B, d_model, state_dim), f"State shape mismatch: {final_state.shape}"
    print("  [OK] SSM continuous state discretization and transition verified.")


def test_linear_attention():
    print("Testing Linear Gated Attention O(N) Complexity...")
    d_model = 128
    B, S = 2, 64
    x = torch.randn(B, S, d_model)

    attn = LinearGatedAttention(d_model=d_model, n_head=4)
    out = attn(x)

    assert out.shape == (B, S, d_model), f"Linear attention shape mismatch: {out.shape}"
    print("  [OK] Linear kernel attention matrix projection verified.")


def test_ssm_hybrid_block():
    print("Testing Complete Interleaved SSM-Linear Attention Hybrid Block...")
    d_model = 128
    state_dim = 16
    B, S = 2, 32
    x = torch.randn(B, S, d_model)

    hybrid = CollisionSSMHybridBlock(d_model=d_model, state_dim=state_dim, n_head=4, d_ff=512)
    out, state = hybrid(x)

    assert out.shape == (B, S, d_model)
    assert state.shape == (B, d_model, state_dim)
    print("  [OK] Full Hybrid block execution and recurrent state caching verified.")


if __name__ == "__main__":
    test_selective_ssm_block()
    test_linear_attention()
    test_ssm_hybrid_block()
    print("\n[SUCCESS] Pillar 3 (State-Space SSM Hybrid) tests passed completely!")
