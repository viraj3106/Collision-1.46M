"""
Test suite for COLLISION Dual-Process System 1 / System 2 Hegelian Dialectic Core (Pillar 2).
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.dialectic_engine import DualProcessCognitiveEngine, HegelianDialecticCore


def test_hegelian_dialectic_flow():
    print("Testing Hegelian Dialectic Thesis-Antithesis-Synthesis Flow...")
    d_model = 256
    B, S = 2, 16
    h = torch.randn(B, S, d_model)

    dialectic = HegelianDialecticCore(d_model=d_model, d_ff=512, n_head=4)
    out = dialectic(h)

    assert "thesis" in out and out["thesis"].shape == (B, S, d_model)
    assert "antithesis" in out and out["antithesis"].shape == (B, S, d_model)
    assert "synthesis" in out and out["synthesis"].shape == (B, S, d_model)
    print("  [OK] Thesis, Antithesis, and Cross-Attention Synthesis shapes valid.")


def test_dual_process_engine():
    print("Testing Dual-Process System 1 vs System 2 Routing...")
    d_model = 256
    vocab_size = 1000
    B, S = 2, 16
    h = torch.randn(B, S, d_model)

    engine = DualProcessCognitiveEngine(d_model=d_model, vocab_size=vocab_size, d_ff=512)

    # 1. Test System 1 force
    res_sys1 = engine(h, force_system=1)
    assert res_sys1["logits"].shape == (B, S, vocab_size)
    assert "System 1" in res_sys1["system_used"]

    # 2. Test System 2 force
    res_sys2 = engine(h, force_system=2)
    assert res_sys2["logits"].shape == (B, S, vocab_size)
    assert "System 2" in res_sys2["system_used"]
    assert "dialectic_tensors" in res_sys2

    # 3. Test Dynamic Blend
    res_auto = engine(h)
    assert res_auto["logits"].shape == (B, S, vocab_size)
    assert (res_auto["complexity"] >= 0.0).all() and (res_auto["complexity"] <= 1.0).all()

    print("  [OK] Cognitive routing & multi-perspective projection verified.")


if __name__ == "__main__":
    test_hegelian_dialectic_flow()
    test_dual_process_engine()
    print("\n[SUCCESS] Pillar 2 (Cognitive Dialectic System 1/2) tests passed completely!")
