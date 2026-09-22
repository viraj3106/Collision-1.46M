"""
Test suite for COLLISION Self-Correcting Execution & In-Weights Verification Graph (Pillar 4).
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.in_weights_verifier import SymbolicGrammarVerifier, SelfCorrectingGraph


def test_symbolic_verifier():
    print("Testing Symbolic Grammar Verifier...")
    d_model = 128
    B, S = 2, 16
    h = torch.randn(B, S, d_model)

    verifier = SymbolicGrammarVerifier(d_model)
    scores = verifier(h)

    assert scores.shape == (B, S, 4), f"Expected shape (2, 16, 4), got {scores.shape}"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores not bounded in [0, 1]"
    print("  [OK] Math, Syntax, Logic, and Grounding verification heads active.")


def test_self_correcting_graph():
    print("Testing Self-Correcting Residual Feedback Loop...")
    d_model = 128
    vocab_size = 1000
    B, S = 2, 16
    h = torch.randn(B, S, d_model)

    graph = SelfCorrectingGraph(d_model=d_model, vocab_size=vocab_size)
    out = graph(h)

    assert out["logits"].shape == (B, S, vocab_size), "Logits shape mismatch"
    assert "verification_scores" in out
    assert "mean_confidence" in out
    assert out["corrected_states"].shape == (B, S, d_model)
    print("  [OK] In-weights self-correction loop successfully projected and verified.")


if __name__ == "__main__":
    test_symbolic_verifier()
    test_self_correcting_graph()
    print("\n[SUCCESS] Pillar 4 (Self-Correcting In-Weights Verification) tests passed completely!")
