"""
Master Test Suite for all 4 COLLISION Revolutionary Pillars.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_parallel_generation import test_parallel_decoder_shapes, test_parallel_sampler_generation
from tests.test_dialectic_engine import test_hegelian_dialectic_flow, test_dual_process_engine
from tests.test_ssm_hybrid import test_selective_ssm_block, test_linear_attention, test_ssm_hybrid_block
from tests.test_in_weights_verifier import test_symbolic_verifier, test_self_correcting_graph


def run_all_pillars():
    print("=" * 70)
    print("  COLLISION REVOLUTIONARY ARCHITECTURE SUITE — 4 PILLARS VERIFICATION")
    print("=" * 70)

    print("\n[PILLAR 1] Non-Autoregressive / Parallel Discrete Diffusion Generation")
    test_parallel_decoder_shapes()
    test_parallel_sampler_generation()

    print("\n[PILLAR 2] Cognitive Neuro-Symbolic & Hegelian Dialectic Dual-Process Core")
    test_hegelian_dialectic_flow()
    test_dual_process_engine()

    print("\n[PILLAR 3] State-Space (SSM) & Linear Attention Mamba Hybrid (O(1) Memory)")
    test_selective_ssm_block()
    test_linear_attention()
    test_ssm_hybrid_block()

    print("\n[PILLAR 4] Self-Correcting Execution & In-Weights Verification Graph")
    test_symbolic_verifier()
    test_self_correcting_graph()

    print("\n" + "=" * 70)
    print("  ALL 4 REVOLUTIONARY PILLARS BUILT, TESTED & 100% OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_pillars()
