"""
Test suite and speed benchmark for COLLISION Parallel Discrete Diffusion Decoder (Pillar 1).
"""

import sys
import os
import time
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.parallel_decoder import CollisionParallelDecoder
from inference.parallel_sampler import ParallelSampler


def test_parallel_decoder_shapes():
    print("Testing Parallel Decoder Forward Shapes...")
    model = CollisionParallelDecoder(
        vocab_size=1000,
        d_model=128,
        n_layer=2,
        n_head=4,
        d_ff=512,
        max_seq_len=128,
        mask_token_id=3
    )

    B, S = 2, 32
    input_ids = torch.randint(0, 1000, (B, S))
    step = torch.tensor([0, 1])

    logits, confidence = model(input_ids, step=step)

    assert logits.shape == (B, S, 1000), f"Unexpected logits shape: {logits.shape}"
    assert confidence.shape == (B, S), f"Unexpected confidence shape: {confidence.shape}"
    assert (confidence >= 0.0).all() and (confidence <= 1.0).all(), "Confidence scores outside [0, 1]"
    print("  [OK] Forward pass shapes and confidence calibration valid.")


def test_parallel_sampler_generation():
    print("Testing Parallel Sampler Multi-Token Generation...")
    model = CollisionParallelDecoder(
        vocab_size=1000,
        d_model=128,
        n_layer=2,
        n_head=4,
        d_ff=512,
        max_seq_len=128,
        mask_token_id=3
    )
    sampler = ParallelSampler(model, mask_token_id=3)

    prompt = torch.tensor([[10, 20, 30, 40]])  # Prompt of 4 tokens
    new_tokens = 24
    num_iterations = 4

    start_t = time.perf_counter()
    result = sampler.sample(
        prompt_ids=prompt,
        max_new_tokens=new_tokens,
        num_iterations=num_iterations,
        temperature=0.8
    )
    elapsed = time.perf_counter() - start_t

    generated_ids = result["generated_ids"]
    assert generated_ids.shape == (1, 4 + new_tokens), f"Output shape mismatch: {generated_ids.shape}"
    # Verify no mask token remains in the final output
    assert (generated_ids != 3).all(), "Mask tokens found in final generation output!"
    assert len(result["history"]) == num_iterations + 1

    tok_per_sec = new_tokens / max(elapsed, 1e-6)
    print(f"  [OK] Generated {new_tokens} tokens in {num_iterations} parallel passes!")
    print(f"  [BENCHMARK] Parallel Speed: {tok_per_sec:.2f} tokens/sec in simulated Python runtime.")


if __name__ == "__main__":
    test_parallel_decoder_shapes()
    test_parallel_sampler_generation()
    print("\n[SUCCESS] Pillar 1 (Parallel Discrete Diffusion) tests passed completely!")
