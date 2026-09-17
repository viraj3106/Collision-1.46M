"""
COLLISION Empirical Scaling Ladder Benchmark.

Evaluates scaling trajectories from 1.46M to 100M parameters across 50 training steps per tier.
Measures parameter counts, step latencies, loss descent, memory footprint, and compute ceilings.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer

TIERS = [
    {
        "name": "COLLISION-1.46M (Original Base)",
        "d_model": 192,
        "n_layer": 4,
        "n_head": 4,
        "d_ff": 384,
        "vocab_size": 4000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-3.5M (Compact)",
        "d_model": 256,
        "n_layer": 6,
        "n_head": 4,
        "d_ff": 512,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-7.2M (Intermediate)",
        "d_model": 320,
        "n_layer": 6,
        "n_head": 8,
        "d_ff": 640,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-10.28M (Flagship Current)",
        "d_model": 384,
        "n_layer": 6,
        "n_head": 8,
        "d_ff": 768,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-15.8M (Expanded)",
        "d_model": 448,
        "n_layer": 8,
        "n_head": 8,
        "d_ff": 896,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-25.2M (Heavy)",
        "d_model": 512,
        "n_layer": 10,
        "n_head": 8,
        "d_ff": 1024,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-50.4M (Ultra)",
        "d_model": 640,
        "n_layer": 12,
        "n_head": 10,
        "d_ff": 1536,
        "vocab_size": 16000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-102.5M (Max Scaled)",
        "d_model": 768,
        "n_layer": 16,
        "n_head": 12,
        "d_ff": 2048,
        "vocab_size": 32000,
        "max_seq_len": 256
    }
]


def load_sample_batch(bin_path: str, batch_size: int, seq_len: int, vocab_size: int, step: int) -> tuple:
    """Loads batch of sequential tokens from dataset bin, clamped to vocab_size."""
    if os.path.exists(bin_path):
        data = np.fromfile(bin_path, dtype=np.uint16)
        total_tokens = len(data)
        offset = (step * batch_size * seq_len) % max(1, total_tokens - batch_size * seq_len - 10)
        chunk = data[offset : offset + batch_size * seq_len + 1].astype(np.int64)
        if len(chunk) == batch_size * seq_len + 1:
            chunk = np.clip(chunk, 0, vocab_size - 1)
            x = torch.from_numpy(chunk[:-1]).reshape(batch_size, seq_len)
            y = torch.from_numpy(chunk[1:]).reshape(batch_size, seq_len)
            return x, y
            
    # Synthetic deterministic fallback if bin is unavailable
    torch.manual_seed(step + 42)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


def benchmark_tier(tier: Dict[str, Any], steps: int = 50, batch_size: int = 4, device: str = "cpu") -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking Tier: {tier['name']}")
    print(f"Layers: {tier['n_layer']} | d_model: {tier['d_model']} | Heads: {tier['n_head']} | Vocab: {tier['vocab_size']}")
    print(f"=======================================================")

    cfg = ModelConfig(
        vocab_size=tier["vocab_size"],
        max_seq_len=tier["max_seq_len"],
        d_model=tier["d_model"],
        n_layer=tier["n_layer"],
        n_head=tier["n_head"],
        d_ff=tier["d_ff"],
        dropout=0.1,
        tie_embeddings=True
    )

    model = CollisionTransformer(cfg)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Exact Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)

    bin_path = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v5_expanded", "train.bin")
    
    step_latencies = []
    losses = []
    tokens_per_step = batch_size * tier["max_seq_len"]

    t_bench_start = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        
        x, y = load_sample_batch(bin_path, batch_size, tier["max_seq_len"], tier["vocab_size"], step)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t_step = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(t_step)
        loss_val = float(loss.item())
        losses.append(loss_val)

        if step == 1 or step % 10 == 0 or step == steps:
            print(f"  Step {step:02d}/{steps:02d} | Loss: {loss_val:.4f} | Latency: {t_step:.1f}ms | Throughput: {tokens_per_step / (t_step / 1000.0):.1f} tok/s")

    total_time_s = time.perf_counter() - t_bench_start
    avg_latency_ms = float(np.mean(step_latencies))
    avg_tokens_per_sec = tokens_per_step / (avg_latency_ms / 1000.0)
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    steps_for_10m = 10_000_000 / tokens_per_step
    est_hours_10m_cpu = (steps_for_10m * (avg_latency_ms / 1000.0)) / 3600.0
    est_hours_10m_gpu = est_hours_10m_cpu / 25.0

    result = {
        "name": tier["name"],
        "parameters": total_params,
        "params_m": round(total_params / 1e6, 2),
        "d_model": tier["d_model"],
        "n_layer": tier["n_layer"],
        "n_head": tier["n_head"],
        "d_ff": tier["d_ff"],
        "vocab_size": tier["vocab_size"],
        "initial_loss": round(losses[0], 4),
        "final_loss_step_50": round(losses[-1], 4),
        "delta_loss": round(losses[0] - losses[-1], 4),
        "avg_step_latency_ms": round(avg_latency_ms, 2),
        "total_time_50_steps_s": round(total_time_s, 2),
        "tokens_per_second": round(avg_tokens_per_sec, 2),
        "ram_mb": round(mem_after, 2),
        "est_hours_10m_tokens_cpu": round(est_hours_10m_cpu, 2),
        "est_hours_10m_tokens_gpu": round(est_hours_10m_gpu, 2)
    }

    print(f"--> Summary: Avg Latency: {avg_latency_ms:.1f}ms | Loss: {losses[0]:.4f} -> {losses[-1]:.4f} | Est 10M on CPU: {est_hours_10m_cpu:.2f}h")
    return result


def main():
    print("=================================================================")
    print("COLLISION FULL SCALING LADDER BENCHMARK (50 STEPS PER TIER)")
    print("=================================================================")

    results = []
    for tier in TIERS:
        res = benchmark_tier(tier, steps=50, batch_size=4, device="cpu")
        results.append(res)

    out_dir = os.path.join(PROJECT_ROOT, "experiments", "scaling_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "scaling_ladder_results.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=================================================================")
    print(f"Benchmark completed for all {len(results)} tiers! Saved to {out_file}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
