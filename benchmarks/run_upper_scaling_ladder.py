"""
COLLISION Upper Scaling Ladder Benchmark (15.8M to Final Max-Scaled).

Runs 50 training optimization steps per tier from 15.8M up to 250M parameters.
Measures step time, memory scaling, convergence rate, and compute boundaries.
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

UPPER_TIERS = [
    {
        "name": "COLLISION-15.8M (Expanded Tier)",
        "d_model": 448,
        "n_layer": 8,
        "n_head": 8,
        "d_ff": 896,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-25.2M (Heavy Tier)",
        "d_model": 512,
        "n_layer": 10,
        "n_head": 8,
        "d_ff": 1024,
        "vocab_size": 8000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-50.4M (Ultra Tier)",
        "d_model": 640,
        "n_layer": 12,
        "n_head": 10,
        "d_ff": 1536,
        "vocab_size": 16000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-102.5M (Sub-Billion Base)",
        "d_model": 768,
        "n_layer": 16,
        "n_head": 12,
        "d_ff": 2048,
        "vocab_size": 32000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-150M (Dense Frontier)",
        "d_model": 896,
        "n_layer": 16,
        "n_head": 14,
        "d_ff": 2560,
        "vocab_size": 32000,
        "max_seq_len": 256
    },
    {
        "name": "COLLISION-250M (Final Max-Scaled Boundary)",
        "d_model": 1024,
        "n_layer": 20,
        "n_head": 16,
        "d_ff": 3072,
        "vocab_size": 32000,
        "max_seq_len": 256
    }
]


def load_batch(bin_path: str, batch_size: int, seq_len: int, vocab_size: int, step: int) -> tuple:
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

    torch.manual_seed(step + 42)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


def benchmark_upper_tier(tier: Dict[str, Any], steps: int = 50, batch_size: int = 4, device: str = "cpu") -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"[*] Benchmarking Tier: {tier['name']}", flush=True)
    print(f"    Architecture: d_model={tier['d_model']}, layers={tier['n_layer']}, heads={tier['n_head']}, d_ff={tier['d_ff']}, vocab={tier['vocab_size']}", flush=True)
    print(f"=======================================================", flush=True)

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
    print(f"    -> Exact Parameter Count: {total_params:,} ({total_params / 1e6:.2f}M)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    bin_path = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v5_expanded", "train.bin")
    
    step_latencies = []
    losses = []
    tokens_per_step = batch_size * tier["max_seq_len"]

    t_bench_start = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        
        x, y = load_batch(bin_path, batch_size, tier["max_seq_len"], tier["vocab_size"], step)
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
            print(f"    [Step {step:02d}/50] Loss: {loss_val:.4f} | Latency: {t_step:.1f}ms | Throughput: {tokens_per_step / (t_step / 1000.0):.1f} tok/s", flush=True)

    total_time_s = time.perf_counter() - t_bench_start
    avg_latency_ms = float(np.mean(step_latencies))
    avg_tokens_per_sec = tokens_per_step / (avg_latency_ms / 1000.0)
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # 10M token training duration estimates:
    steps_for_10m = 10_000_000 / tokens_per_step
    est_hours_10m_cpu = (steps_for_10m * (avg_latency_ms / 1000.0)) / 3600.0
    est_hours_10m_gpu = est_hours_10m_cpu / 30.0 # GPU acceleration factor

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

    print(f"    => Tier Done: Avg Latency: {avg_latency_ms:.1f}ms | Loss Drop: {losses[0]:.4f} -> {losses[-1]:.4f} (Delta -{losses[0]-losses[-1]:.4f}) | Est 10M on CPU: {est_hours_10m_cpu:.1f} hrs\n", flush=True)
    return result


def main():
    print("=================================================================", flush=True)
    print("COLLISION UPPER SCALING LADDER (15.8M -> 250M FINAL MAX SCALED)", flush=True)
    print("=================================================================", flush=True)

    results = []
    for tier in UPPER_TIERS:
        res = benchmark_upper_tier(tier, steps=50, batch_size=4, device="cpu")
        results.append(res)

    out_dir = os.path.join(PROJECT_ROOT, "experiments", "scaling_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "upper_scaling_results.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"=================================================================", flush=True)
    print(f"All upper scaling tiers completed! Saved to {out_file}", flush=True)
    print(f"=================================================================", flush=True)


if __name__ == "__main__":
    main()
