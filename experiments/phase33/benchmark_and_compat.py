import os
import sys
import time
import json
import torch
import shutil
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import generate

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase33")
CP_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase33")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
MODEL_D_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phase32", "collision_10m_production_candidate_v1.pt")
MODEL_E_PATH = os.path.join(CP_DIR, "collision_10m_production_candidate_v2.pt")

def main():
    print("================================================================")
    print("  PHASE 33: INFERENCE BENCHMARK & API COMPATIBILITY TESTING     ")
    print("================================================================")

    if not os.path.exists(MODEL_E_PATH):
        raise FileNotFoundError(f"Model E checkpoint not found at: {MODEL_E_PATH}")

    e_sha = hashlib.sha256(open(MODEL_E_PATH, "rb").read()).hexdigest()
    e_ck = torch.load(MODEL_E_PATH, map_location="cpu")
    e_cfg = ModelConfig(**e_ck["config"])
    e_m = CollisionTransformer(e_cfg)
    e_m.load_state_dict(e_ck["model_state_dict"])
    e_params = sum(p.numel() for p in e_m.parameters())

    print(f"Candidate Model E Checkpoint: {MODEL_E_PATH}")
    print(f"Model E SHA256: {e_sha}")
    print(f"Model E Parameters: {e_params:,}")

    # API Compatibility Test
    print("\n--- FASTAPI ENDPOINT COMPATIBILITY ---")
    from fastapi.testclient import TestClient
    from api.main import app
    from collision.inference.engine import CollisionInferenceEngine

    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200, "Health check failed"
    print("API /health: OK")

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200, "Readiness check failed"
    print("API /ready: OK")

    res_models = client.get("/v1/models")
    assert res_models.status_code == 200, "Models check failed"
    print("API /v1/models: OK")

    cand_engine = CollisionInferenceEngine(model_dir=os.path.join(PROJECT_ROOT, "models", "collision-10m"))
    cand_engine.model = e_m
    res_gen = cand_engine.generate("What is machine learning?", max_tokens=30, temp=0.7)
    assert len(res_gen["text"]) > 0, "Model E output empty"
    print(f"Candidate Model E Inference Engine Test: Passed. Output sample: '{res_gen['text'][:50]}...'")

    # Benchmarking A vs D vs E
    print("\n--- INFERENCE BENCHMARKING (A vs D vs E) ---")
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    benchmark_prompts = [
        "Explain quantum mechanics in simple terms.",
        "What is the difference between a process and a thread?",
        "Define artificial intelligence and describe its main subfields.",
        "How does a neural network backpropagate errors?",
        "What are prime numbers and why are they useful in cryptography?"
    ]

    def load_model(p_path):
        ck = torch.load(p_path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        m = CollisionTransformer(cfg)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        return m

    m_A = load_model(PROD_MODEL_PATH)
    m_D = load_model(MODEL_D_PATH)
    m_E = e_m.eval()

    def benchmark_model(model, name, num_runs=3):
        latencies = []
        tps_list = []
        tot_tokens = 0
        
        generate(model, tokenizer, "Warmup", max_tokens=10, temperature=0.7)
        for _ in range(num_runs):
            for p in benchmark_prompts:
                t0 = time.perf_counter()
                out = generate(model, tokenizer, p, max_tokens=40, temperature=0.7)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                t_gen = len(tokenizer.encode(out))
                tps = (t_gen / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                latencies.append(elapsed_ms)
                tps_list.append(tps)
                tot_tokens += t_gen

        latencies.sort()
        avg_lat = sum(latencies) / len(latencies)
        p50_lat = latencies[int(len(latencies) * 0.50)]
        p95_lat = latencies[int(len(latencies) * 0.95)]
        avg_tps = sum(tps_list) / len(tps_list)

        return {
            "model_name": name,
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "total_tokens": tot_tokens
        }

    bm_A = benchmark_model(m_A, "Model A Baseline")
    bm_D = benchmark_model(m_D, "Model D Phase 32")
    bm_E = benchmark_model(m_E, "Model E Phase 33")

    print(f"Model A Baseline -> Avg Latency: {bm_A['avg_latency_ms']} ms | Tokens/sec: {bm_A['avg_tokens_per_sec']}")
    print(f"Model D Candidate -> Avg Latency: {bm_D['avg_latency_ms']} ms | Tokens/sec: {bm_D['avg_tokens_per_sec']}")
    print(f"Model E Candidate -> Avg Latency: {bm_E['avg_latency_ms']} ms | Tokens/sec: {bm_E['avg_tokens_per_sec']}")

    summary = {
        "candidate_e_sha256": e_sha,
        "candidate_e_parameters": e_params,
        "api_compatibility": "PASS",
        "benchmark_A": bm_A,
        "benchmark_D": bm_D,
        "benchmark_E": bm_E
    }

    out_file = os.path.join(EXP_DIR, "inference_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved inference benchmark results to: {out_file}\n")

if __name__ == "__main__":
    main()
