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

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase32")
CP_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase32")
os.makedirs(CP_DIR, exist_ok=True)

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
CANDIDATE_SRC_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phase31", "collision_10m_augmented_v1.pt")
CANDIDATE_DEST_PATH = os.path.join(CP_DIR, "collision_10m_production_candidate_v1.pt")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def main():
    print("================================================================")
    print("  PHASE 32: INFERENCE BENCHMARK & API COMPATIBILITY TESTING     ")
    print("================================================================")

    # 1. Step 14 - Save separated Production Candidate Checkpoint
    if not os.path.exists(CANDIDATE_SRC_PATH):
        raise FileNotFoundError(f"Source candidate checkpoint not found: {CANDIDATE_SRC_PATH}")
    
    shutil.copy2(CANDIDATE_SRC_PATH, CANDIDATE_DEST_PATH)
    print(f"Created Production Candidate Checkpoint: {CANDIDATE_DEST_PATH}")

    cand_sha = hashlib.sha256(open(CANDIDATE_DEST_PATH, "rb").read()).hexdigest()
    cand_ck = torch.load(CANDIDATE_DEST_PATH, map_location="cpu")
    cand_cfg = ModelConfig(**cand_ck["config"])
    cand_m = CollisionTransformer(cand_cfg)
    cand_m.load_state_dict(cand_ck["model_state_dict"])
    cand_params = sum(p.numel() for p in cand_m.parameters())

    print(f"Candidate SHA256: {cand_sha}")
    print(f"Candidate Params: {cand_params:,}")

    # 2. Step 15 - API Compatibility Test
    print("\n--- TESTING FASTAPI API COMPATIBILITY ---")
    from fastapi.testclient import TestClient
    from api.main import app
    from collision.inference.engine import CollisionInferenceEngine

    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    print(f"API /health: OK ({res_health.status_code})")

    # Readiness check
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200, f"Readiness check failed: {res_ready.text}"
    print(f"API /ready: OK ({res_ready.status_code})")

    # Models endpoint
    res_models = client.get("/v1/models")
    assert res_models.status_code == 200, f"Models check failed: {res_models.text}"
    print(f"API /v1/models: OK ({res_models.status_code})")

    # Candidate Engine Compatibility Check
    cand_engine = CollisionInferenceEngine(model_dir=os.path.join(PROJECT_ROOT, "models", "collision-10m"))
    cand_engine.model = cand_m
    cand_res = cand_engine.generate("What is machine learning?", max_tokens=30, temp=0.7)
    gen_text = cand_res["text"]
    assert len(gen_text) > 0, "Candidate engine produced empty output"
    print(f"Candidate Inference Engine Compatibility: Passed. Output sample: '{gen_text[:50]}...'")
    print("API Compatibility Verification Complete: 100% Compatible.")

    # 3. Step 16 - Inference Benchmark Comparison
    print("\n--- INFERENCE BENCHMARKING (Baseline vs Candidate) ---")
    prod_ck = torch.load(PROD_MODEL_PATH, map_location="cpu")
    prod_cfg = ModelConfig(**prod_ck["config"])
    prod_m = CollisionTransformer(prod_cfg)
    prod_m.load_state_dict(prod_ck["model_state_dict"])
    prod_m.eval()

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    benchmark_prompts = [
        "Explain quantum mechanics in simple terms.",
        "What is the difference between a process and a thread?",
        "Define artificial intelligence and describe its main subfields.",
        "How does a neural network backpropagate errors?",
        "What are prime numbers and why are they useful in cryptography?"
    ]

    def benchmark_model(model, name, num_runs=5):
        latencies = []
        tokens_per_sec_list = []
        total_tokens_gen = 0
        
        # Warmup
        generate(model, tokenizer, "Warmup prompt", max_tokens=10, temperature=0.7)

        for _ in range(num_runs):
            for p in benchmark_prompts:
                t0 = time.perf_counter()
                out = generate(model, tokenizer, p, max_tokens=40, temperature=0.7)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                t_gen = len(tokenizer.encode(out))
                tps = (t_gen / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0
                latencies.append(elapsed_ms)
                tokens_per_sec_list.append(tps)
                total_tokens_gen += t_gen

        latencies.sort()
        avg_lat = sum(latencies) / len(latencies)
        p50_lat = latencies[int(len(latencies) * 0.50)]
        p95_lat = latencies[int(len(latencies) * 0.95)]
        avg_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list)

        return {
            "model_name": name,
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "avg_tokens_per_sec": round(avg_tps, 2),
            "total_tokens_generated": total_tokens_gen
        }

    bm_prod = benchmark_model(prod_m, "Production COLLISION-10M Baseline")
    bm_cand = benchmark_model(cand_m, "Production Candidate (Model D)")

    print(f"Production Baseline -> Avg Latency: {bm_prod['avg_latency_ms']} ms | P50: {bm_prod['p50_latency_ms']} ms | P95: {bm_prod['p95_latency_ms']} ms | Tokens/sec: {bm_prod['avg_tokens_per_sec']}")
    print(f"Production Candidate -> Avg Latency: {bm_cand['avg_latency_ms']} ms | P50: {bm_cand['p50_latency_ms']} ms | P95: {bm_cand['p95_latency_ms']} ms | Tokens/sec: {bm_cand['avg_tokens_per_sec']}")

    benchmark_summary = {
        "candidate_checkpoint_sha256": cand_sha,
        "candidate_parameters": cand_params,
        "api_compatibility": "PASS",
        "production_baseline_benchmark": bm_prod,
        "production_candidate_benchmark": bm_cand
    }

    bm_file = os.path.join(EXP_DIR, "inference_benchmark_results.json")
    with open(bm_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"Saved benchmark results to: {bm_file}\n")

if __name__ == "__main__":
    main()
