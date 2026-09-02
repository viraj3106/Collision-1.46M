import os
import sys
import time
import psutil
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app

def get_process_memory():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB
    except Exception:
        return 0.0

def run_benchmark():
    print("Starting Phase 16 Inference Benchmark...")
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "phase16"), exist_ok=True)
    
    # 1. Measure Model Loading Time
    t0 = time.time()
    client = TestClient(app)
    # Trigger first endpoint to verify it is fully loaded
    client.get("/health")
    load_time = time.time() - t0
    
    # Obtain coldstart metrics
    coldstart_res = client.get("/coldstart").json()
    model_load_time = coldstart_res.get("model_load_time_seconds", load_time)
    
    print(f"Model Load & Warmup Time: {model_load_time:.3f} seconds")
    
    # 2. Sequential Benchmark Loop
    prompt = "What is machine learning?"
    payload = {
        "model": "collision-10m",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
    
    latencies = []
    tokens_per_sec = []
    generated_tokens = []
    memory_usages = []
    
    print("\nRunning 10 sequential inference requests...")
    for i in range(10):
        t_req = time.time()
        res = client.post("/v1/generate", json=payload)
        req_latency = (time.time() - t_req) * 1000
        
        assert res.status_code == 200, f"Request {i+1} failed"
        data = res.json()
        
        # Collect metrics
        latencies.append(req_latency)
        tokens_per_sec.append(data["performance"]["tokens_per_second"])
        generated_tokens.append(data["usage"]["completion_tokens"])
        memory_usages.append(get_process_memory())
        
        print(f" Request {i+1}/10: Generated {data['usage']['completion_tokens']} tokens in {req_latency:.1f}ms ({data['performance']['tokens_per_second']:.1f} tok/s)")
        
    # Calculate statistics
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    avg_tps = np.mean(tokens_per_sec)
    total_tokens = sum(generated_tokens)
    avg_mem = np.mean(memory_usages)
    peak_mem = np.max(memory_usages)
    
    # 3. Write benchmark markdown report
    report_content = f"""# COLLISION-10M Inference Benchmark Report

This report presents performance metrics of the COLLISION-10M model running sequentially on CPU.

## Model Setup
* **Model ID**: `collision-10m`
* **Checkpoint Hash**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
* **Model Loading & Warmup Time**: {model_load_time:.3f} seconds

## Performance Summary (10 Sequential Requests)
* **Prompt**: "{prompt}"
* **Settings**: max_tokens=100, temperature=0.7, top_k=50, top_p=0.9
* **Total Generated Tokens**: {total_tokens}

| Metric | Value |
| :--- | :---: |
| **Average Latency** | {avg_latency:.1f} ms |
| **Median Latency** | {median_latency:.1f} ms |
| **Minimum Latency** | {min_latency:.1f} ms |
| **Maximum Latency** | {max_latency:.1f} ms |
| **Average Throughput** | {avg_tps:.2f} tokens/sec |
| **Average RAM Usage** | {avg_mem:.2f} MB |
| **Peak RAM Usage** | {peak_mem:.2f} MB |

## Execution Logs

"""
    for i in range(10):
        report_content += f"* **Request {i+1}**: Latency: {latencies[i]:.1f} ms | Completion Tokens: {generated_tokens[i]} | Throughput: {tokens_per_sec[i]:.2f} tokens/sec | RAM: {memory_usages[i]:.1f} MB\n"
        
    benchmark_md_path = os.path.join(PROJECT_ROOT, "experiments", "phase16", "inference_benchmark.md")
    with open(benchmark_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nInference benchmark report written to {benchmark_md_path}")

if __name__ == "__main__":
    run_benchmark()
