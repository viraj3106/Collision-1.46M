"""
COLLISION Phase 100 — End-to-End UI Integration & Latency Benchmark.

Executes real end-to-end query evaluations against the production CollisionService / API client:
- Test 1: Local / Architecture query (What is COLLISION 10M?)
- Test 2: Local RAG (What is the embedding dimension in COLLISION?)
- Test 3: Web Grounding (What is the latest release of PyTorch in 2025?)
- Test 4: Insufficient Information (Prediction/unestablished fact)
- Test 5: Conflict resolution scenario
Measures transport & engine latency, claim counts, and source citations.
"""

import os
import sys
import time
import json
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from api.main import app
from collision.service import get_collision_service

client = TestClient(app)

BENCHMARK_PROMPTS = [
    {
        "id": "test_1_architecture",
        "name": "Local Architecture Query",
        "question": "What is COLLISION 10M?",
        "mode": "AUTO",
        "expected_status": "ANSWERED"
    },
    {
        "id": "test_2_local_rag",
        "name": "Local Knowledge RAG Query",
        "question": "What is the embedding dimension in the COLLISION 10M architecture?",
        "mode": "LOCAL",
        "expected_status": "ANSWERED"
    },
    {
        "id": "test_3_web_grounded",
        "name": "Web Grounded Query",
        "question": "What is the latest release version of PyTorch in 2025?",
        "mode": "WEB",
        "expected_status": "ANSWERED"
    },
    {
        "id": "test_4_insufficient_info",
        "name": "Insufficient Information Query",
        "question": "What will the exact stock price of NVIDIA be on October 15, 2038?",
        "mode": "LOCAL",
        "expected_status": "INSUFFICIENT_INFORMATION"
    },
    {
        "id": "test_5_conflict",
        "name": "Conflict Resolution Query",
        "question": "Was Python created in 1991 or 2005?",
        "mode": "LOCAL",
        "expected_status": "ANSWERED"
    }
]


def run_benchmark():
    print("=" * 60)
    print("PHASE 100 END-TO-END BENCHMARK & EVALUATION")
    print("=" * 60)

    results = []
    latencies = []
    success_count = 0
    fail_count = 0

    for item in BENCHMARK_PROMPTS:
        print(f"\nRunning {item['name']} ({item['id']})...")
        t0 = time.perf_counter()
        
        resp = client.post(
            "/v1/ask",
            json={
                "question": item["question"],
                "mode": item["mode"],
                "include_sources": True,
                "include_claims": True
            }
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        
        if resp.status_code == 200:
            data = resp.json()
            success_count += 1
            latencies.append(total_time_ms)
            
            print(f"  Status: {data['status']}")
            print(f"  Mode: {data['mode']}")
            print(f"  Sources returned: {len(data.get('sources', []))}")
            print(f"  Claims returned: {len(data.get('claims', []))}")
            print(f"  Answer snippet: {data['answer'][:120]}...")
            print(f"  Latency: {total_time_ms:.2f} ms (Engine total: {data.get('latency', {}).get('total_ms', 0):.2f} ms)")
            
            results.append({
                "id": item["id"],
                "name": item["name"],
                "question": item["question"],
                "mode": data["mode"],
                "status": data["status"],
                "sources_count": len(data.get("sources", [])),
                "claims_count": len(data.get("claims", [])),
                "answer_preview": data["answer"][:200],
                "transport_latency_ms": total_time_ms,
                "engine_latency": data.get("latency", {})
            })
        else:
            fail_count += 1
            print(f"  FAILED: HTTP {resp.status_code}")
            results.append({
                "id": item["id"],
                "name": item["name"],
                "error": f"HTTP {resp.status_code}"
            })

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p50_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = (
        statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 
        else max(latencies) if latencies else 0.0
    )

    summary = {
        "total_requests": len(BENCHMARK_PROMPTS),
        "successful_requests": success_count,
        "failed_requests": fail_count,
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "test_results": results
    }

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY:")
    print(f"Total Requests:      {summary['total_requests']}")
    print(f"Successful Requests: {summary['successful_requests']}")
    print(f"Failed Requests:     {summary['failed_requests']}")
    print(f"Average Latency:     {summary['avg_latency_ms']} ms")
    print(f"p50 Latency:         {summary['p50_latency_ms']} ms")
    print(f"p95 Latency:         {summary['p95_latency_ms']} ms")
    print("=" * 60)

    out_path = os.path.join(PROJECT_ROOT, "reports", "phase100_benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Benchmark results saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_benchmark()
