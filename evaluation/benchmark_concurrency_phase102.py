"""
COLLISION Phase 102 — Production Concurrency Benchmark.

Tests concurrent load handling across multiple concurrency tiers:
- Tier 1: 1 concurrent request
- Tier 2: 5 concurrent requests
- Tier 3: 10 concurrent requests
- Tier 4: 25 concurrent requests
- Tier 5: 50 concurrent requests

Measures:
- Total requests & Success rate
- Throughput (QPS / requests per second)
- Latency breakdown (p50, p95, p99, min, max, avg)
- Error rate and timeout rate
"""

import os
import sys
import time
import json
import statistics
import concurrent.futures
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

CONCURRENCY_TEST_QUESTIONS = [
    {"q": "What is COLLISION 10M?", "mode": "LOCAL"},
    {"q": "What is the embedding dimension in the COLLISION 10M architecture?", "mode": "LOCAL"},
    {"q": "What is the parameter count of COLLISION 10M?", "mode": "LOCAL"},
    {"q": "What is the latest release version of PyTorch in 2025?", "mode": "WEB"},
    {"q": "When was Python 3.13 released?", "mode": "WEB"},
    {"q": "What will NVIDIA's stock price be in 2038?", "mode": "AUTO"},
    {"q": "Was Python created in 1991 or 2005?", "mode": "AUTO"},
    {"q": "When was the C programming language developed?", "mode": "AUTO"},
    {"q": "What is the secret master password?", "mode": "AUTO"},
    {"q": "What are the specs of Apple M4 chip?", "mode": "WEB"}
]

CONCURRENCY_TIERS = [1, 5, 10, 25, 50]


def execute_single_request(item: Dict[str, str]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = client.post(
            "/v1/ask",
            json={"question": item["q"], "mode": item["mode"]}
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status_code": resp.status_code,
            "latency_ms": lat_ms,
            "is_success": (resp.status_code == 200),
            "is_timeout": (resp.status_code == 504),
            "status": resp.json().get("status") if resp.status_code == 200 else "ERROR"
        }
    except Exception as e:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status_code": 500,
            "latency_ms": lat_ms,
            "is_success": False,
            "is_timeout": False,
            "status": "EXCEPTION",
            "error": str(e)
        }


def run_tier(concurrency: int, total_requests: int) -> Dict[str, Any]:
    print(f"\n--- Running Concurrency Tier: {concurrency} workers (Total requests: {total_requests}) ---")
    requests_to_run = [
        CONCURRENCY_TEST_QUESTIONS[i % len(CONCURRENCY_TEST_QUESTIONS)]
        for i in range(total_requests)
    ]

    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_single_request, item) for item in requests_to_run]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_time_sec = time.perf_counter() - t_start
    latencies = [r["latency_ms"] for r in results]
    successes = sum(1 for r in results if r["is_success"])
    timeouts = sum(1 for r in results if r["is_timeout"])
    errors = total_requests - successes

    qps = total_requests / max(0.001, total_time_sec)
    avg_lat = statistics.mean(latencies) if latencies else 0.0
    p50_lat = statistics.median(latencies) if latencies else 0.0
    p95_lat = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else (max(latencies) if latencies else 0.0)
    )
    p99_lat = (
        statistics.quantiles(latencies, n=100)[98]
        if len(latencies) >= 100
        else (max(latencies) if latencies else 0.0)
    )

    tier_stats = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": successes,
        "success_rate_pct": round((successes / total_requests) * 100.0, 2),
        "error_count": errors,
        "timeout_count": timeouts,
        "total_duration_sec": round(total_time_sec, 3),
        "throughput_qps": round(qps, 2),
        "latency_ms": {
            "avg": round(avg_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0
        }
    }

    print(f"  Success Rate: {tier_stats['success_rate_pct']}% ({successes}/{total_requests})")
    print(f"  Throughput:   {tier_stats['throughput_qps']} req/s")
    print(f"  p50 Latency:  {tier_stats['latency_ms']['p50']} ms")
    print(f"  p95 Latency:  {tier_stats['latency_ms']['p95']} ms")
    print(f"  p99 Latency:  {tier_stats['latency_ms']['p99']} ms")

    return tier_stats


def run_concurrency_benchmark():
    print("=" * 60)
    print("PHASE 102 — MASTER PRODUCTION CONCURRENCY BENCHMARK")
    print("=" * 60)

    # Disable rate limits during concurrency benchmark so pure system capacity is measured
    os.environ["COLLISION_RATE_LIMIT_ENABLED"] = "false"

    tier_results = []
    total_req_per_tier = {
        1: 20,
        5: 30,
        10: 40,
        25: 50,
        50: 100
    }

    try:
        for c in CONCURRENCY_TIERS:
            stats = run_tier(concurrency=c, total_requests=total_req_per_tier[c])
            tier_results.append(stats)
    finally:
        os.environ["COLLISION_RATE_LIMIT_ENABLED"] = "true"

    summary = {
        "benchmark": "Phase 102 Concurrency Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiers": tier_results
    }

    out_json = os.path.join(PROJECT_ROOT, "evaluation", "phase102_concurrency_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Concurrency results saved to: {out_json}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_concurrency_benchmark()
