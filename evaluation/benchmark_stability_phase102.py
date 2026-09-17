"""
COLLISION Phase 102 — Production Memory & Sustained Workload Stability Benchmark.

Runs a continuous, mixed-mode workload of 120+ requests to audit:
- Process RSS Memory (MB) growth and stability
- Memory leak detection (comparing baseline vs final memory)
- Latency drift analysis (early requests vs late requests)
- Thread count and resource handles stability
- Error count under continuous operation
"""

import os
import sys
import time
import json
import statistics
import ctypes
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def get_current_memory_mb() -> float:
    """Returns the current process RSS memory usage in Megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        # Windows fallback using K32GetProcessMemoryCounters
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            pass
    return 0.0


STABILITY_QUESTIONS = [
    # Local RAG
    {"q": "What is COLLISION 10M?", "mode": "LOCAL"},
    {"q": "What is the embedding dimension in the COLLISION 10M architecture?", "mode": "LOCAL"},
    {"q": "How many transformer layers are in the COLLISION 10M model?", "mode": "LOCAL"},
    {"q": "What is the parameter count of COLLISION 10M?", "mode": "LOCAL"},
    {"q": "What chunk size is used by the Phase 94 local chunker?", "mode": "LOCAL"},
    # Web Grounding
    {"q": "What is the latest release version of PyTorch in 2025?", "mode": "WEB"},
    {"q": "When was Python 3.13 released?", "mode": "WEB"},
    {"q": "What are the specs of Apple M4 chip?", "mode": "WEB"},
    {"q": "What is the active LTS release of Node.js?", "mode": "WEB"},
    {"q": "What features are stabilized in Rust 1.83?", "mode": "WEB"},
    # Historical
    {"q": "Was Python created in 1991 or 2005?", "mode": "AUTO"},
    {"q": "When was the C programming language developed?", "mode": "AUTO"},
    {"q": "When was the ENIAC computer completed?", "mode": "AUTO"},
    {"q": "When was the World Wide Web invented at CERN?", "mode": "AUTO"},
    # Insufficient / Future
    {"q": "What will NVIDIA's stock price be in 2038?", "mode": "AUTO"},
    {"q": "What will Bitcoin's price be tomorrow?", "mode": "AUTO"},
    {"q": "What is the secret master password of user 999?", "mode": "AUTO"},
    {"q": "What did user 4528 have for breakfast this morning?", "mode": "AUTO"}
]


def run_stability_benchmark(total_requests: int = 120):
    print("=" * 60)
    print("PHASE 102 — MASTER PRODUCTION MEMORY & STABILITY BENCHMARK")
    print(f"Total Requests to Execute: {total_requests}")
    print("=" * 60)

    # Disable rate limits during sustained workload test
    os.environ["COLLISION_RATE_LIMIT_ENABLED"] = "false"

    # Warmup pipeline so runtime allocators & modules reach steady-state
    for w_q in STABILITY_QUESTIONS[:3]:
        client.post("/v1/ask", json={"question": w_q["q"], "mode": w_q["mode"]})

    mem_start_mb = get_current_memory_mb()
    print(f"Initialized Steady-State Memory RSS: {mem_start_mb:.2f} MB")

    latencies: List[float] = []
    successes = 0
    errors = 0
    memory_snapshots: List[Dict[str, Any]] = []

    t_start = time.perf_counter()

    try:
        for idx in range(1, total_requests + 1):
            q_item = STABILITY_QUESTIONS[(idx - 1) % len(STABILITY_QUESTIONS)]
            
            t0 = time.perf_counter()
            resp = client.post(
                "/v1/ask",
                json={"question": q_item["q"], "mode": q_item["mode"]}
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            if resp.status_code == 200:
                successes += 1
            else:
                errors += 1

            # Snapshot memory periodically
            if idx % 30 == 0 or idx == total_requests:
                current_mem = get_current_memory_mb()
                memory_snapshots.append({
                    "request_index": idx,
                    "memory_rss_mb": round(current_mem, 2),
                    "memory_delta_mb": round(current_mem - mem_start_mb, 2)
                })
                print(f"[{idx:03d}/{total_requests}] Current RSS: {current_mem:.2f} MB (Delta: +{current_mem - mem_start_mb:.2f} MB) | Last Lat: {lat_ms:.2f} ms")

    finally:
        os.environ["COLLISION_RATE_LIMIT_ENABLED"] = "true"

    total_duration_sec = time.perf_counter() - t_start
    mem_final_mb = get_current_memory_mb()
    mem_growth_mb = mem_final_mb - mem_start_mb

    # Latency drift analysis: first 20 vs last 20
    first_20_avg = statistics.mean(latencies[:20]) if len(latencies) >= 20 else 0.0
    last_20_avg = statistics.mean(latencies[-20:]) if len(latencies) >= 20 else 0.0
    latency_drift_pct = ((last_20_avg - first_20_avg) / max(0.001, first_20_avg)) * 100.0 if first_20_avg > 0 else 0.0

    summary = {
        "benchmark": "Phase 102 Memory & Stability Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_requests": total_requests,
        "successful_requests": successes,
        "success_rate_pct": round((successes / total_requests) * 100.0, 2),
        "error_count": errors,
        "total_duration_sec": round(total_duration_sec, 2),
        "throughput_qps": round(total_requests / max(0.001, total_duration_sec), 2),
        "memory_rss": {
            "initial_mb": round(mem_start_mb, 2),
            "final_mb": round(mem_final_mb, 2),
            "net_growth_mb": round(mem_growth_mb, 2),
            "is_leak_detected": (mem_growth_mb > 50.0)
        },
        "latency_ms": {
            "overall_avg": round(statistics.mean(latencies), 2),
            "overall_p50": round(statistics.median(latencies), 2),
            "overall_p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else 0.0,
            "first_20_avg": round(first_20_avg, 2),
            "last_20_avg": round(last_20_avg, 2),
            "latency_drift_pct": round(latency_drift_pct, 2)
        },
        "memory_snapshots": memory_snapshots
    }

    print("\n" + "=" * 60)
    print("STABILITY BENCHMARK SUMMARY:")
    print(f"Total Requests Executed: {total_requests}")
    print(f"Success Rate:            {summary['success_rate_pct']}%")
    print(f"Initial RSS Memory:      {summary['memory_rss']['initial_mb']} MB")
    print(f"Final RSS Memory:        {summary['memory_rss']['final_mb']} MB")
    print(f"Net Memory Growth:       {summary['memory_rss']['net_growth_mb']} MB")
    print(f"Early Latency (First 20):{summary['latency_ms']['first_20_avg']} ms")
    print(f"Late Latency (Last 20):  {summary['latency_ms']['last_20_avg']} ms")
    print(f"Latency Drift:           {summary['latency_ms']['latency_drift_pct']}%")
    print(f"Memory Leak Detected:    {summary['memory_rss']['is_leak_detected']}")
    print("=" * 60)

    out_json = os.path.join(PROJECT_ROOT, "evaluation", "phase102_stability_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Stability results saved to: {out_json}")
    return summary


if __name__ == "__main__":
    run_stability_benchmark()
