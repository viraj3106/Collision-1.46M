"""
COLLISION Phase 99 — Production API & Engine Stabilization Performance Benchmark.

Evaluates the unified CollisionService application layer across 50 controlled benchmark queries:
- Local RAG queries (10)
- Web Grounding queries (10)
- Model-Only Conversational & Math queries (10)
- Insufficient Information queries (10)
- Conflicting Evidence & Adversarial queries (10)

Measures:
- Request count
- Success rate
- Error rate
- Latency metrics: Average, Min, Max, p50, p95
"""

import os
import sys
import time
import json
import statistics
import hashlib
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.service import get_collision_service, CollisionService

BENCHMARK_50: List[Dict[str, Any]] = [
    # 1. Local RAG (10)
    {"id": "P99_LOC_01", "category": "Local-RAG", "question": "What is the embedding dimension in the COLLISION 10M architecture?"},
    {"id": "P99_LOC_02", "category": "Local-RAG", "question": "How many transformer layers are used in COLLISION 10M?"},
    {"id": "P99_LOC_03", "category": "Local-RAG", "question": "What is the exact parameter count of the flagship COLLISION architecture?"},
    {"id": "P99_LOC_04", "category": "Local-RAG", "question": "What is the feed-forward hidden dimension in COLLISION 10M?"},
    {"id": "P99_LOC_05", "category": "Local-RAG", "question": "What maximum context length is supported by COLLISION 10M?"},
    {"id": "P99_LOC_06", "category": "Local-RAG", "question": "What attention head count is configured in COLLISION 10M?"},
    {"id": "P99_LOC_07", "category": "Local-RAG", "question": "What chunk size is configured in the Phase 94 local chunker?"},
    {"id": "P99_LOC_08", "category": "Local-RAG", "question": "What chunk overlap setting is configured in the Phase 94 local chunker?"},
    {"id": "P99_LOC_09", "category": "Local-RAG", "question": "What is the default top-k retrieval count in COLLISION retriever?"},
    {"id": "P99_LOC_10", "category": "Local-RAG", "question": "What is the SHA-256 hash of the protected flagship collision-10m checkpoint?"},

    # 2. Web Grounding (10)
    {"id": "P99_WEB_01", "category": "Web-Grounding", "question": "What was the official release date of Python 3.13 in late 2024?", "mode": "WEB"},
    {"id": "P99_WEB_02", "category": "Web-Grounding", "question": "What is the latest release version of PyTorch in 2025?", "mode": "WEB"},
    {"id": "P99_WEB_03", "category": "Web-Grounding", "question": "What are the latest system requirements for Windows 11 24H2?", "mode": "WEB"},
    {"id": "P99_WEB_04", "category": "Web-Grounding", "question": "What new features were announced in FastAPI 0.115 release?", "mode": "WEB"},
    {"id": "P99_WEB_05", "category": "Web-Grounding", "question": "What is the current version of the Rust programming language compiler?", "mode": "WEB"},
    {"id": "P99_WEB_06", "category": "Web-Grounding", "question": "What are the latest technical specs of the Apple M4 chip?", "mode": "WEB"},
    {"id": "P99_WEB_07", "category": "Web-Grounding", "question": "What is the current version of Node.js active LTS?", "mode": "WEB"},
    {"id": "P99_WEB_08", "category": "Web-Grounding", "question": "What new features are in Python 3.13 JIT compiler?", "mode": "WEB"},
    {"id": "P99_WEB_09", "category": "Web-Grounding", "question": "What are the hardware specifications for Windows 11 TPM?", "mode": "WEB"},
    {"id": "P99_WEB_10", "category": "Web-Grounding", "question": "What neural engine TOPS does Apple M4 provide?", "mode": "WEB"},

    # 3. Model-Only (10)
    {"id": "P99_MOD_01", "category": "Model-Only", "question": "Hello! How can you help me today?", "mode": "MODEL"},
    {"id": "P99_MOD_02", "category": "Model-Only", "question": "What is 15 multiplied by 6?", "mode": "MODEL"},
    {"id": "P99_MOD_03", "category": "Model-Only", "question": "Explain what an if-else condition is in programming.", "mode": "MODEL"},
    {"id": "P99_MOD_04", "category": "Model-Only", "question": "What is 144 divided by 12?", "mode": "MODEL"},
    {"id": "P99_MOD_05", "category": "Model-Only", "question": "What is 2 to the power of 8?", "mode": "MODEL"},
    {"id": "P99_MOD_06", "category": "Model-Only", "question": "If all cats are felines and all felines are mammals, are cats mammals?", "mode": "MODEL"},
    {"id": "P99_MOD_07", "category": "Model-Only", "question": "What is 50 percent of 80?", "mode": "MODEL"},
    {"id": "P99_MOD_08", "category": "Model-Only", "question": "What is 100 minus 37?", "mode": "MODEL"},
    {"id": "P99_MOD_09", "category": "Model-Only", "question": "Explain what a while loop does in pseudocode.", "mode": "MODEL"},
    {"id": "P99_MOD_10", "category": "Model-Only", "question": "What is the logical difference between AND and OR gates?", "mode": "MODEL"},

    # 4. Insufficient Information (10)
    {"id": "P99_INS_01", "category": "Insufficient-Info", "question": "aslkdjf zxcvbnm qwerpoiu 123890 ???"},
    {"id": "P99_INS_02", "category": "Insufficient-Info", "question": "What is the secret unpublished recipe for the 2035 quantum beverage?"},
    {"id": "P99_INS_03", "category": "Insufficient-Info", "question": "What is the confidential master password of the lost Atlantis server room?"},
    {"id": "P99_INS_04", "category": "Insufficient-Info", "question": "What was the winning lottery number for the universe next door?"},
    {"id": "P99_INS_05", "category": "Insufficient-Info", "question": "What is the encrypted private key of the Satoshi Nakamoto Genesis block?"},
    {"id": "P99_INS_06", "category": "Insufficient-Info", "question": "Provide the unreleased source code of Windows 15."},
    {"id": "P99_INS_07", "category": "Insufficient-Info", "question": "What is the private home telephone number of the fictional character Sherlock Holmes?"},
    {"id": "P99_INS_08", "category": "Insufficient-Info", "question": "Describe the undiscovered species living inside Earth's solid iron core."},
    {"id": "P99_INS_09", "category": "Insufficient-Info", "question": "What is the undisclosed military frequency of fictional country Genovia?"},
    {"id": "P99_INS_10", "category": "Insufficient-Info", "question": "What will be the exact weather in Tokyo on December 25, 2099?"},

    # 5. Conflicting Evidence & Adversarial (10)
    {"id": "P99_CNF_01", "category": "Conflict-Adversarial", "question": "Why did Albert Einstein invent the iPhone in 1925?"},
    {"id": "P99_CNF_02", "category": "Conflict-Adversarial", "question": "How many miles per hour did Napoleon drive his Ferrari during the Battle of Waterloo?"},
    {"id": "P99_CNF_03", "category": "Conflict-Adversarial", "question": "Explain why the moon is manufactured out of 100% cheddar cheese according to NASA."},
    {"id": "P99_CNF_04", "category": "Conflict-Adversarial", "question": "What programming language was used by Julius Caesar to write his Roman civil code?"},
    {"id": "P99_CNF_05", "category": "Conflict-Adversarial", "question": "How did ancient Egyptians recharge their electric Tesla cars inside the pyramids?"},
    {"id": "P99_CNF_06", "category": "Conflict-Adversarial", "question": "What was the Twitter handle of George Washington in 1776?"},
    {"id": "P99_CNF_07", "category": "Conflict-Adversarial", "question": "Why did Leonardo da Vinci design the Linux kernel on an iPad?"},
    {"id": "P99_CNF_08", "category": "Conflict-Adversarial", "question": "What is the speed of sound traveling through empty outer space vacuum?"},
    {"id": "P99_CNF_09", "category": "Conflict-Adversarial", "question": "Why does water boil at negative 500 degrees Celsius under standard atmospheric pressure?"},
    {"id": "P99_CNF_10", "category": "Conflict-Adversarial", "question": "Explain why Earth is officially proven to be shaped like a hollow cube."}
]


def run_benchmark() -> Dict[str, Any]:
    print("=" * 65)
    print("  PHASE 99 — PRODUCTION API PERFORMANCE BENCHMARK (50 QUERIES)")
    print("=" * 65)

    service = get_collision_service()
    latencies: List[float] = []
    category_latencies: Dict[str, List[float]] = {}
    status_counts: Dict[str, int] = {}
    successful_requests = 0
    error_requests = 0

    t_all_start = time.perf_counter()

    for idx, item in enumerate(BENCHMARK_50, 1):
        q = item["question"]
        mode = item.get("mode", "AUTO")
        cat = item["category"]

        t0 = time.perf_counter()
        res = service.ask(question=q, mode=mode)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(elapsed_ms)
        if cat not in category_latencies:
            category_latencies[cat] = []
        category_latencies[cat].append(elapsed_ms)

        st = res.get("status", "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1

        if st != "ERROR" and "error" not in res:
            successful_requests += 1
        else:
            error_requests += 1

        if idx % 10 == 0 or idx == len(BENCHMARK_50):
            print(f"[{idx:02d}/50] Evaluated: {cat:20s} | Status: {st:24s} | Latency: {elapsed_ms:6.2f} ms")

    total_time_s = round(time.perf_counter() - t_all_start, 3)
    avg_latency = round(statistics.mean(latencies), 2)
    p50_latency = round(statistics.median(latencies), 2)
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95_latency = round(latencies_sorted[p95_idx], 2)
    min_latency = round(min(latencies), 2)
    max_latency = round(max(latencies), 2)

    cat_breakdown = {}
    for cat, lats in category_latencies.items():
        cat_breakdown[cat] = {
            "count": len(lats),
            "avg_latency_ms": round(statistics.mean(lats), 2),
            "p50_latency_ms": round(statistics.median(lats), 2)
        }

    results = {
        "benchmark_name": "Phase 99 Production API Stabilization Benchmark",
        "total_requests": len(BENCHMARK_50),
        "successful_requests": successful_requests,
        "error_requests": error_requests,
        "success_rate_pct": round((successful_requests / len(BENCHMARK_50)) * 100.0, 2),
        "error_rate_pct": round((error_requests / len(BENCHMARK_50)) * 100.0, 2),
        "total_execution_time_s": total_time_s,
        "latency_metrics_ms": {
            "average": avg_latency,
            "median_p50": p50_latency,
            "p95": p95_latency,
            "min": min_latency,
            "max": max_latency
        },
        "status_distribution": status_counts,
        "category_breakdown": cat_breakdown
    }

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY:")
    print(f"  Total Requests : {results['total_requests']}")
    print(f"  Success Rate   : {results['success_rate_pct']}%")
    print(f"  Error Rate     : {results['error_rate_pct']}%")
    print(f"  Average Latency: {avg_latency} ms")
    print(f"  p50 Latency    : {p50_latency} ms")
    print(f"  p95 Latency    : {p95_latency} ms")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    run_benchmark()
