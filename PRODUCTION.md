# COLLISION Production Operations & Architecture Manual

## 1. System Architecture Overview

COLLISION is an evidence-first, grounded question-answering architecture that strictly enforces factual provenance:

```text
               CLIENT (Web Chat UI / API / SDK)
                             │
                             ▼
                   REVERSE PROXY (Nginx / Caddy)
                             │
                             ▼
               COLLISION FASTAPI SERVICE (Port 8000)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  [Rate Limiter]     [Auth / Sessions]     [Metrics Collector]
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
             POST /v1/ask  (CollisionService)
                             │
                             ▼
                 Adaptive Knowledge Engine
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    Local RAG           Web Grounding        Parametric
   (BM25 + Chunks)     (Safe Web Fetch)     (Model Checkpoint)
         └───────────────────┼───────────────────┘
                             │
                             ▼
                  Multi-Source Evidence Fusion
                             │
                             ▼
                 Grounding & Safety Verifier
           (Fact Verification, Injection Shield,
            Temporal Freshness, Conflict Detector)
                             │
                             ▼
                  Grounded Synthesis Engine
                             │
                             ▼
                    Structured Response
```

---

## 2. Protected Checkpoints & Integrity Guarantees

COLLISION guarantees **ZERO model weight modifications**. All checkpoints are verified on startup and during CI/CD checks:

| Checkpoint Name | Model Path | Checkpoint SHA-256 |
| :--- | :--- | :--- |
| **Flagship 10M** | `models/collision-10m/model.pt` | `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97` |
| **Research V9** | `models/phase91_v9_10m/model.pt` | `98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449` |

To verify integrity manually:
```bash
python -c "from collision.config import validate_checkpoints; assert validate_checkpoints()"
```

---

## 3. Concurrency & Performance Benchmarks

COLLISION Phase 102 was tested under heavy load across multiple concurrency tiers:

| Concurrency Level | Total Requests | Success Rate | Throughput (QPS) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Worker** | 20 | **100.0%** | 23.41 | 14.44 | 542.43 | 569.51 |
| **5 Workers** | 30 | **100.0%** | 68.07 | 70.13 | 97.33 | 99.34 |
| **10 Workers** | 40 | **100.0%** | 71.21 | 132.56 | 172.91 | 176.48 |
| **25 Workers** | 50 | **100.0%** | 69.77 | 305.60 | 430.38 | 479.39 |
| **50 Workers** | 100 | **100.0%** | **73.32** | 486.04 | 713.74 | 824.93 |

### Memory Stability (Long-Run Benchmark)
* **Total Executed Requests:** 120 continuous mixed queries
* **Success Rate:** 100.0% (120/120)
* **Initial Process RSS:** 368.96 MB
* **Final Process RSS:** 369.73 MB
* **Net Memory Growth:** +0.77 MB
* **Memory Leak Status:** **NO LEAK DETECTED** (Rock solid memory envelope)

---

## 4. API Endpoints Reference

### Ask Endpoint (`POST /v1/ask`)
Request Body:
```json
{
  "query": "What is the embedding dimension of COLLISION 10M?",
  "mode": "AUTO",
  "max_tokens": 256,
  "temperature": 0.0
}
```

Response Body:
```json
{
  "answer": "COLLISION 10M operates with 6 transformer layers, an embedding dimension d_model of 384, 8 attention heads, and d_ff of 768.",
  "status": "ANSWERED",
  "mode": "LOCAL",
  "sources": [
    {
      "source_id": "collision_architecture.md",
      "url": "file:///docs/collision_architecture.md",
      "title": "COLLISION Architecture Specification",
      "snippet": "COLLISION 10M operates with 6 transformer layers, an embedding dimension d_model of 384..."
    }
  ],
  "latency_ms": 11.4,
  "model": "collision-10m"
}
```

---

## 5. Security & Safety Defenses

1. **SSRF Protection (`collision/web/fetch.py`)**:
   Prevents fetching internal/private network addresses (e.g., `127.0.0.1`, `169.254.169.254`, `10.0.0.0/8`, `192.168.0.0/16`).
2. **Adversarial & Injection Filtering (`collision/grounding/injection_defense.py`)**:
   Neutralizes jailbreaks, DAN prompts, instruction overrides, and fake system prefixes.
3. **Future & Speculation Guard (`collision/grounding/verifier.py`)**:
   Strictly refuses unknowable claims (future stock prices, lottery predictions, future elections) with explicit, helpful abstention notices.
4. **Rate Limiting (`api/limiter.py`)**:
   Protects against brute-force query flooding with token-bucket IP limits.
