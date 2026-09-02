# Phase 37 Real-World Data & Preference Specification

## 1. Executive Strategy

Phase 37 expands real-world training dataset volume (Collision Dataset V8) and introduces preference optimization dataset (Preference Dataset V1) to determine if scaling real-world data, preference alignment, or their combination enables COLLISION-10M to surpass Production Model A.

Dataset Labels:
- Dataset V8: `REAL_WORLD_PUBLIC_DATA`
- Preference Dataset V1: `CURATED_REALISTIC_DATA`

---

## 2. Dataset V8 Scale-Up Specification

Total Token Target: **250,000 – 500,000 tokens**

### Category Distribution
- 25% Natural Q&A (Technical, conceptual, general science, technology)
- 20% Instruction Following (Formatting, summarization, bullet extraction, transformation)
- 15% Explanations (Deep learning, operating systems, cloud architecture)
- 10% Troubleshooting (Debugging, error stack traces, memory profiling)
- 10% Conversational Language (Multi-turn continuity, clarification requests)
- 10% Reasoning / Problem Solving (Algorithms, math, latency calculations)
- 5% Summarization / Rewriting (Text compression, tone shift)
- 5% Everyday Knowledge (Productivity, team workflows, best practices)

---

## 3. Preference Dataset V1 Specification

Total Preference Pairs: **5,000 – 10,000 pairs**

Schema (`preference_dataset_v1.json`):
```json
{
  "id": "PREF_0001",
  "source_type": "CURATED_REALISTIC_DATA",
  "category": "technical_explanation",
  "prompt": "Explain why connection pooling reduces backend database latency.",
  "chosen": "Connection pooling reuses already established TCP connections rather than opening and closing a new socket per query, which eliminates authentication handshakes and reduces query latency.",
  "rejected": "Connection pooling makes queries faster by compressing SQL text strings before sending them to the database engine."
}
```

Preference Alignment Criteria:
- `chosen`: Correct, direct, coherent, complete, and contextually precise.
- `rejected`: Contains factual hallucination, repetition loops, verbose filler, or instruction non-compliance.

---

## 4. Privacy Filtering Specification

All dataset records pass through automated PII filter:
- Emails -> `[REDACTED_EMAIL]`
- Private IPs -> `[REDACTED_IP]`
- Credentials/API keys -> `[REDACTED_CREDENTIAL]`
