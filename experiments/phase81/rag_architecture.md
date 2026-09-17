# Phase 81 — RAG Architecture Specification

## System Architecture

```
USER QUESTION
     │
     ▼
[QueryRouter] (auto/on/off mode check)
     │
     ├─► OFF/Static Query ──► Direct COLLISION Generation
     │
     └─► ON/Auto Web Query
           │
           ▼
     [BaseSearchProvider] (Mock / DuckDuckGo / Custom API)
           │
           ▼
     [SSRF-Protected Fetcher] (3s timeout, 500KB cap, IP guard)
           │
           ▼
     [Text Extractor] (Strips nav, scripts, boilerplate)
           │
           ▼
     [BM25 Lexical Ranker] (Passage relevance scoring)
           │
           ▼
     [Token-Budget Context Manager] (Adheres to max_seq_len = 256)
           │
           ▼
     [Prompt Injection Isolated Context]
           │
           ▼
     COLLISION-25M / 10M Inference Engine
           │
           ▼
     Answer + Web Sources & Citations
```

## Security & Integrity
* **Prompt Injection Isolation**: Retrieved text is tagged in `CONTEXT (UNTRUSTED WEB DATA)` blocks with explicit system prompt instruction guards.
* **SSRF Guard**: Resolves DNS hostnames and blocks requests targeting `127.0.0.1`, `localhost`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`.
* **Zero Model Weights Modification**: `TRAINING EXECUTED = FALSE`, `MODEL WEIGHTS MODIFIED = FALSE`.
