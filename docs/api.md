# COLLISION REST API & Service Documentation

Phase 99 Production API & Grounded Answering Engine Specification.

This document describes the public HTTP API, Application Service Layer, and CLI entry points for the COLLISION grounded answering system.

---

## 1. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Configure runtime behavior via environment variables:

| Variable | Description | Default |
|---|---|---|
| `COLLISION_HOST` | Server host interface | `127.0.0.1` |
| `COLLISION_PORT` | Server listening port | `8000` |
| `COLLISION_MODEL_PATH` | Path to production model weights | `models/collision-10m/model.pt` |
| `COLLISION_TOKENIZER_PATH` | Path to tokenizer directory | `artifacts/tokenizer` |
| `COLLISION_LOCAL_RAG_ENABLED`| Enable local vector retrieval | `true` |
| `COLLISION_WEB_ENABLED` | Enable live web grounding | `true` |
| `COLLISION_MAX_INPUT_LENGTH` | Maximum question character length | `2000` |
| `COLLISION_LOG_LEVEL` | Application logging level | `INFO` |

### Starting the Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## 2. API Endpoints

### `GET /health`
Liveness probe.

**Response (200 OK):**
```json
{
  "status": "ok",
  "service": "collision",
  "version": "1.0",
  "model": "collision-10m",
  "device": "cpu"
}
```

---

### `GET /ready`
Readiness probe verifying runtime components (model checkpoint, tokenizer, vector index, and web search provider) without running expensive inference.

**Response (200 OK):**
```json
{
  "status": "ready",
  "service": "collision",
  "checks": {
    "model_checkpoint": true,
    "tokenizer": true,
    "local_index": true,
    "web_provider": true
  }
}
```

---

### `POST /v1/ask` (or `/ask`)
Primary grounded question-answering endpoint.

**Request Body (JSON):**
```json
{
  "question": "What is the embedding dimension in the COLLISION 10M architecture?",
  "mode": "AUTO",
  "include_sources": true,
  "include_claims": true
}
```

#### Request Parameters
- `question` (string, required): The query to answer (1 to 2000 characters).
- `mode` (string, optional, default: `"AUTO"`): Routing mode:
  - `"AUTO"`: Automatically routes between local RAG, web grounding, hybrid synthesis, and conversational prior based on confidence.
  - `"LOCAL"`: Forces retrieval from local knowledge documents.
  - `"WEB"`: Forces retrieval from external web sources.
  - `"HYBRID"`: Synthesizes cross-source evidence (local + web).
  - `"MODEL"`: Direct conversational model response without retrieval.
- `include_sources` (boolean, optional, default: `true`): Include source provenance in response.
- `include_claims` (boolean, optional, default: `true`): Include claim verification breakdown.

**Response (200 OK):**
```json
{
  "answer": "COLLISION 10M operates with 6 transformer layers, an embedding dimension d_model of 384, 8 attention heads, and d_ff of 768.",
  "status": "ANSWERED",
  "mode": "LOCAL",
  "confidence": 0.8228,
  "sources": [
    {
      "source_id": "src_1",
      "title": "collision_architecture.md",
      "url": "local://collision_architecture.md",
      "source_type": "LOCAL",
      "retrieval_score": 1.0,
      "relevance": 1.0,
      "snippet": ""
    }
  ],
  "claims": [
    {
      "text": "COLLISION 10M operates with 6 transformer layers, an embedding dimension d_model of 384, 8 attention heads, and d_ff of 768.",
      "support_status": "SUPPORTED",
      "evidence_ids": [
        "collision_architecture.md"
      ]
    }
  ],
  "latency": {
    "routing_ms": 9.65,
    "retrieval_ms": 0.35,
    "generation_ms": 2.8,
    "verification_ms": 2.38,
    "total_ms": 15.37
  },
  "metadata": {
    "answer_type": "EXTRACTIVE_ANSWER",
    "is_fallback_used": false,
    "termination_reason": "direct_extraction",
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

---

## 3. Response Statuses & Semantics

| Status | Description |
|---|---|
| `ANSWERED` | The query was answered with verified evidence or valid conversational completion. |
| `INSUFFICIENT_INFORMATION` | Available evidence is insufficient to answer the query, or query demands confidential/future information. Refusal without hallucination. |
| `CONFLICT` | Available sources provide conflicting evidence. Contradiction is exposed with both sources rather than arbitrarily guessed. |
| `ERROR` | An invalid request or internal execution error occurred. |

---

## 4. Error Handling
Errors are returned as structured JSON without exposing internal Python stack traces:

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Question exceeds maximum allowed length (2500 > 2000 characters)."
  },
  "latency": {
    "total_ms": 0.42
  }
}
```

---

## 5. CLI Usage

Both the HTTP API and CLI share the unified `CollisionService` application layer.

### Ask a question (Text Mode)
```bash
python -m collision ask "How many transformer layers are used in COLLISION 10M?"
```

### Ask a question (JSON Mode)
```bash
python -m collision ask "How many transformer layers are used in COLLISION 10M?" --json
```

### Interactive Chat Session
```bash
python -m collision chat
```

---

## 6. Security & Safety Controls
- **SSRF Protection**: Private and local IP address ranges are blocked during live web fetches.
- **Prompt Injection Defense**: Retrieved web content is strictly isolated as passive data chunks and cannot override system instructions.
- **Claim Verification**: Generative model outputs are checked against retrieved evidence spans before acceptance; unverified claims trigger extraction fallback.
- **Model Checkpoint Immutability**: All model weights remain strictly frozen and read-only.

---

## 7. Frontend API Client (`website/src/api.ts`)

The web UI interacts with the production API using the centralized `collisionApi` client:

```typescript
import { collisionApi } from './api';

// Ask query
const response = await collisionApi.ask("What is COLLISION 10M?");
console.log(response.answer, response.status, response.sources);
```

Environment variable configuration:
- `VITE_COLLISION_API_URL` or `VITE_API_URL` (default: `http://localhost:8000`).
