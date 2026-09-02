# COLLISION Developer API Documentation

This documentation describes the secure local completions REST API wrapping the frozen `COLLISION-10M` base language model.

## 1. Introduction
The COLLISION REST API exposes endpoints to query token completions, check system health, list available models, and manage developer keys. It is built as a lightweight, developer-consumable API foundation.

## 2. Authentication
Authentication is enforced on the completions endpoint using the `Authorization` header.
- **Scheme**: Bearer Token
- **Format**: `Authorization: Bearer col_xxxx`
- Requesting completions without a valid, active key returns `401 Unauthorized`.

## 3. API Keys
API keys are cryptographically generated and prefix-tagged securely:
- **Prefix**: `col_` followed by secure hex identifiers.
- **Security**: The plaintext key is shown **only once** upon generation and is never stored. The database stores only the cryptographically secure SHA256 hash.
- **Revocation**: Keys can be revoked via the developer dashboard or admin endpoints. Once revoked, they cannot be reactivated.

## 4. Models
Currently, only the flagship base model is supported:
- **Active Model ID**: `collision-10m`
- arbitary custom model-loading parameters or path arguments are prohibited to prevent local file vulnerability exploits.

## 5. Text Generation
- **Endpoint**: `POST /v1/generate`
- Submits text prompt and parameters to return generated completions.

### Example Request Body (JSON)
```json
{
  "model": "collision-10m",
  "prompt": "Artificial intelligence is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9
}
```

## 6. Parameters
- `model` (string, required): Must be `"collision-10m"`.
- `prompt` (string, required): Non-empty text prompt. Length must not exceed the context length limit.
- `max_tokens` (int, optional, default: 100): Maximum tokens to generate (1 - 256).
- `temperature` (float, optional, default: 0.7): Randomness scale (gt 0.0).
- `top_k` (int, optional, default: 50): Token restriction bounds (ge 0).
- `top_p` (float, optional, default: 0.9): Cumulative probability boundary (0.0 < top_p <= 1.0).

## 7. Responses
Successful requests receive a `200 OK` response. Every response contains an `X-Request-ID` header.

### Example Response Body (JSON)
```json
{
  "id": "collision-generation-c80c2f8b-dbb1-4171-aa31-e4f603c46e01",
  "object": "text_completion",
  "model": "collision-10m",
  "text": " defined as a system designed to adjust calculated model gradients.",
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 12,
    "total_tokens": 18
  },
  "performance": {
    "latency_ms": 270.4,
    "tokens_per_second": 44.38
  }
}
```

## 8. Errors
Errors return a standard JSON error response containing the type, details, and internal request ID. Stack traces are masked.

### Categories
- `authentication_error` (401): Missing, malformed, or invalid bearer token.
- `rate_limit_error` (429): Limit exceeded.
- `validation_error` (422 / 400): Parameter boundary violation or oversized prompt.
- `model_error` (500): Local model inference failure.
- `server_error` (500): Internal system database or startup failure.

### Example Error Response (JSON)
```json
{
  "error": {
    "type": "authentication_error",
    "message": "Invalid API key.",
    "request_id": "req_8c2014b2d5d8f6d7"
  }
}
```

## 9. Rate Limits
- **Default rate limit**: 60 requests/minute/API key.
- **Header**: Returns `Retry-After: <seconds>` on 429 errors.
- Limits are configurable via `COLLISION_RATE_LIMIT` environment variable.

## 10. Usage Tracking
Every generation logs statistical events to the database:
- Prompt token count
- Completion token count
- Latency (ms)
- Generation timestamp
Usage totals and averages can be reviewed inside the **Developer Dashboard** tab in COLLISION LAB.

## 11. Context Limits
The context length is hard-capped at **256 tokens** (including both prompt and completion). Prompts exceeding this constraint will fail with a `validation_error` (400 Bad Request).

## 12. Current Model Limitations
- **Autocomplete only**: The model is a **base model** and has not been instruction-tuned. It continues prompt sequences rather than responding in conversational dialogue format.
- **Scientific focus**: Trained on declarative concept paragraphs; factual accuracy is not guaranteed.
- **CPU latency**: Generations execution speed is bounded by CPU throughput.
