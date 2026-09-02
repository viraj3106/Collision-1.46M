# COLLISION Quickstart Guide

This guide will help you onboard onto the COLLISION API developer beta and execute your first completion query.

## 1. Account Creation & API Keys
1. Visit the COLLISION Developer Portal.
2. Sign up under the **Create Developer Account** form with your email and password.
3. Log in with your new credentials.
4. Navigate to the **API Keys** tab and click **Generate API Key**.
5. Copy the generated key token (`col_...`) immediately. 

> **Important**: This key is shown **only once** upon generation. Keep it secure; you cannot view it again.

## 2. Authentication
Enforce authorization using standard HTTP Bearer headers:
- Header name: `Authorization`
- Header format: `Bearer col_your_copied_secret_key`

## 3. First Request Examples

### cURL
```bash
curl -X POST https://api.example.com/v1/generate \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 50
  }'
```

### Python
```python
import requests

url = "https://api.example.com/v1/generate"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}
payload = {
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 50
}

response = requests.post(url, json=payload, headers=headers)
print(response.json()["text"])
```

### JavaScript / Node.js
```javascript
fetch("https://api.example.com/v1/generate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  },
  body: JSON.stringify({
    model: "collision-10m",
    prompt: "Artificial intelligence is",
    max_tokens: 50
  })
})
.then(res => res.json())
.then(data => console.log(data.text));
```

## 4. Request Schema (`POST /v1/generate`)
- `model` (string, required): Must be `"collision-10m"`.
- `prompt` (string, required): Non-empty text prompt <= 256 tokens.
- `max_tokens` (int, optional, default 100): Tokens to complete (1 to 256).
- `temperature` (float, optional, default 0.7): Randomness scale (gt 0.0).
- `top_k` (int, optional, default 50): Token restriction bounds (ge 0).
- `top_p` (float, optional, default 0.9): Cumulative boundary (0.0 < top_p <= 1.0).

## 5. Response Schema
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

## 6. Error Responses
Errors return structured JSON objects, masking all internal stack traces or filesystem paths:
- **401 Unauthorized**: Missing, malformed, or invalid API key.
- **403 Forbidden**: Accessing another developer's resource or calling with a revoked key.
- **413 Payload Too Large**: Combined prompt and generation size exceeds 256-token context limit.
- **429 Too Many Requests**: Exceeded sliding-window rate limit (60 requests/minute).
- **400 Bad Request**: Invalid parameters or empty prompt.
- **500 Server Error**: Internal model or DB failure.

## 7. Model Limitations
- **Autocomplete Only**: This is a **base model** and has not been conversational-aligned (SFT/RLHF). It behaves as a pure autocomplete continuation engine.
- **Factual Hallucinations**: It does not possess reasoning capabilities; it is trained on concept summaries.
- **Contextcap**: Context length is hard-capped at **256 tokens** (combined prompt + completed sequence).
