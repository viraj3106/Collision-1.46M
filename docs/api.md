# COLLISION-10M REST API Documentation

*Local Development / Beta Version Only*

This API exposes endpoints for the frozen `collision-10m` model running on local CPU.

## Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn httpx torch numpy pyyaml psutil
   ```

2. **Starting the server**:
   From the project root directory, run:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```

## Endpoints

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {
    "status": "ok",
    "model": "collision-10m",
    "device": "cpu"
  }
  ```

### 2. List Models
* **Endpoint**: `GET /v1/models`
* **Response**:
  ```json
  {
    "data": [
      {
        "id": "collision-10m",
        "object": "model"
      }
    ]
  }
  ```

### 3. Generate Completions
* **Endpoint**: `POST /v1/generate`
* **Request Format** (JSON):
  * `model` (string): Must be `"collision-10m"`.
  * `prompt` (string, required): Input prompt, min length 1, max context 256 tokens.
  * `max_tokens` (int, optional, default: 100): Tokens to generate (range: 1 - 256).
  * `temperature` (float, optional, default: 0.7): Must be > 0.0.
  * `top_k` (int, optional, default: 50): Must be >= 0.
  * `top_p` (float, optional, default: 0.9): Must be > 0.0 and <= 1.0.

* **Example curl Request**:
  ```bash
  curl -X POST https://YOUR_DOMAIN/v1/generate \
    -H "Authorization: Bearer col_YOUR_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "collision-10m",
      "prompt": "What is artificial intelligence?",
      "max_tokens": 100,
      "temperature": 0.7,
      "top_p": 0.9
    }'
  ```

* **Response Format** (JSON):
  ```json
  {
    "id": "collision-generation-<uuid>",
    "object": "text_completion",
    "model": "collision-10m",
    "text": "Generated completion continuation here...",
    "usage": {
      "prompt_tokens": 6,
      "completion_tokens": 82,
      "total_tokens": 88
    },
    "performance": {
      "latency_ms": 1750.4,
      "tokens_per_second": 46.85
    }
  }
  ```

## Limits
* **Maximum Context Length**: 256 tokens.
* **Maximum Generation Tokens**: 256 tokens.
* Prompts exceeding 256 tokens will return `400 Bad Request`.
* Invalid decoding parameters will return `422 Unprocessable Entity` or `400 Bad Request`.

## Client Integration Examples

### Python Client Example

A simple script using the `requests` library to fetch completions:

```python
import requests

def generate_completion(prompt: str, max_tokens: int = 100):
    url = "http://127.0.0.1:8000/v1/generate"
    payload = {
        "model": "collision-10m",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["text"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    prompt = "Artificial intelligence is"
    completion = generate_completion(prompt)
    print("Prompt:", prompt)
    print("Completion:", completion)
```

### JavaScript / TypeScript Client Example

A simple function using standard `fetch`:

```javascript
async function generateCompletion(prompt, maxTokens = 100) {
  const url = 'http://127.0.0.1:8000/v1/generate';
  const payload = {
    model: 'collision-10m',
    prompt: prompt,
    max_tokens: maxTokens,
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status}, detail: ${errorText}`);
    }

    const data = await response.json();
    return data.text;
  } catch (error) {
    console.error("Failed to generate completion:", error);
    return null;
  }
}

// Usage:
// generateCompletion("Artificial intelligence is").then(text => console.log(text));
```

