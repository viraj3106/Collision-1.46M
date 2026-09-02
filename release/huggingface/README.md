---
language: en
license: mit
tags:
- text-generation
- causal-language-model
- small-language-model
- transformers
- research
- educational
datasets:
- collision_dataset_v5_expanded
metrics:
- perplexity
model_name: COLLISION-10M
parameters: 10.28M
---

# COLLISION-10M

## Model Summary

COLLISION-10M is a 10.28M-parameter transformer-based base language model trained from scratch using a CPU-first development approach. 

> [!IMPORTANT]
> **This is a BASE LANGUAGE MODEL.** It is NOT instruction tuned, and it does not act as a conversational assistant. It is primarily a causal text-completion model.

## Model Specifications

- **Parameters**: 10,282,304
- **Layers (n_layer)**: 6
- **d_model (Embedding Size)**: 384
- **Attention heads (n_head)**: 8
- **d_ff (Feedforward size)**: 768
- **Context length**: 256 tokens
- **Training initialization**: Random initialization (from scratch)
- **Training tokens**: 10,000,384 tokens
- **Dataset**: `collision_dataset_v5_expanded`
- **Tokenizer**: Custom BPE Tokenizer (active vocabulary size: 890, model capacity vocab size: 8,000)

## Evaluation

The following metrics represent project-specific benchmarks measured at the best validation checkpoint (Step 2,500). 

> [!NOTE]
> These metrics are specific to the COLLISION pretraining environment. They are not directly comparable to frontier large language models.

- **Validation loss**: 0.7454
- **Validation perplexity**: 2.11
- **Test loss**: 0.5805
- **Test perplexity**: 1.79
- **Repetition rate**: 41.1%
- **Unique token ratio**: 58.9%
- **Termination rate**: 62.5%
- **CPU throughput**: 42.38 tokens/second average
- **Average API latency**: 2317.6 ms (for a 97-token average generation)
- **Memory (RAM)**: 476.3 MB average, 614.1 MB peak

## Training Story

The COLLISION series follows a structured scaling research roadmap:
```
COLLISION-1.46M
        ↓
COLLISION-3.38M
        ↓
Dataset v5
        ↓
COLLISION-10M
        ↓
Inference API
        ↓
COLLISION LAB
```
COLLISION began with the 1.46M baseline pretraining, which highlighted dataset representativeness and sentence leakage issues. Constructing a cleaner, deduplicated split strategy (Dataset v4) improved validation perplexities dramatically. The series scaled to the 3.38M model to check scaling laws, followed by an expansion of training topics to build Dataset v5. Finally, COLLISION-10M was trained under a 10M token budget, culminating in a production API service and the interactive COLLISION LAB interface.

## Intended Use

- Educational projects
- Language-model experimentation
- Local inference
- Text completion
- Small developer experiments
- Research
- Learning transformer inference

## Limitations

- **Parameter count**: 10.28M parameters limit representation capabilities.
- **Context limit**: 256 tokens max context window.
- **Base model structure**: Not instruction tuned; continues text instead of conversing.
- **Limited training data**: Trained on 10M tokens.
- **CPU latency**: Generations are constrained by CPU performance.
- **Generation flaws**: Text repetition can occur; generated information can be incorrect, and it is not a factual database.
- **Safety**: Not a safety-tuned assistant; not suitable for safety-critical applications.

## Quick Start

Run direct local inference on the model using the COLLISION repository:

```bash
# Clone the repository
git clone https://github.com/viraj3106/Collision-1.46M.git
cd Collision-1.46M

# Install release dependencies
pip install -r requirements-release.txt

# Run inference directly (loads checkpoints/tokenizer locally)
python release_inference.py --prompt "Artificial intelligence is" --checkpoint models/collision-10m/model.pt
```

## API Usage

The model can also be accessed through the COLLISION FastAPI server.

### Start local server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Request Completion (cURL)
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  }'
```

### Python Integration
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/generate",
    json={
        "model": "collision-10m",
        "prompt": "Artificial intelligence is",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9
    }
)
print(response.json())
```

### JavaScript/TypeScript Integration
```javascript
const response = await fetch("http://127.0.0.1:8000/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "collision-10m",
    prompt: "Artificial intelligence is",
    max_tokens: 100,
    temperature: 0.7,
    top_k: 50,
    top_p: 0.9
  })
});
const data = await response.json();
console.log(data);
```

## Playground

You can interact with the model via **COLLISION LAB**, a local Streamlit playground frontend:

```
Developer
    ↓
COLLISION LAB (Streamlit UI)
    ↓
FastAPI Server
    ↓
COLLISION-10M Inference Engine
    ↓
Completions Output
```

To launch, start the API server, then in a separate window run:
```bash
streamlit run playground/app.py
```

## Reproducibility

- **Locked replication metrics**: Described in detail in [release/REPRODUCIBILITY.md](file:///v:/collision%20-%201M/release/REPRODUCIBILITY.md).
- **Frozen Checkpoint SHA256**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`

## License

- **Code Base**: Licensed under the **MIT License** (see [LICENSE](file:///v:/collision%20-%201M/LICENSE)).
- **Model Checkpoints**: Licensed under the **MIT License** (see [release/LICENSE_DECISION.md](file:///v:/collision%20-%201M/release/LICENSE_DECISION.md)).
- **Dataset**: Synthetic data generated for research under permissive MIT licensing (see [release/DATASET_LICENSE_AUDIT.md](file:///v:/collision%20-%201M/release/DATASET_LICENSE_AUDIT.md)).
