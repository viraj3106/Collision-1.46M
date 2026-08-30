# COLLISION

Small Models. Real AI.

## What is COLLISION?

COLLISION is a research project designed to explore Transformer causal language modeling in extreme low-resource regimes. It focuses on causal language modeling convergence, CPU-first development, and data quality engineering for tiny model architectures.

## Why does it exist?

COLLISION exists to prove that high-quality, scientifically hygienic training data, coupled with rigorous evaluation protocols, can yield stable and convergent language representation spaces under extreme size constraints (sub-50M parameters) on consumer CPUs without GPU pretraining.

## COLLISION-10M

COLLISION-10M v1.0.0 is the flagship 10.28M-parameter model release under the COLLISION series. It is an experimental base language model trained from scratch with a CPU-first development approach.

## COLLISION-7M

COLLISION-7M is a 6.34M-parameter (6,338,880 parameters) model variant in the COLLISION series designed for scaling analysis. It is configured and trained using the script [train_phase10_7m.py](file:///v:/collision%20-%201M/training/train_phase10_7m.py).

## Quick Start

Follow these steps to run local inference using the frozen model.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd collision
```

### 2. Install Dependencies
```bash
pip install -r requirements-release.txt
```

### 3. Run Direct Local Inference
Execute causal completion directly using the pre-existing inference engine (requires no API or running servers):
```bash
python release_inference.py --prompt "Artificial intelligence is" --checkpoint models/collision-10m/model.pt
```

## API

A production-oriented FastAPI service is provided to query completions via HTTP requests.

### Start the API Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Check Health
```bash
curl -X GET http://localhost:8000/health
```

### List Models
```bash
curl -X GET http://localhost:8000/v1/models
```

### Request Completion
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

For Python and JavaScript clients, see the detailed code examples in the [API Documentation](file:///v:/collision%20-%201M/docs/api.md).

## Playground

The playground consists of a decoupled Streamlit client (`playground/app.py`) that interacts with the FastAPI server backend.

### Launch the Playground
1. Start the API server in one terminal:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the Streamlit frontend in a second terminal:
   ```bash
   streamlit run playground/app.py
   ```

## Model Architecture

| Model Variant | Parameters | Layers (n_layer) | d_model | Attention heads (n_head) | d_ff | Context Length | Weight Tying |
|---|---|---|---|---|---|---|---|
| **COLLISION-10M** | 10,282,304 | 6 | 384 | 8 | 768 | 256 | Enabled (`tie_embeddings: true`) |
| **COLLISION-7M** | 6,338,880 | 8 | 256 | 8 | 512 | 256 | Enabled (`tie_embeddings: true`) |

- **Positional Encoding**: absolute_learned

## Training

### COLLISION-10M
- **Dataset**: `collision_dataset_v5_expanded` (10,000,384 tokens trained)
- **Initialization**: Random initialization (from scratch)
- **Optimizer**: AdamW (lr = 6e-4, min lr = 6e-5, weight decay = 0.01)
- **Schedule**: CosineWarmup (150 warmup steps)
- **Hardware**: CPU

### COLLISION-7M
- **Dataset**: `collision_dataset_v4` (1,536,000 token budget)
- **Initialization**: Random initialization (from scratch)
- **Optimizer**: AdamW (lr = 6e-4, min lr = 6e-5, weight decay = 0.01)
- **Schedule**: CosineWarmup (150 warmup steps)
- **Hardware**: CPU

## Evaluation

COLLISION-10M achieves the following metrics at its best validation checkpoint (Step 2,500):
- **Best Validation Loss**: 0.7454
- **Best Validation Perplexity**: 2.11
- **Test Loss**: 0.5805
- **Test Perplexity**: 1.79
- **Repetition Rate**: 41.1%
- **Unique Token Ratio**: 58.9%
- **Termination Rate**: 62.5%

## Limitations

- **Context Window**: 256 tokens.
- **Small Parameter Count**: 10.28M parameters limit generation capability.
- **Repetitive Output**: Subject to unigram repetition biases common in small models.
- **Not Instruction Tuned**: Will not act as a conversational chatbot assistant (will continue the text prompt).
- **Factual Inaccuracy**: Outputs should not be treated as factually correct.
- **Latencies**: Throughput is bounded by local CPU capabilities.

## Repository Structure

```
collision/
│
├── README.md
├── CITATION.cff
├── requirements-release.txt
│
├── models/
│   └── collision-10m/
│       ├── model.pt
│       ├── config.json
│       ├── tokenizer.json
│       ├── generation_config.json
│       ├── MODEL_CARD.md
│       └── README.md
│
├── inference/
├── api/
├── playground/
│
├── docs/
│   ├── api.md
│   └── experiment_history.md
│
├── release/
│   ├── version.json
│   ├── checksums.sha256
│   ├── MANIFEST.md
│   ├── REPRODUCIBILITY.md
│   ├── DATASET_LICENSE_AUDIT.md
│   ├── LICENSE_DECISION.md
│   ├── GITHUB_RELEASE.md
│   ├── benchmark.md
│   ├── public_claims.md
│   ├── verify_release.py
│   └── huggingface/
│
└── experiments/
```

## License

Subject to the MIT License. See [LICENSE_DECISION.md](file:///v:/collision%20-%201M/release/LICENSE_DECISION.md) for details on code, tokenizer, model weights, and dataset considerations.

## Citation

If you use COLLISION-10M in your research or projects, please cite it using the metadata in [CITATION.cff](file:///v:/collision%20-%201M/CITATION.cff).

## Roadmap

- **Phase 19**: Investigating scaling capabilities and potential architecture improvements (Note: Phase 19 is not currently active).
- **Abuse Prevention & Authentication**: Adding rate limits and auth for API layers.
