# COLLISION-10M v1.0.0 — First Public Developer Release

## Overview

The COLLISION project is pleased to announce the first public developer release of **COLLISION-10M v1.0.0**. This release packages the frozen model checkpoint, local inference pipeline, completions REST API service, and interactive playground in a reproducible open-source release format. 

COLLISION-10M is an **experimental 10.28M-parameter base language model** trained from scratch using a CPU-first development approach. It is intended for developer experimentation, local CPU-only inference research, and education.

## Highlights

- **From-Scratch CPU Training**: Randomly initialized and trained on CPU under a 10M token budget.
- **Scientific Dataset Hygiene**: Trained on audited synthetic scientific data with absolute train/validation/test leakage isolation.
- **Local completions REST API**: Fast completions server built on FastAPI and Uvicorn.
- **Graphical Playground Interface**: Streamlit dashboard client with real-time token logs.
- **Permissive Open-Source Licensing**: Code, weights, and synthetic datasets released under the MIT License.

## Model Specifications

- **Parameters**: 10,282,304
- **Layers (n_layer)**: 6
- **Embedding Dimension (d_model)**: 384
- **Attention Heads (n_head)**: 8
- **Feedforward Dimension (d_ff)**: 768
- **Dropout**: 0.1
- **Weight Tying**: Enabled (`tie_embeddings: true`)
- **Context Length**: 256 tokens
- **Positional Encoding**: absolute_learned

## Evaluation

Metrics evaluated at the best validation checkpoint (Step 2,500):
- **Best Validation Loss**: `0.7454`
- **Best Validation Perplexity**: `2.11`
- **Test Split Loss**: `0.5805`
- **Test Split Perplexity**: `1.79`
- **Unigram Repetition Rate**: `41.1%`
- **Unique Token Ratio**: `58.9%`
- **Sentence Termination Rate**: `62.5%`

## API

FastAPI endpoints running locally:
- `GET /health` - Service health status
- `GET /v1/models` - Active model listings
- `POST /v1/generate` - Text completions

Start local completions server:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

*Note: The API is in Local/Developer Beta and does not feature authentication, API keys, rate-limiting, or billing layers.*

## Playground

Streamlit client user interface:
```bash
streamlit run playground/app.py
```
Interact using sliders for Temperature, Top K, Top P, and inspect raw API payloads.

## Limitations

- **Context Window**: Strictly 256 tokens max.
- **Base Model Behavior**: Not instruction-tuned; continues text instead of conversing.
- **Repetitive Outputs**: Susceptible to unigram repetition biases common in small models.
- **Factual Accuracy**: Factual correctness cannot be assumed; training data is restricted to synthetic concept templates.
- **CPU Throughput**: Throughput (~42.38 tokens/second) is limited by local CPU performance.

## Reproducibility

- **SHA256 Checksum**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- **File Size**: `125,057,611` bytes
- **Locked Env**: See detailed software package version definitions in `release/REPRODUCIBILITY.md` and dependencies in `requirements-release.txt`.

## License

Subject to the MIT License. Code, weights, and dataset splits are licensed permissively.

## Known Issues

- High latency on first token generation due to first-pass model compilation.
- Sub-optimal tokenization efficiency due to vocabulary alignment constraints (active vocabulary has 890 BPE merges, while model embedding capacity is 8,000).
