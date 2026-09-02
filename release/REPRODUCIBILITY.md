# COLLISION-10M Reproducibility Report

This document outlines the environmental configuration, tokenizer, hyper-parameters, and hardware settings required to reproduce the inference execution of the `COLLISION-10M` v1.0.0 model.

## 1. System Environment

- **Python Version**: `3.13.14`
- **Operating System**: Windows (tested on x86_64)

### Dependency Versions
We recommend setting up a virtual environment and installing the packages defined in `requirements-release.txt`:
- `torch` == `2.13.0`
- `numpy` == `2.5.2`
- `fastapi` == `0.141.1`
- `uvicorn` == `0.52.4`
- `streamlit` == `1.62.0`
- `PyYAML` == `6.0.3`
- `psutil` == `7.2.2`
- `httpx` == `0.28.1`
- `requests` == `2.34.2`
- `matplotlib` == `3.11.1`

## 2. Checkpoint Details

- **Checkpoint Path**: `models/collision-10m/model.pt`
- **Release Version**: `1.0.0`
- **File Size**: `125,057,611` bytes
- **SHA256 Hash**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- **Model Parameters**: `10,282,304`
- **Checkpoint Source**: Saved at Step 2500 of Phase 15 training run (representing the best validation epoch).

## 3. Tokenizer Configuration

- **Tokenizer Type**: BPE (Byte Pair Encoding) Custom Tokenizer
- **Source**: `collision/inference/tokenizer.py` (delegating to `data/tokenize.py`)
- **Active Vocabulary Size**: `890` (Model vocab capacity: `8,000`)
- **Special Tokens**:
  - `[PAD]`: 256
  - `[UNK]`: 257
  - `[BOS]`: 258
  - `[EOS]`: 259
- **Compatibility**: Requires `tokenizer/vocab.json`, `tokenizer/merges.json`, and `tokenizer/config.json`.

## 4. Model Architecture & Hyperparameters

- **Layers (n_layer)**: 6
- **Embedding Dimension (d_model)**: 384
- **Attention Heads (n_head)**: 8
- **Feedforward Dimension (d_ff)**: 768
- **Dropout**: 0.1
- **Weight Tying**: Enabled (`tie_embeddings: true`)
- **Context Length**: 256 tokens
- **Positional Encoding**: absolute_learned

## 5. Inference Configurations

- **Default Sampling Settings**:
  - `temperature`: 0.7
  - `top_k`: 50
  - `top_p`: 0.9
  - `max_tokens`: 100
- **Random Seed**:
  - Training initialization: Not recorded.
  - Dataset generation seed: 42.
  - Inference seed: Optional (can be set via CLI/API parameter).

## 6. Benchmark Methodology & Hardware Assumptions

- **Hardware**: Consumer CPU (x86_64 architecture).
- **Run Setup**: Decoupled API performance measurement across 10 sequential prompt generate requests (averaging 97 tokens completed per request).
- **CPU Throughput**: 42.38 tokens/second average.
- **RAM Usage**: Average 476.3 MB, Peak 614.1 MB.
