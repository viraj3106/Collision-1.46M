# COLLISION-1M Phase 3 Experiment Design

This document details the configuration for the COLLISION-1M Phase 3 training run.

## Experiment Specifications

- **Experiment Name**: COLLISION-1M Phase 3
- **Model Parameters**: 1,462,464 parameters (Decoder-only Transformer)
- **Model Layout**:
  - Vocabulary Size: 8000
  - Context Length: 256
  - Embedding Dimension: 128
  - Layers: 3
  - Attention Heads: 4
  - Feed-forward Dimension: 256
- **Compute Device**: CPU
- **Optimizer**: AdamW (Learning rate: 6e-4, Weight decay: 0.01)
- **Dataset Targets**:
  - Target range: 1,000,000 to 5,000,000 tokens
  - Active Dataset Version: [Refer to latest datasets/ collision_dataset metadata]
  - Split: 90% Training, 10% Validation

---

## Pre-Training Status

> [!WARNING]
> **Pre-Training Gate Constraint**
> The model will not undergo serious training until a dataset containing at least 1,000,000 tokens is built and processed through the dataset builder pipeline.

---

## Action Plan
1. Collect high-quality public domain text corpuses and copy them to `data/raw/`.
2. Run `python -m data.build` to generate clean, deduplicated token binaries and metadata.
3. Validate dataset parameters using `python -m data.report`.
4. Initiate training with the CPU configuration: `python -m training.train --config configs/collision_1m_cpu.yaml --max-steps 5000`.
