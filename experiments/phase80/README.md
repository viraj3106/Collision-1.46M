# Phase 80 — 25M Pretraining Data Scaling Experiment

## Executive Summary

Phase 80 is a controlled scaling experiment designed to determine whether the **COLLISION-25M** model (`25,263,936` parameters) is primarily **data-constrained**. It systematically measures model loss, perplexity, generation coherence, and capability probe performance across a 5-condition token budget scaling curve:

* **P80-A**: 1M Tokens (500 steps)
* **P80-B**: 10M Tokens (1,000 steps)
* **P80-C**: 25M Tokens (2,500 steps)
* **P80-D**: 50M Tokens (5,000 steps)
* **P80-E**: 100M Tokens (10,000 steps)

## Research Question

> How does increasing the pretraining token budget affect the performance and utilization of the fixed ~25M parameter COLLISION architecture?

## Model Architecture (Fixed Control)

| Property | Value |
|---|---|
| Architecture | CollisionTransformer |
| Parameters | **25,263,936** |
| Layers (`n_layer`) | 10 |
| Embedding Dim (`d_model`) | 512 |
| Heads (`n_head`) | 8 |
| Feed-Forward Dim (`d_ff`) | 1024 |
| Vocabulary Size | 8,000 |
| Context Length | 256 |
| Tied Embeddings | True |

## Execution Commands

```bash
# 1. Run unit tests
pytest experiments/phase80/test_phase80.py

# 2. Run full experiment (train all conditions, evaluate, generate scaling report)
python experiments/phase80/run_phase80.py
```
