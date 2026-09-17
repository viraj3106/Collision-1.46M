# Phase 79 — 10M → 25M Capacity Scaling Experiment

## Executive Summary

Phase 79 is a controlled scaling experiment designed to compare the foundational performance of the **COLLISION-10M Control** (`~10.28M` parameters) against a newly trained **COLLISION-25M Candidate** (`~25.26M` parameters).

## Research Question

> Does increasing parameter capacity from ~10M to ~25M improve foundational language-model performance when dataset, tokenizer, objective, optimizer, context length, and training procedure are held as constant as possible?

## Hypotheses

* **H1 — Capacity Scaling**: The 25M model will achieve lower validation/test loss than the 10M control.
* **H2 — Generalization**: The 25M model will maintain or improve generation quality rather than simply memorizing the synthetic training distribution.
* **H3 — Capacity Efficiency**: If 25M improves substantially under the same pretraining conditions, parameter capacity remains a useful scaling axis for COLLISION.
* **H4 — Bottleneck Detection**: If 25M shows little improvement despite substantially greater capacity, data diversity/complexity or optimization is likely the dominant bottleneck.

## Model Specifications

| Property | P79-A (10M Control) | P79-B (25M Candidate) |
|---|---|---|
| Layers (`n_layer`) | 6 | 10 |
| Embedding Dim (`d_model`) | 384 | 512 |
| Heads (`n_head`) | 8 | 8 |
| Feed-Forward Dim (`d_ff`) | 768 | 1024 |
| Exact Parameters | **10,282,304** | **25,263,936** |
| Context Length | 256 | 256 |
| Vocabulary Size | 8,000 | 8,000 |
| Tied Embeddings | True | True |

## Reproducibility & Execution

To run tests and execute the experiment:

```bash
# 1. Run unit tests
python experiments/phase79/test_phase79.py

# 2. Run full experiment (train, evaluate, analyze, generate report)
python experiments/phase79/run_phase79.py
```
