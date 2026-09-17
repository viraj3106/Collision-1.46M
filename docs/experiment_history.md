# COLLISION Series: Experiment & Research History

This document details the historical progression of the COLLISION series models, training datasets, and inference frameworks, explaining the motivation and results for each stage of development.

```
COLLISION-1.46M
        ↓
COLLISION-3.38M
        ↓
Dataset v5
        ↓
COLLISION-10M
        ↓
Production checkpoint
        ↓
Inference API
        ↓
COLLISION LAB
```

---

## 1. COLLISION-1.46M (Phase 5 & 6)

### Specifications
- **Parameters**: 1,462,464
- **Architecture**: 3 layers, 128 embedding dim, 4 attention heads, 256 feed-forward dim, 256 context.

### Stage 1: Phase 5 (Baseline Pretraining)
- **Objective**: Establish a baseline causal language model trained completely from scratch on CPU.
- **Dataset**: `collision_dataset_v3` (2.41M tokens).
- **Outcome**: Achieved a best validation loss of `4.1409` (perplexity `62.86`) at step 1500, but suffered from severe overfitting and sequence degradation after step 1500.

### Stage 2: Phase 6 (Dataset Audit & Generalization)
- **Objective**: Resolve data-quality issues identified in Phase 5:
  - Alphabetical train/val splitting caused severe class biases (e.g., Artificial Intelligence was completely missing from the training split but present in validation).
  - High validation sentence leakage of 26.82% due to contiguous splitting on repetitive documents.
- **Action**: Created `collision_dataset_v4` by removing redundant headers, deduplicating paragraphs, and implementing subject-wise deterministic splits (0% leakage).
- **Outcome**: Validation loss dropped from `4.1409` to `1.9363`, and validation perplexity fell to **6.93** (a 55.93 point improvement) without changing the model size or architecture.

---

## 2. COLLISION-3.38M (Phase 10 & 12B)

- **Objective**: Investigate model scaling laws by increasing model depth and width while maintaining CPU efficiency.
- **Specifications**: 3,375,680 parameters (6 layers, 192 embedding dim, 6 attention heads, 384 feed-forward dim).
- **Outcome**: Achieving validation perplexity of **3.75** (Test Loss: `1.3213`), proving that parameter capacity scaling significantly drives convergence.

---

## 3. Instruction Tuning Exploration (Phase 13)

- **Objective**: Test the instruction-following capabilities of tiny architectures using a synthetic conversational dataset (`collision_instruct_v1`).
- **Outcome**: Led to severe overfitting and regression (repetition rate rose to 57.6%, sentence termination rate dropped to 6.7%). This proved that sub-50M base architectures lack the capacity to generalise conversational instructions without losing base reasoning capability.

---

## 4. Dataset v5 Expansion

- **Objective**: Construct a larger, domain-balanced, leak-free corpus with dedicated training, validation, and test splits to support a 10M token training run.
- **Specifications**: `collision_dataset_v5_expanded` containing 1.80M tokens split into Train (1.54M), Val (189k), and Test (65k).

---

## 5. COLLISION-10M (Phase 14 & 15)

- **Objective**: Scale parameters to 10.28M and extend pretraining to a 10,000,000 token budget starting from random initialization.
- **Specifications**: 10,282,304 parameters (6 layers, 384 embedding dim, 8 attention heads, 768 feed-forward dim, tied embeddings).
- **Results**:
  - Converged to **Best Validation Perplexity of 2.11** (loss `0.7454`) at step 2,500.
  - Achieved a **Test Perplexity of 1.79** (loss `0.5805`).
  - Improved lexical variety (Unique Token Ratio: `58.9%`).

---

## 6. Production Checkpoint & API (Phase 16)

- **Objective**: Freeze the best performing weight checkpoint and wrap it in a clean, local REST API service for easy deployment.
- **Action**: The checkpoint at step 2,500 was frozen to `models/collision-10m/model.pt` (SHA256: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`). A FastAPI web service was implemented to expose completions.

---

## 7. COLLISION LAB (Phase 17)

- **Objective**: Create a graphical playground for model interaction.
- **Action**: Developed a decoupled Streamlit client in `playground/app.py` that interacts with the FastAPI service, featuring real-time diagnostic output, generation controls, and session logs.

---

## 8. Parameter Capacity Expansion (Phase 77)

- **Objective**: Test whether scaling model parameters from ~10M to 25M–50M (`J77-25M`, `J77-35M`, `J77-50M`) under the Phase 76 hybrid loss objective improves multi-turn conversational reasoning.
- **Outcome**: **Outcome C — Capacity Not Confirmed**. Parameter expansion without pretraining resulted in zero-coherence collapse during conversational fine-tuning (coherence 0.500 vs 2.470 control).
- **Conclusion**: Parameter scale alone is not the dominant bottleneck.

---

## 9. Pretraining Token Budget Expansion (Phase 78)

- **Objective**: Investigate whether multi-stage foundational pretraining on synthetic domain-balanced text (`collision_dataset_v5_p78`) unlocks larger model capacity (~25M–50M) prior to conversational fine-tuning.
- **Candidates**: `P78-A` (10M / 100 steps), `P78-B` (10M / 400 steps), `P78-C` (25M / 400 steps), `P78-D` (25M / 1000 steps), `P78-E` (50M / 1000 steps).
- **Results**:
  - Foundational pretraining completely prevented zero-coherence collapse across all scales, restoring functional coherence to `2.360` (vs Phase 77 `0.500`).
  - Pretraining 25M model for 1000 steps (`P78-D`) achieved best validation loss of **`0.2511`** and perplexity of **`1.29`**.
- **Outcome**: **Outcome B — Partial Unlock**. Pretraining exposure successfully unlocked capacity stability and convergence, confirming that foundational pretraining is mandatory. However, synthetic template pretraining reaches capacity saturation.
- **Identified Bottleneck**: **DATA DIVERSITY & COMPLEXITY (PRETRAINING DATA QUALITY)**.

---

## 10. RAG Grounded-Answering Validation (Phase 93)

- **Objective**: Rigorously evaluate whether COLLISION V9 can utilize high-quality retrieved context to synthesize accurate, grounded answers across controlled single-hop, multi-hop, and conflict conditions.
- **Protocol**: 100-question controlled benchmark across 10 distinct domains evaluated under 4 controlled context variations (`MODEL_ONLY`, `RELEVANT_CONTEXT`, `IRRELEVANT_CONTEXT`, `CONFLICTING_CONTEXT`).
- **Results**:
  - `MODEL_ONLY_ACCURACY`: **2.00%**
  - `RELEVANT_CONTEXT_ACCURACY`: **1.00%**
  - `GROUNDED_GAIN`: **-1.00%**
  - `TRUE_CONTEXT_UTILIZATION_RATE`: **1.00%**
  - `IRRELEVANT_CONTEXT_CONTAMINATION_RATE`: **0.00%**
  - `CONFLICT_RESOLUTION_RATE`: **0.00%**
  - `HALLUCINATION_RATE`: **99.00%**
  - `GENERATION_FAILURE_RATE`: **24.75%**
- **Outcome**: **Verdict: PHASE_93_CONTEXT_UTILIZATION_FAILURE**. While V9 eliminates token collapse and generation crashes, a 10M base causal transformer does not intrinsically perform reading comprehension or in-context evidence extraction without targeted instruction fine-tuning / cross-attention mechanisms.
- **Safety & Integrity**: All 13 safeguard assertions (`TEST_A` to `TEST_M`) passed with zero model weight alterations.
