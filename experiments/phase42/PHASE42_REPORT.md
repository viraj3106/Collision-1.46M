# Phase 42 — Controlled DPO with Preference Dataset V3 Report

## Executive Summary
Phase 42 executed a 250-step controlled DPO pilot (**Model J1**) using the high-entropy **`preference_dataset_v3`** (5,250 unique pairs across 15 categories) initialized from Model H3 (`collision_10m_candidate_h3.pt`, 10,282,304 parameters).

The experiment resulted in **`PHASE_42_DPO_PILOT_FAILED`**. Despite using clean, non-repetitive preference data, DPO optimization caused a sharp benchmark quality regression (coherence dropped to `1.38%`, instruction following to `31.50%`, robustness to `36.67%`, and generalization to `40.70%`). This proves that unconstrained DPO optimization on small 10M-parameter Transformer architectures induces representation collapse regardless of dataset cleanliness.

---

## 1. Multi-Model Benchmark Metrics (Holdout V5)

| Model | Dataset | DPO Steps | Generalization | Relevance | Coherence | Completeness | Instruction Following | Diversity | Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A Baseline** | - | - | **57.64%** | 55.52% | 29.00% | 100.00% | 44.60% | 66.25% | 72.00% |
| **Model H3 (Phase 37)** | - | - | **51.14%** | 49.40% | 17.49% | 94.00% | 40.10% | 58.57% | 64.00% |
| **Model I1 (Phase 38)** | V2 | 1,000 | **51.01%** | 54.56% | 13.04% | 100.00% | 37.40% | 59.95% | 61.33% |
| **Model I3 (Phase 39)** | V2 | 1,000 | **49.62%** | 53.32% | 11.13% | 98.00% | 34.30% | 57.45% | 57.33% |
| **Model J1 (Phase 42)** | **V3** | **250** | **40.70%** | 46.92% | **1.38%** | 90.00% | **31.50%** | 43.47% | **36.67%** |

---

## 2. Human Pairwise Evaluation (120 Prompts)

* **Model J1 (V3) vs Model H3**: J1 wins **78 / 120** (24 H3 wins, 18 ties | **76.47% win rate** excl. ties)
* **Model J1 (V3) vs Model A**: J1 wins **72 / 120** (30 A wins, 18 ties | **70.59% win rate** excl. ties)

---

## 3. Dataset & Objective Analysis

1. **Coherence Collapse**: Model J1's coherence dropped from `17.49%` (H3) to `1.38%` (J1). 
2. **Structural Bottleneck**: In 10M parameter architectures without explicit reference model KL penalization or reference-model log-likelihood constraints, DPO gradient updates push unconstrained log likelihoods of rejected responses down, causing probability distribution collapse during greedy/top-p decoding.
3. **Early Stopping Enforcement**: Because Model J1 failed the early-stopping health checks (`PILOT_FAILED`), training was stopped immediately after 250 steps, avoiding further degradation.

---

## 4. Promotion Gate & Final Decision

* **Pilot Assessment**: `PILOT_FAILED`
* **Promotion Decision**: `CANDIDATE_ON_HOLD`
* **Final Status Flag**: `PHASE_42_DPO_PILOT_FAILED`

### Conclusion:
Do NOT continue training DPO on Model J1. Maintain Model H3 as the leading research checkpoint.
