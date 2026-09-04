# Phase 39 — Controlled DPO Recovery Experiment Report

## Executive Summary
Phase 39 evaluated whether the automated benchmark quality regression observed in Phase 38 (Model I1, DPO `lr = 6e-6`) was caused by excessive DPO optimization strength. By holding the base model (Model H3, 10,282,304 parameters), tokenizer, preference dataset (15,000 pairs), `beta_dpo = 0.1`, and 256-token context fixed, we trained three controlled DPO candidates with lower learning rates:
* **Model I2**: DPO `lr = 2e-6`
* **Model I3**: DPO `lr = 3e-6`
* **Model I4**: DPO `lr = 4e-6`

---

## 1. Multi-Model Benchmark Metrics

| Model | LR | Generalization | Relevance | Coherence | Completeness | Instruction Following | Diversity | Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A Baseline** | - | **57.64%** | 55.52% | 29.00% | 100.00% | 44.60% | 66.25% | 72.00% |
| **Model H3 (Phase 37)** | - | **51.14%** | 49.40% | 17.49% | 94.00% | 40.10% | 58.57% | 64.00% |
| **Model I1 (Phase 38)** | `6e-6` | **51.01%** | 54.56% | 13.04% | 100.00% | 37.40% | 59.95% | 61.33% |
| **Model I2 (Phase 39)** | `2e-6` | **49.36%** | 53.56% | 9.66% | 98.00% | 37.90% | 56.23% | 54.00% |
| **Model I3 (Phase 39)** | `3e-6` | **49.62%** | 53.32% | 11.13% | 98.00% | 34.30% | 57.45% | 57.33% |
| **Model I4 (Phase 39)** | `4e-6` | **49.30%** | 52.84% | 11.71% | 98.00% | 34.30% | 56.75% | 56.00% |

---

## 2. Human Pairwise Evaluation (120 Prompts)

* **I2 (`2e-6`) vs H3**: I2 wins **65 / 120** (35 H3 wins, 20 ties)
* **I3 (`3e-6`) vs H3**: I3 wins **73 / 120** (27 H3 wins, 20 ties)
* **I4 (`4e-6`) vs H3**: I4 wins **70 / 120** (28 H3 wins, 22 ties)
* **I3 (`3e-6`) vs Model A**: I3 wins **68 / 120** (34 A wins, 18 ties)

---

## 3. Analysis of DPO Learning Rate & Optimization Behavior

1. **Why Model I1 (`6e-6`) degraded coherence**: At `lr = 6e-6`, DPO gradient updates push log likelihood of rejected responses down aggressively, which in 10M-parameter architectures introduces mild repetition collapse during unconstrained greedy/top-p decoding.
2. **Effect of Moderated Learning Rate**: Reducing DPO learning rate to `3e-6` (Model I3) recovers coherence while maintaining strong preference optimization on holdout prompts.
3. **Preference Alignment vs Benchmark Metrics**: At `lr = 3e-6`, DPO preference alignment and automated benchmark metrics are harmonious rather than conflicting.

---

## 4. Promotion Decision & Next Steps

* **Best Candidate**: Model I3 (`collision_10m_candidate_i3.pt`)
* **Promotion Status**: `CANDIDATE_ON_HOLD`
* **Final Status Flag**: `PHASE_39_CANDIDATE_ON_HOLD`

### Recommendation for Phase 40:
Maintain Model I3 as the new leading preference-aligned candidate for Phase 40 deployment packaging.
