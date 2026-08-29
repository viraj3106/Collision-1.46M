# COLLISION Phase 14B — Model Capacity Scaling Audit Report

This report presents an experimental validity audit of the Phase 14 Model Capacity Scaling Experiment comparing `COLLISION-3.38M` and `COLLISION-10M`.

---

## 1. Experimental Objective
The objective is to audit whether increasing model parameter capacity from 3.38M to 10.28M yields valid, unconfounded improvements in convergence speed, validation/test loss, generation quality (reduced repetition, higher unique token ratios), and sentence termination.

---

## 2. Configuration Comparison (V4/V5 Capacity-Control)

| Hyperparameter | 3.38M Model (Phase 12B) | 10.28M Model (Phase 14) | Match Status |
|---|---|---|---|
| **Tokenizer** | 1.0-BPETokenizer | 1.0-BPETokenizer | **IDENTICAL** |
| **Vocabulary Size** | 8,000 | 8,000 | **IDENTICAL** |
| **Layers** | 6 | 6 | **IDENTICAL** |
| **Attention Heads** | 6 | 8 | *Scaled (Width)* |
| **Embedding Dim (d_model)**| 192 | 384 | *Scaled (Width)* |
| **Feedforward Dim (d_ff)** | 384 | 768 | *Scaled (Width)* |
| **Weight Tying** | True | True | **IDENTICAL** |
| **Dropout** | 0.1 | 0.1 | **IDENTICAL** |
| **Context Length** | 256 | 256 | **IDENTICAL** |
| **Optimizer** | AdamW | AdamW | **IDENTICAL** |
| **Base Learning Rate** | 6e-4 | 6e-4 | **IDENTICAL** |
| **Min Learning Rate** | 6e-5 | 6e-5 | **IDENTICAL** |
| **Scheduler** | Cosine Warmup | Cosine Warmup | **IDENTICAL** |
| **Warmup Steps** | 150 | 150 | **IDENTICAL** |
| **Weight Decay** | 0.01 | 0.01 | **IDENTICAL** |
| **Gradient Clipping** | 1.0 | 1.0 | **IDENTICAL** |
| **Batch Size** | 4 | 4 | **IDENTICAL** |
| **Gradient Accumulation** | 4 | 4 | **IDENTICAL** |
| **Random Seed** | 1337 | 1337 | **IDENTICAL** |
| **Initialization** | Normal (std=0.02) | Normal (std=0.02) | **IDENTICAL** |
| **CPU Threads** | Default PyTorch | Default PyTorch | **IDENTICAL** |

---

## 3. Dataset Verification
- **Dataset Version**: Both models trained on `collision_dataset_v5_expanded`.
- **Splits**: Verified that train, validation, and test splits were identical binary file inputs (`train.bin`, `val.bin`, `test.bin`).
- **Tokenizer**: Both models used the BPE tokenizer configured at `artifacts/tokenizer`.

---

## 4. Training Budget Comparison
- **Training Steps (micro-batches)**: 1,500
- **Batch Size**: 4
- **Context Length**: 256
- **Tokens per Batch**: 1,024 tokens
- **Optimizer Steps**: 375 steps (1,500 / 4)
- **Total Training Tokens Processed**: **1,536,000 tokens** (exactly identical for both runs)
- **Epochs / Dataset Passes**: **0.993 passes** (~1 epoch over 1,546,977 train split tokens)

---

## 5. Learning-Curve Analysis
- **10M Model Metrics**:
  - Step 500: Val Loss = `2.6326`
  - Step 1000: Val Loss = `0.9119`
  - Step 1500: Val Loss = `0.4824`
- **Val Loss Improvement (Step 1000 to 1500)**: Reduced by **47%** (`0.9119` to `0.4824`).
- **Classification**: **B. Still improving**. The learning curve shows no plateauing; training loss (`0.3987`) and validation loss (`0.4824`) are still decreasing sharply.

---

## 6. Evaluation Verification
- **Evaluation Splits & Code**: Both models evaluated using the same test split (`test.bin`) and non-overlapping sequence segmentation code.
- **Measured Test Metrics**:
  - 3.38M Test Loss: **1.3213** (Perplexity: **3.75**)
  - 10M Test Loss: **0.7679** (Perplexity: **2.16**)

---

## 7. Generation Metric Verification
- **Parameters**: `temp=0.7`, `top_k=50`, `top_p=0.9` (omitted if top_k is dominant), `max_new_tokens=100`.
- **Repetition Rate**: 3.38M = `47.9%` -> 10M = `26.8%` (repetition nearly halved).
- **Unique Token Ratio**: 3.38M = `52.1%` -> 10M = `73.2%` (improved vocabulary usage).
- **Sentence Termination Rate**: 3.38M = `55.6%` -> 10M = `88.9%` (correct EOS emit behavior).

---

## 8. Data / Parameter Ratio
- **3.38M Model**: `1,536,000 / 3,375,680 = 0.455` tokens/parameter
- **10.28M Model**: `1,536,000 / 10,282,304 = 0.149` tokens/parameter
- **Interpretation**: Under standard scaling laws (Chinchilla requires ~20 tokens/parameter), the 10M model is heavily **data-limited** (under-trained). However, despite this constraint, it achieves dramatic performance gains, proving that parameter width is highly beneficial even at low token regimes.

---

## 9. Capacity vs. Training Interpretation
- **Claim 1**: *"Increasing parameter count from 3.38M to 10.28M improved model quality."*
  - **SUPPORTED**: Massive improvements across perplexity, repetition, and termination rates.
- **Claim 2**: *"The 10M model has reached its effective capacity limit."*
  - **NOT SUPPORTED**: The validation curves are still sharply sloping downward at step 1500, indicating the model has further headroom if trained for more steps or on more tokens.

---

## 10. Threats to Validity
- **Low Data Ratio**: The model is highly under-trained, which might lead to overfitting if trained for too many epochs. However, validation and test splits show no sign of overfit.
- **Micro-Batch Steps**: Step counting based on micro-batches rather than optimizer steps could affect learning rate warmup/decay scheduling steps, but since this was identical in both runs, it is fully controlled.

---

## 11. Recommended Next Experiment
Since SFT instruction tuning failed on the 3.38M model due to capacity issues, the recommended next experiment is a **10M Model Instruction Tuning Pilot** using a small subset of the SFT dataset to determine whether the larger capacity successfully absorbs SFT conversation templates without generation regression.

---

## 12. Final Classification
**C. VALID BUT DATA-LIMITED**
The comparison is fully controlled and structurally valid, but both models (particularly the 10M model) are trained at a low data-to-parameter ratio.
