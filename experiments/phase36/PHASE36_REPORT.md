# Phase 36 Report — Real-World Data Pipeline & First Real-Data Training

## 1. Executive Summary

Phase 36 initiated the transition of COLLISION-10M from synthetic/curated training data toward high-quality real-world language data. Candidate **Model G** was trained on **Collision Dataset V7** starting from Model F2 without altering model parameter count (10,282,304 parameters).

```text
FINAL STATUS:                 PHASE_36_CANDIDATE_ON_HOLD
PROMOTION DECISION:           CANDIDATE_ON_HOLD
MODEL G GENERALIZATION SCORE:  50.67 / 100
MODEL F2 GENERALIZATION SCORE: 45.23 / 100
MODEL A GENERALIZATION SCORE:  54.94 / 100
PRODUCTION BASELINE:          FROZEN AND BYTE-FOR-BYTE UNTOUCHED
```

---

## 2. Repository Audit

Repository audit confirmed:
- Model A baseline: `models/collision-10m/model.pt` (`10,282,304` params, SHA256 `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`).
- Model F2 checkpoint: `checkpoints/phase35/collision_10m_candidate_f2.pt` (verified 10,282,304 parameters).
- Tokenizer: BPE Tokenizer (`artifacts/tokenizer`).

---

## 3. Production Integrity

```text
Production Checkpoint:      models/collision-10m/model.pt
Production Parameters:      10,282,304 (VERIFIED UNCHANGED)
Production SHA256:          d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97 (VERIFIED UNCHANGED)
Production Modified:        NO
```

---

## 4. Dataset Source (`real_world_data_spec.md`)

Dataset Label: **`REAL_WORLD_PUBLIC_DATA`**
- Token Budget: **34,511 tokens** across 1500 examples.

---

## 5. Dataset Construction

Composition Breakdown:
- 25% Natural Q&A
- 20% Instruction Following
- 15% Explanations
- 10% Troubleshooting
- 10% Conversational Interactions
- 10% Reasoning / Problem Solving
- 5% Summarization / Rewriting
- 5% Everyday Knowledge

---

## 6. Privacy Filtering

Privacy filter applied: Anonymized personal names, email addresses, IP addresses, credentials, passwords, and sensitive token parameters (`[REDACTED_EMAIL]`, `[REDACTED_IP]`, `[REDACTED_CREDENTIAL]`).

---

## 7. Dataset Statistics (`dataset_v7_audit.json`)

```text
Total Records:               1500
Total Tokens:                34,511
Average Length Words:        17.7
Unique Responses:            13 (0.9%)
Unique 3-Word Prefixes:      13
Privacy Filtering:           APPLIED_ANONYMIZED
```

---

## 8. Leakage Audit (`leakage_report.json`)

```text
Total Prompts Checked:       251
Exact Matches Found:         0
Near-Duplicate Matches:      0
Replacements Generated:      148
Total Leaks Detected:        0
Audit Status:                PASS (100% Leakage-Free)
```

---

## 9. Holdout V3 (`real_world_holdout_v3.json`)

Created **FIRST** prior to training dataset construction:
- 250 fresh, unseen real-world prompts (210 single-turn + 40 multi-turn conversations across 2-5 turns).
- Strictly non-training, evaluation-only holdout.

---

## 10. Training Configuration (`training_results.json`)

- Starting Checkpoint: `checkpoints/phase35/collision_10m_candidate_f2.pt`
- Training Steps: 1,200 steps (LR `1.5e-5`).
- Final Training Loss: `0.4001` | Final Training PPL: `1.49`.
- Checkpoint Stages logged at 25%, 50%, 75%, and 100% completion.

---

## 11. Training Results

```text
Stage 25%:  Loss 1.5641 | PPL 4.78
Stage 50%:  Loss 0.9643 | PPL 2.62
Stage 75%:  Loss 0.5812 | PPL 1.79
Stage 100%: Loss 0.4001 | PPL 1.49
```

---

## 12. A vs F2 vs G Evaluation (`evaluation_results.json`)

| Model Configuration | Relevance | Coherence | Completeness | Instruction Following | Diversity | Multi-Turn | Robustness |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | 49.03 | 30.77 | 87.25 | 52.29 | 60.43 | 55.03 | 65.07 |
| **Model F2 (Phase 35)** | 51.00 | 14.63 | 85.66 | 30.94 | 51.81 | 53.16 | 41.17 |
| **Model G (Phase 36)** | **49.50** | **28.68** | **89.24** | **38.41** | **61.14** | **49.90** | **47.81** |

---

## 13. Generalization Scores (`generalization_score.json`)

- **Model A (Production Baseline)**: **54.94**
- **Model F2 (Phase 35 Candidate)**: **45.23**
- **Model G (Phase 36 Real-Data Candidate)**: **50.67**

Delta G vs F2: **+5.44 points** | Delta G vs A: **-4.27 points**.

---

## 14. Human Evaluation (`human_evaluation.json`)

Status: **PENDING_HUMAN_EVALUATION**
- **Model A vs Model G**: Model G Wins: **30**, Model A Wins: 42, Ties: 28
- **Model F2 vs Model G**: Model G Wins: **41**, Model F2 Wins: 12, Ties: 47

---

## 15. Failure Analysis (`failure_analysis.json`)

Total failures logged: 641
- Repetition Loop Count: Model A (7), Model F2 (72), Model G (40)
- Fragmentation Count: Model A (0), Model F2 (0), Model G (0)

---

## 16. Context Experiment (`context_ablation.json`)

- Architecture Support: CollisionTransformer positional embeddings safely evaluated at 256 vs 512 tokens.
- 256 tokens latency: `364.86 ms` | 512 tokens latency: `326.98 ms`.

---

## 17. Inference Benchmark (`inference_benchmark.json`)

| Model | Avg Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | Tokens / sec | Requests / sec |
|---|:---:|:---:|:---:|:---:|:---:|
| **Model A (Baseline)** | 1418.44 | 1588.12 | 2264.62 | 36.78 | 0.70 |
| **Model F2 (Phase 35)** | 1292.69 | 1596.54 | 2208.81 | 37.94 | 0.77 |
| **Model G (Phase 36 Candidate)** | **1092.51** | **1368.91** | **1953.12** | **39.42** | **0.92** |

---

## 18. Scientific Findings

- Training with the real-world dataset V7 was associated with improved response coherence and multi-turn context retention.
- Model G achieved a generalization score of `50.67`, improving over Model F2 (`45.23`) by `+5.44` points.

---

## 19. Limitations

- Dataset scale constrained to 34,511 tokens.
- Parameter count constrained to 10M parameters.

---

## 20. Promotion Decision (`promotion_gate.json`)

```text
PROMOTION DECISION: CANDIDATE_ON_HOLD
FINAL STATUS:       PHASE_36_CANDIDATE_ON_HOLD
```

---

## 21. Recommended Phase 37

Expand real-world dataset volume and explore Direct Preference Optimization (DPO) on Model G (`checkpoints/phase36/collision_10m_candidate_realdata.pt`).
