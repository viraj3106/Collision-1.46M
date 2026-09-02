# Phase 30 Report — Real-World Dataset Collection & Controlled Fine-Tuning Experiment

## 1. Executive Summary

Phase 30 evaluates the central research question: **Does carefully cleaned real-world beta feedback improve COLLISION-10M?**

Using a controlled experimental setup, we isolated the effect of fine-tuning `COLLISION-10M` on cleaned beta user feedback. The production model (`COLLISION-10M`, 10,282,304 parameters) remained strictly frozen and untouched.

---

## 2. Baseline Model Specification

```text
Model Architecture:       COLLISION-10M Transformer (12 Layers, 256 Model Dim, 8 Heads)
Parameters:               10,282,304
Production Checkpoint:    models/collision-10m/model.pt
SHA256 Checksum:          d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97
Context Length:           256 Tokens
Vocabulary Size:          8,000 (Active BPE)
```

---

## 3. Real-World Feedback Dataset (v1)

```text
Raw Feedback Evaluated:   11 Examples
Cleaned Training Examples: 6 Examples (Consent=True, Thumbs Up, Filtered Credentials)
Rejected Examples:        5 Examples (Consent Declined, Thumbs Down, Email Leak, API Key Leak, Duplicates)
Train Split:              5 Examples (320 Tokens) -> datasets/collision_instruct_v1/real_world_train.jsonl
Validation Split:         1 Example  (66 Tokens)  -> datasets/collision_instruct_v1/real_world_val.jsonl
Total Dataset Tokens:     386 Tokens
Avg Example Length:       64.33 Tokens
```

---

## 4. Controlled Fine-Tuning Setup

```text
Baseline Checkpoint:      models/collision-10m/model.pt (Unchanged & Frozen)
Experimental Checkpoint:  checkpoints/phase30/collision_10m_realworld_v1.pt
Optimizer:                AdamW (lr=1e-4, weight_decay=0.01)
Fine-Tuning Steps:        50 Steps
Training Time:            18.4 Seconds (CPU execution)
Random Seed:              42
```

---

## 5. Quantitative Evaluation Results

| Metric | COLLISION-10M Baseline | Fine-Tuned (Real-World v1) | Difference / Delta |
|---|:---:|:---:|:---:|
| **Validation Loss** | `5.5977` | `4.7779` | **-0.8198** |
| **Validation Perplexity** | `269.81` | `118.85` | **-150.96 PPL** |
| **Unigram Repetition Rate** | `0.1840` | `0.3420` | **+0.1580** (Increased Repetition) |
| **Out-of-Domain Generalization**| High | Low | Catastrophic Overfitting on Small Dataset |

---

## 6. Qualitative & Preference Evaluation

- **Domain In-Sample Prompts**: The fine-tuned model improved formatting on direct machine learning Q&A patterns matching the feedback data.
- **Out-of-Domain Generalization Prompts**: Fine-tuning on 5 training examples (386 tokens) caused catastrophic overfitting—the model began repeating phrases from the small training set when given general knowledge prompts.
- **Blind Preference Distribution**:
  - Baseline Preferred: 4 / 6 Prompts (66.7%)
  - Fine-Tuned Preferred: 1 / 6 Prompts (16.7%)
  - Ties: 1 / 6 Prompts (16.7%)

---

## 7. Research Conclusion

```text
Result Classification: REGRESSION (Catastrophic Overfitting Due to Small Dataset Size)
```

### Key Research Findings:
1. **Perplexity vs. Generalization Disconnect**: Fine-tuning on a tiny feedback batch significantly reduced validation perplexity on the feedback domain (269.81 → 118.85 PPL), but severely degraded output diversity and generalization across broader knowledge domains.
2. **Dataset Size Requirement**: 5 training examples (386 tokens) are insufficient for fine-tuning `COLLISION-10M` without overfitting. At least 1,000+ diverse, high-quality feedback records are necessary before executing production fine-tuning.
3. **Safety & Baseline Preservation**: Isolating the experimental checkpoint in `checkpoints/phase30/` ensured zero impact on the frozen production model.

---

## 8. Final Production Integrity Gate

```text
Production Checkpoint Path: models/collision-10m/model.pt
Parameter Count:            10,282,304 (VERIFIED UNCHANGED)
SHA256 Checksum:            d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97 (VERIFIED UNCHANGED)
```

---

## 9. Next Recommendation

Proceed to **Phase 31 — Real-World Beta Telemetry Scale-Up & Synthetic Augmentation Pipeline**. Focus on expanding real-world user feedback collection while building synthetic instruction augmentation to prevent dataset scarcity overfitting.
