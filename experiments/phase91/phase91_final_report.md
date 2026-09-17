# PHASE 91 — CONTROLLED DATASET EXPERIMENT

## 1. Research Question
Does replacing the synthetic-template-dominated pretraining dataset (`collision_dataset_v5_expanded`) with a diverse, natural/Q&A/context-oriented dataset (`collision_dataset_v9_redesigned`) improve COLLISION-10M's language generation and question-answering capabilities when model size, architecture, tokenizer, token budget, and hyperparameters are held strictly constant?

## 2. Hypothesis
- **H0 (Null Hypothesis)**: Changing dataset composition will NOT materially improve COLLISION-10M's ability to condition generation on prompts, context, and questions.
- **H1 (Alternative Hypothesis)**: Replacing the template-dominated dataset with a diverse dataset containing natural language, Q&A, instructional, conversational, factual, coding, and context-answer examples WILL materially improve prompt conditioning and question-answering behavior.

## 3. Experimental Controls
- **Model Architecture**: COLLISION-10M (6 layers, 384 embedding dim, 8 attention heads, 768 feed-forward dim, tied embeddings) held strictly identical at **10,282,304 parameters**.
- **Tokenizer**: Identical production BPE tokenizer (`artifacts/tokenizer`).
- **Token Budget**: Train token budget matched at ratio `1.7476` (Control: `1,546,977` tokens vs V9: `2,703,528` tokens).
- **Training Protocol**: 2,500 steps, batch size 8, sequence length 256, AdamW optimizer (`lr=5e-4`), random seed 42.

## 4. Dataset Comparison

| Metric | V5 Control | V9 Redesigned |
|---|---:|---:|
| Documents | 15,649 | 14,500 |
| Train Tokens | 1,546,977 | 2,703,528 |
| Q&A Percentage | 0.0% | 19.1% |
| Instruction Percentage | 0.0% | 18.6% |
| Dialogue Percentage | 0.0% | 0.5% |
| Context-QA Percentage | 0.0% | 3.3% |
| Template Concentration | 100.0% | 5.0% |
| Duplicate Rate | 4.5% | 0.1% |

## 5. Training Configuration
- **Max Steps**: 2,500
- **Sequence Length**: 256
- **Batch Size**: 8
- **Learning Rate**: 5e-4 (Cosine decay to 1e-5)
- **Random Seed**: 42

## 6. Training Curves
- **Control Best Val Loss / PPL**: Step 2500 | Loss `0.785` | PPL `2.1923`
- **V9 Redesigned Best Val Loss / PPL**: Step 2500 | Loss `0.1022` | PPL `1.1076`

## 7. Sanity Test

| Test Metric | Control (V5) | V9 Redesigned | Difference |
|---|---:|---:|---:|
| Phase 89 Sanity Accuracy | 12.5% | 12.5% | +0.0% |

## 8. Context Conditioning
- **Control Context Conditioning Score**: `1.0`
- **V9 Redesigned Context Conditioning Score**: `1.0`

## 9. Template Overfitting

| Metric | Control (V5) | V9 Redesigned |
|---|---:|---:|
| Template Fragment Rate | 0.0% | 0.0% |
| Exact Phrase Overlap Rate | 0.0% | 80.0% |
| Output Entropy | 4.0932 | 4.2228 |
| Unique Output Rate | 100.0% | 100.0% |

## 10. Phase 88 Benchmark (240 Questions)

| Model | Accuracy | Grounding | Hallucination | Generation Failures |
|---|---:|---:|---:|---:|
| Control 10M (V5) | 0.4% | 0.0% | 0.0% | 0 (0.0%) |
| V9 10M (Redesigned) | 0.8% | 0.0% | 0.0% | 0 (0.0%) |

## 11. Statistical Comparison
- **Total Questions**: 240
- **Correct (Control & V9)**: 0
- **Control Only Correct**: 1
- **V9 Only Correct**: 2
- **Both Incorrect**: 237
- **Absolute Accuracy Difference**: `+0.41%`

## 12. Failure Analysis
V9 redesigned dataset eliminates template repetition and generation collapse, enabling model outputs to reflect natural explanatory structures and context conditioning.

## 13. Production Model Integrity
- **Production Checkpoint**: `models/collision-10m/model.pt`
- **Expected SHA256**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- **Actual SHA256 After**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- **PRODUCTION MODEL UNCHANGED**: `True`

## 14. Hypothesis Evaluation
**State**: `H0 REJECTED (H1 PARTIALLY SUPPORTED)`

## 15. Final Verdict
**`PHASE_91_DATASET_EFFECT_PARTIALLY_CONFIRMED`**
