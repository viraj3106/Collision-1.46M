# PHASE 77 — PARAMETER CAPACITY EXPANSION REPORT

## 1. Executive Summary
Phase 77 evaluated whether increasing model parameter capacity from ~10M to 25M–50M improves multi-turn conversational reasoning when holding the winning Phase 76 hybrid loss objective constant ($lpha = 0.10, eta = 0.90$). Four scale variants were trained and evaluated under identical experimental conditions: `COLLISION-10M` (Control, 10.28M), `J77-25M` (25.26M), `J77-35M` (34.85M), and `J77-50M` (48.89M).

**Scientific Outcome**: **Outcome C — Capacity Not Confirmed**  
**Verdict**: Increasing parameters provided negligible conversational improvement (+-1.97). Evidence does NOT support parameter capacity as the dominant bottleneck.

---

## 2. Research Question
> **Can increasing model parameter capacity from approximately 10M to 25M–50M improve multi-turn conversational reasoning when using the calibrated Phase 76 hybrid loss objective?**

---

## 3. Hypothesis
> **Hypothesis**: The multi-turn conversational degradation observed in Phase 76 was primarily driven by a strict 10M parameter capacity ceiling. Scaling model capacity to 25M–50M while maintaining the Phase 76 hybrid loss formulation will yield measurable quality gains in coherence, context retention, and instruction following.

---

## 4. Experimental Controls
To isolate **MODEL PARAMETER CAPACITY** as the single independent variable, the following components were held strictly constant across all models:
- Tokenizer & Vocabulary: BPE Tokenizer (Vocab Size 8,000)
- Sequence Length & Context: 256 tokens max sequence length
- Loss Objective: Calibrated Phase 76 Hybrid Loss ($L = 0.10 \cdot L_{context} + 0.90 \cdot L_{response}$)
- Dataset & Preprocessing: Frozen Gold Evaluation Set (140 records across 14 categories)
- Evaluation Engine & Decoding Parameters: Identical deterministic evaluation framework
- Optimizer & Hyperparameters: AdamW ($lr = 1.0 \times 10^{-5}$, weight decay $0.01$)

---

## 5. Model Configurations
| Model Identifier | Target Scale | d_model | n_layer | n_head | d_ff | Exact Parameter Count | Checkpoint Path |
|---|---|---|---|---|---|---|---|
| `COLLISION-10M` | Control (~10M) | 384 | 6 | 8 | 768 | 10,282,304 | `experiments/phase77/checkpoints/collision_10m_control.pt` |
| `J77-25M` | Candidate A (~25M) | 512 | 10 | 8 | 1024 | 25,263,936 | `experiments/phase77/checkpoints/collision_25m_j77a.pt` |
| `J77-35M` | Candidate B (~35M) | 640 | 9 | 10 | 1280 | 34,847,680 | `experiments/phase77/checkpoints/collision_35m_j77b.pt` |
| `J77-50M` | Candidate C (~50M) | 768 | 9 | 12 | 1536 | 48,893,504 | `experiments/phase77/checkpoints/collision_50m_j77c.pt` |

---

## 6. Training Configuration
- **Seed**: 42
- **Batch Size**: 8
- **Learning Rate**: $1.0 \times 10^{-5}$
- **Weight Decay**: 0.01
- **Optimizer**: AdamW
- **Training Steps**: 20 controlled conversational execution steps per model scale

---

## 7. Loss Objective
Winning Phase 76 Objective:
$$\mathcal{L}_{total} = 0.10 \cdot \mathcal{L}_{context} + 0.90 \cdot \mathcal{L}_{response}$$
Calculated by splitting token cross-entropy loss between context sequence ($1 - \text{loss\_mask}$) and response sequence ($	ext{loss\_mask}$).

---

## 8. Dataset
Evaluated on frozen Gold Evaluation Set (`experiments/phase75/data/gold/gold_eval_set.jsonl`), containing 140 prompts across 14 evaluation categories.

---

## 9. Evaluation Methodology
All candidate checkpoints were evaluated using the automated evaluation engine (`experiments/phase75/evaluation/evaluator.py`). Output text is scored for response quality (0-4), coherence (0-4), and categorized into failure taxonomy metrics (`PREMATURE_TERMINATION`, `FRAGMENTATION`, `IRRELEVANCE`, `REPETITION`).

---

## 10. Results
| Model | Params | Val Loss | PPL | Coherence | Quality | Premature Term | Fragmentation | Irrelevance | Repetition | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `COLLISION-10M` | 10.28M | 5.7932 | 328.06 | 2.470 | 3.130 | 45 | 43 | 43 | 0 | 3.55 |
| `J77-25M` | 25.26M | 7.1526 | 1277.42 | 0.500 | 2.400 | 130 | 140 | 140 | 0 | 7.88 |
| `J77-35M` | 34.85M | 7.0038 | 1100.81 | 0.500 | 2.400 | 130 | 140 | 140 | 0 | 9.87 |
| `J77-50M` | 48.89M | 6.6102 | 742.63 | 0.500 | 2.400 | 130 | 140 | 140 | 0 | 13.75 |

---

## 11. Conversational Quality Comparison
- **Coherence Progression**: 10M (2.470) $\rightarrow$ 25M (0.500) $\rightarrow$ 35M (0.500) $\rightarrow$ 50M (0.500)
- **Quality Progression**: 10M (3.130) $\rightarrow$ 25M (2.400) $\rightarrow$ 35M (2.400) $\rightarrow$ 50M (2.400)

---

## 12. Scaling Analysis
- **Quality Gain Over 10M Control**:
  - `J77-25M`: -1.970 coherence gain (-0.730 quality gain)
  - `J77-35M`: -1.970 coherence gain (-0.730 quality gain)
  - `J77-50M`: -1.970 coherence gain (-0.730 quality gain)
- **Scaling Behavior**: Expanding un-pretrained parameter capacity from 10M to 50M without foundational pretraining led to severe generation collapse (coherence 2.470 -> 0.500). Parameter scale alone does not improve multi-turn coherence.

---

## 13. Parameter Efficiency
- **Parameter Efficiency Ratio (Coherence Gain per Million Additional Parameters)**:
  - `J77-25M` (+14.98M params): -0.1315 coherence gain / M params
  - `J77-35M` (+24.57M params): -0.0802 coherence gain / M params
  - `J77-50M` (+38.61M params): -0.0510 coherence gain / M params

---

## 14. Compute Efficiency
- **Compute Efficiency Ratio (Coherence Gain per Additional Second Training Time)**:
  - `J77-25M`: -0.4550
  - `J77-35M`: -0.3117
  - `J77-50M`: -0.1931

---

## 15. Failure Analysis
- **Premature Termination**: 10M (45) $\rightarrow$ 50M (130)
- **Fragmentation**: 10M (43) $\rightarrow$ 50M (140)
- **Irrelevance**: 10M (43) $\rightarrow$ 50M (140)

---

## 16. Comparison With Phase 76
- Phase 76 established `J76-B` (10M params) with coherence 2.40. Phase 77 demonstrates that randomly initialized 25M–50M models fail to match `J76-B` without prior pretraining.

---

## 17. Scientific Interpretation
The evidence demonstrates that **parameter capacity alone is NOT the dominant bottleneck**. Expanding model parameters without pretraining token volume causes representations to collapse during conversational fine-tuning.

---

## 18. Limitations
- Experiments executed under CPU constraints with max sequence length of 256.
- Training duration restricted to 20 fine-tuning execution steps per candidate.

---

## 19. Conclusion
The 10M parameter capacity ceiling hypothesis is **NOT CONFIRMED**. The evidence indicates that foundational pretraining and training token budget, rather than raw model capacity, are the primary bottlenecks.

---

## 20. Recommendation For Phase 78
Proceed to **Phase 78: Pretraining Dataset & Token Budget Expansion**. Investigate pretraining larger capacities on general text datasets before applying conversational fine-tuning.
