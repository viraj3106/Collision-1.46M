# Phase 33 Plan — Synthetic Diversity Expansion & Multi-Turn Dataset V2

## 1. Executive Baseline & Safety Audit

```text
Production Checkpoint:   models/collision-10m/model.pt
Parameter Count:         10,282,304 (VERIFIED FROZEN)
SHA256 Checksum:         d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97 (VERIFIED FROZEN)
Status:                  STRICTLY FROZEN AND UNTOUCHED
```

---

## 2. Phase 32 Audit Findings & Root Cause Analysis

### Findings from Phase 32 Report
1. **Quantitative Performance**: Model D (`COLLISION-10M + Augmented v1`) achieved an impressive drop in validation perplexity (`322.58 PPL` → `5.12 PPL`) and validation loss (`5.7764` → `1.6327`).
2. **Qualitative & Blind Evaluation Deficit**: Despite low PPL, in blind pairwise evaluation (`A_vs_D`), baseline Model A won **19 prompts**, Model D won **8 prompts**, and **21 prompts** were ties.
3. **Synthetic Dataset Audit Root Cause**:
   - Total synthetic instructions in v1: `120`
   - Total unique responses in v1: `30`
   - Type-Token Ratio (TTR): `0.1621` (extremely narrow vocabulary and templated response structures).
   - **Diagnosis**: Model D suffered from **Synthetic Template Concentration / Memorization**. Low perplexity was an artifact of predicting repetitive synthetic response structures.

---

## 3. Phase 33 Core Research Hypothesis

> **Hypothesis**: Expanding synthetic dataset size and structural diversity (target `>= 400` unique responses, `unique_response_ratio >= 0.75`) while introducing multi-turn conversational structures will resolve template collapse and significantly improve real-world open-ended generation quality in `COLLISION-10M` without increasing model size.

---

## 4. Planned Dataset Composition (Synthetic V2 & Augmented V2)

### Dataset Composition Targets (600–1,000 Examples)
- **Declarative Knowledge (15%)**: Concise factual statements, definitions, and technical concepts.
- **Explanations (20%)**: Step-by-step breakdowns, beginner vs advanced explanations, simple analogies.
- **Question Answering (15%)**: Factual, conceptual, comparison, "why", and "how" questions.
- **Instruction Following (15%)**: Summarization, classification, bulleted extraction, rephrasing.
- **Completion (10%)**: Technical and conceptual continuations.
- **Multi-Turn Conversations (15%)**: 2–4 turn dialogues with follow-ups, clarifications, and context retention.
- **Reasoning / Structured Thinking (10%)**: Tradeoff analysis, step sequencing, logical deduction.

### Domain Coverage
Computer Science, Artificial Intelligence, Machine Learning, Physics, Mathematics, Technology, Space, General Knowledge, Software Engineering, Data Science, Networking, Databases, Cybersecurity, Electronics, Everyday reasoning.

---

## 5. Evaluation Strategy & Success Criteria

### 3-Way Comparative Benchmark
- **Model A**: Production Baseline (`models/collision-10m/model.pt`)
- **Model D**: Phase 32 Candidate (`checkpoints/phase32/collision_10m_production_candidate_v1.pt`)
- **Model E**: Phase 33 Model (`checkpoints/phase33/collision_10m_production_candidate_v2.pt`)

### Evaluation Components
1. **Diversity Audit**: `experiments/phase33/audit_dataset_diversity.py` (V1 vs V2 comparison).
2. **Novel Benchmark Suite V2**: 100+ leakage-free prompts across 15 domains.
3. **Open-Ended Test**: 30 prompts requiring non-templated reasoning.
4. **Multi-Turn Benchmark**: 20 multi-turn dialogues measuring context retention and follow-up correctness.
5. **Blind Pairwise Preference**: 50 prompts comparing A vs D, A vs E, and D vs E.
6. **Inference Benchmarking & API Compatibility**: Latency/throughput + FastAPI endpoint tests.

### Success Gate Criteria
- `[ ]` Parameters: `10,282,304` (architecture unchanged)
- `[ ]` Synthetic V2: `>= 500` examples, `>= 400` unique responses, `unique_response_ratio >= 0.75`
- `[ ]` Leakage: `0 leaks` on benchmark suite v2
- `[ ]` Multi-turn & Open-ended generation quality improved
- `[ ]` Blind preference: Model E beats Model D (`E > D`), and closes gap or beats Baseline A (`E >= A`)
- `[ ]` Test suite passes: `31/31 PASS`
- `[ ]` Production baseline unchanged (`d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`)
