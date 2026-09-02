# Phase 35 Plan — Natural Instruction & Conversation Alignment

## 1. Mission & Objectives

Phase 35 addresses the core findings of Phase 34: while Phase 33 Model E improved over Model D (+6.22 points), it did not yet outperform the production baseline Model A.
- **Primary Objective**: Build **Model F** (controlled variants F1 and F2) starting from Model E to improve natural instruction following, conversational behavior, follow-up understanding, context retention, clarification behavior, troubleshooting, and natural explanations without increasing model size (maintaining 10,282,304 parameters).
- **Target**: `Model F > Model E` and ideally `Model F > Model A` on an unseen real-world holdout V2.

---

## 2. Frozen Production Baseline Safety & Integrity

- Baseline Checkpoint: `models/collision-10m/model.pt`
- Required Parameters: `10,282,304`
- Required SHA256: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- Baseline SHA256 and parameter count recorded in `experiments/phase35/production_integrity_before.json`.
- Verification conducted before and after all training and evaluation operations to ensure production baseline remains byte-for-byte untouched.

---

## 3. Model Lineage & Architecture

```text
Production Baseline Model A (models/collision-10m/model.pt)
               ↓
Phase 32 Model D (checkpoints/phase32/collision_10m_production_candidate_v1.pt)
               ↓
Phase 34 Model E (checkpoints/phase33/collision_10m_production_candidate_v2.pt)
               ↓
Phase 35 Model F1 / F2 (checkpoints/phase35/collision_10m_candidate_f1.pt / f2.pt)
```

- Architecture: 10M Parameter Transformer (Config: `vocab_size=32000`, `max_seq_len=256`, `dim=256`, `n_layers=8`, `n_heads=8`).

---

## 4. Unseen Real-World Holdout V2 (`real_world_holdout_v2.json`)

- Created **FIRST** before training dataset V6 creation.
- 220 fresh unseen prompts (190 single-turn + 30 multi-turn dialogues, 2-5 turns each).
- Task Mix: 25% Natural Q&A, 20% Instruction Following, 15% Explanations, 10% Troubleshooting, 10% Conversational Follow-ups, 10% Reasoning, 5% Summarization / Rewriting, 5% Everyday Knowledge.
- Data Leakage Audit: Target **0 leaks** against all training datasets from Phase 30–34. Output `experiments/phase35/leakage_report.json`.

---

## 5. Collision Dataset V6 & Quality Audit

- Dataset V6 created with natural user language, direct answers, concise explanations, and multi-turn follow-ups.
- Quality Audit output to `experiments/phase35/dataset_v6_audit.json`.

---

## 6. Training & Evaluation Setup

- **F1**: Conservative fine-tuning adaptation on Dataset V6 (LR 1e-5, 300 steps).
- **F2**: Slightly longer adaptation on Dataset V6 (LR 2e-5, 600 steps).
- **Locked Decoding Settings**: `temp=0.7`, `top_k=40`, `top_p=0.9`, `max_tokens=60`, `seed=42`, `context_len=256`.
- **4-Model Evaluation**: Model A vs Model E vs Model F1 vs Model F2 on Holdout V2.
- **Metrics**: LM Loss/PPL, Generation metrics, Failure analysis (`failure_analysis.json`), 0-5 Multi-Turn scores, 100-prompt Blind Human Preference (`human_evaluation.json`), Generalization Scores (`generalization_score.json`), Shadow Beta (`shadow_beta_report.json`), Inference Benchmarks (`inference_benchmark.json`), Unit Tests (`python -m unittest discover tests`).

---

## 7. Promotion Gate Criteria

- Candidate marked `PHASE_35_PASS` only if `F > E` AND `F >= A + 3.0` points, zero leakage, unit tests pass (`>= 31 PASS`), and production integrity is preserved.
