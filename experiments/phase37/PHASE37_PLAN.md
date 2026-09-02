# Phase 37 Plan — Real-World Data Scale-Up + DPO

## 1. Mission & Research Objective

Phase 37 tests whether scaling up high-quality real-world language data (Dataset V8: 250k–500k tokens), applying preference optimization / DPO (`preference_dataset_v1`), or combining both (Candidate H3) enables COLLISION-10M to surpass Production Model A (54.94) and Model G (50.67) without increasing parameter count (10,282,304 parameters) or modifying the frozen production baseline.

---

## 2. Frozen Production Baseline Integrity

- Checkpoint: `models/collision-10m/model.pt`
- Expected Parameters: `10,282,304`
- Expected SHA256: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- Verified baseline integrity recorded in `experiments/phase37/production_integrity_before.json`.

---

## 3. Starting Baseline & Experiment Matrix

Research Baseline: `Model G — Phase 36 Candidate` (`checkpoints/phase36/collision_10m_candidate_realdata.pt`).

Candidates:
1. **Model H1**: Real-world data scale-up on Dataset V8 (250k–500k tokens) starting from Model G. Checkpoint: `checkpoints/phase37/collision_10m_candidate_h1.pt`.
2. **Model H2**: Preference Optimization / DPO on `preference_dataset_v1.json` starting from Model G. Checkpoint: `checkpoints/phase37/collision_10m_candidate_h2.pt`.
3. **Model H3**: Combined adaptation (Dataset V8 scale-up followed by preference optimization). Checkpoint: `checkpoints/phase37/collision_10m_candidate_h3.pt`.

---

## 4. Holdout V4 & Leakage Audit

- Created **FIRST** prior to dataset training.
- 350 fresh unseen prompts (300 single-turn + 50 multi-turn conversations across 2–5 turns).
- Data Leakage Audit against all prior datasets (Phase 30–37). Target: **0 leaks**. Output `experiments/phase37/leakage_report.json`.

---

## 5. Dataset V8 & Preference Dataset V1 Specs

- Dataset V8: 250,000–500,000 tokens of `REAL_WORLD_PUBLIC_DATA` with privacy filtering and quality deduplication (`dataset_v8_audit.json`).
- Preference Dataset V1: 5,000–10,000 preference pairs (`prompt`, `chosen`, `rejected`) labeled `CURATED_REALISTIC_DATA` (`preference_dataset_audit.json`).

---

## 6. Evaluation Framework

- Locked decoding settings: `temp=0.7`, `top_k=40`, `top_p=0.9`, `max_tokens=60`, `seed=42`, `context_len=256`.
- 6-Model Comparison: Model A, Model F2, Model G, Model H1, Model H2, Model H3 on Holdout V4.
- Metrics: Relevance (20%), Coherence (20%), Completeness (15%), Instruction Following (15%), Diversity (10%), Multi-Turn (10%), Failure Robustness (10%).
- Artifacts: `evaluation_results.json`, `human_evaluation.json`, `failure_analysis.json`, `generalization_score.json`, `context_ablation.json`, `inference_benchmark.json`, `promotion_gate.json`, `PHASE37_REPORT.md`.
