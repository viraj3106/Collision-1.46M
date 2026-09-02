# Phase 36 Plan — Real-World Data Pipeline + First Real-Data Training

## 1. Mission & Scope

Phase 36 begins the transition of COLLISION-10M from synthetic/curated training data toward high-quality real-world language data.
- **Research Question**: Does fine-tuning on high-quality real-world language data improve generalization and natural interaction quality without increasing model size or modifying the frozen production model?
- **Constraints**:
  - Do NOT modify frozen production baseline (`models/collision-10m/model.pt`).
  - Maintain parameter count (`10,282,304`).
  - Do NOT jump directly into DPO. Establish a clean real-data baseline first.

---

## 2. Frozen Production Baseline Integrity

- Checkpoint: `models/collision-10m/model.pt`
- Expected Parameters: `10,282,304`
- Expected SHA256: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- Baseline SHA256 and parameter count logged in `experiments/phase36/production_integrity_before.json`.
- Verification conducted before and after all pipeline, training, and evaluation steps.

---

## 3. Model Lineage & Architecture

```text
Production Baseline Model A (models/collision-10m/model.pt)
               ↓
Phase 35 Model F2 (checkpoints/phase35/collision_10m_candidate_f2.pt)
               ↓
Phase 36 Model G (checkpoints/phase36/collision_10m_candidate_realdata.pt)
```

- Starting Checkpoint: `checkpoints/phase35/collision_10m_candidate_f2.pt` (verified 10,282,304 parameters).

---

## 4. Real-World Holdout V3 (`real_world_holdout_v3.json`)

- Created **FIRST** before training dataset V7 creation.
- 250 fresh, unseen real-world prompts (210 single-turn + 40 multi-turn dialogues, 2–5 turns each).
- Data Leakage Audit: Target **0 leaks** against all training datasets from Phase 30–35. Output `experiments/phase36/leakage_report.json`.

---

## 5. Collision Dataset V7 & Quality Audit

- Dataset V7: 100,000–500,000 tokens of high-quality real-world data (`datasets/collision_dataset_v7/collision_dataset_v7.jsonl`).
- Privacy Filtering: Removal/anonymization of PII, names, emails, API keys, credentials.
- Quality Filtering: Deduplication, length thresholding, template elimination.
- Dataset Quality Audit output to `experiments/phase36/dataset_v7_audit.json`.

---

## 6. Training & Evaluation Setup

- **Candidate Model G**: Fine-tuned on Dataset V7 starting from Model F2 state dict (`10,282,304` params) with checkpoint logging at 25%, 50%, 75%, and 100% completion (`training_results.json`).
- **Locked Decoding Settings**: `temp=0.7`, `top_k=40`, `top_p=0.9`, `max_tokens=60`, `seed=42`, `context_len=256`.
- **3-Model Evaluation**: Model A vs Model F2 vs Model G on Holdout V3.
- **Metrics**: LM Loss/PPL, Generation metrics, Failure analysis (`failure_analysis.json`), 0-5 Multi-Turn scores, Blind Human Preference (`human_evaluation.json`), Generalization Scores (`generalization_score.json`), Context Ablation (`context_ablation.json`), Inference Benchmarks (`inference_benchmark.json`), Unit Tests (`python -m unittest discover tests`).

---

## 7. Promotion Gate Criteria

- Candidate marked `PROMOTED` only if `G >= F2 + 3.0` points and `G >= Model A`, 0 leakage, unit tests pass (`>= 31 PASS`), and production integrity is preserved. Otherwise marked `CANDIDATE_ON_HOLD`.
