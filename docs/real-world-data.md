# Real-World Data & Privacy Documentation — Phase 27

## 1. Overview & Architecture

With 15 days remaining until final project freeze, Phase 27 establishes a simplified, production-grade real-world data collection and processing pipeline for COLLISION.

```
React/Vite Website (Playground / Landing)
       │
       ├──> Firebase Authentication + Firestore (Cloud User Session & Records)
       │
       └──> FastAPI API (Local REST & Model Inference Engine)
               │
               ├──> Database (SQLite / Postgres Local Storage & Feedback Table)
               └──> Data Pipeline (data/clean_real_world.py)
                       │
                       ├──> raw/ (Raw immutable feedback logs)
                       ├──> cleaned/ (Cleaned, consent-verified dataset)
                       └──> rejected/ (Rejected / unconsented / invalid entries)
                               │
                               └──> training/prepare_real_world_dataset.py
                                       │
                                       └──> Google Colab / Temporary Compute (COLLISION-11M)
```

---

## 2. Privacy Boundaries & Consent Mechanisms

1. **Consent-First Collection**: Feedback interactions recorded via the COLLISION Playground include an explicit `consent` flag (`true`/`false`). Only consented entries (`consent == true`) are extracted by the data cleaning pipeline.
2. **Credential Exclusion**: The dataset pipeline enforces strict filtering against sensitive tokens (`api_key`, `bearer`, `password=`, `secret_key`, `col_`). Raw passwords, session tokens, and API key secrets are NEVER collected or stored in feedback collections.
3. **Data Segregation**: Original raw feedback submissions stored in `data/real_world/raw/` are NEVER overwritten. Cleaned training datasets and rejected data logs remain in separate isolated directories (`cleaned/` and `rejected/`).

---

## 3. Dataset Format

### Raw Record Schema
```json
{
  "user_id": "1",
  "prompt": "What is machine learning?",
  "model": "collision-10m",
  "response": "Machine learning is a field of computer science...",
  "rating": "thumbs_up",
  "feedback": "Great response!",
  "category": "general",
  "consent": true,
  "timestamp": "2026-09-01T20:45:00Z"
}
```

### Formatted Training Dataset Schema (`datasets/collision_instruct_v1/real_world_formatted.jsonl`)
```json
{
  "instruction": "What is machine learning?",
  "response": "Machine learning is a field of computer science...",
  "category": "general",
  "source": "user_feedback"
}
```

---

## 4. Data Quality Pipeline & Preprocessing

The preprocessing pipeline (`data/clean_real_world.py`) performs automated validation across all records:
- **Empty Record Removal**: Filters out items missing `prompt`, `response`, or `rating`.
- **Deduplication**: Eliminates duplicate `(prompt, response)` pairs.
- **Consent & Rating Validation**: Ensures `consent == true` and rating is a valid positive rating (`thumbs_up`, `up`, `+1`).
- **Sensitive String Scanning**: Scans for credential leaks.
- **Rejection Log**: Records rejected examples alongside explicit rejection reasons in `data/real_world/rejected/real_world_rejected.jsonl`.

---

## 5. Training Workflow & Safety Safeguards

- **Frozen Weights Preservation**: `COLLISION-10M` model weights (`10,282,304` parameters) remain frozen and untouched.
- **Colab Compute Delegation**: Retraining and experiment execution are performed on temporary ML compute (Google Colab / GPUs) using `training/COLAB_TRAINING.md`. No RETRAINING or heavy ML compute is run on constrained AWS instances.
