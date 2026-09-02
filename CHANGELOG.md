# Changelog

All notable changes to the COLLISION project will be documented in this file.

## [1.0.0] - 2026-08-30

This is the first public developer release of the COLLISION-10M model.

### Added
- **COLLISION-10M Flagship Model Weights**: Frozen checkpoint (`models/collision-10m/model.pt`) trained from scratch (random initialization) on CPU with a 10M training-token budget.
- **REST API Completions Service**: Clean, production-oriented completions endpoint built on FastAPI and Uvicorn (`/health`, `/v1/models`, `/v1/generate`).
- **COLLISION LAB Playground**: Decoupled interactive Streamlit application serving as a user interface client.
- **Standalone CLI Inference script**: Direct local inference tool (`release_inference.py`) that executes casual generation on CPU.
- **Model Card & Release Manifest**: Detailed specifications (`MODEL_CARD.md`), evaluation loss/perplexity stats, and manifest (`MANIFEST.md`) of all required files.
- **Reproducibility Package**: Detailed replication instructions (`REPRODUCIBILITY.md`) specifying system environment dependencies, pinned versions, training history, and evaluation checkpoints.
- **Licensing Decision**: Permissive dependency auditing report (`LICENSE_DECISION.md`) and standard MIT License file.
- **Dataset Audit**: Auditing overview (`DATASET_LICENSE_AUDIT.md`) for the synthetic template corpus `collision_dataset_v5_expanded`.
