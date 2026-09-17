# PHASE 73 — CONTROLLED SFT OBJECTIVE REPAIR

This directory contains the Phase 73 SFT objective repair experiment.

## Files:
- `phase73_config.yaml`: Configuration settings for training and evaluation.
- `train_phase73.py`: Training module with response-only loss masking and KL divergence regularization.
- `evaluate_phase73.py`: Evaluation module for scoring baseline and candidate checkpoints.
- `run_phase73.py`: Main execution runner script.
- `reports/phase73_report.md`: Markdown summary report.
- `reports/phase73_metrics.json`: Machine-readable metrics for all candidates.
- `reports/phase73_training.json`: Recorded training loss curves.
- `reports/phase73_generations.json`: Raw generations across evaluation categories.
