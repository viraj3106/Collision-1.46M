# COLLISION-10M Release Manifest

The following files are required to run local or API inference on `COLLISION-10M` v1.0.0. All files listed below have been verified to exist in the repository.

## Model Weights & Configuration

- [model.pt](file:///v:/collision%20-%201M/models/collision-10m/model.pt) — Frozen PyTorch model checkpoint containing state dict parameters.
- [config.json](file:///v:/collision%20-%201M/models/collision-10m/config.json) — Architecture hyperparameters configuration.
- [generation_config.json](file:///v:/collision%20-%201M/models/collision-10m/generation_config.json) — Default generation sampling thresholds (temp, top_k, top_p, max_tokens).
- [tokenizer.json](file:///v:/collision%20-%201M/models/collision-10m/tokenizer.json) — Model vocabulary config reference.
- [MODEL_CARD.md](file:///v:/collision%20-%201M/models/collision-10m/MODEL_CARD.md) — Model description, metrics, training facts, and intended use.

## BPE Tokenizer Assets

- [tokenizer/config.json](file:///v:/collision%20-%201M/models/collision-10m/tokenizer/config.json) — Tokenizer configuration (vocab size, special tokens mapping).
- [tokenizer/vocab.json](file:///v:/collision%20-%201M/models/collision-10m/tokenizer/vocab.json) — BPE token vocabulary mapping.
- [tokenizer/merges.json](file:///v:/collision%20-%201M/models/collision-10m/tokenizer/merges.json) — BPE token merge sequences.
- [tokenizer/stats.json](file:///v:/collision%20-%201M/models/collision-10m/tokenizer/stats.json) — Tokenizer training metrics.
