# COLLISION-10M Production Model Release

This directory contains the production-frozen parameters and assets for the COLLISION-10M model.

## Contents
* `model.pt`: Frozen model checkpoint weights.
* `config.json`: Architecture configuration parameters.
* `tokenizer.json`: BPETokenizer configuration map.
* `tokenizer/`: Tokenizer vocabulary and merge tables.
* `generation_config.json`: Default generation decoding parameters.
* `MODEL_CARD.md`: Details of model metadata, parameters, validation, and SHA256 hashes.

## Verification
* **SHA256 Hash**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
* **Size**: `125,057,611` bytes
* **Architecture**: 6 layers, 384 d_model, 8 heads, 768 d_ff
* **Parameters**: 10,282,304
* **Generalization**: Test Perplexity `1.79`
