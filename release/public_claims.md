# COLLISION Public Claims Audit

This document lists public marketing and technical communication claims that are supported by experimental data, and distinguishes them from unsupported claims.

## SUPPORTED CLAIMS

These claims are verified by project code, configuration files, and training logs:
- **"10.28M-parameter base language model"**: Exactly 10,282,304 parameters.
- **"Trained completely from scratch"**: Initialized randomly without utilizing pretrained weights.
- **"CPU-compatible pretraining and inference"**: Validated training and generation loops run on x86_64 CPU architectures.
- **"REST API Completions Service"**: Exposed completions endpoints using FastAPI.
- **"Interactive Developer Playground"**: Accessible Streamlit interface.
- **"10M training-token pretraining run"**: Exact token budget was 10,000,384 tokens.
- **"Scientifically audited training dataset"**: Complete deduplication and split representativeness checks performed.

## NOT CURRENTLY SUPPORTED CLAIMS

Do **NOT** use these claims in release materials or public communication, as they lack independent empirical evidence:
- **"World's smallest LLM"**: There are multiple research models operating in the range of 1M - 5M parameters.
- **"World's fastest CPU LLM"**: CPU inference speed depends entirely on host hardware, core isolation, and compiler flags; no comparative benchmark exists.
- **"Best small language model"**: No comparative downstream benchmarks (e.g. MMLU-tiny) have been performed.
- **"ChatGPT alternative / ChatGPT equivalent"**: The model is a base model (non-instruction tuned) and has only 10.28M parameters, whereas ChatGPT uses 100B+ parameters.
- **"Production-ready general intelligence / AGI replacement"**: The model is an experimental research release, restricted to scientific concept continuation.
- **"India's first..."**: The project origin and regional priority are undocumented.
