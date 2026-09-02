# COLLISION-10M Release Benchmark Report

This document reports the performance characteristics of `COLLISION-10M` v1.0.0 across training, validation, and local inference regimes.

## 1. Training Metrics

- **Model Parameter Count**: 10,282,304
- **Total Pretraining Tokens**: 10,000,384 tokens
- **Training Device**: Intel/AMD x86_64 Consumer CPU
- **Training Throughput**: ~850 - 900 tokens/second
- **Total Training Steps**: 9,766 steps (Batch size: 4, Gradient Accumulation: 4)

## 2. Evaluation Metrics

These metrics represent model perplexity and loss at the optimal checkpoint (Step 2,500) evaluated on non-overlapping context windows.

- **Best Validation Loss**: `0.7454`
- **Best Validation Perplexity**: `2.11`
- **Test Split Loss**: `0.5805`
- **Test Split Perplexity**: `1.79`

### Generation Quality Metrics
Averaged across 8 standard prompts using default sampling configuration (Temp=0.8, Top_K=50, Top_P=0.9):
- **Repetition Rate (Unigrams)**: `41.1%`
- **Unique Token Ratio**: `58.9%`
- **Repeated Bigram Ratio**: `7.9%`
- **Repeated Trigram Ratio**: `3.0%`
- **Sentence Termination Rate**: `62.5%`

## 3. Local Inference Benchmark

These metrics were recorded sequentially across 10 prompt completion API queries (~97 tokens generated per request) executing on CPU.

- **Average CPU Throughput**: `42.38 tokens/second`
- **Average Latency**: `2,317.6 ms` (per 97 tokens generated)
- **Minimum Latency**: `1,952.8 ms`
- **Maximum Latency**: `2,662.4 ms`
- **Memory Consumption (RAM)**:
  - Average RAM: `476.3 MB`
  - Peak RAM: `614.1 MB`
- **Model Load & Warmup Time**: `0.534 seconds`
