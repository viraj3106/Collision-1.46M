# COLLISION-3M Scaling Experiment Results

## Overview
This document presents the results of the **COLLISION-3M** scaling experiment, comparing it directly against the baseline **COLLISION-1.46M** model. Both models were trained on the identical dataset (`collision_dataset_v4`) under identical hyperparameter conditions on CPU for exactly `1,536,000` training tokens (`1,500` steps).

## Direct Comparison Metrics

| Metric | COLLISION-1.46M (Baseline) | COLLISION-3M (Experiment) | Change (%) |
| :--- | :---: | :---: | :---: |
| **Model Parameters** | 1,462,464 | 3,375,680 | +130.82% |
| **Best Validation Loss** | 1.9363 | 0.9663 | -50.10% |
| **Best Validation Perplexity** | 6.93 | 2.63 | -62.08% |
| **Total Training Time (s)** | 490.5s | 1067.4s | +117.62% |
| **Avg Inference Speed (tok/s)** | 200.83 | 92.12 | -54.13% |

## Observations
- **Loss Improvement**: Validation Loss changed from `1.9363` to `0.9663` (-50.10%).
- **Perplexity Change**: Validation Perplexity changed from `6.93` to `2.63` (-62.08%).
- **Training Efficiency**: Parameter count increased by `130.8%` resulting in a `+117.6%` training time change on CPU.
- **Inference Speed Change**: Generative throughput changed from `200.83` tok/s to `92.12` tok/s (-54.13%).

## Generation Benchmark Outputs

### Prompt: "What is artificial intelligence?"
* **Generated Text**: `What is artificial intelligence? this and intelligence sequence and and maximize necessary intelligence next maximize loss. to and data domain.



Transformers self-attention adapts hidden maintaining `
* **Tokens generated**: `50`
* **Generation time**: `0.6049s`
* **Tokens/second**: `82.66`

### Prompt: "Computer science is"
* **Generated Text**: `Computer science is to to to next of processing intelligence to values. to maximize learning hidden to to to to to pointers the to to and `
* **Tokens generated**: `50`
* **Generation time**: `0.5222s`
* **Tokens/second**: `95.74`

### Prompt: "The future of technology"
* **Generated Text**: `The future of technology and and and and and comus methods.
Key and partics partilasures the s and to toclaccape`
* **Tokens generated**: `50`
* **Generation time**: `0.5055s`
* **Tokens/second**: `98.91`

### Prompt: "An algorithm is"
* **Generated Text**: `An algorithm is and parently algorithm the methodologies. Ke improvellim the enefits arobjectletwe lude efin `
* **Tokens generated**: `50`
* **Generation time**: `0.6962s`
* **Tokens/second**: `71.82`

### Prompt: "Space exploration"
* **Generated Text**: `Space exploration Is memory cby and from memory the data of maximize software. data in and data data in in next To its understand `
* **Tokens generated**: `50`
* **Generation time**: `0.4486s`
* **Tokens/second**: `111.45`

## Reproducibility Variables
* **Git Commit**: `b611024fd25750fa3a46a21d4c02d73d44f2c602`
* **Python Version**: `3.13.14`
* **PyTorch Version**: `2.13.0+cpu`
* **CPU Info**: `AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD`
* **Configuration Hash**: `4bb90f5c77bf3d79d9e0ef963570edb814acea112ee48ace5b91e573ae8a1143`
* **Dataset Version**: `collision_dataset_v4`
* **Tokenizer Version**: `1.0-BPETokenizer`
* **Seed**: `1337`
* **Start Time**: `2026-08-28 21:03:47`
* **End Time**: `2026-08-28 21:21:34`
