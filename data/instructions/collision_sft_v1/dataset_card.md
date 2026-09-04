# Dataset Card: collision_sft_v1

## Overview
`collision_sft_v1` is a high-entropy, multi-domain Supervised Fine-Tuning (SFT) dataset designed specifically for training COLLISION-10M.

* **Total Records**: 4,995
* **Train Split**: 4,500 (90%)
* **Validation Split**: 495 (10%)
* **Unique Prompt Ratio**: 100% (0% exact duplicates)
* **Domains**: 15 balanced domains (333 records per domain)
* **Response Length Buckets**: 33.33% SHORT, 33.33% MEDIUM, 33.33% LONG
* **Context Limit**: 256 tokens total (max total tokens: 240)

## Quality Safeguards
* **Zero PII or Secrets**: Verified clean.
* **No Length Bias**: Eliminates "longer = better" assumption by balancing short concise answers alongside multi-step explanations.
