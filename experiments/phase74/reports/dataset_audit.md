# DATASET AUDIT REPORT — collision_conversation_v1

## EXECUTIVE SUMMARY

Automated audit completed for `collision_conversation_v1` dataset across 1,400 multi-turn conversation records.

---

## 1. SUMMARY METRICS

* **Total Conversations**: `1400` (`1200` Train / `200` Validation)
* **Train / Val Overlap**: `0` (Zero leakage)
* **Total Turn Pairs**: `2380`
* **Average Turns / Conversation**: `1.7`
* **Turn Depth Range**: `1` min / `3` max
* **Average User Turn Length**: `49.59` chars
* **Average Assistant Turn Length**: `80.35` chars
* **Context-Dependent Conversation Rate**: `1.0`

---

## 2. TOKENIZER & INTEGRITY AUDIT

* **Unknown Token Rate**: `0.0`
* **Token Fragmentation Ratio**: `0.6472`
* **Average Tokens / Conversation**: `160.48`
* **Duplicate Prompt Rate**: `0.0`

---

## 3. SHA256 MANIFEST

* `train.jsonl`: `487b8ef657f3dd853c63ae6cffd225996d5fdbb5dbec2aeff5b4396eb5a786f8`
* `validation.jsonl`: `1bcea71369645b3640ae78ab8229e787bb08e81fb79541e734b0a754d8ba3533`
* `dataset_spec.json`: `7624bcf47bc6a315cbcc0f34c20ef292664edba302c09cd6c017bed05bfa92f5`
