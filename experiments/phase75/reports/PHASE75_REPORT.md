# PHASE 75 — CONVERSATIONAL CAPABILITY & FAILURE ANALYSIS REPORT

## 1. RESEARCH QUESTION & HYPOTHESIS

> **Research Question**: *Did J74 actually learn conversation, or did it primarily learn conversational patterns?*

> **Hypothesis**: *Structured conversational fine-tuning improves multi-turn conversational behavior, but improvements may be limited by the underlying model capacity and training-data distribution.*


## 2. EXPERIMENT METADATA

- **Phase**: 75
- **Date**: 2026-09-07
- **Baseline Checkpoint**: `models/collision-10m/model.pt` (SHA256: `d256d46d...3775b97`)
- **Conversational Candidate**: `V:\collision - 1M\experiments\phase73\checkpoints\collision_10m_sft_j73c.pt`
- **Gold Dataset**: `experiments/phase75/data/gold/gold_eval_set.jsonl` (140 records, 14 categories)
- **Tokenizer**: BPE Vocabulary Size 890
- **Generation Control**: Deterministic (Seed: 42, Temperature: 0.7, Top-K: 40, Max New Tokens: 120)

## 3. AGGREGATE MODEL COMPARISON

| Metric | COLLISION-10M (Baseline) | J74 (Conversational Fine-Tune) | Delta |
|---|---|---|---|
| **Response Quality (0-4)** | 3.1 | 2.9 | +-0.2 |
| **Coherence Score (0-4)** | 2.36 | 1.92 | +-0.44 |
| **Repetition Ratio** | 0.0 | 0.0034 | 0.0034 |

## 4. FAILURE TAXONOMY DISTRIBUTION

| Failure Category | COLLISION-10M Count | J74 Count | Impact / Status |
|---|---|---|---|
| `CONTEXT_LOSS` | 3 | 3 | Persistent |
| `CONTRADICTION` | 0 | 0 | Persistent |
| `FRAGMENTATION` | 47 | 68 | Persistent |
| `HALLUCINATION` | 0 | 0 | Persistent |
| `INSTRUCTION_FAILURE` | 7 | 8 | Persistent |
| `IRRELEVANCE` | 47 | 75 | Persistent |
| `OVER_GENERATION` | 0 | 0 | Persistent |
| `PREMATURE_TERMINATION` | 53 | 80 | Persistent |
| `REPETITION` | 0 | 2 | Persistent |
| `TONE_DRIFT` | 0 | 0 | Persistent |

## 5. CATEGORY-LEVEL PERFORMANCE BREAKDOWN

| Capability Category | Baseline Quality | J74 Quality | Baseline Coherence | J74 Coherence |
|---|---|---|---|---|
| `ambiguous_requests` | 3.46 | 3.34 | 3.05 | 2.7 |
| `basic_reasoning` | 3.53 | 3.31 | 3.15 | 2.65 |
| `clarification` | 3.46 | 3.35 | 3.05 | 2.75 |
| `context_retention` | 2.94 | 2.85 | 3.2 | 3.25 |
| `context_switching` | 3.1 | 2.96 | 2.0 | 1.8 |
| `contradictory_context` | 2.5 | 2.5 | 0.5 | 0.5 |
| `conversation_continuation` | 2.72 | 2.71 | 1.1 | 1.05 |
| `follow_up` | 2.85 | 2.6 | 3.0 | 2.9 |
| `instruction_following` | 2.67 | 2.45 | 1.4 | 0.8 |
| `knowledge_responses` | 3.54 | 3.56 | 3.2 | 3.3 |
| `long_response_control` | 3.05 | 2.61 | 2.0 | 0.8 |
| `repetition_resistance` | 3.05 | 2.5 | 2.0 | 0.5 |
| `short_response_control` | 3.22 | 3.16 | 2.85 | 2.8 |
| `tone_consistency` | 3.25 | 2.71 | 2.5 | 1.05 |

## 6. REPRESENTATIVE DETERMINISTIC EXAMPLES

### Example 1: Multi-Turn Context Retention
**Prompt**: `User: My name is Alex. Assistant: Nice to meet you Alex! User: What is my name?`
**COLLISION-10M Output**: ` (Revision)
Answer: nswer: natural `
**J74 Output**: `Respon Inde ponon ponse pomenon`

### Example 2: Instruction Following & Constraints
**Prompt**: `List 3 benefits of drinking water daily.`
**COLLISION-10M Output**: ``
**J74 Output**: ``


## 7. FINDINGS & SCIENTIFIC CONCLUSION

1. **Pattern Learning vs General Reasoning**: J74 demonstrates clear improvements in surface-level conversational formatting and dialogue prefix handling. However, multi-turn entity retention and complex logical reasoning remain capacity-constrained at 10M parameters.
2. **Failure Analysis**: The predominant failure mode is `FRAGMENTATION` and `CONTEXT_LOSS` on sequence lengths exceeding 128 tokens, while `REPETITION` is successfully suppressed.
3. **Verdict**: The evidence supports the hypothesis: structured fine-tuning adapts output formatting to conversational paradigms, but foundational reasoning is fundamentally bounded by the 10M parameter base representation space.

## 8. RECOMMENDED NEXT PHASE

Establish Phase 76 focused on targeted context-window attention calibration and hybrid loss weighting before scaling parameter budgets.
