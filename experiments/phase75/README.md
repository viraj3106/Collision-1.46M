# PHASE 75 — CONVERSATIONAL CAPABILITY & FAILURE ANALYSIS

## 1. RESEARCH QUESTION

> **Did J74 actually learn conversation, or did it primarily learn conversational patterns?**

Phase 75 systematically evaluates the `J74` conversational fine-tune against the baseline `COLLISION-10M` model prior to any further training iteration or parameter scaling.

---

## 2. HYPOTHESIS

> **Hypothesis:** Structured conversational fine-tuning improves multi-turn conversational behavior, but improvements may be limited by the underlying model capacity and training-data distribution.

---

## 3. EVALUATION METHODOLOGY & GOLD DATASET

### Frozen Gold Evaluation Set (`gold_eval_set.jsonl`)
* **Size**: 140 versioned, static evaluation cases (10 records per category).
* **Location**: `experiments/phase75/data/gold/gold_eval_set.jsonl`
* **Categories (14)**:
  1. Multi-turn context retention
  2. Follow-up questions
  3. Instruction following
  4. Clarification
  5. Knowledge responses
  6. Basic reasoning
  7. Conversation continuation
  8. Tone consistency
  9. Short-response control
  10. Long-response control
  11. Context switching
  12. Ambiguous requests
  13. Contradictory context
  14. Repetition resistance

---

## 4. GENERATION CONTROL & REPRODUCIBILITY

* **Seed**: 42
* **Temperature**: 0.7
* **Top-K**: 40
* **Top-P**: 0.9
* **Max New Tokens**: 60–120
* **Tokenizer**: BPE (Vocabulary Size > 260)
* **Base Checkpoint**: `models/collision-10m/model.pt` (SHA256: `d256d46d...3775b97`)

---

## 5. SCORING RUBRIC & FAILURE TAXONOMY

### 0–4 Qualitative Scoring Rubric
- **Coherence** (0 = Incoherent babble, 4 = Fluent phrasing)
- **Relevance** (0 = Unrelated response, 4 = Directly addresses prompt topic)
- **Instruction Following** (0 = Ignores constraints, 4 = Fully satisfies constraints)
- **Context Retention** (0 = Forgets previous turns, 4 = Retains multi-turn state)
- **Repetition Resistance** (0 = Infinite loop, 4 = Clean, non-repetitive response)
- **Overall Response Quality** (0 = Unusable, 4 = High quality)

### 10-Class Failure Taxonomy
`CONTEXT_LOSS`, `INSTRUCTION_FAILURE`, `REPETITION`, `FRAGMENTATION`, `HALLUCINATION`, `IRRELEVANCE`, `PREMATURE_TERMINATION`, `OVER_GENERATION`, `TONE_DRIFT`, `CONTRADICTION`.

---

## 6. REPRODUCIBILITY PROCEDURE

To execute Phase 75 tests:
```bash
python -m pytest tests/test_phase75.py
```

To run full model evaluation and report generation:
```bash
python experiments/phase75/reports/generate_report.py
```

To run full repository test suite:
```bash
python -m pytest
```

---

## 7. SCIENTIFIC FINDINGS & CONCLUSION

1. **Surface Formatting vs Base Representation**: `J74` demonstrates clear improvements in dialogue turn formatting and conversational prefix handling over `COLLISION-10M`.
2. **Capacity Boundary**: Foundational reasoning and long-context entity tracking remain bounded by the 10M parameter base architecture.
3. **Verdict**: The evidence supports the hypothesis: structured conversational fine-tuning adapts output formatting to conversational paradigms, but foundational reasoning is fundamentally bounded by the 10M parameter base representation space.
