# Phase 88 — RAG Evaluation Pipeline Trace & Root Cause Analysis

## Executive Summary
During Phase 87 forensic investigation of Phase 86 evaluation results, a critical evaluator bug was identified: the evaluator was scoring question strings (or raw input prompts) instead of model-generated text. This document provides a complete end-to-end trace of the data flow, highlighting the exact failure point and architectural repair in Phase 88.

---

## 1. End-to-End Pipeline Data Flow

```
[ Benchmark Question Item ]
           │
           ▼
 [ RAGPipeline.process(req) ]
           │
           ├───────────────────────────────┐
           ▼                               ▼
  (Mode = OFF / Non-Web)            (Mode = ON / Web Search)
           │                               │
   router.should_search -> False   router.should_search -> True
           │                               │
   context_text = ""               Search -> Fetch -> Rank
           │                               │
   prompt_formatted = query        context_text = retrieved passages
           │                               prompt_formatted = System + Context + Query
           ▼                               ▼
     RAGResponse                     RAGResponse
       - query                         - query
       - prompt_formatted              - prompt_formatted
       - context_text = ""             - context_text = "..."
       - generated_answer              - generated_answer
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
              [ CollisionInferenceEngine ]
                           │
               generate(prompt_formatted)
                           │
                           ▼
                generated_answer (LLM output)
                           │
                           ▼
               [ Evaluator Invariants ]
                           │
               assert evaluator_input == generated_answer
               assert evaluator_input != query
               assert evaluator_input != prompt_formatted
               assert evaluator_input != context_text
                           │
                           ▼
                  [ Score Benchmark ]
```

---

## 2. Phase 87 Root Cause Analysis

In Phase 86 (`run_phase86.py`, line 138), answer extraction for scoring was implemented as:

```python
# FAULTY CODE IN PHASE 86:
answer_text = resp.context_text if resp.context_text else resp.prompt_formatted
```

### Breakdown of the Failure:
1. **For Non-Web RAG (`MODEL_ONLY` or `AUTO` when search is disabled)**:
   - `resp.context_text` is empty string `""` (evaluates to `False`).
   - `resp.prompt_formatted` defaults to the raw user input query string `request.query`.
   - Result: `answer_text` evaluates to `request.query` (the input question itself!).
2. **Evaluator Scoring Mechanism**:
   - The evaluator checked keyword/gold-fact overlaps against `answer_text`.
   - Exactly **63 out of 240 benchmark questions** happened to contain expected answer keywords or gold facts within the question phrasing itself.
   - 63 / 240 = **26.25%**.
3. **Artifact of Bug**:
   - Every non-web evaluation mode (`MODEL_ONLY`, `AUTO_RAG_PHASE84`, `AUTO_RAG_PHASE85` on static questions) reported an artificial accuracy ceiling of **26.25%**, which was completely independent of the model's actual generation capability.

---

## 3. Potential Field Confusion Points & Structural Audit

| Pipeline Stage | Field | Risk of Confusion | Phase 88 Defense / Invariant |
|---|---|---|---|
| Query Input | `request.query` | High (used as prompt in non-web mode) | Invariant: `generated_answer != query` |
| Context Builder | `resp.context_text` | High (used in Phase 86 fallback) | Invariant: `generated_answer != context_text` |
| Prompt Formatter | `resp.prompt_formatted` | Critical (fallback target in Phase 86) | Invariant: `generated_answer != prompt_formatted` |
| LLM Generator | `resp.generated_answer` | Medium (may be empty or fail) | Explicit field in schema; Invariant: `assert generated_answer is not None and len(generated_answer) > 0` |
| Evaluator Interface | `evaluator_input` | Critical (was receiving fallback string) | Assertion: `assert evaluator_input == generated_answer` |

---

## 4. Phase 88 Safe Answer Extraction Contract

Phase 88 replaces the heuristic fallback with a strict, typed contract:
1. `RAGResponse` explicitly includes `generated_answer: Optional[str]`.
2. Model generation is invoked via `CollisionInferenceEngine.generate(prompt=resp.prompt_formatted)`.
3. If generation fails or returns empty text, `EVALUATION_STATUS` is set to `GENERATION_FAILURE`. The evaluator NEVER substitutes `prompt_formatted`, `context_text`, or `query`.
4. Automated assertions verify `evaluator_input == generated_answer` prior to scoring.
