# PHASE 87 REPORT — COLLISION RAG EVALUATION FORENSIC AUDIT & FAILURE ATTRIBUTION

## Executive Summary
Phase 87 conducted a rigorous, non-destructive forensic audit to investigate why Phase 86 reported `26.25%` accuracy for AUTO_RAG alongside 100% grounding.

## Key Findings
1. **Primary Root Cause: Evaluator Prompt Formatting Bug**:
   In Phase 86 execution (`run_phase86.py`), when web search was not invoked or returned empty context, `answer_text` fell back to `resp.prompt_formatted`, which defaulted to the raw input `query`. The evaluator evaluated the question text itself against gold facts. In 63 out of 240 questions (26.25%), words in the question string accidentally matched the expected answer keywords, creating an artificial 26.25% accuracy across ALL 4 modes (`MODEL_ONLY`, `WEB_RAG`, `AUTO_RAG_PHASE84`, `AUTO_RAG_PHASE85`).
2. **Model & Codebase Integrity**:
   `models/collision-10m/model.pt` SHA256 was verified as `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97` (10,282,304 parameters). `TRAINING EXECUTED = FALSE` and `MODEL WEIGHTS MODIFIED = FALSE`.

## Final Verdict
`PHASE_87_EVALUATION_ISSUES_FOUND`
