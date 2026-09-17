# PHASE 74 — CONVERSATIONAL FOUNDATION EXPERIMENT SPECIFICATION

## 1. RESEARCH QUESTION

Can a ~10M-parameter causal model (`COLLISION-10M`) adapt into a coherent, multi-turn conversational assistant using response-only loss masking and base-model KL divergence regularization, without suffering catastrophic forgetting or token fragmentation?

---

## 2. HYPOTHESIS

1. Fine-tuning on a high-quality, balanced multi-turn conversational dataset (`collision_conversation_v1`) will improve multi-turn context retention and natural conversational flow over Phase 73 (`J73-C`).
2. Response-only loss masking applied across multi-turn user/assistant exchanges will prevent format/prefix template memorization.
3. Base-model KL regularization ($\lambda_{\text{kl}} = 0.05$) against the frozen `COLLISION-10M` base will preserve baseline perplexity and general knowledge representation.

---

## 3. EXPERIMENTAL DESIGN

* **Base Model**: `models/collision-10m/model.pt` (Frozen reference, SHA256: `d256d46d...3775b97`)
* **Reference Candidates**: `J71` (`phase71`), `J73-C` (`phase73`), and Phase 74 Candidate (`J74`).
* **Dataset**: `collision_conversation_v1` (1,400 records total: 1,200 Train / 200 Val).
* **Training Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{response\_ce}} + \lambda_{\text{kl}} \cdot D_{\text{KL}}(P_{\text{base}} \parallel P_{\text{candidate}})$$
  * Prompt & User turn tokens: `loss_mask = 0`
  * Assistant turn tokens: `loss_mask = 1`
  * Padding: `loss_mask = 0`

---

## 4. DATASET SPECIFICATION (`collision_conversation_v1`)

* **Composition**:
  * Casual conversation (8%)
  * General QA & explanations (16%)
  * Technical conversations & reasoning (13%)
  * Follow-up questions & context-dependent queries (16%)
  * Multi-turn dialogue (10%)
  * Clarifications, topic changes, & corrections (14%)
  * Short & detailed answer controls (10%)
  * Friendly acknowledgements & "I don't know" scenarios (8%)
* **Multi-Turn Serialization**:
  ```text
  User: <prompt_turn_1>
  Assistant: <response_turn_1>
  User: <prompt_turn_2>
  Assistant: <response_turn_2>
  ```

---

## 5. FIXED CONVERSATIONAL BENCHMARK (14 CATEGORIES)

Evaluates 14 fixed categories across 140 multi-turn conversation prompts:
1. Casual Conversation
2. Context Retention
3. Follow-Up Understanding
4. Explanation
5. Instruction Following
6. Technical Conversation
7. Topic Switching
8. Clarification
9. Ambiguity
10. Short-Response Control
11. Long-Response Control
12. Consistency
13. Repetition Resistance
14. Conversational Naturalness

---

## 6. PRIMARY SUCCESS CRITERIA

Comparing Phase 74 Candidate (`J74`) against `J73-C`:
* **Overall Capability**: $\ge 0.4149$ (Baseline floor)
* **Coherence**: $\ge 0.3202$ (Baseline floor)
* **Instruction Following & Multi-turn Retention**: $> 0.4500$
* **Fragmentation Rate**: $\le 0.0100$
* **Unigram Repetition**: $\le 0.3141$

---

## 7. REPRODUCIBILITY REQUIREMENTS

* Machine-readable metadata saved to `experiments/phase74/reports/metrics/phase74_metrics.json`.
* Complete unedited raw generations saved to `experiments/phase74/reports/generations/phase74_generations.json`.
* Training loss curves logged in `experiments/phase74/reports/metrics/phase74_training.json`.
* Immutability of reference checkpoints verified before training.
