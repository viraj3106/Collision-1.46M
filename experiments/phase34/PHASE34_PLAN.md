# Phase 34 Plan — Real-World Generalization & Adaptive Fine-Tuning Validation

## 1. Mission & Scope

Phase 34 validates whether the improved COLLISION-10M candidate (Model E) can generalize to genuinely unseen real-world user requests.
- **Research Question**: After fixing synthetic template concentration in Phase 33, does COLLISION-10M actually become better at unseen real-world tasks?
- **Constraints**:
  - Do NOT increase model size (must maintain 10,282,304 parameters).
  - Do NOT modify frozen production model (`models/collision-10m/model.pt`).
  - Do NOT automatically promote any checkpoint to production.

---

## 2. Frozen Production Baseline Integrity

- Checkpoint: `models/collision-10m/model.pt`
- Expected Parameters: `10,282,304`
- Expected SHA256: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
- Baseline must be verified before and after the evaluation phase and remain byte-for-byte unchanged.

---

## 3. Evaluation Setup & Candidates

1. **Model A (Production Baseline)**: `models/collision-10m/model.pt`
2. **Model D (Phase 32 Candidate)**: `checkpoints/phase32/collision_10m_production_candidate_v1.pt`
3. **Model E (Phase 33 Candidate)**: `checkpoints/phase33/collision_10m_production_candidate_v2.pt`

### Decoding Parameters (Identical across models)
- `temperature = 0.7`
- `top_k = 40`
- `top_p = 0.9`
- `max_tokens = 60`
- `seed = 42`
- `context_len = 256`

---

## 4. Benchmark & Data Composition

### Real-World Evaluation Suite (`experiments/phase34/real_world_eval_v1.json`)
- **Target Prompt Count**: 200 unseen prompts (simulating real beta users).
- **Categories**: general questions, beginner technical questions, CS questions, AI/ML questions, troubleshooting, explanations, comparisons, summarization, rewriting, planning, reasoning, follow-up questions, ambiguous requests, incomplete requests, conversational prompts, creative-but-safe prompts, everyday knowledge.
- **Task Mix Target**:
  - 20% knowledge
  - 20% explanation
  - 15% instruction following
  - 10% reasoning
  - 10% comparison
  - 10% summarization/rewrite
  - 10% conversational/multi-turn
  - 5% open-ended
- **Multi-Turn dialogues**: At least 30 conversations (2-5 turns each).
- **Leakage Target**: 0 exact or near-duplicate leaks against all training datasets and previous evaluation suites.

---

## 5. Metrics & Failure Analysis

- **Quantitative Metrics**: Val Loss, Val PPL, Test Loss, avg/median response length, unique response ratio, repeated trigram/4-gram rates, repetition-loop rate, incomplete-response rate, instruction-following score, relevance, coherence, completeness, multi-turn score.
- **Failure Modes Explicitly Audited**: Repetition, Fragmentation, Template behavior, Hallucination, Instruction failure, Topic drift, Context loss, Over-compression, Over-generation (`failure_analysis.json`).
- **Blind Human Preference Evaluation**: At least 75 prompts evaluated pairwise across Model A vs E, D vs E.
- **Real-World Generalization Score**:
  `Generalization Score = 20% relevance + 20% coherence + 15% completeness + 15% instruction following + 10% diversity + 10% multi-turn quality + 10% failure-mode robustness`.
- **PPL Analysis**: Contrast PPL ranking vs Human preference ranking vs Generalization ranking.

---

## 6. System Verification & Promotion Gate

- **Shadow Beta Simulation**: Log shadow metrics to `shadow_beta_report.json`.
- **Inference Benchmark**: Latency (avg, P50, P95), throughput (tokens/sec) logged to `inference_benchmark.json`.
- **API & Unit Testing**: Endpoints `/health`, `/ready`, `/v1/models`, `/v1/generate` and `python -m unittest discover tests` (Target: >= 31/31 PASS).
- **Promotion Check**: Verify if candidate passes all promotion gate requirements. Save `checkpoints/phase34/collision_10m_production_candidate_v3.pt` if candidate passes.
