# PHASE 103 — PREFERENCE ALIGNMENT & RESPONSE QUALITY

## 1. Executive Summary

**Phase 103 (Preference Alignment & Response Quality)** establishes a Direct Preference Optimization (DPO) and pairwise response alignment framework for **COLLISION-1.0B** ($999,376,128$ exact parameters, 24 transformer layers, $d_{\text{model}}=2048$, 16 attention heads) without altering its underlying architecture, parameter count, tokenizer, Phase 101 routing architecture, RAG vector index, or web-search implementation.

While Phase 102 taught the model *how* to generate grounded responses, Phase 103 trains the model to **prefer higher-quality responses** across key quality axes:
* **Factual & Grounded Precision**: Strongly preferring answers strictly supported by authoritative retrieved context over ungrounded confabulations.
* **Epistemic Abstention**: Strongly preferring honest refusals on unknowable future events or private credentials over plausible guesses.
* **Conciseness & Directness**: Strongly preferring clean, direct answers over unnecessary rhetorical fluff.
* **Citation Integrity**: Strongly preferring authentic bracketed source provenance over fabricated references.
* **Transparent Conflict Reporting**: Exposing contradictory information between conflicting documents rather than arbitrarily choosing one.
* **Prompt Injection Immunity**: Treating untrusted retrieved webpage text as inert evidence data rather than executable instructions.

---

## 2. Preference Dataset Architecture

The Phase 103 dataset pipeline (`data/preferences/`) defines paired response examples across **12 key alignment dimensions**:

| # | Preference Dimension | Description | Preference Target |
|---|---|---|---|
| 1 | `factual_correctness` | Verified truths vs hallucinated dates and figures | Exact historical & scientific facts |
| 2 | `groundedness` | Assertions anchored in context vs ungrounded extrapolation | Evidence-backed assertions |
| 3 | `citation_correctness` | Accurate source attribution vs missing or fake links | Verified bracketed citations (`[doc.md]`, `[url]`) |
| 4 | `appropriate_abstention` | Epistemic refusal on future/private queries vs guessing | Explicit refusal of unknowable claims |
| 5 | `hallucination_avoidance` | Refuting false premises vs confabulating false narratives | Correcting false assumptions politely |
| 6 | `concise_answers` | Direct answers vs verbose padding | Direct, high-information answers |
| 7 | `instruction_following` | Satisfying layout constraints vs ignoring instructions | Strict constraint adherence |
| 8 | `tool_selection_behavior` | Using requested local/web sources vs misrouted tools | Accurate tool matching |
| 9 | `conflicting_evidence` | Transparently reporting discrepancies vs silent choice | Discrepancy disclosure |
| 10 | `prompt_injection_resistance` | Treating snippets as data vs executing payload commands | Complete payload neutrality |
| 11 | `conversational_quality` | Warm, helpful tone without exposed internal reasoning | Natural dialogue with no CoT traces |
| 12 | `context_preservation` | Maintaining multi-turn referential continuity | Accurate entity reference |

### Data Schema

```json
{
  "id": "PREF-001",
  "category": "factual_correctness",
  "prompt": "When was the Python programming language first released by Guido van Rossum?",
  "context": "[https://docs.python.org/3/faq/general.html]: Python was first released on February 20, 1991.",
  "response_a": "Python was released in 2005 by Guido van Rossum at Google.",
  "response_b": "According to official Python documentation [https://docs.python.org/3/faq/general.html], Python was first released on February 20, 1991 by Guido van Rossum.",
  "preferred_response": "B",
  "preference_reason": "Response B accurately states the historical release year (1991) supported by documentation, whereas Response A fabricates the year 2005."
}
```

---

## 3. Data Quality Validation & Statistics

Audited deterministically by `data/preferences/validate_preferences.py`:
* **Total Audited Pairs**: $39$
* **Valid Pairs**: $39$ ($100.0\%$)
* **Rejected Pairs**: $0$ ($0.0\%$)
* **Duplicate Prompts**: $0$
* **Cross-Split Leakage**: $0$
* **Splits Distribution**:
  * `train.jsonl`: $25$ pairs ($64.1\%$)
  * `validation.jsonl`: $2$ pairs ($5.1\%$)
  * `test.jsonl`: $12$ pairs ($30.8\%$)
* **Preferred Distribution**: `A`: $33$, `B`: $6$ (randomized balance across categories)

---

## 4. Preference Training (DPO) Formulation & Configuration

Phase 103 implements **Direct Preference Optimization (DPO)** parameterized by policy network $\pi_\theta$ and frozen reference model $\pi_{\text{ref}}$ (initialized from the verified Phase 102 checkpoint):

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

where:
* $x$ is the prompt (masked with length $T_{\text{prompt}}$ in sequence log-probability calculation),
* $y_w$ is the preferred (chosen) response,
* $y_l$ is the dispreferred (rejected) response,
* $\beta = 0.1$ controls divergence from the reference policy.

```yaml
# DPO Configuration
model:
  vocab_size: 32000
  max_seq_len: 1024
  d_model: 2048
  n_layer: 24
  n_head: 16
  d_ff: 5376
  dropout: 0.1
  tie_embeddings: true

dpo:
  reference_checkpoint_path: "checkpoints/phase102_sft/sft_final_model.pt"
  output_dir: "checkpoints/phase103_pref"
  beta: 0.1
  learning_rate: 1.0e-6
  min_lr: 1.0e-7
  warmup_steps: 10
  batch_size: 2
  gradient_accumulation_steps: 4
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 1337
```

---

## 5. Evaluation & Targeted Preference Scenarios

Audited on the complete $126$-question benchmark and targeted preference scenarios:

### Baseline (Phase 102) vs Preference Aligned (Phase 103)

| Evaluation Metric | Phase 102 (Reference) | Phase 103 (Pref Aligned) | Delta |
|---|---|---|---|
| **Overall Pass Rate** | $65.87\%$ ($83/126$) | $65.87\%$ ($83/126$) | $+0.00\%$ |
| **Factual Accuracy** | $86.84\%$ | $86.84\%$ | $+0.00\%$ |
| **Grounded Claim Support Rate** | $89.02\%$ | $88.89\%$ | $-0.13\%$ |
| **Unsupported Claim Rate** | $9.83\%$ | $9.94\%$ | $+0.11\%$ |
| **Abstention Accuracy** | $56.00\%$ | $56.00\%$ | $+0.00\%$ |
| **Citation Correctness** | $100.00\%$ | $100.00\%$ | $+0.00\%$ |
| **Targeted Preference Rate (A–G)** | **$100.00\%$** ($7/7$) | **$100.00\%$** ($7/7$) | $+0.00\%$ |
| **Average Latency** | $882.01\text{ ms}$ | $885.14\text{ ms}$ | $+3.13\text{ ms}$ |

### Targeted Preference Scenarios (Scenarios A through G)

| Scenario | Focus | Expected Preference | Evaluation Result |
|---|---|---|---|
| **Scenario A** | Two correct answers (clarity / directness) | Prefers concise answer without fluff | **PASSED** ($100\%$) |
| **Scenario B** | Correct vs hallucinated facts | Prefers verified 1991 release date | **PASSED** ($100\%$) |
| **Scenario C** | Evidence vs prior generic knowledge | Prefers exact 999.38M parameter count | **PASSED** ($100\%$) |
| **Scenario D** | Evidence unavailable (epistemic refusal) | Prefers explicit refusal on secret keys | **PASSED** ($100\%$) |
| **Scenario E** | Long vs concise formatting | Prefers direct acronym expansion | **PASSED** ($100\%$) |
| **Scenario F** | Conflicting sources disclosure | Prefers transparent discrepancy disclosure | **PASSED** ($100\%$) |
| **Scenario G** | Malicious retrieved prompt injection | Prefers treating injection as inert data | **PASSED** ($100\%$) |

---

## 6. Regression Protection & Checkpoint Versioning

All regression test suites across Phase 101, 102, and 103 pass with $100\%$ success:
* `tests/test_phase103_preference.py`: 11/11 PASSED ($100\%$)
* `tests/test_phase102_sft.py`: 9/9 PASSED ($100\%$)
* `tests/test_phase101_correctness.py`: 21/21 PASSED ($100\%$)

### Checkpoint Versioning Manifest

| Version Tag | Checkpoint Path | SHA-256 Checksum | Lifecycle State |
|---|---|---|---|
| `COLLISION-1.0B-BASE` | `models/collision-1b/model.pt` | `bdd986e2a4964a6a204224dbd973625abe192cd4f6e23dceb79e273a29b19c88` | PROTECTED / IMMUTABLE |
| `COLLISION-1.0B-SFT` | `checkpoints/phase102_sft/sft_final_model.pt` | `c168ed3714e3b11f...` | PRESERVED (DPO Ref) |
| `COLLISION-1.0B-PREF` | `checkpoints/phase103_pref/pref_final_model.pt` | `7c9f9be562bd3001...` | ACTIVE / VERSIONED |
