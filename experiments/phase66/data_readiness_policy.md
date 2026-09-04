# Phase 66 — Data Readiness Policy & Quality Gate Architecture

## 1. PURPOSE

This document defines the strict, non-degradable quality gates required for any real-world dataset collected via the COLLISION Public Beta before it can be considered eligible for model training review.

Software pipeline functionality **never** implies dataset readiness. The policy defined herein shall **never** be weakened or lowered to accommodate slow user data acquisition.

---

## 2. STRICT READINESS GATES

### Gate A — Minimum Quantity Threshold
- **Requirement**: At least **20 genuine clean records** for the immediate milestone, and **100 clean records** for full SFT dataset readiness.
- **Current Baseline**: 7 clean records.
- **Rule**: If clean records < 20, the gate strictly evaluates to **FAILED** (`PHASE_66_DATA_NOT_READY_EXTERNAL_TRAFFIC_REQUIRED`).

### Gate B — Category & Domain Diversity
- **Requirement**: The dataset must NOT be concentrated in a single category.
- **Rule**: If any single domain represents $\ge 70\%$ of the clean dataset, or if active domains with $\ge 5$ records are fewer than 6, diversity status is **`HIGHLY_CONCENTRATED`**. A `HIGHLY_CONCENTRATED` dataset fails Gate B.

### Gate C — Conversation Type Diversity
- **Requirement**: Multi-type interaction coverage (explanatory, how-to, reasoning, troubleshooting, writing, multi-turn).
- **Rule**: The dataset cannot qualify if factual Q&A accounts for $100\%$ of all interactions.

### Gate D — Multi-Turn & Context Tracking
- **Requirement**: Track parent ID, turn numbers, and follow-up flags.
- **Rule**: Multi-turn records must be explicitly tracked and validated for session continuity.

### Gate E — Zero Privacy / PII Violations
- **Requirement**: **0** emails, phone numbers, IP addresses, or secret API keys (`col_`, `sk-`, `bearer`) in the clean dataset split.
- **Rule**: Any privacy violation immediately quarantines the record into `data/real_world/rejected/`.

### Gate F — Explicit User Consent
- **Requirement**: **100%** of training-eligible records must contain explicit `consent = True`.
- **Rule**: Missing, implicit, or `False` consent results in immediate rejection.

### Gate G — Provenance & Audit Trail
- **Requirement**: All clean records must be traceable to genuine API endpoints (`/v1/feedback`), containing timestamp, model version, rating, category, and feedback ID.
- **Rule**: Synthetic test records (`source = 'synthetic_fixture'`) or untraceable records are strictly forbidden from entering the clean dataset.

### Gate H — Deduplication & Contamination
- **Requirement**: No exact or near-duplicate prompt-response pairs (`duplicate_rate < 0.20`).
- **Rule**: Duplicate records are rejected during pipeline cleaning.

---

## 3. NON-DEGRADATION ENFORCEMENT

Under no circumstances may any gate threshold be adjusted downwards (e.g. lowering the milestone target from 20 to 7 records). If human beta traffic is insufficient, the system verdict must remain:

`PHASE_66_DATA_NOT_READY_EXTERNAL_TRAFFIC_REQUIRED`
