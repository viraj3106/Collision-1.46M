# Phase 65 — Real-World Public Beta Data Acquisition Campaign Protocol

## Overview
Phase 65 focuses on transitioning the COLLISION real-world dataset from **7 clean records to 20+ clean records** through targeted organic human interactions across the Public Beta Playground UI and Developer API.

---

## 1. Zero Synthetic Data Rule
- **Strict Real-World Provenance**: Only genuine human interactions submitted through `/v1/feedback` with explicit consent (`consent = true`) are counted in the real-world dataset.
- **Fixture Isolation**: Synthetic fixtures and unit test mocks must remain strictly isolated in test files (`tests/test_*.py`) and NEVER injected into `data/real_world/raw/`.

---

## 2. Target Category Acquisition Scorecard (Milestone 20)

| Category / Domain | Desired New Records | Current Count | Target Minimum |
| :--- | :---: | :---: | :---: |
| Programming | 2+ | 0 | 2+ |
| AI / Machine Learning | 2+ | 0 | 2+ |
| Mathematics | 1+ | 0 | 1+ |
| Science | 1+ | 0 | 1+ |
| Writing | 1+ | 0 | 1+ |
| Reasoning | 2+ | 0 | 2+ |
| Troubleshooting | 1+ | 0 | 1+ |
| Everyday Tasks | 1+ | 0 | 1+ |
| General Knowledge / Other | Remaining | 7 | 7+ |
| Multi-turn Conversations | 2+ | 0 | 2+ |
| Follow-up Questions | 2+ | 0 | 2+ |

---

## 3. Privacy & Quality Constraints
1. **Explicit Consent**: `consent` field MUST be explicitly `true`. Non-consented submissions (`consent = false`) are rejected and logged.
2. **Strict Redaction**: Submissions containing PII (emails, phone numbers, IP addresses) or API credentials/passwords (`col_`, `api_key`) are rejected.
3. **Audit Integrity**: Rejection audit reasons are recorded in `data/real_world/rejected/real_world_rejected.jsonl`.
