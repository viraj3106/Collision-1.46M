# COLLISION Public Beta Real-World Data Acquisition Protocol — Phase 64

## Overview
This protocol outlines the rules and guidelines for collecting genuine, consented real-world beta interaction data for COLLISION-10M.

---

## 1. Zero Synthetic Data Rule
- **Real-World Only**: Only genuine human interactions submitted through the Public Beta UI or Developer API with explicit consent (`consent = true`) are included in the real-world dataset.
- **Fixture Isolation**: Synthetic fixtures and test mocks are restricted exclusively to automated unit/integration tests and must NEVER be stored in `data/real_world/raw/` or counted toward dataset growth metrics.

---

## 2. Target Category & Diversity Guidelines

Participants are encouraged to submit queries across diverse domains and conversation types:

### Domain Taxonomy
- **Programming**: Code debugging, syntax explanations, function implementations, algorithm design.
- **AI / Machine Learning**: Concept explanations, loss functions, neural network architectures.
- **Mathematics**: Algebraic calculations, probability, geometry, logical proofs.
- **Science**: Physics principles, chemistry reactions, biology mechanisms.
- **Writing**: Text drafting, summary generation, style rewriting, editing.
- **Reasoning**: Logical deduction, step-by-step problem solving, comparative analysis.
- **Everyday Tasks**: Planning itineraries, scheduling advice, how-to instructions, productivity tips.
- **Summarization / Troubleshooting / Instructions / Conversation**

### Conversation Types
- Factual Q&A
- Explanations & Instructions
- Troubleshooting guides
- Reasoning & Planning
- Summarization & Rewriting
- Multi-turn conversation history & Follow-up questions

---

## 3. Strict Consent & Privacy Requirements

1. **Explicit Consent**: Consent MUST be explicitly set to `true`. No silent defaults or inferred consent are permitted.
2. **PII Redaction**: Never submit PII (emails, phone numbers, IP addresses) or credentials (API keys, secret tokens, passwords).
3. **Audit Trail**: Every rejected submission is logged in `real_world_rejected.jsonl` with exact rejection reasons.
