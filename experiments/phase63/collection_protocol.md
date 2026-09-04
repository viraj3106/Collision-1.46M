# COLLISION Public Beta Real-World Data Acquisition Protocol

## Overview
This protocol establishes guidelines for acquiring genuine, consented, high-quality real-world interaction data through the COLLISION Public Beta playground and developer API.

---

## 1. Domain Coverage Guidelines

Participants are encouraged to test COLLISION-10M across diverse domains using the established taxonomy:

- **Programming**: Code debugging, syntax explanations, function implementations, algorithm design.
- **AI / Machine Learning**: Concept explanations, loss functions, neural network architectures, hyperparameter tuning.
- **Mathematics**: Algebraic calculations, probability, geometry formulas, logic proofs.
- **Science**: Physics principles, chemistry reactions, biology mechanisms, scientific method steps.
- **Writing**: Text drafting, summary generation, style rewriting, editing feedback.
- **Reasoning**: Logical deduction, step-by-step problem solving, comparative analysis.
- **Everyday Tasks**: Planning itineraries, scheduling advice, how-to instructions, productivity tips.

---

## 2. Conversation-Type & Multi-Turn Guidelines

To ensure conversational depth, feedback events should capture varied conversation types:

- Factual Q&A
- Explanatory responses
- Step-by-step How-To instructions
- Troubleshooting guides
- Logical reasoning
- Summarization & Rewriting
- Multi-turn follow-up interactions (associating `parent_id` for multi-turn history)

---

## 3. Privacy & Consent Rules

1. **Explicit Consent Required**: Users must explicitly opt-in (`consent = true`) before any prompt or response is logged for dataset inclusion.
2. **Strict Privacy Redaction**: Never submit PII (emails, phone numbers, IP addresses) or credentials (API keys, secret tokens, passwords).
3. **No Synthetic Fabrication**: Synthetic test data must remain strictly isolated in test fixtures and never injected into raw real-world feedback batches.
