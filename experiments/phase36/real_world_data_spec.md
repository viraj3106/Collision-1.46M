# Phase 36 Real-World Data Specification

## 1. Executive Strategy

The objective of Collision Dataset V7 is to transition COLLISION-10M from synthetic/template-concentrated data toward high-quality, natural real-world language data.

Dataset Label: `REAL_WORLD_PUBLIC_DATA`

---

## 2. Target Composition & Task Breakdown

Total Token Target: **100,000 – 500,000 tokens**

### Target Task Distribution
- **25% Natural Q&A**: Factual, conceptual, beginner, and technical questions asked in natural user phrasing.
- **20% Instruction Following**: Summarization, rewriting, bullet extraction, classification, format transformation.
- **15% Explanations**: Step-by-step breakdowns, conceptual analogies, intuitive technical explanations.
- **10% Troubleshooting**: Practical debugging, error trace analysis, common pitfalls, step-by-step troubleshooting.
- **10% Conversational Interactions**: Multi-turn dialogues, follow-up questions, clarifications, context retention.
- **10% Reasoning / Problem Solving**: Mathematical, algorithmic, trade-off analysis, logical sequencing.
- **5% Summarization / Rewriting**: Concise text compression, tone conversion, active-voice rephrasing.
- **5% Everyday Knowledge**: Everyday technology support, productivity tips, general advice.

---

## 3. Privacy Filtering Specification

All records passing through the Phase 36 data pipeline must undergo automatic privacy filtering BEFORE training.

### Anonymization Rules
1. **Names & Identifiers**: Replace personal names with generic descriptors (`[USER]`, `[DEVELOPER]`).
2. **Email Addresses**: Anonymize to `user@example.com` or `developer@collision.org`.
3. **Phone Numbers & Addresses**: Strip or mask any numerical phone or physical address strings.
4. **API Keys & Credentials**: Redact strings matching regex patterns for AWS keys, JWT tokens, DB passwords (`[REDACTED_API_KEY]`, `[REDACTED_SECRET]`).
5. **Private URLs**: Mask internal IP addresses and internal staging URLs (`http://10.0.0.1`, `http://staging.internal`).

---

## 4. Quality & Cleaning Pipeline

1. **Length Bounds**: Reject examples with response length < 5 words or > 256 tokens.
2. **Deduplication**: Remove exact instruction and response duplicates.
3. **Template Elimination**: Filter out records exhibiting repetitive synthetic prefix structures (e.g., identical 5-word sentence openings across multiple instructions).
4. **Encoding & Spam Filtering**: Reject corrupted UTF-8 byte sequences, malformed JSON, and spam strings.

---

## 5. Schema Specification (`collision_dataset_v7.jsonl`)

Each record follows the structured COLLISION dataset schema:

```json
{
  "id": "V7_001",
  "source_type": "REAL_WORLD_PUBLIC_DATA",
  "task_type": "natural_qa",
  "category": "technical_qa",
  "instruction": "Why does a database connection pool improve backend performance?",
  "response": "Connection pooling reuses open database TCP connections rather than opening and closing a new socket per query, significantly reducing latency and server overhead.",
  "conversation_id": null
}
```
