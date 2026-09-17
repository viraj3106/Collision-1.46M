# Counterfactual Dataset Redesign Specification (`collision_dataset_v9_redesigned`)

## 1. Design Principles
The redesigned dataset replaces 100% synthetic combinatorial template data with a high-diversity, natural-language, instruction-oriented pretraining corpus.

### Key Objectives:
- Eliminate synthetic sentence template generators (0.0% template generator ratio).
- Introduce genuine instruction-response and multi-turn conversational patterns.
- Incorporate context-conditioned RAG triples (`[Context, Question, Answer]`).
- Ensure high lexical, syntactic, and structural diversity.

---

## 2. Proposed Dataset Mixture

| Content Category | Target Percentage | Target Tokens (10M Budget) | Primary Sources & Formats |
|---|---:|---:|---|
| **Natural Factual & Technical Text** | 40.0% | 4,000,000 | OpenWebText, Wikipedia, ArXiv abstracts, technical documentation |
| **Instruction & Q&A Response Pairs** | 30.0% | 3,000,000 | OpenOrca, Alpaca, Dolly, StackExchange technical Q&A |
| **Context-Conditioned RAG Triples** | 15.0% | 1,500,000 | SQuAD 2.0, MS MARCO context-question-answer tuples |
| **Code & Technical Reasoning** | 10.0% | 1,000,000 | Python, JavaScript, SQL snippets, GSM8K math reasoning steps |
| **Short Factual Q&A** | 5.0% | 500,000 | Concise entity Q&A (tickers, capitals, versions, definitions) |
| **Total** | **100.0%** | **10,000,000** | **High-diversity multi-source corpus** |

---

## 3. Data Processing & Quality Pipeline

1. **Deduplication**: MinHash LSH deduplication at 0.8 Jaccard similarity threshold to remove repeated paragraphs.
2. **Template Detection & Filtering**: Reject any text matching formulaic synthetic n-gram templates (e.g. `X is critical because it functions to...`).
3. **Length & Quality Filters**: Filter out documents under 20 tokens or over 512 tokens; enforce minimum character-per-token ratio (> 1.2).
4. **Contamination Safeguard**: Decontaminate training data against all 240 questions in the Phase 88 benchmark.
