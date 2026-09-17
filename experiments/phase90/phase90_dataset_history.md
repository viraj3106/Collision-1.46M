# COLLISION Dataset Version History

| Dataset Version | Creation Date | Token Count | Documents | Synthetic Ratio | Key Characteristics & Weaknesses | Model Trained | Observed Performance |
|---|---|---:|---:|---:|---|---|---|
| `collision_dataset_v1` | 2026-08-20 | ~250K | ~1,200 | 0.0% | Initial raw scraped technical text. Unbalanced domain distribution, noisy formatting. | COLLISION-1M | Perplexity: 14.2 |
| `collision_dataset_v2` | 2026-08-23 | ~500K | ~2,500 | 15.0% | Cleaned domain text + initial synthetic completion stems. | COLLISION-3M | Perplexity: 8.5 |
| `collision_dataset_v3` | 2026-08-26 | ~1.0M | ~5,000 | 35.0% | Preference-audited dataset with declarative paragraphs. Introduced fixed Q&A prefixes. | COLLISION-5M | Perplexity: 4.1 |
| `collision_dataset_v4` | 2026-08-28 | 2.30M | ~7,516 | 10.0% | Natural paragraph corpus across 5 subjects (AI, Astronomy, CS, Philosophy, Physics). | COLLISION-7M | Perplexity: 2.45 |
| `collision_dataset_v5` | 2026-08-30 | 87.2K | 1,000 | 100.0% | Synthetic controlled mixture prototype (40% Declarative, 25% Explanatory, 20% Q&A, 15% Completion). | Pilot Run | Perplexity: 2.15 |
| `collision_dataset_v5_expanded` | 2026-08-31 | 1.80M | 15,649 | 100.0% | Combinatorial synthetic generator (`build_v5_expanded.py`) with 5 fixed sentence templates and synthetic suffix tags (`(Revision)`, `(Overview)`). | **COLLISION-10M** | **Test PPL: 1.79 \| RAG Benchmark Accuracy: 0.83%** |

### Evolution Analysis
Synthetic template concentration escalated from **0% in v1** to **100% in v5_expanded**. While synthetic data dramatically artificially lowered validation loss and perplexity (1.79 PPL), it severely destroyed generative instruction-following, context-conditioning, and factual Q&A capability.
