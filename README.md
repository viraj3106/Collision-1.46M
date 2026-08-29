# COLLISION Models Series

An experimental suite of small decoder-only Transformer language models (1.46M, 3.38M, and 10.28M parameters) trained completely from scratch on CPU.

![COLLISION Banner](assets/collision-banner.png)

---

## 1. What is COLLISION?
COLLISION is a research project designed to explore Transformer behavior in extreme low-resource regimes, focusing on:
* Causal language modeling convergence when trained on consumer CPUs with zero pretraining.
* The impact of dataset quality, deduplication, and leakage isolation on small-scale generalization.
* Instruction fine-tuning (SFT) dynamics on tiny architectures.
* Model capacity scaling laws (scaling parameters from 1.46M to 10.28M).

---

## 2. Model Variants & Architectures

| Variant | Parameters | Layers | Embedding Dim (d_model) | Heads | Feedforward Dim (d_ff) | Context Length |
|---|---|---|---|---|---|---|
| **COLLISION-1.46M** | 1,462,464 | 3 | 128 | 4 | 256 | 256 |
| **COLLISION-3.38M** | 3,375,680 | 6 | 192 | 6 | 384 | 256 |
| **COLLISION-10M** | 10,282,304 | 6 | 384 | 8 | 768 | 256 |

*Note: All models use weight-tied embeddings and a context length of 256 tokens.*

---

## 3. Dataset Versions
* **collision_dataset_v4**: Cleaned, deduplicated dataset containing 2,072,993 training tokens.
* **collision_dataset_v5_expanded**: Causal dataset expanded with train/validation/test leakage isolation (1,546,977 train tokens).
* **collision_instruct_v1**: Synthetic conversational instruction dataset (20,480 unique examples format: `<|user|>` / `<|assistant|>`).

---

## 4. Phase-by-Phase Experiment Results

### Pretraining Results (V5 Dataset)

| Metric | 3.38M Base Model (Phase 12B) | 10.28M Base Model (Phase 14) | Impact of Scaling |
|---|---|---|---|
| **Test Loss** | 1.3213 | **0.7679** | -42.0% loss reduction |
| **Test Perplexity** | 3.75 | **2.16** | -42.4% perplexity reduction |
| **Avg Repetition Rate** | 47.9% | **26.8%** | Repetition rate nearly halved |
| **Sentence Termination**| 55.6% | **88.9%** | Termination accuracy +60% |
| **Unique Token Ratio** | 52.1% | **73.2%** | Lexical variety +40% |
| **CPU Training Speed** | ~1500 tokens/sec | ~837 tokens/sec | -44% compute throughput |

### Instruction Tuning Results (Phase 13 SFT)
* **Model**: `COLLISION-Instruct-3.37M` (derived from 3.38M base).
* **Evaluation Perplexity**: **9.70** (Test Loss: 2.2720).
* **Key Finding**: SFT tuning on the 3.38M model caused a generation quality regression (repetition rose to 57.6%, sentence termination rate dropped to 6.7%) due to severe structural overfitting on small parameters.

---

## 5. COLLISION LAB Playground
COLLISION LAB is an interactive web playground built with Streamlit for CPU-first inference, checkpoint switching, generation parameter adjustment, and real-time generation diagnostics.

Launch the playground:
```bash
streamlit run dashboard/app.py
```

---

## 6. Standalone CLI Inference
Run causal sequence completion on any checkpoint:
```bash
python release_inference.py --prompt "Artificial intelligence is" --checkpoint checkpoints/phase14/collision-10m-best.pt
```

For interactive conversation with the instruction-following model:
```bash
python -m collision.chat
```

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
