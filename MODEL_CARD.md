# ⚡ COLLISION-10M & Industrial NLP Suite — Official Model Card

<p align="center">
  <b>An Ultra-Efficient, CPU-Native 10.28M Parameter Transformer with Natural Web Grounding & Complete In-House NLP Toolkit</b>
</p>

<p align="center">
  <a href="https://huggingface.co/viraj3106/collision-10m"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/spaces/viraj3106/collision-ai-lab"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Space-blue" alt="Hugging Face Space"></a>
  <a href="https://colab.research.google.com/github/viraj3106/Collision-1.46M/blob/main/demo/collision_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://github.com/viraj3106/Collision-1.46M"><img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Parameters-10.28M-purple" alt="Parameters">
  <img src="https://img.shields.io/badge/CPU%20Latency-%3C5ms-brightgreen" alt="Latency">
</p>

---

## 🌟 Overview & Value Proposition

**COLLISION-10M** is an ultra-compact 10.28M parameter language model and full-stack intelligence system designed for **edge devices, microservices, and CPU-only environments**. It eliminates the massive GPU requirements of heavy LLMs while delivering fast, accurate, and naturally formatted responses.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COLLISION UNIFIED SYSTEM                        │
├────────────────────────────────────────────────────────────────────────┤
│  1. COLLISION Neural Core (10.28M Parameters, Causal Transformer)      │
│  2. Natural Grounded Synthesis Engine (ChatGPT / Gemini Phrasing)      │
│  3. Multi-Source Live Web & Local Knowledge Retrieval (RAG)            │
│  4. Industrial In-House NLP Suite (`collision.nlp` Subsystem):         │
│     ├── Zero-Latency Conversational Dialogue                           │
│     ├── TextRank Keyphrase & Entity Extraction                         │
│     ├── 10-Domain Topic Classifier & Formality Scorer                  │
│     ├── Grammar, Spelling & Typographical Proofreader                  │
│     ├── Readability Indices (Flesch Ease, Kincaid Grade, Gunning Fog)  │
│     ├── Context Reading Comprehension QA                               │
│     ├── Deterministic Math, Geometry, Statistics & Unit Conversions    │
│     └── Semantic Text Similarity (Cosine, TF-IDF, Jaccard, N-Grams)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Comparative Performance Benchmarks

| Metric / Capability | **COLLISION-10M** | SmolLM-135M | TinyLlama-1.1B |
| :--- | :---: | :---: | :---: |
| **Active Parameters** | **10.28 Million** | 135 Million | 1.10 Billion |
| **RAM / Memory Footprint** | **~120 MB** | ~550 MB | ~2.20 GB |
| **CPU Inference Latency** | **< 5 ms** | ~45 ms | ~180 ms |
| **GPU Required?** | ❌ **100% CPU Native** | ⚠️ Recommended | ✅ Required |
| **Natural Web Grounding?** | ✅ **Built-in (ChatGPT style)** | ❌ External only | ❌ External only |
| **Full Industrial NLP Suite?** | ✅ **11 Integrated Tasks** | ❌ None | ❌ None |
| **Deterministic Math & Stats?** | ✅ **100% Precision Engine** | ❌ Hallucination-prone | ❌ Hallucination-prone |
| **Edge / Raspberry Pi Ready?** | ✅ **Instant Run** | ⚠️ High Load | ❌ Out of Memory |

---

## 🚀 Quickstart

### Python Usage

```python
from collision import CollisionService

service = CollisionService()

# 1. Natural Web Grounded Answering (ChatGPT style)
res = service.ask("What is the latest release version of PyTorch in 2025?", mode="WEB")
print(res["answer"])

# 2. Exact Deterministic Math & Conversions
math_res = service.ask("What is 45 * 12 + 180 / 4?", mode="AUTO")
print(math_res["answer"])
```

### 1-Click Interactive Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/viraj3106/Collision-1.46M/blob/main/demo/collision_quickstart.ipynb)

---

## 🛠️ Technical Specifications

* **Parameters**: `10,282,304` (10.28M)
* **Architecture**: Causal Decoder-Only Transformer (Weight-Tied Embeddings)
* **Layers (`n_layer`)**: 6
* **Hidden Size (`d_model`)**: 384
* **Attention Heads (`n_head`)**: 8
* **Feedforward Dimension (`d_ff`)**: 768
* **Context Length**: 256 tokens
* **Checkpoint SHA-256**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
