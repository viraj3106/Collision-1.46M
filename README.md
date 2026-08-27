# COLLISION-1.46M
### A Tiny Language Model Built From Scratch

![COLLISION-1.46M Banner](assets/collision-banner.png)

COLLISION-1.46M is a small decoder-only Transformer language model built and trained completely from scratch in PyTorch, designed specifically for offline CPU-first execution.

*   🧠 **1.46M Parameters** — Programmatically calculated and tied embedding structure.
*   💻 **CPU Trained** — Lightweight execution optimized for standard consumer laptops.
*   🔤 **Custom BPE Tokenizer** — Byte-Level merges restricted to word/chunk boundaries.
*   📚 **2.4M Training Tokens** — Clean educational corpus covering physics, CS, AI, philosophy, and astronomy.
*   📊 **Training Dashboard** — Live tracking Streamlit lab showing loss charts, metrics, and text generators.
*   💾 **Checkpoint System** — Step checkpoint histories, best loss tracking, and emergency keyboard interrupts saving.
*   🔬 **No Pretrained Weights** — Randomly initialized weights, learning from pure text streams.

---

## 1. How it Works (Flowchart)

```
        Text Input
             │
             ▼
       BPE Tokenizer
             │
             ▼
         Token IDs
             │
             ▼
         Embeddings (Token + Positional)
             │
             ▼
     Transformer Blocks (3 Causal Layers)
             │
             ▼
       Self-Attention (4 Heads + Casual Masking)
             │
             ▼
         MLP Block (GELU Activation)
             │
             ▼
     Next Token Prediction (Softmax probabilities)
             │
             ▼
        Loss (Cross Entropy)
             │
             ▼
     Backpropagation (AdamW Optimizer)
             │
             ▼
      Updated Weights
```

---

## 2. Model Specifications

| Metric | COLLISION |
| :--- | :--- |
| **Parameters** | 1,462,464 |
| **Vocabulary Size** | 8,000 |
| **Context Length** | 256 |
| **Transformer Layers** | 3 |
| **Attention Heads** | 4 |
| **Training Data** | 2.41M tokens |
| **Training Device** | CPU |
| **Throughput Speed** | ~4,018 tokens/sec |

---

## 3. CLI Commands Reference

Perform the complete lifecycle of COLLISION-1M using the commands below:

### Setup Structure & Directories
```bash
python -m collision.setup
```

### Train Byte-Pair Encoding (BPE) Tokenizer
```bash
python -m data.tokenize --train
```

### Clean & Build Raw Dataset
```bash
python -m data.build
```

### Run Model Pre-Training Readiness Check
```bash
python -m collision.readiness_check
```

### CPU Benchmark Profiling
Runs exactly 100 steps to profile steps/sec, tokens/sec, memory usage, and outputs estimations for 1K, 5K, and 10K steps.
```bash
python -m training.train --profile --config configs/collision_1m_cpu.yaml
```

### Train COLLISION-1.46M
```bash
python -m training.train --config configs/collision_1m_cpu.yaml
```

### Compare Checkpoints
Compares metrics and outputs side-by-side:
```bash
python -m evaluation.compare
```

### Launch COLLISION LAB Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 4. Scaling Configs
Find templates for `collision_10m.yaml`, `collision_50m.yaml`, and `collision_100m.yaml` inside the `configs/` directory.

---

## Copyright & Licensing

Copyright @viraj3106. All rights reserved.
