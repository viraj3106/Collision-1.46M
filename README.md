# 💥 COLLISION-1M — Decoder-only Language Model Trained From Scratch

COLLISION-1M is a decoder-only Transformer language model ($1,462,464$ parameters) built and trained completely from scratch in PyTorch. It is designed to run locally, offline, and CPU-first on standard machines.

---

## 1. How the Transformer Works & Architecture

The architecture of COLLISION-1M is a GPT-style causal autoregressive language model. The core layers include:
- **Token Embeddings**: Maps vocabulary indices to vectors in $\mathbb{R}^{d_{model}}$.
- **Positional Embeddings**: Learnable spatial embedding vectors mapping input token positions.
- **Causal Multi-Head Self-Attention**: Projects input representations to Query, Key, and Value vectors. Attention scores are calculated as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
Where $M$ is the causal mask (lower triangular mask forcing the model to only attend to historical tokens).
- **Feed-Forward Networks (MLP)**: Projects tokens into a higher dimension ($d_{ff}$), applies a GELU activation, and projects back.
- **Pre-Layer Normalization & Residual Connections**: Applied at each block layer step to stabilize gradients and facilitate backpropagation.
- **Weight Tying**: The input token embedding layer shares weights with the output projection layer (Language Model Head) to reduce parameters and conserve memory.

---

## 2. Programmatic Parameter Calculation

The parameter count is programmatically calculated by analyzing the layers of the configuration:
1. **Embeddings**: $\text{vocab\_size} \times d_{model} + \text{max\_seq\_len} \times d_{model}$
2. **Decoder Blocks**: $n_{layer} \times (\text{Attention} + \text{MLP} + \text{LayerNorms})$
   - **Attention**: $(3 \times d_{model} \times d_{model}) + (3 \times d_{model}) + (d_{model} \times d_{model}) + d_{model}$
   - **MLP**: $(2 \times d_{model} \times d_{ff}) + d_{ff} + d_{model}$
   - **LayerNorms**: $2 \times (2 \times d_{model})$
3. **Final LayerNorm**: $2 \times d_{model}$
4. **LM Head**: $\text{vocab\_size}$ (when weights are tied) or $\text{vocab\_size} \times d_{model} + \text{vocab\_size}$ (when untied).

For `d_model=128`, `d_ff=256`, `n_layer=3`, `n_head=4`, `vocab_size=8000`, the parameter count is exactly:
**1,462,464 parameters**

---

## 3. Pre-Training Pipeline

COLLISION-1M implements a modular pre-training dataset and validation suite:

```
  [Raw Documents] -> [data.build] -> [Deduplication & Cleaning] -> [Train/Val Splits]
                                                                         |
                                                                         v
  [Training safety gates] <- [Pre-Training Readiness] <- [Tokenizer Train/Verification]
```

### 1. Dataset Generation (`data/generate_corpus.py`)
Generates 5,000,000 characters of clean, diverse educational texts across Physics, Computer Science, AI, Astronomy, and Philosophy.
```bash
python -m data.generate_corpus
```

### 2. Dataset Builder (`data/build.py`)
Safely reads files in `data/raw/`, cleans corrupt control characters, normalizes line endings and whitespace, detects/removes duplicates, and deterministically splits the tokens into a $90/10$ train/val split.
```bash
python -m data.build
```

### 3. Dataset Health & Reports (`data/report.py` & `data/stats.py`)
Computes token statistics and classifications:
```bash
python -m data.stats
python -m data.report
```

---

## 4. CLI Commands Reference

Perform the complete lifecycle of COLLISION-1M using the commands below:

### Setup Structure & Directories
```bash
python -m collision.setup
```

### Train BPE Tokenizer
```bash
python -m data.tokenize --train
```

### View Model Configuration Info
```bash
python -m collision.info
```

### Run Model Pre-Training Readiness Check
Verifies dataset integrity, binary splits, tokenizer loopbacks, loads checkpoints, and outputs readiness decisions.
```bash
python -m collision.readiness_check
```

### Run CPU Benchmark Profiling
Runs exactly 100 steps to profile steps/sec, tokens/sec, memory usage, and outputs estimations for 1K, 5K, and 10K steps.
```bash
python -m training.train --profile --config configs/collision_1m_cpu.yaml
```

### Train COLLISION-1M
```bash
python -m training.train --config configs/collision_1m_cpu.yaml
```
*Note: Training will block and warn if the dataset token count is under 1,000,000.*

### Compare Checkpoints
Compares saved step checkpoints in a table along with text generation comparisons side-by-side:
```bash
python -m evaluation.compare
```

### Launch COLLISION LAB Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 5. Scaling Plan

- **Model Configs**: Find scaling configurations in the `configs/` directory:
  - `collision_1m_cpu.yaml`
  - `collision_10m.yaml`
  - `collision_50m.yaml`
  - `collision_100m.yaml`

---

## Copyright & Licensing

Copyright @viraj3106. All rights reserved.
