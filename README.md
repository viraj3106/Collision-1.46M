# COLLISION-1M

COLLISION-1M is a small decoder-only Transformer language model (~1.46M parameters) built and trained completely from scratch in PyTorch, designed specifically for offline CPU-first execution.

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

## 3. CLI Commands Reference

Perform the complete lifecycle of COLLISION-1M using the commands below:

### Setup Structure & Directories
```bash
python -m collision.setup
```

### Clean & Prepare Raw Dataset
```bash
python -m data.prepare
```

### Train Byte-Pair Encoding (BPE) Tokenizer
Trains the custom BPE tokenizer and tokenizes the datasets:
```bash
python -m data.tokenize
```

### View Model Configuration
Displays programmatically calculated model information and parameter counts:
```bash
python -m collision.info
```

### Run Smoke Test
Run a quick liveness test verifying tokenizer, dataset, train, validation, saving, and loading:
```bash
python -m training.train --smoke-test
```

### Train COLLISION-1M
Train the model with configurable parameters:
```bash
python -m training.train --epochs 5 --batch-size 8 --learning-rate 0.001 --cpu-safe
```
- Use `--resume` to continue training from the latest checkpoint.

### Run Text Generation
Generate text autoregressively with temperature, top-k, and top-p:
```bash
python -m inference.generate --prompt "COLLISION-1M is" --temperature 0.8 --max-tokens 50
```

### Run Evaluation
Evaluate a checkpoint's perplexity and run fixed prompt tests:
```bash
python -m evaluation.evaluate
```

### Launch COLLISION LAB Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 4. Scaling & Future Self-Learning

- **Scaling Configs**: Find scaling configurations in the `configs/` directory (e.g., templates for `collision_10m.yaml` and `collision_50m.yaml`).
- **Self-Learning Guidelines**: See [learning/README.md](file:///v:/collision%20-%201M/learning/README.md) for architectural details on how future versions can safely collect feedback, extend datasets, retrain, and validate checkpoints without risk of adversarial corruption or poisoning.
