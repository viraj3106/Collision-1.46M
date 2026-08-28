# COLLISION-1.46M Model Card

## Overview
COLLISION-1.46M is a small decoder-only Transformer language model trained completely from scratch on CPU. This project serves as an educational and research exploration into training behavior, tokenizer vocabulary capacity, dataset quality, generalization capability, and CPU-first execution constraints.

## Model Architecture
Verified architecture parameters from checkpoint metadata:
* **Parameters**: 1,462,464 (Model config capacity 8,000 vocab).
* **Architecture**: Decoder-only causal Transformer.
* **Layers**: 3 blocks.
* **Embedding Dimension**: 128.
* **Attention Heads**: 4.
* **Feedforward Dimension (d_ff)**: 256.
* **Context Length (max_seq_len)**: 256 tokens.
* **Vocabulary Size**: 8,000 capacity (890 active tokens).
* **Weight Tieing**: True (shared embedding and output weights).

## Training
* **Initialization**: Random initialization (no pre-trained weights).
* **Hardware**: CPU-only.
* **Tokenizer**: Custom Byte-Pair Encoding (BPE) restricted to word boundaries.
* **Dataset**: `collision_dataset_v4` (2,072,993 training tokens, 229,010 validation tokens). Covers Physics, Astronomy, Philosophy, CS, AI, and related subjects.
* **Optimizer**: AdamW (learning rate = 6e-4, weight decay = 0.01, Cosine Warmup over 150 steps).
* **Duration**: 1,500 steps, batch size 4 (accumulation 4).

## Evaluation
The model was trained in two phases, illustrating the impact of dataset cleaning and train/validation leakage reduction without increasing parameter count.

* **Phase 5 (First Training Run on v3 Dataset)**:
  - Validation Perplexity: **62.86**
  - Overfitting: Severe sequence-level overfitting and character sequence corruption due to sentence paragraphs leakage (~26% sentence duplicate leakage).
* **Phase 6 (Generalization Experiment on v4 Dataset)**:
  - Validation Perplexity: **6.93**
  - Generalization: Smooth training loss and validation loss convergence, complete removal of duplicate paragraph leakage (0% paragraph leakage), and elimination of gibberish character merges.

## Limitations
COLLISION-1.46M is an experimental small-scale educational model. Intended limitations include:
* **Limited Capacity**: Due to only 1.46M parameters, it cannot form deep logical conclusions or maintain coherent long-term conversation.
* **Repetition**: The model may fallback to repetitive token loops when context length increases.
* **Factual Inaccuracy**: Outputs may contain incorrect, nonsensical, or fabricated assertions.
* **No Instruction Following**: The model is trained as a raw completion agent and does not follow instructional formatting.
* **Short-Context Memory**: The maximum context limit is strictly 256 tokens.

## Intended Use
* Research into small language model behavior.
* Educational exploration of Transformer mechanics.
* Benchmarking CPU inference constraints.

## Not Intended For
* Production environments.
* High-stakes decision making.
* Medical, legal, or financial advice.
* Conversational chatbot use-cases.

## Reproducibility
Launch standalone inference on CPU using:
```bash
python release_inference.py --prompt "What is artificial intelligence?" --max-tokens 100
```
Launch the interactive web playground:
```bash
streamlit run dashboard/app.py
```
