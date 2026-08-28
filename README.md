# COLLISION-1.46M

An experimental 1.46M parameter language model trained from scratch on CPU.

![COLLISION-1.46M Banner](assets/collision-banner.png)

## 1. What is COLLISION?
COLLISION-1.46M is a small decoder-only Transformer language model built and trained completely from scratch (no pretrained weights) in PyTorch, designed specifically for offline, CPU-first execution.

## 2. Why it Exists
COLLISION is a research project designed to explore:
* What happens when a small Transformer model is trained on limited hardware with zero pretraining.
* The impact of dataset quality, duplication rates, and train/validation split bias on generalization.
* CPU-only pretraining limits and performance.

## 3. Architecture
Verified architecture parameters:
* **Parameters**: 1,462,464 parameters (Model capacity 8,000 vocab).
* **Architecture**: Decoder-only causal Transformer.
* **Layers**: 3.
* **Embedding size (d_model)**: 128.
* **Attention heads**: 4.
* **Feedforward dimension (d_ff)**: 256.
* **Context Length**: 256 tokens.
* **Embedding Tieing**: True.

## 4. Training
* **Initialization**: Randomly initialized.
* **Hardware**: Trained completely on consumer CPU.
* **Dataset**: `collision_dataset_v4` (2,072,993 training tokens, 229,010 validation tokens).
* **Tokenizer**: Custom Byte-Level BPE tokenizer with boundary restrictions (890 active vocabulary).
* **Optimizer**: AdamW (lr=6e-4, weight_decay=0.01, Cosine Warmup over 150 steps).

## 5. Phase 5 → Phase 6 Results
The model underwent two distinct pretraining runs:

| Metric | Phase 5 (Step 1500) | Phase 6 (Step 1500) |
| :--- | :---: | :---: |
| **Training Loss** | 2.0934 | 2.1608 |
| **Validation Loss** | 4.1409 | 1.9363 |
| **Validation Perplexity** | 62.86 | **6.93** |
| **Overfitting Indicator** | High (degrades after step 1500) | None (monotonically decreases) |
| **Dataset Version** | `collision_dataset_v3` (with duplicates/leakage) | `collision_dataset_v4` (deduplicated/leak-free) |

*The significant decrease in validation perplexity came entirely from dataset auditing and better train/validation separation, without increasing the model size.*

## 6. COLLISION LAB
COLLISION LAB is a polished public AI playground interface designed for CPU-only inference, checkpoint selection, and visual statistics.

To launch the web playground:
```bash
streamlit run dashboard/app.py
```

## 7. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

## 8. Inference
Run inference using the official standalone entry point script:
```bash
python release_inference.py --prompt "What is artificial intelligence?" --max-tokens 100
```

Supported arguments:
* `--prompt`: Input text prompt.
* `--checkpoint`: Path to model checkpoint file (default: `checkpoints/phase6/collision-1.46m-best.pt`).
* `--tokenizer`: Path to tokenizer directory (default: `artifacts/tokenizer`).
* `--temperature`: Generation temperature (default: `0.8`).
* `--top-k`: Top-K token filtering (default: `50`).
* `--top-p`: Top-P nucleus sampling (default: `0.9`).

## 9. Evaluation
Run checkpoint comparison:
```bash
python -m evaluation.compare
```

## 10. Repository Structure
```
├── checkpoints/          # Saved training checkpoints (Phase 5 & 6)
├── configs/              # Model configuration files
├── dashboard/            # COLLISION LAB Streamlit app (app.py)
├── data/                 # Tokenization and dataset preparation scripts
├── datasets/             # Tokenized dataset bins (v4)
├── evaluation/           # Output evaluation and comparison scripts
├── experiments/          # Evaluation logs, reports, and loss curves
├── inference/            # Legacy inference modules
├── model/                # Transformer architecture files
├── release/              # Release files metadata config
├── MODEL_CARD.md         # Detailed Model Card
├── RELEASE_CHECKLIST.md  # Open-source release verification checklist
├── HF_RELEASE.md         # Hugging Face deployment guidelines
└── release_inference.py  # Standalone CLI entry point
```

## 11. Limitations
COLLISION-1.46M is an experimental small-scale model. Intended limitations:
* **Factual Accuracy**: The model outputs are statistically simulated completion sequences and may be factually incorrect, incomplete, repetitive, or nonsensical.
* **No Instruction Following**: The model has not undergone instruction fine-tuning.
* **Limited Context**: STRICT context length limit of 256 tokens.

## 12. License
Distributed under the MIT License. See [LICENSE](LICENSE) or standard project terms for more information.

## 13. Hugging Face Release
The model is prepared for Hugging Face model repository upload. For full instructions on uploading checkpoints, tokenizers, and model page metadata, see [HF_RELEASE.md](HF_RELEASE.md).
