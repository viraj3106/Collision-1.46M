# Phase 91 Controlled Pretraining Experiment Plan

## 1. Single-Variable Controlled Setup
To rigorously evaluate the hypothesis that dataset composition is the primary bottleneck, Phase 91 will execute a single-variable controlled pretraining experiment:

- **CONTROL**: Pretrain model on `collision_dataset_v5_expanded` (10M tokens).
- **EXPERIMENT**: Pretrain model on `collision_dataset_v9_redesigned` (10M tokens).

---

## 2. Fixed Constants Across Control & Experiment

| Hyperparameter / Component | Fixed Value |
|---|---|
| Model Architecture | 10M Transformer (6 layers, d_model=384, 8 heads, d_ff=768) |
| Total Parameters | 10,282,304 |
| Sequence Length (`max_seq_len`) | 256 |
| Vocabulary Capacity | 8,000 |
| Tokenizer | BPETokenizer |
| Total Token Budget | 10,000,000 tokens |
| Optimizer | AdamW (base lr = 6e-4, min lr = 6e-5, weight decay = 0.01) |
| Scheduler | CosineWarmupScheduler (150 warmup steps) |
| Batch Size & Accumulation | Batch size 4, gradient accumulation 4 |
| Benchmark Evaluation | Frozen 240-Question Independent RAG Benchmark |

---

## 3. Independent Variable
- **DATASET COMPOSITION**: 100% Synthetic Combinatorial Template Data vs 100% High-Diversity Natural/Instruction/RAG Data.
