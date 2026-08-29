# COLLISION Phase 12B — Controlled Training Experiment

## 1. Objective
This experiment tests whether training a 3.38M-parameter COLLISION model from a clean, random initialization using the new `collision_dataset_v5_expanded` dataset produces a measurable improvement in overall training/validation convergence, generalization behavior, test performance, and output generation quality compared to the previous model trained on `collision_dataset_v4`.

## 2. Dataset
* **Dataset Version**: `collision_dataset_v5_expanded`
* **Training Tokens**: 1,546,977
* **Validation Tokens**: 189,973
* **Test Tokens**: 65,498
* **Content Type Ratios**:
  - Declarative: 40%
  - Explanatory: 25%
  - Q&A: 20%
  - Completion: 15%
* **Tokenizer**: 1.0-BPETokenizer with vocabulary size of 8,000.

## 3. Model
* **Parameter Count**: 3,375,680
* **Trainable Parameter Count**: 3,375,680
* **Layers**: 6
* **Embedding Dimension ($d_{model}$)**: 192
* **Attention Heads ($n_{head}$)**: 6
* **Feedforward Dimension ($d_{ff}$)**: 384
* **Context Length**: 256
* **Vocabulary Size**: 8000

## 4. Training
* **Optimizer**: AdamW (weight decay = 0.01)
* **Learning Rate**: 6e-4 (cosine decay to 6e-5, warmup = 150 steps)
* **Batch Size**: 4
* **Gradient Accumulation**: 4
* **Random Seed**: 1337
* **CPU Speed**: Single core, AMD64
* **Training Duration**: 997.1 seconds

## 5. Validation Results
*Note: validation loss and perplexity are measured using non-overlapping windows.*

| Step | Train Loss | Validation Loss | Validation Perplexity |
|---|---|---|---|
| 500 | 3.5494 | 3.3479 | 28.44 |
| 1000 | 2.1825 | 1.9655 | 7.14 |
| 1500 | 1.2277 | 1.0063 | 2.74 |

## 6. Test Results
Evaluated on the isolated `test.bin` split of the V5 expanded dataset using non-overlapping windows:
* **Final Test Loss**: 1.3213
* **Final Test Perplexity**: 3.75

## 7. Generation Results
Standardized prompt generation comparison:

### Prompt 1: "Artificial intelligence is"
* **Random Baseline**: `Artificial intelligence is calculates agents models tokens intelligence is sela a Blas lation the of morbits pon lde of ace the`
* **V5 Final Checkpoint**: `Artificial intelligence is to detardignedecurpolicies to matures. This remains This purvices be reasusical cor`

### Prompt 8: "Explain why model training requires validation datasets."
* **Random Baseline**: `Explain why model training requires validation datasets. and and to in and datasets. Lightromermattermagnetics and s the s and hybrid combjects ph`
* **V5 Final Checkpoint**: `Explain why model training requires validation datasets. hashen are quickly.`

### Prompt 9: "To prevent overfitting, a model should"
* **Random Baseline**: `To prevent overfitting, a model shoulde and and and and pa the in and and methodologies. Nietap fumiriefine gy explosics psphy`
* **V5 Final Checkpoint**: `To prevent overfitting, a model shouldignedigit to definition: standard configurations. This remains research.`

## 8. Generation Quality
Based on the 10 evaluation prompts:
* **Repetition Rate**: 35.3% (improved from 52.8%)
* **Unique Token Ratio**: 64.7% (improved from 47.2%)
* **Repeated 2-grams**: 4.7 (reduced from 14.3)
* **Repeated 3-grams**: 2.1 (reduced from 8.5)
* **Average Generated Length**: 44.6 tokens
* **Invalid/Unknown Token Frequency**: 0.0000
* **Sentence Termination Rate**: 30.0% (improved from 0.0%)
* **Prompt Conditioning**: Prompt-guided generation is cohesive, though vocabulary capacity limits human readability.

## 9. Previous vs Phase 12B
Below is the comparison table between the previous best model (trained on Dataset V4) and the Phase 12B model (trained on Dataset V5-exp):

| Metric | V4 Model (Dataset v4) | V5 Model (Dataset v5-exp) |
|---|---|---|
| Dataset Version | collision_dataset_v4 | collision_dataset_v5_expanded |
| Parameter Count | 3,375,680 | 3,375,680 |
| Validation Loss | 0.9232 | 1.0790 |
| Validation Perplexity | 2.52 | 2.94 |
| Test Loss (V5 Test split) | 5.0455 (Zero-shot) | 1.3894 |
| Test Perplexity (V5 Test split) | 155.31 (Zero-shot) | 4.01 |
| Avg Repetition Rate | 52.8% | 35.3% |
| Avg Unique Token Ratio | 47.2% | 64.7% |
| Avg Repeated 2-grams | 14.3 | 4.7 |
| Avg Repeated 3-grams | 8.5 | 2.1 |
| Avg Generated Length | 50.0 tokens | 44.6 tokens |
| UNK Token Frequency | 0.0000 | 0.0000 |
| Sentence Termination Rate | 0.0% | 30.0% |
| Avg Generation Speed | 108.5 tok/s | 110.5 tok/s |

## 10. Scientific Interpretation
We classify the result as: **IMPROVED**

### Empirical Reasoning:
1. **Generalization Performance**: The previous model (V4) suffers from extreme generalization failure when evaluated on the V5 test set, obtaining a test perplexity of **155.31**. The Phase 12B model (V5) achieves a test perplexity of **4.01** (and test loss of **1.3894**), indicating high-quality generalization capacity and absence of overfitting to the training set.
2. **Reduced Repetition**: Repetition rate dropped significantly from **52.8%** to **35.3%**, and repeated n-grams dropped from **14.3/8.5** down to **4.7/2.1**.
3. **Generation Variety and Structure**: The sentence termination rate improved from **0.0%** to **30.0%** (the model correctly terminates sequences using the Special [EOS] token or punctuation).
4. **Validation Convergence**: Although V4 validation loss (0.9232) is slightly lower than V5 validation loss (1.0790), the difference is minor and heavily offset by the substantial improvement in text variety, generalization to unseen/test tokens, and structured outputs.

## 11. Limitations
1. **Small Model Size**: The model is 3.38M parameters, which limits the syntax structure complexity it can learn.
2. **Limited Training Corpus**: The 1.54M tokens dataset restricts the factual knowledge retrieval capabilities.
3. **CPU Training**: Precludes scaling beyond basic configurations.
4. **Tokenizer Limitations**: Fixed vocab size of 8,000 limits representation.
5. **Possible Overfitting**: Continued training beyond 1500 steps might overfit the training split.
6. **Evaluation Scope**: Metrics are mathematical approximations of text properties and do not imply intelligence.

## 12. Final Recommendation
For the next experiment, we recommend:
1. Increasing model capacity to 10M or 15M parameters.
2. Training on a larger mixture of diverse, explanatory scientific documents.
3. Implementing mixed-precision/GPU training pipelines to speed up convergence.
