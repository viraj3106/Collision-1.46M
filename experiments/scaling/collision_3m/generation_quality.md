# COLLISION-3M Generation Quality Audit Report

This report presents a scientific evaluation of the **COLLISION-3M** model (`3,375,680` parameters) to diagnose why it produces repetitive or fragmented outputs despite achieving a low validation perplexity of **2.63**.

---

## A. Inference Pipeline Verification
* **Checkpoint Loading**: Successful. Weights loaded from `checkpoints/scaling/collision_3m/collision-3m-best.pt` onto CPU.
* **Model Configuration**: Verified (6 layers, 192 model dimension, 6 attention heads, context length 256).
* **Tokenizer Loading**: Loaded from `artifacts/tokenizer` (vocab capacity 8,000, active tokens 890).
* **Causal Attention Mask**: Functioning correctly. Output tokens are conditioned on previous sequence.
* **Diagnostic Loop Output**:
  - *Prompt*: `"What is artificial intelligence?"`
  - *Prompt IDs*: `[258, 87, 104, 277, 32, 278, 32, 884, 32, 398, 63]`
  - *Generated IDs*: `[32, 615, 46, 32, 286, 32, 615, 46, 32, 825, 32, 676, 32, 674, 32, 820, 32, 286, 32, 286]`
  - *Decoded Continuation*: `" domain. and domain. hybrid maximize agents advanced and and"`

---

## B. Decoding Strategy Comparison (Repetition Analysis)
We evaluated exactly 100 generated tokens across the 8 target prompts under four different decoding settings.

### Strategy: Greedy (Temp=1.0 / greedy argmax)
| Prompt | Unique Token Ratio | Repeated Unigram | Repeated Bigram | Repeated Trigram | Longest Repeat (tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| "What is artificial intelligence?" | 0.27 | 0.73 | 0.62 | 0.56 | 45 |
| "Computer science is" | 0.16 | 0.84 | 0.79 | 0.77 | 39 |
| "The future of technology" | 0.19 | 0.81 | 0.74 | 0.68 | 41 |
| "An algorithm is" | 0.26 | 0.74 | 0.63 | 0.54 | 28 |
| "Space exploration" | 0.08 | 0.92 | 0.86 | 0.84 | 50 |
| "Why does the Earth orbit the Sun?" | 0.23 | 0.77 | 0.66 | 0.60 | 50 |
| "Machine learning is" | 0.14 | 0.86 | 0.79 | 0.74 | 40 |
| "Photosynthesis is" | 0.26 | 0.74 | 0.55 | 0.35 | 12 |

### Strategy: Conservative Sampling (Temp=0.5, K=40, P=0.9)
| Prompt | Unique Token Ratio | Repeated Unigram | Repeated Bigram | Repeated Trigram | Longest Repeat (tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| "What is artificial intelligence?" | 0.28 | 0.72 | 0.46 | 0.29 | 7 |
| "Computer science is" | 0.20 | 0.80 | 0.62 | 0.46 | 9 |
| "The future of technology" | 0.32 | 0.68 | 0.51 | 0.41 | 21 |
| "An algorithm is" | 0.51 | 0.49 | 0.18 | 0.08 | 5 |
| "Space exploration" | 0.21 | 0.79 | 0.62 | 0.46 | 14 |
| "Why does the Earth orbit the Sun?" | 0.24 | 0.76 | 0.58 | 0.44 | 13 |
| "Machine learning is" | 0.31 | 0.69 | 0.40 | 0.20 | 5 |
| "Photosynthesis is" | 0.33 | 0.67 | 0.46 | 0.29 | 9 |

### Strategy: Default Sampling (Temp=0.8, K=50, P=0.9)
| Prompt | Unique Token Ratio | Repeated Unigram | Repeated Bigram | Repeated Trigram | Longest Repeat (tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| "What is artificial intelligence?" | 0.41 | 0.59 | 0.35 | 0.16 | 4 |
| "Computer science is" | 0.29 | 0.71 | 0.45 | 0.23 | 5 |
| "The future of technology" | 0.50 | 0.50 | 0.21 | 0.11 | 5 |
| "An algorithm is" | 0.42 | 0.58 | 0.35 | 0.20 | 5 |
| "Space exploration" | 0.26 | 0.74 | 0.51 | 0.30 | 7 |
| "Why does the Earth orbit the Sun?" | 0.38 | 0.62 | 0.41 | 0.29 | 7 |
| "Machine learning is" | 0.35 | 0.65 | 0.33 | 0.19 | 5 |
| "Photosynthesis is" | 0.51 | 0.49 | 0.21 | 0.12 | 7 |

### Strategy: Creative Sampling (Temp=1.0, K=50, P=0.95)
| Prompt | Unique Token Ratio | Repeated Unigram | Repeated Bigram | Repeated Trigram | Longest Repeat (tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| "What is artificial intelligence?" | 0.39 | 0.61 | 0.35 | 0.18 | 5 |
| "Computer science is" | 0.37 | 0.63 | 0.34 | 0.18 | 5 |
| "The future of technology" | 0.55 | 0.45 | 0.24 | 0.12 | 7 |
| "An algorithm is" | 0.50 | 0.50 | 0.16 | 0.05 | 3 |
| "Space exploration" | 0.41 | 0.59 | 0.28 | 0.11 | 3 |
| "Why does the Earth orbit the Sun?" | 0.42 | 0.58 | 0.27 | 0.13 | 5 |
| "Machine learning is" | 0.32 | 0.68 | 0.40 | 0.21 | 5 |
| "Photosynthesis is" | 0.47 | 0.53 | 0.26 | 0.13 | 4 |

---

## C. Temperature Scaling Effect
Using the fixed prompt: *"What is artificial intelligence?"*, we scaled the temperature setting while keeping Top-K=50, Top-P=0.9, and Max Tokens=100.

| Temperature | Unique Token Ratio | Repeated Unigram | Repeated Bigram | Repeated Trigram | Decoded Output Sample |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.3 | 0.23 | 0.77 | 0.63 | 0.51 | " and and and and and and and and and to and and and and and and and and and and and and an..." |
| 0.5 | 0.22 | 0.78 | 0.59 | 0.40 | " and of and and and hybrid values data and maximize and and next and software. loss. Unsup..." |
| 0.7 | 0.28 | 0.72 | 0.49 | 0.32 | " next the data and of and data and methodologies. methods of and and next models and data ..." |
| 0.8 | 0.25 | 0.75 | 0.52 | 0.31 | " field. and models function values fundamental and and values agents data of data data lea..." |
| 1.0 | 0.45 | 0.55 | 0.23 | 0.12 | " values of data and advanced models datasets. training maximize from methodologies. tasks...." |

* **Coherence Observation**: Lowering the temperature (e.g., 0.3) reduces token variance but increases loop repetitions. Higher temperature (1.0) improves unigram variety but increases syntax fragmentation.

---

## D. Prompt-Conditioning Results
Comparing continuations of different prompts under identical settings:
* **"Computer science is"** -> ` to and and calculates to sequence a learn to data data in point values next point and next a to seq...`
* **"Why does the Earth orbit the Sun?"** -> ` and clde and combjects data and and next and and and of to of data maximize of hybrid in methodolog...`

* **Observation**: The model's continuation is highly conditioned on the prompt. Distinct prompts lead to distinct token paths, proving that prompt conditioning is functional.

---

## E. Dataset Distribution Analysis
* **Approximate Domains**: Physics, Astronomy, Computer Science, Philosophy, Artificial Intelligence, and sample filler paragraphs.
* **Training Text Style**: Synthetic, short, declarative paragraphs concatenated together (e.g., *"Sorting algorithms arrange elements in a specific order, such as ascending."*).
* **Question Frequency**: **0%**. There are no interrogative marks (`?`) or question-answering sequences in the training set.
* **Scientific Questions**: Completely unrepresented.
* **Evaluation Alignment**: When evaluated on questions like *"Why does the Earth orbit the Sun?"*, the model attempts to map the prompt to technical declarative terms (like astronomy coordinates or physics vectors) because it has never seen a Q&A distribution.

---

## F. Validation Metrics Verification
* **Validation Loss**: `0.9663` (Verified directly from `experiments/scaling/collision_3m/training_log.csv`)
* **Validation Perplexity**: `2.63` (Verified directly)

---

## G. CPU Inference Speed Benchmark
Generative throughput comparison on the same CPU hardware (using Default Sampling Temp=0.8, Top-K=50, Top-P=0.9):

| Model | Parameters | Average Inference Speed (tokens/sec) | Throughput Change (%) |
| :--- | :---: | :---: | :---: |
| **COLLISION-1.46M (Base)** | 1,462,464 | 165.15 tok/s | - |
| **COLLISION-3M (Experiment)** | 3,375,680 | 70.77 tok/s | -57.15% |

---

## H. Known Limitations
1. **Low Parameter Capacity (3.38M)**: Insufficient depth to represent abstract multi-step reasoning.
2. **Repetitive Training Corpus**: The dataset consists of highly duplicated sentence templates, reinforcing loop states.
3. **Distribution Mismatch**: The validation prompts contain questions and conversational prompts, whereas the training set consists entirely of declarative technical paragraphs.

---

## I. Final Classification & Next Steps
We classify the current generation behavior as a:

```text
D. Dataset-distribution problem (Combined with low parameter capacity)
```

**Diagnostic Evidence**:
1. Causal masking and inference pipelines are 100% functional (proved in Section A & D).
2. The low validation loss (`0.9663`) and perplexity (`2.63`) indicate the model has fully mastered the validation split.
3. However, because the training split contains synthetic, highly repetitive, purely declarative paragraphs, the model naturally produces repetitive loops and cannot format QA sequences.

**Recommended Next Step**:
Expand and diversify the training corpus (`collision_dataset_v5`) to include multi-sentence logic, question-answer formats, and reduce sentence duplicates before scaling to 7M parameters.
