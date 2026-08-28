# COLLISION-v5-Expanded Dataset Quality Report

## 1. Dataset Overview
COLLISION-v5-expanded scales the text distribution of COLLISION-v5 to reach the target token budget of **~1.5M - 1.6M training tokens**. It introduces combinatorial semantic templates across the 6 domains to avoid simple noun-swapping or syntax repetition, generating 15,000+ highly unique paragraphs.

## 2. Actual Content Type Distribution
Targeted vs Measured content types in `collision_dataset_v5_expanded`:
* **Declarative Knowledge (factual assertions)**: 6260 examples (40.00%)
* **Explanatory Text (process explanations)**: 3912 examples (25.00%)
* **Question/Answer Pairs**: 3130 examples (20.00%)
* **Completion Stems**: 2347 examples (15.00%)

## 3. Domain Distribution
* Computer Science: 2634
* Artificial Intelligence: 2559
* Machine Learning: 2600
* Physics: 2631
* Mathematics: 2571
* Space: 2654

## 4. Token Statistics
* **Total Tokens**: 1,802,448
* **Train Tokens**: 1,546,977
* **Val Tokens**: 189,973
* **Test Tokens**: 65,498
* **Average Tokens/Example**: 115.2
* **Unknown/PAD Tokens**: 0 (all text maps to vocabulary successfully)

## 5. Duplicate and Leakage Analysis
* **Exact duplicates filtered**: 0
* **Sentence Leakage Rate**: **0.00%** (deterministic split by base group hashes avoids leakage of derived prompts between Train, Val, and Test splits).

## 6. N-gram Repetition Analysis
* Average sentence length: 16.49 words
* Median sentence length: 13.0 words
* Vocabulary: Same Custom BPE Tokenizer (active size 894)

## 7. Comparison Table (collision_dataset_v4 vs collision_dataset_v5_expanded)

| Metric | collision_dataset_v4 | collision_dataset_v5_expanded |
| :--- | :---: | :---: |
| **Total Examples** | N/A | 15649 |
| **Total Tokens** | 2,302,003 | 1,802,448 |
| **Question/Answer (%)** | 0.00% | 20.00% |
| **Explanatory (%)** | 0.00% | 25.00% |
| **Declarative (%)** | 100.00% | 40.00% |
| **Completion (%)** | 0.00% | 15.00% |
| **Duplicate Rate (%)** | 0.00% | 0.00% |
| **Active Vocab Size** | 890 | 894 |
