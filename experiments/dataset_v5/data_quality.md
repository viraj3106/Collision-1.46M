# COLLISION-v5 Dataset Quality Report

## 1. Dataset Overview
COLLISION-v5 is a CPU-manageable educational and research dataset designed specifically to solve the distribution limitations discovered in COLLISION-v4. This version introduces controlled Q&A formats, explanatory prose, and declarative summaries to promote coherent multi-sentence predictions and correct prompt response behaviors.

## 2. Actual Content Type Distribution
Targeted vs Measured content types in `collision_dataset_v5`:
* **Declarative Knowledge (factual assertions)**: 400 examples (40.00%)
* **Explanatory Text (process explanations)**: 250 examples (25.00%)
* **Question/Answer Pairs**: 200 examples (20.00%)
* **Completion Stems**: 150 examples (15.00%)

## 3. Domain Distribution
* Computer Science: 270
* Artificial Intelligence: 118
* Machine Learning: 149
* Physics: 203
* Mathematics: 109
* Space: 151

## 4. Token Statistics
* **Total Tokens**: 87,209
* **Train Tokens**: 67,914
* **Val Tokens**: 12,441
* **Test Tokens**: 6,854
* **Average Tokens/Example**: 87.2
* **Unknown/PAD Tokens**: 0 (all text maps to vocabulary successfully)

## 5. Duplicate and Leakage Analysis
* **Exact duplicates filtered**: 1000
* **Sentence Leakage Rate**: **0.00%** (deterministic split by base group hashes avoids leakage of derived prompts between Train, Val, and Test splits).

## 6. N-gram Repetition Analysis
* Average sentence length: 13.17 words
* Median sentence length: 13.0 words
* Vocabulary: Same Custom BPE Tokenizer (active size 894)

## 7. Comparison Table (collision_dataset_v4 vs collision_dataset_v5)

| Metric | collision_dataset_v4 | collision_dataset_v5 |
| :--- | :---: | :---: |
| **Total Examples** | N/A | 1000 |
| **Total Tokens** | 2,302,003 | 87,209 |
| **Question/Answer (%)** | 0.00% | 20.00% |
| **Explanatory (%)** | 0.00% | 25.00% |
| **Declarative (%)** | 100.00% | 40.00% |
| **Completion (%)** | 0.00% | 15.00% |
| **Duplicate Rate (%)** | 0.00% | 50.00% |
| **Active Vocab Size** | 890 | 894 |
