# COLLISION-10M Dataset License Audit

This document summarizes the origin, content composition, and licensing audit of the `collision_dataset_v5_expanded` corpus used to train the `COLLISION-10M` v1.0.0 model.

## 1. Dataset Origin & Metadata

- **Name**: `collision_dataset_v5_expanded`
- **Total Unique Documents**: 15,649
- **Domain Distribution**:
  - Mathematics: 2,571
  - Artificial Intelligence: 2,559
  - Computer Science: 2,634
  - Space: 2,654
  - Physics: 2,631
  - Machine Learning: 2,600
- **Total Token Count**: 1,802,448 tokens (Train: 1,546,977 tokens; Val: 189,973 tokens; Test: 65,498 tokens)

## 2. Generation Method & Synthetic Content

- **Method**: The dataset was generated programmatically using a custom *Combinatorial Semantic Variation Generator* (`data/generate_corpus.py`).
- **Input Seed Sentences**: The baseline templates consist of original declarative scientific definitions across 10 subject fields written explicitly for the COLLISION project.
- **Deduplication & Quality Control**: Paragraphs were audited to filter duplicates, strip non-standard formatting prefixes, and split deterministically using a seeded randomizer (`seed=42`).

## 3. External Material & Copyright Audit

- **Copyrighted Text**: The dataset contains **zero** external copyrighted material (such as copyrighted books, private code bases, or web crawls).
- **Synthetic Redistribution**: Although the dataset is synthetic, it is composed of highly structured sentences.
- **Redistribution Status**: Since the templates and generators are original assets developed under the COLLISION codebase, the generated dataset splits (`train.bin`, `val.bin`, `test.bin`) can be safely distributed alongside the model.

## 4. Redistribution and Attribution

- The dataset splits are packaged for developer reproduction under the same terms as the model.
- No third-party copyrights or attributions are required for this synthetic corpus.
