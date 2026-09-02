# COLLISION-10M Distribution Strategy & Large File Audit

This document outlines the hosting boundaries for repository files, classifies large files that must not be committed to Git, and establishes the model distribution workflow.

## 1. Distribution Strategy

To keep the GitHub repository lightweight and optimize developer cloning times, we decouple source code from model checkpoints and dataset binaries:

### GitHub Repository
- **Scope**: Source code, REST API wrappers, interactive playground, local CLI utilities, documentation, research history, and configurations.
- **Repository URL**: `https://github.com/viraj3106/Collision-1.46M`

### Hugging Face Hub
- **Scope**: Large model weights (`model.pt`), BPE tokenizer vocabularies, merges files, model configurations, and model cards.
- **Hugging Face Model Hub Target**: `viraj3106/collision-10m`
- **Hugging Face Datasets Hub Target**: `viraj3106/collision_dataset_v5_expanded`

---

## 2. Large File Audit & Exclusions

The following assets exceed standard GitHub file size guidelines (50MB warning, 100MB block limit) or represent heavy local runtime logs and caches. They must be excluded from Git commits:

| Filename / Directory | Size | Recommended Hosting / Action |
| :--- | :---: | :--- |
| **`models/collision-10m/model.pt`** | `125,057,611` bytes (~125 MB) | Exclude via `.gitignore`. Host on Hugging Face Model Hub. |
| **`checkpoints/**/*.pt`** (e.g., training checkpoints) | `18 MB` to `125 MB` each | Exclude via `.gitignore` (folder-level). Host on archive/cloud storage for development records. |
| **`datasets/**/*.bin`** (Tokenized binary splits) | `~130 KB` to `~3.1 MB` each | Exclude via `.gitignore`. Host on Hugging Face Datasets or dev download archives. |
| **`data/processed/**/*.bin`** (Tokenized processed outputs) | `~130 KB` to `~3.1 MB` each | Exclude via `.gitignore`. Regenerated dynamically using `python -m data.tokenize`. |

These exclusions have been configured in [.gitignore](file:///v:/collision%20-%201M/.gitignore).
