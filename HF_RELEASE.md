# Hugging Face Release Guide for COLLISION-10M & COLLISION NLP Engine

This document outlines the upload process, file layout, and metadata required to publish COLLISION-10M and the COLLISION NLP Engine to the Hugging Face Model Hub and Hugging Face Spaces.

---

## 1. Hugging Face Repositories
* **Model Hub**: `collision-10M/collision-10m`
* **Space Hub**: `collision-10M/collision-ai-lab` (Static SDK)
* **Dataset Hub**: `collision-10M/collision_dataset_v5_expanded`

---

## 2. Model Hub Release Layout (`release/huggingface/`)
The `release/huggingface/` directory is prepared for 1-step deployment to Hugging Face Model Hub:
```
release/huggingface/
├── README.md               # Model Card with rich YAML frontmatter & NLP documentation
├── DESCRIPTION.md          # Short & Long architectural descriptions
├── model_metadata.json     # Parameter specs, SHA-256 checksums, and NLP feature list
├── checksums.sha256        # Verified cryptographic SHA-256 hashes
├── model.pt                # 10.28M parameter model weights (125,057,611 bytes)
├── config.json             # Transformer hyperparameters
├── generation_config.json  # Sampling temperature, top_k, repetition penalty
├── tokenizer.json          # Tokenizer config metadata
└── tokenizer/              # BPE vocab.json, merges.json, config.json, stats.json
```

---

## 3. Hugging Face Space Layout (`release/huggingface_space/`)
The `release/huggingface_space/` directory is ready for 1-step deployment to Hugging Face Spaces:
```
release/huggingface_space/
├── README.md               # Space metadata (sdk: static, title: COLLISION AI & NLP Lab)
├── index.html              # Modern glassmorphism UI portal
├── app.js                  # Client-side reactivity and API integrations
└── styles.css              # Cyber-indigo theme styles
```

---

## 4. Publication Procedure

### Option A: Using `huggingface_hub` Python SDK
```python
from huggingface_hub import HfApi

api = HfApi()

# 1. Upload Model Hub
api.upload_folder(
    folder_path="release/huggingface",
    repo_id="collision-10M/collision-10m",
    repo_type="model"
)

# 2. Upload Space Hub
api.upload_folder(
    folder_path="release/huggingface_space",
    repo_id="collision-10M/collision-ai-lab",
    repo_type="space"
)
```

### Option B: Using Git
```bash
# Model Hub
git clone https://huggingface.co/viraj3106/collision-10m
cp -r release/huggingface/* collision-10m/
cd collision-10m
git add . && git commit -m "Release COLLISION-10M and in-house NLP Engine"
git push origin main

# Space Hub
git clone https://huggingface.co/spaces/viraj3106/collision-ai-lab
cp -r release/huggingface_space/* collision-ai-lab/
cd collision-ai-lab
git add . && git commit -m "Deploy COLLISION AI & NLP Lab Streamlit Space"
git push origin main
```

---

## 5. Verification
Run the Hugging Face release verification suite before deploying:
```bash
python release/verify_huggingface_package.py
```
