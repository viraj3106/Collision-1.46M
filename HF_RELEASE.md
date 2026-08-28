# Hugging Face Release Guide for COLLISION-1.46M

This document outlines the upload process, file layout, and model page metadata required to publish COLLISION-1.46M to the Hugging Face Model Hub.

## 1. Model Repository Name
Recommended name: `viraj3106/collision-1.46m`

## 2. Files to Upload
Upload the following files to the Hugging Face repository root:
* `checkpoints/phase6/collision-1.46m-best.pt` (rename to `collision-1.46m-best.pt` in the hub root)
* `release/release.yaml`
* `artifacts/tokenizer/vocab.json`
* `artifacts/tokenizer/merges.txt`
* `MODEL_CARD.md` (rename to `README.md` on Hugging Face to serve as the Model Card page)
* `release_inference.py` (optional - helper script for hub users)

## 3. Hugging Face Metadata Card (YAML Frontmatter)
Prepend this block to the Hugging Face repository's `README.md` to format the hub page correctly:
```yaml
---
language: en
license: mit
tags:
- text-generation
- custom-transformer
- educational
- cpu-optimized
datasets:
- collision_dataset_v4
metrics:
- perplexity
model_name: COLLISION-1.46M
parameters: 1.46M
---
```

## 4. Upload Checklist
- [ ] Install the Hugging Face CLI tool: `pip install huggingface_hub`
- [ ] Login using write-access token: `huggingface-cli login`
- [ ] Create repository: `huggingface-cli repo create collision-1.46m`
- [ ] Clone repository: `git clone https://huggingface.co/viraj3106/collision-1.46m`
- [ ] Copy files (checkpoint, tokenizer files, and code) to target directory.
- [ ] Commit and push changes:
  ```bash
  git add .
  git commit -m "Initial release of COLLISION-1.46M model and custom BPE tokenizer"
  git push origin main
  ```

## 5. Verification Procedure
To verify the uploaded model hub resources run locally:
```bash
# Clone model hub files
git clone https://huggingface.co/viraj3106/collision-1.46m

# Test inference using the entry point pointing to the cloned files
python release_inference.py --checkpoint collision-1.46m/collision-1.46m-best.pt --tokenizer collision-1.46m/
```
