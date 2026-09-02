# COLLISION-10M License Decision

This document details the licensing considerations and decisions for the public developer release of the `COLLISION-10M` v1.0.0 model weights, inference code, custom tokenizer, and synthetic training dataset.

## 1. Existing Licenses

Prior to this audit:
- The project README referenced the MIT License.
- No standalone LICENSE file was present in the repository root.
- The HF_RELEASE metadata draft proposed the MIT License.

## 2. Release Asset Decisions

### Code Licensing
- **Decision**: **MIT License**
- **Justification**: The training, inference, API, and playground codebases are original implementations. Standard open-source libraries used (FastAPI, PyTorch, Streamlit, Numpy) are fully compatible with MIT licensing.

### Model Weights Licensing
- **Decision**: **MIT License**
- **Justification**: The model weights in `models/collision-10m/model.pt` were trained from scratch under this project. No licensed pretrained weights were utilized.

### Tokenizer Licensing
- **Decision**: **MIT License**
- **Justification**: The tokenizer configuration, vocabulary, and merges JSON files are generated assets based on original code and synthetic text.

### Dataset Licensing
- **Decision**: **MIT License / Public Domain**
- **Justification**: The training dataset `collision_dataset_v5_expanded` is synthetic and was built without external copyrighted materials. It can be freely distributed under the MIT License alongside the model weights.

## 3. Third-Party Dependency Compatibility

All core project dependencies are distributed under permissive open-source licenses:
- **PyTorch**: BSD 3-Clause License
- **NumPy**: BSD 3-Clause License
- **FastAPI**: MIT License
- **Uvicorn**: BSD 3-Clause License
- **Streamlit**: Apache License 2.0
- **Httpx**: BSD 3-Clause License
- **PyYAML**: MIT License
- **psutil**: BSD 3-Clause License

No copyleft (e.g., GPL) or restrictive proprietary licenses are present in the dependency tree.

## 4. Final License Action

An MIT License has been applied to this repository.
- Code: MIT
- Model Checkpoints: MIT
- Dataset: MIT
