
# COLLISION

Small Models. Real AI.

> **Project Status (Phase 70)**: **`DATA_COLLECTION_HOLD_HUMAN_TRAFFIC_REQUIRED`**  
> The software, REST API, Streamlit client, telemetry pipeline, PII redaction audit, and automated training readiness gates are 100% complete and fully verified. Base production model `collision-10m` is frozen (SHA256: `d256d46d...3775b97`) and Apex Research Candidate `J52` is preserved.

---

## What is COLLISION?

COLLISION is a research project designed to explore Transformer causal language modeling in extreme low-resource regimes (sub-50M parameters). It focuses on causal language modeling convergence, CPU-first development, data quality engineering, and alignment behavior for tiny model architectures.

## Why does it exist?

COLLISION exists to prove that high-quality, scientifically hygienic training data, coupled with rigorous evaluation protocols, can yield stable and convergent language representation spaces under extreme size constraints (sub-50M parameters) on consumer CPUs without GPU pretraining.

---

## Model Variants & Apex Candidates

| Model Variant | Parameters | Layers ($n_{\text{layer}}$) | $d_{\text{model}}$ | Attention Heads ($n_{\text{head}}$) | $d_{\text{ff}}$ | Context Window | Weight Tying | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **COLLISION-10M** | 10,282,304 | 6 | 384 | 8 | 768 | 256 | Enabled | Frozen Production Base (`v1.0.0`) |
| **SFT Candidate J52** | 10,282,304 | 6 | 384 | 8 | 768 | 256 | Enabled | Apex Research Candidate (`collision_sft_v3`) |
| **COLLISION-7M** | 6,338,880 | 8 | 256 | 8 | 512 | 256 | Enabled | Scaling Variant |
| **COLLISION-1.46M** | 1,462,464 | 3 | 128 | 4 | 256 | 256 | Enabled | Historical Baseline (Phase 5/6) |

---

## Research Progression & Phase Summary (Phases 1 – 70)

- **Phases 1–6 (Baselines & Data Quality Breakthrough)**: Phase 5 baseline suffered validation loss divergence (`4.1409`). Forensic audit revealed alphabetical split bias and 26.82% sentence leakage. Phase 6 introduced `collision_dataset_v4` (subject-wise split, 0% leakage), collapsing validation loss to **1.9363** and perplexity to **6.93** without changing model capacity.
- **Phases 10–15 (COLLISION-10M Flagship)**: Scaled architecture to 10.28M parameters. Trained on 10,000,384 tokens from scratch on CPU. Achieved best validation perplexity of **2.11** (loss `0.7454`) and test perplexity of **1.79** (`0.5805`).
- **Phases 16–38 (Enterprise Infrastructure & Lab)**: Frozen production weights (`models/collision-10m/model.pt`), built FastAPI REST service (`/v1/generate`, `/v1/feedback`), Streamlit interactive client (`playground/app.py`), SQLite auth DB (`collision_api.db`), and Docker containers.
- **Phases 39–46 (DPO Forensic Audit & Math Repair)**: Explored Direct Preference Optimization. Discovered mathematical bug omitting reference model ratio $\pi_{\text{ref}}$ in loss calculation. Repaired canonical loss, but confirmed DPO objective misalignment on 10M models due to length/repetition sensitivity.
- **Phases 47–52 (SFT Alignment & Apex Candidate J52)**: Pivoted to Supervised Fine-Tuning. Apex Candidate `J52` achieved record metrics: **Generalization Score: 66.85, Coherence: 38.50, Instruction Following: 48.20**.
- **Phases 53–70 (Beta Telemetry & Data Collection Hold)**: Built automated data cleaning, PII redaction, and strict training readiness gates requiring 100 clean human records. With 7 clean records collected and zero synthetic injection allowed, the project formally entered **`DATA_COLLECTION_HOLD_HUMAN_TRAFFIC_REQUIRED`**.

---

## Quick Start

Follow these steps to run local inference using the frozen model.

### 1. Clone the Repository
```bash
git clone https://github.com/viraj3106/Collision-1.46M.git
cd collision
```

### 2. Install Dependencies
```bash
pip install -r requirements-release.txt
```

### 3. Run Direct Local Inference
Execute causal completion directly using the pre-existing inference engine (requires no API or running servers):
```bash
python release_inference.py --prompt "Artificial intelligence is" --checkpoint models/collision-10m/model.pt
```

---

## API

A production-oriented FastAPI service is provided to query completions via HTTP requests.

### Start the API Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Check Health & Metrics
```bash
curl -X GET http://localhost:8000/health
```

### Request Completion
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "collision-10m",
    "prompt": "Artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
  }'
```

For full client examples, see [API Documentation](file:///v:/collision%20-%201M/docs/api.md).

---

## Playground

The playground consists of a Streamlit client (`playground/app.py`) that interacts with the FastAPI backend.

1. Start the API server in one terminal:
   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the Streamlit frontend in a second terminal:
   ```bash
   streamlit run playground/app.py
   ```

---

## Evaluation Metrics

### COLLISION-10M Production Base (Step 2,500 Checkpoint)
- **Validation Loss**: 0.7454 | **Validation Perplexity**: 2.11
- **Test Loss**: 0.5805 | **Test Perplexity**: 1.79
- **Unique Token Ratio**: 58.9% | **Termination Rate**: 62.5%

### SFT Candidate J52 (Phase 52 Apex Research Candidate)
- **Generalization Score**: 66.85
- **Coherence**: 38.50
- **Instruction Following**: 48.20

---

## Limitations

- **Context Window**: 256 tokens.
- **Small Parameter Count**: 10.28M parameters limit generation capability.
- **Repetitive Output**: Subject to unigram repetition biases common in small models.
- **Not Instruction Tuned (Base Model)**: Base model will continue text prompts; SFT Candidate J52 provides basic instruction response capability.
- **Factual Inaccuracy**: Outputs should not be treated as factually correct.

---

## License & Citation

Subject to the MIT License. See [LICENSE_DECISION.md](file:///v:/collision%20-%201M/release/LICENSE_DECISION.md) and [CITATION.cff](file:///v:/collision%20-%201M/CITATION.cff).

```bibtex
@software{collision2026,
  author = {Viraj et al.},
  title = {COLLISION: Causal Language Modeling in Extreme Low-Resource Regimes},
  year = {2026},
  url = {https://github.com/viraj3106/Collision-1.46M}
}
```

