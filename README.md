
# COLLISION

Small Models. Real AI.

<p align="center">
  <a href="https://huggingface.co/collision-10M/collision-10m"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-blue" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/spaces/collision-10M/collision-ai-lab"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Interactive-Space%20Demo-purple" alt="Hugging Face Space"></a>
  <a href="https://colab.research.google.com/github/viraj3106/Collision-1.46M/blob/main/demo/collision_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Parameters-10.28M-purple" alt="Parameters">
  <img src="https://img.shields.io/badge/CPU%20Latency-%3C5ms-brightgreen" alt="Latency">
</p>

> **Project Status (Phase 100 — Production Release)**:  
> Base production model `COLLISION-10M` (SHA256: `d256d46d...3775b97`) is published to the Hugging Face Model Hub, paired with the full **In-House NLP Suite** (`collision.nlp`) and the **Synaptic Cognitive Brain** (`collision.brain`).

---

## What is COLLISION?

COLLISION is an ultra-efficient, CPU-native language model and hybrid intelligence architecture designed for edge devices, microservices, and extreme low-resource environments (sub-50M parameters).

It pairs a 10.28M causal transformer with:
1. **Natural Grounded Synthesis Engine**: Multi-source live web and local vector search.
2. **Industrial In-House NLP Toolkit (`collision.nlp`)**: 11 deterministic NLP tasks (TextRank, Topic Classification, Tone, Grammar Proofreading, SQuAD QA, Exact Math).
3. **Synaptic Cognitive Brain (`collision.brain`)**: Dual-Process System 1/2 controller, Graph-of-Thoughts (GoT) Hegelian Dialectics, Global Workspace Theory (GWT), and Synaptic Working Memory.

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

## 🧠 Synaptic Cognitive Brain (`collision.brain`)

COLLISION features a dual-process cognitive architecture executing non-linear **Graph-of-Thoughts (GoT)**, Hegelian Dialectic synthesis, Global Workspace Theory conscious broadcasting, and Synaptic Working Memory with LTP decay:

```python
from collision.brain import get_collision_brain

brain = get_collision_brain()

# Deliberative Hegelian reasoning (Thesis -> Antithesis -> Synthesis)
res = brain.think(
    query="Can artificial neural networks achieve subjective consciousness or only functional simulation?",
    domain="Philosophy & AI",
    enable_dialectic=True
)

print("Synthesized Stance:", res.answer)
print("Epistemic Certainty:", f"{res.epistemic_certainty*100:.1f}%")
print("Adversarial Resilience:", f"{res.adversarial_resilience*100:.1f}%")
```

### CLI Think Command
```bash
collision think "Should distributed databases prioritize strong consistency or partition availability?" --dialectic
```

---

## 📊 Empirical Scaling Ladder (15.8M $\rightarrow$ 243M Boundary)

We benchmarked model scaling trajectories across 50 training steps per tier:

| Model Tier | Exact Parameters | Architecture ($d/L/H/\text{vocab}$) | $\Delta\text{Loss}$ (50 steps) | Step Latency (CPU) | 10M Token Budget (GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COLLISION-15.8M** | **16,592,064** | $448 / 8 / 8 / 8\text{k}$ | **$-5.14$** | $1.60\text{ s}$ | $\sim 8\text{ mins}$ |
| **COLLISION-25.2M** | **25,263,936** | $512 / 10 / 8 / 8\text{k}$ | **$-5.09$** | $2.30\text{ s}$ | $\sim 12\text{ mins}$ |
| **COLLISION-50.4M** | **53,762,432** | $640 / 12 / 10 / 16\text{k}$ | **$-5.61$** | $4.22\text{ s}$ | $\sim 23\text{ mins}$ |
| **COLLISION-102.5M** | **113,029,888** | $768 / 16 / 12 / 32\text{k}$ | **$-6.64$** | $8.11\text{ s}$ | $\sim 44\text{ mins}$ |
| **COLLISION-150M** | **153,885,696** | $896 / 16 / 14 / 32\text{k}$ | **$-6.15$** | $10.93\text{ s}$ | $\sim 59\text{ mins}$ |
| **COLLISION-250M (Final)** | **243,025,152** | $1024 / 20 / 16 / 32\text{k}$ | **$-6.44$** | $17.22\text{ s}$ | $\sim 1.5\text{ hours}$ |

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

